"""Image validation, file properties, perceptual hash, the shared pixel pipeline, CLIP preprocessing,
training-time augmentation and the fixed test-time degradations.

**One pixel pipeline for both classes.** Real and fake images arrive with different sizes, formats
and metadata (a JPEG with EXIF from a camera vs. a 1024x1024 PNG from a generator). Any of those
differences is a label shortcut. So every image, whatever its source, goes through the same steps
before the encoder sees it: decode -> upright RGB (alpha flattened, EXIF dropped) -> resize the
shorter side to 224 -> centre crop 224x224 -> one JPEG re-encode at a fixed quality. `standardize`
is that pipeline; CLIP's own normalization follows in `clip_pixels`.

Nothing here writes an image to disk.
"""
import io

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

CLIP_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MIN_SIDE = 64                 # "tiny" = shorter side below this many pixels
STANDARD_JPEG_QUALITY = 95    # the one re-encode every image gets in `standardize`
DEGRADATIONS = ("jpeg90", "jpeg70", "jpeg50", "resize0.5", "centercrop0.7")

# The IJG reference luminance quantization table (quality 50). Only its sum is used, so its order
# does not matter.
_IJG_LUMA_SUM = 3688


def _flatten_to_rgb(img):
    """Any PIL mode -> RGB. Transparency is composited on white rather than dropped, because the
    colour hidden under transparent pixels is arbitrary."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        top = arr.max() if arr.size and arr.max() > 0 else 1.0
        img = Image.fromarray((arr / top * 255.0).astype(np.uint8), mode="L")
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def is_grayscale_only(rgb):
    """True when the three channels are identical everywhere (a grayscale picture stored as RGB)."""
    small = np.asarray(rgb.resize((64, 64), Image.BILINEAR), dtype=np.int16)
    return bool((small[..., 0] == small[..., 1]).all() and (small[..., 1] == small[..., 2]).all())


def estimate_jpeg_quality(img):
    """IJG quality (1-100) estimated from a JPEG's luminance quantization table, or None for other
    formats. Uses the table's sum against the reference table's, so it does not depend on whether
    the library returns the table in zigzag or natural order."""
    tables = getattr(img, "quantization", None)
    if img.format != "JPEG" or not tables:
        return None
    luma = np.asarray(list(tables[min(tables)]), dtype=np.float64)
    scale = 100.0 * luma.sum() / _IJG_LUMA_SUM
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return float(min(100.0, max(1.0, round(quality))))


def decode_image(data, min_side=MIN_SIDE):
    """Bytes -> a validated record. Never raises on a bad file: `ok` is False and `reason` says why
    (`corrupt`, `tiny`, `grayscale`). `rgb` is the upright RGB image, or None when not ok.

    The file properties (`format`, `jpeg_quality`, `has_exif`, `file_size`, `width`, `height`) are
    those of the file *as received* -- they feed the shortcut baseline, not the model.
    """
    record = {"ok": False, "reason": None, "rgb": None, "width": None, "height": None, "format": None,
              "mode": None, "jpeg_quality": None, "file_size": len(data) if data is not None else 0,
              "has_exif": None}
    try:
        img = Image.open(io.BytesIO(data))
        record.update(format=img.format, mode=img.mode, width=img.size[0], height=img.size[1])
        record["jpeg_quality"] = estimate_jpeg_quality(img)
        record["has_exif"] = bool(img.getexif()) or "exif" in img.info
        img.load()
        grayscale_mode = img.mode in ("L", "LA", "1", "I;16", "I;16B", "I;16L", "I", "F")
        rgb = _flatten_to_rgb(ImageOps.exif_transpose(img))
    except Exception as exc:  # noqa: BLE001 -- any decode failure is a corrupt file, recorded not raised
        record["reason"] = "corrupt"
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return record
    record.update(width=rgb.size[0], height=rgb.size[1])
    if min(rgb.size) < min_side:
        record["reason"] = "tiny"
    elif grayscale_mode or is_grayscale_only(rgb):
        record["reason"] = "grayscale"
    else:
        record.update(ok=True, rgb=rgb)
    return record


def validate_image(data, min_side=MIN_SIDE):
    """(ok, reason) only -- the cleaning rule of Task 2 in one call."""
    record = decode_image(data, min_side)
    return record["ok"], record["reason"]


# ---------------------------------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------------------------------
def phash64(rgb):
    """64-bit DCT perceptual hash (the usual pHash: 32x32 grayscale -> 2-D DCT -> the 8x8 lowest
    frequencies -> one bit per coefficient, above or below their median). Returned as a Python int.
    Resizing or re-encoding an image flips only a few bits, which is what near-duplicate search uses."""
    from scipy.fft import dct

    gray = np.asarray(rgb.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int(np.packbits(bits).view(">u8")[0])


# ---------------------------------------------------------------------------------------------------
# The shared pipeline and CLIP preprocessing
# ---------------------------------------------------------------------------------------------------
def jpeg_bytes(rgb, quality):
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def jpeg_roundtrip(rgb, quality):
    return Image.open(io.BytesIO(jpeg_bytes(rgb, quality))).convert("RGB")


def resize_center_crop(rgb, size=CLIP_SIZE):
    """CLIP's own geometry: shorter side to `size` (bicubic), then a centre crop of size x size.
    The longer side is rounded down, as CLIP's own processor does, so the two agree pixel for pixel."""
    w, h = rgb.size
    new_w, new_h = (size, max(size, int(size * h / w))) if w <= h else (max(size, int(size * w / h)), size)
    resized = rgb.resize((new_w, new_h), Image.BICUBIC)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))


def standardize(rgb, jpeg_quality=STANDARD_JPEG_QUALITY):
    """The shared pipeline: 224x224 RGB after one JPEG re-encode. Returns (image, jpeg byte count).
    The byte count is the only file property left that still varies between images, so it is what
    the post-pipeline shortcut baseline is run on."""
    fixed = resize_center_crop(rgb)
    if jpeg_quality is None:
        return fixed, None
    data = jpeg_bytes(fixed, jpeg_quality)
    return Image.open(io.BytesIO(data)).convert("RGB"), len(data)


def clip_pixels(rgb):
    """A 224x224 RGB image -> float32 tensor [3, 224, 224], scaled to [0, 1] and normalized with
    CLIP's mean and std. Images of another size get CLIP's resize + centre crop first."""
    if rgb.size != (CLIP_SIZE, CLIP_SIZE):
        rgb = resize_center_crop(rgb)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def preprocess_image(rgb, jpeg_quality=STANDARD_JPEG_QUALITY):
    """Upright RGB of any size -> the model input. Deterministic: same input, same tensor."""
    return clip_pixels(standardize(rgb, jpeg_quality)[0])


def inspect_image(data, min_side=MIN_SIDE, jpeg_quality=STANDARD_JPEG_QUALITY):
    """Everything the metadata table needs from one image, without keeping its pixels: validity,
    the file properties as received, the perceptual hash, and the byte count after the shared
    pipeline (`pipeline_file_size`)."""
    record = decode_image(data, min_side)
    rgb = record.pop("rgb")
    record.update(phash=None, pipeline_file_size=None)
    if record["ok"]:
        record["phash"] = phash64(rgb)
        record["pipeline_file_size"] = standardize(rgb, jpeg_quality)[1]
    return record


# ---------------------------------------------------------------------------------------------------
# Training-time augmentation (train split only) and fixed test-time degradations
# ---------------------------------------------------------------------------------------------------
_INTERPOLATIONS = (Image.BILINEAR, Image.BICUBIC, Image.LANCZOS)


def view_rng(image_id, view, seed=0):
    """A generator that depends only on (image, view, seed), so re-running an interrupted
    extraction reproduces exactly the same augmented views."""
    import hashlib

    digest = hashlib.sha256(f"{seed}|{view}|{image_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def augment_view(rgb, rng, scale=(0.5, 1.5), crop_area=(0.7, 1.0), jpeg_quality=(30, 100), colour=0.2):
    """One augmented view for the CLIP branch: random resize (random interpolation), random crop
    of 70-100% of the area, JPEG at a random quality, mild brightness/contrast. No rotation and no
    flip (rotation resamples pixels; a flip is wrong for images containing text). Ranges are the
    handbook's Section 21 starting points."""
    w, h = rgb.size
    factor = rng.uniform(*scale)
    new_w, new_h = max(MIN_SIDE, round(w * factor)), max(MIN_SIDE, round(h * factor))
    out = rgb.resize((new_w, new_h), _INTERPOLATIONS[int(rng.integers(len(_INTERPOLATIONS)))])
    side = float(np.sqrt(rng.uniform(*crop_area)))
    crop_w, crop_h = max(MIN_SIDE // 2, round(new_w * side)), max(MIN_SIDE // 2, round(new_h * side))
    left = int(rng.integers(0, new_w - crop_w + 1))
    top = int(rng.integers(0, new_h - crop_h + 1))
    out = out.crop((left, top, left + crop_w, top + crop_h))
    out = ImageEnhance.Brightness(out).enhance(1.0 + rng.uniform(-colour, colour))
    out = ImageEnhance.Contrast(out).enhance(1.0 + rng.uniform(-colour, colour))
    return jpeg_roundtrip(out, int(rng.integers(jpeg_quality[0], jpeg_quality[1] + 1)))


def degrade(rgb, name):
    """The fixed, documented degradations of the robustness suites, applied to the original image
    before the shared pipeline: JPEG at quality 90/70/50, a 0.5x resize, a centre crop keeping 70%
    of each side."""
    if name.startswith("jpeg"):
        return jpeg_roundtrip(rgb, int(name[4:]))
    w, h = rgb.size
    if name.startswith("resize"):
        factor = float(name[6:])
        return rgb.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.BICUBIC)
    if name.startswith("centercrop"):
        keep = float(name[10:])
        crop_w, crop_h = max(1, round(w * keep)), max(1, round(h * keep))
        left, top = (w - crop_w) // 2, (h - crop_h) // 2
        return rgb.crop((left, top, left + crop_w, top + crop_h))
    raise ValueError(f"unknown degradation: {name}")


def variant_image(rgb, variant, image_id, seed=0):
    """`clean`, `aug<k>` or a degradation name -> the image that goes into `standardize`."""
    if variant == "clean":
        return rgb
    if variant.startswith("aug"):
        return augment_view(rgb, view_rng(image_id, variant, seed))
    return degrade(rgb, variant)
