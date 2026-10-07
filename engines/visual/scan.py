"""The two passes over the image sources.

Pass 1 (`scan_*`, Tasks 1-2) reads each selected image once to fill its metadata row: validity,
file properties, perceptual hash. No encoder, no GPU.
Pass 2 (`load_*`, Task 5) reads the images that survived cleaning and splitting again, to embed them.

Two passes because the shortcut baselines (Task 4) must pass on the metadata before any encoder
runs. Neither pass writes an image to disk; pass 1 keeps a metadata row, pass 2 an embedding.
"""
from concurrent.futures import ThreadPoolExecutor

from .data import META_COLUMNS
from .data import SparseParquet
from .data import image_bytes_column
from .data import prompt_id_of
from .data import read_file_bytes
from .data import stable_fraction
from .metadata import LICENSES
from .metadata import family_of
from .preprocess import inspect_image

REAL_FAMILY = "real"


def _image_fields(record):
    return {"width": record["width"], "height": record["height"], "format": record["format"],
            "jpeg_quality": record["jpeg_quality"], "file_size": record["file_size"], "has_exif": record["has_exif"],
            "phash": record["phash"], "pipeline_file_size": record["pipeline_file_size"],
            "valid": bool(record["ok"]), "drop_reason": record["reason"]}


def openfake_row(meta, locator):
    """The metadata fields of an OpenFake row that are known before its image is read."""
    fake = int(meta["label"] == "fake")
    image_id = f"openfake:{meta['row_id']}"
    return {
        "image_id": image_id, "source_dataset": "openfake", "generator": meta["model"],
        "generator_family": family_of(meta["model"])[0] if fake else REAL_FAMILY, "is_fake": fake,
        # OpenFake fakes are whole generated images (frames, for the video models); its reals are untouched photos.
        "manipulation_type": ("generated_video_frame" if meta.get("type") == "video" else "fully_generated") if fake else "none",
        "source_image_id": None if fake else image_id, "identity_id": None,
        "prompt_id": prompt_id_of(meta.get("prompt")), "license": LICENSES["openfake"],
        "release_date": meta.get("release_date"), "locator": locator,
    }


def keep_row(image_id, generator, keep_fraction, seed=0):
    """Deterministic thinning of over-represented generators: a row is kept when a hash of its id
    falls below that generator's fraction (default 1 = keep everything)."""
    fraction = keep_fraction.get(generator, keep_fraction.get("*", 1.0)) if keep_fraction else 1.0
    return fraction >= 1.0 or stable_fraction(image_id, seed) < fraction


def scan_openfake_shard(reader, shard, keep_fraction=None, seed=0, threads=4):
    """Pass 1 over one shard: metadata rows for every kept row. The small columns are read first;
    a row group's images are downloaded only if it holds a kept row."""
    sparse = SparseParquet(reader, shard["path"], shard["size"])
    columns = [c for c in META_COLUMNS if c in sparse.columns]
    rows, stats = [], {"rows_in_shard": int(sparse.meta.num_rows), "kept": 0, "row_groups_downloaded": 0}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for group in range(sparse.num_row_groups):
            meta = sparse.read(columns, [group]).to_pylist()
            wanted = [i for i, m in enumerate(meta) if keep_row(f"openfake:{m['row_id']}", m["model"], keep_fraction, seed)]
            if not wanted:
                continue
            blobs = image_bytes_column(sparse.read(["image"], [group]))
            stats["row_groups_downloaded"] += 1
            for i, record in zip(wanted, pool.map(inspect_image, [blobs[i] for i in wanted])):
                rows.append({**openfake_row(meta[i], f"{shard['path']}#{group}#{i}"), **_image_fields(record)})
    stats["kept"] = len(rows)
    return rows, stats


def scan_folder_rows(index_rows, threads=4):
    """Pass 1 over files on disk (ArtiFact, FFHQ): `index_rows` come from data.index_artifact /
    data.index_ffhq and carry `path`."""
    def one(row):
        record = inspect_image(read_file_bytes(row["path"]))
        fake = int(row["is_fake"])
        out = {"image_id": row["image_id"], "source_dataset": row["source_dataset"], "generator": row["generator"],
               "generator_family": family_of(row["generator"])[0] if fake else REAL_FAMILY, "is_fake": fake,
               "manipulation_type": None if fake else "none",
               "source_image_id": row.get("source_image_id") or (None if fake else row["image_id"]),
               "identity_id": row.get("identity_id"), "prompt_id": None,
               "license": LICENSES.get(row["source_dataset"]), "release_date": None, "locator": row["path"]}
        return {**out, **_image_fields(record)}

    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(one, index_rows))


def load_openfake_items(reader, shard, variants_by_locator):
    """Pass 2 for one shard: (image_id, image bytes, variants) for the rows named in
    `variants_by_locator` ({locator: (image_id, [variants])}). Only the row groups that hold a
    wanted row are downloaded."""
    by_group = {}
    for locator, (image_id, variants) in variants_by_locator.items():
        _, group, row = locator.rsplit("#", 2)
        by_group.setdefault(int(group), []).append((int(row), image_id, variants))
    sparse = SparseParquet(reader, shard["path"], shard["size"])
    items = []
    for group in sorted(by_group):
        blobs = image_bytes_column(sparse.read(["image"], [group]))
        items.extend((image_id, blobs[row], variants) for row, image_id, variants in sorted(by_group[group]))
    return items


def load_folder_items(rows):
    """Pass 2 for files on disk: `rows` are (image_id, path, variants)."""
    return [(image_id, read_file_bytes(path), variants) for image_id, path, variants in rows]


def fetch_image_bytes(rows, reader=None, shard_sizes=None):
    """{image_id: bytes} for a handful of rows given as (image_id, source_dataset, locator) --
    used to look at duplicate pairs and to load the capped pixel subset of the heavier baselines.
    The bytes stay in memory."""
    out, by_shard = {}, {}
    for image_id, source, locator in rows:
        if source == "openfake":
            by_shard.setdefault(locator.rsplit("#", 2)[0], {})[locator] = (image_id, ["clean"])
        else:
            out[image_id] = read_file_bytes(locator)
    for path, wanted in by_shard.items():
        for image_id, data, _ in load_openfake_items(reader, {"path": path, "size": shard_sizes[path]}, wanted):
            out[image_id] = data
    return out
