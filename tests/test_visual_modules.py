"""CPU tests for the Visual Engine (docs/PHASE_3_VISUAL_ENGINE.md, section 11). Synthetic data only;
no reported numbers.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_visual_modules.py
"""
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from engines.visual import baselines, calibrate, data, embed, heads, metadata, preprocess, scan, splits
from engines.visual import evaluate as visual_evaluate
from engines.visual.experiment import EmbeddingBank, run_fold
from engines.voice import evaluate as voice_evaluate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def picture(seed=0, fake=False, size=(200, 150)):
    return data.synthetic_picture(np.random.default_rng(seed), fake, size)


def png(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# --- preprocess ----------------------------------------------------------------------------------
def test_preprocess_shape_dtype_range_and_determinism():
    img = picture(1, size=(640, 480))
    a, b = preprocess.preprocess_image(img), preprocess.preprocess_image(img)
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32
    assert torch.equal(a, b), "the same input must give the same tensor"
    low = (0.0 - max(preprocess.CLIP_MEAN)) / min(preprocess.CLIP_STD)
    high = (1.0 - min(preprocess.CLIP_MEAN)) / min(preprocess.CLIP_STD)
    assert low <= float(a.min()) and float(a.max()) <= high
    # Every source size ends at the same shape: FFHQ thumbnails, ArtiFact, a tall phone photo.
    for size in ((128, 128), (200, 200), (300, 900)):
        assert preprocess.preprocess_image(picture(2, size=size)).shape == (3, 224, 224)


def test_clip_preprocessing_matches_the_official_processor():
    transformers = pytest.importorskip("transformers")
    official_class = getattr(transformers, "CLIPImageProcessorPil", transformers.CLIPImageProcessor)
    for size in ((640, 480), (333, 517), (128, 128)):
        img = picture(3, size=size)
        official = official_class()(images=img, return_tensors="pt")["pixel_values"][0]
        ours = preprocess.clip_pixels(preprocess.resize_center_crop(img))
        assert float((official - ours).abs().max()) < 1e-5, size


def test_rgba_palette_and_grayscale_are_handled():
    rgba = picture(4, fake=True).convert("RGBA")
    rgba.putalpha(Image.fromarray(np.tile(np.linspace(0, 255, rgba.size[0], dtype=np.uint8), (rgba.size[1], 1))))
    record = preprocess.decode_image(png(rgba))
    assert record["ok"] and record["rgb"].mode == "RGB" and record["mode"] == "RGBA"
    # Fully transparent pixels become white, not whatever colour was hidden under them.
    assert np.asarray(record["rgb"])[:, 0].min() > 240
    assert preprocess.preprocess_image(record["rgb"]).shape == (3, 224, 224)

    gray = preprocess.decode_image(png(picture(5).convert("L")))
    assert not gray["ok"] and gray["reason"] == "grayscale"
    gray_as_rgb = preprocess.decode_image(png(picture(5).convert("L").convert("RGB")))
    assert not gray_as_rgb["ok"] and gray_as_rgb["reason"] == "grayscale", "grayscale stored as RGB is still grayscale"
    palette = preprocess.decode_image(png(picture(6).convert("P")))
    assert palette["ok"] and palette["rgb"].mode == "RGB"


def test_validate_image_drops_corrupt_tiny_and_grayscale():
    assert preprocess.validate_image(png(picture(7))) == (True, None)
    assert preprocess.validate_image(b"definitely not an image") == (False, "corrupt")
    assert preprocess.validate_image(png(picture(7))[:200]) == (False, "corrupt"), "a truncated file is corrupt"
    assert preprocess.validate_image(png(picture(7, size=(40, 300)))) == (False, "tiny")
    assert preprocess.validate_image(png(picture(7).convert("L"))) == (False, "grayscale")


def test_file_properties_and_jpeg_quality_estimate():
    img = picture(8)
    for quality in (30, 50, 75, 90, 95):
        record = preprocess.decode_image(preprocess.jpeg_bytes(img, quality))
        assert record["format"] == "JPEG" and abs(record["jpeg_quality"] - quality) <= 1
        assert record["has_exif"] is False
    record = preprocess.decode_image(png(img))
    assert record["format"] == "PNG" and record["jpeg_quality"] is None and record["file_size"] == len(png(img))
    exif = Image.Exif()
    exif[0x010F] = "Cam"
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", exif=exif)
    assert preprocess.decode_image(buffer.getvalue())["has_exif"] is True


def test_phash_is_stable_under_resize_and_reencode_and_differs_between_images():
    img = picture(9, size=(400, 300))
    base = preprocess.phash64(img)
    again = preprocess.phash64(preprocess.jpeg_roundtrip(img.resize((200, 150)), 60))
    other = preprocess.phash64(picture(10, size=(400, 300)))
    distance = lambda a, b: bin(a ^ b).count("1")  # noqa: E731
    assert distance(base, again) <= 4 and distance(base, other) > 10


def test_augmented_views_are_reproducible_and_degradations_do_what_they_say():
    img = picture(11, size=(300, 240))
    view = lambda: preprocess.variant_image(img, "aug0", "img-1", seed=0)  # noqa: E731
    assert np.array_equal(np.asarray(view()), np.asarray(view())), "the same (image, view, seed) must give the same view"
    assert not np.array_equal(np.asarray(view()), np.asarray(preprocess.variant_image(img, "aug1", "img-1", seed=0)))
    assert preprocess.variant_image(img, "clean", "img-1") is img
    assert preprocess.degrade(img, "resize0.5").size == (150, 120)
    assert preprocess.degrade(img, "centercrop0.7").size == (210, 168)
    assert preprocess.degrade(img, "jpeg50").size == img.size
    with pytest.raises(ValueError):
        preprocess.degrade(img, "blur3")
    inspected = preprocess.inspect_image(png(img))
    assert inspected["ok"] and "rgb" not in inspected and inspected["pipeline_file_size"] > 0 and inspected["phash"] is not None


# --- data access -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_shards(tmp_path_factory):
    import fsspec

    root = tmp_path_factory.mktemp("openfake")
    rows = data.make_synthetic_openfake(n_prompts=40)
    data.write_synthetic_openfake(root, rows, n_shards=2, rows_per_group=16)
    fs = fsspec.filesystem("file")
    return fs, data.list_parquet_shards(fs, str(root)), rows


def test_sparse_parquet_reads_exactly_the_bytes_it_needs(synthetic_shards):
    fs, shards, rows = synthetic_shards
    reader = data.FsRangeReader(fs)
    sparse = data.SparseParquet(reader, shards[0]["path"], shards[0]["size"], footer_guess=4096)
    assert shards[0]["config"] == "core" and shards[0]["split"] == "train"
    before = reader.bytes_read
    table = sparse.read(["label", "model"])
    assert table.num_rows == sparse.meta.num_rows and sparse.file.misses == 0
    assert reader.bytes_read - before < 0.02 * shards[0]["size"], "metadata must not pull the image column"
    one_group = sparse.read(["image", "label"], [1])
    blobs = data.image_bytes_column(one_group)
    assert len(blobs) == sparse.row_group_rows()[1] and sparse.file.misses == 0
    assert sum(sparse.column_bytes(["image"])) > 0.9 * shards[0]["size"]


def test_metadata_scan_counts_match_the_rows_written(synthetic_shards):
    fs, shards, rows = synthetic_shards
    result = data.scan_all_shards(data.FsRangeReader(fs), shards, threads=2, progress_every=0)
    assert not result["failed"] and sum(s["rows"] for s in result["shards"]) == len(rows)
    labels = Counter(label for s in result["shards"] for label in s["label"])
    assert labels == Counter(r["label"] for r in rows)
    stats = data.prompt_group_stats(result)
    assert stats["prompts_with_real_and_fake"] == 40 and stats["rows_with_empty_prompt"] == 0
    sample = data.time_streamed_sample(data.FsRangeReader(fs), shards, max_images=40, log=lambda *_: None)
    assert sample["images"] >= 32 and sample["sparse_misses"] == 0
    assert any(g["entered_by_seek"] and g["row_group"] > 0 for g in sample["row_groups"]), "the second shard is entered mid-file"


def test_shards_to_fill_quota():
    per_shard = [{"a": 5, "b": 1}, {"a": 5, "b": 1}, {"a": 5}]
    assert data.shards_to_fill_quota(per_shard, 10) == {"a": 2, "b": None}
    assert data.shards_to_fill_quota(per_shard, 2) == {"a": 1, "b": 2}


class _FakeResponse:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}


class _FakeSession:
    """Stands in for the Hub: one redirect to a signed storage URL, which serves byte ranges."""

    def __init__(self, payload):
        self.payload, self.calls, self.expired = payload, [], set()

    def get(self, url, headers=None, follow_redirects=False, timeout=None):
        self.calls.append((url, dict(headers or {})))
        if "huggingface.co" in url:
            return _FakeResponse(302, headers={"Location": f"https://storage.example/blob?sig={len(self.calls)}"})
        if url in self.expired:
            return _FakeResponse(403)
        start, end = (int(x) for x in headers["Range"].split("=")[1].split("-"))
        return _FakeResponse(206, self.payload[start:end + 1])


def test_http_reader_reuses_the_storage_url_and_keeps_the_token_off_the_storage_host():
    payload = bytes(range(256)) * 40
    session = _FakeSession(payload)
    reader = data.HttpRangeReader("org/repo", token="hf_secret", session=session)
    path = "datasets/org/repo/core/train-00000-of-00001.parquet"
    assert reader.read(path, 10, 20) == payload[10:20]
    assert reader.read(path, 1000, 1300) == payload[1000:1300]
    assert reader.hub_requests == 1 and reader.requests == 2, "the Hub is asked once per file, not once per range"
    for url, headers in session.calls:
        assert ("Authorization" in headers) == ("huggingface.co" in url), "the token goes to the Hub only"
    assert session.calls[0][0] == "https://huggingface.co/datasets/org/repo/resolve/main/core/train-00000-of-00001.parquet"
    # A signed URL that has expired is resolved again.
    session.expired.add(reader._storage_url[path])
    assert reader.read(path, 0, 5) == payload[0:5] and reader.hub_requests == 2


def test_secrets_are_scrubbed_and_never_reach_a_json_file(tmp_path):
    token = "hf_abcdefghijklmnopqrstuvwxyz012345"
    value = {"error": f"401 for token {token}", "nested": [{"k": f"Bearer {token}"}], "n": 3}
    clean = data.scrub_secrets(value, [token, None])
    assert token not in repr(clean) and clean["n"] == 3 and "<redacted>" in clean["error"]
    data.save_json(tmp_path / "out.json", value, [token])
    assert token not in (tmp_path / "out.json").read_text(encoding="utf-8")


def test_stable_fraction_and_prompt_id():
    assert data.stable_fraction("row-1", 0) == data.stable_fraction("row-1", 0) != data.stable_fraction("row-1", 1)
    values = [data.stable_fraction(f"row-{i}") for i in range(2000)]
    assert 0 <= min(values) and max(values) < 1 and abs(np.mean(values) - 0.5) < 0.03
    assert data.prompt_id_of("A  Cat\non a mat") == data.prompt_id_of("a cat on a mat")
    assert data.prompt_id_of("") is None and data.prompt_id_of(None) is None


def test_folder_sources(tmp_path):
    artifact = data.write_synthetic_artifact(tmp_path / "artifact", per_folder=5)
    ffhq = data.write_synthetic_ffhq(tmp_path / "ffhq", n=6)
    summary = data.summarize_folder(artifact, log=lambda *_: None)
    assert summary["complete"] and summary["files"] == 4 * 6 and len(summary["csv_files"]) == 4
    described = data.describe_csv(Path(artifact) / "stylegan2" / "metadata.csv")
    assert described["columns"] == ["filename", "image_path", "target", "category"] and described["value_counts"]["target"] == {"1": 5}
    index = data.index_artifact(artifact)
    assert len(index) == 20 and sum(r["is_fake"] for r in index) == 10
    faces = data.index_ffhq(ffhq)
    assert len(faces) == 6 and all(r["identity_id"] == r["source_image_id"] for r in faces)
    headers = data.header_summary([r["path"] for r in faces])
    assert headers["sizes"] == {"128x128": 6} and headers["modes"] == {"RGB": 6}
    (Path(artifact) / "ddpm" / "metadata.csv").write_text("name,label\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data.index_artifact(artifact)   # an unexpected layout is an error, never a guess
    # Attached datasets are found by name, at whatever depth Kaggle mounted them.
    for folder in ("artifact-dataset/metfaces", "datasets/someone/flickr-faces-70k/thumbnails128x128"):
        (tmp_path / "input" / folder).mkdir(parents=True)
    found = data.find_kaggle_inputs(tmp_path / "input")
    assert found[0].endswith("input/artifact-dataset") and any(f.endswith("someone/flickr-faces-70k") for f in found)
    assert data.match_input(found, ["flickr", "faces"], not_under=data.match_input(found, ["artifact"])).endswith("flickr-faces-70k")
    assert data.find_kaggle_inputs(tmp_path / "input", max_entries=1) == [], "a crowded folder is not opened"
    inputs = [str(tmp_path / "artifact-dataset"), str(tmp_path / "artifact-dataset" / "metfaces"), str(tmp_path / "flickr-faces-70k")]
    assert data.match_input(inputs, ["artifact"]) == inputs[0]
    assert data.match_input(inputs, ["faces"], not_under=inputs[0]) == inputs[2], "an ArtiFact sub-folder is not the FFHQ dataset"


# --- metadata, dedupe ------------------------------------------------------------------------------
def test_family_rules():
    assert metadata.family_of("flux.1-dev")[0] == metadata.family_of("flux-1.1-pro")[0] == "flux"
    assert metadata.family_of("midjourney-6")[0] == metadata.family_of("midjourney-7")[0] == "midjourney"
    assert metadata.family_of("sd-1.5")[0] == metadata.family_of("sd-2.1")[0] == "sd-1x-2x"
    assert metadata.family_of("sdxl-juggernaut")[0] == "sdxl" and metadata.family_of("sd-3.5")[0] == "sd-3"
    family, rule = metadata.family_of("some-new-model-2.0")
    assert (family, rule) == ("some-new-model", "fallback"), "an unknown name is flagged, with its version stripped"


def test_near_duplicate_pairs_match_brute_force():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2 ** 63, size=120, dtype=np.uint64)
    flips = [np.uint64(sum(1 << int(b) for b in rng.choice(64, size=k, replace=False))) for k in rng.integers(1, 9, size=60)]
    hashes = np.concatenate([base, base[:60] ^ np.array(flips, dtype=np.uint64), base[:5]])
    for tolerance in (0, 3, 7):
        found = {(int(i), int(j)): int(d) for i, j, d in metadata.near_duplicate_pairs(hashes, tolerance)}
        expected = {}
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                d = bin(int(hashes[i]) ^ int(hashes[j])).count("1")
                if d <= tolerance:
                    expected[(i, j)] = d
        assert found == expected
    with pytest.raises(ValueError):
        metadata.near_duplicate_pairs(hashes, 8)
    assert metadata.popcount64(np.array([0, 1, 255, 2 ** 64 - 1], dtype=np.uint64)).tolist() == [0, 1, 8, 64]


def _table(rows):
    table = metadata.build_table(rows)
    table["valid"] = table["valid"].astype(bool)
    return table


def test_dedupe_keeps_one_per_cluster_and_drops_mixed_label_clusters():
    h = 0x0F0F0F0F0F0F0F0F
    rows = [
        {"image_id": "a", "is_fake": 0, "source_dataset": "x", "phash": h, "valid": True},
        {"image_id": "b", "is_fake": 0, "source_dataset": "y", "phash": h ^ 0b11, "valid": True},      # near duplicate of a
        {"image_id": "c", "is_fake": 1, "source_dataset": "x", "phash": ~h & (2 ** 64 - 1), "valid": True},
        {"image_id": "d", "is_fake": 0, "source_dataset": "x", "phash": ~h & (2 ** 64 - 1), "valid": True},  # same picture, other label
        {"image_id": "e", "is_fake": 1, "source_dataset": "x", "phash": 0x123456789ABCDEF0, "valid": True},
        {"image_id": "f", "is_fake": 1, "source_dataset": "x", "phash": None, "valid": False, "drop_reason": "corrupt"},
    ]
    table, report = metadata.dedupe_table(_table(rows), tolerance=4)
    kept = set(table.loc[table["valid"], "image_id"])
    assert kept == {"a", "e"}
    assert report["clusters"] == 2 and report["removed"] == 3 and report["mixed_label_clusters"] == 1
    assert report["cross_source_clusters"] == 1
    assert set(table.loc[table["drop_reason"] == "duplicate", "image_id"]) == {"b", "c", "d"}
    assert metadata.distance_histogram(_table(rows))[2] == 1 and metadata.distance_histogram(_table(rows))[0] == 1
    # A chunk that holds an invalid image (no hash) must not disturb the other hashes: all 64 bits survive.
    assert _table(rows).set_index("image_id").loc["e", "phash"] == "123456789abcdef0"
    nulls = metadata.null_counts(_table(rows))
    assert nulls["prompt_id"] == 6 and nulls["image_id"] == 0, "unknown fields stay null and are counted"
    assert set(metadata.METADATA_COLUMNS) <= set(table.columns)


# --- splits and leakage ----------------------------------------------------------------------------
def synthetic_table(n_prompts=240, seed=0):
    """A metadata table shaped like the real one: per prompt one real row and fakes from several
    generators (four families), plus ungrouped reals with an identity each."""
    rng = np.random.default_rng(seed)
    generators = {"flux.1-dev": "flux", "flux.1-schnell": "flux", "midjourney-6": "midjourney", "sd-1.5": "sd-1x-2x",
                  "sd-2.1": "sd-1x-2x", "sdxl": "sdxl"}
    rows = []
    for p in range(n_prompts):
        rows.append({"image_id": f"real{p:04d}", "source_dataset": "openfake", "generator": "laion", "generator_family": "real",
                     "is_fake": 0, "prompt_id": f"p{p}", "source_image_id": f"real{p:04d}", "valid": True,
                     "phash": int(rng.integers(0, 2 ** 63)), "width": 640, "height": 480, "format": "JPEG",
                     "jpeg_quality": 85.0, "file_size": int(rng.integers(40000, 90000)), "has_exif": True,
                     "pipeline_file_size": int(rng.integers(9000, 11000))})
        for g in rng.choice(list(generators), size=3, replace=False):
            rows.append({"image_id": f"fake{p:04d}-{g}", "source_dataset": "openfake", "generator": g,
                         "generator_family": generators[g], "is_fake": 1, "prompt_id": f"p{p}", "valid": True,
                         "phash": int(rng.integers(0, 2 ** 63)), "width": 1024, "height": 1024, "format": "PNG",
                         "jpeg_quality": None, "file_size": int(rng.integers(900000, 1500000)), "has_exif": False,
                         "pipeline_file_size": int(rng.integers(9000, 11000))})
    for k in range(60):
        rows.append({"image_id": f"face{k:03d}", "source_dataset": "ffhq", "generator": "ffhq", "generator_family": "real",
                     "is_fake": 0, "identity_id": f"id{k}", "source_image_id": f"face{k:03d}", "valid": True,
                     "phash": int(rng.integers(0, 2 ** 63)), "width": 128, "height": 128, "format": "PNG", "jpeg_quality": None,
                     "file_size": 30000, "has_exif": False, "pipeline_file_size": int(rng.integers(9000, 11000))})
    return _table(rows)


FRACTIONS = {"train": 0.6, "val": 0.1, "known_test": 0.1, "holdout": 0.2}


@pytest.fixture(scope="module")
def split_table():
    table = splits.assign_partitions(synthetic_table(), FRACTIONS, seed=0)
    return splits.add_fold_roles(table, ["flux", "midjourney", "sd-1x-2x", "sdxl"], seed=0, old_families=["sd-1x-2x"])


def test_groups_live_in_exactly_one_partition(split_table):
    assert split_table.groupby("group_id")["split"].nunique().max() == 1
    assert split_table.groupby("prompt_id")["split"].nunique().max() == 1, "a real photo and the fakes of its caption stay together"
    shares = split_table["split"].value_counts(normalize=True)
    assert all(abs(shares[name] - share) < 0.08 for name, share in FRACTIONS.items())
    again = splits.assign_partitions(synthetic_table().sample(frac=1.0, random_state=3), FRACTIONS, seed=0)
    merged = split_table[["image_id", "split"]].merge(again[["image_id", "split"]], on="image_id")
    assert (merged["split_x"] == merged["split_y"]).all(), "placement must not depend on row order"


def test_group_ids_join_rows_through_any_shared_key():
    table = _table([
        {"image_id": "a", "prompt_id": "p1", "valid": True}, {"image_id": "b", "prompt_id": "p1", "identity_id": "i1", "valid": True},
        {"image_id": "c", "identity_id": "i1", "source_image_id": "s1", "valid": True}, {"image_id": "d", "source_image_id": "s1", "valid": True},
        {"image_id": "e", "valid": True}, {"image_id": "f", "prompt_id": "p2", "valid": True}])
    groups = splits.group_ids(table)
    assert len(set(groups[:4])) == 1 and groups[0] == "a", "a-b by prompt, b-c by identity, c-d by source image"
    assert len(set(groups)) == 3


def test_fold_roles_follow_the_addendum(split_table):
    for family in ("flux", "midjourney", "sd-1x-2x", "sdxl"):
        roles = split_table[splits.FOLD_PREFIX + family]
        held = (split_table["generator_family"] == family) & (split_table["is_fake"] == 1)
        assert not roles[held].isin(["train", "val", "known_test"]).any(), "every fake of the held-out family is test-only"
        unseen = split_table[roles == "unseen_test"]
        assert (unseen["split"] == "holdout").all()
        assert set(unseen.loc[unseen["is_fake"] == 1, "generator_family"]) == {family}
        assert (unseen["is_fake"] == 1).sum() == (unseen["is_fake"] == 0).sum(), "reserved reals are matched to the fake side"
        val = split_table[roles == "val"]
        assert family not in set(val["generator_family"]), "validation holds only generators seen in the fold"
    reserved = (split_table["is_fake"] == 0) & (split_table["split"] == "holdout")
    for fold in splits.fold_names(split_table):
        assert not split_table.loc[reserved, splits.FOLD_PREFIX + fold].isin(["train", "val", "known_test"]).any()
    old = split_table[splits.FOLD_PREFIX + splits.OLD_ONLY]
    assert set(split_table.loc[(old == "train") & (split_table["is_fake"] == 1), "generator_family"]) == {"sd-1x-2x"}
    report = splits.leakage_report(split_table, phash_tolerance=4)
    assert report["violations"] == [] and set(report["folds"]) == {"flux", "midjourney", "sd-1x-2x", "sdxl", "old_only"}
    assert report["folds"]["flux"]["balance"]["per_role"]["unseen_test"]["fake"] > 0


def test_leakage_checks_catch_each_planted_violation(split_table):
    column = splits.FOLD_PREFIX + "flux"
    # 1. a held-out fake slipped into training
    bad = split_table.copy()
    victim = bad.index[(bad["generator_family"] == "flux") & (bad["is_fake"] == 1)][0]
    bad.loc[victim, column] = "train"
    with pytest.raises(AssertionError, match="held-out rows in train/val"):
        splits.leakage_report(bad)
    # 2. a reserved real used for training
    bad = split_table.copy()
    victim = bad.index[(bad["is_fake"] == 0) & (bad["split"] == "holdout")][0]
    bad.loc[victim, column] = "train"
    with pytest.raises(AssertionError, match="reserved reals"):
        splits.leakage_report(bad)
    # 3a. a prompt shared between train and the unseen test
    bad = split_table.copy()
    train_row = bad.index[bad[column] == "train"][0]
    unseen_row = bad.index[bad[column] == "unseen_test"][0]
    bad.loc[unseen_row, "prompt_id"] = bad.loc[train_row, "prompt_id"]
    with pytest.raises(AssertionError, match="prompt_id values span more than one split"):
        splits.leakage_report(bad)
    # 3b. a near-duplicate image across splits
    bad = split_table.copy()
    bad.loc[unseen_row, "phash"] = f"{int(bad.loc[train_row, 'phash'], 16) ^ 0b101:016x}"
    with pytest.raises(AssertionError, match="near-duplicate pairs across splits"):
        splits.leakage_report(bad, phash_tolerance=4)
    assert splits.leakage_report(bad, phash_tolerance=4, strict=False)["folds"]["flux"]["phash_pairs_across_splits"] == 1


def test_eligibility_caps_and_balance(split_table):
    eligible = splits.eligible_families(split_table, min_unseen_fakes=10)
    assert eligible["flux"]["eligible"] and eligible["flux"]["holdout"] >= 10
    strict = splits.eligible_families(split_table, min_unseen_fakes=10 ** 6)
    assert not strict["flux"]["eligible"] and "holdout fakes" in strict["flux"]["why_not"]
    capped = splits.apply_generator_caps(split_table, {"fake": {"train": 20}, "real": {}}, seed=0)
    counts = capped[capped["valid"]].groupby(["generator", "split", "is_fake"]).size()
    assert counts.loc[("sdxl", "train", 1)] == 20 and (capped["drop_reason"] == "over_cap").sum() > 0
    assert capped.loc[capped["valid"] & (capped["is_fake"] == 0)].shape[0] == (split_table["is_fake"] == 0).sum()
    labels = np.array([1] * 30 + [0] * 10)
    keep = splits.balanced_subsample(labels, np.random.default_rng(0))
    assert labels[keep].sum() == 10 and (labels[keep] == 0).sum() == 10
    assert len(splits.balanced_subsample(np.array([1, 0, 1, 0]), np.random.default_rng(0))) == 4, "already balanced: nothing dropped"
    assert splits.auc_ci_halfwidth(0.9, 2000, 2000) < splits.auc_ci_halfwidth(0.9, 100, 100)


# --- shortcut baselines ----------------------------------------------------------------------------
def test_shortcut_baseline_flags_a_planted_shortcut_and_passes_without_one(split_table):
    result = baselines.run_shortcut_baselines(split_table, n_resamples=200, seed=0)
    by = {(r["mode"], r["kind"]): r for r in result["results"]}
    assert by[("raw", "logreg")]["separable"] and by[("raw", "xgboost")]["auc"] > 0.99, "format and size give the label away"
    # After the shared pipeline only the re-encoded byte count varies, and here it is unrelated to the label.
    assert not by[("pipeline", "logreg")]["separable"]
    assert by[("pipeline", "logreg")]["ci_low"] <= 0.5 <= by[("pipeline", "logreg")]["ci_high"]
    # FFHQ contributes real images only, so the source alone predicts the label.
    assert by[("source", "logreg")]["separable"]
    assert not result["gate"]["passed"] and any("source" in f for f in result["gate"]["failing"])
    without_ffhq = split_table[split_table["source_dataset"] != "ffhq"]
    assert baselines.run_shortcut_baselines(without_ffhq, n_resamples=200, seed=0, kinds=("logreg",))["gate"]["passed"]
    table = baselines.source_class_table(split_table)
    assert table["ffhq"]["fake"] == 0 and table["ffhq"]["top_sizes"] == {"128x128": 60}


# --- metrics, bootstrap, calibration ---------------------------------------------------------------
def test_bootstrap_reuses_the_voice_code_and_brackets_the_point():
    assert visual_evaluate._stratified_resample is voice_evaluate._stratified_resample
    assert visual_evaluate.compute_eer is voice_evaluate.compute_eer
    rng = np.random.default_rng(0)
    labels = np.array([1] * 300 + [0] * 500)
    probs = np.clip(rng.normal(0.35 + 0.3 * labels, 0.15), 0, 1)
    result = visual_evaluate.evaluate_suite(probs, labels, 0.5, np.where(labels == 1, "gen-a", "laion"), n_resamples=200)
    for name, metric in result["metrics"].items():
        assert metric["ci_low"] <= metric["value"] <= metric["ci_high"], name
    from sklearn.metrics import roc_auc_score

    assert abs(result["metrics"]["roc_auc"]["value"] - roc_auc_score(labels, probs)) < 1e-12
    assert result["counts"] == {"fake": 300, "real": 500}
    c = result["confusion"]
    assert c["tp"] + c["fn"] == 300 and c["fp"] + c["tn"] == 500
    assert result["per_generator"]["gen-a"]["kind"] == "fake" and result["per_generator"]["laion"]["kind"] == "real"
    assert abs(result["per_generator"]["gen-a"]["detection_rate"]["value"] - c["tp"] / 300) < 1e-12
    small = visual_evaluate.bootstrap_metric_cis(probs[::8], labels[::8], 0.5, n_resamples=200)
    width = lambda m: m["ci_high"] - m["ci_low"]  # noqa: E731
    assert width(small["roc_auc"]) > width(result["metrics"]["roc_auc"]), "fewer test images -> a wider CI"


def test_paired_difference_calls_a_tie_a_tie():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 600)
    a = np.clip(rng.normal(0.4 + 0.2 * labels, 0.2), 0, 1)
    tie = visual_evaluate.paired_metric_difference(a, np.clip(a + rng.normal(0, 0.01, 600), 0, 1), labels, n_resamples=200)
    assert tie["verdict"] == "tie" and tie["ci_low"] <= 0 <= tie["ci_high"]
    worse = visual_evaluate.paired_metric_difference(a, rng.uniform(0, 1, 600), labels, n_resamples=200)
    assert worse["verdict"] == "different" and worse["difference"] > 0
    gap = visual_evaluate.unpaired_metric_difference(a, labels, rng.uniform(0, 1, 400), rng.integers(0, 2, 400), n_resamples=200)
    assert gap["excludes_zero"] and gap["difference"] > 0
    grid = visual_evaluate.results_template(["m1"], ["T1", "T3"])
    assert grid == {"m1": {"T1": "not yet run", "T3": "not yet run"}}


def test_ece_on_a_hand_worked_example():
    # 10 predictions at 0.9 (fake) of which 8 are fake; 10 at 0.2 (so 0.8 confident "real") of which 10 are real.
    probs = np.array([0.9] * 10 + [0.2] * 10)
    labels = np.array([1] * 8 + [0] * 2 + [0] * 10)
    # bin of confidence 0.9: accuracy 0.8, gap 0.1, weight 0.5; bin of confidence 0.8: accuracy 1.0, gap 0.2, weight 0.5
    assert abs(visual_evaluate.expected_calibration_error(probs, labels, n_bins=10) - (0.5 * 0.1 + 0.5 * 0.2)) < 1e-12
    perfect = np.array([1.0] * 5 + [0.0] * 5)
    assert visual_evaluate.expected_calibration_error(perfect, np.array([1] * 5 + [0] * 5)) == 0.0
    curve = visual_evaluate.reliability_curve(probs, labels, n_bins=10)
    assert [round(c["fraction_fake"], 2) for c in curve] == [0.0, 0.8]


def test_temperature_scaling_recovers_a_known_temperature_and_keeps_the_ranking():
    rng = np.random.default_rng(2)
    true_logits = rng.normal(0, 2, 20000)
    labels = (rng.uniform(size=20000) < calibrate.sigmoid(true_logits)).astype(int)
    overconfident = true_logits * 3.0
    temperature = calibrate.fit_temperature(overconfident, labels)
    assert abs(temperature - 3.0) < 0.2
    before = visual_evaluate.expected_calibration_error(calibrate.sigmoid(overconfident), labels)
    after = visual_evaluate.expected_calibration_error(calibrate.calibrated_probs(overconfident, temperature), labels)
    assert after < before / 3
    assert visual_evaluate.fast_roc_auc(overconfident, labels) == visual_evaluate.fast_roc_auc(overconfident / temperature, labels)
    assert calibrate.fit_temperature(np.array([1.0, 2.0]), np.array([1, 1])) == 1.0, "one class: nothing to fit"


def test_operating_point_is_chosen_for_recall_and_the_band_brackets_it():
    probs = np.concatenate([np.linspace(0.30, 0.99, 100), np.linspace(0.01, 0.60, 100)])
    labels = np.array([1] * 100 + [0] * 100)
    point = calibrate.choose_operating_point(probs, labels, target_recall=0.95, band_recall=0.99, band_fpr=0.01)
    recall = ((probs >= point["threshold"]) & (labels == 1)).sum() / 100
    assert recall >= 0.95 and ((probs >= np.nextafter(point["threshold"], 1)) & (labels == 1)).sum() / 100 < 0.95 + 0.011
    assert point["band_low"] <= point["threshold"] <= point["band_high"]
    assert ((probs < point["band_low"]) & (labels == 1)).sum() <= 1, "at most 1% of fakes fall below the band"
    assert ((probs >= point["band_high"]) & (labels == 0)).sum() <= 1, "at most 1% of reals fall above the band"
    verdicts = calibrate.verdicts(probs, point["band_low"], point["band_high"])
    assert set(verdicts) == {"likely_real", "uncertain", "likely_ai"}
    separable = calibrate.choose_operating_point(np.array([0.9, 0.95, 0.1, 0.05]), np.array([1, 1, 0, 0]))
    assert separable["band_low"] <= separable["threshold"] <= separable["band_high"]


# --- embeddings and heads --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_encoder():
    pytest.importorskip("transformers")
    return embed.ClipEncoder(model=embed.tiny_clip_for_smoke(), fp16=False)


def test_embedding_shape_and_determinism(tiny_encoder):
    pixels = torch.stack([preprocess.preprocess_image(picture(i, fake=i % 2 == 1)) for i in range(5)])
    vectors = tiny_encoder.embed(pixels)
    assert vectors.shape == (5, tiny_encoder.dim) == (5, 24) and vectors.dtype == np.float32
    check = embed.embedding_is_deterministic(tiny_encoder, pixels)
    assert check["identical"] and check["max_abs_difference"] == 0.0
    assert not any(p.requires_grad for p in tiny_encoder.model.parameters()) and not tiny_encoder.model.training
    # In a different batch the same image agrees to float rounding (not bit for bit: the matrix kernels differ by batch size).
    assert np.allclose(tiny_encoder.embed(pixels[:2]), vectors[:2], atol=1e-5), "an image's embedding must not depend on its batch"


def test_extraction_is_chunked_resumable_and_writes_no_images(tiny_encoder, tmp_path):
    items = [(f"img{i}", png(picture(i, fake=i % 2 == 1)), ["clean", "aug0"] if i < 4 else ["clean", "jpeg50"]) for i in range(8)]
    items.append(("broken", b"not an image", ["clean"]))
    loads = []

    def chunk(key, part):
        def load():
            loads.append(key)
            return part
        return key, load

    store = embed.ChunkStore(tmp_path / "session1" / "embeddings")
    report = embed.extract_embeddings(tiny_encoder, [chunk("a", items[:5]), chunk("b", items[5:])], store, batch_size=3,
                                      log=lambda *_: None)
    assert report["chunks_done"] == 2 and report["vectors_written"] == 16, "8 images x 2 variants; the broken file is skipped"
    ids, variants, vectors = store.load()
    assert vectors.shape == (16, 24) and set(variants) == {"clean", "aug0", "jpeg50"} and "broken" not in set(ids)
    assert {p.suffix for p in (tmp_path / "session1").rglob("*") if p.is_file()} == {".npz"}, "only vectors reach the disk"
    # A later session that is given the first one's output re-does nothing.
    later = embed.ChunkStore(tmp_path / "session2" / "embeddings", resume_dirs=[tmp_path / "session1" / "embeddings"])
    loads.clear()
    again = embed.extract_embeddings(tiny_encoder, [chunk("a", items[:5]), chunk("b", items[5:]), chunk("c", items[:2])], later,
                                     batch_size=3, log=lambda *_: None)
    assert loads == ["c"] and again["chunks_skipped"] == 2 and again["chunks_done"] == 1
    assert later.load()[2].shape[0] == 16 + 4
    bank = EmbeddingBank(ids, variants, vectors)
    x, found = bank.lookup(["img0", "img7", "broken"], "aug0")
    assert found.tolist() == [True, False, False] and x.shape == (1, 24)
    clean_a, _ = bank.lookup(["img1"], "clean")
    clean_b = tiny_encoder.embed(preprocess.preprocess_image(picture(1, fake=True)).unsqueeze(0))
    assert np.allclose(clean_a, clean_b, atol=1e-5)


def _separable(n=400, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, dim)).astype(np.float32)
    x[:, 0] += 2.5 * (y - 0.5)
    return x, y


def test_heads_learn_and_save_load_restores_the_same_predictions(tmp_path):
    x, y = _separable()
    x_val, y_val = _separable(seed=1)
    head, fit = heads.fit_logreg_head(x, y, x_val, y_val)
    assert fit["validation_auc"] > 0.85 and str(fit["chosen_c"]) in fit["validation_auc_by_c"]
    again, _ = heads.fit_logreg_head(x, y, x_val, y_val)
    assert np.array_equal(head.logits(x_val), again.logits(x_val)), "logistic regression is deterministic given its inputs"
    mlp, mlp_fit = heads.fit_mlp_head(x, y, x_val, y_val, seed=0, hidden=16, epochs=60, patience=60, lr=1e-2)
    assert mlp_fit["validation_auc"] > 0.85
    other_seed, _ = heads.fit_mlp_head(x, y, x_val, y_val, seed=1, hidden=16, epochs=5)
    assert not np.array_equal(mlp.logits(x_val), other_seed.logits(x_val)), "the MLP's seed changes its weights"
    for name, model in (("logreg", head), ("mlp", mlp)):
        digest = heads.save_head(tmp_path / f"{name}.npz", model)
        restored = heads.load_head(tmp_path / f"{name}.npz")
        assert len(digest) == 64 and digest == embed.file_sha256(tmp_path / f"{name}.npz")
        assert type(restored) is type(model) and restored.n_params() == model.n_params()
        assert np.array_equal(restored.logits(x_val), model.logits(x_val)), f"{name}: predictions changed after save and load"


def test_run_fold_end_to_end_on_planted_embeddings(split_table):
    """Embeddings in which every fake family but one shares a direction: the fold that holds that
    family out must show the gap, and say which rows it never trained on."""
    rng = np.random.default_rng(0)
    live = split_table[split_table["valid"]]
    ids, variants, vectors = [], [], []
    for image_id, fake, family in zip(live["image_id"], live["is_fake"], live["generator_family"]):
        for variant in ["clean", "aug0"] + list(preprocess.DEGRADATIONS):
            v = rng.normal(0, 1, 12)
            if fake and family != "midjourney":
                v[0] += 3.0          # the cue the seen families share
            ids.append(image_id)
            variants.append(variant)
            vectors.append(v)
    bank = EmbeddingBank(ids, variants, np.array(vectors, dtype=np.float32))
    seen, _ = run_fold(split_table, bank, "flux", "logreg", seed=0, n_aug=1, n_resamples=100)
    blind, extra = run_fold(split_table, bank, "midjourney", "logreg", seed=0, n_aug=1, n_resamples=100)
    assert seen["suites"]["unseen"]["metrics"]["roc_auc"]["value"] > 0.9
    assert blind["suites"]["unseen"]["metrics"]["roc_auc"]["value"] < 0.7 < blind["suites"]["known"]["metrics"]["roc_auc"]["value"]
    assert blind["known_minus_unseen"]["roc_auc"]["excludes_zero"] and blind["known_minus_unseen"]["roc_auc"]["difference"] > 0
    assert {"known", "unseen", "known/jpeg90", "unseen/jpeg50", "unseen/resize0.5"} <= set(blind["suites"])
    assert blind["counts"]["train_vectors"] == 2 * blind["counts"]["train_images"], "clean + one augmented view per training image"
    assert "kept when the training set is balanced" in blind["seed_changes"]
    frame = extra["frames"]["unseen"]
    assert set(frame.loc[frame["is_fake"] == 1, "generator"]) == {"midjourney-6"} and extra["head"].n_params() == 13
    assert 0 < blind["temperature"] and blind["operating_point"]["n_validation"] == blind["counts"]["val"]
    mlp, _ = run_fold(split_table, bank, "flux", "mlp", seed=0, n_aug=1, n_resamples=50, with_ci=False)
    assert "ci_low" not in mlp["suites"]["known"]["metrics"]["roc_auc"] and "initial weights" in mlp["seed_changes"]


# --- pass 1 / pass 2 over the sources --------------------------------------------------------------
def test_scan_and_load_passes_agree(synthetic_shards, tiny_encoder):
    fs, shards, rows = synthetic_shards
    reader = data.FsRangeReader(fs)
    scanned, stats = scan.scan_openfake_shard(reader, shards[0], keep_fraction={}, threads=2)
    table = _table(scanned)
    assert stats["kept"] == stats["rows_in_shard"] == len(table)
    assert set(table.loc[~table["valid"], "drop_reason"]) <= {"corrupt", "tiny", "grayscale"}
    fakes = table[table["is_fake"] == 1]
    assert fakes["source_image_id"].isna().all() and fakes["prompt_id"].notna().all() and (fakes["generator_family"] != "real").all()
    reals = table[table["is_fake"] == 0]
    assert (reals["source_image_id"] == reals["image_id"]).all() and (reals["generator_family"] == "real").all()
    thinned, _ = scan.scan_openfake_shard(reader, shards[0], keep_fraction={"*": 0.3}, threads=2)
    assert 0 < len(thinned) < len(scanned) and {r["image_id"] for r in thinned} <= set(table["image_id"])
    # Pass 2 fetches the very same bytes by locator.
    pick = table[table["valid"]].head(5)
    wanted = {loc: (i, ["clean"]) for i, loc in zip(pick["image_id"], pick["locator"])}
    items = scan.load_openfake_items(reader, shards[0], wanted)
    assert {i for i, _, _ in items} == set(pick["image_id"])
    for image_id, blob, _ in items:
        row = pick.set_index("image_id").loc[image_id]
        assert f"{preprocess.inspect_image(blob)['phash']:016x}" == row["phash"]
    blobs = scan.fetch_image_bytes([(i, "openfake", loc) for i, loc in zip(pick["image_id"], pick["locator"])], reader,
                                   {s["path"]: s["size"] for s in shards})
    assert set(blobs) == set(pick["image_id"])


# --- dataset terms (spec section 12a) --------------------------------------------------------------
def test_dataset_outputs_are_gitignored_and_the_commit_check_blocks_them():
    import check_no_dataset_outputs as check

    assert check.missing_gitignore_entries() == [], "the repo's .gitignore must list every dataset-derived output"
    blocked = ["artifacts/visual/private/metadata.parquet", "artifacts/visual/private/embeddings/emb-openfake-00001.npz",
               "artifacts/visual/private/predictions_v1.parquet", "artifacts/visual/private/weights/v1_final_logreg.npz",
               "artifacts\\visual\\private\\scan\\rows-folders-00000.parquet", "artifacts/visual/sample.jpg",
               "artifacts/Celeb-DF-v2/video1.mp4", "datasets/FaceForensics++/frames/0001.png", "artifacts/visual/per_image_errors.csv"]
    assert [p for p, _ in check.forbidden_paths(blocked)] == blocked
    allowed = ["artifacts/visual/results_v1.json", "artifacts/visual/preflight_task0.json", "artifacts/visual/reliability_v1.png",
               "artifacts/visual/per_generator_v1.csv", "artifacts/visual/openfake_model_counts.csv", "docs/visual_leaderboard.md",
               "engines/visual/metadata.py", "notebooks/05_visual_v1.ipynb", "tests/test_visual_modules.py"]
    assert check.forbidden_paths(allowed) == []


def test_notebook_never_prints_or_stores_the_token():
    source = (ROOT / "notebooks" / "src" / "05_visual_v1.py").read_text(encoding="utf-8")
    assert 'STAGE = os.environ.get("TRUSTMESH_VISUAL_STAGE") or "task0"' in source, "the default stage must be task0"
    for line in source.splitlines():
        if "_token" in line and ("print(" in line or "save_" in line):
            assert "bool(_token)" in line, f"the token itself must never be printed or saved: {line.strip()}"
