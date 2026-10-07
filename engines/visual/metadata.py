"""The per-image metadata table (Task 1), generator families, and perceptual-hash dedupe (Task 2).

One row per image. Every leakage check, per-generator metric and error table reads this table.
Fields that are not known stay null and are counted (`null_counts`), never invented. The table
lists individual dataset images, so under the datasets' terms it is never committed to a public
repository (see `scripts/check_no_dataset_outputs.py`).
"""
import re

import numpy as np
import pandas as pd

METADATA_COLUMNS = [
    "image_id", "source_dataset", "generator", "generator_family", "is_fake", "manipulation_type",
    "source_image_id", "identity_id", "prompt_id", "width", "height", "format", "jpeg_quality",
    "file_size", "has_exif", "phash", "split", "license",
    # Not in the handbook's list, needed by this design:
    "release_date", "pipeline_file_size", "valid", "drop_reason", "dup_cluster", "group_id", "locator",
]
# `fold_role` is one column per fold, named `fold_role__<family>` (see splits.add_fold_roles).

# Licenses as found on each dataset's own page, with where and when they were read. A conflict is
# recorded as a conflict, not resolved here.
LICENSES = {
    "openfake": "cc-by-nc-4.0 (dataset card, read 2026-10-07); the handbook reports CC BY-NC-SA 4.0 in the paper "
                "(not re-verified)",
    "artifact": "not yet looked up (Task 0 records the license field of the Kaggle copy)",
    "ffhq": "Kaggle re-upload license field 'Unknown' (as seen by the user, 2026-10-06); source NVlabs/ffhq-dataset "
            "lists CC BY-NC-SA 4.0",
}

# Proposed generator -> family rules, first match wins. A family groups generators that share an
# architecture lineage, so holding a family out removes every close relative from training.
# These are a starting proposal written from the OpenFake dataset card; they are confirmed (or
# changed) against the real `model` values that Task 0 prints, and the outcome is logged in
# docs/design_decisions.md.
FAMILY_RULES = [
    (r"flux", "flux"),
    (r"^midjourney", "midjourney"),
    (r"^imagen", "imagen"),
    (r"nano-banana", "nano-banana"),
    (r"^gpt-image", "gpt-image"),
    (r"dall", "dall-e"),
    (r"^ideogram", "ideogram"),
    (r"^recraft", "recraft"),
    (r"^grok", "grok"),
    (r"^sdxl|juggernaut|realvis", "sdxl"),
    (r"^sd-3", "sd-3"),
    (r"^sd-[12]|realistic-vision|dreamshaper|epicdream", "sd-1x-2x"),
    (r"^stylegan", "stylegan"),
]


def family_of(generator):
    """(family, rule) for a generator name. `rule` is the pattern that matched, or `fallback` when
    none did -- then the family is the name with its trailing version removed, and the generator
    is listed for a human to check."""
    name = str(generator).strip().lower()
    for pattern, family in FAMILY_RULES:
        if re.search(pattern, name):
            return family, pattern
    stem = re.sub(r"[-_. ]?v?\d+(\.\d+)*[a-z]?$", "", name)
    return (stem or name), "fallback"


def empty_row():
    return {c: None for c in METADATA_COLUMNS}


def phash_hex(value):
    """A 64-bit hash as a 16-character hex string (None stays None)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value if isinstance(value, str) else f"{int(value):016x}"


def build_table(rows):
    """Rows (dicts) -> the metadata DataFrame with every schema column present.

    The 64-bit hash is stored as a 16-character hex string, converted *before* the rows become a
    DataFrame: a numeric column that also holds a missing value is turned into floats by pandas,
    and a float cannot hold 64 bits exactly, so the low bits of every hash would be lost.
    """
    rows = [{**empty_row(), **row, "phash": phash_hex(row.get("phash"))} for row in rows]
    table = pd.DataFrame(rows)
    for column in METADATA_COLUMNS:
        if column not in table.columns:
            table[column] = None
    return table


def null_counts(table, columns=METADATA_COLUMNS):
    """How many rows have no value, per column -- the 'unknown fields are counted' rule."""
    return {c: int(table[c].isna().sum()) for c in columns if c in table.columns}


# ---------------------------------------------------------------------------------------------------
# Near-duplicate search on 64-bit perceptual hashes
# ---------------------------------------------------------------------------------------------------
_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(values):
    """Number of set bits in each uint64."""
    values = np.asarray(values, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values).astype(np.int64)
    return _POPCOUNT8[values.view(np.uint8).reshape(values.shape + (8,))].sum(axis=-1).astype(np.int64)


def hashes_to_uint64(hex_hashes):
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


def near_duplicate_pairs(hashes, max_distance, block=2048):
    """All pairs (i < j) of hashes within `max_distance` bits, with their distance.

    Exact search without comparing every pair: a 64-bit hash is cut into 8 bytes, and two hashes
    that differ in at most 7 bits must agree on at least one whole byte. So only hashes sharing a
    byte are compared. (Hence the limit of 7.)
    """
    if max_distance > 7:
        raise ValueError("near_duplicate_pairs is exact only up to a distance of 7")
    hashes = np.asarray(hashes, dtype=np.uint64)
    found = {}
    for band in range(8):
        keys = ((hashes >> np.uint64(8 * band)) & np.uint64(0xFF)).astype(np.int64)
        order = np.argsort(keys, kind="stable")
        bounds = np.flatnonzero(np.diff(keys[order])) + 1
        for members in np.split(order, bounds):
            if len(members) < 2:
                continue
            members = np.sort(members)
            values = hashes[members]
            for start in range(0, len(members), block):
                distance = popcount64(values[start:start + block, None] ^ values[None, :])
                rows, cols = np.nonzero(distance <= max_distance)
                keep = cols > rows + start
                for r, c in zip(rows[keep], cols[keep]):
                    found[(int(members[r + start]), int(members[c]))] = int(distance[r, c])
    if not found:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(sorted((i, j, d) for (i, j), d in found.items()), dtype=np.int64)


def cluster_pairs(n, pairs):
    """Union-find over index pairs -> a cluster label per item (the smallest index in its cluster)."""
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def distance_histogram(table, max_distance=7):
    """Number of image pairs at each hash distance 0..max_distance among valid rows -- the evidence
    the dedupe tolerance is chosen from (a jump in the count marks where unrelated images start to
    collide)."""
    valid = table[table["valid"] & table["phash"].notna()]
    pairs = near_duplicate_pairs(hashes_to_uint64(valid["phash"]), max_distance)
    counts = np.bincount(pairs[:, 2], minlength=max_distance + 1) if len(pairs) else np.zeros(max_distance + 1, int)
    return {int(d): int(c) for d, c in enumerate(counts)}


def duplicate_examples(table, max_distance=7, per_distance=4):
    """A few (image_id, image_id, distance) pairs at each distance, for looking at real duplicate
    pairs before choosing the tolerance."""
    valid = table[table["valid"] & table["phash"].notna()].reset_index(drop=True)
    pairs = near_duplicate_pairs(hashes_to_uint64(valid["phash"]), max_distance)
    out = []
    for d in range(max_distance + 1):
        for i, j, _ in pairs[pairs[:, 2] == d][:per_distance]:
            out.append((valid["image_id"][i], valid["image_id"][j], d))
    return out


def dedupe_table(table, tolerance):
    """Mark exact and near duplicates (hash distance <= tolerance) as dropped.

    Each cluster of duplicates keeps one image (the smallest image_id). A cluster that mixes real
    and fake images is dropped whole: the same picture carrying both labels has no usable label.
    Returns (table, report). Rows are marked (`valid` False, `drop_reason` 'duplicate'), not deleted,
    so the counts stay auditable.
    """
    table = table.copy()
    live = np.flatnonzero((table["valid"] & table["phash"].notna()).to_numpy())
    hashes = hashes_to_uint64(table["phash"].iloc[live])
    pairs = near_duplicate_pairs(hashes, tolerance)
    clusters = cluster_pairs(len(live), pairs[:, :2])
    table["dup_cluster"] = None
    report = {"tolerance": int(tolerance), "images_checked": int(len(live)), "pairs_within_tolerance": int(len(pairs)),
              "clusters": 0, "removed": 0, "mixed_label_clusters": 0, "cross_source_clusters": 0,
              "removed_by_source": {}}
    ids = table["image_id"].iloc[live].to_numpy()
    fake = table["is_fake"].iloc[live].to_numpy()
    source = table["source_dataset"].iloc[live].to_numpy()
    drop = []
    for label in np.unique(clusters):
        members = np.flatnonzero(clusters == label)
        if len(members) < 2:
            continue
        report["clusters"] += 1
        table.iloc[live[members], table.columns.get_loc("dup_cluster")] = str(ids[members].min())
        report["cross_source_clusters"] += int(len(set(source[members])) > 1)
        if len(set(fake[members])) > 1:
            report["mixed_label_clusters"] += 1
            drop.extend(members)
        else:
            keeper = members[np.argmin(ids[members])]
            drop.extend(m for m in members if m != keeper)
    rows = live[np.array(drop, dtype=int)] if drop else np.array([], dtype=int)
    table.iloc[rows, table.columns.get_loc("valid")] = False
    table.iloc[rows, table.columns.get_loc("drop_reason")] = "duplicate"
    report["removed"] = int(len(rows))
    report["removed_by_source"] = {str(k): int(v) for k, v in table.iloc[rows]["source_dataset"].value_counts().items()}
    return table, report


def cleaning_report(table):
    """Counts of what cleaning dropped and why, per source."""
    dropped = table[~table["valid"].astype(bool)]
    return {"rows": int(len(table)), "valid": int(table["valid"].astype(bool).sum()),
            "dropped_by_reason": {str(k): int(v) for k, v in dropped["drop_reason"].value_counts().items()},
            "dropped_by_source": {str(k): int(v) for k, v in dropped["source_dataset"].value_counts().items()}}
