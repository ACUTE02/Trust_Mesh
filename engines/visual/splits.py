"""Group-aware splits with rotating held-out generator families (Task 3; design from
docs/HANDBOOK_ADDENDUM_01_ROTATING_HOLDOUT.md).

Two layers:

1. **Partition, fixed once.** Images are tied into groups by shared `source_image_id`,
   `identity_id` or `prompt_id` (in OpenFake a real photo and the fakes generated from its caption
   share a prompt). Each *group* is placed whole into one of four partitions -- `train`, `val`,
   `known_test`, `holdout` -- by a hash of its key, so the placement is the same in every session.
   The real images of the `holdout` partition are the reserved real pool.

2. **Fold roles, one set per held-out family.** In the fold for family F:
     train / val / known_test = those partitions without any fake of family F;
     unseen_test              = the fakes of family F in the holdout partition, plus an equal
                                number of reserved reals;
     everything else is unused in this fold (family-F fakes outside the holdout partition are
     never trained on; other families' holdout fakes wait for their own fold).

   Because the unseen test is drawn only from the holdout partition, no prompt, source image or
   identity is shared between a fold's training data and its unseen test, and the reserved reals
   are never in any fold's training data.

Embeddings are computed once per image; folds only change which rows the cheap head reads.
"""
import numpy as np
import pandas as pd

from .data import stable_fraction
from .metadata import cluster_pairs, hashes_to_uint64, near_duplicate_pairs

PARTITIONS = ("train", "val", "known_test", "holdout")
GROUP_KEYS = ("source_image_id", "identity_id", "prompt_id")
FOLD_PREFIX = "fold_role__"
OLD_ONLY = "old_only"


def group_ids(table, keys=GROUP_KEYS):
    """A group label per row: rows that share any non-null key value are one group (transitively).
    The label is the smallest image_id in the group, so it is stable across sessions."""
    n = len(table)
    pairs = []
    for key in keys:
        if key not in table.columns:
            continue
        values = table[key].to_numpy()
        first_seen = {}
        for i, value in enumerate(values):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            token = (key, value)
            if token in first_seen:
                pairs.append((first_seen[token], i))
            else:
                first_seen[token] = i
    clusters = cluster_pairs(n, pairs)
    ids = table["image_id"].to_numpy()
    label_of = {}
    for cluster, image_id in zip(clusters, ids):
        if cluster not in label_of or image_id < label_of[cluster]:
            label_of[cluster] = image_id
    return np.array([label_of[c] for c in clusters], dtype=object)


def partition_of(group_key, fractions, seed=0):
    """The partition a group key hashes into. `fractions` maps partition -> share (summing to 1)."""
    u, edge = stable_fraction(group_key, seed), 0.0
    for name in PARTITIONS:
        edge += fractions[name]
        if u < edge:
            return name
    return PARTITIONS[-1]


def assign_partitions(table, fractions, seed=0):
    """Fill `group_id` and `split` for every valid row; invalid rows get no split."""
    table = table.copy()
    table["group_id"] = group_ids(table)
    valid = table["valid"].astype(bool).to_numpy()
    table["split"] = [partition_of(g, fractions, seed) if ok else None for g, ok in zip(table["group_id"], valid)]
    return table


def apply_generator_caps(table, caps, seed=0):
    """Keep at most `caps[partition]` images per (generator, partition); the rest are marked
    dropped with reason `over_cap`. Which images stay is decided by a hash of the image id, so it
    does not depend on row order. `caps` may hold `{'real': {...}, 'fake': {...}}` per-class caps."""
    table = table.copy()
    live = table[table["valid"].astype(bool)]
    order = pd.Series([stable_fraction(i, seed) for i in live["image_id"]], index=live.index)
    drop = []
    for (generator, split, is_fake), members in live.groupby(["generator", "split", "is_fake"]).groups.items():
        per_class = caps.get("fake" if is_fake else "real", caps)
        cap = per_class.get(split)
        if cap is not None and len(members) > cap:
            drop.extend(order.loc[members].sort_values().index[cap:])
    table.loc[drop, "valid"] = False
    table.loc[drop, "drop_reason"] = "over_cap"
    table.loc[drop, "split"] = None
    return table


def family_counts(table):
    """Valid fake images per family and partition."""
    fakes = table[table["valid"].astype(bool) & (table["is_fake"] == 1)]
    counts = fakes.groupby(["generator_family", "split"]).size().unstack(fill_value=0)
    return counts.reindex(columns=list(PARTITIONS), fill_value=0)


def eligible_families(table, min_unseen_fakes, min_reserved_reals=None):
    """Which families can be held out: a family needs at least `min_unseen_fakes` fakes in the
    holdout partition (its unseen test) and the reserved pool needs as many reals. Returns
    {family: {..counts.., 'eligible': bool, 'why_not': str or None}}."""
    counts = family_counts(table)
    live = table[table["valid"].astype(bool)]
    reserved = int(((live["is_fake"] == 0) & (live["split"] == "holdout")).sum())
    need_reals = min_unseen_fakes if min_reserved_reals is None else min_reserved_reals
    out = {}
    for family, row in counts.iterrows():
        why = None
        if row["holdout"] < min_unseen_fakes:
            why = f"only {int(row['holdout'])} holdout fakes (< {min_unseen_fakes})"
        elif reserved < need_reals:
            why = f"only {reserved} reserved reals (< {need_reals})"
        elif int(counts["train"].sum() - row["train"]) == 0:
            why = "no other family left to train on"
        out[str(family)] = {**{p: int(row[p]) for p in PARTITIONS}, "reserved_reals": reserved,
                            "eligible": why is None, "why_not": why}
    return out


def fold_roles(table, family, seed=0, reals_per_fake=1.0):
    """The role of every row in the fold that holds out `family` (see the module docstring)."""
    valid = table["valid"].astype(bool).to_numpy()
    split = table["split"].to_numpy()
    fake = (table["is_fake"] == 1).to_numpy()
    held = (table["generator_family"] == family).to_numpy() & fake
    roles = np.full(len(table), None, dtype=object)
    for name in ("train", "val", "known_test"):
        roles[valid & (split == name) & ~held] = name
    roles[valid & (split != "holdout") & held] = "unused"
    in_holdout = valid & (split == "holdout")
    roles[in_holdout & fake & ~held] = "unused"
    roles[in_holdout & held] = "unseen_test"
    reserved = np.flatnonzero(in_holdout & ~fake)
    roles[reserved] = "reserved_real"
    n_take = min(len(reserved), int(round(reals_per_fake * int((in_holdout & held).sum()))))
    ids = table["image_id"].to_numpy()
    order = sorted(reserved, key=lambda i: stable_fraction(ids[i], f"{seed}|{family}"))
    roles[np.array(order[:n_take], dtype=int)] = "unseen_test"
    return roles


def old_only_roles(table, old_families):
    """The old-only ablation: train and validate on real images plus fakes from `old_families`
    only. Its tests are the unseen-test sets of the LOFO folds of the modern families, so the
    ablation and the folds are scored on identical images."""
    valid = table["valid"].astype(bool).to_numpy()
    split = table["split"].to_numpy()
    fake = (table["is_fake"] == 1).to_numpy()
    old = table["generator_family"].isin(list(old_families)).to_numpy()
    roles = np.full(len(table), None, dtype=object)
    roles[valid] = "unused"
    for name in ("train", "val", "known_test"):
        roles[valid & (split == name) & (~fake | old)] = name
    roles[valid & (split == "holdout") & ~fake] = "reserved_real"
    return roles


def add_fold_roles(table, families, seed=0, old_families=None):
    """One `fold_role__<family>` column per held-out family, plus `fold_role__old_only`."""
    table = table.copy()
    for family in families:
        table[FOLD_PREFIX + str(family)] = fold_roles(table, family, seed)
    if old_families is not None:
        table[FOLD_PREFIX + OLD_ONLY] = old_only_roles(table, old_families)
    return table


def fold_names(table):
    return [c[len(FOLD_PREFIX):] for c in table.columns if c.startswith(FOLD_PREFIX)]


def class_balance(table, fold):
    """Real/fake counts per role, and per generator within each role, for one fold."""
    column = FOLD_PREFIX + fold
    used = table[table[column].notna() & (table[column] != "unused")]
    per_role = used.groupby([column, "is_fake"]).size().unstack(fill_value=0)
    per_generator = used.groupby([column, "generator"]).size()
    return {"per_role": {str(role): {("fake" if k else "real"): int(v) for k, v in row.items()}
                         for role, row in per_role.iterrows()},
            "per_generator": {f"{role}/{gen}": int(v) for (role, gen), v in per_generator.items()}}


def leakage_report(table, phash_tolerance=0, strict=True):
    """The per-fold leakage checks of the addendum, section 5. Returns a report; with `strict`
    raises AssertionError listing every violation.

      1. the held-out family has zero rows in that fold's train and validation;
      2. reserved reals (real images of the holdout partition) are in no training or validation
         split of any fold;
      3. no perceptual hash (within `phash_tolerance` bits), source_image_id, identity_id or
         prompt_id is shared between two different splits of a fold;
      5. class balance per split and per generator (returned for printing).
    (Check 4, the shortcut-baseline AUC, is reported by baselines.py.)
    """
    violations, report = [], {"folds": {}, "phash_tolerance": int(phash_tolerance)}
    valid = table["valid"].astype(bool)
    reserved = valid & (table["is_fake"] == 0) & (table["split"] == "holdout")
    for fold in fold_names(table):
        column = FOLD_PREFIX + fold
        roles = table[column]
        entry = {"balance": class_balance(table, fold)}
        if fold != OLD_ONLY:
            held = (table["generator_family"] == fold) & (table["is_fake"] == 1)
            entry["held_out_rows_in_train_val"] = int((held & roles.isin(["train", "val"])).sum())
            if entry["held_out_rows_in_train_val"]:
                violations.append(f"fold {fold}: {entry['held_out_rows_in_train_val']} held-out rows in train/val")
        entry["reserved_reals_in_train_val"] = int((reserved & roles.isin(["train", "val", "known_test"])).sum())
        if entry["reserved_reals_in_train_val"]:
            violations.append(f"fold {fold}: {entry['reserved_reals_in_train_val']} reserved reals outside the unseen test")
        used = table[roles.isin(["train", "val", "known_test", "unseen_test"])]
        entry["shared_keys"] = {}
        for key in ("source_image_id", "identity_id", "prompt_id", "group_id"):
            spread = used[used[key].notna()].groupby(key)[column].nunique()
            entry["shared_keys"][key] = int((spread > 1).sum())
            if entry["shared_keys"][key]:
                violations.append(f"fold {fold}: {entry['shared_keys'][key]} {key} values span more than one split")
        hashed = used[used["phash"].notna()].reset_index(drop=True)
        pairs = near_duplicate_pairs(hashes_to_uint64(hashed["phash"]), phash_tolerance) if len(hashed) else []
        split_of = hashed[column].to_numpy() if len(hashed) else np.array([])
        entry["phash_pairs_across_splits"] = int(sum(1 for i, j, _ in pairs if split_of[i] != split_of[j]))
        if entry["phash_pairs_across_splits"]:
            violations.append(f"fold {fold}: {entry['phash_pairs_across_splits']} near-duplicate pairs across splits")
        report["folds"][fold] = entry
    report["violations"] = violations
    if strict and violations:
        raise AssertionError("leakage checks failed: " + "; ".join(violations))
    return report


def balanced_subsample(labels, rng):
    """Indices that keep every example of the smaller class and an equal-sized random sample of
    the larger one. This is the one place a seed changes which training rows a head sees."""
    labels = np.asarray(labels)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n = min(len(pos), len(neg))
    if len(pos) == len(neg) or n == 0:
        return np.arange(len(labels))
    keep_pos = pos if len(pos) == n else rng.choice(pos, n, replace=False)
    keep_neg = neg if len(neg) == n else rng.choice(neg, n, replace=False)
    return np.sort(np.concatenate([keep_pos, keep_neg]))


def auc_ci_halfwidth(auc, n_pos, n_neg):
    """Approximate 95% half-width of a ROC-AUC confidence interval (Hanley & McNeil, 1982). Used
    only to judge, before any model exists, how many test images make a fold's CI usefully narrow."""
    if n_pos < 1 or n_neg < 1:
        return float("nan")
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return float(1.96 * np.sqrt(max(variance, 0.0)))
