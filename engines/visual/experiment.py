"""One fold, end to end: train a head on cached embeddings, calibrate it on that fold's validation
split, pick the operating point there, then score each test suite once (Tasks 6-8).

Validation holds only generators seen in the fold, so the temperature, the threshold and the
uncertain band never see the held-out family.
"""
import numpy as np
import pandas as pd

from .calibrate import calibrated_probs
from .calibrate import choose_operating_point
from .calibrate import fit_temperature
from .calibrate import sigmoid
from .evaluate import evaluate_suite
from .evaluate import expected_calibration_error
from .evaluate import point_metrics
from .evaluate import unpaired_metric_difference
from .heads import fit_logreg_head
from .heads import fit_mlp_head
from .preprocess import DEGRADATIONS
from .splits import FOLD_PREFIX
from .splits import balanced_subsample

JPEG_SUITES = ("jpeg90", "jpeg70", "jpeg50")


class EmbeddingBank:
    """Cached embeddings addressed by (image id, variant)."""

    def __init__(self, image_ids, variants, embeddings):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.index = {(str(i), str(v)): k for k, (i, v) in enumerate(zip(image_ids, variants))}
        self.dim = int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0

    def variants(self):
        return sorted({v for _, v in self.index})

    def lookup(self, image_ids, variant):
        """(matrix, found mask) for `image_ids` under one variant; images without that embedding
        (for example a file that failed to decode in pass 2) are left out and counted by the caller."""
        rows = [self.index.get((str(i), variant)) for i in image_ids]
        found = np.array([r is not None for r in rows], dtype=bool)
        return self.embeddings[[r for r in rows if r is not None]], found


def fold_rows(table, fold, role):
    return table[table[FOLD_PREFIX + fold] == role]


def train_matrix(bank, rows, n_aug):
    """Training matrix: each image's clean embedding plus its cached augmented views."""
    xs, ys = [], []
    labels = rows["is_fake"].astype(int).to_numpy()
    for variant in ["clean"] + [f"aug{k}" for k in range(n_aug)]:
        x, found = bank.lookup(rows["image_id"], variant)
        xs.append(x)
        ys.append(labels[found])
    return np.concatenate(xs), np.concatenate(ys)


def suite_frame(bank, rows, variant, head, temperature):
    """Per-image predictions of one suite: image id, label, generator, raw and calibrated P(fake)."""
    x, found = bank.lookup(rows["image_id"], variant)
    kept = rows[found]
    logits = head.logits(x) if len(x) else np.zeros(0)
    return pd.DataFrame({"image_id": kept["image_id"].to_numpy(), "is_fake": kept["is_fake"].astype(int).to_numpy(),
                         "generator": kept["generator"].to_numpy(), "prob_raw": sigmoid(logits),
                         "prob": calibrated_probs(logits, temperature)}), int((~found).sum())


def what_a_seed_changes(kind, subsampled):
    """Said in words, because a seed that changes nothing must not be reported as variance."""
    parts = []
    if subsampled:
        parts.append("which images of the larger class are kept when the training set is balanced")
    if kind == "mlp":
        parts.append("the MLP's initial weights and mini-batch order")
    return "; ".join(parts) if parts else "nothing: this fit is deterministic given its inputs, so one run is reported"


def run_fold(table, bank, fold, kind="logreg", seed=0, n_aug=2, n_resamples=1000, with_ci=True,
             target_recall=0.95, band_recall=0.99, band_fpr=0.01, device=None, extra_tests=None):
    """Train, calibrate and score one fold. Returns (result, {"frames": {suite: per-image predictions}, "head": the fitted head}).

    `extra_tests` adds named test sets ({name: rows}) scored like the unseen test -- the old-only
    ablation uses it to be scored on every modern family's unseen-test images.
    """
    train, val = fold_rows(table, fold, "train"), fold_rows(table, fold, "val")
    rng = np.random.default_rng(seed)
    keep = balanced_subsample(train["is_fake"].astype(int).to_numpy(), rng)
    subsampled = len(keep) < len(train)
    train = train.iloc[keep]
    x_train, y_train = train_matrix(bank, train, n_aug)
    x_val, val_found = bank.lookup(val["image_id"], "clean")
    y_val = val["is_fake"].astype(int).to_numpy()[val_found]
    result = {"fold": fold, "kind": kind, "seed": seed, "n_aug_views": n_aug,
              "seed_changes": what_a_seed_changes(kind, subsampled),
              "counts": {"train_images": int(len(train)), "train_vectors": int(len(y_train)),
                         "train_fake_images": int((train["is_fake"] == 1).sum()),
                         "val": int(len(y_val)), "val_fake": int(y_val.sum())}}
    if len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2:
        result["not_run"] = "training or validation split has a single class"
        return result, {"frames": {}, "head": None}
    if kind == "mlp":
        import torch

        head, fit = fit_mlp_head(x_train, y_train, x_val, y_val, seed=seed, device=device or torch.device("cpu"))
    else:
        head, fit = fit_logreg_head(x_train, y_train, x_val, y_val)
    val_logits = head.logits(x_val)
    temperature = fit_temperature(val_logits, y_val)
    val_probs = calibrated_probs(val_logits, temperature)
    operating = choose_operating_point(val_probs, y_val, target_recall, band_recall, band_fpr)
    result.update(fit=fit, temperature=temperature, operating_point=operating, n_params=head.n_params(),
                  validation={"ece_before": expected_calibration_error(sigmoid(val_logits), y_val),
                              "ece_after": expected_calibration_error(val_probs, y_val),
                              "roc_auc": point_metrics(val_probs, y_val, operating["threshold"])["roc_auc"]})

    tests = {"known": fold_rows(table, fold, "known_test"), "unseen": fold_rows(table, fold, "unseen_test")}
    tests.update(extra_tests or {})
    suites, frames, missing = {}, {}, {}
    for test_name, rows in tests.items():
        if len(rows) == 0:
            continue
        for variant in ("clean",) + DEGRADATIONS:
            name = test_name if variant == "clean" else f"{test_name}/{variant}"
            frame, lost = suite_frame(bank, rows, variant, head, temperature)
            if len(frame) == 0:
                continue
            labels, probs = frame["is_fake"].to_numpy(), frame["prob"].to_numpy()
            if with_ci:
                suite = evaluate_suite(probs, labels, operating["threshold"],
                                       frame["generator"].to_numpy() if variant == "clean" else None, n_resamples, seed)
            else:
                suite = {"metrics": {k: {"value": v} for k, v in point_metrics(probs, labels, operating["threshold"]).items()},
                         "counts": {"fake": int(labels.sum()), "real": int((labels == 0).sum())}}
            suite["ece_before_calibration"] = expected_calibration_error(frame["prob_raw"].to_numpy(), labels)
            suite["share_uncertain"] = float(((probs >= operating["band_low"]) & (probs < operating["band_high"])).mean())
            suites[name] = suite
            frames[name] = frame
            missing[name] = lost
    result["suites"] = suites
    result["missing_embeddings"] = {k: v for k, v in missing.items() if v}
    if with_ci and "known" in frames and "unseen" in frames:
        k, u = frames["known"], frames["unseen"]
        result["known_minus_unseen"] = {
            metric: unpaired_metric_difference(k["prob"].to_numpy(), k["is_fake"].to_numpy(), u["prob"].to_numpy(),
                                               u["is_fake"].to_numpy(), metric, operating["threshold"], n_resamples, seed)
            for metric in ("roc_auc", "eer")}
    return result, {"frames": frames, "head": head}


def metric_value(result, suite, metric="roc_auc"):
    """A suite metric's point value from a fold result, or None when that suite was not scored."""
    try:
        return result["suites"][suite]["metrics"][metric]["value"]
    except (KeyError, TypeError):
        return None
