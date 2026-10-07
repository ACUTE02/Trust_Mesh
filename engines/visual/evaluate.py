"""Metrics and confidence intervals for the Visual Engine.

Convention: **fake is the positive class** (label 1) and a score is P(fake).
  FPR = real images flagged as fake;  FNR = fake images that got through.

The stratified bootstrap and the EER come from the Voice Engine's evaluate.py (the code fixed for
issue #16) rather than being written again.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from ..voice.evaluate import _stratified_resample
from ..voice.evaluate import compute_eer

FAKE_LABEL = 1
NOT_RUN = "not yet run"
SUITE_METRICS = ("roc_auc", "pr_auc", "eer", "fpr", "fnr", "ece")


def fast_roc_auc(scores, labels):
    """ROC-AUC from ranks (ties averaged). NaN when only one class is present."""
    labels = np.asarray(labels)
    pos = labels == FAKE_LABEL
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: over equal-width confidence bins, the average gap between confidence and accuracy,
    weighted by how many predictions fall in each bin. Confidence is the probability of the
    predicted class, max(p, 1 - p)."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    if len(probs) == 0:
        return float("nan")
    confidence = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype(int) == labels).astype(np.float64)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        members = bins == b
        total += members.mean() * abs(correct[members].mean() - confidence[members].mean())
    return float(total)


def reliability_curve(probs, labels, n_bins=10):
    """Points for a reliability diagram: per bin of P(fake), the mean prediction, the observed
    fraction of fakes and the count."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    return [{"bin": int(b), "mean_predicted": float(probs[bins == b].mean()),
             "fraction_fake": float((labels[bins == b] == FAKE_LABEL).mean()), "count": int((bins == b).sum())}
            for b in np.unique(bins)]


def confusion_at(probs, labels, threshold):
    """Counts at `threshold` (predict fake when P(fake) >= threshold)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    flagged, fake = probs >= threshold, labels == FAKE_LABEL
    return {"tp": int((flagged & fake).sum()), "fn": int((~flagged & fake).sum()),
            "fp": int((flagged & ~fake).sum()), "tn": int((~flagged & ~fake).sum())}


def point_metrics(probs, labels, threshold):
    """Every suite metric as a plain number. A metric that needs both classes is NaN without them."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    c = confusion_at(probs, labels, threshold)
    both = (c["tp"] + c["fn"]) > 0 and (c["fp"] + c["tn"]) > 0
    return {
        "roc_auc": fast_roc_auc(probs, labels),
        "pr_auc": float(average_precision_score(labels == FAKE_LABEL, probs)) if both else float("nan"),
        "eer": float(compute_eer(probs, labels, FAKE_LABEL)["eer"]) if both else float("nan"),
        "fpr": c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else float("nan"),
        "fnr": c["fn"] / (c["fn"] + c["tp"]) if (c["fn"] + c["tp"]) else float("nan"),
        "ece": expected_calibration_error(probs, labels),
    }


def _resample_indices(labels, rng):
    """Stratified resample; falls back to a plain one when a class is missing (fake-only sets)."""
    labels = np.asarray(labels)
    if (labels == FAKE_LABEL).all() or (labels != FAKE_LABEL).all():
        return rng.choice(len(labels), len(labels))
    return _stratified_resample(labels, rng, FAKE_LABEL)


def bootstrap_metric_cis(probs, labels, threshold, n_resamples=1000, seed=0, alpha=0.05, metrics=SUITE_METRICS):
    """{metric: {value, ci_low, ci_high}} with percentile bootstrap CIs, resampling within each
    class so the class counts never change."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    point = point_metrics(probs, labels, threshold)
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in metrics}
    for _ in range(n_resamples):
        idx = _resample_indices(labels, rng)
        values = point_metrics(probs[idx], labels[idx], threshold)
        for m in metrics:
            samples[m].append(values[m])
    out = {}
    for m in metrics:
        finite = np.array([v for v in samples[m] if np.isfinite(v)])
        low, high = (np.quantile(finite, [alpha / 2, 1 - alpha / 2]) if len(finite) else (np.nan, np.nan))
        out[m] = {"value": float(point[m]), "ci_low": float(low), "ci_high": float(high)}
    return out


def rate_ci(flags, n_resamples=1000, seed=0, alpha=0.05):
    """A proportion (for example the share of fakes detected) with a bootstrap CI."""
    flags = np.asarray(flags, dtype=np.float64)
    if len(flags) == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = flags[rng.integers(0, len(flags), size=(n_resamples, len(flags)))].mean(axis=1)
    return {"value": float(flags.mean()), "ci_low": float(np.quantile(means, alpha / 2)),
            "ci_high": float(np.quantile(means, 1 - alpha / 2)), "n": int(len(flags))}


def per_generator_results(probs, labels, generators, threshold, n_resamples=1000, seed=0):
    """Per fake generator: detection rate at the threshold, and ROC-AUC of that generator's fakes
    against all the suite's real images. Per real source: the false-positive rate."""
    probs, labels, generators = np.asarray(probs, dtype=np.float64), np.asarray(labels), np.asarray(generators)
    real = labels != FAKE_LABEL
    out = {}
    for g in sorted(set(generators.tolist())):
        members = generators == g
        if labels[members].max() == FAKE_LABEL:
            subset = members | real
            auc = bootstrap_metric_cis(probs[subset], labels[subset], threshold, n_resamples, seed, metrics=("roc_auc",))
            out[str(g)] = {"kind": "fake", "n": int(members.sum()), "roc_auc_vs_all_reals": auc["roc_auc"],
                           "detection_rate": rate_ci(probs[members] >= threshold, n_resamples, seed)}
        else:
            out[str(g)] = {"kind": "real", "n": int(members.sum()),
                           "false_positive_rate": rate_ci(probs[members] >= threshold, n_resamples, seed)}
    return out


def evaluate_suite(probs, labels, threshold, generators=None, n_resamples=1000, seed=0):
    """Everything reported for one test suite: metrics with CIs, the confusion matrix at the
    operating threshold, class counts and per-generator results."""
    labels = np.asarray(labels)
    result = {"metrics": bootstrap_metric_cis(probs, labels, threshold, n_resamples, seed),
              "confusion": confusion_at(probs, labels, threshold), "threshold": float(threshold),
              "counts": {"fake": int((labels == FAKE_LABEL).sum()), "real": int((labels != FAKE_LABEL).sum())},
              "n_resamples": int(n_resamples)}
    if generators is not None:
        result["per_generator"] = per_generator_results(probs, labels, generators, threshold, n_resamples, seed)
    return result


def paired_metric_difference(probs_a, probs_b, labels, metric="roc_auc", threshold=0.5, n_resamples=1000, seed=0,
                             alpha=0.05):
    """CI of metric(a) - metric(b) for two models scored on the same images (each resample uses
    one set of indices for both). `excludes_zero` False means the two are a tie at this sample size."""
    a, b, labels = np.asarray(probs_a, dtype=np.float64), np.asarray(probs_b, dtype=np.float64), np.asarray(labels)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = _resample_indices(labels, rng)
        diffs.append(point_metrics(a[idx], labels[idx], threshold)[metric] -
                     point_metrics(b[idx], labels[idx], threshold)[metric])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point = point_metrics(a, labels, threshold)[metric] - point_metrics(b, labels, threshold)[metric]
    return {"metric": metric, "difference": float(point), "ci_low": float(low), "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0), "verdict": "different" if (low > 0 or high < 0) else "tie"}


def results_template(models, suites):
    """The reporting grid: one row per model, one column per suite, every cell 'not yet run'
    until a real run fills it."""
    return {model: {suite: NOT_RUN for suite in suites} for model in models}


def mean_std(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)) if len(values) > 1 else None, "n": len(values)}


def unpaired_metric_difference(probs_a, labels_a, probs_b, labels_b, metric="roc_auc", threshold=0.5,
                               n_resamples=1000, seed=0, alpha=0.05):
    """CI of metric(a) - metric(b) for two different test sets (each resampled on its own) -- the
    known-minus-unseen gap of a fold."""
    a, la = np.asarray(probs_a, dtype=np.float64), np.asarray(labels_a)
    b, lb = np.asarray(probs_b, dtype=np.float64), np.asarray(labels_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        ia, ib = _resample_indices(la, rng), _resample_indices(lb, rng)
        diffs.append(point_metrics(a[ia], la[ia], threshold)[metric] - point_metrics(b[ib], lb[ib], threshold)[metric])
    diffs = [d for d in diffs if np.isfinite(d)]
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2]) if diffs else (np.nan, np.nan)
    point = point_metrics(a, la, threshold)[metric] - point_metrics(b, lb, threshold)[metric]
    return {"metric": metric, "difference": float(point), "ci_low": float(low), "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0)}
