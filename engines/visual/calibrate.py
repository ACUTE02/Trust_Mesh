"""Calibration, the operating threshold and the uncertain band (Task 7).

All three are fitted on a fold's validation split only, which holds seen generators only. Whether
that calibration still holds on an unseen family is measured (ECE per suite), not assumed.

For TrustMesh's KYC screening a missed fake costs more than a false alarm, so the operating
threshold is the one that catches a target share of validation fakes, and anything inside the
uncertain band goes to a person. The visual score never rejects anyone on its own.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def binary_nll(logits, labels):
    """Mean negative log-likelihood of binary labels under sigmoid(logits)."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_temperature(logits, labels, bounds=(0.05, 20.0)):
    """The single number T > 0 that minimizes the NLL of sigmoid(logits / T) on validation data.
    T > 1 means the model was over-confident. Ranking (ROC-AUC) is unchanged by it."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return 1.0
    result = minimize_scalar(lambda log_t: binary_nll(logits / np.exp(log_t), labels),
                             bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(result.x))


def calibrated_probs(logits, temperature):
    return sigmoid(np.asarray(logits, dtype=np.float64) / temperature)


def threshold_for_recall(probs, labels, target_recall):
    """The highest threshold at which at least `target_recall` of the fakes are still flagged."""
    fakes = np.sort(np.asarray(probs, dtype=np.float64)[np.asarray(labels) == 1])
    if len(fakes) == 0:
        return 0.5
    allowed_misses = int(np.floor((1.0 - target_recall) * len(fakes) + 1e-9))
    return float(fakes[allowed_misses])


def threshold_for_fpr(probs, labels, target_fpr):
    """The lowest threshold at which at most `target_fpr` of the real images are flagged."""
    reals = np.sort(np.asarray(probs, dtype=np.float64)[np.asarray(labels) == 0])
    if len(reals) == 0:
        return 0.5
    allowed = int(np.floor(target_fpr * len(reals) + 1e-9))
    return float(np.nextafter(reals[len(reals) - 1 - allowed], np.inf))


def choose_operating_point(probs, labels, target_recall=0.95, band_recall=0.99, band_fpr=0.01):
    """Operating threshold and uncertain band from validation predictions.

      threshold : catches `target_recall` of validation fakes (the KYC setting: favour recall);
      band low  : below it, at most 1 - `band_recall` of validation fakes are found ("likely real");
      band high : above it, at most `band_fpr` of validation reals are found ("likely AI");
      in between: "uncertain", sent to a person.

    If the model separates validation so well that low would exceed high, the band collapses onto
    the threshold. The targets are settings, logged with the result, not facts about the data.
    """
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    threshold = threshold_for_recall(probs, labels, target_recall)
    low = threshold_for_recall(probs, labels, band_recall)
    high = threshold_for_fpr(probs, labels, band_fpr)
    if low > high:
        low = high = threshold
    low, high = min(low, threshold), max(high, threshold)
    return {"threshold": float(threshold), "band_low": float(low), "band_high": float(high),
            "target_recall": target_recall, "band_recall": band_recall, "band_fpr": band_fpr,
            "validation_share_uncertain": float(((probs >= low) & (probs < high)).mean()) if len(probs) else None,
            "n_validation": int(len(probs))}


def verdicts(probs, band_low, band_high):
    """`likely_real`, `uncertain` or `likely_ai` per image."""
    probs = np.asarray(probs, dtype=np.float64)
    return np.where(probs < band_low, "likely_real", np.where(probs >= band_high, "likely_ai", "uncertain"))
