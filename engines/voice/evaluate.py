"""EER, DET curve and the silence-leakage probe.

Convention throughout: score = "how bonafide is this", label 1 = bonafide (the target class).
  FAR (false accept rate) = spoof clips scored above threshold  -> accepted as bonafide
  FRR (false reject rate)  = bonafide clips scored below threshold -> rejected as spoof
EER is where those two curves cross, found by linear interpolation between the two bracketing
operating points rather than by taking the nearest point, which is only an approximation.
"""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(scores, labels, positive_label=1):
    """Equal Error Rate. Returns eer, the threshold at it, and FAR/FRR there."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    y = (labels == positive_label).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return {"eer": float("nan"), "threshold": float("nan"), "far": float("nan"),
                "frr": float("nan"), "note": "only one class present; EER undefined"}

    fpr, tpr, thresholds = roc_curve(y, scores)
    far, frr = fpr, 1.0 - tpr  # FAR rises with threshold-looseness, FRR falls
    diff = far - frr
    sign_change = np.flatnonzero(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)
    if len(sign_change) == 0:  # curves touch rather than cross (ties/small samples)
        idx = int(np.nanargmin(np.abs(diff)))
        return {"eer": float((far[idx] + frr[idx]) / 2), "threshold": float(thresholds[idx]),
                "far": float(far[idx]), "frr": float(frr[idx]), "interpolated": False}
    i = int(sign_change[0])
    # Linear interpolation of the crossing between operating points i and i+1.
    weight = diff[i] / (diff[i] - diff[i + 1])
    eer = float(far[i] + weight * (far[i + 1] - far[i]))
    frr_at = float(frr[i] + weight * (frr[i + 1] - frr[i]))
    threshold = float(thresholds[i] + weight * (thresholds[i + 1] - thresholds[i]))
    return {"eer": (eer + frr_at) / 2, "threshold": threshold, "far": eer, "frr": frr_at,
            "interpolated": True}


def det_curve_points(scores, labels, positive_label=1, max_points=512):
    """FAR/FRR pairs for plotting a DET curve, thinned to at most `max_points`."""
    y = (np.asarray(labels) == positive_label).astype(int)
    fpr, tpr, thresholds = roc_curve(y, np.asarray(scores, dtype=np.float64))
    far, frr = fpr, 1.0 - tpr
    if len(far) > max_points:
        keep = np.unique(np.linspace(0, len(far) - 1, max_points).astype(int))
        far, frr, thresholds = far[keep], frr[keep], thresholds[keep]
    return {"far": far.tolist(), "frr": frr.tolist(), "thresholds": thresholds.tolist()}


def per_class_accuracy(scores, labels, threshold, positive_label=1):
    scores, labels = np.asarray(scores), np.asarray(labels)
    predicted_positive = scores > threshold
    is_positive = labels == positive_label
    return {
        "threshold": float(threshold),
        "bonafide_accuracy": float(predicted_positive[is_positive].mean()) if is_positive.any() else None,
        "spoof_accuracy": float((~predicted_positive[~is_positive]).mean()) if (~is_positive).any() else None,
        "overall_accuracy": float((predicted_positive == is_positive).mean()),
        "n_bonafide": int(is_positive.sum()),
        "n_spoof": int((~is_positive).sum()),
    }


def _stratified_resample(labels, rng, positive_label=1):
    """Resample with replacement *within* each class, so every resample keeps the original class
    counts -- a plain resample could drop the (small) bonafide class and leave EER undefined."""
    pos = np.flatnonzero(labels == positive_label)
    neg = np.flatnonzero(labels != positive_label)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_eer_ci(scores, labels, n_resamples=1000, seed=0, alpha=0.05, positive_label=1):
    """Percentile bootstrap CI for EER, stratified by class."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = _stratified_resample(labels, rng, positive_label)
        samples.append(compute_eer(scores[idx], labels[idx], positive_label)["eer"])
    samples = np.array(samples)
    return {
        "eer": compute_eer(scores, labels, positive_label)["eer"],
        "ci_low": float(np.quantile(samples, alpha / 2)),
        "ci_high": float(np.quantile(samples, 1 - alpha / 2)),
        "ci_width": float(np.quantile(samples, 1 - alpha / 2) - np.quantile(samples, alpha / 2)),
        "bootstrap_std": float(samples.std()),
        "n_resamples": n_resamples,
        "n_bonafide": int((labels == positive_label).sum()),
        "n_spoof": int((labels != positive_label).sum()),
    }


def bootstrap_ci_at_counts(scores, labels, n_bonafide, n_spoof, n_resamples=1000, seed=0, alpha=0.05,
                           positive_label=1):
    """CI width the same scores would have had with a smaller test set of the given class counts.
    Used to say how much the larger Phase 2B test set narrowed the CI versus Phase 2's, on the same
    model, instead of comparing CIs from different models."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    pos = np.flatnonzero(labels == positive_label)
    neg = np.flatnonzero(labels != positive_label)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, n_bonafide), rng.choice(neg, n_spoof)])
        samples.append(compute_eer(scores[idx], labels[idx], positive_label)["eer"])
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"n_bonafide": int(n_bonafide), "n_spoof": int(n_spoof), "ci_low": float(low),
            "ci_high": float(high), "ci_width": float(high - low)}


def bootstrap_eer_difference(scores_a, labels_a, scores_b, labels_b, n_resamples=1000, seed=0,
                             alpha=0.05, paired=False, positive_label=1):
    """Percentile bootstrap CI for EER(a) - EER(b).

    paired=True: a and b are two models scored on the *same* clips in the same order, so each resample
    draws one set of indices used for both -- this cancels the test-set sampling noise the two share.
    paired=False: two different test sets (e.g. a random split vs. a speaker-disjoint split), each
    resampled independently.
    """
    sa, la = np.asarray(scores_a, dtype=np.float64), np.asarray(labels_a)
    sb, lb = np.asarray(scores_b, dtype=np.float64), np.asarray(labels_b)
    if paired and (len(sa) != len(sb) or not np.array_equal(la, lb)):
        raise ValueError("paired bootstrap needs both models scored on the same clips in the same order")
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        ia = _stratified_resample(la, rng, positive_label)
        ib = ia if paired else _stratified_resample(lb, rng, positive_label)
        diffs.append(compute_eer(sa[ia], la[ia], positive_label)["eer"] -
                     compute_eer(sb[ib], lb[ib], positive_label)["eer"])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point = compute_eer(sa, la, positive_label)["eer"] - compute_eer(sb, lb, positive_label)["eer"]
    return {"difference": float(point), "ci_low": float(low), "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0), "paired": paired, "n_resamples": n_resamples}


def speaker_leakage_call(diff):
    """Phase 2 printed "speaker leakage" for any positive difference, including +0.0009. The rule is
    now: only a CI that excludes zero counts. `diff` = EER(speaker-disjoint) - EER(random split), so a
    positive, significant difference means the random split flatters the model."""
    ci = f"{diff['difference']:+.4f} EER, 95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]"
    if not diff["excludes_zero"]:
        return {"verdict": "not_detectable",
                "text": f"no leakage detectable at this sample size ({ci}; the CI includes zero)"}
    if diff["difference"] > 0:
        return {"verdict": "leakage_detected",
                "text": f"speaker leakage detected: the random split looks better ({ci}, CI excludes zero)"}
    return {"verdict": "random_split_worse",
            "text": f"the random split is significantly *worse* ({ci}) -- not leakage; investigate the splits"}


def per_attack_eer(scores, labels, attacks, positive_label=1):
    """EER of all bonafide clips against each spoofing attack separately."""
    scores, labels, attacks = np.asarray(scores), np.asarray(labels), np.asarray(attacks)
    bonafide = labels == positive_label
    out = {}
    for attack in sorted(set(attacks[~bonafide].tolist())):
        mask = bonafide | (attacks == attack)
        out[attack] = {"eer": compute_eer(scores[mask], labels[mask], positive_label)["eer"],
                       "n_spoof": int((attacks == attack).sum())}
    return out


def evaluate_scores(scores, labels, positive_label=1):
    eer = compute_eer(scores, labels, positive_label)
    return {
        **eer,
        "per_class_at_eer_threshold": per_class_accuracy(scores, labels, eer["threshold"], positive_label),
        "det_curve": det_curve_points(scores, labels, positive_label),
    }


def duration_leakage_probe(stats, labels, positive_label=1):
    """How much can clip timing alone tell you about the label?

    Scores each clip by a duration measure and computes the EER of that single feature. An EER far
    from 50% means the dataset's recording/segmentation artifacts leak the label, independent of any
    spoofing artifact -- the known ASVspoof silence-duration shortcut. Both directions are tried per
    measure, since which sign favours bonafide is itself an empirical question; the reported EER is
    the better (lower) of the two, i.e. the strongest cheat available from that measure.
    """
    labels = np.asarray(labels)
    out = {}
    for key in ("total_s", "lead_silence_s", "trail_silence_s", "speech_s", "silence_ratio"):
        values = np.array([s[key] for s in stats], dtype=np.float64)
        if np.allclose(values, values[0]):
            out[key] = {"eer": float("nan"), "note": "constant measure"}
            continue
        forward = compute_eer(values, labels, positive_label)["eer"]
        backward = compute_eer(-values, labels, positive_label)["eer"]
        better = min(forward, backward)
        y = (labels == positive_label).astype(float)
        out[key] = {
            "eer": float(better),
            "direction": "higher_is_bonafide" if forward <= backward else "lower_is_bonafide",
            "point_biserial_r": float(np.corrcoef(values, y)[0, 1]),
            "mean_bonafide": float(values[labels == positive_label].mean()),
            "mean_spoof": float(values[labels != positive_label].mean()),
        }
    return out


def leakage_verdict(probe, strong=0.35, mild=0.45):
    """Turn the probe into a stated finding. `strong`/`mild` are EER thresholds: the further below
    50% a single duration measure gets, the more label information timing alone carries."""
    usable = {k: v["eer"] for k, v in probe.items() if not np.isnan(v["eer"])}
    if not usable:
        return {"level": "unknown", "worst_measure": None, "worst_eer": None,
                "text": "No usable duration measure; leakage could not be assessed."}
    worst_measure = min(usable, key=usable.get)
    worst = usable[worst_measure]
    level = "strong" if worst < strong else "mild" if worst < mild else "none"
    text = {
        "strong": f"Timing alone separates the classes well ({worst_measure} EER {worst:.4f}), so this data "
                  f"carries the known ASVspoof silence-duration shortcut.",
        "mild": f"Timing alone is weakly predictive ({worst_measure} EER {worst:.4f}) -- some leakage, but far "
                f"from a free solution.",
        "none": f"No meaningful timing leakage: the best duration measure only reaches EER {worst:.4f} "
                f"({worst_measure}), close to the 0.5 of a coin flip.",
    }[level]
    return {"level": level, "worst_measure": worst_measure, "worst_eer": float(worst), "text": text}
