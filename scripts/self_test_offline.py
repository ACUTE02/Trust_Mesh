"""Self-test, offline part (ST1, ST3, ST4 of PHASE_2_SELF_TEST_RUNBOOK.md). No training, no network.

Recomputes the headline numbers from the saved artifacts, estimates the voice thresholds out of sample by
repeated half-splits of the saved test scores, and bootstraps an interval for the served seed's EER.
Prints aggregates only and writes them to data/self_test/offline.json (local, not committed).

Run: .venv\\Scripts\\python.exe scripts/self_test_offline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "self_test"
TOL = 0.0005
HIGH_FPR, MEDIUM_FPR = 0.01, 0.10  # the rule in scripts/calibrate_voice_thresholds.py


def eer(labels, scores):
    """Equal error rate with bonafide (label 1) as the target, same convention as the project."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"tolerance": TOL}

    # --- ST1 graph --------------------------------------------------------------------------------
    from serving.graph_service import GraphService
    g = GraphService()
    g.load(with_gat=False)
    measured = g.verification["test_pr_auc"]
    result["graph"] = {"claimed_seed42": 0.7474, "measured": measured, "diff": abs(measured - 0.7474),
                       "n_test": g.verification["n_test"], "pass": abs(measured - 0.7474) <= TOL}

    # --- ST1 voice --------------------------------------------------------------------------------
    z = np.load(ROOT / "artifacts" / "voice" / "phase2c_test_scores.npz")
    per_seed = {}
    for seed in (42, 43, 44):
        key = f"D_ssl_mean_only_P1_s{seed}"
        per_seed[seed] = eer(z[f"{key}__labels"], z[f"{key}__scores"])
    values = np.array(list(per_seed.values()))
    mean, std = float(values.mean()), float(values.std())
    result["voice"] = {
        "per_seed": per_seed, "mean": mean, "std_population": std, "std_sample": float(values.std(ddof=1)),
        "claimed_mean": 0.0166, "claimed_std": 0.0019, "claimed_seed42": 0.0194,
        "pass_mean": abs(mean - 0.0166) <= TOL, "pass_std": abs(std - 0.0019) <= TOL,
        "pass_seed42": abs(per_seed[42] - 0.0194) <= TOL,
    }

    # --- ST3 thresholds out of sample ------------------------------------------------------------------
    labels = z["D_ssl_mean_only_P1_s42__labels"]
    spoof_prob = 1.0 / (1.0 + np.exp(z["D_ssl_mean_only_P1_s42__scores"].astype(np.float64)))
    bona, spoof = spoof_prob[labels == 1], spoof_prob[labels == 0]

    def thresholds(b):
        return float(np.quantile(b, 1 - HIGH_FPR)), float(np.quantile(b, 1 - MEDIUM_FPR))

    def rates(b, s, high, medium):
        return {"genuine_flagged_high": float((b >= high).mean()), "genuine_flagged_medium_or_high": float((b >= medium).mean()),
                "fake_flagged_high": float((s >= high).mean()), "fake_flagged_medium_or_high": float((s >= medium).mean())}

    high, medium = thresholds(bona)
    in_sample = rates(bona, spoof, high, medium)
    rng = np.random.default_rng(20261008)
    runs = []
    for _ in range(200):
        bi, si = rng.permutation(len(bona)), rng.permutation(len(spoof))
        b_fit, b_eval = bona[bi[:len(bona) // 2]], bona[bi[len(bona) // 2:]]
        s_eval = spoof[si[len(spoof) // 2:]]
        h, m = thresholds(b_fit)
        runs.append({**rates(b_eval, s_eval, h, m), "high": h, "medium": m})
    result["thresholds"] = {
        "served": {"high": high, "medium": medium}, "in_sample": in_sample, "n_bonafide": int(len(bona)), "n_spoof": int(len(spoof)),
        "splits": 200,
        "out_of_sample": {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs])),
                              "min": float(np.min([r[k] for r in runs])), "max": float(np.max([r[k] for r in runs]))}
                          for k in runs[0]},
    }

    # --- ST4 bootstrap interval for the served seed --------------------------------------------------
    scores = z["D_ssl_mean_only_P1_s42__scores"]
    b_idx, s_idx = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    boot = []
    for _ in range(1000):
        idx = np.concatenate([rng.choice(b_idx, len(b_idx)), rng.choice(s_idx, len(s_idx))])
        boot.append(eer(labels[idx], scores[idx]))
    result["seed42_bootstrap"] = {"eer": per_seed[42], "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                                  "resamples": 1000, "stratified": True}

    (OUT / "offline.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
