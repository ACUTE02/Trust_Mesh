"""Derive voice verdict thresholds from each model's saved test scores (no training).

"high" is the spoof probability that flags 1%, and "medium" the one that flags 10%, of the known-bonafide
test clips (spoof probability = sigmoid(-logit), because the heads' logit is evidence for *bonafide*).
These are the same test scores the model's EER was reported on, so the thresholds are calibrated
in-sample: 1% / 10% describes how they were chosen, it is NOT an out-of-sample false-flag rate.
(The graph engine uses the same rule on validation-period nodes, which is held out.)
A model whose test-score file isn't present locally gets no entry; the service then falls back to
an explicitly-labelled uncalibrated default instead of inventing a threshold.

Run: .venv\\Scripts\\python.exe scripts/calibrate_voice_thresholds.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving import config  # noqa: E402

HIGH_FPR, MEDIUM_FPR = 0.01, 0.10
SOURCES = {  # served model -> (npz file, key prefix of the seed-42 P1 run)
    "C": ("phase2b_test_scores.npz", "C_ssl_P1_s42"),
    "D": ("phase2c_test_scores.npz", "D_ssl_mean_only_P1_s42"),
}


def main():
    out, missing = {}, {}
    for model, (fname, prefix) in SOURCES.items():
        path = config.ARTIFACTS_DIR / "voice" / fname
        if not path.exists() or f"{prefix}__scores" not in np.load(path).files:
            missing[model] = f"{fname} / {prefix} not found locally"
            continue
        z = np.load(path)
        logits, labels = z[f"{prefix}__scores"].astype(np.float64), z[f"{prefix}__labels"]
        assert logits[labels == 1].mean() > logits[labels == 0].mean(), "expected bonafide (label 1) logits to be higher"
        spoof_prob = 1.0 / (1.0 + np.exp(logits))
        bona = spoof_prob[labels == 1]
        out[model] = {
            "high": float(np.quantile(bona, 1.0 - HIGH_FPR)),
            "medium": float(np.quantile(bona, 1.0 - MEDIUM_FPR)),
            "source": f"{fname} [{prefix}]: high flags {HIGH_FPR:.0%}, medium flags {MEDIUM_FPR:.0%} of "
                      f"{len(bona)} bonafide in-domain test clips",
        }
        print(model, out[model])
    config.VOICE_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    config.VOICE_THRESHOLDS.write_text(json.dumps(out, indent=2))
    for model, why in missing.items():
        print(f"model {model}: no thresholds written -- {why}")


if __name__ == "__main__":
    main()
