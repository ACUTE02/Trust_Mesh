# %% [markdown]
# # Phase 2 -- Voice Engine (EchoAuth core): LFCC + LCNN spoof detection on ASVspoof 2019 LA
#
# Implements `docs/PHASE_2_VOICE_ENGINE.md`. Generated from `notebooks/src/03_voice_baseline.py` by
# `scripts/build_notebooks.py` -- edit the source, not this .ipynb.
#
# 1. **Data + EDA** -- stream a subset of ASVspoof 2019 LA, report class balance, durations, sample rate.
# 2. **Preprocessing** -- resample -> silence trim -> fixed length -> LFCC + deltas, as a reusable
#    pipeline used identically for train, test and the out-of-domain clips.
# 3. **Silence-leakage check** -- can clip timing alone predict the label? Measured before and after
#    trimming, and confirmed by training the same model on untrimmed features.
# 4. **LCNN** -- one architecture, trained properly, class-weighted for the bonafide/spoof imbalance.
# 5. **Evaluation** -- EER (interpolated at the FAR/FRR crossing), DET curve, per-class accuracy.
# 6. **Cross-dataset spot-check** -- a few In-the-Wild clips the model never saw (stretch goal).
#
# **Two honesty notes that shape this notebook, both explained in `docs/design_decisions.md`:**
# - The loadable HuggingFace copies of ASVspoof 2019 LA contain **only the evaluation partition**, so
#   the official protocol (train on attacks A01-A06, test on unseen A07-A19) is not reachable. We
#   split the eval partition ourselves; attack types therefore appear on both sides, which is
#   optimistic. Unseen-attack generalization is **not** measured here.
# - A random clip-level split would put the same speaker in train and test. The headline split is
#   therefore **speaker-disjoint**, and the random-split number is reported alongside it as a
#   measurement of that leak -- the audio counterpart of Phase 1's temporal-vs-random gap.
#
# **Kaggle:** Accelerator = GPU, Internet = on, Run All, then **Save Version -> Save & Run All (Commit)**
# or the outputs will not persist.

# %% [markdown]
# ## Setup

# %%
# Kaggle: Accelerator = GPU, Internet = on.
import os
import subprocess
import sys

# True only when tests/smoke_notebook_03.py runs this notebook's code on synthetic audio.
SMOKE_TEST = os.environ.get("TRUSTMESH_SMOKE") == "1"


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


if not SMOKE_TEST:
    # `datasets` streams the audio; `soundfile` decodes the FLAC bytes. Everything else (LFCC, LCNN)
    # is plain torch, so there is no torchaudio/librosa version coupling to worry about.
    for module_name, package in [("datasets", "datasets"), ("soundfile", "soundfile")]:
        try:
            __import__(module_name)
        except ImportError:
            pip_install(package)

# %%
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path(os.environ.get("TRUSTMESH_OUTPUT_DIR", "./artifacts/voice"))
DOCS_DIR = Path(os.environ.get("TRUSTMESH_DOCS_DIR", "./docs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE_TEST else "cpu")

# How many clips to stream out of the 71,237-clip LA evaluation partition. This is the number that
# must be quoted whenever a result from this notebook is quoted -- "started small" must not quietly
# become the final claim. ~6000 clips is roughly 370 MB of the 4.39 GB parquet.
N_CLIPS = 120 if SMOKE_TEST else 6000
CLIP_SECONDS = 4.0
TEST_FRACTION = 0.3
VAL_FRACTION = 0.2          # of the train portion
EPOCHS = 3 if SMOKE_TEST else 30
BATCH_SIZE = 8 if SMOKE_TEST else 32
LEARNING_RATE = 3e-4
PATIENCE = 6
N_OOD_CLIPS = 4 if SMOKE_TEST else 10
N_BOOTSTRAP = 50 if SMOKE_TEST else 1000  # stratified resamples for every EER comparison

RESULTS_PATH = OUTPUT_DIR / "results.json"
RESULTS = {"smoke_test": SMOKE_TEST}


def save_progress(key, value):
    RESULTS[key] = value
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")
    print(f"[progress] saved '{key}' -> {RESULTS_PATH}")


print("device:", DEVICE, "| SMOKE_TEST:", SMOKE_TEST, "| torch", torch.__version__)

# %% [markdown]
# ## Shared voice-engine code (inlined from `engines/voice/`)

# %% inline engines/voice/data.py

# %% inline engines/voice/preprocess.py

# %% inline engines/voice/evaluate.py

# %% inline engines/voice/model.py

# %% [markdown]
# ## 1. Data loading + EDA

# %%
t0 = time.time()
if SMOKE_TEST:
    clips = make_synthetic_asvspoof(N_CLIPS)
    dataset_name = "synthetic (smoke test)"
else:
    print(f"streaming {N_CLIPS} clips from {ASVSPOOF_DATASET} ...")
    clips = load_asvspoof_subset(N_CLIPS, seed=SEED)
    dataset_name = f"{ASVSPOOF_DATASET} (split '{ASVSPOOF_SPLIT}' = LA evaluation partition)"
print(f"loaded {len(clips)} clips in {time.time() - t0:.0f}s")

durations = np.array([len(c["waveform"]) / c["sample_rate"] for c in clips])
sample_rates = sorted({c["sample_rate"] for c in clips})
speakers = sorted({c["speaker_id"] for c in clips})
counts = class_counts(clips)
weight_for_bonafide = pos_weight(clips)

eda = {
    "dataset": dataset_name,
    "clips_requested": N_CLIPS,
    "clips_loaded": len(clips),
    "bonafide": counts["bonafide"],
    "spoof": counts["spoof"],
    "bonafide_share": counts["bonafide"] / max(counts["total"], 1),
    "pos_weight_for_bonafide": weight_for_bonafide,
    "speakers": len(speakers),
    "sample_rates_hz": sample_rates,
    "duration_s": {"mean": float(durations.mean()), "min": float(durations.min()),
                   "max": float(durations.max()), "median": float(np.median(durations)),
                   "total_hours": float(durations.sum() / 3600)},
}
print(json.dumps(eda, indent=2))
save_progress("eda", eda)

# %%
eda_md = f"""# Voice Data Exploration -- ASVspoof 2019 LA

Source: `{eda['dataset']}`, streamed with the HuggingFace `datasets` library.

**Subset actually used for every number in this phase: {eda['clips_loaded']} clips** (of the 71,237 in
the LA evaluation partition), sampled with a shuffle buffer at seed {SEED}. This is the figure to quote;
it is a subset, not the full partition.

## Class balance
- Bonafide: {eda['bonafide']}
- Spoof: {eda['spoof']}
- Bonafide share: {eda['bonafide_share']:.2%}

Spoof outnumbers bonafide by about {eda['pos_weight_for_bonafide']:.1f} to 1, so the training loss is
class-weighted by that ratio and the headline metric is EER rather than accuracy -- the same
reasoning that made PR-AUC the metric for the graph engine.

## Clip durations (as provided, before any trimming)
- Mean: {eda['duration_s']['mean']:.2f} s
- Median: {eda['duration_s']['median']:.2f} s
- Min: {eda['duration_s']['min']:.2f} s
- Max: {eda['duration_s']['max']:.2f} s
- Total audio: {eda['duration_s']['total_hours']:.2f} hours

## Sample rate
- As provided: {', '.join(str(s) for s in eda['sample_rates_hz'])} Hz (the pipeline resamples to 16 kHz if needed).

## Speakers
- Distinct speaker IDs: {eda['speakers']}

Speaker identity matters for the split: a random clip-level split would place the same speaker on
both sides, letting the model key on voice/channel identity instead of spoofing artifacts. The
headline split in this phase is speaker-disjoint; see `docs/design_decisions.md`.

## Known caveat
These clips all come from the LA **evaluation** partition, so the same spoofing systems (A07-A19)
appear in both the train and test halves of our split. Performance against *unseen* attacks is not
measured here.
"""
(DOCS_DIR / "voice_data_exploration.md").write_text(eda_md, encoding="utf-8")
print(f"wrote {DOCS_DIR / 'voice_data_exploration.md'}")

# %% [markdown]
# ## 2. Preprocessing pipeline
#
# One `VoicePreprocessor` builds every feature tensor used below (train, test, and the out-of-domain
# clips in task 6), so those paths cannot silently diverge. Two variants are built: **trimmed**
# (the real pipeline) and **untrimmed** (identical apart from the silence trim), which exists purely
# to measure how much the model relies on silence -- task 3.

# %%
t0 = time.time()
pre_trimmed = VoicePreprocessor(clip_seconds=CLIP_SECONDS, trim=True)
pre_untrimmed = VoicePreprocessor(clip_seconds=CLIP_SECONDS, trim=False)

x_trimmed, y_all, silence_stats = pre_trimmed.featurize_clips(clips)
x_untrimmed, _, _ = pre_untrimmed.featurize_clips(clips)
print(f"features {tuple(x_trimmed.shape)} (clips, channel, LFCC+deltas, frames) in {time.time() - t0:.0f}s")

feature_config = {
    "front_end": "LFCC + delta + delta-delta",
    "n_coeffs": pre_trimmed.n_coeffs,
    "n_features": pre_trimmed.n_features,
    "n_frames": pre_trimmed.n_frames,
    "clip_seconds": CLIP_SECONDS,
    "window_ms": 20.0,
    "hop_ms": 10.0,
    "n_linear_filters": 70,
    "trim_top_db": pre_trimmed.top_db,
    "why_lfcc": "LFCC is the official ASVspoof 2019/2021 baseline front-end; linear filter spacing "
                "keeps high-frequency resolution where vocoder/TTS artifacts live, which mel spacing "
                "compresses away.",
}
save_progress("feature_config", feature_config)

# %% [markdown]
# ## 3. Silence-leakage check
#
# ASVspoof 2019 LA is known to leak the label through silence duration: bonafide and spoof clips were
# produced differently, so leading/trailing silence alone can separate them without any spoofing
# artifact. The probe scores each clip by a timing measure alone and computes that measure's EER --
# an EER far from 0.5 means timing alone is a shortcut. It is run **before** trimming and **after**,
# and then confirmed by training the same model on untrimmed features (task 4).

# %%
labels_np = y_all.numpy()
probe_before = duration_leakage_probe(silence_stats, labels_np)
verdict_before = leakage_verdict(probe_before)

# After trimming, the model sees a fixed-length window of trimmed speech: leading/trailing silence is
# gone by construction, so those measures become constant. What can still carry signal is how much
# speech remained, so that is what the "after" probe scores.
stats_after = [{"total_s": s["speech_s"], "lead_silence_s": 0.0, "trail_silence_s": 0.0,
                "speech_s": s["speech_s"], "silence_ratio": 0.0} for s in silence_stats]
probe_after = duration_leakage_probe(stats_after, labels_np)
verdict_after = leakage_verdict(probe_after)

print("BEFORE trimming (EER of each timing measure used alone; 0.5 = no information):")
for key, value in probe_before.items():
    if not math.isnan(value["eer"]):
        print(f"  {key:18s} EER {value['eer']:.4f}  r={value['point_biserial_r']:+.3f}  "
              f"bonafide mean {value['mean_bonafide']:.3f}s vs spoof {value['mean_spoof']:.3f}s")
print(f"  -> {verdict_before['text']}")
print("\nAFTER trimming (leading/trailing silence removed; only residual speech duration remains):")
for key, value in probe_after.items():
    if not math.isnan(value["eer"]):
        print(f"  {key:18s} EER {value['eer']:.4f}  r={value['point_biserial_r']:+.3f}")
print(f"  -> {verdict_after['text']}")

leakage = {
    "before_trim": probe_before, "after_trim": probe_after,
    "verdict_before": verdict_before, "verdict_after": verdict_after,
    "worst_eer_before": verdict_before["worst_eer"], "worst_eer_after": verdict_after["worst_eer"],
    "note": "Every model input is a fixed-length window, so absolute clip duration cannot reach the "
            "model regardless; these probes measure what the raw data leaks, and the trimmed-vs-"
            "untrimmed models below measure whether the model actually exploited it.",
}
save_progress("silence_leakage", leakage)

# %% [markdown]
# ## 4. LCNN training
#
# Three runs of the same LCNN, differing only in one variable each, so each comparison is clean:
# - `lcnn_trimmed_speaker_disjoint` -- **the headline model**.
# - `lcnn_untrimmed_speaker_disjoint` -- same split, silence left in: does the model exploit silence?
# - `lcnn_trimmed_random_split` -- same features, speakers allowed on both sides: how much does
#   speaker leakage inflate the number?

# %%
def sub_split(indices, fraction, seed, disjoint):
    """Split a subset of clip indices, returning global indices."""
    subset = [clips[i] for i in indices]
    splitter = speaker_disjoint_split if disjoint else random_split
    left, right = splitter(subset, fraction, seed)
    return [indices[i] for i in left], [indices[i] for i in right]


SPLITS = {}
for split_name, disjoint in [("speaker_disjoint", True), ("random", False)]:
    trainval_idx, test_idx = (speaker_disjoint_split if disjoint else random_split)(clips, TEST_FRACTION, SEED)
    train_idx, val_idx = sub_split(trainval_idx, VAL_FRACTION, SEED + 1, disjoint)
    SPLITS[split_name] = {"train": train_idx, "val": val_idx, "test": test_idx,
                          "report": split_report(clips, train_idx, test_idx),
                          "val_counts": class_counts([clips[i] for i in val_idx])}
    report = SPLITS[split_name]["report"]
    print(f"{split_name:16s} train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} | "
          f"test bonafide {report['test']['bonafide']} spoof {report['test']['spoof']} | "
          f"val bonafide {SPLITS[split_name]['val_counts']['bonafide']} | "
          f"speakers shared between train and test: {report['shared_speakers']}")
    if report["shared_speakers"] and split_name == "speaker_disjoint":
        print("  WARNING: speaker-disjoint split fell back to a random split (too few speakers to "
              "keep both classes on both sides) -- the headline number is NOT speaker-disjoint")
save_progress("splits", {k: {**v["report"], "n_train": len(v["train"]), "n_val": len(v["val"]),
                             "n_test": len(v["test"]), "val_counts": v["val_counts"]}
                         for k, v in SPLITS.items()})

# %%
RUNS = [
    ("lcnn_trimmed_speaker_disjoint", "trimmed", "speaker_disjoint", "headline model"),
    ("lcnn_untrimmed_speaker_disjoint", "untrimmed", "speaker_disjoint", "silence-leakage comparison"),
    ("lcnn_trimmed_random_split", "trimmed", "random", "speaker-leakage comparison"),
]
FEATURES = {"trimmed": x_trimmed, "untrimmed": x_untrimmed}
MODEL_RESULTS, SCORES, MODELS = {}, {}, {}

for name, feature_set, split_name, role in RUNS:
    print(f"\n=== {name} ({role}) ===")
    x, split = FEATURES[feature_set], SPLITS[split_name]
    train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]
    train_clips = [clips[i] for i in train_idx]

    torch.manual_seed(SEED)
    model = LCNN(n_features=pre_trimmed.n_features, n_frames=pre_trimmed.n_frames)
    t0 = time.time()
    fit = train_lcnn(model, x[train_idx], y_all[train_idx], pos_weight(train_clips), DEVICE,
                     epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                     x_val=x[val_idx], y_val=y_all[val_idx], patience=PATIENCE, seed=SEED)
    scores = predict_scores(model, x[test_idx], DEVICE)
    report = evaluate_scores(scores, y_all[test_idx].numpy())

    MODEL_RESULTS[name] = {
        "features": feature_set, "split": split_name, "role": role,
        "eer": report["eer"], "eer_threshold": report["threshold"],
        "per_class_at_eer_threshold": report["per_class_at_eer_threshold"],
        "best_val_eer": fit["best_val_eer"], "best_epoch": fit["best_epoch"],
        "validation_usable": fit["validation_usable"], "epochs_run": fit["epochs_run"],
        "train_seconds": round(time.time() - t0, 1),
        "train_loss_first_last": [fit["history"][0]["train_loss"], fit["history"][-1]["train_loss"]],
    }
    SCORES[name] = {"scores": scores, "labels": y_all[test_idx].numpy(), "det": report["det_curve"]}
    MODELS[name] = model
    accuracy = report["per_class_at_eer_threshold"]
    print(f"  test EER {report['eer']:.4f} | bonafide acc {accuracy['bonafide_accuracy']:.4f} | "
          f"spoof acc {accuracy['spoof_accuracy']:.4f} | {fit['epochs_run']} epochs, "
          f"{MODEL_RESULTS[name]['train_seconds']:.0f}s")
    save_progress("models", MODEL_RESULTS)

# %% [markdown]
# ## 5. Evaluation: EER, DET curve, per-class accuracy

# %%
headline = "lcnn_trimmed_speaker_disjoint"
headline_eer = MODEL_RESULTS[headline]["eer"]
silence_effect = MODEL_RESULTS["lcnn_untrimmed_speaker_disjoint"]["eer"] - headline_eer
speaker_effect = headline_eer - MODEL_RESULTS["lcnn_trimmed_random_split"]["eer"]
# The two splits have different test sets, so this is an unpaired bootstrap: only a CI that excludes
# zero is reported as leakage (Phase 2 labelled a +0.0009 difference "speaker leakage" on sign alone).
speaker_leakage = bootstrap_eer_difference(
    SCORES[headline]["scores"], SCORES[headline]["labels"],
    SCORES["lcnn_trimmed_random_split"]["scores"], SCORES["lcnn_trimmed_random_split"]["labels"],
    n_resamples=N_BOOTSTRAP, seed=SEED)
speaker_call = speaker_leakage_call(speaker_leakage)

print(f"{'model':34s} {'EER':>8s} {'bonafide acc':>13s} {'spoof acc':>10s}")
for name, result in MODEL_RESULTS.items():
    accuracy = result["per_class_at_eer_threshold"]
    print(f"{name:34s} {result['eer']:8.4f} {accuracy['bonafide_accuracy']:13.4f} "
          f"{accuracy['spoof_accuracy']:10.4f}")
print(f"\nheadline EER (trimmed features, speaker-disjoint split): {headline_eer:.4f}")
print(f"untrimmed minus trimmed EER: {silence_effect:+.4f} "
      f"({'silence helped the model' if silence_effect < 0 else 'silence did not help the model'})")
print(f"speaker-disjoint minus random-split EER: {speaker_call['text']}")

import matplotlib.pyplot as plt

plt.figure(figsize=(5.5, 5))
for name in MODEL_RESULTS:
    det = SCORES[name]["det"]
    plt.plot(np.array(det["far"]) * 100, np.array(det["frr"]) * 100,
             label=f"{name} (EER {MODEL_RESULTS[name]['eer']:.3f})")
plt.plot([0, 100], [0, 100], "k--", linewidth=0.8, label="EER line")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("False accept rate (%)  -- spoof accepted as bonafide")
plt.ylabel("False reject rate (%)  -- bonafide rejected")
plt.title("DET curves, ASVspoof 2019 LA subset")
plt.legend(fontsize=7)
plt.grid(True, which="both", linewidth=0.3)
plt.tight_layout()
det_path = OUTPUT_DIR / "det_curves.png"
plt.savefig(det_path, dpi=120)
plt.show()
print(f"wrote {det_path}")

save_progress("evaluation", {
    "headline_model": headline,
    "headline_eer": headline_eer,
    "untrimmed_minus_trimmed_eer": silence_effect,
    "speaker_disjoint_minus_random_eer": speaker_effect,
    "speaker_leakage_bootstrap": speaker_leakage,
    "speaker_leakage_verdict": speaker_call,
    "det_curves": {name: SCORES[name]["det"] for name in MODEL_RESULTS},
})

# %% [markdown]
# ## 6. Cross-dataset spot-check (stretch goal)
#
# A handful of In-the-Wild clips, which come from real recordings rather than ASVspoof's studio
# conditions. With this few clips the numbers are indicative only, never a headline. Non-fatal: if the
# dataset can't be reached or its labels can't be interpreted, that is recorded and the notebook
# continues.

# %%
ood = {"status": "not_run", "n_clips": 0}
try:
    if SMOKE_TEST:
        ood_clips = make_synthetic_asvspoof(N_OOD_CLIPS, seed=7)
        ood_source = "synthetic (smoke test)"
    else:
        print(f"streaming {N_OOD_CLIPS} clips from {IN_THE_WILD_DATASET} ...")
        ood_clips = load_in_the_wild_clips(N_OOD_CLIPS, seed=SEED)
        ood_source = IN_THE_WILD_DATASET

    x_ood, _, _ = pre_trimmed.featurize_clips(ood_clips, progress_every=None)
    ood_scores = predict_scores(MODELS[headline], x_ood, DEVICE)
    ood_labels = [c.get("label") for c in ood_clips]
    labelled = [(s, l) for s, l in zip(ood_scores, ood_labels) if l is not None]

    ood = {
        "status": "ok",
        "source": ood_source,
        "n_clips": len(ood_clips),
        "raw_label_values": sorted({str(c.get("raw_label")) for c in ood_clips}) if not SMOKE_TEST else ["synthetic"],
        "label_column": ood_clips[0].get("label_column") if not SMOKE_TEST else "label",
        "scores": [float(s) for s in ood_scores],
        "labels": [None if l is None else int(l) for l in ood_labels],
        "mean_score_bonafide": float(np.mean([s for s, l in labelled if l == BONAFIDE])) if any(l == BONAFIDE for _, l in labelled) else None,
        "mean_score_spoof": float(np.mean([s for s, l in labelled if l == SPOOF])) if any(l == SPOOF for _, l in labelled) else None,
    }
    if len({l for _, l in labelled}) == 2:
        ood["eer"] = compute_eer(np.array([s for s, _ in labelled]), np.array([l for _, l in labelled]))["eer"]
        ood["eer_caveat"] = (f"EER over only {len(labelled)} clips -- indicative, not a reportable "
                             f"operating point (in-domain EER used {len(SPLITS['speaker_disjoint']['test'])} clips).")
    else:
        ood["eer"] = None
        ood["eer_caveat"] = "Only one class present among the sampled clips; EER undefined."
    print(json.dumps({k: v for k, v in ood.items() if k != "scores"}, indent=2))
except Exception as exc:  # noqa: BLE001 -- a stretch goal must not sink the run
    ood = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
           "note": "Cross-dataset spot-check skipped; in-domain results above are unaffected."}
    print(f"cross-dataset spot-check failed, continuing: {ood['error']}")
save_progress("cross_dataset_spot_check", ood)

# %% [markdown]
# ## 7. Closing analysis and save

# %%
findings = [
    f"Headline: an LCNN on LFCC features reaches {headline_eer:.4f} EER on a held-out, speaker-disjoint "
    f"test split of {len(SPLITS['speaker_disjoint']['test'])} clips drawn from a {len(clips)}-clip subset "
    f"of the ASVspoof 2019 LA evaluation partition.",
    f"Silence leakage: before trimming, the most predictive timing measure alone reached EER "
    f"{leakage['worst_eer_before']:.4f} ({verdict_before['worst_measure']}); after trimming, "
    f"{leakage['worst_eer_after']:.4f}. {verdict_before['text']}",
]
if silence_effect < -0.01:
    findings.append(f"Training on untrimmed features scored {abs(silence_effect):.4f} EER *better* "
                    f"({MODEL_RESULTS['lcnn_untrimmed_speaker_disjoint']['eer']:.4f}), i.e. the model does "
                    f"exploit silence when it is left in -- trimming removes a real shortcut rather than a "
                    f"hypothetical one.")
elif silence_effect > 0.01:
    findings.append(f"Training on untrimmed features scored {silence_effect:.4f} EER *worse* "
                    f"({MODEL_RESULTS['lcnn_untrimmed_speaker_disjoint']['eer']:.4f}), so the leftover silence "
                    f"was noise for this model, not a usable shortcut.")
else:
    findings.append(f"Trimmed and untrimmed models land within {abs(silence_effect):.4f} EER of each other, "
                    f"so the silence cue was not decisive for this model either way.")
findings.append(f"Speaker leakage (random-split EER {MODEL_RESULTS['lcnn_trimmed_random_split']['eer']:.4f} vs "
                f"speaker-disjoint {headline_eer:.4f}): {speaker_call['text']}.")
if ood.get("status") == "ok" and ood.get("eer") is not None:
    findings.append(f"Out of domain: on {ood['n_clips']} In-the-Wild clips the same model scores EER "
                    f"{ood['eer']:.4f}. {ood['eer_caveat']}")
elif ood.get("status") == "ok":
    findings.append(f"Out of domain: {ood['n_clips']} In-the-Wild clips scored, but {ood['eer_caveat'][0].lower()}"
                    f"{ood['eer_caveat'][1:]}")
else:
    findings.append("Out of domain: the spot-check did not run, so nothing is claimed about generalization.")
findings.append("Scope limit: every clip comes from the LA evaluation partition, so the same spoofing "
                "systems appear in train and test. Performance against unseen attack types is not measured.")

closing_paragraph = " ".join(findings)
print(closing_paragraph)
save_progress("closing_paragraph", closing_paragraph)

for name, model in MODELS.items():
    torch.save(model.state_dict(), OUTPUT_DIR / f"{name}.pt")
    print(f"wrote {OUTPUT_DIR / f'{name}.pt'}")
np.savez_compressed(OUTPUT_DIR / "test_scores.npz",
                    **{f"{name}__{field}": SCORES[name][field] for name in SCORES for field in ("scores", "labels")})
print(f"wrote {OUTPUT_DIR / 'test_scores.npz'}")
print(f"\nFull results: {RESULTS_PATH}")
print(json.dumps({k: v for k, v in RESULTS.items() if k != "evaluation"}, indent=2, default=float))
