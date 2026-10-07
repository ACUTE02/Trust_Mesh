# %% [markdown]
# # Phase 2B -- Voice Engine depth pass: honest accuracy, not shortcut accuracy
#
# Implements `docs/PHASE_2B_VOICE_DEPTH.md`. Generated from `notebooks/src/04_voice_depth.py` by
# `scripts/build_notebooks.py` -- edit the source, not this .ipynb.
#
# - **Task 1 -- more data:** every bonafide clip in the LA evaluation partition (~7.3k, vs 619 in Phase 2)
#   plus a capped sample of spoof clips, decoded once into an on-disk feature cache.
# - **Task 2 -- attack-disjoint split:** attack IDs come from the official protocol file shipped in the
#   same HuggingFace repo, and are used only after it reproduces every clip's label and speaker exactly.
#   Shared-attack and attack-disjoint models are scored on *the same test clips*.
# - **Task 3 -- one variable per run:** A = LCNN on the larger data; B = A + augmentation; C = frozen
#   XLS-R front-end + layer-weighted head. Selection on validation only; test touched once per model.
# - **Task 4 -- silence:** trimmed stays the headline; untrimmed is labelled a shortcut. Four ablations
#   separate silence *duration* from silence *content*.
# - **Task 5 -- rigour:** 3 seeds for the headline and best model, stratified bootstrap 95% CIs (1000
#   resamples) on every test EER, class counts next to every number.
# - **Task 6 -- In-the-Wild:** rerun with the fixed loader, >= 100 clips per class, failures recorded.
#
# Nothing here is compared with published ASVspoof numbers: this is not the official protocol.
#
# **Kaggle:** GPU (T4), Internet on. Recommended: skip the interactive Run All and go straight to
# **Save Version -> Save & Run All (Commit)** -- a commit re-runs everything from scratch anyway, so an
# interactive run first doubles the GPU time. See `docs/kaggle_setup.md`.

# %% [markdown]
# ## Setup

# %%
# Kaggle: Accelerator = GPU, Internet = on.
import os
import subprocess
import sys

# True only when tests/smoke_notebook_04.py runs this notebook's code on synthetic data.
SMOKE_TEST = os.environ.get("TRUSTMESH_SMOKE") == "1"


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


if not SMOKE_TEST:
    for module_name, package in [("datasets", "datasets"), ("soundfile", "soundfile"),
                                 ("transformers", "transformers"), ("fsspec", "fsspec")]:
        try:
            __import__(module_name)
        except ImportError:
            pip_install(package)

# %%
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

RUN_START = time.time()
SEED = 42
SEEDS = [42, 43, 44]
torch.manual_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = Path(os.environ.get("TRUSTMESH_OUTPUT_DIR", "./artifacts/voice"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# The spec asks for the cache in /kaggle/working so a crash doesn't force re-decoding. It is deleted at
# the end of a *successful* run (DELETE_CACHE_AT_END) so the committed output stays small.
CACHE_DIR = Path(os.environ.get("TRUSTMESH_CACHE_DIR", str(OUTPUT_DIR / "cache")))
DELETE_CACHE_AT_END = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE_TEST else "cpu")

SPOOF_PER_BONAFIDE = 4           # all bonafide + 4 spoof per bonafide (spec: 1:4 to 1:6). Lower this, never cut bonafide.
CLIP_SECONDS = 4.0
SILENCE_WINDOW_S = 0.2           # silence-only ablation: exactly this much genuine silence per clip (21 LFCC frames)
FIXED_SILENCE_S = 0.5            # zeros_fixed / noise_fixed ablations
NOISE_DBFS = -60.0
TEST_FRACTION, VAL_FRACTION = 0.3, 0.2
N_HELD_OUT_ATTACKS = 4           # of the 13 LA eval attacks (A07-A19)

EPOCHS = 3 if SMOKE_TEST else 25
PATIENCE = 5
BATCH_SIZE = 8 if SMOKE_TEST else 64
LEARNING_RATE = 3e-4
SSL_MODEL = "facebook/wav2vec2-xls-r-300m"
SSL_FALLBACK_MODEL = "facebook/wav2vec2-base"
SSL_BATCH = 4 if SMOKE_TEST else 16
SSL_EPOCHS = 3 if SMOKE_TEST else 60
SSL_LR, SSL_PATIENCE = 1e-3, 8
# Phase 2C Model F re-runs the frozen encoder every training batch (on-the-fly augmentation), so one
# epoch costs far more wall-clock than Model C's cached-feature training -- Model C historically
# converged within 13-44 epochs (Phase 2B), so this caps Model F's worst case without changing its
# early-stopping patience. Same protocol otherwise (lr, patience, batch size).
SSL_EPOCHS_ONTHEFLY = 3 if SMOKE_TEST else 30
# Model F is Phase 2C's most expensive run by a wide margin (it re-runs the frozen encoder on every
# training batch, even after issues #26/#27's fixes) and is a smaller, more speculative check than D or
# E. Off by default to keep a real Kaggle attempt to a couple of hours (user's call, 2026-09-30, after
# three slow/stuck attempts); set TRUSTMESH_RUN_MODEL_F=1 to include it. The smoke test always sets this
# so Model F's code path stays covered regardless of this default.
RUN_MODEL_F = SMOKE_TEST or os.environ.get("TRUSTMESH_RUN_MODEL_F") == "1"
# Model E's full-dataset sequence cache (~15 GB) overran this Kaggle environment's real disk quota even
# after Task 4's cleanup freed 7.17 GB -- a second, distinct disk failure (issue #28) mid-way through
# building it. Off by default for the same reason as Model F, until the disk budget is worked out
# (e.g. a smaller clip subset just for this model); set TRUSTMESH_RUN_MODEL_E=1 to include it.
RUN_MODEL_E = SMOKE_TEST or os.environ.get("TRUSTMESH_RUN_MODEL_E") == "1"

N_ITW_PER_CLASS = 3 if SMOKE_TEST else 100
N_BOOTSTRAP = 50 if SMOKE_TEST else 1000
PHASE2_TEST_COUNTS = {"bonafide": 182, "spoof": 1643}   # Phase 2's headline test set, for the CI-shrinkage check
TIME_BUDGET_HOURS = 10.0                                # Kaggle sessions stop at 12 h

CLIP_SAMPLES = int(16000 * CLIP_SECONDS)
SILENCE_SAMPLES = int(16000 * SILENCE_WINDOW_S)
RESULTS_PATH = OUTPUT_DIR / "results_phase2b.json"
RESULTS = {"smoke_test": SMOKE_TEST, "not_run": {}}
# Phase 2C (docs/PHASE_2C_VOICE_SSL_DEPTH.md) writes its own file, never merged into results_phase2b.json,
# so the two phases stay separable per the spec.
RESULTS2C_PATH = OUTPUT_DIR / "results_phase2c.json"
RESULTS2C = {"smoke_test": SMOKE_TEST, "not_run": {}}


def save_progress(key, value):
    RESULTS[key] = value
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")


def save_progress_2c(key, value):
    RESULTS2C[key] = value
    RESULTS2C_PATH.write_text(json.dumps(RESULTS2C, indent=2, default=float), encoding="utf-8")


def elapsed_hours():
    return (time.time() - RUN_START) / 3600


def within_budget(step):
    """Optional steps are skipped (and recorded as not run) rather than risking the whole session."""
    if elapsed_hours() < TIME_BUDGET_HOURS:
        return True
    RESULTS["not_run"][step] = f"time budget: {elapsed_hours():.1f} h elapsed of {TIME_BUDGET_HOURS} h"
    save_progress("not_run", RESULTS["not_run"])
    print(f"SKIPPED {step}: {RESULTS['not_run'][step]}")
    return False


def within_budget_2c(step):
    """Same budget check as `within_budget`, for Phase 2C steps -- records into results_phase2c.json's
    own `not_run`, not Phase 2B's, since Phase 2C runs in the same Kaggle session against the same
    TIME_BUDGET_HOURS clock but must not write into results_phase2b.json."""
    if elapsed_hours() < TIME_BUDGET_HOURS:
        return True
    RESULTS2C["not_run"][step] = f"time budget: {elapsed_hours():.1f} h elapsed of {TIME_BUDGET_HOURS} h"
    save_progress_2c("not_run", RESULTS2C["not_run"])
    print(f"SKIPPED {step}: {RESULTS2C['not_run'][step]}")
    return False


print("device:", DEVICE, "| SMOKE_TEST:", SMOKE_TEST, "| torch", torch.__version__)

# %% [markdown]
# ## Shared voice-engine code (inlined from `engines/voice/`)

# %% inline engines/voice/data.py

# %% inline engines/voice/preprocess.py

# %% inline engines/voice/evaluate.py

# %% inline engines/voice/model.py

# %% inline engines/voice/features.py

# %% inline engines/voice/ssl.py

# %% [markdown]
# ## Task 1 -- data: every bonafide clip, a capped spoof sample, and the attack IDs

# %%
ITW_ARCHIVE = os.environ.get("TRUSTMESH_ITW_ARCHIVE", IN_THE_WILD_ARCHIVE)
ITW_PROTOCOL = os.environ.get("TRUSTMESH_ITW_PROTOCOL", "hf")

t0 = time.time()
if SMOKE_TEST:
    # Bonafide pads long enough that the real 0.2 s silence window fits; spoof pads stay far longer,
    # so the planted silence leakage is still there for the ablations to find.
    synthetic = make_synthetic_asvspoof(240, n_attacks=6, bonafide_pad_s=(0.12, 0.3))
    table = None
    meta = {"label": np.array([c["label"] for c in synthetic]),
            "utterance_id": [c["utterance_id"] for c in synthetic],
            "speaker_id": [c["speaker_id"] for c in synthetic],
            "columns": ["path", "audio", "label", "notes"], "note_keys": ["speaker_id", "utterance_id"]}
    protocol_path = write_synthetic_protocol(synthetic, OUTPUT_DIR / "synthetic_protocol.txt")
    dataset_name = "synthetic (smoke test)"
else:
    print(f"downloading {ASVSPOOF_DATASET} (4.39 GB parquet, cached by `datasets`) ...")
    table = load_asvspoof_table()
    meta = table_metadata(table)
    protocol_path = None
    dataset_name = f"{ASVSPOOF_DATASET} (split '{ASVSPOOF_SPLIT}' = LA evaluation partition)"
print(f"metadata for {len(meta['label'])} clips in {time.time() - t0:.0f}s")

# Log what the dataset itself exposes before reaching for the protocol file.
schema = {"columns": meta["columns"], "note_keys": meta["note_keys"],
          "attack_id_in_dataset": any(k in ("attack", "attack_id", "system_id") for k in meta["columns"] + meta["note_keys"])}
print("dataset schema:", schema)

try:
    protocol, protocol_source = load_eval_protocol(path=protocol_path)
    verification = verify_protocol(protocol, meta["utterance_id"], meta["label"], meta["speaker_id"])
except Exception as exc:  # noqa: BLE001 -- no protocol means no attack IDs, which is a recorded outcome
    protocol, protocol_source = {}, None
    verification = {"usable": False, "error": f"{type(exc).__name__}: {exc}"}
ATTACKS_AVAILABLE = bool(verification["usable"])
print("protocol verification:", verification)
print("attack-disjoint protocol:", "AVAILABLE" if ATTACKS_AVAILABLE else "NOT available -- unseen attacks stay unmeasured")

all_attacks = [protocol[u]["attack"] if u in protocol else "unknown" for u in meta["utterance_id"]]
selected = select_subset(meta["label"], SPOOF_PER_BONAFIDE, seed=SEED)
labels_np = meta["label"][selected].astype(np.int64)
speakers = [meta["speaker_id"][i] for i in selected]
attacks = [all_attacks[i] for i in selected]
N = len(selected)

from collections import Counter

data_summary = {
    "dataset": dataset_name,
    "schema": schema,
    "protocol_source": protocol_source,
    "protocol_verification": verification,
    "attacks_available": ATTACKS_AVAILABLE,
    "partition_counts": {"clips": int(len(meta["label"])), "bonafide": int((meta["label"] == BONAFIDE).sum()),
                         "spoof": int((meta["label"] == SPOOF).sum())},
    "spoof_per_bonafide_requested": SPOOF_PER_BONAFIDE,
    # The *actual* loaded counts -- the numbers to quote, never the requested ones.
    "loaded": {"clips": int(N), "bonafide": int((labels_np == BONAFIDE).sum()), "spoof": int((labels_np == SPOOF).sum()),
               "speakers": len(set(speakers))},
    "attack_counts_loaded": dict(sorted(Counter(attacks).items())),
    "phase2_loaded_for_comparison": {"clips": 6000, "bonafide": 619, "spoof": 5381},
}
print(json.dumps(data_summary["loaded"], indent=2))
save_progress("data", data_summary)

# %%
# One decode pass -> on-disk cache of every input the rest of the notebook needs. After a crash, rerunning
# this cell reuses the cache instead of decoding ~37k clips again.
def iter_selected_clips():
    if SMOKE_TEST:
        for i in selected:
            yield synthetic[i]
    else:
        for row in table.select(selected.tolist()):
            wave, sr = _decode_audio(row["audio"])
            yield {"waveform": wave, "sample_rate": sr}


extractor = LFCCExtractor().to(DEVICE)
t0 = time.time()
cache = build_feature_cache(iter_selected_clips(), N, CACHE_DIR, extractor, DEVICE, CLIP_SAMPLES, SILENCE_SAMPLES,
                            fixed_silence_s=FIXED_SILENCE_S, noise_dbfs=NOISE_DBFS, seed=SEED,
                            config_key=f"{dataset_name}|{SEED}|{SPOOF_PER_BONAFIDE}|{N}|{int(selected.sum())}",
                            progress_every=None if SMOKE_TEST else 2000)
print(f"feature cache ready in {time.time() - t0:.0f}s, {cache_size_gb(CACHE_DIR):.2f} GB in {CACHE_DIR}")

silence_by_class = {}
for key in ("lead_silence_s", "trail_silence_s", "speech_s", "silence_ratio"):
    values = np.array([s[key] for s in cache["stats"]])
    silence_by_class[key] = {"bonafide_mean": float(values[labels_np == BONAFIDE].mean()),
                             "spoof_mean": float(values[labels_np == SPOOF].mean())}
probe_before = duration_leakage_probe(cache["stats"], labels_np)
save_progress("silence_stats", {"by_class": silence_by_class, "timing_probe_before_trim": probe_before,
                                "timing_probe_verdict": leakage_verdict(probe_before),
                                "clips_with_silence_window": int(cache["has_silence_window"].sum())})
print("silence by class:", json.dumps(silence_by_class, indent=1))

# %% [markdown]
# ## Splits
#
# **P1 -- speaker-disjoint, shared attacks** (the base protocol, as in Phase 2).
# **P2 -- speaker- *and* attack-disjoint:** the *same* speaker partition, with spoofs from the held-out
# attacks removed from train/val and the test set restricted to bonafide + held-out-attack spoofs.
# Because the P2 test set is a subset of the P1 test set, a P1 model ("attacks seen in training") and a
# P2 model ("attacks unseen") can be compared on identical clips.

# %%
meta_clips = [{"speaker_id": speakers[i], "label": int(labels_np[i]), "attack": attacks[i]} for i in range(N)]


def sub_split(indices, fraction, seed):
    subset = [meta_clips[i] for i in indices]
    left, right = speaker_disjoint_split(subset, fraction, seed)
    return [indices[i] for i in left], [indices[i] for i in right]


def counts_of(idx):
    lab = labels_np[np.asarray(idx, dtype=np.int64)]
    return {"bonafide": int((lab == BONAFIDE).sum()), "spoof": int((lab == SPOOF).sum()), "total": int(len(lab))}


def has_both_classes(idx):
    c = counts_of(idx)
    return c["bonafide"] > 0 and c["spoof"] > 0


trainval, test = speaker_disjoint_split(meta_clips, TEST_FRACTION, SEED)
train, val = sub_split(trainval, VAL_FRACTION, SEED + 1)
SPLITS = {"P1": {"train": train, "val": val, "test": test}}
held_out, held_out_draws = [], 0
if ATTACKS_AVAILABLE:
    # Held-out attacks are a seeded draw. A draw is rejected only if it leaves some P2 split without a
    # class (e.g. every val spoof came from held-out attacks) -- a structural constraint, not tuning.
    for attempt in range(20):
        candidate = choose_held_out_attacks(attacks, N_HELD_OUT_ATTACKS, seed=SEED + attempt)
        p2 = {"train": drop_attacks(train, attacks, candidate), "val": drop_attacks(val, attacks, candidate),
              "test": keep_only_attacks(test, attacks, candidate)}
        if all(has_both_classes(p2[part]) for part in p2):
            held_out, held_out_draws, SPLITS["P2"] = candidate, attempt + 1, p2
            break
    if "P2" not in SPLITS:
        ATTACKS_AVAILABLE = False
        RESULTS["not_run"]["attack_disjoint_protocol"] = "no held-out-attack draw kept both classes in every P2 split"
else:
    RESULTS["not_run"]["attack_disjoint_protocol"] = ("attack IDs unavailable or failed verification -- "
                                                      "unseen-attack generalization remains unmeasured")

split_summary = {}
for name, split in SPLITS.items():
    train_speakers = {speakers[i] for i in split["train"]}
    test_speakers = {speakers[i] for i in split["test"]}
    split_summary[name] = {
        part: {**counts_of(split[part]), "attacks": sorted({attacks[i] for i in split[part]} - {"bonafide"})}
        for part in ("train", "val", "test")
    }
    split_summary[name]["shared_speakers_train_test"] = len(train_speakers & test_speakers)
    split_summary[name]["attack_overlap_train_test"] = sorted(
        set(split_summary[name]["train"]["attacks"]) & set(split_summary[name]["test"]["attacks"]))
    for part in ("train", "val", "test"):
        c = split_summary[name][part]
        print(f"{name} {part:5s}: {c['bonafide']:5d} bonafide / {c['spoof']:5d} spoof | attacks {c['attacks']}")
    print(f"{name} shared speakers train/test: {split_summary[name]['shared_speakers_train_test']} | "
          f"attack overlap train/test: {split_summary[name]['attack_overlap_train_test'] or 'none'}")
    for part in ("train", "val", "test"):
        c = split_summary[name][part]
        if not (c["bonafide"] and c["spoof"]):
            raise RuntimeError(f"{name} {part} split is missing a class: {c}")
if split_summary["P1"]["shared_speakers_train_test"]:
    print("WARNING: P1 fell back to a random split -- the headline is NOT speaker-disjoint")
save_progress("splits", {**split_summary, "held_out_attacks": held_out, "held_out_draws_needed": held_out_draws})

# %% [markdown]
# ## Task 3 -- models, one variable per run (on P1)
#
# A: LCNN, trimmed LFCC | B: A + augmentation (training batches only) | C: frozen XLS-R + layer-weighted head.

# %%
MODEL_RESULTS, SCORES, MODELS = {}, {}, {}
MODEL_RESULTS_2C, SCORES_2C, MODELS_2C = {}, {}, {}  # Phase 2C's own store -- see `run()`'s `store` param
LCNN_FRAMES = extractor.n_frames(CLIP_SAMPLES)
SILENCE_FRAMES = extractor.n_frames(SILENCE_SAMPLES)
ssl_features, ssl_info = None, {}


def fetch_for(features, augment=False, seed=SEED):
    if features == "trimmed":
        return wave_lfcc_source(cache["wave_trimmed"], extractor, DEVICE, augment=augment, seed=seed)
    if features == "silence_only":
        return lfcc_array_source(cache["lfcc_silence_only"], DEVICE)
    if features == "ssl":
        return array_source(ssl_features, DEVICE)
    return lfcc_array_source(cache["lfcc"][features], DEVICE)


def build_model(kind, features):
    if kind == "ssl":
        return SSLHead(ssl_info["n_layers"], ssl_info["feature_dim"])
    return LCNN(n_features=extractor.n_features, n_frames=SILENCE_FRAMES if features == "silence_only" else LCNN_FRAMES)


def run(name, group, kind, features, split_name, seed, augment=False, role="", split=None,
        batch_size=None, model_override=None, fetch_override=None, store=None, epochs=None, verbose=False):
    """Train on train, select on val, then score the test split exactly once.

    `store` = (results_dict, models_dict, scores_dict, save_fn), defaulting to the Phase 2B globals so
    every existing call site is unchanged. Phase 2C passes its own store so its runs never touch
    results_phase2b.json (the spec says the two phases' output files must stay separable).

    `verbose=True` (Phase 2C's Model F, the one run here slow enough that Kaggle's log otherwise shows
    nothing for a long time between "training ..." and the result line) prints one line per epoch, so
    the log stays visibly alive instead of looking stuck.
    """
    results_dict, models_dict, scores_dict, save_fn = store or (MODEL_RESULTS, MODELS, SCORES, save_progress)
    split = split or SPLITS[split_name]
    print(f"training {name} ({role}) ...")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_override if model_override is not None else build_model(kind, features)
    fetch = fetch_override if fetch_override is not None else fetch_for(features, augment, seed)
    lab = labels_np[np.asarray(split["train"])]
    weight = float((lab == SPOOF).sum() / max((lab == BONAFIDE).sum(), 1))
    bs = batch_size if batch_size is not None else BATCH_SIZE
    is_ssl_kind = kind in ("ssl", "ssl_seq")
    ep = epochs if epochs is not None else (SSL_EPOCHS if is_ssl_kind else EPOCHS)
    t_start = time.time()
    fit = train_model(model, fetch, labels_np, split["train"], weight, DEVICE, val_idx=split["val"],
                      epochs=ep, batch_size=bs,
                      lr=SSL_LR if is_ssl_kind else LEARNING_RATE,
                      patience=SSL_PATIENCE if is_ssl_kind else PATIENCE, seed=seed, verbose=verbose)
    test_scores = score_model(model, fetch, split["test"], batch_size=bs)
    test_labels = labels_np[np.asarray(split["test"])]
    ci = bootstrap_eer_ci(test_scores, test_labels, n_resamples=N_BOOTSTRAP, seed=seed)
    record = {
        "group": group, "kind": kind, "features": features, "split": split_name, "seed": seed,
        "augment": augment, "role": role, "test_eer": ci["eer"], "test_ci": [ci["ci_low"], ci["ci_high"]],
        "test_counts": counts_of(split["test"]), "train_counts": counts_of(split["train"]),
        "best_val_eer": fit["best_val_eer"], "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
        "validation_usable": fit["validation_usable"], "train_seconds": round(time.time() - t_start, 1),
        "n_params": model.n_params() if hasattr(model, "n_params") else None,
    }
    if kind == "ssl":
        record["layer_weights"] = model.layer_weights()
    results_dict[name], models_dict[name] = record, model
    scores_dict[name] = {"test_idx": np.asarray(split["test"]), "scores": test_scores, "labels": test_labels}
    save_fn("models", results_dict)
    print(f"{name:34s} val EER {fit['best_val_eer'] if fit['best_val_eer'] is not None else float('nan'):.4f} | "
          f"test EER {ci['eer']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] | "
          f"{record['test_counts']['bonafide']} bf / {record['test_counts']['spoof']} sp | "
          f"{fit['epochs_run']} ep {record['train_seconds']:.0f}s")
    return record


# Run A: data alone (the headline), 3 seeds.
for seed in SEEDS:
    run(f"A_lcnn_trimmed_P1_s{seed}", "A_lcnn_trimmed", "lcnn", "trimmed", "P1", seed, role="headline")

# Run B: A + augmentation, first seed (more seeds below if it wins on validation).
run(f"B_lcnn_trimmed_aug_P1_s{SEEDS[0]}", "B_lcnn_trimmed_aug", "lcnn", "trimmed", "P1", SEEDS[0], augment=True,
    role="augmentation")

# %%
# Run C: frozen SSL front-end. Encoder runs once over every clip; pooled layer features are cached.
t0 = time.time()
ssl_encoder = None
try:
    if SMOKE_TEST:
        ssl_encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=DEVICE, fp16=False)
        ssl_info["model"] = "tiny random wav2vec2 (smoke test)"
    else:
        for candidate in (SSL_MODEL, SSL_FALLBACK_MODEL):
            try:
                ssl_encoder = SSLEncoder(candidate, device=DEVICE, fp16=True)
                ssl_info["model"] = candidate
                break
            except Exception as exc:  # noqa: BLE001 -- try the fallback, record why
                ssl_info.setdefault("load_failures", []).append(f"{candidate}: {type(exc).__name__}: {exc}")
    if ssl_encoder is None:
        raise RuntimeError(f"no SSL encoder could be loaded: {ssl_info.get('load_failures')}")
    ssl_features = build_ssl_cache(ssl_encoder, cache["wave_trimmed"], CACHE_DIR / "ssl_pooled.npy",
                                   batch_size=SSL_BATCH, progress_every=None if SMOKE_TEST else 4000)
    ssl_info.update({"n_layers": ssl_encoder.n_layers, "feature_dim": ssl_encoder.feature_dim,
                     "pooling": "per-layer mean + std over frames", "frozen": True,
                     "extract_seconds": round(time.time() - t0, 1)})
    for seed in SEEDS:
        run(f"C_ssl_P1_s{seed}", "C_ssl_frozen", "ssl", "ssl", "P1", seed, role="frozen SSL front-end")
except Exception as exc:  # noqa: BLE001
    ssl_info["error"] = f"{type(exc).__name__}: {exc}"
    ssl_info["traceback"] = traceback.format_exc()
    RESULTS["not_run"]["run_C_ssl"] = ssl_info["error"]
    print("Run C failed, continuing without it:", ssl_info["error"])
save_progress("ssl", ssl_info)

# %%
# Model selection on validation only: compare the first seed of each family.
first_seed_runs = {}
for run_name, record in MODEL_RESULTS.items():
    if record["seed"] == SEEDS[0] and record["split"] == "P1":
        first_seed_runs[record["group"]] = run_name
val_by_group = {g: MODEL_RESULTS[n]["best_val_eer"] for g, n in first_seed_runs.items()
                if MODEL_RESULTS[n]["best_val_eer"] is not None}
best_group = min(val_by_group, key=val_by_group.get)
print("validation EER by family (first seed):", {g: round(v, 4) for g, v in val_by_group.items()})
print("best family on validation:", best_group)

# The best family gets 3 seeds too (A and C already have them).
if best_group == "B_lcnn_trimmed_aug" and within_budget("extra_seeds_B"):
    for seed in SEEDS[1:]:
        run(f"B_lcnn_trimmed_aug_P1_s{seed}", "B_lcnn_trimmed_aug", "lcnn", "trimmed", "P1", seed, augment=True,
            role="augmentation")

# Untrimmed: the silence shortcut, kept only as a comparison. Never a result.
run(f"U_lcnn_untrimmed_P1_s{SEEDS[0]}", "U_lcnn_untrimmed_SHORTCUT", "lcnn", "untrimmed", "P1", SEEDS[0],
    role="shortcut comparison -- NOT a result")
save_progress("model_selection", {"validation_eer_first_seed": val_by_group, "best_group": best_group,
                                  "rule": "lowest validation EER (P1 val split), first seed of each family"})

# %% [markdown]
# ## Task 2 -- attack-disjoint (P2): does it work on attacks it never saw?

# %%
ATTACK_GAPS = {}
if ATTACKS_AVAILABLE and within_budget("attack_disjoint_runs"):
    family_runs = {"A_lcnn_trimmed": ("lcnn", "trimmed", False), "B_lcnn_trimmed_aug": ("lcnn", "trimmed", True)}
    if "C_ssl_frozen" in val_by_group:
        family_runs["C_ssl_frozen"] = ("ssl", "ssl", False)
    p1_test_pos = {clip: k for k, clip in enumerate(SPLITS["P1"]["test"])}
    p2_test = SPLITS["P2"]["test"]
    for group, (kind, features, augment) in family_runs.items():
        p2_name = f"{group}_P2_s{SEEDS[0]}"
        run(p2_name, f"{group}_P2", kind, features, "P2", SEEDS[0], augment=augment, role="attack-disjoint")
        seen_name = first_seed_runs[group]
        seen_scores = SCORES[seen_name]["scores"][[p1_test_pos[c] for c in p2_test]]  # slice, don't rescore
        unseen_scores = SCORES[p2_name]["scores"]
        p2_labels = labels_np[np.asarray(p2_test)]
        gap = bootstrap_eer_difference(unseen_scores, p2_labels, seen_scores, p2_labels, n_resamples=N_BOOTSTRAP,
                                       seed=SEED, paired=True)
        p2_attacks = [attacks[i] for i in p2_test]
        ATTACK_GAPS[group] = {
            "test_counts": counts_of(p2_test), "held_out_attacks": held_out,
            "eer_attacks_seen_in_training": compute_eer(seen_scores, p2_labels)["eer"],
            "eer_attacks_unseen": compute_eer(unseen_scores, p2_labels)["eer"],
            "unseen_minus_seen": gap,
            "per_attack_seen": per_attack_eer(seen_scores, p2_labels, p2_attacks),
            "per_attack_unseen": per_attack_eer(unseen_scores, p2_labels, p2_attacks),
        }
        g = ATTACK_GAPS[group]
        print(f"{group:20s} same {g['test_counts']['bonafide']} bf / {g['test_counts']['spoof']} sp test clips: "
              f"seen {g['eer_attacks_seen_in_training']:.4f} -> unseen {g['eer_attacks_unseen']:.4f} "
              f"(gap {gap['difference']:+.4f}, CI [{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}])")
    save_progress("attack_disjoint", ATTACK_GAPS)

# %% [markdown]
# ## Task 4 -- silence: what is the shortcut made of?
#
# All ablations use the same P1 split and are compared with paired bootstraps on the same test clips.

# %%
SILENCE = {}
if within_budget("silence_ablations"):
    for mode in SILENCE_ABLATIONS:
        run(f"S_{mode}_P1_s{SEEDS[0]}", f"S_{mode}", "lcnn", mode, "P1", SEEDS[0], role="silence ablation")

    eligible = cache["has_silence_window"]
    silence_split = {part: [i for i in SPLITS["P1"][part] if eligible[i]] for part in ("train", "val", "test")}
    silence_ok = all(counts_of(silence_split[p])["bonafide"] and counts_of(silence_split[p])["spoof"]
                     for p in ("train", "val", "test"))
    if silence_ok:
        run(f"S_silence_only_P1_s{SEEDS[0]}", "S_silence_only", "lcnn", "silence_only", "P1", SEEDS[0],
            role="silence ablation", split=silence_split)
    else:
        RESULTS["not_run"]["silence_only"] = f"too few clips with {SILENCE_WINDOW_S}s of silence in a class: " + \
            json.dumps({p: counts_of(silence_split[p]) for p in silence_split})

    trimmed_ref, untrimmed_ref = first_seed_runs["A_lcnn_trimmed"], f"U_lcnn_untrimmed_P1_s{SEEDS[0]}"

    def paired(a, b, clips=None):
        """EER(a) - EER(b) on the same test clips (optionally restricted to `clips`)."""
        def take(name):
            idx, sc = SCORES[name]["test_idx"], SCORES[name]["scores"]
            if clips is None:
                return sc, labels_np[idx]
            pos = {c: k for k, c in enumerate(idx)}
            return sc[[pos[c] for c in clips]], labels_np[np.asarray(clips)]
        sa, la = take(a)
        sb, lb = take(b)
        return bootstrap_eer_difference(sa, la, sb, lb, n_resamples=N_BOOTSTRAP, seed=SEED, paired=True)

    comparisons = {"untrimmed_minus_trimmed": paired(untrimmed_ref, trimmed_ref)}
    for mode in SILENCE_ABLATIONS:
        name = f"S_{mode}_P1_s{SEEDS[0]}"
        comparisons[f"{mode}_minus_trimmed"] = paired(name, trimmed_ref)
        comparisons[f"{mode}_minus_untrimmed"] = paired(name, untrimmed_ref)
    silence_only_name = f"S_silence_only_P1_s{SEEDS[0]}"
    if silence_only_name in SCORES:
        test_e = silence_split["test"]
        so = SCORES[silence_only_name]
        comparisons["silence_only_ci"] = bootstrap_eer_ci(so["scores"], so["labels"], n_resamples=N_BOOTSTRAP, seed=SEED)
        # Same clips (those with enough silence) for every model in this comparison.
        comparisons["untrimmed_minus_trimmed_on_silence_subset"] = paired(untrimmed_ref, trimmed_ref, clips=test_e)
        comparisons["silence_only_minus_trimmed_on_silence_subset"] = paired(silence_only_name, trimmed_ref, clips=test_e)
        comparisons["silence_only_minus_untrimmed_on_silence_subset"] = paired(silence_only_name, untrimmed_ref, clips=test_e)

    # Verdicts, each stated with what it does NOT show.
    def significant(d, sign):
        return d["excludes_zero"] and (d["difference"] < 0 if sign < 0 else d["difference"] > 0)

    findings = []
    shortcut = comparisons["untrimmed_minus_trimmed"]
    shortcut_present = significant(shortcut, -1)
    findings.append(("shortcut", f"Untrimmed vs trimmed on the same clips: {shortcut['difference']:+.4f} EER "
                     f"(CI [{shortcut['ci_low']:+.4f}, {shortcut['ci_high']:+.4f}]) -- "
                     + ("the silence shortcut is present." if shortcut_present
                        else "no significant shortcut here, so there is nothing for the duration/replacement "
                             "ablations below to decompose; their numbers are recorded, not interpreted.")))
    dur = comparisons["zeros_original_duration_minus_trimmed"]
    dur_vs_untrimmed = comparisons["zeros_original_duration_minus_untrimmed"]
    if shortcut_present:
        if significant(dur, -1):
            text = (f"Silence replaced by digital zeros *at its original length* still beats trimmed by "
                    f"{-dur['difference']:.4f} EER, so silence duration/position alone reaches the model.")
            text += (f" It still trails untrimmed by {dur_vs_untrimmed['difference']:.4f} EER (CI excludes zero), so "
                     f"the silence's content adds information on top of its length." if significant(dur_vs_untrimmed, +1)
                     else " It is not significantly worse than untrimmed, so duration accounts for most of the shortcut.")
        else:
            text = ("Zeros at the original silence length do not beat trimmed significantly, so duration/position "
                    "alone does not reproduce the shortcut.")
        findings.append(("duration", text + " Does NOT show: which property of the content matters."))
    for mode in ("zeros_fixed", "noise_fixed"):
        d = comparisons[f"{mode}_minus_trimmed"]
        findings.append((mode, f"{mode}: {d['difference']:+.4f} EER vs trimmed (CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]) -- "
                         + ("not significantly different from trimmed" if not d["excludes_zero"]
                            else "significantly different from trimmed, so the replacement itself carries or disrupts signal")
                         + ". Does NOT separate duration from content on its own."))
    if "silence_only_ci" in comparisons:
        so_ci = comparisons["silence_only_ci"]
        carries = so_ci["ci_high"] < 0.5
        findings.append(("content", f"A classifier given only {SILENCE_WINDOW_S}s of each clip's genuine silence "
                         f"(fixed length, so no duration information) reaches EER {so_ci['eer']:.4f} "
                         f"[{so_ci['ci_low']:.4f}, {so_ci['ci_high']:.4f}] on {so_ci['n_bonafide']} bf / {so_ci['n_spoof']} sp "
                         + ("-- the silence *content* carries label information beyond duration."
                            if carries else "-- not distinguishable from chance, so silence content alone is not informative.")
                         + " Does NOT show: which acoustic property (noise floor, spectral shape, codec) it is, and it "
                         "only covers clips with at least that much silence."))
    SILENCE = {"comparisons": comparisons, "findings": [{"test": k, "text": t} for k, t in findings]}
    for _, t in findings:
        print("-", t)
    save_progress("silence_ablation", SILENCE)

# %% [markdown]
# ## Free the LFCC variant caches Task 3/4 needed and nothing after this needs
#
# `cache["lfcc"]` (untrimmed + the 3 silence ablations) and `cache["lfcc_silence_only"]` are only read
# by the runs just above; every model score is already saved in `SCORES`, so the raw arrays aren't
# needed again -- not by Task 5/6 below, and not by Phase 2C, which extends this notebook in the same
# Kaggle session. Freeing them here (~7 GB) rather than only at the very end is what makes room for
# Phase 2C's own cache needs later in this session (issue #25: an earlier run overran Kaggle's disk
# because nothing was freed until the end).

# %%
import gc

_freed_gb = 0.0
for _variant in list(cache["lfcc"].keys()):
    cache["lfcc"][_variant] = None
cache["lfcc_silence_only"] = None
gc.collect()
for _path in [CACHE_DIR / f"lfcc_{v}.npy" for v in LFCC_VARIANTS] + [CACHE_DIR / "lfcc_silence_only.npy"]:
    if _path.exists():
        _freed_gb += _path.stat().st_size / 1e9
        _path.unlink()
print(f"freed {_freed_gb:.2f} GB of LFCC variant caches no longer needed after Task 4")

# %% [markdown]
# ## Task 5 -- rigour: seeds, CIs, and how much the CI shrank

# %%
def group_summary(group):
    runs = [r for r in MODEL_RESULTS.values() if r["group"] == group]
    eers = [r["test_eer"] for r in runs]
    first = min(runs, key=lambda r: r["seed"])
    return {"group": group, "kind": first["kind"], "features": first["features"], "split": first["split"],
            "role": first["role"], "seeds": [r["seed"] for r in runs], "eer_per_seed": eers,
            "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers)) if len(eers) > 1 else None,
            "ci_first_seed": first["test_ci"], "test_counts": first["test_counts"], "train_counts": first["train_counts"]}


LEADERBOARD = sorted((group_summary(g) for g in {r["group"] for r in MODEL_RESULTS.values()}),
                     key=lambda row: (row["split"], row["eer_mean"]))
headline = group_summary("A_lcnn_trimmed")
best = group_summary(best_group)
headline_first = first_seed_runs["A_lcnn_trimmed"]
ci_full = bootstrap_eer_ci(SCORES[headline_first]["scores"], SCORES[headline_first]["labels"], n_resamples=N_BOOTSTRAP, seed=SEED)
ci_phase2_size = bootstrap_ci_at_counts(SCORES[headline_first]["scores"], SCORES[headline_first]["labels"],
                                        PHASE2_TEST_COUNTS["bonafide"], PHASE2_TEST_COUNTS["spoof"],
                                        n_resamples=N_BOOTSTRAP, seed=SEED)
ci_shrinkage = {"ci_width_at_phase2b_test_size": ci_full["ci_width"], "phase2b_test_counts": headline["test_counts"],
                "ci_width_at_phase2_test_size": ci_phase2_size["ci_width"], "phase2_test_counts": PHASE2_TEST_COUNTS,
                "width_ratio": ci_full["ci_width"] / ci_phase2_size["ci_width"] if ci_phase2_size["ci_width"] else None,
                "method": "same headline model's test scores, bootstrapped at Phase 2's class counts vs the full Phase 2B test set"}
print(f"{'group':30s} {'split':5s} {'seeds':>5s} {'EER mean':>9s} {'std':>7s}  CI (first seed)       test bf/sp")
for row in LEADERBOARD:
    std = f"{row['eer_std']:.4f}" if row["eer_std"] is not None else "   -  "
    print(f"{row['group']:30s} {row['split']:5s} {len(row['seeds']):5d} {row['eer_mean']:9.4f} {std:>7s}  "
          f"[{row['ci_first_seed'][0]:.4f}, {row['ci_first_seed'][1]:.4f}]   "
          f"{row['test_counts']['bonafide']}/{row['test_counts']['spoof']}")
ratio_text = f"{ci_shrinkage['width_ratio']:.2f}" if ci_shrinkage["width_ratio"] is not None else "n/a"
print(f"\nCI width on the headline: {ci_full['ci_width']:.4f} at {headline['test_counts']['bonafide']} bonafide test clips vs "
      f"{ci_phase2_size['ci_width']:.4f} at Phase 2's {PHASE2_TEST_COUNTS['bonafide']} (ratio {ratio_text})")
save_progress("leaderboard", LEADERBOARD)
save_progress("ci_shrinkage", ci_shrinkage)

# %% [markdown]
# ## Task 6 -- In-the-Wild, rerun with the fixed loader

# %%
ITW = {"status": "not_run"}
if within_budget("in_the_wild"):
    try:
        archive = ZipAudioArchive(ITW_ARCHIVE, protocol=ITW_PROTOCOL)
        itw_clips, itw_report = load_archive_clips_balanced(archive, N_ITW_PER_CLASS, seed=SEED)
        itw_labels = np.array([c["label"] for c in itw_clips])
        trimmer = VoicePreprocessor(clip_seconds=CLIP_SECONDS, trim=True)
        raw_pre = VoicePreprocessor(clip_seconds=CLIP_SECONDS, trim=False)
        itw_waves = np.stack([to_int16(trimmer.prepare_waveform(c["waveform"], c["sample_rate"])[0]) for c in itw_clips])
        with torch.no_grad():
            itw_untrimmed = np.stack([
                extractor(torch.from_numpy(quantize(raw_pre.prepare_waveform(c["waveform"], c["sample_rate"])[0]))
                          .unsqueeze(0).to(DEVICE)).half().cpu().numpy()[0] for c in itw_clips])
        itw_fetch = {"trimmed": wave_lfcc_source(itw_waves, extractor, DEVICE),
                     "untrimmed": lfcc_array_source(itw_untrimmed, DEVICE)}
        if ssl_encoder is not None:
            itw_ssl = np.concatenate([ssl_encoder.pooled(torch.from_numpy(itw_waves[s:s + SSL_BATCH].astype(np.float32) / INT16_SCALE))
                                      .cpu().numpy().astype(np.float16) for s in range(0, len(itw_waves), SSL_BATCH)])
            itw_fetch["ssl"] = array_source(itw_ssl, DEVICE)

        to_score = {"headline_A": headline_first, "best_" + best_group: first_seed_runs[best_group],
                    "untrimmed_SHORTCUT": f"U_lcnn_untrimmed_P1_s{SEEDS[0]}"}
        if "C_ssl_frozen" in first_seed_runs:
            to_score["ssl_C"] = first_seed_runs["C_ssl_frozen"]
        itw_scores = {}
        for label, name in to_score.items():
            features = MODEL_RESULTS[name]["features"]
            sc = score_model(MODELS[name], itw_fetch[features], np.arange(len(itw_clips)))
            ci = bootstrap_eer_ci(sc, itw_labels, n_resamples=N_BOOTSTRAP, seed=SEED)
            itw_scores[label] = {"model": name, "eer": ci["eer"], "ci": [ci["ci_low"], ci["ci_high"]],
                                 "in_domain_eer": MODEL_RESULTS[name]["test_eer"]}
            print(f"In-the-Wild {label:24s} EER {ci['eer']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
                  f"(in-domain {MODEL_RESULTS[name]['test_eer']:.4f})")
        ITW = {"status": "ok", "source": ITW_ARCHIVE, "report": itw_report,
               "counts": {"bonafide": int((itw_labels == BONAFIDE).sum()), "spoof": int((itw_labels == SPOOF).sum())},
               "models": itw_scores}
    except Exception as exc:  # noqa: BLE001 -- recorded as a new issue, never claimed as generalization
        ITW = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        print("In-the-Wild failed -- log this traceback as a new issue:\n", ITW["traceback"])
save_progress("in_the_wild", ITW)

# %% [markdown]
# ## Save: DET curves, weights, scores, closing paragraph

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))
for name, record in MODEL_RESULTS.items():
    if record["seed"] != SEEDS[0]:
        continue
    det = det_curve_points(SCORES[name]["scores"], SCORES[name]["labels"])
    style = "--" if record["split"] == "P2" else ":" if record["group"].startswith(("U_", "S_")) else "-"
    plt.plot(np.array(det["far"]) * 100, np.array(det["frr"]) * 100, style,
             label=f"{record['group']} [{record['split']}] EER {record['test_eer']:.3f}")
plt.plot([0.01, 100], [0.01, 100], "k-", linewidth=0.5, alpha=0.4)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("False accept rate (%) -- spoof accepted as bonafide")
plt.ylabel("False reject rate (%) -- bonafide rejected")
plt.title("Phase 2B DET curves (solid: P1, dashed: attack-disjoint P2, dotted: shortcut/ablations)")
plt.legend(fontsize=6, loc="lower left")
plt.grid(True, which="both", linewidth=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "det_curves_phase2b.png", dpi=130)
plt.show()

for name, model in MODELS.items():
    torch.save(model.state_dict(), OUTPUT_DIR / f"phase2b_{name}.pt")
np.savez_compressed(OUTPUT_DIR / "phase2b_test_scores.npz",
                    **{f"{n}__{f}": SCORES[n][f] for n in SCORES for f in ("test_idx", "scores", "labels")})

# --- Closing paragraph, generated from this run's numbers -------------------------------------------
parts = [
    f"Headline (LCNN on trimmed LFCC, speaker-disjoint, shared attacks): EER {headline['eer_mean']:.4f} +/- "
    f"{headline['eer_std']:.4f} over {len(headline['seeds'])} seeds (95% CI of the first seed "
    f"[{headline['ci_first_seed'][0]:.4f}, {headline['ci_first_seed'][1]:.4f}]) on {headline['test_counts']['bonafide']} "
    f"bonafide / {headline['test_counts']['spoof']} spoof test clips, trained on {data_summary['loaded']['bonafide']} "
    f"bonafide clips (Phase 2: 619). At Phase 2's test size the same model's CI would be "
    f"{ci_shrinkage['ci_width_at_phase2_test_size']:.4f} wide; here it is {ci_shrinkage['ci_width_at_phase2b_test_size']:.4f}.",
]
if best_group != "A_lcnn_trimmed":
    parts.append(f"Best family on validation: {best_group}, test EER {best['eer_mean']:.4f}"
                 + (f" +/- {best['eer_std']:.4f} over {len(best['seeds'])} seeds" if best["eer_std"] is not None else "")
                 + f" (CI [{best['ci_first_seed'][0]:.4f}, {best['ci_first_seed'][1]:.4f}]).")
else:
    parts.append("Neither augmentation nor the frozen SSL front-end beat the plain LCNN on validation.")
if ATTACK_GAPS:
    gap_text = "; ".join(
        f"{g}: {v['eer_attacks_seen_in_training']:.4f} seen -> {v['eer_attacks_unseen']:.4f} unseen "
        f"(gap {v['unseen_minus_seen']['difference']:+.4f}, CI [{v['unseen_minus_seen']['ci_low']:+.4f}, "
        f"{v['unseen_minus_seen']['ci_high']:+.4f}])" for g, v in ATTACK_GAPS.items())
    parts.append(f"Attack-disjoint (held-out {', '.join(held_out)}, same test clips): {gap_text}.")
else:
    parts.append("Attack-disjoint: not measured -- " + RESULTS["not_run"].get(
        "attack_disjoint_protocol", RESULTS["not_run"].get("attack_disjoint_runs", "not run")) + ".")
if SILENCE:
    parts.append("Silence ablations: " + " ".join(f["text"] for f in SILENCE["findings"]))
else:
    parts.append("Silence ablations: not run (" + RESULTS["not_run"].get("silence_ablations", "unknown") + ").")
if ITW.get("status") == "ok":
    itw_text = "; ".join(f"{k} {v['eer']:.4f} [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] (in-domain {v['in_domain_eer']:.4f})"
                         for k, v in ITW["models"].items())
    parts.append(f"In-the-Wild ({ITW['counts']['bonafide']} bonafide / {ITW['counts']['spoof']} spoof clips, "
                 f"{ITW['report']['failed']} failed to load): {itw_text}.")
else:
    parts.append(f"In-the-Wild: {ITW.get('status')} ({ITW.get('error', RESULTS['not_run'].get('in_the_wild', ''))}) "
                 "-- no generalization claim.")
parts.append("Scope limits that remain: every clip is from the LA evaluation partition (not the official "
             "train/eval protocol), so no comparison with published ASVspoof numbers is valid; attack-disjoint uses "
             f"{len(held_out)} held-out attacks from one seeded draw; the SSL encoder is frozen, not fine-tuned.")
closing_paragraph = " ".join(parts)
print(closing_paragraph)
save_progress("closing_paragraph", closing_paragraph)
save_progress("run_hours", round(elapsed_hours(), 2))
print(f"\nPhase 2B results: {RESULTS_PATH} | elapsed {elapsed_hours():.2f} h")

# %% [markdown]
# # Phase 2C -- Voice Engine: SSL head upgrade (Tier 1 only)
#
# Implements `docs/PHASE_2C_VOICE_SSL_DEPTH.md`. Extends this notebook in place (same file, same
# Kaggle session, same `TIME_BUDGET_HOURS` clock) rather than a new notebook, reusing Phase 2B's
# frozen SSL encoder, cached features, splits and In-the-Wild data computed above. Writes its own
# `results_phase2c.json`, never merged into `results_phase2b.json`.
#
# Model C (Phase 2B's best, frozen XLS-R + layer-weighted head) is the fixed baseline. This phase
# asks one question: does a richer head on top of the same frozen features close more of its 0.10
# In-the-Wild EER, without touching the frozen encoder or reusing any held-out test set twice.

# %% [markdown]
# ## Task 0 -- pre-flight: what Model C already contains
#
# Model C's pooling is already logged above as `ssl_info["pooling"]` = `"per-layer mean + std over
# frames"` -- the spec's premise that Model C pools mean-only does **not** hold in this codebase
# (`SSLEncoder.pooled` in `ssl.py` concatenates mean and std before `SSLHead` ever sees it). Task 1 is
# reframed per that finding: Model D ablates the std half of Model C's *own* pooling (mean-only),
# instead of adding std as a "new" variable Model C already has.

# %%
PHASE2C = {"prerequisite": None}
if "C_ssl_frozen" not in first_seed_runs or ssl_encoder is None:
    PHASE2C["prerequisite"] = ("Model C (Run C) did not complete -- Phase 2C needs its frozen encoder, "
                               "cache, and learned layer weights, so nothing below can run.")
    RESULTS2C["not_run"]["phase2c"] = PHASE2C["prerequisite"]
    save_progress_2c("not_run", RESULTS2C["not_run"])
    print("SKIPPING Phase 2C:", PHASE2C["prerequisite"])
else:
    c_name = first_seed_runs["C_ssl_frozen"]
    c_model = MODELS[c_name]
    c_layer_weights = c_model.layer_weights()
    sorted_layers = sorted(range(len(c_layer_weights)), key=lambda i: -c_layer_weights[i])
    half = len(c_layer_weights) // 2
    preflight = {
        "model_c_pooling": ssl_info.get("pooling"),
        "spec_premise_mean_only_holds": ssl_info.get("pooling") == "mean over frames",
        "layer_weights_sorted": [{"layer": i, "weight": c_layer_weights[i]} for i in sorted_layers],
        "top_layer": sorted_layers[0],
        "early_vs_late": ("early/middle layers carry more weight" if sum(c_layer_weights[:half]) > 0.5
                          else "late layers carry more weight"),
        "model_c_n_params": c_model.n_params(),
    }
    print("Model C pooling (already computed by Run C, not re-derived):", preflight["model_c_pooling"])
    print("Model C's learned layer weights, sorted heaviest first:", json.dumps(preflight["layer_weights_sorted"], indent=1))
    print(preflight["early_vs_late"] + " -- tests the 'early layers carry more spoofing-relevant detail' hypothesis.")
    save_progress_2c("preflight", preflight)

# %% [markdown]
# ## Task 1 -- Model D: mean-only ablation of Model C's own pooling
#
# The one-variable comparison the pre-flight finding makes honest: Model C (mean+std) vs. Model D
# (mean only), on the *same* cached features -- no new encoder pass, since `ssl.py`'s
# `SSLEncoder.pooled` already concatenates `[mean, std]` per layer and Model D just reads the first
# half of the existing cache (`array_mean_only_source`).

# %%
if PHASE2C["prerequisite"] is None:
    def fetch_D(seed):
        return array_mean_only_source(ssl_features, ssl_encoder.hidden_size, DEVICE)

    def build_D():
        return SSLHead(ssl_info["n_layers"], ssl_encoder.hidden_size)

# %% [markdown]
# ## Task 2 -- Model E: a lightweight temporal model on frozen SSL tokens
#
# Caching every layer's full per-frame sequence is infeasible: at ~200 frames x 1024-dim x
# `ssl_info["n_layers"]` layers x 2 bytes (float16) x the loaded clip count, that is on the order of
# hundreds of GB -- far past what Kaggle's `/kaggle/working` can hold. Model E therefore keeps only the
# single most-weighted layer from Model C's learned weights above (logged in the pre-flight step), so
# there is no per-layer weighting mechanism here (unlike `SSLHead`, which mixes every layer) -- that
# trade-off is the point of this deviation from the spec's literal wording.
#
# **Disk, not just compute, is the constraint (issue #25):** even a *single* layer's full-dataset
# sequence cache is ~15 GB, and a first attempt that pre-built it as a disk cache (like every other SSL
# run here) failed a real Kaggle run with "tried to use more disk space than is available" once stacked
# on top of the feature/SSL caches this notebook already holds. Model E therefore reads its sequence
# on the fly per batch instead -- `wave_ssl_sequence_source`, the same frozen-encoder-per-batch pattern
# Model F already uses for a different reason (augmentation) -- trading GPU time for zero added disk.
# A single-layer BiLSTM (chosen over a TCN: fewer moving parts to implement correctly, and
# `torch.nn.LSTM` is a tested, batched primitive) is kept deliberately small (hidden size 64) so the
# comparison tests temporal modeling, not added capacity.
#
# **Issue #26 (a second real Kaggle finding, on top of #25):** the first fix for #25 made Model E read
# its sequence on the fly *every* call, including the validation-EER check `train_model` runs every
# epoch and the one final test-scoring call. Model E has **no augmentation at all**, so its features
# for a given clip never change between calls -- recomputing them from the raw waveform every epoch was
# pure wasted GPU time on top of the already-expensive per-batch encoder forward, and is very likely
# why a real run was still going after 8+ hours. The correct fix, now that Task 4's early cache cleanup
# (above) frees real headroom: build the sequence **once**, as a disk cache, the way every other SSL
# run here already works -- the one-time extraction pass costs about what Run C's full-dataset pooled
# cache cost in Phase 2B (~2-5 minutes on the whole loaded dataset), and every epoch after that reads
# it, exactly like Model D and Model C do. Freed right after Model E's P2 run below (not held to the
# end), so it doesn't coexist with Model F's own disk-light but GPU-heavy training.

# %%
E_AVAILABLE = RUN_MODEL_E and PHASE2C["prerequisite"] is None and within_budget_2c("model_E_sequence_cache")
if not RUN_MODEL_E and PHASE2C["prerequisite"] is None:
    RESULTS2C["not_run"]["model_E"] = ("skipped this run (RUN_MODEL_E=False) -- its full-dataset sequence "
                                       "cache overran this environment's disk quota on a real attempt "
                                       "(issue #28); set TRUSTMESH_RUN_MODEL_E=1 to include it")
    save_progress_2c("not_run", RESULTS2C["not_run"])
    print("SKIPPING Model E:", RESULTS2C["not_run"]["model_E"])
if E_AVAILABLE:
    E_LAYER = preflight["top_layer"]
    t0 = time.time()
    try:
        e_seq_cache = build_ssl_sequence_cache(ssl_encoder, cache["wave_trimmed"], E_LAYER,
                                               CACHE_DIR / f"ssl_seq_layer{E_LAYER}.npy", batch_size=SSL_BATCH,
                                               progress_every=None if SMOKE_TEST else 4000)
        cache_cost = {"layer_used": E_LAYER, "shape": list(e_seq_cache.shape),
                      "single_layer_cache_gb": round(e_seq_cache.nbytes / 1e9, 2),
                      "extract_seconds": round(time.time() - t0, 1)}
        print("Model E sequence cache:", json.dumps(cache_cost, indent=1))
        save_progress_2c("model_e_cache_cost", cache_cost)

        def fetch_E(seed):
            return array_source(e_seq_cache, DEVICE)

        def build_E():
            return SSLSeqHead(ssl_encoder.hidden_size, lstm_hidden=64)
    except Exception as exc:  # noqa: BLE001 -- Model E is optional; record why, continue without it
        RESULTS2C["not_run"]["model_E"] = f"{type(exc).__name__}: {exc}"
        save_progress_2c("not_run", RESULTS2C["not_run"])
        print("Model E sequence cache failed, continuing without it:", RESULTS2C["not_run"]["model_E"])
        E_AVAILABLE = False

# %% [markdown]
# ## Task 3 -- Model F: training-time augmentation on Model C's recipe
#
# Same architecture as Model C (mean+std pooling, same layer-weighted head), but augmentation must
# happen *before* the frozen encoder's forward pass -- unlike every other SSL run here, which reads
# the pre-computed cache, Model F's **training** batches re-run the frozen encoder (`wave_ssl_source`).
# Its validation/test calls do not: they are never augmented, so they read Phase 2B's already-built
# `ssl_features` cache instead of recomputing identical features from the waveform every epoch (the
# other half of issue #26 -- the same redundant-eval mistake Model E made). Augmentation is Run B's
# menu (noise, gain, SpecAugment) plus one light codec/compression simulation that Phase 2B explicitly
# skipped for time -- see `simulate_low_bitrate` in `preprocess.py` for why it is a documented mu-law +
# bandwidth-reduction proxy, not a real MP3 round-trip.
#
# **Still the slowest model here even after the #26 fix**, because every *training* batch genuinely
# needs a live encoder forward pass -- that part can't be cached. Its epoch cap is lowered to
# `SSL_EPOCHS_ONTHEFLY` (Model C converged in 13-44 epochs on the cached path, so this doesn't cut
# training short in practice) and it prints one line per epoch (`verbose=True`) so a long run shows
# visible progress in Kaggle's log instead of looking stuck between the start and result lines.

# %%
if PHASE2C["prerequisite"] is None:
    def fetch_F(seed):
        return wave_ssl_source(cache["wave_trimmed"], ssl_encoder, DEVICE, augment=True, seed=seed,
                               codec_sim=True, pooled_cache=ssl_features)

    def build_F():
        return SSLHead(ssl_info["n_layers"], ssl_info["feature_dim"])

# %% [markdown]
# ## Train D, E, F: P1 (3 seeds), P2 attack-disjoint (first seed), In-the-Wild
#
# Same protocol Model C got in Phase 2B, so every comparison below is apples-to-apples.

# %%
PHASE2C_FAMILIES = {}
if PHASE2C["prerequisite"] is None:
    STORE_2C = (MODEL_RESULTS_2C, MODELS_2C, SCORES_2C, save_progress_2c)
    # (group, role, build_fn, fetch_fn, augment, batch_size, epochs, verbose) -- epochs/verbose default
    # to run()'s own SSL_EPOCHS/False for D and E (cheap, cache-backed); Model F re-runs the frozen
    # encoder every batch, so it gets a lower epoch cap and per-epoch progress prints (see SSL_EPOCHS_ONTHEFLY).
    specs = [("D_ssl_mean_only", "mean-only pooling ablation", build_D, fetch_D, False, BATCH_SIZE, None, False)]
    if E_AVAILABLE:
        specs.append(("E_ssl_bilstm", "temporal model (BiLSTM) on frozen SSL tokens", build_E, fetch_E, False,
                      BATCH_SIZE, None, False))
    if RUN_MODEL_F:
        specs.append(("F_ssl_augmented", "training-time augmentation on Model C's recipe", build_F, fetch_F, True,
                      SSL_BATCH, SSL_EPOCHS_ONTHEFLY, True))
    else:
        RESULTS2C["not_run"]["model_F"] = ("skipped this run to keep total Kaggle time down (RUN_MODEL_F=False) "
                                           "-- set TRUSTMESH_RUN_MODEL_F=1 to include it; Model C/D/E are unaffected")
        save_progress_2c("not_run", RESULTS2C["not_run"])
        print("SKIPPING Model F:", RESULTS2C["not_run"]["model_F"])

    for group, role, build_fn, fetch_fn, augment, bs, ep, verb in specs:
        kind = "ssl_seq" if group == "E_ssl_bilstm" else "ssl"
        seed_names = []
        for seed in SEEDS:
            if not within_budget_2c(f"{group}_P1"):
                break
            name = f"{group}_P1_s{seed}"
            run(name, group, kind, group.lower(), "P1", seed, augment=augment, role=role,
               model_override=build_fn(), fetch_override=fetch_fn(seed), batch_size=bs, store=STORE_2C,
               epochs=ep, verbose=verb)
            seed_names.append(name)
        if seed_names:
            PHASE2C_FAMILIES[group] = {"kind": kind, "seed_names": seed_names, "first_seed": seed_names[0]}

# %% [markdown]
# ## Attack-disjoint (P2): same test clips as Model C's comparison

# %%
PHASE2C_ATTACK_GAPS = {}
if PHASE2C["prerequisite"] is None and ATTACKS_AVAILABLE and PHASE2C_FAMILIES and within_budget_2c("phase2c_attack_disjoint"):
    p1_test_pos = {clip: k for k, clip in enumerate(SPLITS["P1"]["test"])}
    p2_test = SPLITS["P2"]["test"]
    p2_labels = labels_np[np.asarray(p2_test)]
    for group, fam in PHASE2C_FAMILIES.items():
        spec = next(s for s in specs if s[0] == group)
        _, role, build_fn, fetch_fn, augment, bs, ep, verb = spec
        seed = SEEDS[0]
        p2_name = f"{group}_P2_s{seed}"
        run(p2_name, f"{group}_P2", fam["kind"], group.lower(), "P2", seed, augment=augment,
           role="attack-disjoint", model_override=build_fn(), fetch_override=fetch_fn(seed), batch_size=bs,
           store=STORE_2C, epochs=ep, verbose=verb)
        seen_scores = SCORES_2C[fam["first_seed"]]["scores"][[p1_test_pos[c] for c in p2_test]]
        unseen_scores = SCORES_2C[p2_name]["scores"]
        gap = bootstrap_eer_difference(unseen_scores, p2_labels, seen_scores, p2_labels, n_resamples=N_BOOTSTRAP,
                                       seed=SEED, paired=True)
        PHASE2C_ATTACK_GAPS[group] = {
            "test_counts": counts_of(p2_test), "held_out_attacks": held_out,
            "eer_attacks_seen_in_training": compute_eer(seen_scores, p2_labels)["eer"],
            "eer_attacks_unseen": compute_eer(unseen_scores, p2_labels)["eer"],
            "unseen_minus_seen": gap,
        }
        g = PHASE2C_ATTACK_GAPS[group]
        print(f"{group:20s} same {g['test_counts']['bonafide']} bf / {g['test_counts']['spoof']} sp test clips: "
              f"seen {g['eer_attacks_seen_in_training']:.4f} -> unseen {g['eer_attacks_unseen']:.4f} "
              f"(gap {gap['difference']:+.4f}, CI [{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}])")
    save_progress_2c("attack_disjoint", PHASE2C_ATTACK_GAPS)

# %% [markdown]
# ## Free Model E's sequence cache -- nothing after this point reads it
#
# In-the-Wild scores Model E by re-running the encoder directly on the ~200 ITW clips (cheap), not by
# reading this cache, and Model F's remaining training never touches it either.

# %%
if "e_seq_cache" in dir() and e_seq_cache is not None:
    e_seq_cache = None
    gc.collect()
    p = CACHE_DIR / f"ssl_seq_layer{E_LAYER}.npy"
    if p.exists():
        freed = p.stat().st_size / 1e9
        p.unlink()
        print(f"freed {freed:.2f} GB (Model E's sequence cache, no longer needed)")

# %% [markdown]
# ## In-the-Wild: D, E, F scored on the same clips Model C was scored on

# %%
PHASE2C_ITW = {}
if (PHASE2C["prerequisite"] is None and PHASE2C_FAMILIES and ITW.get("status") == "ok"
        and within_budget_2c("phase2c_in_the_wild")):
    # Recomputed (cheap: ~200 clips), not read back from ITW["models"], which only kept the EER/CI
    # summary, not the raw per-clip scores a paired comparison needs.
    c_itw_raw = score_model(MODELS[c_name], itw_fetch["ssl"], np.arange(len(itw_clips))) if "ssl" in itw_fetch else None
    try:
        itw_int16 = itw_waves  # trimmed, int16 -- same clips as Model C's ITW pass above
        for group, fam in PHASE2C_FAMILIES.items():
            model = MODELS_2C[fam["first_seed"]]
            if group == "D_ssl_mean_only":
                feats = ssl_encoder.mean_only(torch.from_numpy(np.asarray(itw_ssl, dtype=np.float32)))
                sc = score_model(model, lambda idx, training, feats=feats: feats[np.asarray(idx)].to(DEVICE),
                                 np.arange(len(itw_int16)))
            elif group == "E_ssl_bilstm":
                seq_chunks = [ssl_encoder.sequence(torch.from_numpy(itw_int16[s:s + SSL_BATCH].astype(np.float32) / INT16_SCALE), E_LAYER)
                             for s in range(0, len(itw_int16), SSL_BATCH)]
                itw_seq = torch.cat(seq_chunks).cpu()
                sc = score_model(model, lambda idx, training, itw_seq=itw_seq: itw_seq[np.asarray(idx)].to(DEVICE),
                                 np.arange(len(itw_int16)))
            else:  # F_ssl_augmented -- on-the-fly, no augmentation at eval (fetch_override has augment=True
                   # only applied when training=True, and score_model always calls with training=False)
                fetch = wave_ssl_source(itw_int16, ssl_encoder, DEVICE)
                sc = score_model(model, fetch, np.arange(len(itw_int16)))
            ci = bootstrap_eer_ci(sc, itw_labels, n_resamples=N_BOOTSTRAP, seed=SEED)
            in_domain = MODEL_RESULTS_2C[fam["first_seed"]]["test_eer"]
            entry = {"model": fam["first_seed"], "eer": ci["eer"], "ci": [ci["ci_low"], ci["ci_high"]],
                    "in_domain_eer": in_domain}
            if c_itw_raw is not None:
                vs_c = bootstrap_eer_difference(sc, itw_labels, c_itw_raw, itw_labels,
                                               n_resamples=N_BOOTSTRAP, seed=SEED, paired=True)
                entry["vs_model_c"] = {"difference": vs_c["difference"], "ci_low": vs_c["ci_low"],
                                       "ci_high": vs_c["ci_high"], "excludes_zero": vs_c["excludes_zero"]}
            PHASE2C_ITW[group] = entry
            print(f"In-the-Wild {group:20s} EER {ci['eer']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
                  f"(in-domain {in_domain:.4f})")
        save_progress_2c("in_the_wild", PHASE2C_ITW)
    except Exception as exc:  # noqa: BLE001 -- recorded, never claimed as generalization
        RESULTS2C["not_run"]["in_the_wild"] = f"{type(exc).__name__}: {exc}"
        save_progress_2c("not_run", RESULTS2C["not_run"])
        print("Phase 2C In-the-Wild failed:", RESULTS2C["not_run"]["in_the_wild"])
elif PHASE2C["prerequisite"] is None:
    RESULTS2C["not_run"]["in_the_wild"] = "Phase 2B's own In-the-Wild pass did not succeed, so there are no clips to reuse"
    save_progress_2c("not_run", RESULTS2C["not_run"])

# %% [markdown]
# ## Results: one leaderboard, C carried over unchanged, plus D, E, F

# %%
if PHASE2C["prerequisite"] is None:
    def group_summary_2c(group):
        runs = [r for r in MODEL_RESULTS_2C.values() if r["group"] == group]
        eers = [r["test_eer"] for r in runs]
        first = min(runs, key=lambda r: r["seed"])
        return {"group": group, "kind": first["kind"], "features": first["features"], "split": first["split"],
                "role": first["role"], "seeds": [r["seed"] for r in runs], "eer_per_seed": eers,
                "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers)) if len(eers) > 1 else None,
                "ci_first_seed": first["test_ci"], "test_counts": first["test_counts"],
                "train_counts": first["train_counts"], "n_params": first["n_params"]}

    c_summary = group_summary("C_ssl_frozen")
    c_summary["n_params"] = c_model.n_params()
    PHASE2C_LEADERBOARD = [c_summary] + [group_summary_2c(g) for g in PHASE2C_FAMILIES]
    print(f"{'group':22s} {'seeds':>5s} {'EER mean':>9s} {'std':>7s}  CI (first seed)       params")
    for row in PHASE2C_LEADERBOARD:
        std = f"{row['eer_std']:.4f}" if row["eer_std"] is not None else "   -  "
        print(f"{row['group']:22s} {len(row['seeds']):5d} {row['eer_mean']:9.4f} {std:>7s}  "
              f"[{row['ci_first_seed'][0]:.4f}, {row['ci_first_seed'][1]:.4f}]   {row['n_params']}")
    save_progress_2c("leaderboard", PHASE2C_LEADERBOARD)

    # --- Closing paragraph ---------------------------------------------------------------------------
    c_itw_eer = ITW["models"][[k for k, v in ITW["models"].items() if v["model"] == c_name][0]]["eer"] \
        if ITW.get("status") == "ok" and any(v["model"] == c_name for v in ITW["models"].values()) else None
    parts_2c = [f"Model C carried over from Phase 2B: test EER {c_summary['eer_mean']:.4f}"
               + (f" +/- {c_summary['eer_std']:.4f}" if c_summary["eer_std"] is not None else "") + "."]
    improved = []
    for group in PHASE2C_FAMILIES:
        entry = PHASE2C_ITW.get(group)
        if entry and "vs_model_c" in entry and entry["vs_model_c"]["excludes_zero"] and entry["vs_model_c"]["difference"] < 0:
            improved.append(group)
    if c_itw_eer is None:
        parts_2c.append("In-the-Wild comparison unavailable: not run.")
    elif improved:
        parts_2c.append(f"Beat Model C's In-the-Wild EER {c_itw_eer:.4f} outside noise: {', '.join(improved)}.")
    else:
        parts_2c.append(f"None of {', '.join(PHASE2C_FAMILIES)} beat Model C's In-the-Wild EER {c_itw_eer:.4f} "
                        "outside noise -- Model C remains a legitimate final answer.")
    regressed = [g for g, v in PHASE2C_ATTACK_GAPS.items()
                if v["unseen_minus_seen"]["excludes_zero"] and v["unseen_minus_seen"]["difference"] > 0
                and v["unseen_minus_seen"]["difference"] > ATTACK_GAPS.get("C_ssl_frozen", {}).get(
                    "unseen_minus_seen", {}).get("difference", 0)]
    parts_2c.append(f"Attack-disjoint gap regressed vs Model C in: {', '.join(regressed)}." if regressed
                    else "No variant made the attack-disjoint gap significantly worse than Model C's.")
    param_text = "; ".join(f"{g}: {PHASE2C_FAMILIES[g] and MODEL_RESULTS_2C[PHASE2C_FAMILIES[g]['first_seed']]['n_params']} params"
                           for g in PHASE2C_FAMILIES)
    parts_2c.append(f"Added parameters vs Model C ({c_model.n_params()}): {param_text}.")
    closing_2c = " ".join(parts_2c)
    print(closing_2c)
    save_progress_2c("closing_paragraph", closing_2c)
    save_progress_2c("run_hours", round(elapsed_hours(), 2))

    plt.figure(figsize=(7, 6))
    for name, record in {**{c_name: MODEL_RESULTS[c_name]}, **MODEL_RESULTS_2C}.items():
        scores_src = SCORES if name == c_name else SCORES_2C
        if record["seed"] != SEEDS[0]:
            continue
        det = det_curve_points(scores_src[name]["scores"], scores_src[name]["labels"])
        style = "--" if record["split"] == "P2" else "-"
        plt.plot(np.array(det["far"]) * 100, np.array(det["frr"]) * 100, style,
                 label=f"{record['group']} [{record['split']}] EER {record['test_eer']:.3f}")
    plt.plot([0.01, 100], [0.01, 100], "k-", linewidth=0.5, alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("False accept rate (%) -- spoof accepted as bonafide")
    plt.ylabel("False reject rate (%) -- bonafide rejected")
    plt.title("Phase 2C DET curves: C, D, E, F (solid: P1, dashed: attack-disjoint P2)")
    plt.legend(fontsize=6, loc="lower left")
    plt.grid(True, which="both", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "det_curves_phase2c.png", dpi=130)
    plt.show()

    for name, model in MODELS_2C.items():
        torch.save(model.state_dict(), OUTPUT_DIR / f"phase2c_{name}.pt")
    if SCORES_2C:
        np.savez_compressed(OUTPUT_DIR / "phase2c_test_scores.npz",
                            **{f"{n}__{f}": SCORES_2C[n][f] for n in SCORES_2C for f in ("test_idx", "scores", "labels")})
    print(f"\nPhase 2C results: {RESULTS2C_PATH} | elapsed {elapsed_hours():.2f} h")

# %% [markdown]
# ## Delete the feature cache (after both phases are done with it)

# %%
if DELETE_CACHE_AT_END and CACHE_DIR.exists():
    import gc

    size = cache_size_gb(CACHE_DIR)
    cache = ssl_features = None  # release the memory-mapped files first; open memmaps can't be deleted on every OS
    gc.collect()
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    if CACHE_DIR.exists():
        print(f"WARNING: could not fully delete {CACHE_DIR} ({cache_size_gb(CACHE_DIR):.2f} GB left) -- "
              "delete it before committing if output size matters")
    else:
        print(f"deleted feature cache ({size:.2f} GB) so the committed output stays small")
print(f"\nresults: {RESULTS_PATH} (Phase 2B) and {RESULTS2C_PATH} (Phase 2C) | total elapsed {elapsed_hours():.2f} h")
