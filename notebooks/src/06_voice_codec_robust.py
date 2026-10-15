# %% [markdown]
# # Phase 2D -- codec-robust voice head (Model F2)
#
# Implements `docs/PHASE_2D_CODEC_ROBUSTNESS.md` and its addendum. Generated from
# `notebooks/src/06_voice_codec_robust.py` by `scripts/build_notebooks.py` -- edit the source, not this .ipynb.
# The criteria this notebook is judged by are written in `docs/design_decisions.md` ("Phase 2D") and were fixed
# before any run.
#
# **Question.** Compressed audio (AAC, MP3, Opus, phone) pushes genuine speech towards "fake" for the served
# voice model. Does training the same small head on features of codec-processed copies of the training clips
# fix that, without hurting clean audio?
#
# **What this notebook does**
# 1. Rebuilds exactly the speaker-disjoint P1 split Model D used (and checks it against the recorded hash).
# 2. Turns every clip into pooled XLS-R features: the clean clip, 3 augmented copies of each training clip, and
#    every codec condition of the test clips. Real codecs through ffmpeg, one clip at a time, nothing but
#    feature vectors stored, in chunks, so a crash or the 12 h session limit loses little.
# 3. Trains a Model D control (clean features only) and F2 (clean + augmented) with the same recipe, 3 seeds.
# 4. Evaluates both on every condition with paired bootstrap intervals, and writes aggregates only.
#
# **Two switches, in the cell right below.** `RUN_MODE = "PILOT"` measures speed and size on about 500 clips and
# stops. Change it to `"FULL"` for the real run. Read `docs/KAGGLE_RUNBOOK_PHASE_2D.md` first.
#
# Nothing here changes the served model. Heads written by this notebook stay private until the owner has checked
# the dataset terms.

# %% [markdown]
# ## Setup

# %%
# Kaggle: Accelerator = GPU, Internet = on (needed to download the dataset and the encoder).
import os
import shutil
import subprocess
import sys

# True only when tests/smoke_notebook_06.py runs this notebook's code on synthetic data.
SMOKE_TEST = os.environ.get("TRUSTMESH_SMOKE") == "1"

# ---- the switches --------------------------------------------------------------------------------------------
RUN_MODE = os.environ.get("TRUSTMESH_RUN_MODE", "PILOT")   # "PILOT" (measure and stop) or "FULL"
PILOT_CLIPS = 24 if SMOKE_TEST else 500                    # clips the pilot processes
EVAL_PER_CONDITION = 60 if SMOKE_TEST else 4000            # test clips per condition (all bonafide + spoof up to this); None = all
MAX_EXTRACT_HOURS = 9.0                                    # extraction stops cleanly after this and says how to continue
# Directories holding chunk files from an earlier session (e.g. an attached notebook output). Leave empty on a first run.
RESUME_DIRS = [d for d in os.environ.get("TRUSTMESH_RESUME_DIRS", "").split(os.pathsep) if d]
assert RUN_MODE in ("PILOT", "FULL"), "RUN_MODE must be 'PILOT' or 'FULL'"


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


if not SMOKE_TEST:
    for module_name, package in [("datasets", "datasets"), ("soundfile", "soundfile"),
                                 ("transformers", "transformers"), ("fsspec", "fsspec")]:
        try:
            __import__(module_name)
        except ImportError:
            pip_install(package)
if shutil.which("ffmpeg") is None:
    try:
        __import__("imageio_ffmpeg")
    except ImportError:
        pip_install("imageio-ffmpeg")

# %%
import json
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
CACHE_DIR = Path(os.environ.get("TRUSTMESH_CACHE_DIR", str(OUTPUT_DIR / "cache_phase2d")))
DELETE_CACHE_AT_END = True      # only after a complete, successful FULL run
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE_TEST else "cpu")

# Same data and split settings as Phase 2B/2C (notebook 04). Do not change: they define "the same split as Model D".
SPOOF_PER_BONAFIDE = 4
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(16000 * CLIP_SECONDS)
TEST_FRACTION, VAL_FRACTION = 0.3, 0.2
# Model D's recipe (notebook 04, Phase 2C): kept exactly.
SSL_MODEL = "facebook/wav2vec2-xls-r-300m"
SSL_BATCH = 4 if SMOKE_TEST else 16
SSL_EPOCHS = 3 if SMOKE_TEST else 60
SSL_LR, SSL_PATIENCE = 1e-3, 8
BATCH_SIZE = 8 if SMOKE_TEST else 64
N_BOOTSTRAP = 50 if SMOKE_TEST else 1000
CHUNK_SIZE = 16 if SMOKE_TEST else 256
WORKERS = 2 if SMOKE_TEST else min(4, os.cpu_count() or 2)
PILOT_DIR = CACHE_DIR / "pilot"

# What the real Model D run (results_phase2b.json / phase2c_test_scores.npz) recorded for the P1 split.
EXPECTED_P1 = {"train": {"bonafide": 4099, "spoof": 16099}, "val": {"bonafide": 1091, "spoof": 4329},
               "test": {"bonafide": 2165, "spoof": 8992}}
EXPECTED_TEST_IDX_SHA256 = "5e6a50f79e4cafbc370b28f9569bc9e925449402d761e798588f1f4a4c905f2a"

RESULTS_PATH = OUTPUT_DIR / "results_phase2d.json"
RESULTS = {"smoke_test": SMOKE_TEST, "run_mode": RUN_MODE, "not_run": {}}


def save_progress(key, value):
    RESULTS[key] = value
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2, default=float), encoding="utf-8")


print("device:", DEVICE, "| mode:", RUN_MODE, "| SMOKE_TEST:", SMOKE_TEST, "| torch", torch.__version__)

# %% [markdown]
# ## Shared voice-engine code (inlined from `engines/voice/`)

# %% inline engines/voice/data.py

# %% inline engines/voice/preprocess.py

# %% inline engines/voice/evaluate.py

# %% inline engines/voice/model.py

# %% inline engines/voice/features.py

# %% inline engines/voice/ssl.py

# %% inline engines/voice/codec.py

# %% inline engines/voice/robust.py

# %% [markdown]
# ## Step 1 -- the same data and the same split as Model D

# %%
t0 = time.time()
if SMOKE_TEST:
    synthetic = make_synthetic_asvspoof(240, n_attacks=6, bonafide_pad_s=(0.12, 0.3))
    table = None
    meta = {"label": np.array([c["label"] for c in synthetic]),
            "utterance_id": [c["utterance_id"] for c in synthetic],
            "speaker_id": [c["speaker_id"] for c in synthetic]}
    dataset_name = "synthetic (smoke test)"
else:
    print(f"downloading {ASVSPOOF_DATASET} (4.39 GB parquet, cached by `datasets`) ...")
    table = load_asvspoof_table()
    meta = table_metadata(table)
    dataset_name = f"{ASVSPOOF_DATASET} (split '{ASVSPOOF_SPLIT}')"
print(f"metadata for {len(meta['label'])} clips in {time.time() - t0:.0f}s")

selected = select_subset(meta["label"], SPOOF_PER_BONAFIDE, seed=SEED)
labels_np = meta["label"][selected].astype(np.int64)
speakers = [meta["speaker_id"][i] for i in selected]
N = len(selected)
print(f"{N} clips selected ({int((labels_np == BONAFIDE).sum())} bonafide / {int((labels_np == SPOOF).sum())} spoof)")


def get_clip(position):
    """Decode clip number `position` of the selected subset: (float32 waveform, sample rate)."""
    if SMOKE_TEST:
        clip = synthetic[int(selected[position])]
        return clip["waveform"], clip["sample_rate"]
    return _decode_audio(table[int(selected[position])]["audio"])


meta_clips = [{"speaker_id": speakers[i], "label": int(labels_np[i])} for i in range(N)]


def sub_split(indices, fraction, seed):
    subset = [meta_clips[i] for i in indices]
    left, right = speaker_disjoint_split(subset, fraction, seed)
    return [indices[i] for i in left], [indices[i] for i in right]


def counts_of(idx):
    lab = labels_np[np.asarray(idx, dtype=np.int64)]
    return {"bonafide": int((lab == BONAFIDE).sum()), "spoof": int((lab == SPOOF).sum())}


# Identical calls to notebook 04's P1 split.
trainval, test = speaker_disjoint_split(meta_clips, TEST_FRACTION, SEED)
train, val = sub_split(trainval, VAL_FRACTION, SEED + 1)
test_idx_hash = __import__("hashlib").sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest()
split_identity = {
    "counts": {"train": counts_of(train), "val": counts_of(val), "test": counts_of(test)},
    "shared_speakers_train_test": len({speakers[i] for i in train} & {speakers[i] for i in test}),
    "test_idx_sha256": test_idx_hash, "expected_test_idx_sha256": EXPECTED_TEST_IDX_SHA256,
    "matches_model_d_hash": test_idx_hash == EXPECTED_TEST_IDX_SHA256,
    "matches_model_d_counts": {k: counts_of(v) for k, v in (("train", train), ("val", val), ("test", test))} == EXPECTED_P1,
}
print(json.dumps(split_identity, indent=1))
if not SMOKE_TEST:
    # Stop rather than compare against a different split: every number below would be unusable.
    assert split_identity["matches_model_d_counts"], "P1 split counts differ from Model D's recorded split"
    assert split_identity["matches_model_d_hash"], "P1 test indices differ from Model D's recorded split"
assert split_identity["shared_speakers_train_test"] == 0, "speaker overlap between train and test"
save_progress("split_identity", split_identity)

# %%
# Evaluation subset: all bonafide test clips plus spoofs up to EVAL_PER_CONDITION clips (seeded, fixed once).
test_arr = np.asarray(test, dtype=np.int64)
if EVAL_PER_CONDITION is None or len(test_arr) <= EVAL_PER_CONDITION:
    eval_positions = test_arr
else:
    bonafide_test = test_arr[labels_np[test_arr] == BONAFIDE]
    spoof_test = test_arr[labels_np[test_arr] == SPOOF]
    n_spoof = max(1, EVAL_PER_CONDITION - len(bonafide_test))
    rng = np.random.default_rng(SEED + 7)
    eval_positions = np.sort(np.concatenate([bonafide_test, rng.choice(spoof_test, min(n_spoof, len(spoof_test)), replace=False)]))
eval_set = set(int(p) for p in eval_positions)
trainval_set = set(int(p) for p in trainval)

# What this machine can really produce. Substitutions and missing codecs are recorded, not hidden.
FFMPEG = find_ffmpeg()
CONDITIONS = resolve_conditions(FFMPEG)
print("ffmpeg:", FFMPEG)
print("training conditions:", CONDITIONS["training"])
print("held-out conditions:", CONDITIONS["held_out"])
if CONDITIONS["substituted"] or CONDITIONS["unavailable"]:
    print("SUBSTITUTED:", CONDITIONS["substituted"], "| UNAVAILABLE:", CONDITIONS["unavailable"])
if len(CONDITIONS["training"]) < AUGMENTATIONS_PER_CLIP:
    raise RuntimeError(f"only {len(CONDITIONS['training'])} training conditions can be produced here")
if not CONDITIONS["held_out"]:
    raise RuntimeError("no held-out condition can be produced here, so unseen codecs cannot be tested")
EVAL_CONDITIONS = CONDITIONS["training"] + CONDITIONS["held_out"]
AUG_ASSIGN_ARRAY = assign_conditions(N, CONDITIONS["training"], AUGMENTATIONS_PER_CLIP, seed=SEED)
AUG_ASSIGN = {int(p): AUG_ASSIGN_ARRAY[int(p)].tolist() for p in trainval}

# %%
# Encoder: XLS-R 300M, frozen, exactly as Model D's. No silent fallback to a different encoder: a different
# encoder would make "same as Model D" false.
if SMOKE_TEST:
    ssl_encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=DEVICE, fp16=False)
    ssl_name = "tiny random wav2vec2 (smoke test)"
else:
    ssl_encoder = SSLEncoder(SSL_MODEL, device=DEVICE, fp16=True)
    ssl_name = SSL_MODEL
HIDDEN, N_LAYERS = ssl_encoder.hidden_size, ssl_encoder.n_layers
VERSION_BYTES = bytes_per_version(N_LAYERS, HIDDEN)
print(f"encoder {ssl_name}: {N_LAYERS} layers x {HIDDEN} values, {VERSION_BYTES / 1000:.1f} KB per stored version")

PLAN = version_counts(N, len(trainval), len(eval_positions), len(EVAL_CONDITIONS), AUGMENTATIONS_PER_CLIP)
PLAN["bytes_per_version"] = VERSION_BYTES
PLAN["total_cache_gb"] = PLAN["total"] * VERSION_BYTES / 1e9
PLAN["eval_clips_per_condition"] = {"total": int(len(eval_positions)), **counts_of(eval_positions)}
print("planned versions:", json.dumps(PLAN, indent=1))
save_progress("plan", {**PLAN, "conditions": CONDITIONS, "ssl_model": ssl_name, "eval_per_condition_cap": EVAL_PER_CONDITION,
                       "augmentations_per_clip": AUGMENTATIONS_PER_CLIP})

# %% [markdown]
# ## Step 2 -- pilot: speed and size on a small sample, then stop

# %%
if RUN_MODE == "PILOT":
    rng = np.random.default_rng(SEED + 99)
    n_tv = max(1, int(round(PILOT_CLIPS * 0.8)))
    pilot_tv = rng.choice(np.asarray(trainval, dtype=np.int64), min(n_tv, len(trainval)), replace=False)
    pilot_te = rng.choice(eval_positions, min(PILOT_CLIPS - len(pilot_tv), len(eval_positions)), replace=False)
    pilot_positions = sorted(int(p) for p in np.concatenate([pilot_tv, pilot_te]))
    pilot_set = set(pilot_positions)
    shutil.rmtree(PILOT_DIR, ignore_errors=True)
    print(f"PILOT: {len(pilot_positions)} clips ({len(pilot_tv)} training, {len(pilot_te)} test)")
    stats = extract_features(get_clip, pilot_positions, ssl_encoder, CLIP_SAMPLES, PILOT_DIR,
                             trainval=trainval_set & pilot_set, aug_conditions=CONDITIONS["training"],
                             aug_assign=AUG_ASSIGN, eval_positions=eval_set & pilot_set,
                             eval_conditions=EVAL_CONDITIONS, ffmpeg=FFMPEG, seed=SEED, chunk_clips=CHUNK_SIZE,
                             batch_size=SSL_BATCH, workers=WORKERS)
    measured_bytes = sum(p.stat().st_size for p in PILOT_DIR.glob("chunk_*.npz"))
    projection = project_cost(stats["seconds"], stats["versions"], measured_bytes, PLAN["total"])
    disk = shutil.disk_usage(CACHE_DIR if CACHE_DIR.exists() else OUTPUT_DIR)
    pilot = {"clips": len(pilot_positions), "versions_made": stats["versions"],
             "seconds_total": round(stats["seconds"], 1), "seconds_per_clip": round(stats["seconds"] / len(pilot_positions), 2),
             "seconds_per_version": round(projection["seconds_per_version"], 3),
             "seconds_split": {k: round(stats[k], 1) for k in ("seconds_decode", "seconds_conditions", "seconds_encoder")},
             "projected_versions_full": PLAN["total"], "projected_extraction_hours": round(projection["projected_hours"], 2),
             "projected_cache_gb": round(projection["projected_cache_gb"], 2), "free_disk_gb": round(disk.free / 1e9, 1),
             "disk_ok_with_margin": disk.free >= DISK_FACTOR * projection["projected_cache_gb"] * 1e9,
             "note": "straight-line projection from this sample; head training time is not included (minutes)."}
    save_progress("pilot", pilot)
    print(json.dumps(pilot, indent=1))
    print(f"\nPILOT DONE. Full extraction is projected at about {pilot['projected_extraction_hours']:.1f} hours and "
          f"{pilot['projected_cache_gb']:.1f} GB of cache. Free disk: {pilot['free_disk_gb']:.0f} GB.")
    print("If that fits, set RUN_MODE = 'FULL' in the setup cell and run again (Save & Run All).")
    shutil.rmtree(PILOT_DIR, ignore_errors=True)

# %% [markdown]
# ## Step 3 -- full extraction (chunked, resumable)

# %%
FULL = RUN_MODE == "FULL"
if FULL:
    required = PLAN["total"] * VERSION_BYTES
    disk_report = check_disk(CACHE_DIR, required)
    print(f"cache needs about {disk_report['required_gb']:.1f} GB; free {disk_report['free_gb']:.1f} GB (margin x{DISK_FACTOR:g}) -- ok")
    deadline = RUN_START + MAX_EXTRACT_HOURS * 3600
    extraction = extract_features(get_clip, list(range(N)), ssl_encoder, CLIP_SAMPLES, CACHE_DIR,
                                  trainval=trainval_set, aug_conditions=CONDITIONS["training"], aug_assign=AUG_ASSIGN,
                                  eval_positions=eval_set, eval_conditions=EVAL_CONDITIONS, ffmpeg=FFMPEG,
                                  resume_dirs=RESUME_DIRS, seed=SEED, chunk_clips=CHUNK_SIZE, batch_size=SSL_BATCH,
                                  workers=WORKERS, deadline=deadline)
    save_progress("extraction", {k: (round(v, 1) if isinstance(v, float) else v) for k, v in extraction.items()})
    print(json.dumps(RESULTS["extraction"], indent=1))
    if extraction["stopped_early"]:
        print("\nSTOPPED EARLY on the time limit. Nothing is lost: the finished chunks are in the output. Start a new "
              "session with this output attached as input, set TRUSTMESH_RESUME_DIRS to its cache folder, and run again.")
        FULL = False

# %% [markdown]
# ## Step 4 -- train the control (clean only) and F2 (clean + augmented), 3 seeds

# %%
head_results, SEED_SCORES = {}, {}
if FULL:
    loaded = load_chunks([CACHE_DIR, *RESUME_DIRS], kinds=("clean", "aug"))
    clean, aug = loaded["clean"], loaded["aug"]
    assert len(clean["positions"]) == N, f"expected {N} clean versions, found {len(clean['positions'])}"
    pool = np.concatenate([clean["data"], aug["data"]])
    pool_positions = np.concatenate([clean["positions"], aug["positions"]])
    pool_is_aug = np.concatenate([np.zeros(len(clean["positions"]), bool), np.ones(len(aug["positions"]), bool)])
    del loaded, clean, aug
    pool_labels = labels_np[pool_positions]
    train_set, val_set = set(int(p) for p in train), set(int(p) for p in val)
    in_train = np.fromiter((int(p) in train_set for p in pool_positions), bool, len(pool_positions))
    in_val = np.fromiter((int(p) in val_set for p in pool_positions), bool, len(pool_positions))
    control_rows = np.flatnonzero(in_train & ~pool_is_aug)
    f2_rows = np.flatnonzero(in_train)
    val_rows = np.flatnonzero(in_val)                    # clean + augmented validation clips, both models
    val_clean_rows = np.flatnonzero(in_val & ~pool_is_aug)
    print(f"control trains on {len(control_rows)} rows, F2 on {len(f2_rows)}, validation {len(val_rows)} "
          f"({len(val_clean_rows)} clean + {len(val_rows) - len(val_clean_rows)} augmented)")

    def pool_fetch(idx, training):
        return torch.from_numpy(np.asarray(pool[np.asarray(idx)], dtype=np.float32)).to(DEVICE)

    def train_head(name, rows, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SSLHead(N_LAYERS, HIDDEN)
        lab = pool_labels[rows]
        weight = float((lab == SPOOF).sum() / max((lab == BONAFIDE).sum(), 1))
        t_start = time.time()
        fit = train_model(model, pool_fetch, pool_labels, rows, weight, DEVICE, val_idx=val_rows, epochs=SSL_EPOCHS,
                          batch_size=BATCH_SIZE, lr=SSL_LR, patience=SSL_PATIENCE, seed=seed, verbose=False)
        head_results[name] = {"seed": seed, "train_rows": int(len(rows)), "n_params": model.n_params(),
                              "best_val_eer_mixed": fit["best_val_eer"], "best_epoch": fit["best_epoch"],
                              "epochs_run": fit["epochs_run"], "train_seconds": round(time.time() - t_start, 1),
                              "layer_weights": model.layer_weights()}
        torch.save({"state_dict": model.state_dict(), "n_layers": N_LAYERS, "hidden": HIDDEN, "mean_only": True},
                   OUTPUT_DIR / f"phase2d_{name}.pt")
        print(f"{name}: val EER {fit['best_val_eer']:.4f} at epoch {fit['best_epoch']}, {head_results[name]['train_seconds']:.0f}s, "
              f"{model.n_params()} parameters")
        return model

    models = {}
    for seed in SEEDS:
        models[f"control_s{seed}"] = train_head(f"control_s{seed}", control_rows, seed)
        models[f"F2_s{seed}"] = train_head(f"F2_s{seed}", f2_rows, seed)
        save_progress("heads", head_results)

    # Clean validation scores (for the threshold at the clean-validation EER) and every row's scores on the clean test.
    def score_rows(model, rows):
        return score_model(model, pool_fetch, rows)

    val_clean_labels = pool_labels[val_clean_rows]
    clean_test_rows = np.flatnonzero(~pool_is_aug & np.isin(pool_positions, test_arr))
    clean_test_rows = clean_test_rows[np.argsort(pool_positions[clean_test_rows])]
    clean_test_pos = pool_positions[clean_test_rows]
    clean_test_labels = pool_labels[clean_test_rows]
    for key, model in models.items():
        SEED_SCORES[key] = {"val_clean": score_rows(model, val_clean_rows), "test_clean": score_rows(model, clean_test_rows)}
    del pool, models

# %% [markdown]
# ## Step 5 -- evaluate every condition

# %%
if FULL:
    ev = load_chunks([CACHE_DIR, *RESUME_DIRS], kinds=("eval",))["eval"]
    assert ev["data"] is not None, "no evaluation features found"
    # Heads are small; reload them from disk so the evaluation does not depend on the training cell's memory.
    heads = {}
    for key in SEED_SCORES:
        blob = torch.load(OUTPUT_DIR / f"phase2d_{key}.pt", map_location="cpu")
        h = SSLHead(blob["n_layers"], blob["hidden"])
        h.load_state_dict(blob["state_dict"])
        heads[key] = h.to(DEVICE).eval()

    def eval_scores(key, rows):
        out = []
        with torch.no_grad():
            for s in range(0, len(rows), 256):
                x = torch.from_numpy(np.asarray(ev["data"][rows[s:s + 256]], dtype=np.float32)).to(DEVICE)
                out.append(heads[key](x).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    def mean_over_seeds(model, per_seed):
        return np.mean([per_seed[f"{model}_s{s}"] for s in SEEDS], axis=0)

    def per_seed_eers(model, per_seed, labels):
        return {str(s): compute_eer(per_seed[f"{model}_s{s}"], labels)["eer"] for s in SEEDS}

    # Threshold at the EER on the clean validation clips, per model (score = bonafide logit; lower = more fake).
    thresholds = {m: compute_eer(mean_over_seeds(m, {k: v["val_clean"] for k, v in SEED_SCORES.items()}),
                                 val_clean_labels)["threshold"] for m in ("control", "F2")}

    def evaluate_block(label, per_seed, labels, groups=None, clustered=False):
        d_scores, f_scores = mean_over_seeds("control", per_seed), mean_over_seeds("F2", per_seed)
        block = {"n": {"bonafide": int((labels == BONAFIDE).sum()), "spoof": int((labels == SPOOF).sum()), "rows": int(len(labels))}}
        for name, scores in (("control", d_scores), ("F2", f_scores)):
            ci = bootstrap_eer_ci(scores, labels, n_resamples=N_BOOTSTRAP, seed=SEED)
            block[name] = {"eer": ci["eer"], "ci": [ci["ci_low"], ci["ci_high"]],
                           "per_seed_eer": per_seed_eers(name, per_seed, labels),
                           "genuine_flagged_at_clean_val_eer": share_flagged(scores, labels, thresholds[name])}
        diff = (clustered_paired_bootstrap(f_scores, d_scores, labels, groups, n_resamples=N_BOOTSTRAP, seed=SEED)
                if clustered else
                bootstrap_eer_difference(f_scores, labels, d_scores, labels, n_resamples=N_BOOTSTRAP, seed=SEED, paired=True))
        block["difference_F2_minus_control"] = diff
        block["status"] = criterion_status(diff)
        print(f"{label:22s} control EER {block['control']['eer']:.4f} | F2 EER {block['F2']['eer']:.4f} | "
              f"diff {diff['difference']:+.4f} [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}] | "
              f"flagged genuine control {block['control']['genuine_flagged_at_clean_val_eer']:.2f} "
              f"F2 {block['F2']['genuine_flagged_at_clean_val_eer']:.2f}")
        return block

    evaluation = {"thresholds_clean_val_eer": thresholds, "conditions": {}}
    evaluation["clean_full_test"] = evaluate_block("clean (full test)", {k: v["test_clean"] for k, v in SEED_SCORES.items()},
                                                   clean_test_labels)
    pooled = {"scores": {k: [] for k in SEED_SCORES}, "labels": [], "groups": []}
    clean_row_of = {int(p): i for i, p in enumerate(clean_test_pos)}
    for ci_idx, name in enumerate(EVAL_CONDITIONS):
        rows = np.flatnonzero(ev["cond"] == ci_idx)
        rows = rows[np.argsort(ev["positions"][rows])]
        pos = ev["positions"][rows]
        labels = labels_np[pos]
        per_seed = {k: eval_scores(k, rows) for k in SEED_SCORES}
        block = evaluate_block(name, per_seed, labels)
        block["role"] = "training_menu" if name in CONDITIONS["training"] else "held_out_never_trained"
        # The same clips, clean: the baseline each condition is read against.
        take = np.array([clean_row_of[int(p)] for p in pos])
        block["same_clips_clean"] = evaluate_block(f"  {name} (clean, same clips)",
                                                   {k: v["test_clean"][take] for k, v in SEED_SCORES.items()}, labels)
        evaluation["conditions"][name] = block
        if name in CONDITIONS["training"]:
            for k in SEED_SCORES:
                pooled["scores"][k].append(per_seed[k])
            pooled["labels"].append(labels)
            pooled["groups"].append(pos)
    pooled_scores = {k: np.concatenate(v) for k, v in pooled["scores"].items()}
    evaluation["pooled_training_menu_conditions"] = evaluate_block(
        "pooled codec set", pooled_scores, np.concatenate(pooled["labels"]), groups=np.concatenate(pooled["groups"]),
        clustered=True)
    save_progress("evaluation", evaluation)

    # --- the pre-registered criteria, read mechanically --------------------------------------------------------
    c1 = evaluation["clean_full_test"]["status"]["not_worse"]
    per_cond_ok = {n: b["status"]["not_worse"] for n, b in evaluation["conditions"].items() if b["role"] == "training_menu"}
    c2a = all(per_cond_ok.values())
    c2b = evaluation["pooled_training_menu_conditions"]["status"]["better"]
    verdict = {"criterion_1_clean_not_worse": c1, "criterion_2a_no_condition_worse": c2a,
               "criterion_2a_by_condition": per_cond_ok, "criterion_2b_pooled_better": c2b,
               "all_pass": bool(c1 and c2a and c2b),
               "held_out_reported_not_judged": {n: b["difference_F2_minus_control"] for n, b in evaluation["conditions"].items()
                                                if b["role"] != "training_menu"}}
    save_progress("criteria", verdict)
    print("\nPRE-REGISTERED CRITERIA:", json.dumps({k: v for k, v in verdict.items() if k != "held_out_reported_not_judged"}, indent=1))
    print("Held-out conditions are reported, not judged. A failed criterion is a negative result: do not retune and re-run.")

# %% [markdown]
# ## Wrap-up

# %%
if FULL:
    RESULTS["closing"] = ("Aggregates only; no clip identifiers or per-clip scores are stored. Head weights "
                          f"(phase2d_*.pt, {len(SEED_SCORES)} files) are private until the dataset terms are checked.")
    elapsed = (time.time() - RUN_START) / 3600
    save_progress("elapsed_hours", round(elapsed, 2))
    if DELETE_CACHE_AT_END:
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print("feature cache deleted (the run is complete); results and heads are in", OUTPUT_DIR)
    print(f"done in {elapsed:.1f} h. Download results_phase2d.json and the phase2d_*.pt files.")
elif RUN_MODE == "FULL":
    print("Extraction is not finished; see the message above. Nothing was trained.")
else:
    print("Pilot only. Nothing was trained.")
