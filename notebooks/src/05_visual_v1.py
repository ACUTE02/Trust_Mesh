# %% [markdown]
# # Phase 3 -- Visual Engine, Version 1: frozen CLIP + a small head
#
# Implements `docs/PHASE_3_VISUAL_ENGINE.md` with the split design of
# `docs/HANDBOOK_ADDENDUM_01_ROTATING_HOLDOUT.md`. Generated from `notebooks/src/05_visual_v1.py` by
# `scripts/build_notebooks.py` -- edit the source, not this .ipynb.
#
# **This notebook runs in stages, and stops after the stage named in `STAGE`** (first code cell):
#
# | Stage | What it does | Needs |
# |---|---|---|
# | `task0` | Pre-flight: what is really attached, real OpenFake counts read from metadata only, a timed streamed sample, CLIP check, disk budget. **Default. Nothing else runs.** | -- |
# | `scan` | Tasks 1-2, pass 1: one metadata row per image (no encoder) | the `PLAN` decided from Task 0 |
# | `split` | Tasks 2-4: dedupe, group-aware splits with rotating held-out families, leakage checks, shortcut baselines | `scan` output |
# | `extract` | Task 5: CLIP embeddings, resumable chunks | `split` output, shortcut gate |
# | `heads` | Tasks 6-9: heads, calibration, evaluation suites | `extract` output |
#
# **Kaggle settings:** GPU, Internet on, the `HF_TOKEN` secret attached (Add-ons -> Secrets), the
# ArtiFact and FFHQ-thumbnails datasets added as inputs, and the notebook kept **private** (later
# stages write per-image metadata and embeddings to its output, which the datasets' terms do not
# allow in a public place). Go straight to **Save Version -> Save & Run All (Commit)**.
#
# Every number printed here is from this run. Anything not run says "not yet run".

# %%
import os
import subprocess
import sys

# True only when tests/smoke_notebook_05.py runs this notebook's code on synthetic data.
SMOKE_TEST = os.environ.get("TRUSTMESH_SMOKE") == "1"

# ---- THE STAGE SWITCH -------------------------------------------------------------------------------
# The last stage this run executes: task0 -> scan -> split -> extract -> heads.
# It stays "task0" until the Task 0 results have been read and the PLAN below has been decided.
STAGE = os.environ.get("TRUSTMESH_VISUAL_STAGE") or "task0"


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


if not SMOKE_TEST:
    for module_name, package in [("pyarrow", "pyarrow"), ("huggingface_hub", "huggingface_hub"),
                                 ("transformers", "transformers"), ("fsspec", "fsspec"), ("xgboost", "xgboost")]:
        try:
            __import__(module_name)
        except ImportError:
            pip_install(package)

# %%
import glob
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RUN_START = time.time()
SEED = 42
SEEDS = [42, 43, 44]
torch.manual_seed(SEED)
np.random.seed(SEED)

STAGES = ("task0", "scan", "split", "extract", "heads")
if STAGE not in STAGES:
    raise ValueError(f"STAGE must be one of {STAGES}, got {STAGE!r}")

OUTPUT_DIR = Path(os.environ.get("TRUSTMESH_OUTPUT_DIR", "./artifacts/visual"))
# Everything per-image (metadata rows, embeddings, predictions) and the trained weights go here.
# This folder is in .gitignore and must never reach a public repository or a public Kaggle dataset.
PRIVATE_DIR = OUTPUT_DIR / "private"
for folder in (OUTPUT_DIR, PRIVATE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
# Output of an earlier session of this notebook, attached as an input, so finished chunks are reused.
RESUME_DIRS = [d for d in os.environ.get("TRUSTMESH_RESUME_DIRS", "").split(os.pathsep) if d] or \
    sorted(glob.glob("/kaggle/input/*/artifacts/visual/private"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE_TEST else "cpu")

N_AUG_VIEWS = 2                      # augmented views cached per training image (handbook: 2-3)
N_BOOTSTRAP = 30 if SMOKE_TEST else 1000
EMBED_BATCH = 8 if SMOKE_TEST else 64
THREADS = 2 if SMOKE_TEST else 4     # image decode threads (Kaggle gives 4 CPU cores)
SCAN_THREADS = 2 if SMOKE_TEST else 32
TARGET_RECALL, BAND_RECALL, BAND_FPR = 0.95, 0.99, 0.01   # operating point settings (KYC: favour recall)
TIME_BUDGET_HOURS = 9.0              # Kaggle stops a session at 12 h; stop cleanly well before
SHOW_DUPLICATE_PAIRS = SMOKE_TEST    # draws real images into the notebook output: private notebooks only
ACCEPT_RESIDUAL_SHORTCUT = SMOKE_TEST  # set True only after reading the shortcut-baseline numbers
# Heavier pixel baselines (Task 9): off by default, run only after Tasks 1-8 have real results.
RUN_SMALL_CNN = SMOKE_TEST or os.environ.get("TRUSTMESH_RUN_SMALL_CNN") == "1"
RUN_RESNET50 = SMOKE_TEST or os.environ.get("TRUSTMESH_RUN_RESNET50") == "1"

# ---- THE PLAN: decided from the Task 0 results, not before ------------------------------------------
# Every None below is a decision that depends on real counts or timings. A later stage refuses to
# run while a value it needs is None, so nothing downstream can start on a guess.
PLAN = {
    "openfake_splits": ("train", "validation"),  # which OpenFake `core` splits are read
    "openfake_shards": None,          # how many shards to read, in a fixed seeded order
    "keep_fraction": None,            # {generator: share of its rows kept}, to thin the biggest generators
    "artifact_per_folder": None,      # images taken per ArtiFact generator folder
    "ffhq_images": None,              # FFHQ thumbnails taken (0 drops the source)
    "partition_fractions": {"train": 0.6, "val": 0.1, "known_test": 0.1, "holdout": 0.2},
    "caps": None,                     # max images per generator per partition
    "phash_tolerance": None,          # dedupe distance, chosen from the real distance histogram
    "min_unseen_fakes": None,         # smallest unseen test a family needs to be held out
    "old_families": None,             # families that count as "old" for the old-only ablation
}
if SMOKE_TEST:
    PLAN.update(openfake_shards=99, keep_fraction={}, artifact_per_folder=10, ffhq_images=12,
                caps={"fake": {"train": 40, "val": 20, "known_test": 20, "holdout": 30},
                      "real": {"train": 200, "val": 60, "known_test": 60, "holdout": 80}},
                phash_tolerance=4, min_unseen_fakes=3, old_families=["sd-1x-2x", "stylegan", "ddpm"])

RESULTS_PATH = OUTPUT_DIR / "results_v1.json"
PREFLIGHT_PATH = OUTPUT_DIR / "preflight_task0.json"
RESULTS = {"smoke_test": SMOKE_TEST, "stage": STAGE, "not_run": {}}
PREFLIGHT = {"smoke_test": SMOKE_TEST, "failed_steps": {}}
SECRETS = []  # filled with the token so it can be scrubbed from anything written or printed


def stage_on(name):
    return STAGES.index(name) <= STAGES.index(STAGE)


def require(*keys):
    missing = [k for k in keys if PLAN.get(k) is None]
    if missing:
        raise RuntimeError(f"PLAN{missing} not decided yet. These are set from the Task 0 results "
                           "(see docs/design_decisions.md, Phase 3) -- run STAGE='task0' first and report it back.")


def save_results(key, value):
    RESULTS[key] = value
    save_json(RESULTS_PATH, RESULTS, SECRETS)


def save_preflight(key, value):
    PREFLIGHT[key] = value
    save_json(PREFLIGHT_PATH, PREFLIGHT, SECRETS)


def elapsed_hours():
    return (time.time() - RUN_START) / 3600


def within_budget(step):
    if elapsed_hours() < TIME_BUDGET_HOURS:
        return True
    RESULTS["not_run"][step] = f"time budget: {elapsed_hours():.1f} h elapsed of {TIME_BUDGET_HOURS} h"
    save_results("not_run", RESULTS["not_run"])
    print(f"STOPPED {step}: {RESULTS['not_run'][step]}")
    return False


def preflight_step(name, fn):
    """Run one pre-flight check. A failure is recorded and printed, and the next check still runs."""
    print(f"\n=== Task 0: {name} ===")
    started = time.time()
    try:
        value = fn()
        save_preflight(name, value)
        print(f"[{name}] done in {time.time() - started:.0f}s")
        return value
    except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
        PREFLIGHT["failed_steps"][name] = scrub_secrets(f"{type(exc).__name__}: {exc}", SECRETS)[:600]
        save_preflight("failed_steps", PREFLIGHT["failed_steps"])
        print(f"[{name}] FAILED: {PREFLIGHT['failed_steps'][name]}")
        print(scrub_secrets(traceback.format_exc(), SECRETS)[-1500:])
        return None


def disk_report(paths=("/kaggle/working", "/kaggle/temp", "/tmp", ".")):
    out = {}
    for p in paths:
        if os.path.isdir(p):
            usage = shutil.disk_usage(p)
            out[p] = {"total_gb": round(usage.total / 1e9, 1), "used_gb": round(usage.used / 1e9, 1),
                      "free_gb": round(usage.free / 1e9, 1)}
    return out


print("device:", DEVICE, "| SMOKE_TEST:", SMOKE_TEST, "| STAGE:", STAGE, "| torch", torch.__version__)

# %% [markdown]
# ## Shared code (inlined from `engines/`)
#
# The Voice Engine's `evaluate.py` comes first: the Visual Engine reuses its stratified bootstrap and
# EER (the code fixed for issue #16) instead of writing them again.

# %% inline engines/voice/evaluate.py

# %% inline engines/visual/preprocess.py

# %% inline engines/visual/data.py

# %% inline engines/visual/metadata.py

# %% inline engines/visual/scan.py

# %% inline engines/visual/splits.py

# %% inline engines/visual/evaluate.py

# %% inline engines/visual/calibrate.py

# %% inline engines/visual/baselines.py

# %% inline engines/visual/embed.py

# %% inline engines/visual/heads.py

# %% inline engines/visual/experiment.py

# %% inline engines/visual/pixel_baselines.py

# %% [markdown]
# ## Task 0 -- pre-flight: report, do not assume
#
# Each check below writes what it found to `preflight_task0.json` (aggregate numbers only -- no
# per-image rows). A check that fails is recorded with its error and the rest still run.

# %%
# --- 0a. Environment, disk, token (presence only) -----------------------------------------------------
def check_environment():
    import PIL
    import pyarrow
    import transformers
    import huggingface_hub

    info = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
            "pyarrow": pyarrow.__version__, "huggingface_hub": huggingface_hub.__version__, "pillow": PIL.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "cpu_count": os.cpu_count(),
            "cuda": torch.cuda.is_available(), "disk": disk_report(), "kaggle_inputs": find_kaggle_inputs(),
            "resume_dirs": RESUME_DIRS}
    if torch.cuda.is_available():
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
                        for i in range(torch.cuda.device_count())]
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info["ram_gb"] = None
    print(json.dumps(info, indent=1))
    return info


preflight_step("environment", check_environment)

_token = read_hf_token()
if _token:
    SECRETS.append(_token)
save_preflight("hf_token_present", bool(_token))   # only the fact that it exists -- never its value
print("HF_TOKEN secret present:", bool(_token))

# %%
# --- 0b. OpenFake: what the repository really holds ---------------------------------------------------
if SMOKE_TEST:
    import fsspec

    OPENFAKE_FS = fsspec.filesystem("file")
    OPENFAKE_ROOT = os.environ["TRUSTMESH_OPENFAKE_DIR"]
else:
    from huggingface_hub import HfFileSystem

    OPENFAKE_FS = HfFileSystem(token=_token)
    OPENFAKE_ROOT = f"datasets/{OPENFAKE_REPO}"
OPENFAKE_SHARDS, OPENFAKE_READER = [], None


def check_openfake_files():
    global OPENFAKE_SHARDS, OPENFAKE_READER
    OPENFAKE_SHARDS = list_parquet_shards(OPENFAKE_FS, OPENFAKE_ROOT)
    by_split = {}
    for s in OPENFAKE_SHARDS:
        entry = by_split.setdefault(f"{s['config']}/{s['split']}", {"shards": 0, "gb": 0.0})
        entry["shards"] += 1
        entry["gb"] += s["size"] / 1e9
    out = {"root": OPENFAKE_ROOT, "parquet_shards": len(OPENFAKE_SHARDS),
           "total_gb": round(sum(s["size"] for s in OPENFAKE_SHARDS) / 1e9, 1),
           "by_split": {k: {"shards": v["shards"], "gb": round(v["gb"], 1)} for k, v in sorted(by_split.items())}}
    try:
        card = OPENFAKE_FS.cat_file(f"{OPENFAKE_ROOT}/README.md").decode("utf-8", "replace")
        out["card_license"] = next((line.split(":", 1)[1].strip() for line in card.splitlines()
                                    if line.startswith("license:")), None)
    except Exception as exc:  # noqa: BLE001
        out["card_license"] = f"could not read the dataset card: {type(exc).__name__}"
    out["paper_license_per_handbook"] = "CC BY-NC-SA 4.0 (the handbook's reading of the paper; not re-verified here)"
    try:   # the repository ships a note on what its v2 repair changed; record it rather than guess
        out["v2_repair_summary"] = json.loads(OPENFAKE_FS.cat_file(f"{OPENFAKE_ROOT}/corrections/v2_repair_summary.json"))
    except Exception as exc:  # noqa: BLE001
        out["v2_repair_summary"] = f"not read: {type(exc).__name__}"
    out["other_files"] = sorted(p.rsplit("/", 1)[-1] for p in OPENFAKE_FS.find(OPENFAKE_ROOT) if not p.endswith(".parquet"))[:30]

    # Two ways to read byte ranges; time a footer-sized read with each and keep the faster.
    readers = {"fsspec": FsRangeReader(OPENFAKE_FS)}
    if not SMOKE_TEST:
        readers["http"] = HttpRangeReader(OPENFAKE_REPO, _token)
    timings = {}
    for name, reader in readers.items():
        try:
            t0 = time.time()
            for s in OPENFAKE_SHARDS[:3]:
                reader.read(s["path"], max(0, s["size"] - 65536), s["size"])
            timings[name] = round((time.time() - t0) / max(1, len(OPENFAKE_SHARDS[:3])), 3)
        except Exception as exc:  # noqa: BLE001
            timings[name] = f"failed: {type(exc).__name__}: {scrub_secrets(str(exc), SECRETS)[:200]}"
    usable = {k: v for k, v in timings.items() if isinstance(v, float)}
    # Prefer the plain HTTP reader when it works: it asks the Hub once per file and then reads from storage directly,
    # which keeps tens of thousands of small range reads clear of the Hub's request limits.
    chosen = "http" if "http" in usable else min(usable, key=usable.get)
    OPENFAKE_READER = readers[chosen]
    out["range_read_seconds_per_request"] = timings
    out["range_reader_used"] = chosen
    print(json.dumps(out, indent=1))
    return out


preflight_step("openfake_files", check_openfake_files)

# %%
# --- 0c. OpenFake: real counts from the metadata columns only (no image is downloaded) ----------------
OPENFAKE_SCAN, OPENFAKE_COUNTS = None, None


def check_openfake_counts():
    global OPENFAKE_SCAN, OPENFAKE_COUNTS
    OPENFAKE_SCAN = scan_all_shards(OPENFAKE_READER, OPENFAKE_SHARDS, threads=SCAN_THREADS,
                                    max_seconds=None if SMOKE_TEST else 45 * 60, progress_every=0 if SMOKE_TEST else 25)
    # Parquet files without an image column (the repository's `corrections/` tables) are listed, not counted.
    other = [s for s in OPENFAKE_SCAN["shards"] if "image" not in s["columns"]]
    OPENFAKE_SCAN["shards"] = [s for s in OPENFAKE_SCAN["shards"] if "image" in s["columns"]]
    done = OPENFAKE_SCAN["shards"]
    rows = []
    for shard in done:
        for (label, model, kind, date), n in shard["counts"].items():
            rows.append({"config": shard["config"], "split": shard["split"], "shard": shard["path"].rsplit("/", 1)[-1],
                         "label": label, "model": model, "type": kind, "release_date": date, "rows": n})
    OPENFAKE_COUNTS = pd.DataFrame(rows)
    # Aggregate tables (counts per shard and per model -- not per-image rows).
    OPENFAKE_COUNTS.to_csv(OUTPUT_DIR / "openfake_shard_counts.csv", index=False)
    per_model = (OPENFAKE_COUNTS.groupby(["config", "split", "label", "model", "type", "release_date"], dropna=False)["rows"]
                 .sum().reset_index().sort_values(["config", "split", "label", "rows"], ascending=[True, True, True, False]))
    per_model.to_csv(OUTPUT_DIR / "openfake_model_counts.csv", index=False)

    group_rows = np.concatenate([s["row_group_rows"] for s in done]) if done else np.array([0])
    out = {
        "shards_scanned": len(done), "shards_failed": OPENFAKE_SCAN["failed"][:10],
        "shards_skipped_for_time": len(OPENFAKE_SCAN["skipped_for_time"]), "scan_seconds": round(OPENFAKE_SCAN["seconds"], 1),
        "metadata_bytes_read_mb": round(OPENFAKE_READER.bytes_read / 1e6, 1), "range_requests": OPENFAKE_READER.requests,
        "columns": done[0]["columns"] if done else None, "total_rows": int(sum(s["rows"] for s in done)),
        "rows_by_split": {f"{c}/{s}": int(n) for (c, s), n in OPENFAKE_COUNTS.groupby(["config", "split"])["rows"].sum().items()},
        "rows_by_split_and_label": {f"{c}/{s}/{l}": int(n) for (c, s, l), n in
                                    OPENFAKE_COUNTS.groupby(["config", "split", "label"])["rows"].sum().items()},
        "label_values": {str(k): int(v) for k, v in OPENFAKE_COUNTS.groupby("label")["rows"].sum().items()},
        "type_values": {str(k): int(v) for k, v in OPENFAKE_COUNTS.groupby("type", dropna=False)["rows"].sum().items()},
        "distinct_model_values": int(OPENFAKE_COUNTS["model"].nunique()),
        "distinct_fake_generators": int(OPENFAKE_COUNTS[OPENFAKE_COUNTS["label"] == "fake"]["model"].nunique()),
        "real_sources": {str(k): int(v) for k, v in
                         OPENFAKE_COUNTS[OPENFAKE_COUNTS["label"] == "real"].groupby("model")["rows"].sum().items()},
        "distinct_release_dates": int(OPENFAKE_COUNTS["release_date"].nunique()),
        "row_groups_per_shard_median": float(np.median([s["row_groups"] for s in done])) if done else None,
        "rows_per_row_group_quantiles": {q: float(np.quantile(group_rows, float(q))) for q in ("0.05", "0.5", "0.95")},
        "image_mb_per_row_by_split": {
            f"{c}/{sp}": round(sum(s["image_bytes"] for s in done if (s["config"], s["split"]) == (c, sp)) / 1e6 /
                               max(1, sum(s["rows"] for s in done if (s["config"], s["split"]) == (c, sp))), 3)
            for c, sp in sorted({(s["config"], s["split"]) for s in done})},
        "sparse_read_misses": int(sum(s["sparse_misses"] for s in done)),
        "hub_requests": getattr(OPENFAKE_READER, "hub_requests", None),
        "parquet_files_without_images": [{"file": "/".join(s["path"].rsplit("/", 2)[-2:]), "rows": s["rows"],
                                          "columns": s["columns"]} for s in other],
        "prompt_groups_all": prompt_group_stats(OPENFAKE_SCAN),
        "prompt_groups_core_train_validation": prompt_group_stats(
            {"shards": [s for s in done if s["config"] == "core" and s["split"] in ("train", "validation")]}),
    }
    # Fake generators with the proposed family and which rule produced it ("fallback" = needs a human look).
    fakes = OPENFAKE_COUNTS[OPENFAKE_COUNTS["label"] == "fake"]
    generators = []
    for model, part in fakes.groupby("model"):
        family, rule = family_of(model)
        generators.append({"model": model, "rows": int(part["rows"].sum()),
                           "rows_by_split": {f"{c}/{s}": int(n) for (c, s), n in part.groupby(["config", "split"])["rows"].sum().items()},
                           "types": sorted(set(part["type"].astype(str))), "release_dates": sorted(set(part["release_date"].astype(str))),
                           "proposed_family": family, "family_rule": rule})
    out["fake_generators"] = sorted(generators, key=lambda g: -g["rows"])
    families = Counter()
    for g in generators:
        families[g["proposed_family"]] += g["rows"]
    out["proposed_families"] = dict(families.most_common())
    out["generators_on_fallback_rule"] = [g["model"] for g in generators if g["family_rule"] == "fallback"]

    print(json.dumps({k: v for k, v in out.items() if k not in ("fake_generators",)}, indent=1))
    print(f"\n{'fake generator':38s} {'rows':>9s}  {'type':16s} {'released':10s} proposed family (rule)")
    for g in out["fake_generators"]:
        print(f"{g['model'][:38]:38s} {g['rows']:9d}  {','.join(g['types'])[:16]:16s} {','.join(g['release_dates'])[:10]:10s} "
              f"{g['proposed_family']} ({g['family_rule']})")
    return out


preflight_step("openfake_counts", check_openfake_counts)

# %%
# --- 0d. OpenFake: a timed streamed sample (images downloaded, decoded, then dropped) -----------------
OPENFAKE_SAMPLE = None


def core_train_shards():
    return [s for s in OPENFAKE_SHARDS if s["config"] == "core" and s["split"] == "train"]


def seeded_shard_order(shards):
    """A fixed order for reading shards: the same in every session, unrelated to file names."""
    return sorted(shards, key=lambda s: stable_fraction(s["path"], SEED))


def check_streaming():
    global OPENFAKE_SAMPLE
    pool = core_train_shards()
    picks = seeded_shard_order(pool)[:2]
    OPENFAKE_READER.requests = OPENFAKE_READER.bytes_read = 0
    OPENFAKE_SAMPLE = time_streamed_sample(OPENFAKE_READER, picks, max_images=40 if SMOKE_TEST else 2400,
                                           max_seconds=60 if SMOKE_TEST else 900, decode_threads=THREADS)
    out = {k: v for k, v in OPENFAKE_SAMPLE.items() if k != "row_groups"}
    out["shards_sampled"] = [p["path"].rsplit("/", 1)[-1] for p in picks]
    out["row_groups_read"] = len(OPENFAKE_SAMPLE["row_groups"])
    out["seek_row_groups"] = [g for g in OPENFAKE_SAMPLE["row_groups"] if g["entered_by_seek"]]
    out["what_is_timed"] = ("download = one HTTP range per row group's image column; decode = decode + upright RGB + "
                            "perceptual hash + the shared resize/JPEG pipeline, on THREADS threads")
    print(json.dumps(out, indent=1))
    return out


preflight_step("streaming_sample", check_streaming)

# %%
# --- 0e. Can per-generator quotas be filled, and how long would the planned sample take? --------------
def check_quota_feasibility():
    pool = seeded_shard_order(core_train_shards())
    scanned = {s["path"]: s for s in (OPENFAKE_SCAN or {"shards": []})["shards"]}
    ordered = [scanned[s["path"]] for s in pool if s["path"] in scanned]
    if not ordered or OPENFAKE_SAMPLE is None:
        raise RuntimeError("needs the metadata scan and the streamed sample above")
    per_shard_family, per_shard_model = [], []
    for shard in ordered:
        models = Counter(m for l, m in zip(shard["label"], shard["model"]) if l == "fake")
        per_shard_model.append(models)
        fam = Counter()
        for m, n in models.items():
            fam[family_of(m)[0]] += n
        per_shard_family.append(fam)
    total_family = sum(per_shard_family, Counter())
    mean_shard_gb = float(np.mean([s["size"] for s in ordered])) / 1e9
    mean_rows = float(np.mean([s["rows"] for s in ordered]))
    mbps = OPENFAKE_SAMPLE["mb_per_second"] or float("nan")
    decode_ips = OPENFAKE_SAMPLE["decode_images_per_second"] or float("nan")
    out = {"shards_in_pool": len(ordered), "mean_shard_gb": round(mean_shard_gb, 2), "mean_rows_per_shard": round(mean_rows, 1),
           "shards_are_shuffled_note": "whole shards are read in a fixed seeded order; a quota for a generator is filled "
                                       "by reading shards until enough of its rows have gone by",
           "families_total_fakes_in_pool": dict(total_family.most_common()), "quotas": {}}
    for quota in (250, 500, 1000, 2000, 4000):
        need = shards_to_fill_quota(per_shard_family, quota)
        reachable = {f: n for f, n in need.items() if n is not None}
        out["quotas"][str(quota)] = {
            "families_that_reach_it": len(reachable), "families_that_never_do": sorted(f for f, n in need.items() if n is None),
            "shards_needed_per_family": reachable,
            "shards_for_the_rarest_reachable_family": max(reachable.values()) if reachable else None,
        }
    # Hours for reading K whole shards: one pass downloads every row group; decode time assumes every row is decoded
    # (an upper bound -- thinning the biggest generators lowers it). Pass 2 (CLIP) downloads the same bytes again.
    plans = []
    for k in sorted({10, 20, 40, 80, 160, 320, len(ordered)}):
        if k > len(ordered):
            continue
        gb, rows = k * mean_shard_gb, k * mean_rows
        download_h, decode_h = gb * 1000 / mbps / 3600, rows / decode_ips / 3600
        timed = bool(np.isfinite(download_h + decode_h))   # False only if the sample above measured no time at all
        families_min = min((sum((c for c in per_shard_family[:k]), Counter())[f] for f in total_family), default=0)
        plans.append({"shards": k, "gb_per_pass": round(gb, 0), "rows": int(rows),
                      "download_hours_per_pass": round(download_h, 2) if timed else None,
                      "decode_hours_if_every_row_decoded": round(decode_h, 2) if timed else None,
                      "hours_per_pass_sequential": round(download_h + decode_h, 2) if timed else None,
                      "fits_one_session_per_pass": bool(download_h + decode_h < TIME_BUDGET_HOURS) if timed else None,
                      "sessions_needed_for_both_passes": int(np.ceil(2 * (download_h + decode_h) / TIME_BUDGET_HOURS)) if timed else None,
                      "fakes_of_the_rarest_family": int(families_min)})
    out["plans"] = plans
    out["pass_rule"] = ("the planned sample must fit in sessions that resume from chunked writes, none longer than "
                        f"{TIME_BUDGET_HOURS} h (Kaggle's limit is 12 h); this table is the evidence, the choice is made after reading it")
    print(json.dumps({k: v for k, v in out.items() if k != "quotas"}, indent=1))
    for quota, q in out["quotas"].items():
        print(f"quota {quota:>5s} fakes per family: {q['families_that_reach_it']} families reach it; rarest of those needs "
              f"{q['shards_for_the_rarest_reachable_family']} shards; never reached by: {q['families_that_never_do']}")
    return out


preflight_step("quota_feasibility", check_quota_feasibility)

# %%
# --- 0f. ArtiFact and FFHQ: what is attached, and what it really contains -----------------------------
# Found by folder name under /kaggle/input. If a name does not match (the environment check above prints the
# attached folders), write the folder's path into the override instead.
ARTIFACT_DIR_OVERRIDE = None
FFHQ_DIR_OVERRIDE = None
_inputs = find_kaggle_inputs()
ARTIFACT_DIR = ARTIFACT_DIR_OVERRIDE or os.environ.get("TRUSTMESH_ARTIFACT_DIR") or match_input(_inputs, ["artifact"])
FFHQ_DIR = FFHQ_DIR_OVERRIDE or os.environ.get("TRUSTMESH_FFHQ_DIR") or match_input(_inputs, ["ffhq"], ARTIFACT_DIR) \
    or match_input(_inputs, ["flickr", "faces"], ARTIFACT_DIR) or match_input(_inputs, ["thumbnails"], ARTIFACT_DIR)


def check_artifact():
    if not ARTIFACT_DIR:
        return {"attached": False, "note": "no attached dataset folder has 'artifact' in its name -- add awsaf49/artifact-dataset as an input"}
    summary = summarize_folder(ARTIFACT_DIR, max_seconds=60 if SMOKE_TEST else 900)
    out = {"attached": True, "path": ARTIFACT_DIR, "files": summary["files"], "gb": round(summary["bytes"] / 1e9, 2),
           "walk_complete": summary["complete"], "walk_seconds": round(summary["seconds"], 1),
           "top_level_folders": len(summary["by_top_dir"]),
           "by_folder": {k: v for k, v in sorted(summary["by_top_dir"].items())}, "csv_files": summary["csv_files"][:80]}
    described = {}
    for rel in summary["csv_files"][:80]:
        try:
            described[rel] = describe_csv(os.path.join(ARTIFACT_DIR, rel))
        except Exception as exc:  # noqa: BLE001
            described[rel] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    out["csv"] = described
    try:
        index = index_artifact(ARTIFACT_DIR)
        frame = pd.DataFrame(index)
        out["index_rows"] = int(len(frame))
        out["real_images"] = int((frame["is_fake"] == 0).sum())
        out["fake_images"] = int((frame["is_fake"] == 1).sum())
        out["per_folder_class_counts"] = {str(g): {"real": int((p["is_fake"] == 0).sum()), "fake": int((p["is_fake"] == 1).sum())}
                                         for g, p in frame.groupby("generator")}
        out["fake_generator_folders"] = sorted(frame[frame["is_fake"] == 1]["generator"].unique().tolist())
        out["real_source_folders"] = sorted(frame[frame["is_fake"] == 0]["generator"].unique().tolist())
        sample = frame.groupby("generator", group_keys=False).head(60)["path"].tolist()
        out["header_sample"] = header_summary(sample, threads=THREADS * 2)
    except Exception as exc:  # noqa: BLE001 -- the README's layout did not hold; the raw listing above is the record
        out["index_error"] = f"{type(exc).__name__}: {exc}"[:400]
    print(json.dumps({k: v for k, v in out.items() if k not in ("csv", "by_folder")}, indent=1))
    for rel, d in list(described.items())[:40]:
        print(f"  {rel}: columns {d.get('columns')} rows {d.get('rows')} target {d.get('value_counts', {}).get('target')}")
    return out


def check_ffhq():
    if not FFHQ_DIR:
        return {"attached": False, "note": "no attached dataset folder matches 'ffhq' / 'flickr faces' -- add the FFHQ thumbnails dataset as an input"}
    paths = list_images(FFHQ_DIR)
    out = {"attached": True, "path": FFHQ_DIR, "image_files": len(paths),
           "gb": round(sum(os.path.getsize(p) for p in paths) / 1e9, 2),
           "headers": header_summary(paths, threads=THREADS * 2),
           "license_on_kaggle_page": "Unknown (as seen by the user, 2026-10-06)",
           "license_at_source": "NVlabs/ffhq-dataset README: CC BY-NC-SA 4.0 (as cited in the spec; not re-read here)"}
    out["expected_70000_files"] = out["image_files"] == 70000
    out["all_128x128_rgb"] = out["headers"]["sizes"] == {"128x128": len(paths)} and set(out["headers"]["modes"]) == {"RGB"}
    print(json.dumps(out, indent=1))
    return out


preflight_step("artifact", check_artifact)
preflight_step("ffhq", check_ffhq)

# %%
# --- 0g. Source versus label: can "which dataset" or "what size" give the label away? ------------------
def check_source_vs_label():
    out = {}
    counts = PREFLIGHT.get("openfake_counts") or {}
    if counts:
        by = counts["rows_by_split_and_label"]
        out["openfake"] = {"real": sum(v for k, v in by.items() if k.endswith("/real")),
                           "fake": sum(v for k, v in by.items() if k.endswith("/fake")),
                           "sizes_in_streamed_sample": (PREFLIGHT.get("streaming_sample") or {}).get("sizes"),
                           "formats_in_streamed_sample": (PREFLIGHT.get("streaming_sample") or {}).get("format")}
    art = PREFLIGHT.get("artifact") or {}
    if art.get("attached"):
        out["artifact"] = {"real": art.get("real_images"), "fake": art.get("fake_images"),
                           "sizes_in_header_sample": (art.get("header_sample") or {}).get("sizes"),
                           "formats_in_header_sample": (art.get("header_sample") or {}).get("formats")}
    ffhq = PREFLIGHT.get("ffhq") or {}
    if ffhq.get("attached"):
        out["ffhq"] = {"real": ffhq.get("image_files"), "fake": 0, "sizes": ffhq["headers"]["sizes"],
                       "formats": ffhq["headers"]["formats"]}
    out["one_class_sources"] = [k for k, v in out.items() if isinstance(v, dict) and (not v.get("real") or not v.get("fake"))]
    out["note"] = ("a source that contributes only one class makes 'source' (and its image size) a label shortcut; "
                   "Task 4's source-only baseline measures it and decides what happens to that source")
    print(json.dumps(out, indent=1))
    return out


preflight_step("source_vs_label", check_source_vs_label)

# %%
# --- 0h. CLIP ViT-L/14: is it loadable, what does it output, how fast is it ---------------------------
CLIP_ENCODER = None


def check_clip():
    global CLIP_ENCODER
    from transformers import CLIPImageProcessor

    t0 = time.time()
    if SMOKE_TEST:
        CLIP_ENCODER = ClipEncoder(model=tiny_clip_for_smoke(), device=DEVICE, fp16=False)
        processor = CLIPImageProcessor()
    else:
        CLIP_ENCODER = ClipEncoder(CLIP_MODEL, device=DEVICE, fp16=True)
        processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL)
    config = CLIP_ENCODER.model.config
    out = {"model": "tiny random CLIP (smoke test)" if SMOKE_TEST else CLIP_MODEL, "loader": "transformers.CLIPVisionModelWithProjection",
           "load_seconds": round(time.time() - t0, 1), "image_size": int(config.image_size), "patch_size": int(config.patch_size),
           "hidden_size_pre_projection": int(config.hidden_size), "projection_dim": int(config.projection_dim),
           "official_preprocessing": {k: str(getattr(processor, k, None)) for k in
                                      ("size", "crop_size", "resample", "do_center_crop", "do_normalize", "image_mean", "image_std",
                                       "do_convert_rgb", "rescale_factor")}}
    # The embedding this engine stores, measured rather than read from the config.
    picture = synthetic_picture(np.random.default_rng(0), False, (640, 480))
    pixels = clip_pixels(resize_center_crop(picture)).unsqueeze(0)
    vector = CLIP_ENCODER.embed(pixels)
    out["embedding_shape_produced"] = list(vector.shape)
    out["embedding_is"] = "the projected image embedding (image_embeds), not L2-normalized, stored as float32"
    with torch.no_grad():
        raw = CLIP_ENCODER.model(pixel_values=pixels.to(DEVICE).half() if CLIP_ENCODER.half else pixels.to(DEVICE))
    out["pre_projection_pooled_dim"] = int(raw.last_hidden_state.shape[-1])
    # Our preprocessing (no JPEG step) against the official processor on the same picture.
    official = processor(images=picture, return_tensors="pt")["pixel_values"][0]
    out["official_processor_class"] = type(processor).__name__
    out["max_abs_difference_vs_official_preprocessing"] = float((official - pixels[0]).abs().max())
    out["one_grey_level_in_normalized_units"] = round(1 / 255 / min(CLIP_STD), 4)
    try:   # the PIL-backed official processor resamples exactly as this code does; the default one may use torchvision
        from transformers import CLIPImageProcessorPil

        official_pil = CLIPImageProcessorPil()(images=picture, return_tensors="pt")["pixel_values"][0]
        out["max_abs_difference_vs_official_pil_preprocessing"] = float((official_pil - pixels[0]).abs().max())
    except Exception as exc:  # noqa: BLE001 -- older transformers have no separate PIL class
        out["max_abs_difference_vs_official_pil_preprocessing"] = f"not available: {type(exc).__name__}"
    batch = torch.stack([preprocess_image(synthetic_picture(np.random.default_rng(i), i % 2 == 1)) for i in range(4)])
    out["determinism"] = embedding_is_deterministic(CLIP_ENCODER, batch)
    n_batches, size = (2, 8) if SMOKE_TEST else (6, EMBED_BATCH)
    speed_batch = torch.randn(size, 3, 224, 224)
    CLIP_ENCODER.embed(speed_batch)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        CLIP_ENCODER.embed(speed_batch)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    out["encoder_images_per_second"] = round(n_batches * size / max(time.perf_counter() - t0, 1e-9), 1)
    out["encoder_fp16"] = CLIP_ENCODER.half
    print(json.dumps(out, indent=1))
    return out


preflight_step("clip", check_clip)

# %%
# --- 0i. Disk budget for the embedding cache; Deepfake-Eval-2024 access; Synthbuster ------------------
def check_disk_budget():
    measured = (PREFLIGHT.get("clip") or {}).get("embedding_shape_produced")
    dim = measured[-1] if measured else 768
    f = PLAN["partition_fractions"]
    vectors_per_image = f["train"] * (1 + N_AUG_VIEWS) + f["val"] + (f["known_test"] + f["holdout"]) * (1 + len(DEGRADATIONS))
    out = {"embedding_dim_used": int(dim),
           "embedding_dim_source": "measured in the CLIP check above" if measured else
                                   "ASSUMED 768 from the handbook, because the CLIP check above did not produce one",
           "bytes_per_vector_float32": int(dim) * 4,
           "vectors_per_image_average": round(vectors_per_image, 2),
           "how": f"train images: 1 clean + {N_AUG_VIEWS} augmented views; validation: 1; test images: 1 clean + "
                  f"{len(DEGRADATIONS)} degradations ({', '.join(DEGRADATIONS)})",
           "projected_cache_gb": {str(n): round(n * vectors_per_image * dim * 4 / 1e9, 2) for n in (60000, 100000, 150000, 250000, 500000)},
           "metadata_table_size": "not yet measured (the scan stage reports it)",
           "disk_now": disk_report(), "images_written_to_disk": 0}
    print(json.dumps(out, indent=1))
    return out


def check_deepfake_eval_access():
    if SMOKE_TEST:
        return {"status": "not checked (smoke test)"}
    from huggingface_hub import HfApi

    api = HfApi(token=_token)
    out = {"repo": DEEPFAKE_EVAL_REPO, "user_submitted_request": "2026-10-06 (per the spec)", "use": "evaluation only; never training or tuning"}
    try:
        if hasattr(api, "auth_check"):
            api.auth_check(DEEPFAKE_EVAL_REPO, repo_type="dataset")
        else:
            api.list_repo_files(DEEPFAKE_EVAL_REPO, repo_type="dataset")
        out["status"] = "access granted to this token"
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        out["status"] = ("gated: access not granted to this token yet" if "Gated" in name else
                         "repository not found under this id (check the id)" if "NotFound" in name else f"check failed: {name}")
    print(json.dumps(out, indent=1))
    return out


preflight_step("disk_budget", check_disk_budget)
preflight_step("deepfake_eval_2024", check_deepfake_eval_access)
save_preflight("synthbuster", {
    "status": "not attached -- not needed until the end of V1 (the user uploads it to a private Kaggle dataset later)",
    "what_it_is": "Zenodo record 10066460: 9,000 fake images from nine generators, no real images, CC-BY-NC-SA 4.0 (per the spec)",
    "consequence_of_no_reals": "on its own it supports only the detection rate on fakes. ROC-AUC, EER or FPR need real images "
                               "from another source: the reserved real pool of this notebook (never trained on) is the planned "
                               "source, and any such number carries the caveat that reals and fakes come from different sources."})
save_preflight("run_hours", round(elapsed_hours(), 2))
print("\n" + "=" * 100)
print(f"Task 0 finished in {elapsed_hours():.2f} h. Failed checks: {list(PREFLIGHT['failed_steps']) or 'none'}")
print(f"Wrote {PREFLIGHT_PATH}, openfake_model_counts.csv and openfake_shard_counts.csv (aggregate counts only).")
if STAGE == "task0":
    print("STAGE is 'task0': nothing below this cell runs. Report this log and preflight_task0.json back before any extraction.")
print("=" * 100)

# %% [markdown]
# ## Stage control for everything after Task 0
#
# A stage runs only if `STAGE` reaches it, nothing earlier has stopped the run, and every `PLAN`
# value it needs has been decided. A stage that cannot run says why and the notebook finishes
# cleanly -- it never raises, because a failed Kaggle version keeps none of its output.

# %%
STOPPED = None
SHARD_SIZES = {s["path"]: s["size"] for s in OPENFAKE_SHARDS}


def halt(reason):
    global STOPPED
    if STOPPED is None:
        STOPPED = reason
        RESULTS["not_run"]["halted"] = reason
        save_results("not_run", RESULTS["not_run"])
    print("HALTED:", reason)


def ready(stage, *keys):
    if not stage_on(stage):
        print(f"stage '{stage}': not yet run (STAGE is '{STAGE}')")
        return False
    if STOPPED:
        print(f"stage '{stage}': not run -- {STOPPED}")
        return False
    missing = [k for k in keys if PLAN.get(k) is None]
    if missing:
        halt(f"stage '{stage}' needs PLAN{missing}, which are decided from the Task 0 results and are still None")
        return False
    return True


def chunk_files(subdir, pattern):
    """Finished chunk files of a stage: this session's, then those of attached earlier sessions."""
    found = {}
    for base in [PRIVATE_DIR] + [Path(d) for d in RESUME_DIRS]:
        for path in sorted((base / subdir).glob(pattern)):
            found.setdefault(path.name, path)
    return found


save_results("plan", {k: (list(v) if isinstance(v, tuple) else v) for k, v in PLAN.items()})

# %% [markdown]
# ## Tasks 1-2 (stage `scan`) -- pass 1: one metadata row per image
#
# Each selected image is read once: validity, file properties as received, perceptual hash, and the
# byte count after the shared pipeline. No encoder runs. One chunk file per shard (or per block of
# files), so a session that stops is resumed by attaching its output as an input.

# %%
SCAN_COMPLETE = False
if ready("scan", "openfake_shards", "keep_fraction", "artifact_per_folder", "ffhq_images"):
    scan_dir = PRIVATE_DIR / "scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    reusable = chunk_files("scan", "rows-*.parquet")
    pool = [s for s in OPENFAKE_SHARDS if s["config"] == "core" and s["split"] in PLAN["openfake_splits"]]
    work = []
    for shard in seeded_shard_order(pool)[:PLAN["openfake_shards"]]:
        key = "openfake-" + shard["path"].rsplit("/", 1)[-1].replace(".parquet", "")
        work.append((key, lambda shard=shard: scan_openfake_shard(OPENFAKE_READER, shard, PLAN["keep_fraction"], SEED, THREADS)))
    folder_index = []
    if ARTIFACT_DIR and PLAN["artifact_per_folder"]:
        by_folder = {}
        for row in index_artifact(ARTIFACT_DIR):
            by_folder.setdefault(row["generator"], []).append(row)
        for rows in by_folder.values():
            folder_index += sorted(rows, key=lambda r: stable_fraction(r["image_id"], SEED))[:PLAN["artifact_per_folder"]]
    if FFHQ_DIR and PLAN["ffhq_images"]:
        folder_index += sorted(index_ffhq(FFHQ_DIR), key=lambda r: stable_fraction(r["image_id"], SEED))[:PLAN["ffhq_images"]]
    block = 20 if SMOKE_TEST else 5000
    for i in range(0, len(folder_index), block):
        part = folder_index[i:i + block]
        work.append((f"folders-{i // block:05d}", lambda part=part: (scan_folder_rows(part, THREADS), {"kept": len(part)})))

    scan_log = {"chunks_total": len(work), "chunks_reused": 0, "chunks_written": 0, "rows_written": 0}
    SCAN_COMPLETE = True
    for n, (key, fn) in enumerate(work, 1):
        name = f"rows-{key}.parquet"
        if name in reusable:
            scan_log["chunks_reused"] += 1
            continue
        if not within_budget("scan"):
            SCAN_COMPLETE = False
            break
        t0 = time.time()
        rows, stats = fn()
        build_table(rows).to_parquet(scan_dir / (name + ".tmp"), index=False)
        os.replace(scan_dir / (name + ".tmp"), scan_dir / name)
        scan_log["chunks_written"] += 1
        scan_log["rows_written"] += len(rows)
        print(f"scan {n}/{len(work)} {key}: {len(rows)} rows {stats} in {time.time() - t0:.0f}s | elapsed {elapsed_hours():.2f} h")
    scan_log["complete"] = SCAN_COMPLETE
    save_results("scan", scan_log)
    print("scan:", scan_log)
    if not SCAN_COMPLETE:
        halt("the scan stopped at the time budget; attach this version's output as an input and run again to resume")

# %% [markdown]
# ## Task 2 (stage `split`) -- cleaning, and the evidence for the dedupe tolerance

# %%
TABLE, FOLDS, OLD_FAMILIES, SHORTCUT, GATE_OK = None, [], [], None, False
if ready("split"):
    files = chunk_files("scan", "rows-*.parquet")
    if not files:
        halt("no scan chunks found: run the scan stage first (or attach its output as an input)")
    else:
        TABLE = pd.concat([pd.read_parquet(p) for p in files.values()], ignore_index=True).drop_duplicates("image_id")
        TABLE = TABLE.reset_index(drop=True)
        TABLE["valid"] = TABLE["valid"].astype(bool)
        histogram = distance_histogram(TABLE)
        cleaning = {"scan_chunks": len(files), "table_mb_in_memory": round(TABLE.memory_usage(deep=True).sum() / 1e6, 1),
                    "cleaning": cleaning_report(TABLE), "null_counts": null_counts(TABLE),
                    "sources_before_dedupe": source_class_table(TABLE),
                    "pairs_at_each_hash_distance": histogram}
        save_results("cleaning", cleaning)
        print(json.dumps(cleaning, indent=1))
        if SHOW_DUPLICATE_PAIRS:
            # Private notebooks only: this draws dataset images into the notebook's rendered output.
            import io as _io

            import matplotlib.pyplot as plt
            from PIL import Image as _Image

            examples = duplicate_examples(TABLE, 7, per_distance=2)[:10]
            by_id = TABLE.set_index("image_id")
            wanted = sorted({i for a, b, _ in examples for i in (a, b)})
            blobs = fetch_image_bytes([(i, by_id.loc[i, "source_dataset"], by_id.loc[i, "locator"]) for i in wanted],
                                      OPENFAKE_READER, SHARD_SIZES)
            if examples:
                fig, axes = plt.subplots(len(examples), 2, figsize=(5, 2.4 * len(examples)), squeeze=False)
                for axis_row, (a, b, d) in zip(axes, examples):
                    for ax, image_id in zip(axis_row, (a, b)):
                        ax.imshow(_Image.open(_io.BytesIO(blobs[image_id])).convert("RGB"))
                        ax.set_title(f"distance {d}", fontsize=7)
                        ax.axis("off")
                plt.tight_layout()
                plt.show()
        if PLAN["phash_tolerance"] is None:
            halt("PLAN['phash_tolerance'] is not chosen yet: pick it from the pair counts above (and a look at real pairs)")

# %% [markdown]
# ## Tasks 3-4 (stage `split`) -- dedupe, rotating held-out families, leakage checks, shortcut baselines
#
# Groups (shared prompt, source image or identity) go whole into one partition. Each eligible family
# gets a fold in which all its fakes are unseen. The shortcut baselines then ask whether file
# properties or the source alone give the label away; the embedding stage does not start unless
# they cannot, or the residual has been read and accepted explicitly.

# %%
if TABLE is not None and ready("split", "phash_tolerance", "caps", "min_unseen_fakes", "old_families"):
    TABLE, dedupe_report = dedupe_table(TABLE, PLAN["phash_tolerance"])
    TABLE = assign_partitions(TABLE, PLAN["partition_fractions"], SEED)
    TABLE = apply_generator_caps(TABLE, PLAN["caps"], SEED)
    eligibility = eligible_families(TABLE, PLAN["min_unseen_fakes"])
    FOLDS = [f for f, e in eligibility.items() if e["eligible"]]
    present = set(TABLE.loc[TABLE["valid"], "generator_family"])
    OLD_FAMILIES = [f for f in PLAN["old_families"] if f in present]
    TABLE = add_fold_roles(TABLE, FOLDS, SEED, old_families=OLD_FAMILIES or None)
    leakage = leakage_report(TABLE, PLAN["phash_tolerance"], strict=False)
    live = TABLE[TABLE["valid"]]
    split_report = {
        "dedupe": dedupe_report, "rows_valid_after_dedupe_and_caps": int(len(live)),
        "dropped_by_reason": {str(k): int(v) for k, v in TABLE.loc[~TABLE["valid"], "drop_reason"].value_counts().items()},
        "partition_counts": {f"{s}/{'fake' if f else 'real'}": int(n) for (s, f), n in live.groupby(["split", "is_fake"]).size().items()},
        "groups": int(live["group_id"].nunique()), "family_eligibility": eligibility, "folds": FOLDS,
        "old_families_for_ablation": OLD_FAMILIES, "reserved_real_pool": int(((live["is_fake"] == 0) & (live["split"] == "holdout")).sum()),
        "leakage_violations": leakage["violations"],
        "leakage_per_fold": {f: {k: v for k, v in e.items() if k != "balance"} for f, e in leakage["folds"].items()},
        "class_balance_per_fold": {f: e["balance"]["per_role"] for f, e in leakage["folds"].items()},
        "generator_counts_per_fold": {f: e["balance"]["per_generator"] for f, e in leakage["folds"].items()},
        "sources_after_dedupe": source_class_table(TABLE),
        "auc_ci_halfwidth_at_fold_sizes": {
            f: {str(auc): round(auc_ci_halfwidth(auc, e["holdout"], min(e["holdout"], e["reserved_reals"])), 4) for auc in (0.7, 0.9, 0.99)}
            for f, e in eligibility.items() if e["eligible"]},
    }
    print(json.dumps({k: v for k, v in split_report.items() if k not in ("generator_counts_per_fold", "leakage_per_fold")}, indent=1))
    for fold, balance in split_report["class_balance_per_fold"].items():
        print(f"fold {fold:24s} " + " | ".join(f"{role}: {c.get('real', 0)} real / {c.get('fake', 0)} fake" for role, c in balance.items()))
    if leakage["violations"]:
        halt("leakage checks failed: " + "; ".join(leakage["violations"]))
    elif not FOLDS:
        halt("no family is eligible to be held out at PLAN['min_unseen_fakes']")

    # Task 4: shortcut baselines, before any encoder runs.
    SHORTCUT = run_shortcut_baselines(TABLE, N_BOOTSTRAP, SEED)
    print(f"\n{'features':10s} {'model':8s} {'ROC-AUC':>8s}  95% CI            separable   train/test")
    for r in SHORTCUT["results"]:
        if r["auc"] is None:
            print(f"{r['mode']:10s} {r['kind']:8s} not scored: {r.get('note')}")
        else:
            print(f"{r['mode']:10s} {r['kind']:8s} {r['auc']:8.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  {str(r['separable']):9s}  "
                  f"{r['n_train']}/{r['n_test']}")
    GATE_OK = bool(SHORTCUT["gate"]["passed"] or ACCEPT_RESIDUAL_SHORTCUT)
    split_report["shortcut_baselines"] = SHORTCUT
    split_report["shortcut_gate"] = {**SHORTCUT["gate"], "residual_accepted_explicitly": bool(ACCEPT_RESIDUAL_SHORTCUT and not SHORTCUT["gate"]["passed"])}
    print("shortcut gate:", json.dumps(split_report["shortcut_gate"], indent=1))
    TABLE.to_parquet(PRIVATE_DIR / "metadata.parquet", index=False)   # per-image rows: private output only
    split_report["metadata_table_mb_on_disk"] = round((PRIVATE_DIR / "metadata.parquet").stat().st_size / 1e6, 2)
    save_results("split", split_report)

# %% [markdown]
# ## Task 5 (stage `extract`) -- frozen CLIP embeddings
#
# Every image goes through the same pipeline (decode -> RGB -> 224x224 -> one JPEG re-encode -> CLIP
# normalization). Training images get the clean embedding plus augmented views; validation images
# only the clean one; test images the clean one plus each fixed degradation. Only vectors are saved.

# %%
EXTRACT_COMPLETE, STORE = False, None
if TABLE is not None and ready("extract"):
    if not GATE_OK:
        halt("the shortcut baselines separate the classes after the shared pipeline. Fix the pipeline (or rebalance sources) "
             "and re-run them; or, having read the numbers, set ACCEPT_RESIDUAL_SHORTCUT = True and carry the caveat")
    else:
        if CLIP_ENCODER is None:
            CLIP_ENCODER = ClipEncoder(model=tiny_clip_for_smoke(), device=DEVICE, fp16=False) if SMOKE_TEST else \
                ClipEncoder(CLIP_MODEL, device=DEVICE, fp16=True)
        VARIANTS = {"train": ["clean"] + [f"aug{k}" for k in range(N_AUG_VIEWS)], "val": ["clean"],
                    "known_test": ["clean"] + list(DEGRADATIONS), "holdout": ["clean"] + list(DEGRADATIONS)}
        live = TABLE[TABLE["valid"]]
        chunks, group_block, file_block = [], 4 if SMOKE_TEST else 8, 40 if SMOKE_TEST else 2000
        openfake = live[live["source_dataset"] == "openfake"]
        by_block = {}
        for image_id, locator, split in zip(openfake["image_id"], openfake["locator"], openfake["split"]):
            path, group, _ = locator.rsplit("#", 2)
            by_block.setdefault((path, int(group) // group_block), {})[locator] = (image_id, VARIANTS[split])
        for (path, block_index), wanted in sorted(by_block.items()):
            key = f"openfake-{path.rsplit('/', 1)[-1].replace('.parquet', '')}-g{block_index:03d}"
            chunks.append((key, lambda path=path, wanted=wanted: load_openfake_items(
                OPENFAKE_READER, {"path": path, "size": SHARD_SIZES[path]}, wanted)))
        files = live[live["source_dataset"] != "openfake"].sort_values("image_id")
        file_rows = [(i, p, VARIANTS[s]) for i, p, s in zip(files["image_id"], files["locator"], files["split"])]
        for i in range(0, len(file_rows), file_block):
            chunks.append((f"folders-{i // file_block:05d}", lambda part=file_rows[i:i + file_block]: load_folder_items(part)))

        STORE = ChunkStore(PRIVATE_DIR / "embeddings", "emb", [Path(d) / "embeddings" for d in RESUME_DIRS])
        probe = torch.stack([preprocess_image(synthetic_picture(np.random.default_rng(i), i % 2 == 1)) for i in range(4)])
        determinism = embedding_is_deterministic(CLIP_ENCODER, probe)

        def chunks_within_budget():
            for key, load in chunks:
                if not STORE.has(key) and not within_budget("extract"):
                    return
                yield key, load

        extract_report = extract_embeddings(CLIP_ENCODER, chunks_within_budget(), STORE, EMBED_BATCH, SEED, THREADS)
        EXTRACT_COMPLETE = all(STORE.has(key) for key, _ in chunks)
        extract_report.update(chunks_total=len(chunks), complete=EXTRACT_COMPLETE, determinism_check=determinism,
                              encoder=CLIP_ENCODER.name, embedding_dim=CLIP_ENCODER.dim, variants=VARIANTS,
                              images_to_embed=int(len(live)), images_written_to_disk=0, disk_after=disk_report())
        save_results("extract", extract_report)
        print(json.dumps({k: v for k, v in extract_report.items() if k != "variants"}, indent=1))
        if not EXTRACT_COMPLETE:
            halt("embedding extraction stopped at the time budget; attach this version's output as an input and run again to resume")

# %% [markdown]
# ## Tasks 6-8 (stage `heads`) -- heads, calibration, evaluation suites
#
# Per fold: logistic regression (C chosen on that fold's validation split) and a one-hidden-layer
# MLP. Temperature, operating threshold and uncertain band are fitted on the fold's validation
# split, which holds seen generators only. Fake is the positive class. Confidence intervals are 95%
# stratified bootstraps on the first seed; further seeds are run only where a seed changes something.

# %%
FOLD_RESULTS, FOLD_EXTRAS, BANK = {}, {}, None
HEAD_KWARGS = dict(n_aug=N_AUG_VIEWS, n_resamples=N_BOOTSTRAP, target_recall=TARGET_RECALL, band_recall=BAND_RECALL,
                   band_fpr=BAND_FPR, device=DEVICE)


def run_seeds(fold, kind, extra_tests=None):
    """First seed with CIs; more seeds (point values only) unless the fit is deterministic."""
    runs, extras = [], None
    for k, seed in enumerate(SEEDS):
        result, extra = run_fold(TABLE, BANK, fold, kind, seed, with_ci=(k == 0), extra_tests=extra_tests, **HEAD_KWARGS)
        runs.append(result)
        if k == 0:
            extras = extra
        if result.get("not_run") or result["seed_changes"].startswith("nothing"):
            break
    suites = sorted({s for r in runs for s in r.get("suites", {})})
    across = {s: {m: mean_std([metric_value(r, s, m) for r in runs]) for m in ("roc_auc", "eer")} for s in suites}
    return {"first_seed": runs[0], "seeds_run": [r["seed"] for r in runs], "seed_changes": runs[0]["seed_changes"],
            "across_seeds": across}, extras


def cell(result, suite, metric="roc_auc"):
    """'0.9731 [0.9690, 0.9770]' for a results table, or 'not yet run'."""
    try:
        m = result["first_seed"]["suites"][suite]["metrics"][metric]
        return f"{m['value']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]"
    except (KeyError, TypeError):
        return NOT_RUN


if TABLE is not None and ready("heads"):
    if STORE is None or not EXTRACT_COMPLETE:
        halt("the embedding cache is not complete; finish the extract stage first")
    else:
        BANK = EmbeddingBank(*STORE.load())
        print(f"embedding bank: {len(BANK.index)} vectors of {BANK.dim} numbers; variants {BANK.variants()}")
        for fold in FOLDS:
            FOLD_RESULTS[fold], FOLD_EXTRAS[fold] = {}, {}
            for kind in ("logreg", "mlp"):
                t0 = time.time()
                FOLD_RESULTS[fold][kind], FOLD_EXTRAS[fold][kind] = run_seeds(fold, kind)
                print(f"fold {fold:20s} {kind:6s} known AUC {cell(FOLD_RESULTS[fold][kind], 'known')} | unseen AUC "
                      f"{cell(FOLD_RESULTS[fold][kind], 'unseen')} | seeds {FOLD_RESULTS[fold][kind]['seeds_run']} | {time.time() - t0:.0f}s")
            # MLP against logistic regression on the same test images (first seeds).
            versus = {}
            for suite in ("known", "unseen"):
                a = FOLD_EXTRAS[fold]["mlp"]["frames"].get(suite)
                b = FOLD_EXTRAS[fold]["logreg"]["frames"].get(suite)
                if a is not None and b is not None and len(a) == len(b):
                    versus[suite] = paired_metric_difference(a["prob"].to_numpy(), b["prob"].to_numpy(), a["is_fake"].to_numpy(),
                                                             "roc_auc", 0.5, N_BOOTSTRAP, SEED)
            FOLD_RESULTS[fold]["mlp_minus_logreg"] = versus
            save_results("folds", FOLD_RESULTS)

# %%
# Old-only ablation (its own row, labelled an ablation): train on real images plus the old families
# only, then score it on the unseen-test images of every other fold -- the same images those folds use.
OLD_ONLY_RESULT = None
if BANK is not None and ready("heads") and FOLD_PREFIX + OLD_ONLY in TABLE.columns:
    modern = [f for f in FOLDS if f not in OLD_FAMILIES]
    tests = {f"unseen[{f}]": fold_rows(TABLE, f, "unseen_test") for f in modern}
    OLD_ONLY_RESULT, old_extras = run_seeds(OLD_ONLY, "logreg", extra_tests=tests)
    versus = {}
    for f in modern:
        a = old_extras["frames"].get(f"unseen[{f}]")
        b = FOLD_EXTRAS[f]["logreg"]["frames"].get("unseen")
        if a is not None and b is not None:
            merged = a.merge(b[["image_id", "prob"]], on="image_id", suffixes=("_old_only", "_lofo"))
            versus[f] = paired_metric_difference(merged["prob_old_only"].to_numpy(), merged["prob_lofo"].to_numpy(),
                                                 merged["is_fake"].to_numpy(), "roc_auc", 0.5, N_BOOTSTRAP, SEED)
        print(f"old-only -> {f:20s} unseen AUC {cell(OLD_ONLY_RESULT, f'unseen[{f}]')} | LOFO fold on the same images "
              f"{cell(FOLD_RESULTS[f]['logreg'], 'unseen')}")
    OLD_ONLY_RESULT["old_only_minus_lofo_on_same_images"] = versus
    OLD_ONLY_RESULT["old_families"] = OLD_FAMILIES
    save_results("old_only_ablation", OLD_ONLY_RESULT)
elif stage_on("heads") and BANK is not None:
    RESULTS["not_run"]["old_only_ablation"] = "none of PLAN['old_families'] is present in the data"
    save_results("not_run", RESULTS["not_run"])

# %%
# Final model: the head retrained on every family, including those held out in the folds. It has no
# untouched unseen test of its own -- the folds above are its generalization *estimate*, not a measurement.
FINAL_RESULT = None
if BANK is not None and ready("heads"):
    final_fold = "final_all_families"
    TABLE[FOLD_PREFIX + final_fold] = [s if ok and s in ("train", "val", "known_test") else ("unused" if ok else None)
                                       for s, ok in zip(TABLE["split"], TABLE["valid"])]
    FINAL_RESULT, final_extras = run_seeds(final_fold, "logreg")
    weights_dir = PRIVATE_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)
    hashes = {}
    if final_extras["head"] is not None:
        hashes["v1_final_logreg.npz"] = save_head(weights_dir / "v1_final_logreg.npz", final_extras["head"])
    for fold in FOLDS:
        for kind in ("logreg", "mlp"):
            head = FOLD_EXTRAS[fold][kind]["head"]
            if head is not None:
                name = f"v1_fold_{fold}_{kind}.npz".replace("/", "_")
                hashes[name] = save_head(weights_dir / name, head)
    FINAL_RESULT["weights_sha256"] = hashes
    FINAL_RESULT["weights_location"] = "artifacts/visual/private/weights (kept out of any public repository until the dataset licenses are checked)"
    FINAL_RESULT["what_it_cannot_claim"] = (
        "This head has seen every family, so it has no unseen-generator test of its own. Its known-test number is in-domain. "
        "The leave-one-family-out folds are an estimate of how it behaves on a new family, not a measurement of this model.")
    TABLE.drop(columns=[FOLD_PREFIX + final_fold], inplace=True)
    save_results("final_model", FINAL_RESULT)
    print(f"final model (all families): known AUC {cell(FINAL_RESULT, 'known')}; weights hashed: {len(hashes)} files")

# %% [markdown]
# ## Task 9 -- heavier pixel baselines (off by default)
#
# A small CNN from scratch and a fine-tuned ResNet-50, on a capped subset of images held in memory.
# They run only when their flag is on, which is only after Tasks 1-8 have real results.

# %%
PIXEL_RESULTS = {}
if BANK is not None and ready("heads") and (RUN_SMALL_CNN or RUN_RESNET50) and FOLDS:
    pixel_fold = FOLDS[0]
    pixel_caps = {"train": 24, "val": 12, "known_test": 12, "unseen_test": 12} if SMOKE_TEST else \
        {"train": 6000, "val": 1000, "known_test": 1500, "unseen_test": 1500}
    side = 64 if SMOKE_TEST else 224
    arrays = {}
    for role, cap in pixel_caps.items():
        rows = fold_rows(TABLE, pixel_fold, role)
        rows = rows.iloc[np.argsort([stable_fraction(i, SEED) for i in rows["image_id"]])]
        rows = pd.concat([rows[rows["is_fake"] == c].head(cap // 2) for c in (0, 1)])   # balanced, capped
        blobs = fetch_image_bytes(zip(rows["image_id"], rows["source_dataset"], rows["locator"]), OPENFAKE_READER, SHARD_SIZES)
        pixels = [(pixel_array(blobs[i], side), y) for i, y in zip(rows["image_id"], rows["is_fake"]) if i in blobs]
        pixels = [(x, y) for x, y in pixels if x is not None]
        arrays[role] = (np.stack([x for x, _ in pixels]), np.array([y for _, y in pixels], dtype=int))
    builders = {}
    if RUN_SMALL_CNN:
        builders["small_cnn"] = lambda: SmallCNN()
    if RUN_RESNET50:
        builders["resnet50"] = lambda: build_resnet50(pretrained=not SMOKE_TEST)
    for name, build in builders.items():
        print(f"training {name} on {len(arrays['train'][0])} images of fold {pixel_fold} ...")
        model, fit = train_pixel_model(build(), *arrays["train"], *arrays["val"], DEVICE, epochs=1 if SMOKE_TEST else 8, seed=SEED)
        entry = {"fold": pixel_fold, "fit": fit, "image_side": side, "images": {r: int(len(a[0])) for r, a in arrays.items()}, "suites": {}}
        for suite, role in (("known", "known_test"), ("unseen", "unseen_test")):
            x, y = arrays[role]
            probs = sigmoid(pixel_logits(model, x, DEVICE))
            entry["suites"][suite] = evaluate_suite(probs, y, 0.5, None, N_BOOTSTRAP, SEED)
        PIXEL_RESULTS[name] = entry
    save_results("pixel_baselines", PIXEL_RESULTS)
elif stage_on("heads"):
    RESULTS["not_run"]["pixel_baselines"] = "RUN_SMALL_CNN and RUN_RESNET50 are off (they run only after Tasks 1-8 have real results)"
    save_results("not_run", RESULTS["not_run"])

# %% [markdown]
# ## Report: tables, plots, per-generator errors, closing paragraph

# %%
SUITE_COLUMNS = ["known", "unseen", "known/jpeg90", "known/jpeg70", "known/jpeg50", "unseen/jpeg90", "unseen/jpeg70", "unseen/jpeg50"]
SUITE_NAMES = {"known": "T1 known generators", "unseen": "T3 unseen family", "known/jpeg90": "T4 known, JPEG 90",
               "known/jpeg70": "T4 known, JPEG 70", "known/jpeg50": "T4 known, JPEG 50", "unseen/jpeg90": "T4 unseen, JPEG 90",
               "unseen/jpeg70": "T4 unseen, JPEG 70", "unseen/jpeg50": "T4 unseen, JPEG 50"}
MODEL_ROWS = {"logreg": "V1 logistic regression", "mlp": "V1 MLP"}
SHORTCUT_ROW = "shortcut baseline (file properties after the shared pipeline)"

# The reporting grid: every cell is "not yet run" until a real run fills it.
report = {"metric": "ROC-AUC with 95% bootstrap CI (first seed)", "folds": {}, "deferred_suites": {
    "T2 sibling generator": "not run (run only if the real metadata has a clean sibling pair)",
    "T5 screenshots": NOT_RUN, "T6 resize": NOT_RUN + " (0.5x embeddings are cached by the extract stage)",
    "T7 crop": NOT_RUN + " (centre-crop embeddings are cached by the extract stage)", "T8 AI-edited pairs": NOT_RUN,
    "T9 real-world (Deepfake-Eval-2024)": NOT_RUN + " (once, after the V1 head is frozen, if access is granted)"}}
shortcut_cell = NOT_RUN
if SHORTCUT is not None:
    worst = max((r for r in SHORTCUT["results"] if r["mode"] == "pipeline" and r["auc"] is not None),
                key=lambda r: abs(r["auc"] - 0.5), default=None)
    if worst:
        shortcut_cell = f"{worst['auc']:.4f} [{worst['ci_low']:.4f}, {worst['ci_high']:.4f}] ({worst['kind']}, known-test partition, all families)"
for fold in (FOLDS or ["(folds not built yet)"]):
    grid = results_template([SHORTCUT_ROW] + list(MODEL_ROWS.values()), [SUITE_NAMES[s] for s in SUITE_COLUMNS])
    grid[SHORTCUT_ROW][SUITE_NAMES["known"]] = shortcut_cell
    for kind, row_name in MODEL_ROWS.items():
        for suite in SUITE_COLUMNS:
            if fold in FOLD_RESULTS and kind in FOLD_RESULTS[fold]:
                grid[row_name][SUITE_NAMES[suite]] = cell(FOLD_RESULTS[fold][kind], suite)
    for name, entry in PIXEL_RESULTS.items():
        if entry["fold"] == fold:
            grid[f"{name} (pixel baseline, capped subset)"] = {
                SUITE_NAMES[s]: (f"{entry['suites'][s]['metrics']['roc_auc']['value']:.4f} [{entry['suites'][s]['metrics']['roc_auc']['ci_low']:.4f}, "
                                 f"{entry['suites'][s]['metrics']['roc_auc']['ci_high']:.4f}]" if s in entry["suites"] else NOT_RUN)
                for s in SUITE_COLUMNS}
    report["folds"][fold] = grid
report["old_only_ablation"] = ({f: cell(OLD_ONLY_RESULT, f"unseen[{f}]") for f in FOLDS if f not in OLD_FAMILIES}
                               if OLD_ONLY_RESULT else NOT_RUN)
save_results("report", report)
for fold, grid in report["folds"].items():
    print(f"\nFold: {fold} held out")
    for row_name, cells in grid.items():
        print(f"  {row_name}")
        for suite, value in cells.items():
            print(f"      {suite:22s} {value}")

# %%
if FOLD_RESULTS and ready("heads"):
    import matplotlib.pyplot as plt

    # Per-image predictions (private) and the aggregate per-generator table (aggregate, may be committed).
    frames, per_generator = [], []
    for fold in FOLDS:
        for suite, frame in FOLD_EXTRAS[fold]["logreg"]["frames"].items():
            frames.append(frame.assign(fold=fold, suite=suite))
        first = FOLD_RESULTS[fold]["logreg"]["first_seed"]
        for suite in ("known", "unseen"):
            for generator, entry in first.get("suites", {}).get(suite, {}).get("per_generator", {}).items():
                rate = entry.get("detection_rate") or entry.get("false_positive_rate")
                per_generator.append({"fold": fold, "suite": suite, "generator": generator, "kind": entry["kind"], "n": entry["n"],
                                      "flagged_rate": rate["value"], "flagged_rate_ci_low": rate["ci_low"], "flagged_rate_ci_high": rate["ci_high"],
                                      "roc_auc_vs_all_reals": (entry.get("roc_auc_vs_all_reals") or {}).get("value")})
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(PRIVATE_DIR / "predictions_v1.parquet", index=False)
    pd.DataFrame(per_generator).to_csv(OUTPUT_DIR / "per_generator_v1.csv", index=False)

    # Reliability diagrams: known and unseen, per fold (logistic regression, first seed, after temperature scaling).
    fig, axes = plt.subplots(1, len(FOLDS), figsize=(3.2 * len(FOLDS), 3.2), squeeze=False)
    for ax, fold in zip(axes[0], FOLDS):
        for suite, style in (("known", "o-"), ("unseen", "s--")):
            frame = FOLD_EXTRAS[fold]["logreg"]["frames"].get(suite)
            if frame is not None:
                curve = reliability_curve(frame["prob"].to_numpy(), frame["is_fake"].to_numpy())
                ax.plot([c["mean_predicted"] for c in curve], [c["fraction_fake"] for c in curve], style, markersize=3, label=suite)
        ax.plot([0, 1], [0, 1], "k-", linewidth=0.5, alpha=0.4)
        ax.set_title(f"held out: {fold}", fontsize=8)
        ax.set_xlabel("predicted P(fake)", fontsize=7)
        ax.set_ylabel("observed share of fakes", fontsize=7)
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reliability_v1.png", dpi=130)
    plt.show()

    # DET curves on the unseen test of each fold.
    plt.figure(figsize=(6.5, 5.5))
    for fold in FOLDS:
        frame = FOLD_EXTRAS[fold]["logreg"]["frames"].get("unseen")
        if frame is not None and frame["is_fake"].nunique() == 2:
            det = det_curve_points(frame["prob"].to_numpy(), frame["is_fake"].to_numpy(), positive_label=1)
            plt.plot(np.array(det["far"]) * 100, np.array(det["frr"]) * 100,
                     label=f"{fold} (AUC {metric_value(FOLD_RESULTS[fold]['logreg']['first_seed'], 'unseen'):.3f})")
    plt.plot([0.01, 100], [0.01, 100], "k-", linewidth=0.5, alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Real images flagged as fake (%)")
    plt.ylabel("Fake images missed (%)")
    plt.title("V1 logistic regression: DET curve on each fold's unseen family")
    plt.legend(fontsize=6, loc="lower left")
    plt.grid(True, which="both", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "det_curves_v1.png", dpi=130)
    plt.show()

# %%
# --- Closing paragraph, generated from this run's numbers ---------------------------------------------
parts = []
if not FOLD_RESULTS:
    parts.append(f"Visual Engine V1: stage '{STAGE}' ran; no model has been trained in this run, so every V1 number is not yet run."
                 + (f" Stopped: {STOPPED}." if STOPPED else ""))
else:
    lr = {f: FOLD_RESULTS[f]["logreg"] for f in FOLDS}
    gaps = {f: r["first_seed"].get("known_minus_unseen", {}).get("roc_auc") for f, r in lr.items()}
    gap_text = "; ".join(f"{f}: known {cell(lr[f], 'known')}, unseen {cell(lr[f], 'unseen')}, gap {g['difference']:+.4f} "
                         f"[{g['ci_low']:+.4f}, {g['ci_high']:+.4f}]" for f, g in gaps.items() if g)
    parts.append(f"V1 logistic regression on frozen CLIP embeddings, leave-one-family-out over {len(FOLDS)} families (ROC-AUC, 95% bootstrap CI, "
                 f"first seed). Known-minus-unseen gap per fold -- {gap_text}.")
    real_gaps = [f for f, g in gaps.items() if g and g["excludes_zero"] and g["difference"] > 0]
    values = [g["difference"] for g in gaps.values() if g]
    if values:
        parts.append(f"The gap's CI excludes zero for {len(real_gaps)} of {len(values)} folds ({', '.join(real_gaps) or 'none'}); "
                     f"across folds it runs from {min(values):+.4f} to {max(values):+.4f}. Folds share training data, so they are "
                     "shown one by one, not pooled.")
    unseen_auc = {f: metric_value(lr[f]["first_seed"], "unseen") for f in FOLDS}
    unseen_auc = {f: v for f, v in unseen_auc.items() if v is not None}
    if unseen_auc:
        worst_family = min(unseen_auc, key=unseen_auc.get)
        parts.append(f"The family V1 generalizes to worst is {worst_family} (unseen ROC-AUC {cell(lr[worst_family], 'unseen')}).")
    verdicts_seen = Counter(v["verdict"] for f in FOLDS for v in FOLD_RESULTS[f].get("mlp_minus_logreg", {}).values())
    parts.append(f"MLP against logistic regression on the same test images: {dict(verdicts_seen)} over the folds' known and unseen "
                 "tests ('tie' = the CI of the difference includes zero).")
    ece = [(f, lr[f]["first_seed"]["suites"]["unseen"]["ece_before_calibration"], lr[f]["first_seed"]["suites"]["unseen"]["metrics"]["ece"]["value"])
           for f in FOLDS if "unseen" in lr[f]["first_seed"].get("suites", {})]
    if ece:
        better = sum(1 for _, before, after in ece if after < before)
        parts.append("Temperature scaling was fitted on seen generators only; on the unseen families ECE went " +
                     "; ".join(f"{f} {before:.4f} -> {after:.4f}" for f, before, after in ece) +
                     f" (lower in {better} of {len(ece)} folds).")
    parts.append(f"Seeds: logistic regression -- {lr[FOLDS[0]]['seed_changes']}; MLP -- {FOLD_RESULTS[FOLDS[0]]['mlp']['seed_changes']}.")
    if OLD_ONLY_RESULT:
        parts.append("Old-only ablation (trained on " + ", ".join(OLD_FAMILIES) + " plus real images), scored on each modern family's "
                     "unseen test: " + "; ".join(f"{f} {v}" for f, v in report["old_only_ablation"].items()) + ".")
    if SHORTCUT is not None and not SHORTCUT["gate"]["passed"]:
        parts.append("Caveat carried by every number above: after the shared pipeline a file-property or source-only baseline still "
                     "separates the classes (" + "; ".join(SHORTCUT["gate"]["failing"]) + ").")
    if FINAL_RESULT:
        parts.append(FINAL_RESULT["what_it_cannot_claim"])
closing_paragraph = " ".join(parts)
print(closing_paragraph)
save_results("closing_paragraph", closing_paragraph)
save_results("run_hours", round(elapsed_hours(), 2))
print(f"\nresults: {RESULTS_PATH} | preflight: {PREFLIGHT_PATH} | stage: {STAGE} | elapsed {elapsed_hours():.2f} h"
      + (f" | halted: {STOPPED}" if STOPPED else ""))
