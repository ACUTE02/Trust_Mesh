"""Phase 2D machinery: resumable pooled-feature extraction for codec-processed clips, a disk budget check,
cost projection for the pilot, and the clip-clustered paired bootstrap used by the pre-registered criteria.

The encoder is frozen, so every version of a clip (clean, augmented, evaluation condition) is turned into
a pooled feature vector once and stored; the head trains on those vectors. Only the mean-pooled half is
kept (Model D uses nothing else): [25 layers, hidden] float16 = 51,200 bytes per version for XLS-R 300M.

Storage is a directory of chunk files, `chunk_00000.npz`, `chunk_00001.npz`, ... Each holds every version
produced for a fixed run of clips. A chunk is written to a temporary name and renamed when complete, so a
file that exists is a finished chunk; after a crash or a session limit the run resumes at the first
missing chunk. Nothing but pooled vectors, positions and condition indices is stored: no waveforms.
"""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .codec import apply_condition, condition_rng
from .evaluate import compute_eer
from .features import INT16_SCALE, to_int16
from .preprocess import fix_length, resample_linear, trim_silence

CHUNK_CLIPS = 256
DISK_FACTOR = 1.3
BYTES_PER_VALUE = 2  # float16


class DiskBudgetError(RuntimeError):
    pass


def prepare_clip(wave, sr, clip_samples, top_db=30.0):
    """Resample, trim leading/trailing silence, repeat-pad or crop to `clip_samples`, quantise to int16.
    The same steps (and the same trim rule) as the clean cache of Phase 2B, applied to a codec's output."""
    wave, sr = resample_linear(np.asarray(wave, dtype=np.float32), sr)
    _, stats = trim_silence(wave, sr, top_db)
    return to_int16(fix_length(wave[stats["start_sample"]:stats["end_sample"]], clip_samples))


# --- cost and disk ------------------------------------------------------------------------------------------
def version_counts(n_clips, n_trainval, n_eval_clips, n_eval_conditions, per_clip=3):
    clean = int(n_clips)
    aug = int(n_trainval) * per_clip
    evaluation = int(n_eval_clips) * int(n_eval_conditions)
    return {"clean": clean, "augmented": aug, "evaluation": evaluation, "total": clean + aug + evaluation}


def bytes_per_version(n_layers, hidden_size):
    return int(n_layers) * int(hidden_size) * BYTES_PER_VALUE


def check_disk(path, required_bytes, factor=DISK_FACTOR):
    """Plain-language stop if the free space under `path` is below `factor` x the cache size."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    report = {"required_gb": required_bytes / 1e9, "needed_with_margin_gb": factor * required_bytes / 1e9,
              "free_gb": free / 1e9, "ok": free >= factor * required_bytes}
    if not report["ok"]:
        raise DiskBudgetError(
            f"Not enough free disk space. The feature cache needs about {report['required_gb']:.1f} GB "
            f"(and {factor:g}x that is required before starting: {report['needed_with_margin_gb']:.1f} GB), "
            f"but only {report['free_gb']:.1f} GB is free. Nothing was started. Lower EVAL_PER_CONDITION or "
            f"the number of training clips, or free space, and run again.")
    return report


def project_cost(measured_seconds, measured_versions, measured_bytes, total_versions):
    """Straight-line projection from the pilot. It is an estimate, not a promise: the pilot says so."""
    per_version = measured_seconds / max(measured_versions, 1)
    return {"seconds_per_version": per_version, "projected_seconds": per_version * total_versions,
            "projected_hours": per_version * total_versions / 3600.0,
            "projected_cache_gb": measured_bytes / max(measured_versions, 1) * total_versions / 1e9}


# --- chunk store --------------------------------------------------------------------------------------------
def chunk_name(index):
    return f"chunk_{int(index):05d}.npz"


def find_chunk(index, dirs):
    for d in dirs:
        candidate = Path(d) / chunk_name(index)
        if candidate.exists():
            return candidate
    return None


def load_chunks(dirs, kinds=("clean", "aug", "eval")):
    """Concatenate every finished chunk found in `dirs` (first directory wins per chunk index).
    Returns {kind: {"positions", "data", ...extra index arrays}}; missing kinds come back empty."""
    indices = sorted({int(p.stem.split("_")[1]) for d in dirs if Path(d).exists() for p in Path(d).glob("chunk_*.npz")
                      if not p.stem.endswith(".tmp")})
    parts = {k: [] for k in kinds}
    for index in indices:
        with np.load(find_chunk(index, dirs)) as z:
            for kind in kinds:
                if f"{kind}_positions" in z.files and len(z[f"{kind}_positions"]):
                    parts[kind].append({name[len(kind) + 1:]: z[name] for name in z.files if name.startswith(kind + "_")}
                                       | {"data": z[kind]})
    out = {}
    for kind, items in parts.items():
        if not items:
            out[kind] = {"positions": np.zeros(0, dtype=np.int64), "data": None}
            continue
        merged = {key: np.concatenate([it[key] for it in items]) for key in items[0]}
        out[kind] = merged
    return out


def extract_features(get_clip, positions, encoder, clip_samples, out_dir, trainval, aug_conditions, aug_assign,
                     eval_positions, eval_conditions, ffmpeg=None, resume_dirs=(), seed=42, chunk_clips=CHUNK_CLIPS,
                     batch_size=16, workers=4, progress_every=1, deadline=None, log=print):
    """Produce every version of every clip in `positions`, chunk by chunk, skipping finished chunks.

    get_clip(position) -> (waveform float32, sample_rate).     encoder: SSLEncoder (frozen).
    trainval:       set of positions that get the clean version plus augmented versions.
    aug_assign:     {position: [index into aug_conditions] * 3}.
    eval_positions: set of positions that get one version per name in eval_conditions.
    deadline:       time.time() value after which the run stops cleanly at the next chunk boundary.
    Returns a stats dict (versions made, seconds split into decode / conditions / encoder, stopped_early).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [out_dir, *map(Path, resume_dirs)]
    hidden = encoder.hidden_size
    positions = list(positions)
    n_chunks = (len(positions) + chunk_clips - 1) // chunk_clips
    stats = {"versions": 0, "chunks_done": 0, "chunks_reused": 0, "seconds": 0.0, "seconds_decode": 0.0,
             "seconds_conditions": 0.0, "seconds_encoder": 0.0, "stopped_early": False, "n_chunks": n_chunks}
    started = time.time()
    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        for c in range(n_chunks):
            if find_chunk(c, dirs) is not None:
                stats["chunks_reused"] += 1
                continue
            if deadline is not None and time.time() > deadline:
                stats["stopped_early"] = True
                log(f"  time limit reached after chunk {c - 1}; stopping cleanly. Run again to continue.")
                break
            chunk_positions = positions[c * chunk_clips:(c + 1) * chunk_clips]

            t = time.time()
            waves = {p: get_clip(p) for p in chunk_positions}
            stats["seconds_decode"] += time.time() - t

            jobs = []   # (kind, position, condition name or None, extra index)
            for p in chunk_positions:
                jobs.append(("clean", p, None, 0))
                if p in trainval:
                    for slot, ci in enumerate(aug_assign[p]):
                        jobs.append(("aug", p, aug_conditions[int(ci)], int(slot) * 1000 + int(ci)))
                if p in eval_positions:
                    for ei, name in enumerate(eval_conditions):
                        jobs.append(("eval", p, name, ei))

            def make(job):
                kind, p, name, _ = job
                wave, sr = waves[p]
                if name is None:
                    return prepare_clip(wave, sr, clip_samples)
                wave16, _ = resample_linear(np.asarray(wave, dtype=np.float32), sr)
                rng = condition_rng(seed + (1_000_000 if kind == "eval" else 0), p, name)
                return prepare_clip(apply_condition(wave16, name, rng, ffmpeg), 16000, clip_samples)

            t = time.time()
            clips = list(pool.map(make, jobs))
            stats["seconds_conditions"] += time.time() - t

            t = time.time()
            feats = np.zeros((len(jobs), encoder.n_layers, hidden), dtype=np.float16)
            for s in range(0, len(jobs), batch_size):
                batch = torch.from_numpy(np.stack(clips[s:s + batch_size]).astype(np.float32) / INT16_SCALE)
                pooled = encoder.pooled(batch)[..., :hidden]
                feats[s:s + len(batch)] = pooled.cpu().numpy().astype(np.float16)
            stats["seconds_encoder"] += time.time() - t

            arrays = {}
            for kind in ("clean", "aug", "eval"):
                idx = [i for i, j in enumerate(jobs) if j[0] == kind]
                arrays[f"{kind}_positions"] = np.array([jobs[i][1] for i in idx], dtype=np.int64)
                arrays[kind] = feats[idx] if idx else np.zeros((0, encoder.n_layers, hidden), dtype=np.float16)
                if kind == "aug":
                    arrays["aug_slot"] = np.array([jobs[i][3] // 1000 for i in idx], dtype=np.int16)
                    arrays["aug_cond"] = np.array([jobs[i][3] % 1000 for i in idx], dtype=np.int16)
                if kind == "eval":
                    arrays["eval_cond"] = np.array([jobs[i][3] for i in idx], dtype=np.int16)
            tmp = out_dir / f"chunk_{c:05d}.tmp.npz"
            np.savez(tmp, **arrays)
            os.replace(tmp, out_dir / chunk_name(c))

            stats["versions"] += len(jobs)
            stats["chunks_done"] += 1
            stats["seconds"] = time.time() - started
            if progress_every and (stats["chunks_done"] % progress_every == 0 or c == n_chunks - 1):
                rate = stats["seconds"] / max(stats["versions"], 1)
                log(f"  chunk {c + 1}/{n_chunks}: {stats['versions']} versions made, {rate:.2f} s per version "
                    f"(decode {stats['seconds_decode']:.0f}s, conditions {stats['seconds_conditions']:.0f}s, "
                    f"encoder {stats['seconds_encoder']:.0f}s)")
    finally:
        pool.shutdown(wait=True)
    stats["seconds"] = time.time() - started
    return stats


# --- evaluation helpers -------------------------------------------------------------------------------------
def clustered_paired_bootstrap(scores_a, scores_b, labels, groups, n_resamples=1000, seed=0, alpha=0.05):
    """Percentile bootstrap of EER(a) - EER(b) that resamples whole clips.

    Rows are (clip, condition) versions; several rows share a clip, so resampling rows would treat copies of
    one clip as independent and make the interval too narrow. Clips are resampled within class (bonafide /
    spoof) and all of a drawn clip's rows are kept, for both models alike.
    """
    sa, sb = np.asarray(scores_a, dtype=np.float64), np.asarray(scores_b, dtype=np.float64)
    labels, groups = np.asarray(labels), np.asarray(groups)
    unique, first = np.unique(groups, return_index=True)
    group_label = labels[first]
    rows_of = {g: np.flatnonzero(groups == g) for g in unique}
    pos, neg = unique[group_label == 1], unique[group_label != 1]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        drawn = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        idx = np.concatenate([rows_of[g] for g in drawn])
        diffs.append(compute_eer(sa[idx], labels[idx])["eer"] - compute_eer(sb[idx], labels[idx])["eer"])
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    point = compute_eer(sa, labels)["eer"] - compute_eer(sb, labels)["eer"]
    return {"difference": float(point), "ci_low": float(low), "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0), "clustered": True, "n_resamples": n_resamples,
            "n_clips_bonafide": int(len(pos)), "n_clips_spoof": int(len(neg)), "n_rows": int(len(labels))}


def criterion_status(difference):
    """Reads a bootstrap difference dict (F2 minus control) the way the pre-registered criteria do.
    Negative difference = F2 has the lower (better) EER.
    not_worse: the interval does not lie wholly above zero (criteria 1 and 2a).
    better:    the interval lies wholly below zero (criterion 2b)."""
    return {"not_worse": bool(difference["ci_low"] <= 0), "better": bool(difference["ci_high"] < 0)}


def share_flagged(scores, labels, threshold):
    """Share of genuine (label 1) clips scored below `threshold`, i.e. what a caller would see flagged."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    genuine = scores[labels == 1]
    return float((genuine < threshold).mean()) if len(genuine) else float("nan")
