"""On-disk feature cache for Phase 2B, and the data sources training reads from.

One decode pass over the selected clips writes everything later steps need, as memory-mapped
arrays, so a kernel crash mid-run never forces the audio to be downloaded and decoded again:

  wave_trimmed.npy     int16 [N, samples]   trimmed, fixed-length waveforms. Runs A and B both read
                                            this and compute LFCC on the GPU per batch -- B only adds
                                            augmentation, so A vs. B changes exactly one variable. The
                                            SSL encoder (Run C) reads it too.
  lfcc_<variant>.npy   float16 [N, F, T]    untrimmed and silence-ablation features (comparisons only).
  lfcc_silence_only.npy                     LFCC of a fixed window of genuine silence per clip.
  clip_stats.json                           silence stats + which clips have enough silence.
  manifest.json                             what was built; a matching manifest means "reuse".

Every waveform is quantized to int16 before its LFCC is computed, so cached and on-the-fly
features see the same signal precision.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .preprocess import SILENCE_ABLATIONS, ablate_silence, augment_waveforms, fix_length
from .preprocess import resample_linear, silence_window, spec_augment, trim_silence

INT16_SCALE = 32767.0
LFCC_VARIANTS = ("untrimmed",) + SILENCE_ABLATIONS


def to_int16(wave):
    return np.clip(np.round(wave * INT16_SCALE), -32768, 32767).astype(np.int16)


def quantize(wave):
    return to_int16(wave).astype(np.float32) / INT16_SCALE


def _manifest_matches(path, expected):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("config") == expected
    except (json.JSONDecodeError, OSError):
        return False


def build_feature_cache(clip_iter, n_clips, cache_dir, extractor, device, clip_samples, silence_samples,
                        top_db=30.0, fixed_silence_s=0.5, noise_dbfs=-60.0, seed=42, batch_size=128,
                        config_key=None, progress_every=2000):
    """Single pass: decode each clip once, build every waveform variant, LFCC them on `device` in
    batches, and write the memmaps. Reuses an existing cache when its manifest matches."""
    from numpy.lib.format import open_memmap

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    expected = {"n_clips": int(n_clips), "clip_samples": int(clip_samples), "silence_samples": int(silence_samples),
                "top_db": top_db, "fixed_silence_s": fixed_silence_s, "noise_dbfs": noise_dbfs,
                "variants": list(LFCC_VARIANTS), "key": config_key}
    if _manifest_matches(manifest_path, expected):
        print(f"reusing feature cache in {cache_dir}")
        return open_feature_cache(cache_dir)

    n_feat = extractor.n_features
    frames, silence_frames = extractor.n_frames(clip_samples), extractor.n_frames(silence_samples)
    waves = open_memmap(cache_dir / "wave_trimmed.npy", mode="w+", dtype=np.int16, shape=(n_clips, clip_samples))
    lfcc = {v: open_memmap(cache_dir / f"lfcc_{v}.npy", mode="w+", dtype=np.float16, shape=(n_clips, n_feat, frames))
            for v in LFCC_VARIANTS}
    lfcc_silence = open_memmap(cache_dir / "lfcc_silence_only.npy", mode="w+", dtype=np.float16,
                               shape=(n_clips, n_feat, silence_frames))
    stats_out, has_silence = [], np.zeros(n_clips, dtype=bool)
    pending = {v: [] for v in LFCC_VARIANTS + ("silence_only",)}

    def flush(variant, target):
        rows = pending[variant]
        if not rows:
            return
        batch = torch.from_numpy(np.stack([w for _, w in rows])).to(device)
        with torch.no_grad():
            feats = extractor(batch).half().cpu().numpy()
        for (row, _), f in zip(rows, feats):
            target[row] = f
        pending[variant] = []

    start_time = time.time()
    count = 0
    for i, clip in enumerate(clip_iter):
        wave, sr = resample_linear(np.asarray(clip["waveform"], dtype=np.float32), clip["sample_rate"])
        _, stats = trim_silence(wave, sr, top_db)
        stats_out.append({k: stats[k] for k in ("total_s", "lead_silence_s", "trail_silence_s",
                                                  "speech_s", "silence_ratio", "all_silence")})
        waves[i] = to_int16(fix_length(wave[stats["start_sample"]:stats["end_sample"]], clip_samples))

        rng = np.random.default_rng(seed + i)
        variant_waves = {"untrimmed": fix_length(wave, clip_samples)}
        for mode in SILENCE_ABLATIONS:
            variant_waves[mode] = fix_length(ablate_silence(wave, sr, mode, stats, fixed_silence_s, noise_dbfs, rng),
                                             clip_samples)
        for variant, w in variant_waves.items():
            pending[variant].append((i, quantize(w)))
            if len(pending[variant]) >= batch_size:
                flush(variant, lfcc[variant])

        window = silence_window(wave, stats, silence_samples)
        has_silence[i] = window is not None
        pending["silence_only"].append((i, quantize(window if window is not None else np.zeros(silence_samples, np.float32))))
        if len(pending["silence_only"]) >= batch_size:
            flush("silence_only", lfcc_silence)

        count = i + 1
        if progress_every and count % progress_every == 0:
            rate = count / max(time.time() - start_time, 1e-6)
            print(f"  featurized {count}/{n_clips} clips ({rate:.0f} clips/s)")
    if count != n_clips:
        raise RuntimeError(f"expected {n_clips} clips, got {count}")
    for variant in LFCC_VARIANTS:
        flush(variant, lfcc[variant])
    flush("silence_only", lfcc_silence)

    for array in [waves, lfcc_silence, *lfcc.values()]:
        array.flush()
    (cache_dir / "clip_stats.json").write_text(
        json.dumps({"stats": stats_out, "has_silence_window": has_silence.tolist()}), encoding="utf-8")
    manifest_path.write_text(json.dumps({"config": expected, "built_seconds": round(time.time() - start_time, 1)},
                                        indent=2), encoding="utf-8")
    return open_feature_cache(cache_dir)


def open_feature_cache(cache_dir):
    cache_dir = Path(cache_dir)
    extra = json.loads((cache_dir / "clip_stats.json").read_text(encoding="utf-8"))
    return {
        "wave_trimmed": np.load(cache_dir / "wave_trimmed.npy", mmap_mode="r"),
        "lfcc": {v: np.load(cache_dir / f"lfcc_{v}.npy", mmap_mode="r") for v in LFCC_VARIANTS},
        "lfcc_silence_only": np.load(cache_dir / "lfcc_silence_only.npy", mmap_mode="r"),
        "stats": extra["stats"],
        "has_silence_window": np.array(extra["has_silence_window"], dtype=bool),
    }


def cache_size_gb(cache_dir):
    return sum(p.stat().st_size for p in Path(cache_dir).glob("*") if p.is_file()) / 1e9


# ---------------------------------------------------------------------------------------------------
# Data sources: fetch(idx, training) -> model input on `device`. Augmentation only when training.
# ---------------------------------------------------------------------------------------------------
def lfcc_array_source(array, device):
    """Cached float16 LFCC [N, F, T] -> LCNN input [B, 1, F, T]."""
    def fetch(idx, training):
        return torch.from_numpy(np.asarray(array[np.asarray(idx)], dtype=np.float32)).unsqueeze(1).to(device)
    return fetch


def wave_lfcc_source(waves_int16, extractor, device, augment=False, seed=0):
    """int16 waveforms -> (optionally augmented) LFCC on `device` -> LCNN input [B, 1, F, T]."""
    generator = torch.Generator(device=device).manual_seed(seed) if augment else None

    def fetch(idx, training):
        w = torch.from_numpy(np.asarray(waves_int16[np.asarray(idx)], dtype=np.float32) / INT16_SCALE).to(device)
        apply = augment and training
        if apply:
            w = augment_waveforms(w, generator)
        with torch.no_grad():
            feats = extractor(w, filterbank_augment=spec_augment(generator) if apply else None)
        return feats.unsqueeze(1)
    return fetch


def array_source(array, device):
    """Cached pooled SSL features [N, layers, dim] -> SSLHead input. Also used as-is for Phase 2C
    Model E's per-frame sequence cache [N, frames, hidden] -> SSLSeqHead input, since both are just a
    memmap slice on the leading axis."""
    def fetch(idx, training):
        return torch.from_numpy(np.asarray(array[np.asarray(idx)], dtype=np.float32)).to(device)
    return fetch


def array_mean_only_source(pooled_array, hidden_size, device):
    """Phase 2C Model D: the same cached mean+std pooled SSL features as Model C, sliced to the mean
    half only. No extra encoder pass -- `ssl.py`'s `pooled()` already concatenates mean and std, so
    this ablates Model D's one variable (std pooling) by reading less of the existing cache."""
    def fetch(idx, training):
        arr = np.asarray(pooled_array[np.asarray(idx)], dtype=np.float32)[..., :hidden_size]
        return torch.from_numpy(arr).to(device)
    return fetch


def wave_ssl_sequence_source(waves_int16, ssl_encoder, layer, device):
    """int16 waveforms -> on-the-fly per-frame sequence of one frozen SSL layer (Phase 2C Model E).

    Deliberately not cached to disk: a full-dataset single-layer sequence cache is tens of GB (see
    `ssl.py`'s `build_ssl_sequence_cache`), and stacking it on top of the feature cache this notebook
    already builds overran Kaggle's available disk on a real run (Version 2, 2026-09-29 -- issue #25).
    Same trade as `wave_ssl_source`: no disk cost, more GPU time re-running the frozen encoder.
    """
    def fetch(idx, training):
        w = torch.from_numpy(np.asarray(waves_int16[np.asarray(idx)], dtype=np.float32) / INT16_SCALE).to(device)
        return ssl_encoder.sequence(w, layer)
    return fetch


def wave_ssl_source(waves_int16, ssl_encoder, device, augment=False, seed=0, codec_sim=False, pooled_cache=None):
    """int16 waveforms -> on-the-fly frozen-SSL mean+std pooled features (Phase 2C Model F).

    Augmentation must happen upstream of the frozen encoder's forward pass, so -- unlike every other
    SSL run, which reads the pre-computed cache in `ssl.py` -- training calls (`training=True`) re-run
    the (frozen, no-grad) encoder every batch. Slower than the cache, but the encoder's own weights
    never see a gradient.

    `pooled_cache`, if given (Phase 2B's already-built `ssl_features`, same encoder/pooling/clips): used
    for every non-training call instead of re-running the encoder. Validation/test are never augmented,
    so their features are identical to that cache's -- recomputing them from the waveform every epoch
    (as an earlier version of this function did) was pure wasted GPU time, since validation runs once
    per epoch for up to `SSL_EPOCHS` epochs and test runs once more after training.
    """
    generator = torch.Generator(device=device).manual_seed(seed) if augment else None

    def fetch(idx, training):
        idx = np.asarray(idx)
        if not training and pooled_cache is not None:
            return torch.from_numpy(np.asarray(pooled_cache[idx], dtype=np.float32)).to(device)
        w = torch.from_numpy(np.asarray(waves_int16[idx], dtype=np.float32) / INT16_SCALE).to(device)
        if augment and training:
            w = augment_waveforms(w, generator, codec_sim=codec_sim)
        return ssl_encoder.pooled(w)
    return fetch
