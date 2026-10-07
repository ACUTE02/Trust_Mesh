"""Audio front-end: resample -> silence trim -> fixed length -> LFCC (+deltas), plus the silence
ablations and training-time augmentation used in Phase 2B.

**Why LFCC, not mel.** LFCC is the front-end of the official ASVspoof 2019/2021 baselines. Its
filters are spaced linearly, so it keeps resolution at high frequencies, which is where vocoder and
TTS artifacts live. Mel spacing compresses exactly that region because it is tuned to human
perception, which is the right trade for ASR and the wrong one for spoof detection.

LFCC is implemented here in plain torch (linear triangular filterbank -> log -> DCT-II -> deltas)
rather than via `torchaudio.transforms.LFCC`, so the pipeline has no torchaudio dependency.
`LFCCExtractor` is batched and runs on any device; `VoicePreprocessor` uses the same extractor for
single clips, so cached features and on-the-fly (augmented) features are the same computation.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

TARGET_SR = 16000


def resample_linear(waveform, sr, target_sr=TARGET_SR):
    """Linear-interpolation resample. ASVspoof LA is already 16 kHz, so this is normally a no-op;
    it exists so an odd-rate out-of-domain clip doesn't silently go through at the wrong rate."""
    if sr == target_sr:
        return waveform, sr
    n_out = int(round(len(waveform) * target_sr / sr))
    x_in = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, waveform).astype(np.float32), target_sr


def frame_energy_db(waveform, sr, frame_ms=20.0, hop_ms=10.0):
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(waveform) < frame:
        waveform = np.pad(waveform, (0, frame - len(waveform)))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12), frame, hop


def trim_silence(waveform, sr, top_db=30.0, frame_ms=20.0, hop_ms=10.0):
    """Energy-based leading/trailing silence trim (librosa.effects.trim-style, no dependency).

    Returns (trimmed waveform, stats). Stats carry the leading/trailing silence and speech durations
    (what the silence-leakage probe measures) and the speech boundaries in samples (what the silence
    ablations cut at).
    """
    db, frame, hop = frame_energy_db(waveform, sr, frame_ms, hop_ms)
    voiced = np.flatnonzero(db > db.max() - top_db)
    total_s = len(waveform) / sr
    if len(voiced) == 0:  # nothing above the threshold: keep the clip, report it as all silence
        return waveform, {"total_s": total_s, "lead_silence_s": 0.0, "trail_silence_s": 0.0,
                          "speech_s": 0.0, "silence_ratio": 1.0, "all_silence": True,
                          "start_sample": 0, "end_sample": len(waveform)}
    start = int(voiced[0] * hop)
    end = min(len(waveform), int(voiced[-1] * hop + frame))
    trimmed = waveform[start:end]
    lead_s, trail_s = start / sr, (len(waveform) - end) / sr
    return trimmed, {
        "total_s": total_s,
        "lead_silence_s": lead_s,
        "trail_silence_s": trail_s,
        "speech_s": len(trimmed) / sr,
        "silence_ratio": (lead_s + trail_s) / total_s if total_s else 0.0,
        "all_silence": False,
        "start_sample": start,
        "end_sample": end,
    }


def fix_length(waveform, target_len):
    """Crop or repeat-pad to a fixed sample count (repeat-padding is the usual ASVspoof practice --
    zero-padding would inject exactly the silence cue this pipeline is trying to remove)."""
    if len(waveform) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(waveform) >= target_len:
        return waveform[:target_len]
    repeats = int(np.ceil(target_len / len(waveform)))
    return np.tile(waveform, repeats)[:target_len].astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# Silence ablations (Phase 2B, Task 4). Each variant removes a different part of what the silence
# could be leaking, so comparing them says *which* part the untrimmed model was using:
#
#   variant                    silence duration   silence content (what's actually in it)
#   untrimmed                        kept              kept
#   zeros_original_duration          kept              removed (digital zeros)
#   zeros_fixed / noise_fixed        removed           removed (identical zeros / identical-level noise)
#   silence_only (fixed window)      removed           kept -- and nothing else is kept
#   trimmed                          removed           removed
# ---------------------------------------------------------------------------------------------------
SILENCE_ABLATIONS = ("zeros_fixed", "noise_fixed", "zeros_original_duration")


def ablate_silence(waveform, sr, mode, stats, fixed_seconds=0.5, noise_dbfs=-60.0, rng=None):
    """Rebuild a clip with its leading and trailing silence replaced according to `mode`.

    Both ends are replaced, not just the leading one: trailing silence also leaks the label (it is
    the second-strongest timing cue), so replacing only the lead would leave a known cue in place and
    confound the test.
    """
    start, end = stats["start_sample"], stats["end_sample"]
    speech = waveform[start:end]
    if mode == "zeros_fixed":
        pad = np.zeros(int(fixed_seconds * sr), dtype=np.float32)
        return np.concatenate([pad, speech, pad])
    if mode == "noise_fixed":
        n = int(fixed_seconds * sr)
        rng = rng if rng is not None else np.random.default_rng(0)
        noise = (rng.standard_normal(2 * n) * 10 ** (noise_dbfs / 20)).astype(np.float32)
        return np.concatenate([noise[:n], speech, noise[n:]])
    if mode == "zeros_original_duration":
        return np.concatenate([np.zeros(start, dtype=np.float32), speech,
                               np.zeros(len(waveform) - end, dtype=np.float32)])
    raise ValueError(f"unknown silence ablation: {mode}")


def silence_window(waveform, stats, n_samples):
    """Exactly `n_samples` of the clip's genuine silence (leading, then trailing), or None if the clip
    has less silence than that. A fixed-size window means silence *duration* cannot reach the model;
    only what the silence contains can."""
    start, end = stats["start_sample"], stats["end_sample"]
    silence = np.concatenate([waveform[:start], waveform[end:]])
    if len(silence) < n_samples:
        return None
    return silence[:n_samples].astype(np.float32)


# ---------------------------------------------------------------------------------------------------
# LFCC
# ---------------------------------------------------------------------------------------------------
def linear_filterbank(n_filters, n_fft, sr):
    """Triangular filters spaced linearly in Hz (the 'linear' in LFCC)."""
    edges = torch.linspace(0.0, sr / 2.0, n_filters + 2)
    bin_freqs = torch.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    bank = torch.zeros(n_filters, len(bin_freqs))
    for i in range(n_filters):
        low, center, high = edges[i], edges[i + 1], edges[i + 2]
        rising = (bin_freqs - low) / torch.clamp(center - low, min=1e-6)
        falling = (high - bin_freqs) / torch.clamp(high - center, min=1e-6)
        bank[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return bank


def dct_matrix(n_out, n_in):
    """Orthonormal DCT-II."""
    n = torch.arange(n_in, dtype=torch.float32)
    k = torch.arange(n_out, dtype=torch.float32).unsqueeze(1)
    mat = torch.cos(torch.pi * k * (2 * n + 1) / (2 * n_in)) * np.sqrt(2.0 / n_in)
    mat[0] = mat[0] / np.sqrt(2.0)
    return mat


def deltas(features, width=2):
    """Standard regression deltas over a +/-`width` frame window, edges replicated. Accepts
    [coeffs, frames] or [batch, coeffs, frames]."""
    squeeze = features.dim() == 2
    x = features.unsqueeze(0) if squeeze else features
    padded = F.pad(x, (width, width), mode="replicate")
    n_frames = x.shape[-1]
    out = torch.zeros_like(x)
    for n in range(1, width + 1):
        out += n * (padded[..., width + n: width + n + n_frames] - padded[..., width - n: width - n + n_frames])
    out = out / (2.0 * sum(n * n for n in range(1, width + 1)))
    return out.squeeze(0) if squeeze else out


class LFCCExtractor(torch.nn.Module):
    """Batched LFCC (+deltas): [batch, samples] -> [batch, n_coeffs*3, frames], on any device.

    `filterbank_augment`, if given, is applied to the log linear-filterbank energies before the DCT.
    That is where frequency masking means something (masking whole frequency bands); masking rows
    of the cepstrum afterwards would not correspond to any frequency region.
    """

    def __init__(self, sample_rate=TARGET_SR, n_fft=512, win_ms=20.0, hop_ms=10.0, n_filters=70,
                 n_coeffs=20, add_deltas=True):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = int(sample_rate * win_ms / 1000)
        self.hop_length = int(sample_rate * hop_ms / 1000)
        self.n_coeffs = n_coeffs
        self.add_deltas = add_deltas
        self.register_buffer("filterbank", linear_filterbank(n_filters, n_fft, sample_rate))
        self.register_buffer("dct", dct_matrix(n_coeffs, n_filters))
        self.register_buffer("window", torch.hann_window(self.win_length))

    @property
    def n_features(self):
        return self.n_coeffs * (3 if self.add_deltas else 1)

    def n_frames(self, n_samples):
        return n_samples // self.hop_length + 1

    def forward(self, waves, filterbank_augment=None):
        spec = torch.stft(waves, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=True, pad_mode="reflect", return_complex=True)
        log_fbank = torch.log(torch.matmul(self.filterbank, spec.abs() ** 2) + 1e-10)
        if filterbank_augment is not None:
            log_fbank = filterbank_augment(log_fbank)
        feats = torch.matmul(self.dct, log_fbank)
        if self.add_deltas:
            d1 = deltas(feats)
            feats = torch.cat([feats, d1, deltas(d1)], dim=1)
        return feats


class VoicePreprocessor:
    """waveform -> (n_coeffs*3, n_frames) LFCC+delta+delta-delta tensor.

    Defaults follow the ASVspoof LFCC baseline: 20 ms window / 10 ms hop, 70 linear filters,
    20 cepstral coefficients, plus first and second deltas = 60 feature dimensions.
    """

    def __init__(self, sample_rate=TARGET_SR, clip_seconds=4.0, n_fft=512, win_ms=20.0, hop_ms=10.0,
                 n_filters=70, n_coeffs=20, trim=True, top_db=30.0, add_deltas=True):
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.clip_samples = int(sample_rate * clip_seconds)
        self.trim = trim
        self.top_db = top_db
        self.extractor = LFCCExtractor(sample_rate, n_fft, win_ms, hop_ms, n_filters, n_coeffs, add_deltas)
        self.n_coeffs = n_coeffs
        self.add_deltas = add_deltas
        self.hop_length = self.extractor.hop_length

    @property
    def n_features(self):
        return self.extractor.n_features

    @property
    def n_frames(self):
        return self.extractor.n_frames(self.clip_samples)

    def prepare_waveform(self, waveform, sr):
        waveform, sr = resample_linear(np.asarray(waveform, dtype=np.float32), sr, self.sample_rate)
        stats = {"total_s": len(waveform) / sr, "lead_silence_s": 0.0, "trail_silence_s": 0.0,
                 "speech_s": len(waveform) / sr, "silence_ratio": 0.0, "all_silence": False,
                 "start_sample": 0, "end_sample": len(waveform)}
        if self.trim:
            waveform, stats = trim_silence(waveform, sr, self.top_db)
        return fix_length(waveform, self.clip_samples), stats

    @torch.no_grad()
    def lfcc(self, waveform):
        wave = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
        return self.extractor(wave)[0]

    def __call__(self, waveform, sr):
        fixed, stats = self.prepare_waveform(waveform, sr)
        return self.lfcc(fixed), stats

    def featurize_clips(self, clips, progress_every=1000):
        """Returns (features [N,1,F,T] float32, labels [N], per-clip silence stats)."""
        features, labels, stats = [], [], []
        for i, clip in enumerate(clips):
            feats, clip_stats = self(clip["waveform"], clip["sample_rate"])
            features.append(feats)
            labels.append(clip["label"])
            stats.append(clip_stats)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  featurized {i + 1}/{len(clips)} clips")
        x = torch.stack(features).unsqueeze(1)
        return x, torch.tensor(labels, dtype=torch.long), stats


# ---------------------------------------------------------------------------------------------------
# Training-time augmentation (Phase 2B Run B). Applied to training batches only -- never to
# validation or test, which the data sources enforce by only calling these when `training=True`.
# ---------------------------------------------------------------------------------------------------
def simulate_low_bitrate(waves, generator, downsample_factor=4, quant_bits=6, mu=255.0, p_apply=0.3):
    """Cheap proxy for a low-bitrate lossy-codec (e.g. MP3) round-trip: linear-interpolation
    downsample/upsample (the bandwidth loss a low bitrate causes) plus mu-law companding quantized to
    `quant_bits` (the quantization noise a low bitrate causes). This is NOT a real MP3 encode/decode --
    no differentiable, per-batch-fast MP3 codec exists -- it is a documented approximation of the two
    dominant artifacts a low-bitrate codec introduces, applied to a random subset of clips per batch.
    waves: [batch, samples] on any device, expected in [-1, 1]."""
    batch, n = waves.shape
    device = waves.device
    apply = (torch.rand(batch, 1, generator=generator, device=device) < p_apply).float()

    narrow = F.interpolate(waves.unsqueeze(1), size=max(n // downsample_factor, 1), mode="linear",
                           align_corners=False)
    wide = F.interpolate(narrow, size=n, mode="linear", align_corners=False).squeeze(1)

    x = wide.clamp(-1.0, 1.0)
    companded = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    levels = 2 ** quant_bits - 1
    quantized = torch.round((companded + 1) / 2 * levels) / levels * 2 - 1
    expanded = torch.sign(quantized) * (torch.expm1(quantized.abs() * math.log1p(mu)) / mu)

    return waves + apply * (expanded - waves)


def augment_waveforms(waves, generator, snr_db=(10.0, 40.0), gain_db=(-6.0, 6.0), p_noise=0.5,
                      codec_sim=False):
    """Additive white noise at a random per-clip SNR (applied to a random half of clips) and a random
    per-clip gain. waves: [batch, samples] on any device. `codec_sim=True` (Phase 2C Run F only) adds
    `simulate_low_bitrate` on top -- Phase 2B explicitly skipped a codec augmentation for time."""
    batch = waves.shape[0]
    device = waves.device

    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand(batch, 1, generator=generator, device=device)

    signal_power = waves.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-10)
    noise_power = signal_power / 10 ** (uniform(*snr_db) / 10)
    noise = torch.randn(waves.shape, generator=generator, device=device) * noise_power.sqrt()
    apply = (torch.rand(batch, 1, generator=generator, device=device) < p_noise).float()
    waves = waves + apply * noise
    waves = waves * 10 ** (uniform(*gain_db) / 20)
    if codec_sim:
        waves = simulate_low_bitrate(waves, generator)
    return waves


def spec_augment(generator, n_freq_masks=2, max_freq_width=8, n_time_masks=2, max_time_width=40):
    """SpecAugment-style masking for the log-filterbank tensor [batch, filters, frames]. Masked
    regions are set to each clip's mean log energy, so masking doesn't inject an artificial floor."""

    def apply(log_fbank):
        batch, n_filters, n_frames = log_fbank.shape
        device = log_fbank.device
        out = log_fbank.clone()
        fill = log_fbank.mean(dim=(1, 2), keepdim=True).expand_as(out)
        filters = torch.arange(n_filters, device=device).view(1, -1, 1)
        frames = torch.arange(n_frames, device=device).view(1, 1, -1)
        mask = torch.zeros_like(out, dtype=torch.bool)
        for _ in range(n_freq_masks):
            width = torch.randint(0, max_freq_width + 1, (batch, 1, 1), generator=generator, device=device)
            start = (torch.rand(batch, 1, 1, generator=generator, device=device) * (n_filters - width).clamp_min(1)).long()
            mask |= (filters >= start) & (filters < start + width)
        for _ in range(n_time_masks):
            width = torch.randint(0, max_time_width + 1, (batch, 1, 1), generator=generator, device=device)
            start = (torch.rand(batch, 1, 1, generator=generator, device=device) * (n_frames - width).clamp_min(1)).long()
            mask |= (frames >= start) & (frames < start + width)
        return torch.where(mask, fill, out)

    return apply
