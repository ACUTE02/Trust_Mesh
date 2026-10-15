"""Codec and channel conditions for Phase 2D (docs/PHASE_2D_CODEC_ROBUSTNESS.md + addendum 1).

Real codecs go through ffmpeg (decode -> encode -> decode, per clip, nothing stored); room echo and
low-SNR noise are synthetic and need no external data. Condition names are fixed here so a result
file, a test and the runbook always mean the same thing.

Training menu (used to build augmented training clips):
    aac32, aac32_via48k, mp3_64, opus24, echo, noise_lowsnr
Held out (evaluation only, never trained on):
    aac16, opus12, mulaw8k, mp3_32

Everything takes and returns 16 kHz mono float32. The caller trims and fixes the length afterwards, in
exactly the way the clean clip is prepared, so a codec's delay or padding is treated like the service
would treat it.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CODEC_SR = 16000

TRAINING_MENU = ("aac32", "aac32_via48k", "mp3_64", "opus24", "echo", "noise_lowsnr")
HELD_OUT = ("aac16", "opus12", "mulaw8k", "mp3_32")
ALL_CONDITIONS = TRAINING_MENU + HELD_OUT

# name -> (file extension, encoder arguments, sample rate to pass through before encoding or None)
CODEC_SPECS = {
    "aac32": (".m4a", ["-c:a", "aac", "-b:a", "32k"], None),
    "aac32_via48k": (".m4a", ["-c:a", "aac", "-b:a", "32k"], 48000),
    "mp3_64": (".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"], None),
    "opus24": (".opus", ["-c:a", "libopus", "-b:a", "24k"], None),
    "aac16": (".m4a", ["-c:a", "aac", "-b:a", "16k"], None),
    "opus12": (".opus", ["-c:a", "libopus", "-b:a", "12k"], None),
    "mp3_32": (".mp3", ["-c:a", "libmp3lame", "-b:a", "32k"], None),
    # G.711 telephony: 8 kHz, 8-bit mu-law, then back to 16 kHz.
    "mulaw8k": (".wav", ["-ar", "8000", "-c:a", "pcm_mulaw"], None),
    # Used only when libopus is missing from the available ffmpeg; named so the substitution is visible.
    "aac24_for_opus24": (".m4a", ["-c:a", "aac", "-b:a", "24k"], None),
    "aac12_for_opus12": (".m4a", ["-c:a", "aac", "-b:a", "12k"], None),
}
SUBSTITUTES = {"opus24": "aac24_for_opus24", "opus12": "aac12_for_opus12"}
SYNTHETIC = ("echo", "noise_lowsnr")

ECHO_RT60_S = (0.2, 0.8)       # reverberation time of the synthetic room
NOISE_SNR_DB = (5.0, 20.0)     # additive white noise, signal-to-noise range
AUGMENTATIONS_PER_CLIP = 3


class FFmpegMissing(RuntimeError):
    pass


def find_ffmpeg():
    """Path to an ffmpeg binary: the system one if present, else the `imageio-ffmpeg` package's."""
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 -- the message below is what the owner needs to see
        raise FFmpegMissing(
            "No ffmpeg found: it is not on the PATH and the imageio-ffmpeg package could not provide one "
            f"({type(exc).__name__}). Install it with: pip install imageio-ffmpeg") from exc


def _run(ffmpeg, *args):
    done = subprocess.run([str(ffmpeg), "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.strip()[:300]}")


def codec_roundtrip(wave, name, ffmpeg):
    """float32 16 kHz -> codec `name` -> float32 16 kHz."""
    import soundfile as sf

    ext, args, via_rate = CODEC_SPECS[name]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "in.wav"
        sf.write(source, np.asarray(wave, dtype=np.float32), CODEC_SR, subtype="PCM_16")
        current = source
        if via_rate:
            current = tmp / "up.wav"
            _run(ffmpeg, "-i", source, "-ar", via_rate, "-c:a", "pcm_s16le", current)
        coded = tmp / f"coded{ext}"
        _run(ffmpeg, "-i", current, *args, coded)
        back = tmp / "back.wav"
        _run(ffmpeg, "-i", coded, "-ac", 1, "-ar", CODEC_SR, "-c:a", "pcm_s16le", back)
        out, _ = sf.read(back, dtype="float32")
    return np.ascontiguousarray(out, dtype=np.float32)


def synthetic_rir(rng, rt60_s, sr=CODEC_SR):
    """Room impulse response: unit direct path plus Gaussian noise decaying 60 dB over `rt60_s`."""
    n = int(sr * rt60_s)
    t = np.arange(n) / sr
    tail = rng.normal(0.0, 1.0, n) * 10.0 ** (-3.0 * t / rt60_s)   # amplitude: -60 dB at t = rt60
    tail *= 0.3 / (np.sqrt(np.mean(tail ** 2)) + 1e-12)            # reverberant energy about -10 dB re direct
    tail[0] = 1.0
    return tail.astype(np.float32)


def apply_echo(wave, rng):
    rt60 = float(rng.uniform(*ECHO_RT60_S))
    rir = synthetic_rir(rng, rt60)
    n = len(wave) + len(rir) - 1
    size = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(wave, size) * np.fft.rfft(rir, size), size)[:len(wave)]
    # keep the loudness of the input: the model should see reverberation, not a level change
    out *= np.sqrt(np.mean(wave.astype(np.float64) ** 2) + 1e-12) / (np.sqrt(np.mean(out ** 2)) + 1e-12)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def apply_noise(wave, rng):
    snr_db = float(rng.uniform(*NOISE_SNR_DB))
    power = float(np.mean(wave.astype(np.float64) ** 2)) + 1e-12
    noise = rng.normal(0.0, np.sqrt(power / 10.0 ** (snr_db / 10.0)), len(wave))
    return np.clip(wave + noise, -1.0, 1.0).astype(np.float32)


def apply_condition(wave, name, rng, ffmpeg=None):
    """Apply one named condition. `rng` only matters for the synthetic ones."""
    if name == "echo":
        return apply_echo(np.asarray(wave, dtype=np.float32), rng)
    if name == "noise_lowsnr":
        return apply_noise(np.asarray(wave, dtype=np.float32), rng)
    if name not in CODEC_SPECS:
        raise KeyError(f"unknown condition {name!r}")
    return codec_roundtrip(wave, name, ffmpeg or find_ffmpeg())


def condition_rng(seed, position, name):
    """Deterministic generator for one (clip, condition): the same clip always gets the same echo/noise."""
    return np.random.default_rng([int(seed), int(position), sum(ord(c) * (i + 1) for i, c in enumerate(name))])


def resolve_conditions(ffmpeg, probe=None):
    """Which named conditions this machine can really produce, with every substitution made explicit.

    Returns {"training": [...], "held_out": [...], "substituted": {asked: used}, "unavailable": {name: why}}.
    `probe(name)` should raise if the condition cannot be produced; the default runs a short real round trip.
    """
    def default_probe(name):
        codec_roundtrip(np.random.default_rng(0).normal(0, 0.1, CODEC_SR // 2).astype(np.float32), name, ffmpeg)

    probe = probe or default_probe
    usable, substituted, unavailable = {}, {}, {}
    for name in ALL_CONDITIONS:
        if name in SYNTHETIC:
            usable[name] = name
            continue
        try:
            probe(name)
            usable[name] = name
        except Exception as exc:  # noqa: BLE001
            unavailable[name] = f"{type(exc).__name__}: {str(exc)[:120]}"
            fallback = SUBSTITUTES.get(name)
            if fallback:
                try:
                    probe(fallback)
                    usable[name] = fallback
                    substituted[name] = fallback
                except Exception as exc2:  # noqa: BLE001
                    unavailable[fallback] = f"{type(exc2).__name__}: {str(exc2)[:120]}"
    return {"training": [usable[n] for n in TRAINING_MENU if n in usable],
            "held_out": [usable[n] for n in HELD_OUT if n in usable],
            "substituted": substituted, "unavailable": unavailable}


def assign_conditions(n_clips, menu, per_clip=AUGMENTATIONS_PER_CLIP, seed=42):
    """[n_clips, per_clip] indices into `menu`: for each clip, `per_clip` distinct conditions, fixed by seed."""
    if len(menu) < per_clip:
        raise ValueError(f"the training menu has {len(menu)} usable conditions; {per_clip} are needed per clip")
    out = np.zeros((n_clips, per_clip), dtype=np.int16)
    for i in range(n_clips):
        out[i] = np.random.default_rng([int(seed), i]).choice(len(menu), size=per_clip, replace=False)
    return out
