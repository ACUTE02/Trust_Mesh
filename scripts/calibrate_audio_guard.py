"""How often does each audio-guard check fire? (serving/audio_guard.py). No training, no model, no network.

Measures the guard on every local clip, on re-encoded copies of them, on speech with added noise, and on
synthetic non-speech, and prints COUNTS per group. Nothing per clip is printed for the owner's recordings.
Per-group minima/maxima are printed so the thresholds in serving/audio_guard.py can be checked against them.

Clean clips : the 8 dataset clips in data/demo_clips (4 genuine, 4 fake) and 5 LibriSpeech clips.
Compressed  : the same 13 clips re-encoded as AAC 32 kb/s, AAC 32 kb/s through 48 kHz, MP3 128 kb/s, Opus 24 kb/s
              and 8 kHz mu-law telephony (all through ffmpeg).
Noisy       : the 13 clean clips with white noise added at 20, 10, 5 and 0 dB signal-to-noise.
Owner       : the owner's own recordings (3 genuine voice memos, 3 voice clones), read from --folder.

Run: .venv\\Scripts\\python.exe scripts/calibrate_audio_guard.py --folder <your-own-clips-folder>
Results (aggregates) are written to data/self_test/audio_guard_calibration.json (git-ignored).
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engines.voice.preprocess import trim_silence  # noqa: E402
from serving import audio_guard as g  # noqa: E402

SR = 16000
FF = None


def ffmpeg(*args):
    done = subprocess.run([FF, "-y", "-loglevel", "error", *map(str, args)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip()[:200])


def load16(path):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "x.wav"
        ffmpeg("-i", path, "-ac", 1, "-ar", SR, "-c:a", "pcm_s16le", out)
        return sf.read(out, dtype="float32")[0]


CODECS = {"AAC 32 kb/s": (".m4a", ["-c:a", "aac", "-b:a", "32k"], False),
          "AAC 32 kb/s through 48 kHz": (".m4a", ["-c:a", "aac", "-b:a", "32k"], True),
          "MP3 128 kb/s": (".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"], False),
          "Opus 24 kb/s": (".opus", ["-c:a", "libopus", "-b:a", "24k"], False),
          "8 kHz mu-law telephony": (".wav", ["-ar", 8000, "-c:a", "pcm_mulaw"], False)}


def recode(wave, name):
    ext, args, via48 = CODECS[name]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / "s.wav"
        sf.write(src, wave, SR, subtype="PCM_16")
        cur = src
        if via48:
            cur = d / "u.wav"
            ffmpeg("-i", src, "-ar", 48000, "-c:a", "pcm_s16le", cur)
        coded = d / f"c{ext}"
        ffmpeg("-i", cur, *args, coded)
        back = d / "b.wav"
        ffmpeg("-i", coded, "-ac", 1, "-ar", SR, "-c:a", "pcm_s16le", back)
        return sf.read(back, dtype="float32")[0]


def measure(wave, native_rate=None):
    trimmed, _ = trim_silence(np.asarray(wave, dtype=np.float32), SR, 30.0)
    return g.check(trimmed, native_rate=native_rate)


def as_uploaded_8k(wave):
    """A true 8 kHz telephone file as the service sees it: decoded at 8 kHz, then up-sampled by interpolation."""
    from engines.voice.preprocess import resample_linear
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src, out = d / "s.wav", d / "t.wav"
        sf.write(src, wave, SR, subtype="PCM_16")
        ffmpeg("-i", src, "-ar", 8000, "-c:a", "pcm_mulaw", out)
        w8, rate = sf.read(out, dtype="float32")
    return resample_linear(w8, rate)[0], rate


def synthetic():
    rng = np.random.default_rng(1)
    t = np.arange(4 * SR) / SR

    def pink(n):
        spec = np.fft.rfft(rng.normal(size=n))
        k = np.arange(len(spec))
        k[0] = 1
        y = np.fft.irfft(spec / np.sqrt(k), n)
        return (y / np.abs(y).max() * 0.5).astype(np.float32)

    return {
        "white noise (loud)": np.clip(rng.normal(0, .5, len(t)), -1, 1), "white noise (quiet)": rng.normal(0, .05, len(t)),
        "pink noise": pink(len(t)), "pure tone 170 Hz": .3 * np.sin(2 * np.pi * 170 * t),
        "pure tone 1 kHz": .3 * np.sin(2 * np.pi * 1000 * t),
        "tone with 4 Hz tremolo": .3 * np.sin(2 * np.pi * 170 * t) * (1 + .6 * np.sin(2 * np.pi * 4 * t)),
        "three-note chord": .2 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t)),
        "square wave 200 Hz": .3 * np.sign(np.sin(2 * np.pi * 200 * t)),
        "tone plus noise": .3 * np.sin(2 * np.pi * 170 * t) + rng.normal(0, .05, len(t)),
        "frequency sweep 100-3000 Hz": .3 * np.sin(2 * np.pi * (100 * t + (3000 - 100) / 8 * t ** 2)),
    }, rng


def main():
    global FF
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="folder with the LibriSpeech *.flac files and real/ and fake/ recordings")
    folder = Path(parser.parse_args().folder)
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()

    clean = {}
    for p in sorted((ROOT / "data" / "demo_clips").glob("*.wav")):
        clean[f"dataset {p.stem.replace('asvspoof_', '')}"] = sf.read(p, dtype="float32")[0]
    for p in sorted(folder.glob("84-*.flac")):
        clean[f"librispeech {p.stem[-4:]}"] = load16(p)
    groups = {"clean": [measure(w) for w in clean.values()]}
    for codec in CODECS:
        groups[codec] = [measure(recode(w, codec)) for w in clean.values()]
    uploaded = [as_uploaded_8k(w) for w in clean.values()]
    groups["8 kHz mu-law file, as uploaded (rule 3)"] = [measure(w, native_rate=r) for w, r in uploaded]
    groups["same, spectral rules only (rule 3 off)"] = [measure(w) for w, _ in uploaded]
    synth, rng = synthetic()
    for snr in (20, 10, 5, 0):
        noisy = []
        for w in clean.values():
            trimmed, _ = trim_silence(w, SR, 30.0)
            sigma = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2)) / 10 ** (snr / 20)
            noisy.append(measure(np.clip(w + rng.normal(0, sigma, len(w)).astype(np.float32), -1, 1)))
        groups[f"speech + white noise at {snr} dB SNR"] = noisy
    groups["owner genuine voice memos"] = [measure(load16(p)) for p in sorted((folder / "real").iterdir())]
    groups["owner voice clones"] = [measure(load16(p)) for p in sorted((folder / "fake").iterdir())]
    non_speech = {k: measure(w.astype(np.float32)) for k, w in synth.items()}
    non_speech["silence (zeros)"] = g.check(np.zeros(SR, dtype=np.float32))

    def stats(items, key):
        vals = [i[key] for i in items if i.get(key) is not None]
        return {"min": float(min(vals)), "median": float(np.median(vals)), "max": float(max(vals))} if vals else None

    report = {"thresholds": {"spectral_movement_min": g.SPEECH_MOVEMENT_MIN, "hf_to_mid_min_db": g.HF_TO_MID_MIN_DB,
                             "cutoff_min_hz": g.CUTOFF_MIN_HZ, "min_native_rate_hz": g.MIN_NATIVE_RATE_HZ},
              "groups": {}, "non_speech": {}}
    print(f"{'group':42s} {'n':>3s} {'rejected as':>11s} {'band-limit':>10s}   spectral movement   hf/mid dB (median)")
    print(f"{'':42s} {'':>3s} {'not speech':>11s} {'warning':>10s}   min .. max          min .. max")
    for name, items in groups.items():
        rejected = sum(not i["speech_like"] for i in items)
        warned = sum(i["low_quality_audio"] for i in items)
        mv, hf = stats(items, "spectral_movement"), stats(items, "hf_to_mid_db")
        report["groups"][name] = {"n": len(items), "rejected_as_not_speech": rejected, "band_limit_warning": warned,
                                  "spectral_movement": mv, "hf_to_mid_db": hf, "cutoff_hz": stats(items, "cutoff_hz")}
        print(f"{name:42s} {len(items):3d} {rejected:11d} {warned:10d}   {mv['min']:.3f} .. {mv['max']:.3f}      "
              f"{hf['min']:.1f} .. {hf['max']:.1f}")
    print("\nSynthetic non-speech (should be rejected):")
    for name, r in non_speech.items():
        report["non_speech"][name] = {"rejected": not r["speech_like"], "spectral_movement": r["spectral_movement"]}
        print(f"  {name:32s} spectral movement {r['spectral_movement']:.3f}  ->  "
              f"{'REJECTED' if not r['speech_like'] else 'passes as speech'}")
    out = ROOT / "data" / "self_test" / "audio_guard_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
