"""Phase 2D, step A2: is Model F's simulated low-bitrate codec close to a real codec? Diagnosis only.

`engines.voice.preprocess.simulate_low_bitrate` (Model F's stand-in for MP3: 4x down/up-sampling plus mu-law
quantisation to 6 bits) is compared with real AAC 32 kb/s on genuine clips already on this machine: the 4
dataset clips and the 5 LibriSpeech clips (the files are read, never copied). For each clip the served Model D
head scores the original, the simulated version and the real-codec versions, and the drop in the raw score is
compared.

Pre-registered meaning of "close" (written before running): the mean score drop under the simulation lies
between half and twice the mean drop under real AAC 32 kb/s, AND every clip drops under both.

Run: .venv\\Scripts\\python.exe scripts/phase2d_check_simulation.py --folder E:\\TrustMesh\\my_voice_tests
Prints aggregates and per-clip drops for the public dataset clips only; writes data/self_test/phase2d_a2.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_phone_audio as d  # noqa: E402
from engines.voice.preprocess import simulate_low_bitrate  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="folder holding the LibriSpeech 84-*.flac files")
    folder = Path(parser.parse_args().folder)
    import imageio_ffmpeg
    d.FF = imageio_ffmpeg.get_ffmpeg_exe()
    scorer = d.Scorer()

    clips = {}
    for p in sorted((ROOT / "data" / "demo_clips").glob("asvspoof_bonafide_*.wav")):
        clips[f"dataset {p.stem.split('_')[-1]}"] = sf.read(p, dtype="float32")[0]
    for p in sorted(folder.glob("84-*.flac")):
        clips[f"librispeech {p.stem[-4:]}"] = d.load16(p)
    assert len(clips) == 9, f"expected 9 genuine clips, found {len(clips)}"

    generator = torch.Generator().manual_seed(0)
    rows = {}
    for name, wave in clips.items():
        base = scorer.score(wave)["logit"]
        prepared = torch.from_numpy(scorer.prepare(wave)).unsqueeze(0)             # the 4 s clip Model F augmented
        sim_wave = simulate_low_bitrate(prepared, generator, p_apply=1.0)[0].numpy()
        sim = scorer.score(sim_wave, mode="none_start")["logit"]
        aac = scorer.score(d.reencode(_as_wav(name, wave), "aac_32", False, _tmp()))["logit"]
        aac48 = scorer.score(d.reencode(_as_wav(name, wave), "aac_32", True, _tmp()))["logit"]
        rows[name] = {"original": base, "simulated": sim, "aac32": aac, "aac32_via48k": aac48}

    drop = lambda key: np.array([rows[n]["original"] - rows[n][key] for n in rows])  # noqa: E731
    sim_drop, aac_drop, aac48_drop = drop("simulated"), drop("aac32"), drop("aac32_via48k")
    ratio = float(sim_drop.mean() / aac_drop.mean()) if aac_drop.mean() else float("nan")
    close = bool(0.5 <= ratio <= 2.0 and (sim_drop > 0).all() and (aac_drop > 0).all())
    rank = float(np.corrcoef(np.argsort(np.argsort(sim_drop)), np.argsort(np.argsort(aac_drop)))[0, 1])
    result = {
        "n_clips": len(rows), "metric": "drop in the raw (bonafide-logit) score; larger = looks more fake",
        "mean_original_score": float(np.mean([r["original"] for r in rows.values()])),
        "mean_drop_simulated": float(sim_drop.mean()), "mean_drop_real_aac32": float(aac_drop.mean()),
        "mean_drop_real_aac32_via48k": float(aac48_drop.mean()),
        "ratio_simulated_to_real_aac32": ratio, "rank_correlation_of_per_clip_drops": rank,
        "clips_that_drop_simulated": int((sim_drop > 0).sum()), "clips_that_drop_real_aac32": int((aac_drop > 0).sum()),
        "bands_after_simulation": {b: sum(scorer_band(r["simulated"], scorer) == b for r in rows.values()) for b in ("low", "medium", "high")},
        "bands_after_real_aac32": {b: sum(scorer_band(r["aac32"], scorer) == b for r in rows.values()) for b in ("low", "medium", "high")},
        "close_by_pre_registered_rule": close,
    }
    out = ROOT / "data" / "self_test" / "phase2d_a2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"result": result, "per_clip": rows}, indent=2))
    print(json.dumps(result, indent=1))


_TMP = None


def _tmp():
    global _TMP
    import tempfile
    if _TMP is None:
        _TMP = Path(tempfile.mkdtemp(prefix="tm_a2_"))
    return _TMP


def _as_wav(name, wave):
    path = _tmp() / (name.replace(" ", "_") + ".wav")
    sf.write(path, wave, d.SR, subtype="PCM_16")
    return path


def scorer_band(logit, scorer):
    p = 1.0 / (1.0 + np.exp(logit))
    return "high" if p >= scorer.thr["high"] else "medium" if p >= scorer.thr["medium"] else "low"


if __name__ == "__main__":
    main()
