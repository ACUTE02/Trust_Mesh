"""Compression test on clean lossless clips of one other speaker. Diagnosis only: no training, no model,
threshold or served-code change.

Takes the FLAC files given on the command line, scores each as it is, then scores copies re-encoded as
AAC 32 kb/s, AAC 32 kb/s after a 48 kHz step, and MP3 128 kb/s. Scoring is the same in-process path as
scripts/diagnose_phone_audio.py (the service's own code and weights). The re-encoded copies go to the
folder given with --work, which must be outside the repository.

Run: .venv\\Scripts\\python.exe scripts/diagnose_other_speaker.py --work <folder outside the repo> a.flac b.flac ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_phone_audio as d  # noqa: E402

CONDITIONS = [("original (lossless)", None, False), ("AAC 32 kb/s", "aac_32", False),
              ("AAC 32 kb/s after a 48 kHz step", "aac_32", True), ("MP3 128 kb/s", "mp3_128", False)]
SHORT_SECONDS = 4.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    work = Path(args.work).resolve()
    assert ROOT not in work.parents and work != ROOT, "--work must be outside the repository"
    work.mkdir(parents=True, exist_ok=True)

    import imageio_ffmpeg
    d.FF = imageio_ffmpeg.get_ffmpeg_exe()
    scorer = d.Scorer()

    clips, rows = [], {}
    for f in map(Path, args.files):
        info = sf.info(str(f))
        wave = d.load16(f)
        src = work / f"{f.stem}.wav"          # a 16 kHz WAV copy; the original file is only read
        sf.write(src, wave, d.SR, subtype="PCM_16")
        trimmed, _ = d.trim_silence(wave, d.SR, 30.0)
        clips.append({"file": f.name, "native_rate_hz": info.samplerate, "duration_s": round(info.frames / info.samplerate, 2),
                      "under_4_s": info.frames / info.samplerate < SHORT_SECONDS, "sound_after_trim_s": round(len(trimmed) / d.SR, 2)})
        rows[f.name] = {}
        for name, codec, via48 in CONDITIONS:
            w = wave if codec is None else d.reencode(src, codec, via48, work)
            rows[f.name][name] = scorer.score(w)

    summary = {}
    for name, _, _ in CONDITIONS:
        got = [rows[c["file"]][name] for c in clips]
        summary[name] = {"n": len(got), "low": sum(g["band"] == "low" for g in got), "medium": sum(g["band"] == "medium" for g in got),
                         "high": sum(g["band"] == "high" for g in got), "mean_raw_score": float(np.mean([g["logit"] for g in got])),
                         "min_raw_score": float(min(g["logit"] for g in got)), "max_raw_score": float(max(g["logit"] for g in got)),
                         "mean_score": float(np.mean([g["p"] for g in got]))}
    base = CONDITIONS[0][0]
    for name, _, _ in CONDITIONS[1:]:
        drops = [rows[c["file"]][base]["logit"] - rows[c["file"]][name]["logit"] for c in clips]
        summary[name]["mean_drop_in_raw_score"] = float(np.mean(drops))
        summary[name]["clips_whose_raw_score_dropped"] = int(sum(x > 0 for x in drops))
    result = {"clips": clips, "thresholds": {"medium": scorer.thr["medium"], "high": scorer.thr["high"]}, "summary": summary}
    (work / "other_speaker_results.json").write_text(json.dumps({"result": result, "per_clip": rows}, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
