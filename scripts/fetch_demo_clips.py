"""Download a few real labelled clips for the demo (no live microphone pipeline exists).

Reads a handful of ASVspoof 2019 LA clips (the dataset Models C/D were trained on) straight out of one
Hugging Face parquet shard with range requests -- only the first few ~6 MB row groups are fetched, not
the 486 MB shard -- and writes them as WAV into
data/demo_clips/ with a manifest.json holding each clip's true label.

Optional `--in-the-wild N` adds out-of-domain clips; that dataset is a ~7 GB zip, so it needs the
`datasets` package and a fast connection and is off by default.

Run: .venv\\Scripts\\python.exe scripts/fetch_demo_clips.py
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving import config  # noqa: E402

SHARD = "datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA/data/test-00000-of-00009.parquet"
LABEL_NAMES = {0: "bonafide", 1: "spoof"}  # the dataset's ClassLabel: 0 = bonafide, 1 = spoof


def asvspoof_clips(per_class, seed):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    pf = pq.ParquetFile(HfFileSystem().open(SHARD, "rb"))
    rng = np.random.default_rng(seed)
    wanted, picked = {0: per_class, 1: per_class}, []
    # ~10% of clips are bonafide, so one or two 100-row groups (~6 MB each) hold enough of both classes.
    # (Scanning every group's label column first looks cheaper but makes fsspec fetch ~5 MB per seek.)
    for rg in rng.permutation(min(pf.num_row_groups, 8)):
        for row in pf.read_row_group(int(rg), columns=["path", "audio", "label"]).to_pylist():
            if wanted[row["label"]] > 0:
                wanted[row["label"]] -= 1
                picked.append(row)
        if not any(wanted.values()):
            break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--in-the-wild", type=int, default=0, metavar="N")
    args = parser.parse_args()

    out = config.DEMO_CLIPS_DIR
    out.mkdir(parents=True, exist_ok=True)
    manifest, counts = [], {0: 0, 1: 0}
    for row in asvspoof_clips(args.per_class, args.seed):
        label = row["label"]
        counts[label] += 1
        wave, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        name = f"asvspoof_{LABEL_NAMES[label]}_{counts[label]}.wav"
        sf.write(out / name, wave, sr)
        manifest.append({"file": name, "dataset": "ASVspoof 2019 LA (in-domain)",
                         "true_label": LABEL_NAMES[label], "utterance_id": row["path"]})
        print("wrote", name)

    if args.in_the_wild:
        from engines.voice.data import BONAFIDE, load_in_the_wild_clips
        for i, clip in enumerate(load_in_the_wild_clips(args.in_the_wild, seed=args.seed)):
            truth = "bonafide" if clip["label"] == BONAFIDE else "spoof" if clip["label"] is not None else "unknown"
            name = f"inthewild_{truth}_{i + 1}.wav"
            sf.write(out / name, clip["waveform"], clip["sample_rate"])
            manifest.append({"file": name, "dataset": "In-the-Wild (out-of-domain)", "true_label": truth})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} clips to {out}")


if __name__ == "__main__":
    main()
