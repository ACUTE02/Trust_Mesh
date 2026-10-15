"""CPU smoke run of notebooks/06_voice_codec_robust.ipynb on synthetic data (Phase 2D).

Executes every code cell of the *built* notebook, in order, in one namespace: first as a PILOT (must measure and
stop, training nothing), then as a FULL run with a tiny randomly initialised encoder and real ffmpeg codecs. The
numbers it prints are meaningless; what is checked is that every stage runs, the result file has the promised keys,
and the run holds no per-clip data.

Usage: .venv\Scripts\python.exe tests/smoke_notebook_06.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from build_notebooks import SRC_DIR, build_notebook, code_cells  # noqa: E402


def run_notebook(mode, out_dir):
    os.environ.update({"TRUSTMESH_SMOKE": "1", "TRUSTMESH_RUN_MODE": mode, "TRUSTMESH_OUTPUT_DIR": str(out_dir),
                       "TRUSTMESH_CACHE_DIR": str(out_dir / "cache"), "MPLBACKEND": "Agg"})
    namespace = {"__name__": "__main__"}
    for i, src in enumerate(code_cells(build_notebook(SRC_DIR / "06_voice_codec_robust.py"))):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
        except Exception:
            print(f"\nSMOKE FAILED in code cell {i}:\n{src[:400]}\n")
            raise
    return namespace


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="trustmesh_voice2d_smoke_"))
    start = time.time()

    run_notebook("PILOT", out_dir)
    pilot = json.loads((out_dir / "results_phase2d.json").read_text(encoding="utf-8"))
    assert "pilot" in pilot and "evaluation" not in pilot, "the pilot must measure and stop"
    for key in ("seconds_per_clip", "projected_extraction_hours", "projected_cache_gb", "free_disk_gb", "seconds_split"):
        assert key in pilot["pilot"], f"pilot is missing {key}"
    assert not list((out_dir).glob("phase2d_*.pt")), "the pilot must not train anything"

    run_notebook("FULL", out_dir)
    results = json.loads((out_dir / "results_phase2d.json").read_text(encoding="utf-8"))
    for key in ("split_identity", "plan", "extraction", "heads", "evaluation", "criteria"):
        assert key in results, f"results_phase2d.json missing {key}"
    ev = results["evaluation"]
    assert "clean_full_test" in ev and "pooled_training_menu_conditions" in ev
    names = set(ev["conditions"])
    assert {"aac32", "echo", "noise_lowsnr", "mulaw8k"} <= names, names
    assert all(b["role"] in ("training_menu", "held_out_never_trained") for b in ev["conditions"].values())
    assert results["heads"]["F2_s42"]["n_params"] == results["heads"]["control_s42"]["n_params"]
    for seed in (42, 43, 44):
        assert (out_dir / f"phase2d_F2_s{seed}.pt").exists() and (out_dir / f"phase2d_control_s{seed}.pt").exists()
    text = json.dumps(results)
    assert "utterance_id" not in text and "LA_E_" not in text, "no clip identifiers may be written"
    assert not (out_dir / "cache").exists(), "the cache is deleted after a complete run"
    print(f"\nSMOKE OK in {time.time() - start:.0f}s; results at {out_dir}")


if __name__ == "__main__":
    main()
