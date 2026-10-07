"""End-to-end CPU smoke run of notebooks/03_voice_baseline.ipynb on synthetic audio.

Executes every code cell of the *built* notebook, in order, in one namespace -- the same code Kaggle
runs, except that TRUSTMESH_SMOKE=1 swaps in a tiny synthetic ASVspoof-shaped set and small budgets.
Catches wiring bugs before spending GPU time. Numbers it prints are meaningless and must never be
reported.

Usage: .venv\\Scripts\\python.exe tests/smoke_notebook_03.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_notebooks import SRC_DIR, build_notebook, code_cells  # noqa: E402


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="trustmesh_voice_smoke_"))
    os.environ["TRUSTMESH_SMOKE"] = "1"
    os.environ["TRUSTMESH_OUTPUT_DIR"] = str(out_dir / "artifacts")
    os.environ["TRUSTMESH_DOCS_DIR"] = str(out_dir / "docs")
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace = {"__name__": "__main__"}
    cells = code_cells(build_notebook(SRC_DIR / "03_voice_baseline.py"))
    start = time.time()
    for i, src in enumerate(cells):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
        except Exception:
            print(f"\nSMOKE FAILED in code cell {i}:\n{src[:400]}\n")
            raise

    results = json.loads((out_dir / "artifacts" / "results.json").read_text(encoding="utf-8"))
    expected = ["eda", "feature_config", "silence_leakage", "splits", "models", "evaluation",
                "cross_dataset_spot_check", "closing_paragraph"]
    missing = [k for k in expected if k not in results]
    assert not missing, f"results.json missing keys: {missing}"
    assert results["smoke_test"] is True
    assert (out_dir / "docs" / "voice_data_exploration.md").exists(), "EDA doc not written"
    assert (out_dir / "artifacts" / "det_curves.png").exists(), "DET plot not written"
    for name in ("lcnn_trimmed_speaker_disjoint", "lcnn_untrimmed_speaker_disjoint",
                 "lcnn_trimmed_random_split"):
        assert name in results["models"], f"missing model result: {name}"
        assert (out_dir / "artifacts" / f"{name}.pt").exists(), f"missing weights: {name}"
    assert results["splits"]["speaker_disjoint"]["shared_speakers"] == 0, "headline split leaks speakers"
    print(f"\nSMOKE OK: {len(cells)} code cells in {time.time() - start:.0f}s; results at {out_dir}")


if __name__ == "__main__":
    main()
