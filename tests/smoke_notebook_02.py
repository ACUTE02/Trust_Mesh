"""End-to-end CPU smoke run of notebooks/02_graph_full.ipynb on synthetic data.

Executes every code cell of the *built* notebook, in order, in one namespace -- the same code Kaggle
runs, except that TRUSTMESH_SMOKE=1 swaps in a tiny synthetic graph and small epoch/trial budgets.
Catches wiring bugs (undefined names, shape errors, broken result saving) before spending GPU time.
Numbers it prints are meaningless and must never be reported.

Usage: python tests/smoke_notebook_02.py
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
    out_dir = Path(tempfile.mkdtemp(prefix="trustmesh_smoke_"))
    os.environ["TRUSTMESH_SMOKE"] = "1"
    os.environ["TRUSTMESH_OUTPUT_DIR"] = str(out_dir)
    namespace = {"__name__": "__main__"}
    cells = code_cells(build_notebook(SRC_DIR / "02_graph_full.py"))
    start = time.time()
    for i, src in enumerate(cells):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
        except Exception:
            print(f"\nSMOKE FAILED in code cell {i}:\n{src[:400]}\n")
            raise
    results = json.loads((out_dir / "results_phase1d.json").read_text(encoding="utf-8"))
    expected = ["setup", "reproduction", "models", "rgcn", "camouflage_before_after", "expanded_search",
                "robustness_best_gnn", "closing_paragraph", "standing_finding_changed"]
    missing = [k for k in expected if k not in results]
    assert not missing, f"results_phase1d.json missing keys: {missing}"
    assert results["smoke_test"] is True
    print(f"\nSMOKE OK: {len(cells)} code cells in {time.time() - start:.0f}s; results at {out_dir}")


if __name__ == "__main__":
    main()
