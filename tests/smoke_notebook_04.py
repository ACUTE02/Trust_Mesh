"""End-to-end CPU smoke run of notebooks/04_voice_depth.ipynb on synthetic data.

Executes every code cell of the *built* notebook, in order, in one namespace -- the same code Kaggle
runs, except that TRUSTMESH_SMOKE=1 swaps in synthetic ASVspoof-shaped clips, a synthetic protocol
file, a tiny randomly-initialised wav2vec2 (no download), and a local In-the-Wild-shaped zip read
through the same fsspec zip code path as the real hf:// archive. Numbers it prints are meaningless.

Usage: .venv\\Scripts\\python.exe tests/smoke_notebook_04.py
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
from engines.voice.data import write_synthetic_archive  # noqa: E402


def main():
    out_dir = Path(tempfile.mkdtemp(prefix="trustmesh_voice2b_smoke_"))
    archive = write_synthetic_archive(out_dir / "release_in_the_wild.zip", n_real=5, n_fake=5)
    os.environ.update({
        "TRUSTMESH_SMOKE": "1",
        "TRUSTMESH_OUTPUT_DIR": str(out_dir / "artifacts"),
        "TRUSTMESH_CACHE_DIR": str(out_dir / "cache"),
        "TRUSTMESH_ITW_ARCHIVE": archive,
        "TRUSTMESH_ITW_PROTOCOL": "file",
        "MPLBACKEND": "Agg",
    })
    namespace = {"__name__": "__main__"}
    cells = code_cells(build_notebook(SRC_DIR / "04_voice_depth.py"))
    start = time.time()
    for i, src in enumerate(cells):
        try:
            exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
        except Exception:
            print(f"\nSMOKE FAILED in code cell {i}:\n{src[:400]}\n")
            raise

    results = json.loads((out_dir / "artifacts" / "results_phase2b.json").read_text(encoding="utf-8"))
    expected = ["data", "silence_stats", "splits", "models", "ssl", "model_selection", "attack_disjoint",
                "silence_ablation", "leaderboard", "ci_shrinkage", "in_the_wild", "closing_paragraph"]
    missing = [k for k in expected if k not in results]
    assert not missing, f"results_phase2b.json missing keys: {missing}"
    assert results["smoke_test"] is True
    assert results["data"]["attacks_available"], "synthetic protocol should verify"
    assert results["splits"]["P1"]["shared_speakers_train_test"] == 0
    assert not results["splits"]["P2"]["attack_overlap_train_test"], "P2 must hold attacks out"
    assert results["in_the_wild"]["status"] == "ok", results["in_the_wild"].get("error")
    assert results["in_the_wild"]["report"]["failed"] >= 1, "the broken archive member should be recorded"
    groups = {row["group"] for row in results["leaderboard"]}
    for group in ("A_lcnn_trimmed", "B_lcnn_trimmed_aug", "C_ssl_frozen", "U_lcnn_untrimmed_SHORTCUT",
                  "S_zeros_fixed", "S_noise_fixed", "S_zeros_original_duration", "S_silence_only"):
        assert group in groups, f"missing leaderboard group {group}"
    a_row = next(r for r in results["leaderboard"] if r["group"] == "A_lcnn_trimmed")
    assert len(a_row["seeds"]) == 3 and a_row["eer_std"] is not None
    assert (out_dir / "artifacts" / "det_curves_phase2b.png").exists()

    # Phase 2C (docs/PHASE_2C_VOICE_SSL_DEPTH.md): its own results file, never merged into Phase 2B's.
    results_2c = json.loads((out_dir / "artifacts" / "results_phase2c.json").read_text(encoding="utf-8"))
    expected_2c = ["preflight", "model_e_cache_cost", "leaderboard", "in_the_wild", "attack_disjoint",
                  "closing_paragraph"]
    missing_2c = [k for k in expected_2c if k not in results_2c]
    assert not missing_2c, f"results_phase2c.json missing keys: {missing_2c}"
    assert results_2c["smoke_test"] is True
    assert "phase2c" not in results.get("not_run", {}), "Phase 2C bookkeeping must not land in results_phase2b.json"
    groups_2c = {row["group"] for row in results_2c["leaderboard"]}
    for group in ("C_ssl_frozen", "D_ssl_mean_only", "E_ssl_bilstm", "F_ssl_augmented"):
        assert group in groups_2c, f"missing Phase 2C leaderboard group {group}"
    d_row = next(r for r in results_2c["leaderboard"] if r["group"] == "D_ssl_mean_only")
    assert len(d_row["seeds"]) == 3 and d_row["n_params"] is not None
    assert results_2c["preflight"]["model_c_pooling"] == "per-layer mean + std over frames"
    assert (out_dir / "artifacts" / "det_curves_phase2c.png").exists()
    for group in ("D_ssl_mean_only_P1_s42", "E_ssl_bilstm_P1_s42", "F_ssl_augmented_P1_s42"):
        assert (out_dir / "artifacts" / f"phase2c_{group}.pt").exists()

    assert not (out_dir / "cache").exists(), "cache should be deleted at the end of a successful run"
    print(f"\nSMOKE OK: {len(cells)} code cells in {time.time() - start:.0f}s; results at {out_dir}")


if __name__ == "__main__":
    main()
