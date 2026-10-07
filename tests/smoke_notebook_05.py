"""End-to-end CPU smoke run of notebooks/05_visual_v1.ipynb on synthetic data.

Executes every code cell of the *built* notebook, in order, in one namespace -- the same code Kaggle
runs, except that TRUSTMESH_SMOKE=1 swaps in synthetic OpenFake-shaped Parquet shards read through
the same byte-range reader (on the local filesystem), an ArtiFact-shaped folder, an FFHQ-shaped
folder, and a tiny randomly-initialised CLIP (no download). Numbers it prints are meaningless.

Two runs:
  1. STAGE unset -> the default, `task0`: only the pre-flight may run, and nothing per-image may be written.
  2. STAGE = heads -> every task, including the flagged pixel baselines.

Usage: .venv\\Scripts\\python.exe tests/smoke_notebook_05.py
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
from engines.visual.data import make_synthetic_openfake  # noqa: E402
from engines.visual.data import write_synthetic_artifact  # noqa: E402
from engines.visual.data import write_synthetic_ffhq  # noqa: E402
from engines.visual.data import write_synthetic_openfake  # noqa: E402

FAKE_TOKEN = "hf_SMOKETESTTOKENdoNotLeak0123456789"


class Tee:
    """Passes everything printed through to the console and keeps a copy, so the test can check
    that the token was never printed."""

    def __init__(self, stream):
        self.stream, self.parts = stream, []

    def write(self, text):
        self.parts.append(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def text(self):
        return "".join(self.parts)


def run_notebook(out_dir, stage, inputs=None):
    inputs = inputs or out_dir
    os.environ.update({
        "TRUSTMESH_SMOKE": "1",
        "TRUSTMESH_OUTPUT_DIR": str(out_dir / "artifacts"),
        "TRUSTMESH_OPENFAKE_DIR": str(inputs / "openfake"),
        "TRUSTMESH_ARTIFACT_DIR": str(inputs / "artifact"),
        "TRUSTMESH_FFHQ_DIR": str(inputs / "ffhq"),
        "HF_TOKEN": FAKE_TOKEN,
        "MPLBACKEND": "Agg",
    })
    os.environ.pop("TRUSTMESH_VISUAL_STAGE", None)
    if stage:
        os.environ["TRUSTMESH_VISUAL_STAGE"] = stage
    namespace = {"__name__": "__main__"}
    cells = code_cells(build_notebook(SRC_DIR / "05_visual_v1.py"))
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        for i, src in enumerate(cells):
            try:
                exec(compile(src, f"<notebook cell {i}>", "exec"), namespace)
            except Exception:
                print(f"\nSMOKE FAILED in code cell {i} (stage {stage or 'default'}):\n{src[:400]}\n")
                raise
    finally:
        sys.stdout = tee.stream
    assert FAKE_TOKEN not in tee.text(), "the token was printed"
    return namespace, len(cells)


def no_secret_anywhere(folder):
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix in (".json", ".csv", ".txt", ".md"):
            assert FAKE_TOKEN not in path.read_text(encoding="utf-8", errors="replace"), f"the token leaked into {path}"


def main():
    start = time.time()
    base = Path(tempfile.mkdtemp(prefix="trustmesh_visual_smoke_"))

    def make_inputs(out_dir):
        write_synthetic_openfake(out_dir / "openfake", make_synthetic_openfake(n_prompts=70), n_shards=3, rows_per_group=16)
        write_synthetic_artifact(out_dir / "artifact", per_folder=12)
        write_synthetic_ffhq(out_dir / "ffhq", n=16)

    # ---- Run 1: the default stage is task0 and nothing else --------------------------------------
    first = base / "task0_only"
    make_inputs(first)
    namespace, _ = run_notebook(first, stage=None)
    assert namespace["STAGE"] == "task0", "the default stage must be task0"
    preflight = json.loads((first / "artifacts" / "preflight_task0.json").read_text(encoding="utf-8"))
    assert not preflight["failed_steps"], preflight["failed_steps"]
    for key in ("environment", "openfake_files", "openfake_counts", "streaming_sample", "quota_feasibility", "artifact",
                "ffhq", "source_vs_label", "clip", "disk_budget", "deepfake_eval_2024", "synthbuster", "hf_token_present"):
        assert key in preflight, f"preflight_task0.json missing {key}"
    assert preflight["hf_token_present"] is True
    assert preflight["openfake_counts"]["sparse_read_misses"] == 0, "metadata must be read by exact byte ranges"
    assert preflight["openfake_counts"]["label_values"].keys() == {"real", "fake"}
    assert preflight["openfake_counts"]["distinct_fake_generators"] == 9
    assert preflight["openfake_counts"]["prompt_groups_all"]["prompts_with_real_and_fake"] == 70
    assert preflight["streaming_sample"]["images"] > 0 and preflight["streaming_sample"]["seek_row_groups"]
    assert preflight["ffhq"]["image_files"] == 16 and preflight["ffhq"]["all_128x128_rgb"]
    assert preflight["artifact"]["fake_images"] == 24 and preflight["artifact"]["real_images"] == 24
    assert "ffhq" in preflight["source_vs_label"]["one_class_sources"]
    assert preflight["clip"]["determinism"]["identical"]
    # Within one grey level of the default official processor; exact against its PIL-backed variant when there is one.
    assert preflight["clip"]["max_abs_difference_vs_official_preprocessing"] <= preflight["clip"]["one_grey_level_in_normalized_units"] + 1e-3
    pil_difference = preflight["clip"]["max_abs_difference_vs_official_pil_preprocessing"]
    assert isinstance(pil_difference, str) or pil_difference < 1e-5
    private = first / "artifacts" / "private"
    assert not any(p.is_file() for p in private.rglob("*")), "task0 must not write any per-image file"
    results = json.loads((first / "artifacts" / "results_v1.json").read_text(encoding="utf-8"))
    assert "folds" not in results and "extract" not in results, "nothing after Task 0 may run at the default stage"
    assert "not yet run" in results["closing_paragraph"]
    assert all(v == "not yet run" for grid in results["report"]["folds"].values() for row in grid.values() for v in row.values())
    no_secret_anywhere(first / "artifacts")
    print(f"\nSMOKE (task0 only) OK in {time.time() - start:.0f}s\n" + "=" * 100)

    # ---- Run 2: every stage ------------------------------------------------------------------------
    second = base / "all_stages"
    make_inputs(second)
    namespace, n_cells = run_notebook(second, stage="heads")
    out = second / "artifacts"
    results = json.loads((out / "results_v1.json").read_text(encoding="utf-8"))
    assert results["smoke_test"] is True and "halted" not in results["not_run"], results["not_run"]
    for key in ("plan", "scan", "cleaning", "split", "extract", "folds", "old_only_ablation", "final_model",
                "pixel_baselines", "report", "closing_paragraph"):
        assert key in results, f"results_v1.json missing {key}"
    cleaning = results["cleaning"]["cleaning"]
    for reason in ("corrupt", "tiny", "grayscale"):
        assert cleaning["dropped_by_reason"].get(reason, 0) >= 1, f"the planted {reason} file was not dropped"
    split = results["split"]
    assert split["dedupe"]["removed"] >= 2, "the planted exact and near duplicates should be removed"
    assert not split["leakage_violations"]
    assert len(split["folds"]) >= 3, split["family_eligibility"]
    for fold, checks in split["leakage_per_fold"].items():
        assert checks.get("held_out_rows_in_train_val", 0) == 0 and checks["reserved_reals_in_train_val"] == 0
        assert not any(checks["shared_keys"].values()) and checks["phash_pairs_across_splits"] == 0
    raw = [r for r in split["shortcut_baselines"]["results"] if r["mode"] == "raw" and r["auc"] is not None]
    assert raw and all(r["separable"] for r in raw), "the planted file-property shortcut must be caught on raw properties"
    assert results["extract"]["complete"] and results["extract"]["determinism_check"]["identical"]
    assert results["extract"]["images_written_to_disk"] == 0
    for fold in split["folds"]:
        for kind in ("logreg", "mlp"):
            first_seed = results["folds"][fold][kind]["first_seed"]
            assert {"known", "unseen", "known/jpeg50", "unseen/jpeg90"} <= set(first_seed["suites"]), (fold, kind)
            metric = first_seed["suites"]["unseen"]["metrics"]["roc_auc"]
            assert metric["ci_low"] <= metric["value"] <= metric["ci_high"]
            assert "threshold" in first_seed["operating_point"] and first_seed["temperature"] > 0
        assert "roc_auc" in results["folds"][fold]["logreg"]["first_seed"]["known_minus_unseen"]
    assert results["final_model"]["weights_sha256"]["v1_final_logreg.npz"]
    assert "estimate" in results["final_model"]["what_it_cannot_claim"]
    assert set(results["pixel_baselines"]) == {"small_cnn", "resnet50"}
    for name in ("reliability_v1.png", "det_curves_v1.png", "per_generator_v1.csv"):
        assert (out / name).exists(), name
    for name in ("metadata.parquet", "predictions_v1.parquet", "weights/v1_final_logreg.npz"):
        assert (out / "private" / name).exists(), name
    image_files = [p for p in out.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".webp", ".bmp")]
    assert not image_files, f"no dataset image may be written to the output: {image_files[:3]}"
    no_secret_anywhere(out)

    # ---- Run 3: resume -- a second session reuses the first one's chunks ---------------------------
    os.environ["TRUSTMESH_RESUME_DIRS"] = str(out / "private")
    # Same inputs, new output folder -- as on Kaggle, where the data paths do not change between sessions.
    third = base / "resume"
    run_notebook(third, stage="extract", inputs=second)
    resumed = json.loads((third / "artifacts" / "results_v1.json").read_text(encoding="utf-8"))
    os.environ.pop("TRUSTMESH_RESUME_DIRS")
    assert resumed["scan"]["chunks_written"] == 0 and resumed["scan"]["chunks_reused"] == resumed["scan"]["chunks_total"]
    assert resumed["extract"]["chunks_done"] == 0 and resumed["extract"]["chunks_skipped"] == resumed["extract"]["chunks_total"]

    print(f"\nSMOKE OK: {n_cells} code cells, 3 runs (task0 only, all stages, resume) in {time.time() - start:.0f}s; outputs at {base}")


if __name__ == "__main__":
    main()
