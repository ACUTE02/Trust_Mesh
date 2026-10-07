"""Plain-language check of the files a fresh clone does not contain.

The repository holds no dataset, no model weights and no encoder (licences, size). When one is missing the
services used to stop with a long traceback. `report()` says, in words, what is missing, where it goes and
where it comes from. Nothing here downloads or changes anything.
"""
from . import config

KAGGLE = "the Output panel of the Kaggle notebook that trained it (see docs/kaggle_setup.md)"


def _items():
    graph, voice = config.ARTIFACTS_DIR / "graph", config.ARTIFACTS_DIR / "voice"
    encoder = config.DATA_DIR / "models" / "wav2vec2-xls-r-300m"
    elliptic = config.ELLIPTIC_ROOT / "raw" / "elliptic_txs_features.csv"
    processed = config.ELLIPTIC_ROOT / "processed" / "data.pt"
    return [
        # (group, required, path(s) of which ONE must exist, what it is, where it comes from)
        ("graph", True, [graph / "xgboost_tier1_tier2.json"], "the graph scoring model (XGBoost weights)",
         f"notebook 01: {KAGGLE}"),
        ("graph", True, [graph / "gat_weights.pt"], "the graph explanation model (GAT weights)", f"notebook 01: {KAGGLE}"),
        ("graph", False, [elliptic, processed], "the Elliptic dataset",
         "downloaded automatically on first start (about 700 MB, needs internet); or copy the three "
         "elliptic_txs_*.csv files from the Kaggle dataset ellipticco/elliptic-data-set into data/elliptic/raw/"),
        ("voice", True, [voice / "phase2c_D_ssl_mean_only_P1_s42.pt", voice / "phase2b_C_ssl_P1_s42.pt"],
         "a voice model head (Model D, or Model C as the fallback)", f"notebook 04: {KAGGLE}"),
        ("voice", False, [voice / "phase2c_test_scores.npz"], "Model D's saved test scores (used to set the warning levels)",
         f"notebook 04: {KAGGLE}; then run scripts/calibrate_voice_thresholds.py"),
        ("voice", False, [encoder / "pytorch_model.bin"], "the speech encoder (wav2vec2 XLS-R 300M, 1.27 GB)",
         "downloaded automatically from Hugging Face on first start (needs internet); or put config.json and "
         "pytorch_model.bin from facebook/wav2vec2-xls-r-300m into data/models/wav2vec2-xls-r-300m/"),
    ]


def missing(group=None):
    """[(required, path, what, source)] for every item of `group` (or all) with none of its files present."""
    rows = []
    for g, required, paths, what, source in _items():
        if group in (None, g) and not any(p.exists() for p in paths):
            rows.append((required, paths[0], what, source))
    return rows


def _relative(path):
    for base, name in ((config.ARTIFACTS_DIR, "artifacts"), (config.DATA_DIR, "data")):
        try:
            return f"{name}/{path.relative_to(base).as_posix()}"
        except ValueError:
            continue
    return str(path)


def report(group=None):
    """(text, blocking): a message listing what is missing, and whether any of it is required."""
    rows = missing(group)
    if not rows:
        return "", False
    lines = ["", "TrustMesh: some files this repository does not contain are missing.",
             "A fresh clone has no dataset, no model weights and no speech encoder. Put them here:", ""]
    for required, path, what, source in rows:
        lines += [f"  {'REQUIRED' if required else 'optional'}: {_relative(path)}", f"      what: {what}", f"      from: {source}", ""]
    lines.append("The full list is in README.md under \"Files you must supply\".")
    return "\n".join(lines), any(r[0] for r in rows)
