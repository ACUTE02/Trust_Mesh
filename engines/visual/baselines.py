"""Shortcut baselines (Task 4): can the label be guessed without looking at the picture?

A tiny classifier is trained on file properties only, and another on the source dataset only. If
either separates real from fake, a real model would use the same shortcut, so the pipeline has to
be fixed before any encoder runs.

The rule for "separates" is the confidence interval, not a cut-off someone picked: the baseline
passes only if the 95% bootstrap CI of its ROC-AUC includes 0.5 (same reasoning as issue #16).

Three feature sets:
  raw      -- the properties of each file as received (width, height, aspect ratio, format, JPEG
              quality, file size, EXIF present). Expected to fail: that is why the shared pipeline exists.
  pipeline -- the same seven properties after the shared pipeline. All of them are then identical
              for every image except the byte count of the re-encoded JPEG, so this measures what
              shortcut is left for the model.
  source   -- the source dataset alone.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import fast_roc_auc
from .evaluate import _resample_indices
from .preprocess import CLIP_SIZE
from .preprocess import STANDARD_JPEG_QUALITY

SHORTCUT_COLUMNS = ("width", "height", "aspect_ratio", "format", "jpeg_quality", "file_size", "has_exif")
SHORTCUT_MODES = ("raw", "pipeline", "source")


def shortcut_matrix(table, mode, categories=None):
    """(X, feature names, categories) for one feature set. `categories` fixes the one-hot columns
    so train and test matrices line up."""
    n = len(table)
    if mode == "source":
        values = table["source_dataset"].astype(str).to_numpy()
        categories = sorted(set(values)) if categories is None else categories
        return (np.array([[float(v == c) for c in categories] for v in values]).reshape(n, len(categories)),
                [f"source={c}" for c in categories], categories)
    if mode == "pipeline":
        width = height = np.full(n, float(CLIP_SIZE))
        formats = np.array(["JPEG"] * n)
        quality = np.full(n, float(STANDARD_JPEG_QUALITY))
        size = table["pipeline_file_size"].astype(float).to_numpy()
        exif = np.zeros(n)
    elif mode == "raw":
        width, height = table["width"].astype(float).to_numpy(), table["height"].astype(float).to_numpy()
        formats = table["format"].astype(str).to_numpy()
        quality = table["jpeg_quality"].astype(float).fillna(-1.0).to_numpy()
        size = table["file_size"].astype(float).to_numpy()
        exif = table["has_exif"].astype(float).fillna(0.0).to_numpy()
    else:
        raise ValueError(f"unknown shortcut feature set: {mode}")
    categories = sorted(set(formats)) if categories is None else categories
    columns = [width, height, width / np.maximum(height, 1.0), quality, np.log1p(size), exif]
    names = ["width", "height", "aspect_ratio", "jpeg_quality", "log_file_size", "has_exif"]
    for c in categories:
        columns.append((formats == c).astype(float))
        names.append(f"format={c}")
    return np.column_stack(columns), names, categories


def _fit_predict(kind, x_train, y_train, x_test, seed):
    if kind == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                              random_state=seed, n_jobs=2, eval_metric="logloss")
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def shortcut_baseline(table, mode, kind="logreg", train_split="train", test_split="known_test",
                      n_resamples=1000, seed=0):
    """Train on one partition, score another, and report ROC-AUC with a stratified bootstrap CI.
    `separable` is True when the CI excludes 0.5 in either direction."""
    live = table[table["valid"].astype(bool)]
    train, test = live[live["split"] == train_split], live[live["split"] == test_split]
    result = {"mode": mode, "kind": kind, "n_train": int(len(train)), "n_test": int(len(test)),
              "auc": None, "ci_low": None, "ci_high": None, "separable": None}
    y_train, y_test = train["is_fake"].astype(int).to_numpy(), test["is_fake"].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        result["note"] = "a split has only one class; the baseline cannot be trained or scored"
        return result
    x_train, names, categories = shortcut_matrix(train, mode)
    x_test, _, _ = shortcut_matrix(test, mode, categories)
    scores = _fit_predict(kind, x_train, y_train, x_test, seed)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = _resample_indices(y_test, rng)
        samples.append(fast_roc_auc(scores[idx], y_test[idx]))
    low, high = np.quantile(samples, [0.025, 0.975])
    result.update(auc=fast_roc_auc(scores, y_test), ci_low=float(low), ci_high=float(high),
                  separable=bool(low > 0.5 or high < 0.5), features=names, n_resamples=int(n_resamples))
    return result


def run_shortcut_baselines(table, n_resamples=1000, seed=0, kinds=("logreg", "xgboost")):
    """Every feature set with every model kind, plus the gate. The gate looks at what the model
    will actually receive -- the pipeline and source feature sets -- and passes only if neither
    is separable. The raw result is reported next to it as the size of the shortcut the pipeline
    removes."""
    results = [shortcut_baseline(table, mode, kind, n_resamples=n_resamples, seed=seed)
               for mode in SHORTCUT_MODES for kind in kinds]
    gated = [r for r in results if r["mode"] in ("pipeline", "source")]
    undecided = [r for r in gated if r["separable"] is None]
    failing = [r for r in gated if r["separable"]]
    gate = {"passed": not failing and not undecided,
            "failing": [f"{r['mode']}/{r['kind']} AUC {r['auc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                        for r in failing],
            "undecided": [f"{r['mode']}/{r['kind']}: {r.get('note')}" for r in undecided],
            "rule": "pass only if the 95% bootstrap CI of ROC-AUC includes 0.5 for every pipeline and source baseline"}
    return {"results": results, "gate": gate}


def source_class_table(table):
    """Per source dataset: real and fake counts, and the most common sizes and formats -- the
    source-versus-label picture the pre-flight asks for."""
    live = table[table["valid"].astype(bool)]
    out = {}
    for source, rows in live.groupby("source_dataset"):
        sizes = (rows["width"].astype(int).astype(str) + "x" + rows["height"].astype(int).astype(str)).value_counts()
        out[str(source)] = {"real": int((rows["is_fake"] == 0).sum()), "fake": int((rows["is_fake"] == 1).sum()),
                            "top_sizes": {str(k): int(v) for k, v in sizes.head(5).items()},
                            "formats": {str(k): int(v) for k, v in rows["format"].value_counts().items()}}
    return out
