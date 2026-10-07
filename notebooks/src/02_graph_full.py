# %% [markdown]
# # Phase 1D -- Graph Engine, Full Build: RGCN, CARE-GNN, Leaderboard, Expanded Search
#
# Implements `docs/PHASE_1D_GRAPH_ENGINE_FULL.md`. Generated from `notebooks/src/02_graph_full.py` by
# `scripts/build_notebooks.py` -- edit the source, not this .ipynb.
#
# 1. **Setup + reproduction check** -- same data loading, temporal split, class weighting, and
#    training protocol as `01_graph_baseline.ipynb`. XGBoost (Tier-1, Tier-1+2) and the Phase 1C
#    wide-search GraphSAGE are refit from their stored best hyperparameters, and checked against the
#    earlier runs' numbers, so the Phase 1D leaderboard compares like with like.
# 2. **RGCN** with label-free relation types (payment direction; direction x hub), plus an untyped
#    single-relation control on the same edges.
# 3. **CARE-GNN** (similarity-aware neighbor selection + the paper's threshold update), with a
#    before/after comparison on the Phase 1C camouflaged subset at a matched false-positive rate.
# 4. **Leaderboard** of every model trained in this notebook (the full Phase 1-1D leaderboard is
#    merged locally from both results files by `scripts/build_graph_leaderboard.py`).
# 5. **Expanded Optuna search** (up to 100 trials, time-boxed) on the best GNN.
# 6. **3-seed robustness check** on the final best GNN.
# 7. **Closing analysis** and save.
#
# Every task saves to `artifacts/graph/results_phase1d.json` as soon as it finishes.
#
# **Kaggle:** Accelerator = GPU, Internet = on, Run All. Expected runtime: roughly 1-5 hours, dominated
# by the expanded search, which is capped at 4 hours (`SEARCH_TIMEOUT_SEC`).

# %% [markdown]
# ## Setup

# %%
# Kaggle: Accelerator = GPU, Internet = on.
import os
import subprocess
import sys

# True only when tests/smoke_notebook_02.py runs this notebook's code on synthetic data on a CPU.
SMOKE_TEST = os.environ.get("TRUSTMESH_SMOKE") == "1"


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


if not SMOKE_TEST:
    for module_name, package in [("torch_geometric", "torch_geometric"), ("xgboost", "xgboost"),
                                 ("optuna", "optuna"), ("networkx", "networkx")]:
        try:
            __import__(module_name)
        except ImportError:
            pip_install(package)

# %%
import json
import math
import time
from pathlib import Path

import numpy as np
import optuna
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATA_ROOT = Path("./data/elliptic")
OUTPUT_DIR = Path(os.environ.get("TRUSTMESH_OUTPUT_DIR", "./artifacts/graph"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not SMOKE_TEST else "cpu")

DEFAULT_EPOCHS = 5 if SMOKE_TEST else 40          # "defaults" protocol, as for GraphSAGE/GAT/GIN in notebook 01
REPRO_REFIT_EPOCHS = 20 if SMOKE_TEST else 150    # notebook 01's early-stopped refit budget
REFIT_MAX_EPOCHS = 20 if SMOKE_TEST else 300      # early-stopped refits in this notebook (patience 15)
SEARCH_TRIALS = 3 if SMOKE_TEST else 100
SEARCH_MAX_EPOCHS = 20 if SMOKE_TEST else 300
SEARCH_TIMEOUT_SEC = 120 if SMOKE_TEST else 4 * 3600
ROBUSTNESS_SEEDS = [SEED, SEED + 1, SEED + 2]
CAMOUFLAGE_TARGET_FPR = 0.05

# Real numbers from earlier Kaggle runs (artifacts/graph/results.json). Used ONLY to check that this
# notebook reproduces notebook 01's setup; they are not Phase 1D results.
REFERENCE = {
    "split_counts": [26905, 4330, 15329],
    "pos_weight": 8.107989167230873,
    "xgboost_tier1": 0.7447529434085257,
    "xgboost_tier12": 0.7474345053830228,
    "graphsage_wide_search": 0.5770002559534624,
    "xgboost_tier12_3seed_mean": 0.7482402411555441,
    "xgboost_tier12_3seed_std": 0.003011179506437757,
    "graphsage_default_directed": 0.5114779015333686,
}
# Best config from notebook 01's 50-trial wider GraphSAGE search (Phase 1C).
GRAPHSAGE_WIDE_PARAMS = {"hidden_dim": 256, "lr": 0.001031859734651928, "dropout": 0.22808425766001517}

RESULTS_PATH = OUTPUT_DIR / "results_phase1d.json"
RESULTS = {"smoke_test": SMOKE_TEST}


def save_progress(key, value):
    RESULTS[key] = value
    RESULTS_PATH.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"[progress] saved '{key}' -> {RESULTS_PATH}")


optuna.logging.set_verbosity(optuna.logging.WARNING)
print("device:", DEVICE, "| SMOKE_TEST:", SMOKE_TEST, "| torch", torch.__version__)

# %% [markdown]
# ## Shared graph-engine code (inlined from `engines/graph/`)

# %% inline engines/graph/data.py

# %% inline engines/graph/features.py

# %% inline engines/graph/baseline.py

# %% inline engines/graph/relations.py

# %% inline engines/graph/gnn.py

# %% inline engines/graph/care_gnn.py

# %% inline engines/graph/train.py

# %% inline engines/graph/evaluate.py

# %% [markdown]
# ## 1. Setup and reproduction check

# %%
data = make_synthetic_elliptic() if SMOKE_TEST else load_elliptic(DATA_ROOT)
n_nodes, n_features = data.num_nodes, data.num_node_features
train_mask, val_mask, test_mask = temporal_masks(data)
counts = list(split_counts(train_mask, val_mask, test_mask))
pos_weight = class_pos_weight(data.y, train_mask)
time_stats = edge_time_stats(data.edge_index, data.time_step)

x_d, ei_d, y_d = data.x.to(DEVICE), data.edge_index.to(DEVICE), data.y.to(DEVICE)
train_d, val_d, test_d = train_mask.to(DEVICE), val_mask.to(DEVICE), test_mask.to(DEVICE)
pos_weight_t = torch.tensor([pos_weight], device=DEVICE, dtype=torch.float32)


def forward_plain(model):
    return model(x_d, ei_d)


def check_against_reference(name, value, tolerance):
    ref = REFERENCE[name]
    if SMOKE_TEST:
        return {"this_run": value, "reference": ref, "match": None, "note": "skipped (synthetic smoke data)"}
    if isinstance(ref, list):
        match = list(value) == ref
    else:
        match = abs(value - ref) <= tolerance
    status = "OK" if match else "MISMATCH"
    print(f"  reproduction {name}: this run {value} vs reference {ref} -> {status}")
    return {"this_run": value, "reference": ref, "tolerance": tolerance, "match": bool(match)}


print(f"nodes {n_nodes} | features {n_features} | split (train/val/test) {counts} | pos_weight {pos_weight:.4f}")
print(f"edges: {time_stats['edges']} total, {time_stats['same_step']} within a time step, "
      f"{time_stats['cross_step']} crossing time steps")
REPRODUCTION = {
    "split_counts": check_against_reference("split_counts", counts, 0),
    "pos_weight": check_against_reference("pos_weight", pos_weight, 1e-9),
}
save_progress("setup", {"nodes": n_nodes, "features": n_features, "split_counts": counts,
                        "pos_weight": pos_weight, "edge_time_stats": time_stats})

MODEL_RESULTS, PROBS, MODELS = {}, {}, {}


def record_model(name, family, protocol, probs, model=None, extra=None):
    test_score = pr_auc(probs, data.y, test_mask)
    val_score = pr_auc(probs, data.y, val_mask)
    MODEL_RESULTS[name] = {"family": family, "protocol": protocol, "pr_auc_test": test_score,
                           "pr_auc_val": val_score, **(extra or {})}
    PROBS[name] = probs
    if model is not None:
        MODELS[name] = model
    print(f"{name:28s} test PR-AUC {test_score:.4f} | val {val_score:.4f} | {protocol}")
    save_progress("models", MODEL_RESULTS)
    return test_score

# %%
# XGBoost baselines, refit from the stored Phase 1 / 1B best params.
X_tier1 = data.x.numpy()
t0 = time.time()
tier2 = tier2_graph_features(data.edge_index, n_nodes)
print(f"Tier-2 graph features computed in {time.time() - t0:.0f}s")
X_tier12 = np.concatenate([X_tier1, tier2], axis=1)
train_np, y_np = train_mask.numpy(), data.y.numpy()

for name, X, params, protocol in [
    ("xgboost_tier1", X_tier1, XGB_TIER1_PARAMS, "raw features; params from 25-trial Optuna search (Phase 1)"),
    ("xgboost_tier12", X_tier12, XGB_TIER12_PARAMS, "raw + Tier-2 graph features; params from 25-trial Optuna search (Phase 1B)"),
]:
    model = fit_xgb(X[train_np], y_np[train_np], params, pos_weight, SEED)
    probs = torch.from_numpy(model.predict_proba(X)[:, 1]).float()
    score = record_model(name, "xgboost", protocol, probs, model=model)
    REPRODUCTION[name] = check_against_reference(name, score, 0.002)

save_progress("reproduction", REPRODUCTION)

# %%
# Best GNN before Phase 1D: GraphSAGE with the Phase 1C wide-search config, refit with notebook 01's
# protocol (early stopping on val, patience 15, 150 epochs max). This is the "before" model for CARE-GNN.
torch.manual_seed(SEED)
sage = GraphSAGEFlex(n_features, GRAPHSAGE_WIDE_PARAMS["hidden_dim"], GRAPHSAGE_WIDE_PARAMS["dropout"]).to(DEVICE)
opt = torch.optim.Adam(sage.parameters(), lr=GRAPHSAGE_WIDE_PARAMS["lr"], weight_decay=5e-4)
fit = train_full_batch(sage, opt, forward_plain, y_d, train_d, pos_weight_t, epochs=REPRO_REFIT_EPOCHS,
                       val_mask=val_d, patience=15, verbose=True)
score = record_model("graphsage_wide_search", "graphsage",
                     "directed edges; Phase 1C 50-trial search config; early-stopped refit",
                     predict_probs(sage, forward_plain), model=sage,
                     extra={"params": GRAPHSAGE_WIDE_PARAMS, "best_epoch": fit["best_epoch"]})
# GPU nondeterminism moved earlier GNN re-runs by ~0.001; allow more slack for a 256-dim model.
REPRODUCTION["graphsage_wide_search"] = check_against_reference("graphsage_wide_search", score, 0.02)
save_progress("reproduction", REPRODUCTION)

# %% [markdown]
# ## 2. RGCN with label-free relation types
#
# Elliptic's edges are untyped, so relation types have to be constructed. The spec's candidates, and
# why they were or weren't used (details in `docs/design_decisions.md`):
# - **Same-step vs. cross-step edges**: only meaningful if cross-step edges exist; the edge count
#   above shows how many do.
# - **Labeled-adjacent vs. unknown-only**: rejected. Typing an edge by its endpoints' labels leaks
#   test labels into the graph, and because test-period labels are hidden at training time, the
#   relation mix would differ between the train and test periods.
# - **Payment direction** (used): upstream vs. downstream neighbor. Label-free, meaningful for money
#   flow, and adds the downstream messages the notebook 01 GNNs never saw.
# - **Direction x hub source** (used as a variant): splits out messages from the top-1% degree nodes.
#
# All RGCNs use the "defaults" protocol (hidden 64, dropout 0.3, lr 0.01, 40 epochs), the same as
# GraphSAGE/GAT/GIN defaults in notebook 01. The `single` variant is the control: same edges in both
# directions, no typing -- so `direction` minus `single` isolates the effect of relation typing.

# %%
RELATIONS = {}
for scheme in RELATION_SCHEMES:
    ei_r, et_r, names, info = build_relations(data.edge_index, n_nodes, scheme)
    RELATIONS[scheme] = {"edge_index": ei_r.to(DEVICE), "edge_type": et_r.to(DEVICE), "names": names,
                         "info": info, "counts": relation_counts(et_r, names)}
    print(f"{scheme:14s} relations: {RELATIONS[scheme]['counts']} {info}")


def forward_relational(scheme):
    rel = RELATIONS[scheme]
    return lambda model: model(x_d, rel["edge_index"], rel["edge_type"])


for scheme in RELATION_SCHEMES:
    torch.manual_seed(SEED)
    rgcn = RGCN(n_features, len(RELATIONS[scheme]["names"]), hidden_channels=64, dropout=0.3).to(DEVICE)
    opt = torch.optim.Adam(rgcn.parameters(), lr=0.01, weight_decay=5e-4)
    fit = train_full_batch(rgcn, opt, forward_relational(scheme), y_d, train_d, pos_weight_t, epochs=DEFAULT_EPOCHS)
    record_model(f"rgcn_{scheme}", "rgcn", "defaults (hidden 64, dropout 0.3, lr 0.01, 40 epochs)",
                 predict_probs(rgcn, forward_relational(scheme)), model=rgcn,
                 extra={"relation_scheme": scheme, "relation_counts": RELATIONS[scheme]["counts"],
                        "relation_info": RELATIONS[scheme]["info"],
                        "train_loss_first_last": [fit["loss_history"][0], fit["loss_history"][-1]]})

typing_effect = MODEL_RESULTS["rgcn_direction"]["pr_auc_test"] - MODEL_RESULTS["rgcn_single"]["pr_auc_test"]
hub_effect = MODEL_RESULTS["rgcn_direction_hub"]["pr_auc_test"] - MODEL_RESULTS["rgcn_direction"]["pr_auc_test"]
best_rgcn = max((n for n in MODEL_RESULTS if n.startswith("rgcn_")), key=lambda n: MODEL_RESULTS[n]["pr_auc_test"])
rgcn_summary = {
    "relation_typing_effect_direction_minus_single": typing_effect,
    "hub_split_effect_direction_hub_minus_direction": hub_effect,
    "best_rgcn": best_rgcn,
    "best_rgcn_pr_auc_test": MODEL_RESULTS[best_rgcn]["pr_auc_test"],
    "reference_graphsage_default_directed_same_protocol": REFERENCE["graphsage_default_directed"],
    "reference_graphsage_wide_search": REFERENCE["graphsage_wide_search"],
    "cross_step_edges": time_stats["cross_step"],
}
print(f"\nrelation typing (direction - single): {typing_effect:+.4f} | hub split (direction_hub - direction): "
      f"{hub_effect:+.4f} | best RGCN: {best_rgcn} ({MODEL_RESULTS[best_rgcn]['pr_auc_test']:.4f})")
save_progress("rgcn", rgcn_summary)

# %% [markdown]
# ## 3. CARE-GNN
#
# Implementation and every simplification are documented in the `care_gnn.py` cell above and in
# `docs/design_decisions.md`. In short: the paper's label-aware similarity, top-p neighbor
# selection per relation, the paper's reward-based threshold update (+/- tau per epoch, stop after
# 10 epochs of rewards summing within +/-2), and relation-aware aggregation. Simplified: similarity
# from input features shared by both layers, class-weighted loss instead of balanced sampling,
# full-batch, and payment-direction relations.
#
# Relation scheme fixed to `direction` in advance (not picked from the RGCN test results above).
# Trained with early stopping on val, like the wide-search GraphSAGE it is compared against.

# %%
CARE_SCHEME = "direction"
CARE_PARAMS = {"hidden_dim": 64, "dropout": 0.3, "lr": 0.01, "lambda_simi": 2.0,
               "init_threshold": 0.5, "step_size": 0.02}

torch.manual_seed(SEED)
care = CAREGNN(n_features, len(RELATIONS[CARE_SCHEME]["names"]), hidden_channels=CARE_PARAMS["hidden_dim"],
               dropout=CARE_PARAMS["dropout"], lambda_simi=CARE_PARAMS["lambda_simi"],
               init_threshold=CARE_PARAMS["init_threshold"], step_size=CARE_PARAMS["step_size"]).to(DEVICE)
opt = torch.optim.Adam(care.parameters(), lr=CARE_PARAMS["lr"], weight_decay=5e-4)
t0 = time.time()
fit = train_full_batch(care, opt, forward_relational(CARE_SCHEME), y_d, train_d, pos_weight_t,
                       epochs=REFIT_MAX_EPOCHS, val_mask=val_d, patience=15, verbose=True)
care_diag = {**care.diagnostics(), "best_epoch": fit["best_epoch"], "epochs_run": fit["epochs_run"],
             "thresholds_at_restored_best_epoch": [float(t) for t in care.thresholds],
             "threshold_history_every_10_epochs": care.threshold_history[::10],
             "train_seconds": round(time.time() - t0, 1)}
record_model("care_gnn", "care_gnn",
             "simplified CARE-GNN (see design_decisions.md); direction relations; early-stopped (patience 15)",
             predict_probs(care, forward_relational(CARE_SCHEME)), model=care,
             extra={"params": CARE_PARAMS, "diagnostics": care_diag})
print("CARE-GNN diagnostics:", {k: v for k, v in care_diag.items() if k != "threshold_history_every_10_epochs"})

# %%
# Camouflage before/after at a matched false-positive rate. Same subset definitions as Phase 1C.
subsets, licit_test_idx = camouflage_subsets(data.edge_index, data.y, test_mask, n_nodes)
CAMOUFLAGE_MODELS = ["xgboost_tier12", "graphsage_wide_search", best_rgcn, "care_gnn"]
CAMO = {name: camouflage_report(PROBS[name], subsets, licit_test_idx, CAMOUFLAGE_TARGET_FPR)
        for name in CAMOUFLAGE_MODELS}

cols = ["all_illicit", "zero_illicit_neighbors/camouflaged", "zero_illicit_neighbors/non_camouflaged",
        "mostly_licit_or_unknown/camouflaged", "mostly_licit_or_unknown/non_camouflaged"]
print(f"recall at matched {CAMOUFLAGE_TARGET_FPR:.0%} FPR on licit test nodes (subset sizes: "
      + ", ".join(f"{c}={CAMO['care_gnn']['subsets'][c]['n']}" for c in cols) + ")")
print(f"{'model':24s}" + "".join(f"{c.split('/')[-1][:14]:>16s}" for c in cols) + "   actual FPR")
for name in CAMOUFLAGE_MODELS:
    row = CAMO[name]["subsets"]
    cells = "".join(f"{row[c]['recall_at_matched_fpr']:16.4f}" if row[c]["n"] else f"{'n/a':>16s}" for c in cols)
    print(f"{name:24s}{cells}   {CAMO[name]['actual_fpr_on_licit_test']:.4f}")


def recall_delta(a, b, subset):
    """Recall difference (a - b) on a subset at matched FPR, with a normal-approximation standard error."""
    ma, mb = CAMO[a]["subsets"][subset], CAMO[b]["subsets"][subset]
    if min(ma["n"], mb["n"]) < MIN_SUBSET_SIZE:
        return None
    ra, rb, n = ma["recall_at_matched_fpr"], mb["recall_at_matched_fpr"], ma["n"]
    se = math.sqrt(max(ra * (1 - ra), 1e-9) / n + max(rb * (1 - rb), 1e-9) / n)
    delta = ra - rb
    return {"delta": delta, "approx_se": se, "beyond_2se": abs(delta) > 2 * se}


# Does CARE-GNN's selector act on camouflaged fraud as intended? For illicit and licit test targets,
# the share of incoming edges kept, split by the neighbor's label.
predict_probs(care, forward_relational(CARE_SCHEME))  # refresh last_keep in eval mode
keep = care.last_keep.cpu()
src_r, dst_r = RELATIONS[CARE_SCHEME]["edge_index"].cpu()
selector_keep_rates = {}
for target_label, target_name in [(ILLICIT, "illicit_test_targets"), (LICIT, "licit_test_targets")]:
    target_sel = test_mask[dst_r] & (data.y[dst_r] == target_label)
    rates = {}
    for nb_label, nb_name in [(LICIT, "licit"), (ILLICIT, "illicit"), (UNKNOWN, "unknown")]:
        sel = target_sel & (data.y[src_r] == nb_label)
        rates[f"{nb_name}_neighbors"] = {"edges": int(sel.sum()),
                                         "kept_share": float(keep[sel].float().mean()) if sel.any() else None}
    selector_keep_rates[target_name] = rates
print("\nCARE-GNN selector keep rates:", json.dumps(selector_keep_rates, indent=1))

CAMO_COMPARISONS = {
    subset: {
        "care_minus_graphsage": recall_delta("care_gnn", "graphsage_wide_search", subset),
        "care_minus_xgboost_tier12": recall_delta("care_gnn", "xgboost_tier12", subset),
        "graphsage_minus_xgboost_tier12": recall_delta("graphsage_wide_search", "xgboost_tier12", subset),
    }
    for subset in ["all_illicit", "zero_illicit_neighbors/camouflaged", "mostly_licit_or_unknown/camouflaged"]
}


def describe_delta(d, a_name, b_name):
    if d is None:
        return f"{a_name} vs {b_name}: subset too small to compare"
    verdict = ("higher" if d["delta"] > 0 else "lower") if d["beyond_2se"] else "no clear difference (within ~2 SE)"
    return f"{a_name} vs {b_name}: {d['delta']:+.4f} (SE ~{d['approx_se']:.4f}) -> {verdict}"


camo_key = "zero_illicit_neighbors/camouflaged"
camouflage_finding = (
    f"On the camouflaged subset (illicit test nodes with no illicit-labeled neighbors, n="
    f"{CAMO['care_gnn']['subsets'][camo_key]['n']}), recall at a matched {CAMOUFLAGE_TARGET_FPR:.0%} FPR: "
    f"CARE-GNN {CAMO['care_gnn']['subsets'][camo_key]['recall_at_matched_fpr']:.4f}, "
    f"GraphSAGE (wide search) {CAMO['graphsage_wide_search']['subsets'][camo_key]['recall_at_matched_fpr']:.4f}, "
    f"XGBoost Tier-1+2 {CAMO['xgboost_tier12']['subsets'][camo_key]['recall_at_matched_fpr']:.4f}. "
    + describe_delta(CAMO_COMPARISONS[camo_key]["care_minus_graphsage"], "CARE-GNN", "GraphSAGE") + "; "
    + describe_delta(CAMO_COMPARISONS[camo_key]["care_minus_xgboost_tier12"], "CARE-GNN", "XGBoost Tier-1+2") + "."
)
print("\n" + camouflage_finding)
save_progress("camouflage_before_after", {
    "models": CAMO, "comparisons": CAMO_COMPARISONS, "selector_keep_rates": selector_keep_rates,
    "finding": camouflage_finding,
    "note": "Phase 1C reported recall at p>0.5 within one model; across models, recall is compared at a "
            "matched false-positive rate on licit test nodes because the models are calibrated differently.",
})

# %% [markdown]
# ## 4. Leaderboard (models trained in this notebook)

# %%


def print_leaderboard(results):
    rows = sorted(results.items(), key=lambda kv: kv[1]["pr_auc_test"], reverse=True)
    print(f"{'model':28s} {'family':10s} {'test PR-AUC':>11s} {'val PR-AUC':>10s}  protocol")
    for name, r in rows:
        print(f"{name:28s} {r['family']:10s} {r['pr_auc_test']:11.4f} {r['pr_auc_val']:10.4f}  {r['protocol']}")


print_leaderboard(MODEL_RESULTS)

# %% [markdown]
# ## 5. Expanded search on the best GNN
#
# "Best" = highest temporal-test PR-AUC among the GNNs above, as the spec defines it. Picking a model
# by its test score makes that model's leaderboard number slightly optimistic; the 3-seed check below
# and the separate validation scores are the guard against over-reading it. The search itself selects
# on validation (time steps 31-35) only, with early stopping, as in every earlier phase.

# %%
GNN_FAMILIES = {"graphsage", "rgcn", "care_gnn"}
gnn_names = [n for n, r in MODEL_RESULTS.items() if r["family"] in GNN_FAMILIES]
search_base = max(gnn_names, key=lambda n: MODEL_RESULTS[n]["pr_auc_test"])
search_family = MODEL_RESULTS[search_base]["family"]
search_scheme = MODEL_RESULTS[search_base].get("relation_scheme", CARE_SCHEME if search_family == "care_gnn" else None)
print(f"best GNN so far: {search_base} ({MODEL_RESULTS[search_base]['pr_auc_test']:.4f}) -> searching family "
      f"'{search_family}'" + (f" with relation scheme '{search_scheme}'" if search_scheme else ""))

SEARCH_SPACES = {
    "graphsage": {"hidden_dim": [128, 256, 512]},
    "rgcn": {"hidden_dim": [32, 64, 128, 256]},
    "care_gnn": {"hidden_dim": [32, 64, 128], "lambda_simi": (0.5, 4.0), "init_threshold": (0.2, 0.9),
                 "step_size": (0.005, 0.05)},
}
COMMON_RANGES = {"lr": (1e-4, 3e-2), "dropout": (0.1, 0.6), "max_epochs": (min(30, SEARCH_MAX_EPOCHS), SEARCH_MAX_EPOCHS)}


def suggest_params(trial, family):
    params = {
        "lr": trial.suggest_float("lr", *COMMON_RANGES["lr"], log=True),
        "dropout": trial.suggest_float("dropout", *COMMON_RANGES["dropout"]),
        "max_epochs": trial.suggest_int("max_epochs", *COMMON_RANGES["max_epochs"]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", SEARCH_SPACES[family]["hidden_dim"]),
    }
    if family == "care_gnn":
        params["lambda_simi"] = trial.suggest_float("lambda_simi", *SEARCH_SPACES[family]["lambda_simi"])
        params["init_threshold"] = trial.suggest_float("init_threshold", *SEARCH_SPACES[family]["init_threshold"])
        params["step_size"] = trial.suggest_float("step_size", *SEARCH_SPACES[family]["step_size"], log=True)
    return params


def build_model(family, params, seed, scheme):
    torch.manual_seed(seed)
    if family == "graphsage":
        model, forward = GraphSAGEFlex(n_features, params["hidden_dim"], params["dropout"]), forward_plain
    elif family == "rgcn":
        model = RGCN(n_features, len(RELATIONS[scheme]["names"]), params["hidden_dim"], params["dropout"])
        forward = forward_relational(scheme)
    else:
        model = CAREGNN(n_features, len(RELATIONS[scheme]["names"]), params["hidden_dim"], params["dropout"],
                        lambda_simi=params["lambda_simi"], init_threshold=params["init_threshold"],
                        step_size=params["step_size"])
        forward = forward_relational(scheme)
    model = model.to(DEVICE)
    return model, forward, torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=5e-4)


def objective(trial):
    params = suggest_params(trial, search_family)
    model, forward, optimizer = build_model(search_family, params, SEED, search_scheme)
    try:
        fit = train_full_batch(model, optimizer, forward, y_d, train_d, pos_weight_t, epochs=params["max_epochs"],
                               val_mask=val_d, patience=10)
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as e:
        print(f"  trial {trial.number} failed: {type(e).__name__}: {e}")
        raise optuna.TrialPruned()
    finally:
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return fit["best_val_pr_auc"]


def progress_callback(study, trial):
    done = len(study.trials)
    if done % 10 == 0 or done == SEARCH_TRIALS:
        n_ok = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        best = f"{study.best_value:.4f}" if n_ok else "n/a"
        print(f"  {done} trials ({n_ok} completed) | best val PR-AUC {best} | elapsed {time.time() - search_start:.0f}s")


search_start = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=SEARCH_TRIALS, timeout=SEARCH_TIMEOUT_SEC, callbacks=[progress_callback])
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if not completed:
    save_progress("expanded_search", {"base_model": search_base, "family": search_family,
                                      "trials_total": len(study.trials), "trials_completed": 0})
    raise RuntimeError("every search trial failed (see messages above) -- log this as a real bug; tasks 1-4 are saved")
search_best = study.best_params

at_boundary = []
for key, value in search_best.items():
    if key in SEARCH_SPACES[search_family] and isinstance(SEARCH_SPACES[search_family][key], list):
        if value == max(SEARCH_SPACES[search_family][key]):
            at_boundary.append(f"{key}={value} (max option)")
        continue
    lo, hi = COMMON_RANGES.get(key) or SEARCH_SPACES[search_family][key]
    if key in ("lr", "step_size"):
        near_lo, near_hi = value <= lo * 1.5, value >= hi / 1.5
    else:
        span = hi - lo
        near_lo, near_hi = value <= lo + 0.05 * span, value >= hi - 0.05 * span
    if near_lo or near_hi:
        at_boundary.append(f"{key}={value:.4g} (near {'lower' if near_lo else 'upper'} bound)")

print(f"search: {len(completed)}/{len(study.trials)} trials completed in {time.time() - search_start:.0f}s "
      f"(cap {SEARCH_TRIALS} trials / {SEARCH_TIMEOUT_SEC}s)")
print(f"best params: {search_best} | best val PR-AUC {study.best_value:.4f}")
print("at a search-space boundary:", at_boundary or "none")

final_model, final_forward, final_opt = build_model(search_family, search_best, SEED, search_scheme)
fit = train_full_batch(final_model, final_opt, final_forward, y_d, train_d, pos_weight_t, epochs=REFIT_MAX_EPOCHS,
                       val_mask=val_d, patience=15, verbose=True)
final_name = f"{search_family}_expanded_search"
record_model(final_name, search_family,
             f"{len(completed)}-trial Optuna search (val-selected, early stopping); early-stopped refit",
             predict_probs(final_model, final_forward), model=final_model,
             extra={"params": search_best, "relation_scheme": search_scheme, "best_epoch": fit["best_epoch"]})
search_change = MODEL_RESULTS[final_name]["pr_auc_test"] - MODEL_RESULTS[search_base]["pr_auc_test"]
save_progress("expanded_search", {
    "base_model": search_base, "family": search_family, "relation_scheme": search_scheme,
    "trials_requested": SEARCH_TRIALS, "trials_completed": len(completed), "trials_total": len(study.trials),
    "timeout_sec": SEARCH_TIMEOUT_SEC, "seconds": round(time.time() - search_start, 1),
    "best_params": search_best, "best_val_pr_auc": study.best_value, "at_boundary": at_boundary,
    "pr_auc_test": MODEL_RESULTS[final_name]["pr_auc_test"], "change_vs_base": search_change,
    "search_space": {**{k: list(v) for k, v in COMMON_RANGES.items()}, **SEARCH_SPACES[search_family]},
})

# %% [markdown]
# ## 6. Robustness check: 3 seeds on the final best GNN

# %%
final_best_gnn = max((n for n, r in MODEL_RESULTS.items() if r["family"] in GNN_FAMILIES),
                     key=lambda n: MODEL_RESULTS[n]["pr_auc_test"])
print(f"final best GNN by test PR-AUC: {final_best_gnn} ({MODEL_RESULTS[final_best_gnn]['pr_auc_test']:.4f})")

# Re-run the winner with exactly the protocol that produced its leaderboard number.
if final_best_gnn == final_name:
    robust_family, robust_params, robust_scheme = search_family, search_best, search_scheme
    robust_epochs, robust_early_stop = REFIT_MAX_EPOCHS, True
elif final_best_gnn == "care_gnn":
    robust_family, robust_params, robust_scheme = "care_gnn", CARE_PARAMS, CARE_SCHEME
    robust_epochs, robust_early_stop = REFIT_MAX_EPOCHS, True
elif final_best_gnn == "graphsage_wide_search":
    robust_family, robust_params, robust_scheme = "graphsage", GRAPHSAGE_WIDE_PARAMS, None
    robust_epochs, robust_early_stop = REPRO_REFIT_EPOCHS, True
else:  # an untuned RGCN variant won: defaults protocol, fixed epochs, no early stopping
    robust_family, robust_scheme = "rgcn", MODEL_RESULTS[final_best_gnn]["relation_scheme"]
    robust_params = {"hidden_dim": 64, "dropout": 0.3, "lr": 0.01}
    robust_epochs, robust_early_stop = DEFAULT_EPOCHS, False

robust_scores = []
for seed in ROBUSTNESS_SEEDS:
    model, forward, optimizer = build_model(robust_family, robust_params, seed, robust_scheme)
    train_full_batch(model, optimizer, forward, y_d, train_d, pos_weight_t, epochs=robust_epochs,
                     val_mask=val_d if robust_early_stop else None, patience=15 if robust_early_stop else None)
    robust_scores.append(pr_auc(predict_probs(model, forward), data.y, test_mask))
    print(f"  seed {seed}: test PR-AUC {robust_scores[-1]:.4f}")
    del model, optimizer
robust_mean, robust_std = float(np.mean(robust_scores)), float(np.std(robust_scores))
print(f"{final_best_gnn}: {robust_mean:.4f} +/- {robust_std:.4f} across seeds {ROBUSTNESS_SEEDS}")
save_progress("robustness_best_gnn", {"model": final_best_gnn, "family": robust_family, "params": robust_params,
                                      "relation_scheme": robust_scheme, "max_epochs": robust_epochs,
                                      "early_stopping": robust_early_stop, "seeds": ROBUSTNESS_SEEDS,
                                      "scores": robust_scores, "mean": robust_mean, "std": robust_std})

# %%
# Camouflage check on the final best GNN too, if the expanded search produced a new one.
if final_best_gnn not in CAMO:
    CAMO[final_best_gnn] = camouflage_report(PROBS[final_best_gnn], subsets, licit_test_idx, CAMOUFLAGE_TARGET_FPR)
    final_camo = CAMO[final_best_gnn]["subsets"][camo_key]
    print(f"{final_best_gnn} camouflaged-subset recall at matched FPR: {final_camo['recall_at_matched_fpr']:.4f}")
    save_progress("camouflage_final_best_gnn", CAMO[final_best_gnn])

# %% [markdown]
# ## 7. Closing analysis and save

# %%
# Compare against the XGBoost Tier-1+2 refit from THIS run (guaranteed same setup); the 3-seed mean from
# notebook 01 is quoted alongside for its variance.
xgb_this_run = MODEL_RESULTS["xgboost_tier12"]["pr_auc_test"]
gap_to_xgb = robust_mean - xgb_this_run
standing_finding_changed = robust_mean >= xgb_this_run
care_vs_sage = CAMO_COMPARISONS[camo_key]["care_minus_graphsage"]
care_vs_xgb = CAMO_COMPARISONS[camo_key]["care_minus_xgboost_tier12"]

overall = (
    f"Overall: the best GNN after Phase 1D is {final_best_gnn}, at {robust_mean:.4f} +/- {robust_std:.4f} "
    f"test PR-AUC over 3 seeds, vs XGBoost on raw + classical graph features at {xgb_this_run:.4f} in this run "
    f"({REFERENCE['xgboost_tier12_3seed_mean']:.4f} +/- {REFERENCE['xgboost_tier12_3seed_std']:.4f} over 3 seeds "
    f"in notebook 01), a gap of {gap_to_xgb:+.4f}. "
    + ("The standing finding CHANGES: a message-passing GNN now matches or beats tuned XGBoost."
       if standing_finding_changed else
       "The standing finding holds: message-passing GNNs still trail tuned XGBoost on this dataset.")
)
if care_vs_sage is None or care_vs_xgb is None:
    targeted = "The camouflaged subset was too small for a before/after comparison."
else:
    care_helped = care_vs_sage["delta"] > 0 and care_vs_sage["beyond_2se"]
    care_hurt = care_vs_sage["delta"] < 0 and care_vs_sage["beyond_2se"]
    care_matches_xgb = not (care_vs_xgb["delta"] < 0 and care_vs_xgb["beyond_2se"])
    targeted = (
        "Targeted defense: "
        + ("CARE-GNN's neighbor selection improved recall on the camouflaged subset over the pre-1D best GNN "
           if care_helped else
           "CARE-GNN's neighbor selection made camouflaged-subset recall worse than the pre-1D best GNN "
           if care_hurt else
           "CARE-GNN's neighbor selection did not measurably change camouflaged-subset recall vs the pre-1D best GNN ")
        + f"({care_vs_sage['delta']:+.4f}, SE ~{care_vs_sage['approx_se']:.4f}), and on that subset it "
        + ("matches or beats XGBoost" if care_matches_xgb else "still trails XGBoost")
        + f" ({care_vs_xgb['delta']:+.4f}), at a matched {CAMOUFLAGE_TARGET_FPR:.0%} false-positive rate."
    )
typing = (
    f"Relation typing: RGCN with payment-direction relations vs the same edges untyped: "
    f"{rgcn_summary['relation_typing_effect_direction_minus_single']:+.4f} test PR-AUC; "
    f"best RGCN {best_rgcn} at {rgcn_summary['best_rgcn_pr_auc_test']:.4f}."
)
closing_paragraph = " ".join([overall, targeted, typing])
print(closing_paragraph)

reproduction_ok = all(v.get("match") is not False for v in REPRODUCTION.values())
if not reproduction_ok and not SMOKE_TEST:
    closing_paragraph += (" WARNING: at least one reproduction check against notebook 01 failed (see "
                          "'reproduction'), so this notebook's numbers may not be directly comparable to earlier phases.")
    print("\nWARNING: reproduction mismatch -- see RESULTS['reproduction']")

save_progress("closing_paragraph", closing_paragraph)
save_progress("standing_finding_changed", bool(standing_finding_changed))

for name, model in MODELS.items():
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), OUTPUT_DIR / f"phase1d_{name}.pt")
    else:
        model.save_model(str(OUTPUT_DIR / f"phase1d_{name}.json"))
print(f"saved {len(MODELS)} models to {OUTPUT_DIR}")
print_leaderboard(MODEL_RESULTS)
print(f"\nFull results: {RESULTS_PATH}")
print(json.dumps(RESULTS, indent=2))
