"""Camouflage subsets (same definitions as Phase 1C) and threshold-matched recall for cross-model comparison."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.utils import to_undirected

from .data import ILLICIT, LICIT

CAMOUFLAGE_DEFINITION_NAMES = ("mostly_licit_or_unknown", "zero_illicit_neighbors")
MIN_SUBSET_SIZE = 20


def neighbor_label_stats(edge_index, y, num_nodes):
    """Per node: number of real (non-self) undirected neighbors, and how many are labeled illicit."""
    src, dst = to_undirected(edge_index.cpu())
    not_self = src != dst
    is_illicit = (y.cpu() == ILLICIT).long()
    real_degree = torch.zeros(num_nodes, dtype=torch.long)
    real_degree.scatter_add_(0, dst[not_self], torch.ones(int(not_self.sum()), dtype=torch.long))
    illicit_neighbors = torch.zeros(num_nodes, dtype=torch.long)
    illicit_neighbors.scatter_add_(0, dst[not_self], is_illicit[src[not_self]])
    return real_degree, illicit_neighbors


def camouflage_subsets(edge_index, y, test_mask, num_nodes):
    """Illicit test nodes split into camouflaged / non-camouflaged under both Phase 1C definitions.
    Isolated illicit nodes belong to neither side (they aren't hiding among anything)."""
    real_degree, illicit_neighbors = neighbor_label_stats(edge_index, y, num_nodes)
    frac = illicit_neighbors.float() / real_degree.clamp(min=1).float()
    has_neighbors = real_degree > 0
    definitions = {
        "mostly_licit_or_unknown": has_neighbors & (frac < 0.5),
        "zero_illicit_neighbors": has_neighbors & (illicit_neighbors == 0),
    }
    y_cpu, test_cpu = y.cpu(), test_mask.cpu()
    illicit_test = (test_cpu & (y_cpu == ILLICIT)).nonzero(as_tuple=True)[0]
    licit_test = (test_cpu & (y_cpu == LICIT)).nonzero(as_tuple=True)[0]
    subsets = {"all_illicit": illicit_test}
    for name, mask in definitions.items():
        subsets[f"{name}/camouflaged"] = illicit_test[mask[illicit_test]]
        subsets[f"{name}/non_camouflaged"] = illicit_test[(~mask[illicit_test]) & has_neighbors[illicit_test]]
    return subsets, licit_test


def threshold_at_fpr(probs, licit_idx, target_fpr):
    """Score threshold at which roughly `target_fpr` of licit test nodes are flagged. Comparing
    models' recall at the same false-positive rate removes calibration differences that make a
    fixed 0.5 cutoff unfair across models."""
    negatives = probs[licit_idx].double().numpy()
    threshold = float(np.quantile(negatives, 1.0 - target_fpr))
    actual_fpr = float((negatives > threshold).mean())
    return threshold, actual_fpr


def subset_metrics(probs, idx, licit_idx, threshold_matched):
    if len(idx) == 0:
        return {"n": 0, "recall_at_0.5": None, "recall_at_matched_fpr": None,
                "pr_auc_vs_licit": None, "pr_auc_chance_baseline": None}
    pos = probs[idx]
    scores = torch.cat([pos, probs[licit_idx]]).numpy()
    labels = np.concatenate([np.ones(len(idx)), np.zeros(len(licit_idx))])
    return {
        "n": int(len(idx)),
        "recall_at_0.5": float((pos > 0.5).float().mean()),
        "recall_at_matched_fpr": float((pos > threshold_matched).float().mean()),
        "pr_auc_vs_licit": float(average_precision_score(labels, scores)),
        "pr_auc_chance_baseline": float(len(idx) / (len(idx) + len(licit_idx))),
    }


def camouflage_report(probs, subsets, licit_idx, target_fpr=0.05):
    threshold, actual_fpr = threshold_at_fpr(probs, licit_idx, target_fpr)
    report = {
        "target_fpr": target_fpr,
        "matched_threshold": threshold,
        "actual_fpr_on_licit_test": actual_fpr,
        "subsets": {name: subset_metrics(probs, idx, licit_idx, threshold) for name, idx in subsets.items()},
    }
    return report
