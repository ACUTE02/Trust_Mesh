"""CPU smoke/unit tests for engines/graph on tiny synthetic graphs. No real data, no reported numbers.

Run: python -m pytest tests/test_graph_modules.py   (needs torch, torch_geometric, xgboost, networkx)
"""
import math

import torch

from engines.graph.care_gnn import CAREGNN, top_p_per_target
from engines.graph.data import (ILLICIT, UNKNOWN, class_pos_weight, edge_time_stats,
                                make_synthetic_elliptic, temporal_masks)
from engines.graph.evaluate import camouflage_report, camouflage_subsets, threshold_at_fpr
from engines.graph.gnn import RGCN, GraphSAGEFlex
from engines.graph.relations import RELATION_SCHEMES, build_relations, relation_counts
from engines.graph.train import pr_auc, predict_probs, train_full_batch


def test_top_p_per_target_keeps_closest_neighbors():
    # target 0 has 4 incoming edges (distances 0.4, 0.1, 0.3, 0.2); target 1 has 1; target 2 has 2
    dst = torch.tensor([0, 0, 0, 0, 1, 2, 2])
    dist = torch.tensor([0.4, 0.1, 0.3, 0.2, 0.9, 0.5, 0.6])
    keep = top_p_per_target(dst, dist, torch.tensor(0.5), num_nodes=3)
    # ceil(0.5*4)=2 -> distances 0.1, 0.2; ceil(0.5*1)=1 -> its only edge; ceil(0.5*2)=1 -> 0.5
    assert keep.tolist() == [False, True, False, True, True, True, False]

    keep_all = top_p_per_target(dst, dist, torch.tensor(1.0), num_nodes=3)
    assert keep_all.all()
    keep_min = top_p_per_target(dst, dist, torch.tensor(0.0), num_nodes=3)
    assert keep_min.tolist() == [False, True, False, False, True, True, False]  # at least one each


def test_relations_are_label_free_and_cover_both_directions():
    data = make_synthetic_elliptic(nodes_per_step=8)
    e = data.edge_index.size(1)
    for scheme in RELATION_SCHEMES:
        ei, et, names, _ = build_relations(data.edge_index, data.num_nodes, scheme)
        assert ei.size(1) == 2 * e and et.numel() == 2 * e
        assert int(et.max()) < len(names)
        assert sum(relation_counts(et, names).values()) == 2 * e
    ei, et, _, _ = build_relations(data.edge_index, data.num_nodes, "direction")
    assert torch.equal(ei[:, :e], data.edge_index) and torch.equal(ei[:, e:], data.edge_index.flip(0))
    assert edge_time_stats(data.edge_index, data.time_step)["cross_step"] == 0


def _fit(model, forward, data, epochs=15, **kw):
    train, val, _ = temporal_masks(data)
    pw = torch.tensor([class_pos_weight(data.y, train)])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return train_full_batch(model, opt, forward, data.y, train, pw, epochs=epochs, **kw)


def test_all_gnns_train_and_predict():
    torch.manual_seed(0)
    data = make_synthetic_elliptic(nodes_per_step=10)
    ei, et, names, _ = build_relations(data.edge_index, data.num_nodes, "direction")
    _, _, test = temporal_masks(data)

    sage = GraphSAGEFlex(data.num_node_features, 16)
    fit = _fit(sage, lambda m: m(data.x, data.edge_index), data)
    assert len(fit["loss_history"]) == 15

    rgcn = RGCN(data.num_node_features, len(names), 16)
    _fit(rgcn, lambda m: m(data.x, ei, et), data)

    care = CAREGNN(data.num_node_features, len(names), 16, step_size=0.05)
    fit = _fit(care, lambda m: m(data.x, ei, et), data, epochs=30, val_mask=temporal_masks(data)[1], patience=30)
    probs = predict_probs(care, lambda m: m(data.x, ei, et))
    assert probs.shape == (data.num_nodes,) and torch.isfinite(probs).all()
    assert 0.0 <= pr_auc(probs, data.y, test) <= 1.0
    diag = care.diagnostics()
    assert len(diag["final_thresholds"]) == 2 and all(0.0 <= t <= 1.0 for t in diag["final_thresholds"])
    assert sum(diag["threshold_updates"]) > 0, "threshold update rule never fired"
    assert care.last_keep is not None and care.last_keep.dtype == torch.bool


def test_camouflage_report_shapes_and_matched_fpr():
    data = make_synthetic_elliptic(nodes_per_step=10)
    _, _, test = temporal_masks(data)
    subsets, licit = camouflage_subsets(data.edge_index, data.y, test, data.num_nodes)
    probs = torch.rand(data.num_nodes)
    thr, fpr = threshold_at_fpr(probs, licit, 0.1)
    assert 0.0 <= fpr <= 0.15
    report = camouflage_report(probs, subsets, licit, 0.1)
    for name, m in report["subsets"].items():
        if m["n"]:
            assert 0.0 <= m["recall_at_matched_fpr"] <= 1.0 and not math.isnan(m["pr_auc_vs_licit"])
    all_idx = subsets["all_illicit"]
    assert bool((data.y[all_idx] == ILLICIT).all()) and not bool((data.y[licit] == UNKNOWN).any())
