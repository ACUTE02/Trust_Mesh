"""Graph scoring: the real XGBoost Tier-1+2 model (Phase 1D's best graph model), plus a clearly
secondary GAT attention explanation.

The score served here is XGBoost's. GAT/GraphSAGE scored far lower on Elliptic (0.26-0.59 PR-AUC vs
0.7482 for XGBoost), so they are used only for the structural explanation, never for the score.
"""
import json
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb

from engines.graph.data import ILLICIT, LICIT, UNKNOWN, load_elliptic
from engines.graph.features import tier2_graph_features

from . import config

LABEL_NAMES = {LICIT: "licit", ILLICIT: "illicit", UNKNOWN: "unknown"}
# Verdict bands are defined by the false-flag rate on held-out *known-licit* nodes (validation
# steps 31-35, which the model never trained on): "high" flags 1% of licit nodes, "medium" 10%.
HIGH_FPR, MEDIUM_FPR = 0.01, 0.10


def split_of(time_step):
    t = int(time_step)
    return "train" if t <= 30 else "val" if t <= 35 else "test"


class GAT(torch.nn.Module):
    """Same architecture as the Phase 1B GAT whose weights are in artifacts/graph/gat_weights.pt."""

    def __init__(self, in_channels, hidden_channels=64, heads=4, dropout=0.3, out_channels=1):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x, (ei1, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1), ei1, alpha1


def compute_thresholds(scores, labels, time_step):
    """Verdict thresholds from validation-period known-licit scores (see HIGH_FPR/MEDIUM_FPR)."""
    val = (labels == LICIT) & (time_step >= 31) & (time_step <= 35)
    licit_scores = scores[val]
    return {
        "high": float(np.quantile(licit_scores, 1.0 - HIGH_FPR)),
        "medium": float(np.quantile(licit_scores, 1.0 - MEDIUM_FPR)),
        "rule": f"high = flags {HIGH_FPR:.0%} and medium = flags {MEDIUM_FPR:.0%} of known-licit "
                "validation-period (steps 31-35) nodes",
        "n_val_licit": int(val.sum()),
    }


class GraphService:
    def __init__(self):
        self._lock = threading.Lock()
        self.ready = False
        self.error = None

    # -- loading ---------------------------------------------------------------------------------
    def load(self, with_gat=True):
        t0 = time.time()
        config.SERVING_CACHE.mkdir(parents=True, exist_ok=True)
        self.data = load_elliptic(config.ELLIPTIC_ROOT)
        n = self.data.num_nodes
        self.time_step = self.data.time_step.numpy()
        self.labels = self.data.y.numpy()
        self.tx_id = self._load_tx_ids()

        tier2_path = config.SERVING_CACHE / "tier2_features.npy"
        if tier2_path.exists():
            self.tier2 = np.load(tier2_path)
        else:  # ~minutes on CPU (PageRank + clustering over 200k nodes), so cached once
            self.tier2 = tier2_graph_features(self.data.edge_index, n)
            np.save(tier2_path, self.tier2)
        self.features = np.concatenate([self.data.x.numpy().astype(np.float32), self.tier2], axis=1)

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(config.XGB_WEIGHTS))
        assert self.model.n_features_in_ == self.features.shape[1], (
            f"XGBoost expects {self.model.n_features_in_} features, built {self.features.shape[1]}")
        self.scores = self.model.predict_proba(self.features)[:, 1].astype(np.float32)

        self.thresholds = self._load_or_make_thresholds()
        self.verification = self._verify()
        self.gat = self._load_gat() if with_gat else None
        self.ready = True
        self.load_seconds = round(time.time() - t0, 1)

    def _load_tx_ids(self):
        import os
        import pandas as pd
        from torch_geometric.datasets import EllipticBitcoinDataset
        ds = EllipticBitcoinDataset(root=str(config.ELLIPTIC_ROOT))
        feat_csv = next(p for p in ds.raw_paths if "feat" in os.path.basename(p).lower())
        return pd.read_csv(feat_csv, header=None, usecols=[0]).iloc[:, 0].to_numpy()

    def _load_or_make_thresholds(self):
        if config.GRAPH_THRESHOLDS.exists():
            return json.loads(config.GRAPH_THRESHOLDS.read_text())
        thresholds = compute_thresholds(self.scores, self.labels, self.time_step)
        config.GRAPH_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
        config.GRAPH_THRESHOLDS.write_text(json.dumps(thresholds, indent=2))
        return thresholds

    def _verify(self):
        """Test-period PR-AUC of the loaded weights: a startup check that these are the real model
        (Phase 1D: 0.7474 for this seed-42 model, 0.7482 +/- 0.0030 over 3 seeds)."""
        from sklearn.metrics import average_precision_score
        test = (self.labels != UNKNOWN) & (self.time_step >= 36)
        return {"test_pr_auc": float(average_precision_score(self.labels[test] == ILLICIT, self.scores[test])),
                "n_test": int(test.sum()), "reported_seed42": 0.7474345053830228}

    def _load_gat(self):
        gat = GAT(self.features.shape[1] - 4)  # GAT was trained on the 165 raw features only
        gat.load_state_dict(torch.load(config.GAT_WEIGHTS, map_location="cpu"))
        gat.eval()
        with torch.no_grad():
            out, ei, alpha = gat(self.data.x, self.data.edge_index)
        src, dst = ei
        keep = src != dst  # GATConv adds self-loops; they aren't real neighbors
        return {"prob": torch.sigmoid(out).numpy(), "src": src[keep].numpy(), "dst": dst[keep].numpy(),
                "alpha": alpha.mean(dim=1)[keep].numpy()}

    # -- queries ---------------------------------------------------------------------------------
    def verdict(self, score):
        if score >= self.thresholds["high"]:
            return "high"
        if score >= self.thresholds["medium"]:
            return "medium"
        return "low"

    def check_node(self, node_id):
        if not 0 <= node_id < self.data.num_nodes:
            raise KeyError(f"node_id {node_id} out of range [0, {self.data.num_nodes - 1}]")

    def score(self, node_id):
        self.check_node(node_id)
        score = float(self.scores[node_id])
        label = int(self.labels[node_id])
        step = int(self.time_step[node_id])
        split = split_of(step)
        return {
            "node_id": node_id, "tx_id": int(self.tx_id[node_id]), "time_step": step,
            "score": score, "verdict": self.verdict(score),
            "model": "xgboost_tier1_tier2", "thresholds": {k: self.thresholds[k] for k in ("medium", "high")},
            "ground_truth": LABEL_NAMES[label], "split": split,
            "graph_features": {"degree": float(self.tier2[node_id, 0]), "out_in_degree_ratio": float(self.tier2[node_id, 1]),
                               "pagerank": float(self.tier2[node_id, 2]), "clustering": float(self.tier2[node_id, 3])},
            "in_sample": split == "train" and label != UNKNOWN,
        }

    def explain(self, node_id, top_k=5):
        self.check_node(node_id)
        if self.gat is None:
            return {"node_id": node_id, "available": False, "reason": "GAT explanation disabled"}
        g = self.gat
        incoming = np.flatnonzero(g["dst"] == node_id)
        order = incoming[np.argsort(-g["alpha"][incoming])][:top_k]
        neighbors = [{"node_id": int(g["src"][i]), "attention": float(g["alpha"][i]),
                      "xgb_score": float(self.scores[g["src"][i]]),
                      "ground_truth": LABEL_NAMES[int(self.labels[g["src"][i]])]} for i in order]
        return {
            "node_id": node_id, "available": True,
            "label": "Graph-structure explanation from a secondary model (GAT, layer-1 attention averaged "
                     "over heads). It is NOT the source of the risk score, which comes from XGBoost "
                     "Tier-1+2; GAT's own PR-AUC on this task is 0.2557 vs XGBoost's 0.7482.",
            "gat_probability": float(g["prob"][node_id]), "xgb_score": float(self.scores[node_id]),
            "n_incoming_neighbors": int(len(incoming)),
            "top_attended_in_neighbors": neighbors,
            "note": None if len(incoming) else "No incoming edges: this node only sends payments, so GAT "
                                               "has no neighbors to attend to.",
        }

    def node_properties(self, node_id):
        s = self.score(node_id)
        return {k: s[k] for k in ("node_id", "tx_id", "time_step", "score", "verdict", "ground_truth", "split")}

    def top_scored(self, n=50, split="test"):
        """Highest-scored nodes in a split (used to seed the cases table with real rows)."""
        mask = self.time_step >= 36 if split == "test" else np.ones(len(self.scores), dtype=bool)
        idx = np.flatnonzero(mask)
        return [int(i) for i in idx[np.argsort(-self.scores[idx])][:n]]
