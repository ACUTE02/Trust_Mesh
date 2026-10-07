"""CARE-GNN (Dou et al., CIKM 2020, "Enhancing Graph Neural Network-based Fraud Detectors against
Camouflaged Fraudsters"), reimplemented from the paper's description for full-batch PyG.

What is implemented, per the paper:
  1. Label-aware similarity: a one-layer classifier on node features predicts P(illicit). The
     distance between two nodes is the gap between their predictions, and the classifier is trained
     with its own supervised loss added to the main loss (weight `lambda_simi`).
  2. Similarity-aware neighbor selection: per relation, each node keeps its ceil(p_r * degree_r)
     closest neighbors (at least one) and drops the rest before aggregating.
  3. Threshold update ("RL module"): after each training epoch, per relation, if the average distance
     of the selected neighbors went down, p_r += tau; otherwise p_r -= tau. A relation stops updating
     once the last 10 rewards sum to within +/-2. The paper frames this as a Bernoulli MDP; there is
     no learned policy network in the paper either, and none here.
  4. Relation-aware aggregation: mean over selected neighbors per relation, each relation's
     embedding scaled by its p_r, concatenated with the node's own embedding, then linear + ReLU.

Deliberate simplifications (stated so the result isn't mistaken for a line-by-line port):
  - Similarity is computed once per forward pass from the *input features*, and the same neighbor
    selection is used for both layers. The paper recomputes distances per layer from layer embeddings.
  - Two aggregation layers (to match the 2-hop depth of the other GNNs) + a linear classifier.
  - Imbalance is handled with the same class-weighted BCE as every other model here, instead of the
    paper's minority-class-balanced mini-batch sampling. Training is full-batch.
  - Relations come from `relations.build_relations` (payment direction), because Elliptic has no
    domain relation types like the paper's datasets (e.g. Yelp R-U-R / R-T-R / R-S-R).
"""
import math

import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter


def top_p_per_target(dst, dist, p, num_nodes):
    """Boolean keep-mask over edges: for every target node, keep the ceil(p * in-degree) incoming
    edges with the smallest distance (always at least one when the node has any incoming edges)."""
    num_edges = dst.numel()
    if num_edges == 0:
        return torch.zeros(0, dtype=torch.bool, device=dst.device)
    order = torch.argsort(dist, stable=True)
    order = order[torch.argsort(dst[order], stable=True)]  # grouped by target, ascending distance
    dst_sorted = dst[order]
    counts = torch.bincount(dst_sorted, minlength=num_nodes)
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(num_edges, device=dst.device) - starts[dst_sorted]
    k = torch.clamp(torch.ceil(p * counts.float()), min=1).long()
    keep = torch.zeros(num_edges, dtype=torch.bool, device=dst.device)
    keep[order] = rank < k[dst_sorted]
    return keep


class CARELayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels * (num_relations + 1), out_channels)

    def forward(self, h, edge_index, edge_type, keep, thresholds):
        parts = [h]
        src, dst = edge_index
        for r in range(thresholds.numel()):
            sel = (edge_type == r) & keep
            agg = scatter(h[src[sel]], dst[sel], dim=0, dim_size=h.size(0), reduce="mean")
            parts.append(thresholds[r] * agg)
        return F.relu(self.lin(torch.cat(parts, dim=-1)))


class CAREGNN(torch.nn.Module):
    def __init__(self, in_channels, num_relations, hidden_channels=64, dropout=0.3,
                 lambda_simi=2.0, init_threshold=0.5, step_size=0.02, convergence_window=10):
        super().__init__()
        self.simi = torch.nn.Linear(in_channels, 1)
        self.layer1 = CARELayer(in_channels, hidden_channels, num_relations)
        self.layer2 = CARELayer(hidden_channels, hidden_channels, num_relations)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        self.lambda_simi = lambda_simi
        self.step_size = step_size
        self.convergence_window = convergence_window
        self.num_relations = num_relations
        self.register_buffer("thresholds", torch.full((num_relations,), float(init_threshold)))

        self._simi_logits = None
        self._epoch_avg_dist = None
        self._prev_avg_dist = [None] * num_relations
        self.rewards = [[] for _ in range(num_relations)]
        self.converged = [False] * num_relations
        self.threshold_history = []
        self.last_keep = None

    @torch.no_grad()
    def _select(self, simi_logits, edge_index, edge_type):
        prob = torch.sigmoid(simi_logits).squeeze(-1)
        src, dst = edge_index
        # Distance between the predicted class distributions [p, 1-p]: L1 = 2|p_i - p_j|; the
        # constant factor doesn't change the ranking, so it is dropped.
        dist = (prob[src] - prob[dst]).abs()
        keep = torch.zeros_like(dst, dtype=torch.bool)
        avg_dist = []
        for r in range(self.num_relations):
            idx = (edge_type == r).nonzero(as_tuple=True)[0]
            k = top_p_per_target(dst[idx], dist[idx], self.thresholds[r], prob.numel())
            keep[idx[k]] = True
            avg_dist.append(float(dist[idx[k]].mean()) if k.any() else math.nan)
        return keep, avg_dist

    def forward(self, x, edge_index, edge_type):
        simi_logits = self.simi(x)
        keep, avg_dist = self._select(simi_logits, edge_index, edge_type)
        if self.training:
            self._simi_logits = simi_logits
            self._epoch_avg_dist = avg_dist
        self.last_keep = keep

        h = self.layer1(x, edge_index, edge_type, keep, self.thresholds)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer2(h, edge_index, edge_type, keep, self.thresholds)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(-1)

    def auxiliary_loss(self, y, train_mask, pos_weight):
        """Supervised loss for the label-aware similarity classifier (called by the training loop)."""
        logits = self._simi_logits.squeeze(-1)[train_mask]
        return self.lambda_simi * F.binary_cross_entropy_with_logits(
            logits, y[train_mask].float(), pos_weight=pos_weight
        )

    @torch.no_grad()
    def on_epoch_end(self):
        """The paper's threshold update, driven by the average selected-neighbor distance."""
        for r, cur in enumerate(self._epoch_avg_dist):
            if self.converged[r] or math.isnan(cur):
                continue
            prev = self._prev_avg_dist[r]
            self._prev_avg_dist[r] = cur
            if prev is None:
                continue
            reward = 1 if prev - cur >= 0 else -1
            self.thresholds[r] = torch.clamp(self.thresholds[r] + reward * self.step_size, 0.0, 1.0)
            self.rewards[r].append(reward)
            window = self.rewards[r][-self.convergence_window:]
            if len(window) == self.convergence_window and abs(sum(window)) <= 2:
                self.converged[r] = True
        self.threshold_history.append([float(t) for t in self.thresholds])

    def diagnostics(self):
        return {
            "final_thresholds": [float(t) for t in self.thresholds],
            "converged": list(self.converged),
            "threshold_updates": [len(r) for r in self.rewards],
            "last_avg_selected_distance": self._epoch_avg_dist,
        }
