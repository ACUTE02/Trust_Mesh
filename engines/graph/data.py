"""Elliptic loading and the temporal split shared by every graph-engine model."""
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset

EXPECTED_NODES = 203769
EXPECTED_FEATURES = 165
N_LOCAL_FEATURES = 93  # features 0-92 are local; 93-164 are precomputed 1-hop neighborhood aggregates
LICIT, ILLICIT, UNKNOWN = 0, 1, 2
TRAIN_STEPS = (1, 30)
VAL_STEPS = (31, 35)
TEST_STEPS = (36, 49)


def load_elliptic(root):
    dataset = EllipticBitcoinDataset(root=str(root))
    data = dataset[0]
    assert data.num_nodes == EXPECTED_NODES, f"unexpected node count: {data.num_nodes}"
    assert data.num_node_features == EXPECTED_FEATURES, f"unexpected feature count: {data.num_node_features}"

    # PyG drops txId and time_step when it builds data.x, so the time step can't be read from x
    # (data.x[:, 0] is a normalized feature). Take it from the raw features CSV, whose row order
    # matches data.x / data.y.
    features_csv = next(p for p in dataset.raw_paths if "feat" in os.path.basename(p).lower())
    steps = pd.read_csv(features_csv, header=None, usecols=[1]).iloc[:, 0].to_numpy()
    data.time_step = torch.from_numpy(steps).long()
    assert len(data.time_step) == data.num_nodes, "time_step row count doesn't match data.x"
    return data


def make_synthetic_elliptic(nodes_per_step=20, num_features=EXPECTED_FEATURES, seed=0):
    """Tiny Elliptic-shaped graph for local smoke tests only -- never used for reported numbers.
    Same attributes and label coding as `load_elliptic`, edges only within a time step (like
    Elliptic), a few hub nodes, and label/homophily signal so models have something to learn."""
    g = torch.Generator().manual_seed(seed)
    num_steps = TEST_STEPS[1]
    n = nodes_per_step * num_steps
    time_step = torch.arange(1, num_steps + 1).repeat_interleave(nodes_per_step)

    y = torch.full((n,), UNKNOWN, dtype=torch.long)
    r = torch.rand(n, generator=g)
    y[r < 0.40] = LICIT
    y[r < 0.12] = ILLICIT

    x = torch.randn(n, num_features, generator=g)
    x[:, :5] += 1.5 * (y == ILLICIT).float().unsqueeze(1)

    src, dst = [], []
    for t in range(1, num_steps + 1):
        idx = (time_step == t).nonzero(as_tuple=True)[0]
        hub = idx[0]
        for i in idx.tolist():
            if i != hub.item():
                src.append(i)
                dst.append(hub.item())
        for _ in range(len(idx)):
            a, b = idx[torch.randint(len(idx), (2,), generator=g)].tolist()
            if a != b and (y[a] == y[b] or torch.rand(1, generator=g).item() < 0.3):
                src.append(a)
                dst.append(b)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, time_step=time_step)


def temporal_masks(data):
    """Train / val / test masks over labeled nodes only (unknowns have no ground truth)."""
    labeled = data.y != UNKNOWN
    t = data.time_step

    def steps(lo, hi):
        return labeled & (t >= lo) & (t <= hi)

    return steps(*TRAIN_STEPS), steps(*VAL_STEPS), steps(*TEST_STEPS)


def class_pos_weight(y, train_mask):
    y_train = y[train_mask].cpu().numpy()
    return float((y_train == LICIT).sum() / max((y_train == ILLICIT).sum(), 1))


def split_counts(train_mask, val_mask, test_mask):
    return tuple(int(m.sum()) for m in (train_mask, val_mask, test_mask))


def edge_time_stats(edge_index, time_step):
    """How many edges connect nodes in the same time step vs. different steps."""
    ei = edge_index.cpu()
    same = int((time_step[ei[0]] == time_step[ei[1]]).sum())
    total = int(ei.size(1))
    return {"edges": total, "same_step": same, "cross_step": total - same}


def to_numpy_features(data):
    return data.x.cpu().numpy().astype(np.float32)
