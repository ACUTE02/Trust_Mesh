"""Relation typing for Elliptic's untyped edges (RGCN and CARE-GNN).

Elliptic edges are directed payment flows: u -> v means an output of transaction u is spent by v.
The GraphSAGE / GAT / GIN models in notebook 01 passed messages only along that direction, so
each node saw its upstream (payer-side) neighbors and never its downstream ones.

Schemes (all label-free and available at inference time, so no label leakage into the graph):
  single     -- both directions, one relation. Control: same edges as `direction`, no typing.
  direction  -- relation 0: message from an upstream neighbor (original edge u -> v)
                relation 1: message from a downstream neighbor (reversed edge v -> u)
  direction_hub -- `direction` x whether the message source is a hub (undirected degree at or
                above the given percentile), 4 relations. Hubs are likely services such as
                exchanges or mixers, whose neighbors may be less informative.
"""
import torch
from torch_geometric.utils import degree, to_undirected

RELATION_SCHEMES = ("single", "direction", "direction_hub")


def hub_degree_threshold(edge_index, num_nodes, percentile=99.0):
    deg = degree(to_undirected(edge_index.cpu())[0], num_nodes=num_nodes)
    return float(torch.quantile(deg, percentile / 100.0))


def build_relations(edge_index, num_nodes, scheme, hub_percentile=99.0):
    """Returns (edge_index, edge_type, relation_names, info)."""
    ei = edge_index.cpu()
    rev = ei.flip(0)
    both = torch.cat([ei, rev], dim=1)
    direction = torch.cat([torch.zeros(ei.size(1), dtype=torch.long), torch.ones(rev.size(1), dtype=torch.long)])

    if scheme == "single":
        return both, torch.zeros(both.size(1), dtype=torch.long), ["any"], {}
    if scheme == "direction":
        return both, direction, ["from_upstream", "from_downstream"], {}
    if scheme == "direction_hub":
        threshold = hub_degree_threshold(ei, num_nodes, hub_percentile)
        deg = degree(to_undirected(ei)[0], num_nodes=num_nodes)
        source_is_hub = (deg[both[0]] >= threshold).long()
        edge_type = direction * 2 + source_is_hub
        names = ["from_upstream", "from_upstream_hub", "from_downstream", "from_downstream_hub"]
        info = {
            "hub_percentile": hub_percentile,
            "hub_degree_threshold": threshold,
            "hub_nodes": int((deg >= threshold).sum()),
        }
        return both, edge_type, names, info
    raise ValueError(f"unknown relation scheme: {scheme}")


def relation_counts(edge_type, relation_names):
    counts = torch.bincount(edge_type, minlength=len(relation_names))
    return {name: int(c) for name, c in zip(relation_names, counts)}
