"""Tier-2 classical graph features (Phase 1B ablation), computed exactly as in notebook 01."""
import networkx as nx
import numpy as np
from torch_geometric.utils import degree, to_undirected

TIER2_FEATURE_NAMES = ["degree", "out_in_degree_ratio", "pagerank", "clustering"]


def tier2_graph_features(edge_index, num_nodes):
    ei = edge_index.cpu()
    ei_u = to_undirected(ei)
    degree_arr = degree(ei_u[0], num_nodes=num_nodes).numpy()

    g_undirected = nx.Graph()
    g_undirected.add_nodes_from(range(num_nodes))
    g_undirected.add_edges_from(ei_u.t().numpy().tolist())

    g_directed = nx.DiGraph()
    g_directed.add_nodes_from(range(num_nodes))
    g_directed.add_edges_from(ei.t().numpy().tolist())

    in_deg = dict(g_directed.in_degree())
    out_deg = dict(g_directed.out_degree())
    in_arr = np.array([in_deg[i] for i in range(num_nodes)], dtype=np.float32)
    out_arr = np.array([out_deg[i] for i in range(num_nodes)], dtype=np.float32)
    io_ratio = (out_arr + 1.0) / (in_arr + 1.0)  # +1 smoothing avoids division by zero

    pagerank = nx.pagerank(g_directed, tol=1e-4, max_iter=100)
    pagerank_arr = np.array([pagerank[i] for i in range(num_nodes)], dtype=np.float32)

    clustering = nx.clustering(g_undirected)
    clustering_arr = np.array([clustering[i] for i in range(num_nodes)], dtype=np.float32)

    return np.stack([degree_arr, io_ratio, pagerank_arr, clustering_arr], axis=1)
