"""Message-passing GNNs: GraphSAGE (same architecture as notebook 01) and RGCN."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, SAGEConv


class GraphSAGEFlex(torch.nn.Module):
    """2-layer GraphSAGE, identical to GraphSAGEFlex in notebooks/01_graph_baseline.ipynb."""

    def __init__(self, in_channels, hidden_channels=64, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


class RGCN(torch.nn.Module):
    """2-layer RGCN: a separate weight matrix per relation (mean-aggregated), plus a self weight."""

    def __init__(self, in_channels, num_relations, hidden_channels=64, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, 1, num_relations)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_type).squeeze(-1)
