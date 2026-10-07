# Graph Data Exploration -- Elliptic Bitcoin Dataset

Loaded via `torch_geometric.datasets.EllipticBitcoinDataset`.

## Counts
- Nodes: 203769
- Edges: 234355
- Features per node: 165

## Label distribution
- licit: 42019 (20.62%)
- illicit: 4545 (2.23%)
- unknown: 157205 (77.15%)

Roughly 77% of nodes are unlabeled ("unknown"). These are excluded from train/test metrics
throughout this notebook (semi-supervised setting, not dropped because it was inconvenient).

## Degree distribution
- Mean degree: 2.300
- Median degree: 2.0
- Max degree: 473
- Nodes with degree 0: 0

Elliptic is known to have a power-law-ish degree distribution with a handful of very
high-degree hub nodes (large mixers/exchanges) sitting far above the median -- visible above
in the mean/median vs. max gap.

## Time steps
- 49 discrete time steps present, range 1-49.
- Nodes per step: mean 4158.6, min 1089, max 7880.
