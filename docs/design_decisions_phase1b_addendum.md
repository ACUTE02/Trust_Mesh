

## Phase 1B additions

- **Convergence check (task 1):** still dropping noticeably in the final epochs -- the default 40-epoch run was undertrained; task 2's tuning pass below searches a wider epoch range with early stopping.
- **Fair tuning pass (task 2):** best GraphSAGE config {'hidden_dim': 128, 'lr': 0.0019010245319870357, 'dropout': 0.24607232426760908, 'max_epochs': 49} reached
  0.5479 PR-AUC on the temporal test set (vs. 0.5115 untuned,
  vs. 0.7448 for the XGBoost Tier-1 baseline).
- **GAT (task 3):** 0.2557 PR-AUC on the temporal test set. Attention-based
  explanation example: node 154134 (predicted prob
  0.9999), top attended neighbors:
  [{'node': 152455, 'attention_weight': 1.1295293234070414e-06}].
- **Tier-2 graph-feature ablation (task 4):** XGBoost on raw features + {degree, out/in-degree ratio,
  PageRank, clustering coefficient} reached 0.7474 PR-AUC, vs.
  0.7448 for Tier-1-only.
- **Robustness check (task 5):** best overall model was **xgboost_tier1_tier2**
  (0.7482 +/- 0.0030 PR-AUC across seeds [42, 43, 44]).

### Final finding

On this temporal split, neither GraphSAGE (tuned: 0.5479) nor GAT (0.2557) beats a plain tuned XGBoost on raw features (0.7448); adding classical graph features (degree, out/in-degree ratio, PageRank, clustering) closes essentially the whole gap and then some, at 0.7474 PR-AUC. On Elliptic, under a realistic time-based evaluation, graph *structure* (via message-passing GNNs) does not earn its complexity here -- graph-derived tabular features captured with a much simpler and cheaper model do the same job or better.
