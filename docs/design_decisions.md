# Design Decisions — Phase 1 Graph Engine

## Notebook logic: inline vs. importing `engines/graph/*.py`

**Decision:** `notebooks/01_graph_baseline.ipynb` duplicates the minimal logic inline rather than
importing from `engines/graph/*.py` modules.

**Why:** the notebook's only job is to run unmodified on Kaggle by upload (per
`docs/PHASE_1_GRAPH_ENGINE.md`, "Required deliverable format"). Kaggle's default upload flow
doesn't bring along sibling files from this repo — making the notebook depend on
`engines/graph/*.py` would mean either (a) getting internet-enabled `pip install` of this repo
working reliably from a private/uncommitted repo on Kaggle, or (b) manually uploading each module
file alongside the notebook and hoping the working directory lines up. Both are extra failure
points for zero benefit, since the whole point of Phase 1 is one self-contained `.ipynb`. Inline
duplication trades a small amount of code duplication for "click Run All and it works."

`engines/graph/*.py` (the testable, version-controlled counterpart) is a separate, not-yet-built
deliverable — see the open items below.

## GraphSAGE training: full-batch by default, neighbor sampling only as an OOM fallback

**Decision:** train GraphSAGE full-batch (whole graph, one forward/backward pass per epoch) by
default, and only fall back to `NeighborLoader` mini-batches if the full-batch pass raises
`torch.cuda.OutOfMemoryError`.

**Why:** the Elliptic graph is small (203,769 nodes, 234,355 edges, 165 features) — comfortably
under a few hundred MB on any Kaggle GPU — so neighbor sampling buys nothing here and costs a real
dependency (`pyg-lib` or `torch-sparse`, both built against an exact torch+CUDA version) that isn't
reliably installable against Kaggle's current torch build. See the bug log below — this was found
the hard way, not decided up front.

## Known bugs hit and fixed during Phase 1

### 1. Time step read from the wrong place

`EllipticBitcoinDataset.process()` (PyG's loader) builds `data.x` from only the 165 real
`trans_feat_*`/`agg_feat_*` columns — it does **not** keep `txId` or `time_step` as a feature
column. The first attempt read `data.x[:, 0]` assuming it was the time step; it was actually a
normalized feature value, silently producing a near-empty, meaningless temporal split (889 train /
0 val / 1 test nodes, 0% train positives). Fixed by reading the raw
`elliptic_txs_features.csv` PyG already downloads and taking its `time_step` column (index 1) in
the same row order used to build `data.x`/`data.y`, with an assertion on row-count alignment and a
sanity print showing illicit nodes actually spread across time steps 1–49.

### 2. `NeighborLoader` ImportError on Kaggle's torch build

`NeighborLoader` requires a working `pyg-lib` or `torch-sparse` install matched to the exact
torch+CUDA build. On a fresh Kaggle GPU notebook running `torch 2.10.0+cu128`, no prebuilt wheel
existed yet at `data.pyg.org`'s wheel index, so `NeighborSampler` raised `ImportError:
'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` the first time it was used —
even though the earlier `pip install` step had exited 0. Fixed by making neighbor sampling an
on-demand fallback (see above) instead of the default path, so the common case never needs the
fragile dependency at all.

## Phase 1B additions

Extends the Phase 1 comparison per `docs/PHASE_1B_GRAPH_ENGINE_DEPTH.md`, same notebook
(`notebooks/01_graph_baseline.ipynb`), all five tasks run in one Kaggle GPU session. Numbers below
are from the final clean re-run (after the GAT target-node fix).

- **Convergence check (task 1):** the default GraphSAGE run (hidden=64, lr=0.01, 40 epochs) was
  still dropping noticeably at epoch 40 (loss 1.3781 → 0.2488, still falling) — it was
  undertrained, not just genuinely weaker.
- **Fair tuning pass (task 2):** Optuna search (15 trials) over hidden_dim/lr/dropout/max_epochs
  with early stopping on the steps-31–35 validation slice. Best config (hidden_dim=128,
  lr≈0.0019, dropout≈0.246, max_epochs=49) early-stopped its final refit at epoch 102 (best val
  PR-AUC 0.8660) and reached **0.5479 PR-AUC** on the temporal test set (0.5487 on the Phase 1C
  re-run with identical code and seed: GPU nondeterminism gives about ±0.001 run-to-run) — better than the
  untrained default (0.5115) but still well short of the XGBoost Tier-1 baseline (0.7448). More
  training closed some of the gap, not all of it.
- **GAT (task 3):** **0.2557 PR-AUC** on the temporal test set — the weakest of all models tried.
  Attention-based "why was this flagged" example (after fixing target-node selection to require a
  real, non-self-loop incoming edge — the first attempt picked a node with no incoming edges in the
  directed graph, so it had nothing to attend to; it is not isolated, as the EDA shows no degree-0 nodes):
  node `154134`, predicted prob 0.9999, top attended neighbor `152455` (attention weight
  ≈1.13e-6). The near-zero attention weight on the one real neighbor it has is itself informative
  — GAT is barely using graph structure for this node, consistent with its poor overall PR-AUC.
- **Tier-2 graph-feature ablation (task 4):** XGBoost on raw features + {degree, out/in-degree
  ratio, PageRank, clustering coefficient (computed via NetworkX)} reached **0.7474 PR-AUC** —
  slightly *above* the Tier-1-only baseline (0.7448), and above every GNN tried.
- **Robustness check (task 5):** the overall winner was XGBoost Tier-1+Tier-2. Across 3 seeds
  (42 → 0.7474, 43 → 0.7523, 44 → 0.7450): **0.7482 ± 0.0030 PR-AUC** — a small, well-behaved
  spread.

### Final finding (Day 1's actual deliverable)

On this temporal split, neither GraphSAGE (tuned: 0.5479) nor GAT (0.2557) beats a plain tuned
XGBoost on raw features (0.7448); adding classical graph features (degree, out/in-degree ratio,
PageRank, clustering) closes essentially the whole gap and then some, at 0.7474 PR-AUC. On
Elliptic, under a realistic time-based evaluation, graph *structure* (via message-passing GNNs)
does not earn its complexity here — graph-derived tabular features captured with a much simpler
and cheaper model do the same job or better.

## Phase 1C decisions

Implemented as sections 13–16 of `notebooks/01_graph_baseline.ipynb`, per
`docs/PHASE_1C_GRAPH_ENGINE_ADVANCED.md`. Results from the Kaggle run follow in the next section.

- **Third architecture: GIN, not RGCN or GCN.** RGCN needs typed relations; Elliptic's edges are
  untyped, so RGCN would collapse to a single-relation GCN. GCN's symmetric-normalized mean
  aggregation is too close to GraphSAGE's mean aggregator to count as a different family. GIN (sum
  aggregation + MLP) is injective over neighbor multisets. It can tell "1 illicit neighbor" from
  "10", which mean-style aggregators blur. BatchNorm inside GIN's MLPs follows the original paper
  and guards against activation blow-up when summing over hub nodes (max degree 473). Trained with
  the same defaults GAT got in Phase 1B, so the three architectures start on equal footing.
- **GNNExplainer run twice.** Once on the best GNN so far (per spec), and once on GAT at node
  `154134`, the node whose raw attention weight (≈1.13e-6) was reported in Phase 1B. Same model and
  node, two explanation methods: an apples-to-apples comparison. Each explanation also gets a direct
  **edge-ablation check**: remove every edge touching the node and measure the prediction change.
  For a 2-layer GNN that cuts all neighborhood information, so "does the neighbor matter?" is
  measured rather than inferred from a mask value. Feature importance is split into Elliptic's
  local block (features 0–92) and its precomputed neighborhood-aggregate block (93–164). If the
  aggregate block carries the weight, the raw features already contain the graph signal that
  message passing would otherwise supply.
- **Camouflage subset definitions.** Two, both restricted to illicit test nodes with ≥1 real
  neighbor: *mostly licit/unknown* (<50% of real neighbors labeled illicit, the spec's wording)
  and *zero illicit neighbors* (strict). Each is compared against non-camouflaged illicit nodes
  that have neighbors. The verdict uses **recall at p>0.5**, which doesn't depend on prevalence.
  PR-AUC is reported against a shared licit negative pool, next to its chance baseline, because
  PR-AUC shifts with how many positives a subset has. Subsets under 20 nodes aren't reported as a
  finding.
- **Bigger search.** The notebook checks whether GraphSAGE's Phase 1B best `hidden_dim` sat at the
  top of `[32, 64, 128]` and, if so, widens it to `[64, 128, 256]`. The Phase 1B run picked 128, so
  widening is expected. 50 trials each. GAT never got an Optuna search in Phase 1B (only
  GraphSAGE did), so its 50-trial search is a *first* tuning pass, compared against the untuned
  0.2557. A change under 0.01 PR-AUC is reported as a plateau.
- **Partial-progress saving.** Each Phase 1C task writes to
  `artifacts/graph/results_phase1c_partial.json` as soon as it finishes, so a crash or Kaggle
  timeout part-way still leaves a record of what completed (the spec asks for done-vs-not-done
  reporting).

## Phase 1C results

All four tasks completed in one Kaggle GPU run.

| Model (temporal test, steps 36–49) | PR-AUC |
|---|---|
| XGBoost, raw + classical graph features (3 seeds) | **0.7482 ± 0.0030** |
| XGBoost, raw features | 0.7448 |
| GraphSAGE, 50-trial wider search (best GNN) | 0.5770 |
| GraphSAGE, Phase 1B tuned | 0.5487 |
| GAT, first tuning pass (50 trials) | 0.2910 |
| GIN (defaults) | 0.2654 |
| GAT (untuned defaults) | 0.2557 |

- **GIN (task 1): 0.2654**, near GAT and far below GraphSAGE. The extra expressiveness of sum
  aggregation doesn't help here: sum-over-neighbors doesn't find signal that mean-over-neighbors
  missed. (Train loss 1.75 → 0.37, no divergence.)
- **GNNExplainer vs. attention (task 2).** Best GNN at that point was Phase 1B GraphSAGE. It also
  flags node `154134` correctly, so both explanations use the same node. Three separate signals,
  which do *not* all agree:
  - *Edge ablation agrees with the attention weight.* Cutting the node's only edge changes the
    prediction by +0.0001 (GAT) and −0.00001 (GraphSAGE). The neighbor genuinely doesn't matter.
  - *GNNExplainer's edge mask disagrees.* It scores that same edge **0.928**. Here the mask isn't
    faithful to the model. A likely reason: the output doesn't change with the edge, so the mask
    gets no useful gradient and settles by regularization rather than by importance. So
    GNNExplainer's edge mask alone isn't safe as a "why was this flagged" answer. Pair it with an
    ablation check.
  - *Feature mask: the block split is uninformative, but the top features aren't.* The
    54% / 46% local-vs-aggregate share is roughly the feature-count ratio (93/165 = 56% local),
    because nearly all features sit at a ~0.16 mask floor. The notebook's auto-generated sentence
    ("the prediction rests on the node's own features, 54% local") overstates this. The real
    signal: the two standout features for **both** GAT and GraphSAGE are **129 and 141** (mask
    ≈0.64–0.78). Both are in Elliptic's precomputed neighborhood-aggregate block.

  Put together: for this node, neighborhood information does influence the prediction, but it
  arrives through features the dataset authors already aggregated, not through message passing.
  That fits with graph-derived tabular features beating GNNs. It's one node, though, so treat it as
  illustrative, not proven in general.
- **Camouflage (task 3), Phase 1B GraphSAGE, 901 illicit test nodes (0 isolated).** Camouflage is
  **not** a vulnerability here. If anything it runs the other way. Illicit nodes with *zero*
  illicit neighbors: recall **0.867** (n=686). Those with illicit neighbors: **0.777** (n=215). The
  "mostly licit/unknown" definition gives the same pattern (0.857, n=714, vs 0.802, n=187). Both
  gaps are under the 0.10 "sharp" threshold, but both point *against* the predicted weakness. The
  PR-AUCs (0.55 vs 0.25) aren't directly comparable: their chance baselines differ (0.045 vs
  0.015). **Definition caveat:** ~77% of nodes are unlabeled, so "no illicit neighbors" mostly means
  "neighbors are unlabeled" (686 of 901 illicit test nodes qualify). Camouflage is the norm in this
  subset, not the exception. This measures camouflage relative to *known* labels, not true hidden
  fraud.
- **Bigger search (task 4).**
  - *GraphSAGE:* Phase 1B's best `hidden_dim` (128) was at the boundary, so the range widened to
    `[64, 128, 256]`. 50 trials gave **0.5770** (+0.028 vs Phase 1B), which counts as a real change,
    not a plateau. But the new best is at the boundary *again*: `hidden_dim=256` and
    `max_epochs=100` are both range maxima. So this search hasn't plateaued, and widening further
    could still help. At +0.028 per widening, though, closing a 0.17 gap to XGBoost is implausible.
  - *GAT:* first tuning pass, 50 trials: **0.2910** (+0.035 vs untuned).
  - *Validation-to-test gap:* every GNN drops sharply from validation to test (GraphSAGE val 0.876
    → test 0.577; GAT val 0.727 → test 0.291). The steps-31–35 validation slice overstates
    performance on steps 36–49. So "best on validation" is a weak predictor of "best on test", which
    is one more temporal-shift symptom, like the Phase 1 random-vs-temporal gap. Not measured here,
    but consistent with the Elliptic paper (Weber et al., 2019): a dark-market shutdown around
    step 43 is reported to break models trained on earlier steps.

### Closing paragraph (Phase 1C)

Phase 1C does not change the Phase 1B headline; it survives more scrutiny. A third architecture
(GIN, 0.265), a 50-trial wider GraphSAGE search (0.577) and GAT's first tuning pass (0.291) leave
the best message-passing GNN 0.17 PR-AUC behind XGBoost on raw + classical graph features
(0.748 ± 0.003). The explainability check suggests why. On the explained illicit node, cutting its
only edge doesn't move either GNN's prediction, and both models lean hardest on two of Elliptic's
*precomputed neighborhood-aggregate* features. Neighborhood information is already in the raw
features, so message passing has little left to add. Two supporting results: camouflage (fraud
among non-illicit neighbors) doesn't hurt the GNN, and GNNExplainer's edge mask proved unreliable
without an ablation check. The honest framing for Day 1: on Elliptic, the graph matters, but a
tabular model already captures what matters about it.

### Notebook wording corrections

The notebook's auto-generated `comparison_to_attention` and `phase1c_closing_paragraph` strings
from this run treat a 0.928 edge-mask value as agreement, and treat the 54% local share as
evidence. Corrected versions are stored alongside the originals in `results.json`
(`comparison_to_attention_corrected`, `phase1c_closing_paragraph_corrected`). The notebook logic
has been fixed so future runs generate the corrected framing.

## Phase 1D decisions

Per `docs/PHASE_1D_GRAPH_ENGINE_FULL.md`. The Voice Engine (EchoAuth) is deferred and its two days
go to the Graph Engine; see `project.md` §4. Numbers get added here only after a real run.

### Code layout: new notebook, logic in `engines/graph/`
- **`notebooks/02_graph_full.ipynb`, not more cells in notebook 01.** Notebook 01 is already too
  large for the editing tools, and re-running Phase 1–1C (with its Optuna searches) before every
  Phase 1D run would burn hours of Kaggle GPU for nothing.
- **Logic lives in tested modules.** `engines/graph/` holds `data`, `features`, `baseline`,
  `relations`, `gnn`, `care_gnn`, `train`, `evaluate`. This clears the Phase 1 open item for
  everything Phase 1D uses. `scripts/build_notebooks.py` inlines the modules into the notebook,
  so Kaggle still needs just one uploaded file, and it resolves the Phase 1 inline-vs-import
  question without duplicating code. The notebook source is `notebooks/src/02_graph_full.py`; the
  `.ipynb` is generated.
- **Local verification, CPU only, no real data:** `tests/test_graph_modules.py` includes a
  hand-worked check of CARE-GNN's top-p selection. `tests/smoke_notebook_02.py` runs every cell of
  the built notebook on a synthetic Elliptic-shaped graph. Both pass. Smoke numbers are meaningless
  and never reported, and the leaderboard script refuses to merge a smoke-test results file.
- **Reproduction check before any new model.** Notebook 02 re-derives the split counts and
  `pos_weight`, and refits XGBoost Tier-1 / Tier-1+2 and wide-search GraphSAGE from their stored
  best params. It checks them against the earlier runs (XGBoost within 0.002, GraphSAGE within
  0.02). A mismatch is flagged in the closing paragraph, since the unified leaderboard is only valid
  if notebook 02 reproduces notebook 01's setup.

### RGCN edge typing
- **Cross-time-step edges (spec "Type B").** Elliptic's edges are reported to connect only
  transactions in the same time step, which would make this type empty. The notebook counts
  same-step vs. cross-step edges rather than assuming. The chosen scheme doesn't depend on the answer.
- **Labeled-adjacent vs. unknown-only (spec "Type C"): rejected.** Typing edges by endpoint labels
  leaks test labels into the graph structure. And since test-period labels are hidden during
  training, the relation mix would differ systematically between the train and test periods, a
  built-in distribution shift.
- **Used: payment direction.** Relation 0 is a message from an upstream neighbor (original edge
  u → v); relation 1 is from a downstream neighbor (the reversed edge). It's label-free, available
  at inference, and meaningful for money flow. Notebook 01's GNNs only passed messages along the
  original direction, so they never saw downstream neighbors.
- **Variant: direction × hub source (4 relations).** Hubs are nodes at or above the 99th percentile
  of undirected degree (label-free). They are plausibly exchanges or mixers, whose messages may be
  less informative.
- **Control: `rgcn_single`.** The same bidirectional edges with one relation type, so
  `rgcn_direction − rgcn_single` isolates the effect of typing from the effect of adding reverse
  edges.
- **Protocol:** all RGCNs use the "defaults" protocol (hidden 64, dropout 0.3, lr 0.01, 40 epochs),
  the same as GraphSAGE / GAT / GIN defaults in notebook 01.

### CARE-GNN: which version
Reimplemented from the paper's description (Dou et al., CIKM 2020), not ported from the authors'
code. See the `engines/graph/care_gnn.py` docstring.

- **Implemented as described:**
  - *Label-aware similarity:* a one-layer classifier predicts P(illicit); distance between nodes is
    the gap in their predictions; the classifier is trained with its own loss, weight λ = 2.
  - *Top-p neighbor selection:* per relation, each node keeps its ceil(p_r × degree) closest
    neighbors, at least one.
  - *Threshold update:* each epoch, per relation, p_r moves by ±τ depending on whether the average
    selected-neighbor distance fell; a relation stops updating once its last 10 rewards sum within
    ±2. This is the paper's "RL module", a reward-driven update rule. Neither the paper nor this
    code learns a policy network.
  - *Relation-aware aggregation:* per-relation mean, scaled by p_r, concatenated with the node's own
    embedding.
- **Simplified:**
  - Similarity comes from input features and is shared by both layers. The paper recomputes it per
    layer from embeddings.
  - Class-weighted BCE, consistent with every other model here, instead of the paper's
    balanced mini-batch sampling.
  - Training is full-batch.
  - Relations are payment direction, since Elliptic has no domain relation types.
- **Relation scheme fixed in advance,** not picked from RGCN test scores. Trained with early
  stopping on validation, like the wide-search GraphSAGE it's compared against.
- **Selector diagnostic:** for illicit and licit test targets, the share of incoming edges kept,
  split by neighbor label. This shows whether the selector actually prunes licit neighbors of
  fraud nodes, the core CARE-GNN idea.

### Camouflage before/after metric
The subset definitions are identical to Phase 1C. What changes is the metric. Phase 1C compared
subsets *within one model* at a fixed 0.5 cutoff. Phase 1D compares *different models* (CARE-GNN vs
wide-search GraphSAGE vs XGBoost), which are calibrated differently, so a fixed cutoff isn't fair.
Recall is compared at each model's threshold that flags **5% of licit test nodes** (matched
false-positive rate), with a normal-approximation standard error. A difference within ~2 SE is
reported as no clear difference.

### Expanded search, robustness, leaderboard
- **Which GNN gets the search:** "best GNN" is picked by test PR-AUC, as the spec defines it. That
  makes the winner's number slightly optimistic. The search itself selects on validation only, and
  the 3-seed robustness check guards against over-reading one run.
- **Search budget:** up to 100 trials, capped at 4 hours, with the completed-trial count recorded.
  Widened ranges (e.g. lr down to 1e-4, GraphSAGE hidden_dim up to 512, epochs up to 300), since
  Phase 1C's best config sat at the edge of its search space. A boundary check runs again afterwards.
- **Robustness:** 3 seeds on the final best GNN, each re-run with exactly the protocol that
  produced its leaderboard number.
- **Unified leaderboard:** `scripts/build_graph_leaderboard.py` merges `results.json` (Phase 1–1C,
  untouched) and `results_phase1d.json` into `docs/graph_leaderboard.md`. One split, one metric,
  with a protocol column so defaults / tuned / searched models aren't silently mixed. With
  `--merge` it also copies Phase 1D into `results.json["phase1d"]`.

## Phase 1D results

Full Kaggle run, `notebooks/02_graph_full.ipynb`. All 5 reproduction checks against notebook 01
passed (split counts, `pos_weight`, both XGBoost refits within 0.002, wide-search GraphSAGE within
0.02), so this leaderboard is a like-for-like comparison. See `docs/graph_leaderboard.md` for the
full table.

- **Edges are 100% same-time-step.** `cross_step_edges: 0` — Elliptic really does only connect
  transactions within one time step, confirming the assumption behind rejecting the spec's
  cross-time-step relation type; it would have been an empty relation.
- **RGCN.** `direction` (0.4490) beat the untyped `single` control (0.4397) by **+0.0093** — typing
  by payment direction helped a little. `direction_hub` (0.3393) was **worse than both** by over
  0.10 — splitting hub-sourced messages into their own relation hurt, most likely because it thins
  each relation's edge count without adding signal (the paper's hub intuition doesn't transfer
  here). All three RGCNs trail every GraphSAGE variant, default and tuned alike (best RGCN 0.4490
  vs GraphSAGE default 0.5115) — on Elliptic, RGCN's separate per-relation weights cost more in
  parameters/overfitting than the relation split gains.
- **CARE-GNN: 0.4576** — below GraphSAGE (0.577–0.591) and even below `rgcn_direction`, though
  above the other two RGCNs. The threshold update converged on both relations (final p ≈ 0.24 for
  upstream, 0.50 for downstream, i.e. it settled on keeping fewer upstream neighbors and left
  downstream unchanged). The selector diagnostic shows it did discriminate: for illicit test
  targets it keeps a similar share of licit/illicit/unknown neighbors (0.68–0.76 — no
  suppression), but for **licit** test targets it aggressively drops illicit neighbors (kept share
  0.29 vs 0.73 for licit/unknown neighbors of the same targets). So the mechanism is working almost
  backwards from the paper's intent for this task: the selector protects *licit* nodes from
  contamination by nearby fraud, rather than protecting the *fraud* signal from camouflage. That
  tracks with the simplification of computing similarity once from raw features rather than
  per-layer from evolving embeddings — the model likely uses feature-similarity as a proxy for
  "how normal do I look" rather than a working camouflage detector.
- **Camouflage before/after, matched 5% FPR (`zero_illicit_neighbors/camouflaged`, n=686):**
  CARE-GNN 0.6210, GraphSAGE (wide search) 0.6429, XGBoost Tier-1+2 0.7726. CARE-GNN vs GraphSAGE:
  **−0.0219 (SE ≈0.026), no clear difference** — the targeted defense did not measurably improve
  recall on exactly the nodes it was built to help, and both trail XGBoost by roughly 0.13–0.15.
  **The targeted-defense hypothesis is not supported** by this implementation and this dataset.
- **Expanded search (100 trials, GraphSAGE):** 0.5794 (+0.0024 vs the Phase 1C wide search) — a
  much smaller gain than Phase 1C's own widening (+0.028), despite pushing `hidden_dim` to 512 (the
  top of the range) and `lr` down near the floor. Real diminishing returns, not a plateau exactly —
  the new best again sits at or near 3 of 4 search-space boundaries (`hidden_dim=512`,
  `dropout≈0.10`, `max_epochs≈289`), so a further widening might still move the number a little,
  but the trend across three widenings (0.549 → 0.577 → 0.579) says not by much.
- **3-seed robustness, best GNN (`graphsage_expanded_search`):** **0.5913 ± 0.0085** — consistent
  with the single-seed 0.5794, gap to XGBoost Tier-1+2 (0.7474 this run, 0.7482 ± 0.0030 over 3
  seeds in notebook 01) is **−0.156**, well outside the combined seed noise of both.

### Closing paragraph (Phase 1D)

The standing finding holds, more strongly than before. After a third relation-typing scheme
(RGCN, three variants), a proper CARE-GNN implementation with the paper's own threshold-update
rule, and a 100-trial search pushed to the edge of a widened hyperparameter space, the best
message-passing GNN (tuned GraphSAGE, 0.591 ± 0.009 PR-AUC over 3 seeds) still trails XGBoost on
raw + classical graph features (0.748 ± 0.003) by 0.156 PR-AUC — a bigger, not smaller, gap than
Phase 1C's 0.170 vs the wide-search GraphSAGE alone once RGCN and CARE-GNN are added to the
comparison, since both underperformed GraphSAGE. CARE-GNN's targeted camouflage defense also did
not close any part of the gap on exactly the subset it targets (camouflaged illicit nodes): its
recall there is statistically indistinguishable from plain GraphSAGE's, and its selector, on
inspection, appears to protect licit nodes from fraud contamination rather than protect fraud
signal from camouflage — a mechanism mismatch traceable to computing node similarity once from raw
features instead of the paper's per-layer embeddings. On Elliptic, under this evaluation, neither
more relation structure nor a purpose-built anti-camouflage architecture changes the Phase 1B
conclusion: tabular models on graph-derived features remain the stronger choice.

## Phase 2 decisions — Voice Engine

Per `docs/PHASE_2_VOICE_ENGINE.md`. The Voice Engine was deferred in Phase 1D and un-deferred on
2026-09-16. Built as `notebooks/03_voice_baseline.ipynb` + `engines/voice/`, following the same
modules-inlined-for-Kaggle pattern as Phase 1D. Results from the Kaggle run follow in the next section.

### Dataset: which ASVspoof copy, and what that costs us
The spec asked for the "~50MB HuggingFace sample subset". No such sample dataset was found. What
exists on HuggingFace is:
- `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` (and the near-identical `omkar177/ASVspoof2019_LA`)
  — parquet, no loading script, 71,237 clips, 4.39 GB, **evaluation partition only**.
- `LanceaKing/asvspoof2019` — has the official train/dev/eval partitions, but needs
  `trust_remote_code`, which current `datasets` versions no longer support. Not usable.

**Decision:** stream a controlled subset of `SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` (default
`N_CLIPS = 6000`, ~370 MB) instead of downloading 4.39 GB, and split it ourselves. The clip count is
recorded in `results.json` and printed into `docs/voice_data_exploration.md` so "started small" can't
quietly become the final claim.

**What this costs, stated plainly:** everything comes from the LA *evaluation* partition, so the same
spoofing systems (A07–A19) appear in both halves of our split. The official protocol tests against
*unseen* attacks (train on A01–A06). **Unseen-attack generalization is therefore not measured in this
phase**, and the EER here is optimistic relative to a proper ASVspoof submission.

### Speaker-disjoint split as the headline
A random clip-level split puts the same speaker in train and test, letting the model key on voice or
channel identity instead of spoofing artifacts. The `notes` field carries `speaker_id`, so the
headline split assigns whole speakers to one side only. The random-split model is trained anyway and
reported next to it, so the size of that leak is measured rather than assumed — the audio counterpart
of Phase 1's temporal-vs-random gap. Speakers are assigned whole, so a split can leave one side
single-class (EER undefined); the splitter reshuffles up to 12 times and, failing that, falls back to
a random split and reports `shared_speakers > 0` so the fallback is visible, never silent.

### Front-end: LFCC, not mel
LFCC is the front-end of the official ASVspoof 2019/2021 baselines. Its linear filter spacing keeps
resolution at high frequencies, which is where vocoder and TTS artifacts concentrate; mel spacing
compresses exactly that band because it is tuned to human perception. Config follows the baseline:
20 ms window / 10 ms hop, 70 linear triangular filters, 20 cepstral coefficients, plus first and
second deltas = 60 dimensions, over a fixed 4-second window.

**Implemented in plain torch, not `torchaudio`.** LFCC, the energy-based silence trim and the
resampler are ~100 lines in `engines/voice/preprocess.py`. That removes a hard dependency whose
version is coupled to torch (torchaudio is in maintenance mode and its decoding backend has moved),
keeps the local smoke test dependency-free, and makes the front-end unit-testable.

### Silence-leakage check: measured three ways
ASVspoof 2019 LA is known to leak the label through silence duration. The check is:
1. **Probe before trimming** — score clips by one timing measure alone (total duration, leading
   silence, trailing silence, speech duration, silence ratio) and compute that measure's EER. An EER
   far from 0.5 means timing alone is a shortcut. Both score directions are tried, and the stronger
   (lower-EER) one reported, since which direction favours bonafide is an empirical question.
2. **Probe after trimming** — leading/trailing silence is gone by construction, so those measures go
   constant and are reported as such; residual speech duration is still probed.
3. **The empirical confirmation the spec asks for** — the same LCNN is trained on untrimmed features
   and compared to the trimmed model. If leaving silence in improves EER, the model really was
   exploiting the shortcut. Fixed-length windowing means absolute duration can never reach the model
   directly; this comparison is about what survives inside the window.

### EER computed at the crossing, not the nearest point
`compute_eer` builds the FAR/FRR curves from `roc_curve` and **linearly interpolates between the two
operating points that bracket the crossing**, rather than reporting the nearest sampled point, which
is only an approximation. A unit test checks it against a brute-force 20,001-threshold sweep, plus
hand-worked cases (perfect separation = 0.0, perfect inversion = 1.0, a constructed 0.25 case). Single-class
input returns NaN with a note instead of a misleading number.

### Three runs, one variable each
`lcnn_trimmed_speaker_disjoint` (headline), `lcnn_untrimmed_speaker_disjoint` (silence comparison),
`lcnn_trimmed_random_split` (speaker-leakage comparison). One architecture, as in the graph engine —
no architecture zoo. Class-weighted BCE for the ~9:1 spoof/bonafide imbalance, early stopping on
validation EER.

### Out-of-domain spot-check is deliberately fragile-tolerant
`mueller91/In-The-Wild` has no documented column schema, so the loader detects the audio and label
columns and records the raw label values it saw. The whole step is wrapped so a failure records a
reason and the run continues; with 5–10 clips any EER is explicitly labelled indicative, not an
operating point.

## Phase 2 results

One Kaggle GPU run, 6000 clips streamed from the LA evaluation partition (619 bonafide / 5381 spoof,
67 speakers, 5.2 hours). Speaker-disjoint split: 3282 train / 893 val / 1825 test, 0 shared speakers.

| Model | Features | Split | Test EER | Bonafide acc | Spoof acc |
|---|---|---|---|---|---|
| **LCNN (headline)** | **trimmed** | **speaker-disjoint** | **0.2380** | 0.758 | 0.762 |
| LCNN | untrimmed | speaker-disjoint | 0.0097 | 0.989 | 0.990 |
| LCNN | trimmed | random | 0.2371 | 0.763 | 0.763 |

Accuracies are at each model's EER threshold. DET curves: `artifacts/voice/det_curves.png`.

### The silence shortcut is real and large; what it's made of is still a hypothesis
- **Timing probe, before trimming:** leading silence alone separates the classes at **EER 0.180**,
  silence share of the clip at 0.183. Bonafide clips average 1.10 s of leading silence vs 0.30 s for
  spoofed ones. This reproduces the known ASVspoof 2019 LA silence artifact on our subset.
- **After trimming:** leading/trailing silence is gone (constant measures); residual speech duration
  is only weakly predictive (EER 0.400).
- **The empirical confirmation:** the identical model trained on untrimmed audio scores **EER 0.0097**
  against 0.2380 trimmed — a 0.228 EER gap, roughly 24× fewer errors. The model *does* exploit
  silence when it can.
- **A hypothesis, not a finding:** the best timing-only measure tops out at EER 0.180, while the
  model given the actual silent samples reaches 0.0097. One explanation is that the silence's
  *content* matters beyond its length — e.g. the recorded noise floor/room tone of bonafide audio vs.
  near-digital silence in synthesised clips. But the gap doesn't prove it: the model also sees where
  speech sits in the window and how silence and speech interact, which a single timing number can't
  capture. **Phase 2B's silence ablations test this directly** (zeros at the original length,
  fixed-length zeros/noise, and a classifier on a fixed window of silence alone). Until they run, it
  stays a hypothesis.
- **Why 0.0097 must not be the reported number:** it is far better than the official LFCC-GMM
  baseline range, which for a small CNN is itself the sign of a shortcut. It is the number a pipeline
  without a trim step would have shipped. *(Correction, 2026-09-28: an earlier version of this section
  claimed 0.0097 "beat the winner of the 2019 challenge". That was wrong on the facts — the winner was
  around 0.2% EER — and wrong in principle: that result is on the official protocol, ours isn't, so no
  published-number comparison is valid.)*

### Speaker leakage: not detectable at this sample size
Random split 0.2371 vs speaker-disjoint 0.2380: a 0.0009 difference, from one seed on two different
test sets with only 182–194 bonafide clips each. **"Not detectable here" is the strongest claim this
supports** — not "there is no speaker leakage". It does contrast with the graph engine, where the
random split inflated GraphSAGE by a clear +0.26 PR-AUC. One possible reason is that spoof detection
keys on synthesis artifacts rather than voice identity, but that is untested. The speaker-disjoint
split stays the headline anyway; it's the defensible choice.

*(Correction, 2026-09-28: the Phase 2 notebook printed "random split looks better -> speaker leakage"
for this +0.0009, because its rule looked only at the sign of the difference. The rule now uses a
stratified bootstrap CI of the difference (1,000 resamples) and calls leakage only when the CI
excludes zero; otherwise it prints "no leakage detectable at this sample size". Unit-tested for both
branches. The Phase 2 closing paragraph also said the random-split EER "is not better than" the
speaker-disjoint one; it was slightly lower, just not significantly.)*

### What limits the headline
- **Data-limited.** Only 353 bonafide clips in training. Best validation EER came at epoch 3 while
  training loss kept falling (1.16 → 0.17), so the model overfits within a few epochs. Raising
  `N_CLIPS` is the most obvious lever.
- **Not comparable to published ASVspoof EERs.** The official protocol trains on attacks A01–A06 (the
  25,380-clip train partition) and tests on unseen A07–A19. Here, train and test share attack types,
  which makes the task easier, while training data is far smaller, which makes it harder. The
  0.2380 is valid for this setup only.

### Cross-dataset spot-check: did not run (bug, now fixed)
It failed with `LibsndfileError` opening
`zip://release_in_the_wild/10157.wav::hf://datasets/mueller91/In-The-Wild@…/release_in_the_wild.zip`.
In-the-Wild ships as one zip, so `datasets` returns a chained fsspec URL instead of bytes, and libsndfile
can't open that. `_decode_audio` now opens such URLs with fsspec and passes soundfile the bytes. A
regression test reproduces the exact failure with a local `zip://…::file://…` URL, and confirms plain
`sf.read` still fails on it. **Nothing is claimed about generalization until a re-run produces a number.**

### Closing paragraph (Phase 2)
An LCNN on LFCC features reaches **0.238 EER** on a speaker-disjoint test split drawn from a 6000-clip
subset of the ASVspoof 2019 LA evaluation partition. The most important result is not that number but
what it was protected from: the same model with silence left in scores 0.0097 EER — far better than
the official LFCC-GMM baseline range, which for a small CNN is itself the sign of a shortcut. Trimming
removes that shortcut, and the 0.228 EER gap measures how much of a naive result would have been
leakage. What the shortcut is made of (silence length vs. silence content) is a hypothesis that Phase
2B's ablations test. Speaker leakage was not detectable at this sample size. The voice engine's honest
number is modest and data-limited; unseen-attack and out-of-domain performance remain unmeasured.

## Phase 2B decisions — Voice Engine depth pass (real run complete; results in `E:\TrustMesh\PROJECT_LOG.md` → Phase 2B)

Per `docs/PHASE_2B_VOICE_DEPTH.md`. Built as `notebooks/04_voice_depth.ipynb` (source
`notebooks/src/04_voice_depth.py`) on new modules `engines/voice/features.py` and `ssl.py` plus extended
`data.py`, `preprocess.py`, `evaluate.py`, `model.py`. Numbers get added here only after a real run.

### Pre-flight corrections (done before any run)
- **Leakage label:** the sign-only rule that called a +0.0009 difference "speaker leakage" is replaced
  by a stratified bootstrap CI (1,000 resamples); "leakage detected" only when the CI excludes zero.
  Applied to notebook 03 and used for every comparison in notebook 04. Unit-tested for both branches.
- **Claims:** removed the false "0.0097 beat the 2019 challenge winner" everywhere; the silence-content
  explanation is labelled a hypothesis everywhere until the ablations test it.
- **Issue logs:** the repo `project.md` and `E:\TrustMesh\PROJECT_LOG.md` had independent numbering.
  Nothing had been overwritten. PROJECT_LOG.md's numbering is now canonical and both tables are identical.
  While syncing, one older factual error surfaced and was corrected: issue #3 said the Phase 1B GAT
  example node had no neighbors and that "Elliptic has many degree-0 nodes", but the EDA shows none —
  the node simply had no *incoming* edges in the directed graph.

### Task 1 — data
- **Download the parquet once instead of streaming.** Keeping every bonafide clip means touching every
  row anyway, and streaming (not training) was the slow part of Phase 2.
- **Every bonafide clip (~7.3k) + 4 spoof per bonafide**, at the 1:4 end of the spec's range to keep
  the run inside Kaggle's disk and time limits. `SPOOF_PER_BONAFIDE` is the knob; bonafide is never cut.
  The *loaded* counts are what gets recorded.
- **Feature cache, deviation from the letter of the spec:** the spec asked to cache trimmed and
  untrimmed LFCC. Trimmed features are cached as **int16 waveforms** instead, with LFCC computed on
  the GPU per batch. Run B needs waveform-level augmentation (noise, gain) that changes every epoch, so
  it can't read precomputed LFCC — and if A read cached LFCC while B read waveforms, A vs. B would
  differ in more than augmentation. Both read the same cache; the spec's intent (no re-decoding after a
  crash) is kept. Untrimmed and ablation features are cached as float16 LFCC. Every variant is quantized
  to int16 before its LFCC, so all comparisons see the same signal precision. The cache lives in
  `/kaggle/working/cache` as the spec asks and is deleted at the end of a successful run (after
  confirming the files are really gone) so the committed output stays small.

### Task 2 — attack-disjoint split
- **Attack IDs exist after all.** The parquet has none (`notes` holds only utterance ID, speaker ID and
  subset), but the same HuggingFace repo ships the official protocol file
  `protocols/ASVspoof2019.LA.cm.eval.trl.txt` (`SPEAKER UTTERANCE - ATTACK KEY`). That is authoritative
  metadata, not a proxy label. It is **only used if it reproduces every one of the 71,237 clips' label
  and speaker exactly** (`verify_protocol`); any mismatch disables the attack-disjoint protocol, and
  unseen-attack generalization is then reported as unmeasured.
- **P2 reuses P1's speaker partition.** Spoofs from held-out attacks are removed from train/val, and
  the P2 test set is bonafide + held-out-attack spoofs, which is a subset of the P1 test set. So a P1
  model ("attacks seen in training") and a P2 model ("attacks unseen") are compared on **identical
  clips** with a paired bootstrap. The P1 model's scores on those clips are sliced from its one test
  pass, so the test split is still touched once per model.
- **Held-out attacks:** 4 of the 13 eval attacks, one seeded draw. A draw is rejected only if it
  leaves some P2 split without a class (a structural constraint, found by the smoke test; the number of
  draws is recorded), never because of results. One draw is a limit: which attacks are held out
  matters, and this measures the gap for that draw only.

### Task 3 — models, one variable per run
- **A:** the Phase 2 LCNN, on the larger data. **B:** A plus augmentation, on training batches only:
  additive white noise at 10–40 dB SNR on half the clips, ±6 dB gain, and SpecAugment-style masking
  (2 frequency masks of up to 8 filters, 2 time masks of up to 40 frames). **Frequency masking is
  applied to the log linear-filterbank energies before the DCT**; masking rows of the cepstrum afterwards
  wouldn't correspond to any frequency band. Codec/compression simulation (optional in the spec) is not
  implemented.
- **C:** frozen `facebook/wav2vec2-xls-r-300m` (falls back to `wav2vec2-base` if it can't load, and
  records which), fp16, run once over every trimmed clip. Each layer's frames are pooled to mean and
  std and cached, and a head learns a softmax weight per layer plus a small MLP. The learned layer
  weights are reported. No fine-tuning.
- **Selection:** the lowest validation EER among the first seed of each family. The winner gets 3 seeds
  (A and C always do). Run D (the optional extra head/loss) is not implemented.

### Task 4 — silence ablations
Four variants, chosen so the comparison separates silence *length* from silence *content*:

| Variant | Silence length | Silence content |
|---|---|---|
| untrimmed (shortcut reference) | kept | kept |
| zeros at original length | kept | removed |
| fixed-length zeros / fixed-length noise (spec's a, b) | removed | removed |
| 0.2 s of genuine silence only (spec's c) | removed (fixed window) | kept, nothing else |

- **Deviation 1:** both leading *and* trailing silence are replaced, not only the leading one.
  Trailing silence is the second-strongest timing cue (EER 0.366 alone in Phase 2); leaving it in would
  confound the test.
- **Deviation 2:** "zeros at original length" is added. The spec's (a) removes length and content
  together, so on its own it can't say which one mattered.
- The silence-only classifier sees exactly 0.2 s of genuine silence per clip, so duration can't reach
  it. Only clips with that much silence are used, and trimmed/untrimmed are re-scored on the same clips.
- **Verdicts are only interpreted if the shortcut is significant.** Each one states what it doesn't show.

### Task 5 — rigour
- Every test EER carries a stratified 95% bootstrap CI (1,000 resamples; each class resampled within
  itself so the small bonafide class never vanishes from a resample). Same-clip comparisons use a
  paired bootstrap; different test sets use an unpaired one.
- **How much the CI shrank** is measured on one model: the headline model's own test scores,
  bootstrapped at Phase 2's class counts (182/1643) vs. the full Phase 2B test set. Comparing CIs from
  different models would mix sample size with model quality.

### Task 6 — In-the-Wild
The loader opens the 8 GB zip once through fsspec, reads labels from the archive's own `meta.csv`
(file/label columns detected, not assumed), and samples up to 100 clips per class. Every failed clip is
recorded with its error. It's scored with the headline model, the best model, the SSL model and the
untrimmed shortcut model. The untrimmed one is expected to degrade most, if the shortcut is
dataset-specific.

### Budget guard
Optional steps (extra seeds for B, attack-disjoint runs, ablations, In-the-Wild) check a 10-hour
budget first (Kaggle stops sessions at 12 h). A skipped step is recorded under `not_run` with the reason.

## Phase 2C decisions — SSL head upgrade, Tier 1 only (implemented; results pending the Kaggle run)

Per `docs/PHASE_2C_VOICE_SSL_DEPTH.md`. Extends `notebooks/04_voice_depth.ipynb` in place (same
notebook, same Kaggle session and `TIME_BUDGET_HOURS` clock as Phase 2B) rather than a new notebook,
so it reuses Phase 2B's frozen encoder, cache, splits and In-the-Wild data. Writes its own
`artifacts/voice/results_phase2c.json`, never merged into `results_phase2b.json` — `run()` now takes a
`store` parameter (results/models/scores dicts + a save function) so Phase 2C's calls never touch
Phase 2B's globals or output file.

### Pre-flight: the spec's Task 1 premise is false in this codebase (issue #23)
The spec assumed Model C pools mean-only ("should be mean pooling over time, per the Phase 2B spec")
and asked to confirm this rather than guess. `ssl_info["pooling"]` (logged since Phase 2B) already
reads `"per-layer mean + std over frames"` — `SSLEncoder.pooled` concatenates mean and std before
`SSLHead` ever sees it. Confirmed with the user before implementing: **Model D is reframed as a
mean-only ablation of Model C's own pooling** (does the std half actually earn its place?), not as
"adding std" to a model that already has it.

### Model D — mean-only ablation
`array_mean_only_source` slices the first `hidden_size` columns of the existing mean+std cache — no
new encoder pass, no new cache file. `SSLEncoder.mean_only()` is the same slice, used directly in a
unit test to pin the relationship between `pooled()` and the ablation.

### Model E — BiLSTM on frozen SSL tokens, one layer only, read on the fly (two deviations, both logged)
The spec asks for "the same layer-weighting mechanism as Model C" over the per-frame sequence. Caching
every layer's full frame sequence is infeasible: ~200 frames × 1024 dims × 25 layers × 2 bytes (fp16)
× ~37k clips is on the order of hundreds of GB, far past `/kaggle/working`. Model E therefore keeps
only the **single most-weighted layer from Model C's learned weights** (found in the pre-flight step)
and has no layer-weighting mechanism at all — there is nothing to weight with one layer. A single-layer
bidirectional LSTM (hidden size 64, chosen over a TCN because `torch.nn.LSTM` is a tested, batched
primitive and there was nothing to gain from a from-scratch temporal block here) mean-pools over time
into the same small MLP head as `SSLHead`. The parameter count added vs. Model C's head (`n_params` on
every record) is logged, so "did it help enough to justify the extra capacity" has an answer, not just
"did it help".

**Second deviation, forced by a real Kaggle failure (issue #25):** even a *single* layer's full-dataset
sequence cache is ~15 GB. A first implementation built it as a disk cache, same as every other SSL run
here — this failed Kaggle Version 2 ("tried to use more disk space than is available") once stacked on
top of Phase 2B's still-resident feature/SSL caches. Model E now reads its sequence **on the fly per
training batch** instead (`wave_ssl_sequence_source`, `features.py`) — the same frozen-encoder-per-
batch pattern Model F already uses for a different reason (augmentation must happen before the encoder
sees the clip). This costs GPU time, not disk; `results_phase2c.json`'s `model_e_cache_cost` key now
logs the disk size this choice avoids, not a built cache's actual size. `build_ssl_sequence_cache` in
`ssl.py` still exists and is still unit-tested (it may be useful for a smaller subset later), but the
notebook's main flow no longer calls it.

### Model F — augmentation on Model C's recipe
Same architecture as Model C, but training-time augmentation must happen before the frozen encoder's
forward pass, so `wave_ssl_source` re-runs the (frozen, `@torch.no_grad()`) encoder every training
batch instead of reading the pooled cache — the only Phase 2C model that does this, and the most
expensive one, so its `run()` calls use `SSL_BATCH` (16) instead of `BATCH_SIZE` (64) to bound memory.
Augmentation is Run B's menu (noise, gain, SpecAugment) plus one addition the spec calls out
explicitly, a codec/compression simulation that Phase 2B skipped for time: `simulate_low_bitrate`
(`preprocess.py`) is a documented **proxy**, not a real MP3 round-trip — no differentiable, per-batch-
fast MP3 codec exists — combining a linear-interpolation downsample/upsample (bandwidth loss) with
mu-law companding quantized to 6 bits (quantization noise), applied to a random 30% of clips per batch.

### Results and comparison
One leaderboard row per model (C carried over unchanged from Phase 2B's own results, plus D/E/F),
each with P1 test EER (3 seeds, mean ± std, 95% CI), the P2 seen→unseen gap with its CI (paired
bootstrap, same test clips as Model C's P2 comparison), and In-the-Wild EER with its CI — reusing
Phase 2B's own bootstrap/CI code (issue #16's fix), not re-derived. The In-the-Wild comparison against
Model C uses a **paired** bootstrap difference (same ~200 clips for every model), not just CI overlap,
so "beat Model C outside noise" has a real significance test behind it. The generated closing paragraph
states plainly if none of D/E/F beats Model C outside noise, since Model C (0.0175 in-domain, no
significant attack-disjoint gap, 0.10 In-the-Wild) is already a legitimate final answer.

### Local checks (all pass; Kaggle run pending after the issue #25 fix)
7 unit tests (`tests/test_voice_phase2c.py`) covering the mean-only slice, the (still-available but no
longer notebook-used) sequence cache utility, `SSLSeqHead` training, `wave_ssl_sequence_source`'s
on-the-fly path, and `wave_ssl_source`'s train/eval augmentation split. The existing
`tests/smoke_notebook_04.py` now also asserts on `results_phase2c.json` (all four leaderboard groups
present, the pre-flight's `model_c_pooling` value, the DET plot and per-model weight files). Caught and
fixed during local smoke testing: issue #24, a memmap left open by an earlier disk-cache version of
Model E blocked `shutil.rmtree` at the end of the run (same class of bug as issue #19). Caught and
fixed after a real Kaggle failure: issue #25, disk space exceeded — see Model E's decision above.

## Phase 3 decisions — Visual Engine Version 1 (implemented locally; no Kaggle run yet, no numbers)

Per `docs/PHASE_3_VISUAL_ENGINE.md` and `docs/HANDBOOK_ADDENDUM_01_ROTATING_HOLDOUT.md`. Built as
`notebooks/05_visual_v1.ipynb` (source `notebooks/src/05_visual_v1.py`) on `engines/visual/*.py`.
Every Visual Engine metric is **not yet run**. This section records decisions made while writing the
code, and what is still open until Task 0 reports real counts and timings.

### The notebook runs in stages and stops after Task 0 by default
`STAGE` is `task0` unless changed. The later stages (`scan`, `split`, `extract`, `heads`) each need
`PLAN` values that are `None` until they are decided from real counts (how many shards, the dedupe
tolerance, the minimum fold size, the caps, which families are "old"). A stage that is missing one
stops the run with a message; it does not raise, because a failed Kaggle version keeps none of its
output. So no embedding can be extracted on a guess.

### What was looked up before any Kaggle run (public Hugging Face API, read from this machine on 2026-10-07)
These shaped the code. They are page and API figures, not a Kaggle run; Task 0 re-measures all of them.
- OpenFake is stored as Parquet. Configs: `core` (train 2,300,034 rows; validation 58,453; test
  91,397) and `reddit` (test 36,087). Total 2,485,971 rows, matching the page the user saw.
- `core` has 636 Parquet files, 3,420.5 GB: 608 train files (named `train-000XX-of-00032-000YY`),
  15 validation, 13 test.
- Two train shards were opened by footer only: about 3,990 rows each in 58 row groups of about 69
  rows, about 128 MB per row group (about 1.85 MB per image). Both shards held the same mix of models,
  so shards look shuffled rather than sorted by generator (two shards only; Task 0 checks all).
- Columns beyond the ones the spec lists: `row_id`, `image_sha256`, `prompt_kind`, `prompt_model`,
  `prompt_revision`, `prompt_version`.
- The dataset card says this is a **v2 release**, with v1 kept at the tag `v1.0`, and that v2 added
  many generators. That is a possible explanation for issue #31 (the handbook's "18 generators,
  963,342 fakes"); it is not verified against v1's counts.
- The card states that synthetic images "were generated from a shared prompt bank derived from the
  captioned real images". So a real photo and the fakes made from its caption can share a prompt.
- OpenFake's own `core/test` split already holds out generators **and** uses different real-image
  sources (DOCCI, ImageNet) from train (LAION, Pexels). That mixes two shifts, so it is not used as a
  LOFO fold; whether it becomes an extra suite is undecided.
- The card's license field is `cc-by-nc-4.0`.

### Reading OpenFake: byte ranges, not a download and not row-by-row streaming
A row group's small columns (label, model, prompt, ...) are a few kilobytes sitting after its
~128 MB image column. `SparseParquet` reads the footer, computes the byte range of exactly the
column chunks wanted, and fetches only those. This gives the two things Task 0 asks for: counts of
the whole dataset from metadata alone, and a seek (any row group can be fetched without the ones
before it). A test asserts the reader makes zero reads outside the planned ranges.

`HttpRangeReader` remembers each file's signed storage URL after the Hub's first redirect, so the ~60
range reads of a shard cost the Hub one request. A full metadata scan is about 36,000 small reads;
asking the Hub for each would run into its request limits. The token is sent to the Hub only, never
to the storage host (tested with a stand-in session). Task 0 times this reader and the plain
`HfFileSystem` one and records both.

### Consequence of shuffled shards: quotas are filled by reading whole shards
If shards are a shuffled mix, the cheapest way to get N images of a rare generator is to read shards
in a fixed seeded order until N have gone by; fetching single rows would still download a whole
~128 MB row group each. Task 0 therefore reports, for quotas of 250 to 4,000 fakes per family, how
many shards the rarest family needs, and the projected hours per pass. The sampling plan is chosen
from that table, not before it.

### Two passes over the images, because the spec orders the shortcut baseline before the encoder
Pass 1 (`scan`) reads each image once for its metadata row: validity, file properties as received,
perceptual hash, and the byte count after the shared pipeline. No encoder. Pass 2 (`extract`) reads
the surviving images again to embed them. This costs the download twice. If Task 0's timings show two
passes do not fit, the alternative (one pass that hashes and embeds together, with the shortcut
baseline computed afterwards) changes the spec's order and is the user's decision, not made here.

### The shared pixel pipeline
decode -> upright RGB (EXIF orientation applied, EXIF then gone; alpha composited on white) -> resize
the shorter side to 224 and centre-crop 224x224 (CLIP's own geometry) -> one JPEG re-encode at
quality 95 -> CLIP mean/std. Both classes and every source go through it. Augmented views and test
degradations are applied to the original image, before this pipeline.
- Our resize is checked against the official `CLIPImageProcessor`: identical to its PIL-backed
  variant, and within one grey level of the default (torchvision-backed) one. Task 0 records both
  differences on Kaggle's installed version.
- Stored vector: CLIP's projected image embedding (`image_embeds`), not L2-normalized, float32. The
  head L2-normalizes. The dimension is read from the model on Kaggle (the handbook says 768).

### Cleaning and dedupe
- "Tiny" is a shorter side under 64 pixels (keeps FFHQ's 128 and ArtiFact's 200). "Grayscale-only"
  includes an RGB file whose three channels are identical.
- pHash is implemented here (32x32 grey -> DCT -> 8x8 low frequencies -> median threshold), no extra
  dependency. Near-duplicate search is exact up to distance 7 without comparing every pair.
- A duplicate cluster keeps one image; a cluster that mixes real and fake images is dropped whole,
  because the same picture with both labels has no usable label.
- **The tolerance is not chosen yet.** The `split` stage prints the number of pairs at each distance
  0-7 and stops until `PLAN["phash_tolerance"]` is set from that and from a look at real pairs
  (drawn only in a private notebook).

### Splits (addendum design)
- Groups: rows sharing a `prompt_id`, `source_image_id` or `identity_id`, transitively. For OpenFake
  the prompt is the link between a real photo and the fakes of its caption; a fake's
  `source_image_id` stays null (unknown) and is counted as null.
- A group is placed whole, by a hash of its key, into one of four partitions: train 0.6, val 0.1,
  known_test 0.1, holdout 0.2 (starting fractions). The real images of `holdout` are the reserved
  real pool.
- Fold for family F: train/val/known_test are those partitions without F's fakes; the unseen test is
  F's fakes **in the holdout partition** plus an equal number of reserved reals.
- **Why the unseen test uses only the holdout partition, though the addendum says "all fakes of
  family F are test-only".** All of F's fakes are indeed kept out of training. But F's fakes share
  prompts with other families' fakes and with real photos; scoring F's fakes from the train
  partition would put the same prompt on both sides of the fold, which the addendum's own leakage
  check 3 forbids. So those rows are unused in fold F (never trained on, not scored). The cost is a
  smaller unseen test per fold; the holdout fraction is the knob.
- `fold_role` is stored as one column per fold (`fold_role__<family>`), not one column, because a
  row has a different role in each fold.
- Family grouping: `metadata.FAMILY_RULES` is a **proposal** written from the dataset card (for
  example SD 1.x and 2.x together, Midjourney 6 and 7 together, all Flux variants together). Task 0
  prints every real `model` value with its proposed family and flags names that matched no rule.
  The grouping, the minimum fold size and the "old" families for the ablation are decided after that.
- The old-only ablation is scored on the modern folds' own unseen-test images, so it and the folds
  are compared on identical images (paired bootstrap).

### Shortcut baselines
Three feature sets, each with logistic regression and XGBoost: `raw` (file properties as received),
`pipeline` (the same seven properties after the shared pipeline, where only the re-encoded byte
count still varies), and `source` (the dataset alone). The gate is the CI rule from the spec: pass
only if the 95% bootstrap CI of ROC-AUC includes 0.5 for every `pipeline` and `source` baseline.
`raw` is reported beside it as the size of the shortcut the pipeline removes. If the gate fails, the
`extract` stage does not start unless `ACCEPT_RESIDUAL_SHORTCUT` is set after reading the numbers,
and the caveat is then written into the closing paragraph. FFHQ is real-only, so the `source`
baseline is expected to flag it; what to do with FFHQ is decided from that measured result.

### Heads, seeds, calibration
- Logistic regression (L-BFGS, C from {0.01, 0.1, 1, 10, 100} by validation ROC-AUC) and a
  one-hidden-layer MLP (256 units).
- **What a seed changes** is written into every result. Logistic regression is deterministic given
  its inputs; a seed only changes which images of the larger class are kept when the training set is
  balanced. If no balancing is needed, one run is reported and no seed variance is claimed. For the
  MLP a seed also sets the initial weights and batch order. CIs are computed on the first seed.
- Temperature, operating threshold and uncertain band are fitted on the fold's validation split
  (seen generators only). Starting settings, to be revisited with real data: threshold at 95% recall
  of validation fakes; band from the 99%-recall threshold to the 1%-false-positive threshold. ECE is
  reported before and after calibration, separately for known and unseen.

### Dataset terms (spec section 12a)
Everything per-image (metadata rows, embeddings, predictions) and all weights are written under
`artifacts/visual/private/`, which `.gitignore` excludes. `scripts/check_no_dataset_outputs.py`
fails if such a path is staged or if `.gitignore` loses a required entry, and can install itself as
a pre-commit hook once the folder is a git repository (it is not one yet). `README.md` has the
"Datasets and citations" section; a citation not yet looked up says so. Task 0 writes aggregate
counts only.

### Heavier pixel baselines (Task 9)
`RUN_SMALL_CNN` and `RUN_RESNET50` are off. When on, they train on a capped, class-balanced subset
held in memory as uint8 arrays (nothing on disk). The smoke test exercises both.

### Still open until Task 0 reports back
How many shards to read and whether two passes fit; the dedupe tolerance; the family grouping and
minimum fold size; per-generator caps; which families are "old"; whether FFHQ stays; the real
ArtiFact layout (the code assumes the README's `metadata.csv` with `image_path` and `target`, and
raises rather than guesses if that is wrong); Kaggle's real free disk; whether Deepfake-Eval-2024
access has been granted.

## Infra & Demo decisions (serving layer, fusion, Neo4j, frontend)

Spec: `docs/PHASE_2_INFRA_AND_DEMO.md` (in the parent `docs/` folder). Code: `serving/`, `frontend/`,
`docker-compose.yml`. Nothing here trains a model; it loads and serves the Phase 1D and Phase 2B/2C results.

### The graph risk score is XGBoost Tier-1+2, and the GNN only explains
The score `/graph/score` returns comes from `artifacts/graph/xgboost_tier1_tier2.json`. GraphSAGE and GAT
scored 0.26-0.59 PR-AUC against XGBoost's 0.7482, so serving a GNN score would contradict the project's own
Phase 1D finding. GAT stays for one thing XGBoost cannot give: which in-neighbours the layer-1 attention
weighted most. The explanation endpoint labels itself "secondary model, not the source of the score" in the
response text, and a unit test pins that wording. GAT (not GraphSAGE) is the explainer because GraphSAGE has
no per-edge attention to read; `gat_weights.pt` is the Phase 1B GAT whose attention example the notebook already
reported, and the served explanation reproduces that example exactly (node 154134, neighbour 152455).
The server checks itself at startup: the loaded weights must reproduce the test-period PR-AUC of the seed-42
model (0.74743; `/health` shows it).

### Verdict bands: a flag rate on known-benign data
"low / medium / high" needed a rule that is not a guess. `high` is the score above which 1% of known-benign examples
are flagged, `medium` the score above which 10% are. For the graph the benign examples are known-licit nodes in the
validation period (time steps 31-35), which the model never trained on, so that is a held-out rate. For voice the
benign examples are the bonafide clips of the in-domain test split, the same scores the model was evaluated on, so
**the voice thresholds are calibrated in-sample on the test set: "1% / 10% of bonafide flagged" describes how they were
chosen, not an out-of-sample false-flag rate.** A cleaner version would calibrate on a validation split.
The rule, not the number, is the decision: the numbers are written to `artifacts/serving/*.json` with the rule beside
them. Graph: high 0.5786, medium 0.0359. Voice Model D: high 0.6578, medium 0.0239 (Model C: 0.8910 / 0.0805).
A medium graph score is a low raw probability: XGBoost was trained with `scale_pos_weight` (about 8), so its
probabilities are inflated for the rare class and are a ranking, not a calibrated chance of fraud.

### Voice: Model D is served (seed 42), Model C is the fallback
Model D (mean-only pooling, 133,402 head parameters) is the default. The served file is
`phase2c_D_ssl_mean_only_P1_s42.pt`: **seed 42 of three Model D seeds**. The headline result remains the 3-seed mean,
EER 0.0166 +/- 0.0019; seed 42's EER recomputed from its saved test scores is 0.0194, a rough single-seed number and not
a new headline. The weights were not in the repository at first (they stayed in the Kaggle output), so the service
served Model C with `fallback_used` set until they were copied in on 2026-10-08; the fallback path remains and is tested.
D was not reconstructed locally: it is a head trained on cached Kaggle features and training is not allowed here.
Checks done when D went live: the checkpoint loads into the D head with all keys matched; `phase2c_test_scores.npz`
(seed 42) uses the same test split as Model C's scores and has bonafide scores above spoof scores. The scores were
**not** recomputed from audio (that needs the Kaggle feature cache).
The head outputs a logit for *bonafide* (label 1 is the EER target class in `engines/voice/data.py`), so the
voice risk score is `1 - sigmoid(logit)`; a test pins that and the calibration script asserts the polarity from
the saved scores. Pre-processing mirrors how the cache was built (trim, 4 s, int16 round-trip, then the
encoder's own normalisation). A model without calibration scores gets an explicitly labelled uncalibrated
threshold (0.5 / 0.2), never an invented one. The 8-clip demo result (all four bonafide low, all four spoof high
under D) is a plumbing and polarity check, not an accuracy measurement.

### Fusion: a weighted average with placeholder weights, nothing fit
`combined = w_graph * graph + w_voice * voice`, weights normalised, default 0.5 / 0.5, adjustable per call and in
the UI. Bands: medium from 0.30, high from 0.60. **These weights and bands are not fit to any labeled joint
data, because none exists**: Elliptic (transactions) and ASVspoof / In-the-Wild (voice) share no case. The two
scores are also not calibrated against each other (an inflated imbalance-weighted XGBoost probability against a
different model's spoof probability on a different distribution). So the combined number demonstrates the
mechanism, has no validated accuracy, and is never shown without both components beside it. A real calibration
needs cases that have both a known transaction-fraud outcome and a known voice-spoof outcome. A demo case
pairs a transaction with a clip that were never part of one fraud case; the UI and API say so.

### Neo4j: community 5 in Docker, ids shared with the models
Elliptic loads as `(:Tx)-[:SENDS_TO]->(:Tx)` with the real directed edges (234,355) and one node per transaction
(203,769) carrying its ground-truth label, split, XGBoost score and verdict, the four Tier-2 graph features and
the 165-feature vector. `Tx.id` is the row index the PyG graph and the models use (Elliptic's own txId is kept as
`tx_id`), so a score and its graph neighbourhood line up with no mapping table. The loader is idempotent and runs
as its own compose service. Queries: neighbours within N hops, known-illicit transactions within N hops, shortest
path between two transactions, and flagged / known-illicit neighbour counts. Hop counts are validated integers
(Cypher cannot parameterise a variable-length bound), capped at 4. If Neo4j is down, the scoring endpoints still
work and the query endpoints return 503 with the reason, instead of silently falling back to something else.

### The `cases` table
SQLite, one row per case, with nullable voice and combined columns: a graph-only case has NULL voice fields, a
voice-only case has a NULL node, and combined fields exist only when both inputs did. Scoring the same node again
refreshes its graph-only row; each node-plus-clip pairing is a new row. `flagged_cases` is a view of medium/high
cases ordered by risk (combined if present, else graph, else voice). `scripts/seed_cases.py` fills it with the
top-scoring test-period transactions through the API, so the table opens with real rows.

### Frontend: Streamlit first, replaced by React on 2026-10-08
The first version of the demo page was a single Streamlit file. The owner replaced it with a React frontend (see "Frontend decisions" below); Streamlit is no longer in the repository.

### Docker layout
Two images: the API/loader image (CPU torch, transformers, xgboost, torch-geometric) and a small nginx one holding the built React site.
Weights (`artifacts/`) and datasets (`data/`) are bind-mounted, not baked in: the image stays small and a weights
file dropped into `artifacts/voice/` takes effect on restart. Ports are bound to 127.0.0.1 only; the Neo4j password
in the compose file is a local-demo credential for a database that is not exposed beyond the machine.

## Terms findings (WP2, read 2026-10-08)

This is a record of what the official pages say, read on 2026-10-08. **It is not legal advice.** Where the
terms do not answer a question directly the row says "owner decision needed".

| Item | Licence as found | Source read | Citation requested |
|---|---|---|---|
| Elliptic Data Set | **CC BY-NC-ND 4.0** ("Attribution-NonCommercial-NoDerivatives 4.0 International") | Kaggle's own record for `ellipticco/elliptic-data-set`: https://www.kaggle.com/api/v1/datasets/view/ellipticco/elliptic-data-set (the HTML page https://www.kaggle.com/datasets/ellipticco/elliptic-data-set did not render for the reader tool; the API record is the same listing). Licence text: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en | From the dataset description: "[1] Elliptic, www.elliptic.co." and "[2] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson, "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics", KDD '19 Workshop on Anomaly Detection in Finance, August 2019, Anchorage, AK, USA." |
| ASVspoof 2019 (LA) | **Open Data Commons Attribution License (ODC-By)**; the licence governs the database, "not the contents of the Database individually" | Edinburgh DataShare item https://datashare.ed.ac.uk/handle/10283/3336 and its `LICENSE_text.txt` (https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LICENSE_text.txt). The Hugging Face copy this project loads (https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) states the same licence and "The original dataset was not modified." | DataShare: "Yamagishi, Junichi; Todisco, Massimiliano; Sahidullah, Md; Delgado, Héctor; Wang, Xin; Evans, Nicolas; Kinnunen, Tomi; Lee, Kong Aik; Vestman, Ville; Nautsch, Andreas. (2019). ASVspoof 2019", DOI https://doi.org/10.7488/ds/2555 |
| In-the-Wild | **Apache License 2.0** ("This dataset and the associated documentation are licensed under the Apache License, Version 2.0") | https://deepfake-total.com/in_the_wild | Müller, Czempin, Dieckmann, Froghyar, Böttinger, "Does audio deepfake detection generalize?", Interspeech 2022 |
| Schibsted Grotesk (font) | SIL Open Font License 1.1 | https://raw.githubusercontent.com/schibsted/schibsted-grotesk/main/OFL.txt | none required on the page; the licence file ships with the font files |
| Martian Mono (font) | SIL Open Font License 1.1 | https://raw.githubusercontent.com/evilmartians/mono/main/OFL.txt | same |

The frontend spec names two font families, not three; both are open-licensed, so no substitution was needed.
Whether Schibsted Grotesk has tabular figures is recorded in the frontend section below.

### What each licence means for this project

| Question | Elliptic (CC BY-NC-ND 4.0) | ASVspoof 2019 LA (ODC-By) | In-the-Wild (Apache 2.0) |
|---|---|---|---|
| Show aggregate metrics (PR-AUC, EER) publicly | Yes: a measurement is not a copy of the data. Attribution given. | Yes, with attribution. | Yes, with attribution. |
| Show per-sample predictions publicly (a node's score, its neighbours, a clip's score) in a page, screenshot or video | **Owner decision needed.** The licence lets you "reproduce and Share the Licensed Material, in whole or in part, for NonCommercial purposes only" but only "produce and reproduce, but not Share, Adapted Material". Showing a few real transaction ids and edges is sharing a part of the data (allowed with attribution, non-commercial). A model score per transaction is arguably derived from the data; whether it is "Adapted Material" in the licence's copyright sense is not settled by the text. Conservative reading: do not publish per-node scores. | Permitted by the database licence with attribution. The licence does not cover the audio contents themselves (the speech comes from the VCTK corpus, whose own terms were **not read: unverified**). | Permitted with the licence and attribution notice. The clips are recordings of public figures; the licence does not address personality or privacy rights. |
| Show dataset clips or raw rows publicly | A part of the data may be shared non-commercially with attribution; the project still does not do it. | Database licence permits it; underlying audio rights unverified. The project does not do it. | Permitted with notice; the project does not do it. |
| Publish trained weights | **Owner decision needed.** If weights count as Adapted Material they may not be shared; the licence text does not say. Conservative reading: do not publish the XGBoost / GNN weights. | Nothing in ODC-By forbids it; attribution required. | Not trained on it (test only). |
| Is a portfolio "NonCommercial"? | **Owner decision needed.** The licence defines it as "not primarily intended for or directed towards commercial advantage or monetary compensation". A student portfolio shown to employers is probably within that, but it is the owner's call. | Not restricted. | Not restricted. |

**Consequences applied in this build (conservative choices):**
- The public-facing rule already in the frontend spec stands and is now backed by a licence reading: no Elliptic
  per-node prediction, no dataset clip, and no weights in the repository, screenshots or video. The graph case desk
  is a live local demo; its screens are not saved under `docs/frontend_screenshots/`.
- `.gitignore` excludes `artifacts/**` weights and score files and `data/` (WP7).
- The runsheet tells the owner that showing the graph demo's real node scores in a public video, or deploying
  it publicly, is his decision under CC BY-NC-ND, and gives the safe alternative (show the findings pages and
  the voice desk with his own recording; keep the graph desk off screen or blurred).
- No stop condition was triggered: nothing the spec shows publicly is clearly forbidden, because the spec
  already keeps per-sample predictions out of public material. The Elliptic rows above are the open decisions.

## Frontend decisions (WP4 and WP5, 2026-10-08)

Spec: `PHASE_2_FRONTEND_DESIGN.md` (v2) in the parent `docs/` folder. Code: `frontend/`.

### Stack
React 19, Vite 7, TypeScript, Tailwind CSS 4, React Router; Vitest and Testing Library for tests. The built site is
static files served by nginx, which also proxies `/api` to the FastAPI container, so the browser only ever talks to
one origin. Fonts are self-hosted from the `@fontsource-variable` packages (no request to a font CDN). Streamlit
(`frontend/app.py`, `Dockerfile.frontend`, port 8501) is removed; the frontend is now on port 8080.

### One data module, checked two ways
Every number on the pages lives in `frontend/src/data/evidence.json` with its value, spread or interval, what it
measures, run date, phase code and source (a file plus a JSON path, or a file plus a sentence that must appear in it).
- `scripts/check_frontend_numbers.py` resolves every source and compares it with the value at the decimals shown:
  68 entries, all match.
- `frontend/src/test/numbers.test.tsx` runs inside `npm run build`. It renders each page and fails if a
  result-looking number appears outside an element fed by the data module or by the live API, and it scans the
  component sources for inline result literals. It caught three real slips during the build (a screen-reader summary,
  a dataset name with a year, example ids).

### Mismatches found between the spec, the log and the result files
- **GAT tuned and GIN.** The spec and the project log give 0.291 and 0.265 (the 2026-09-15 run, also in
  `docs/graph_leaderboard.md`). `artifacts/graph/results.json` on disk is from a later re-run of the same notebook
  (file dated 2026-09-23) and holds 0.3279 and 0.2638; the GNN training is not seeded identically across runs. The
  page shows the log's numbers, sourced to the log, and each marker's note states the re-run value. The ranking and
  the headline are the same either way, so this was not a stop condition. Logged as issue #45.
- **The gap to XGBoost.** The spec says "about 0.156". That is 0.7474 (XGBoost, seed 42) minus 0.5913 (the best GNN's
  3-seed mean), as the log computes it. Mean against mean it would be 0.157. The page shows the log's 0.156 and the
  marker says which two numbers it subtracts.
- **Explanation cross-check.** The spec's wording ("GAT attention and GNNExplainer disagree ... attention was right")
  matches the log (issue #5), not the auto-generated sentence still inside `results.json`, which says they agree. The
  page follows the log.
- **Fonts.** The spec asks to verify "three fonts or font families"; it names two. Both are OFL.

### Deviations from the spec, and why
- **No "top contributing features" on the graph desk.** The runbook says to leave the API untouched apart from WP3/WP3b,
  and Elliptic's features are anonymised, so a list of feature numbers would explain little. The desk shows the four
  graph statistics instead, and the page's Limits section says the omission plainly.
- **Two additive API fields** (`graph_features` on `/graph/score`, trimmed-silence seconds on `/voice/score`) were added
  so the desk can show degree, PageRank and clustering, and mark the trimmed silence. Nothing existing changed.
- **Example transactions carry no label text.** The first build's menu said "labelled illicit"; a dataset label on a
  static page is dataset content, so the menu now gives only the row number. Scores and labels appear only after a
  live request.
- **The voice-page instrument shows no "difference" figure.** 0.2380 minus 0.0097 is not a number recorded anywhere,
  and the data module holds only recorded numbers.
- **Tabular figures.** Schibsted Grotesk has them (measured in the browser: "1111" and "0000" are the same width with
  the feature on), but the same feature widens the full stop and the comma ("0 . 7730", "203 , 769"). Only the digit
  runs get tabular figures; punctuation stays proportional (`Digits` in `components/evidence.tsx`).
- **Audio is converted in the browser** to 16 kHz mono WAV before upload, so a microphone recording (WebM/Opus, which
  the API's decoder cannot read) and any file the browser can decode both work, and uploads stay small.
- **Verdicts are words and line weight, not colour**: filled, outlined, dashed.
- **The pattern legend and the citations are in the footer of every page**, not only the landing page.

### Tokens
The proposed colours passed in both themes without change (`scripts/check_contrast.py`: 34 pairs, 0 failures; the
tightest is dark-theme warning text on a panel at 4.53:1). Added: a secondary text colour (#4A524E light, #B3BAB4
dark), a rule colour and a track colour, derived from the same greys. Both fonts are SIL OFL 1.1 (Terms findings).

### Critique (WP5)
Screenshots of the three pages at 390, 768 and 1440 px in both themes were reviewed against the spec's "discarded" list.
- Not present: dark mesh hero, gradient, glow, drop shadows, three identical cards, all-caps eyebrow labels, 01/02/03
  numbering, arrows on links, counting-up numbers, scroll-triggered motion, hover lifts.
- **Generic default found and changed:** the theme switch was a bordered button that dominated the header on a phone,
  the stock "toggle in the corner". It is now a plain text control.
- **Generic default found and changed:** the status register and the case card were four-column tables that overflowed
  a 390 px screen by 9 px. They now stack into labelled rows on a phone.
- **Decorative element removed:** the rule under the header. It separated nothing that the spacing did not already
  separate; every remaining rule divides rows of a register.
- Kept on purpose: the neighbourhood rings (they encode hop distance) and the red edge on caveat notes (it marks a
  warning, and the note is also `role="note"` text).

### Accessibility checks (WP5)
| Check | Result |
|---|---|
| Contrast, every token pair, both themes | Pass (script) |
| Horizontal scroll at 390, 768, 1440 px, 3 pages, 2 themes | None (measured by the capture script, 18 of 18) |
| Instrument by keyboard | It is a native radio group; arrow keys switch. Covered by a component test. Not tried with a screen reader. |
| Demos by keyboard | Native buttons, inputs and a select. The file picker is a visible button that triggers the hidden input. **Not verified by tabbing through in a browser, and not verified with a screen reader.** |
| Visible focus | 3 px outline on every focusable element (`:focus-visible`), from the stylesheet. Not screenshotted. |
| Reduced motion | The bar transition is disabled under `prefers-reduced-motion`. Checked in the CSS, **not verified with the OS setting switched on**. |
| Text equivalent for every chart | Dot plots are real tables (label, value, spread in text); the instrument has a screen-reader sentence with both values; the neighbourhood drawing has a described label and a table; waveform and spectrogram have a caption that states the trimmed silence in seconds. |
| Patterns never alone | Shortcut rows say "shortcut, not a result"; the dashed Visual row says "Designed, not built"; verdicts are words. |
| Stacked tables on phones | They use CSS `display: block`, which some screen readers stop announcing as a table. Accepted; the cells carry their labels as text. |

### Revision 1: two layers (2026-10-08)
The owner found the first build too technical for a visitor who is not an engineer. The copy rules of the design spec
are replaced by `PHASE_2_FRONTEND_REVISION_1.md`: plain language by default, exact terms only under "For engineers".
- **One toggle per page, kept for the session** (`sessionStorage`), rather than a toggle per section: a reader is either
  an engineer or not, and thirteen small toggles would be clutter.
- **Jargon is enforced, not just avoided.** A test fails the build if the default layer of any page contains PR-AUC, EER,
  FPR, XGBoost, GNN, wav2vec, "temporal split", "attack-disjoint" or "calibrat...".
- **Plain figures are derived entries**, each naming its parent entry, a scale and a formula. The checker recomputes
  them, and refuses a derived entry that points at another derived entry. The page test now treats any decimal and any
  "N in 100" as a result number, so a plain figure typed by hand fails the build.
- **"Recordings", not "calls".** The revision's examples say calls; the data are research recordings, and "calls" would
  be a new claim.
- **The Cypher shown in the Technical view comes from the service** (`/graph/query/cypher`, built from the same functions
  the store executes), not from a copy in the frontend that could drift.
- **The landing page lost two sections** ("What the evidence shows", "What was cut"). Their content is on the engine
  pages and in the status list; the revision asks to cut repeats.
- **Disabled buttons are not faded.** Fading failed contrast. They are a dashed outline with secondary-text colour, and
  the voice desk says "Add a recording first."
- **`/health` is retried** for about 75 seconds, because a page opened during start-up otherwise stayed on "service not
  running".

### Self-test follow-ups (2026-10-08)
- **Input guards belong to the service, not the model.** Silence is refused and under one second of sound is flagged.
  The one-second line is an engineering choice, not a measured threshold. Noise and tones are still scored: detecting
  "not speech" would need a voice-activity model, which is out of scope.
- **`record=false`** on the scoring endpoints lets the verify and self-test scripts run without touching the cases table.
- **Recording mode is a display mask, not a security boundary.** The API still returns real values; the site hides them
  so a video does not show them. It is for recording on the owner's machine, not for a public deployment.
- **The site no longer waits for the API.** nginx looks the API up per request, so a missing weight file shows as "the
  scoring service did not answer" on a working site instead of a stack that never opens.
- **The owner's voice-clip results stay outside the repository.** The self-test report carries counts only; the per-clip
  table is next to the clips. The owner asked for both a table in the report and for per-clip data never to enter the
  repository; the second instruction was followed because the report is committed.
- **The phone-audio finding went on the voice page as a plain limit, with no number**, because three pairs is not a rate.

### Dataset content check
The saved screenshots are the pages' empty states: aggregate results, no scored case, no clip, no label, no weight.
`scripts/check_no_dataset_outputs.py` passes on them. The transaction row numbers in the example menu are identifiers,
not dataset content. Screens of a scored case are not saved anywhere in the repository.

### Audio-quality guard (2026-10-15)

Two checks run before the voice model (`serving/audio_guard.py`, called from `serving/voice_service.py`). Both are
heuristics fixed once from a handful of local clips. They are **not validated detectors**, they never change the score,
and they never change the risk thresholds. `TRUSTMESH_AUDIO_GUARD=0` switches them off (debugging only).

| Rule | Measure | Threshold | Where the threshold came from |
|---|---|---|---|
| Speech check: refuse | Spectral movement: standard deviation of the spectral centroid over the loud frames, divided by its mean (25 ms frames, 10 ms hop, loud = within 25 dB of the loudest frame) | at least 0.30 | Lowest value among 84 real-speech clips and versions: 0.477. Highest synthetic non-speech apart from a frequency sweep: 0.159 (pink noise). 0.30 is a round number in that gap, about 1.6 times below the quietest speech and 1.9 times above the highest noise |
| Band-limit warning, rule 1 | Median over loud frames of the energy at 5 to 7.9 kHz relative to 0.3 to 3.4 kHz, in dB | at least -55 dB | Lowest of 13 clean clips: -52.0 dB. The threshold is 3 dB lower, rounded |
| Band-limit warning, rule 2 | Cut-off frequency: highest frequency where the average loud-frame spectrum is within 55 dB of its 0.3 to 3.4 kHz level | at least 7600 Hz | The 13 clean clips lie between 7880 and 8000 Hz. The threshold is 280 Hz below the lowest |
| Band-limit warning, rule 3 | The file's own sample rate | at least 16 kHz | By definition: a file recorded below 16 kHz has nothing above half its rate. Needed because the service up-samples by simple interpolation, which fills the missing band with mirror images and hides it from rules 1 and 2 |

"Clean" means the 8 dataset clips on this machine (4 genuine, 4 fake) and 5 LibriSpeech clips (which were themselves cut
from MP3 files). Re-encoded versions of the same 13 clips (AAC 32 kb/s, AAC 32 kb/s through 48 kHz, MP3 128 kb/s, Opus
24 kb/s, 8 kHz mu-law telephony) and the owner's 6 recordings were measured afterwards. The thresholds come from the
clean clips only; they were not moved to catch more of the owner's recordings (the owner's 3 voice clones, at -47.9 to
-41.2 dB, are not flagged).

**What the first suggested measure did.** The addendum's example, the frequency below which 99% of the energy lies, did
not separate compressed from clean speech here (it sits at about 3 kHz for both, because speech energy is low-frequency
anyway). It was measured and not used.

**Counts** (`scripts/calibrate_audio_guard.py`; n clips; "rejected" = refused as not speech; "warned" = band-limit warning):

| Group | n | Rejected | Warned |
|---|---|---|---|
| Clean | 13 | 0 | 0 |
| AAC 32 kb/s | 13 | 0 | 0 |
| AAC 32 kb/s through 48 kHz | 13 | 0 | 0 |
| MP3 128 kb/s | 13 | 0 | 13 (rule 2: the encoder low-passes at about 7.4 kHz) |
| Opus 24 kb/s | 13 | 0 | 0 |
| 8 kHz mu-law telephony, converted back to 16 kHz | 13 | 0 | 13 |
| 8 kHz mu-law file as uploaded (decoded at 8 kHz, up-sampled by the service) | 13 | 0 | 13 (rule 3 only; the spectral rules alone: 0) |
| Speech with white noise at 20 / 10 / 5 dB SNR | 13 each | 0 / 0 / 0 | 0 / 0 / 0 |
| Speech with white noise at 0 dB SNR | 13 | **5** | 0 |
| Owner's genuine voice memos (AAC 32 kb/s) | 3 | 0 | **3** |
| Owner's voice clones (MP3 128 kb/s) | 3 | 0 | 0 |
| Synthetic non-speech (noise x3, tones x4, chord, square, tone plus noise) | 10 | 9 | n/a |
| Synthetic frequency sweep | 1 | **0** | n/a |
| Silence | 1 | refused earlier as silent | n/a |

**Real speech wrongly rejected:** none of the 84 clean, compressed and owner clips. Speech buried in noise is the
exception: 5 of 13 clips at 0 dB signal-to-noise were rejected (none at 5 dB).

**What the warning does and does not catch.** It fires on telephone-band audio, on 8 kHz files, on MP3 with a low-pass,
and on the owner's 3 genuine phone recordings. It did **not** fire on the clean clips re-encoded as AAC 32 kb/s (0 of 26,
with or without the 48 kHz step) or as Opus 24 kb/s (0 of 13), although the model scores those lower (see "Why do genuine
phone recordings score as fake?" in the status report). So it detects a band-limited recording, not codec damage in
general. It is a warning that the result is unreliable, and it is silent on some audio for which the result is
unreliable too.

**Known gaps.** A frequency sweep passes the speech check. Singing, whispering, music and speech in loud noise were not
tested. 84 clips from at least three speakers is a handful. No accuracy figure is attached to either check, and none
appears on the site.

**Where it shows.** The voice desk and the combined desk show the warning in plain words by default
("This audio looks phone-quality or heavily compressed. This model was trained on clean recordings, so the result is
unreliable.") with the measures and thresholds under "For engineers" and in the technical view. A clip that is not
speech is refused with a 422 and the sentence "This does not look like speech, so nothing was scored."

## Phase 2D: codec-robust voice head (Model F2). Design and pre-registered criteria (2026-10-15)

Written before any Phase 2D run, and not to be changed afterwards. Nothing has been trained. The spec is
`docs/PHASE_2D_CODEC_ROBUSTNESS.md` with its addendum 1; where they differ the addendum wins.

**Question.** Lossy compression pushes genuine speech towards "fake" for the served voice model (log issue #56). Does
training the same head on features of codec-processed copies of the training clips fix that, without hurting clean audio?

**What changes and what does not.** The encoder stays frozen (XLS-R 300M). The head is Model D's recipe (mean-only
pooling over 25 layers, 133,402 parameters, same learning rate, patience, batch size, seeds 42/43/44). The only change is
the training data: each training clip contributes its clean version plus 3 augmented versions, chosen with a fixed seed
from the training menu. Same speaker-disjoint P1 split as Model D (split seed 42; identity checked by the test-index hash
and counts in the notebook). The comparison model is a Model D control trained in the same notebook on clean features only,
with the same code, so nothing depends on an earlier run.

**Conditions (names fixed).** Training menu: `aac32`, `aac32_via48k`, `mp3_64`, `opus24`, `echo` (synthetic room impulse
response from exponentially decaying noise, RT60 0.2 to 0.8 s, no external data), `noise_lowsnr` (white noise, 5 to 20 dB
SNR). Held out, evaluation only, never trained on: `aac16`, `opus12`, `mulaw8k` (8 kHz G.711 through ffmpeg, back to 16
kHz), `mp3_32`. Codecs are real ffmpeg codecs applied to the full original clip; the clip is then trimmed and fixed to 4 s
exactly like a clean clip. If a codec is missing from the available ffmpeg the notebook records it and substitutes openly
(`opus24` -> `aac24_for_opus24`, `opus12` -> `aac12_for_opus12`); the result file lists any substitution.

**Validation and tuning.** Early stopping uses a validation set that mixes the clean and the augmented versions of the
validation clips, for the control and for F2 alike. Hyperparameters are Model D's; nothing is tuned on test, and no
setting is changed after seeing test results.

**Evaluation.** The P1 test clips, each processed with every condition (training menu and held out). EER per condition and
pooled, with stratified bootstrap intervals (1000 resamples). Differences are F2 minus control, paired bootstrap. On the
pooled codec set the bootstrap resamples whole clips (all of a clip's condition versions together), because versions of one
clip are not independent. The pooled codec set is the six training-menu conditions; held-out conditions are reported
separately. The decision uses each model's score averaged over its three seeds; per-seed EERs are reported next to it.
Also reported for each model and condition: the share of genuine clips flagged at the threshold that gives the EER on
clean validation clips (what a caller would see).

**Success criteria (pre-registered, addendum version).**
1. Clean audio: the 95% interval of the EER difference (F2 minus control) on the clean test clips does not lie wholly
   above zero (F2 is not significantly worse).
2. Codec audio: (a) on every single training-menu condition the interval of the EER difference does not lie wholly above
   zero; and (b) on the pooled training-menu test set the interval lies wholly below zero (F2 is significantly better).
3. Held-out conditions (`aac16`, `opus12`, `mulaw8k`, `mp3_32`) are reported separately. Not a pass or fail criterion.
4. The owner's own recordings are a demo only, never a criterion.
If 1 or 2 fails, it is reported as a negative result. No retuning and re-running until it passes.

**Known limits stated in advance.**
- To fit Kaggle's disk, each condition is evaluated on a fixed seeded subset of the P1 test clips: all bonafide plus
  spoofs up to `EVAL_PER_CONDITION` clips in total (default 4000); the clean comparison is also given on the full test
  set. The subset and its class counts are written into the results file.
- F2 sees four versions of each training clip, the control one. F2 therefore takes more gradient steps per epoch. This
  is part of what "train on the augmented data" means; it is not a separate arm.
- The synthetic echo and noise are simple; real rooms and real calls are not covered. Real AAC 32 kb/s and Model F's
  simulated codec differ clip by clip (Phase 2D A2: mean drops 8.3 vs 11.0, per-clip rank correlation -0.17 on 9
  clips), so the simulation is not used.
- Dataset terms: the trained head weights stay out of git until the owner has checked the dataset terms. The results file
  holds aggregates only (no clip identifiers, no per-clip scores).

## Phase 2D FULL results (Kaggle run, read 2026-10-17)

Source: `artifacts/voice/results_phase2d.json` (git-ignored; aggregates only) and the six head files (private). Every number
below is copied from that file, except the last subsection, which is a separate local check and says so. The criteria were
fixed before the run (section "Phase 2D: codec-robust voice head" above). Nothing is served from this run: the served model,
the thresholds and the site are unchanged, and the thresholds have not been recalibrated.

**The run.** Split identical to Model D's: the test-index hash and the train/val/test counts match the recorded Model D split
(train 4099 bonafide / 16099 spoof, val 1091 / 4329, test 2165 / 8992; 0 shared speakers). All ten conditions were produced
by the available ffmpeg with no substitution and none unavailable. 153,629 feature versions (36,775 clean, 76,854 augmented,
40,000 evaluation), 7.87 GB planned, extraction 13,829 s (conditions 11,168 s, encoder 2,610 s, decoding 43 s), 4.15 hours in
all. Both models have 133,402 parameters. Each condition is evaluated on the fixed subset of 4000 test clips (2165 bonafide /
1835 spoof); "clean, full test" uses all 11,157 test clips. EER differences are F2 minus control (negative = F2 better), paired
bootstrap, 1000 resamples; the pooled row resamples whole clips (24,000 rows).

| Condition | Control EER [95% CI] | F2 EER [95% CI] | Difference [95% CI] | Genuine flagged, control / F2 |
|---|---|---|---|---|
| Clean, full test | 0.0254 [0.0225, 0.0285] | 0.0199 [0.0166, 0.0224] | -0.0055 [-0.0088, -0.0031] | 0.0513 / 0.0319 |
| aac32 | 0.0360 [0.0296, 0.0431] | 0.0283 [0.0223, 0.0343] | -0.0077 [-0.0139, -0.0029] | 0.5982 / 0.0596 |
| aac32_via48k | 0.0439 [0.0371, 0.0512] | 0.0342 [0.0278, 0.0402] | -0.0097 [-0.0161, -0.0045] | 0.9095 / 0.1012 |
| mp3_64 | 0.0283 [0.0234, 0.0338] | 0.0262 [0.0202, 0.0305] | -0.0022 [-0.0076, +0.0016] | 0.1912 / 0.0790 |
| opus24 | 0.0286 [0.0234, 0.0343] | 0.0263 [0.0213, 0.0309] | -0.0023 [-0.0071, +0.0022] | 0.1894 / 0.0961 |
| echo | 0.2434 [0.2305, 0.2568] | 0.1099 [0.0993, 0.1199] | -0.1335 [-0.1483, -0.1193] | 1.0000 / 0.2402 |
| noise_lowsnr | 0.0619 [0.0545, 0.0703] | 0.0148 [0.0114, 0.0189] | -0.0471 [-0.0545, -0.0397] | 0.2988 / 0.0240 |
| **Pooled training-menu conditions** | 0.1622 [0.1565, 0.1681] | 0.0410 [0.0384, 0.0441] | -0.1212 [-0.1248, -0.1173] | 0.5312 / 0.1000 |
| aac16 (held out) | 0.0910 [0.0822, 0.1008] | 0.0684 [0.0594, 0.0753] | -0.0226 [-0.0315, -0.0155] | 0.9885 / 0.4333 |
| opus12 (held out) | 0.0409 [0.0343, 0.0474] | 0.0388 [0.0322, 0.0457] | -0.0021 [-0.0071, +0.0032] | 0.7861 / 0.4305 |
| mulaw8k (held out) | 0.0338 [0.0273, 0.0393] | 0.0323 [0.0263, 0.0387] | -0.0015 [-0.0071, +0.0049] | 0.4000 / 0.0970 |
| mp3_32 (held out) | 0.0322 [0.0262, 0.0387] | 0.0328 [0.0262, 0.0388] | +0.0006 [-0.0050, +0.0060] | 0.5769 / 0.0753 |

"Genuine flagged" is the share of genuine test clips scored below each model's own threshold at the EER on clean validation
clips (raw score 1.2046 for the control, 0.1938 for F2). On the same clean test clips the two models flag 0.0513 and 0.0319.
Per-seed clean EERs: control 0.0249, 0.0259, 0.0263; F2 0.0197, 0.0203, 0.0194 (seeds 42, 43, 44).

**Pre-registered criteria, read mechanically (all_pass = true).**
1. Clean audio: F2 is not worse than the control (interval -0.0088 to -0.0031, wholly below zero). Pass.
2. (a) No training-menu condition is worse: all six pass. On mp3_64 and opus24 the interval includes zero, so F2 is not shown
   better there either. (b) Pooled training-menu set: the interval lies wholly below zero. Pass.
3. Held-out conditions, reported and not judged: the interval excludes zero on one of four (aac16). On opus12, mulaw8k and
   mp3_32 it includes zero (mp3_32's point difference is +0.0006).

**Plain reading: reduced, not removed.** Training on codec-processed copies reduced the damage that compression does to genuine
speech; it did not remove it. The share of genuine clips flagged at the clean-validation threshold fell, for example, from
0.5982 to 0.0596 (aac32) and from 0.9095 to 0.1012 (aac32 through 48 kHz), but it is still above the clean figure of 0.0319
for every compressed condition, and F2 remains weak on echo (EER 0.1099, 0.2402 of genuine flagged), on the held-out aac16
(EER 0.0684, 0.4333 flagged) and on the held-out opus12 (EER 0.0388, 0.4305 flagged).

**Caveats, stated plainly.**
- **The control is not the served Model D.** The control's clean EER on the full test set is 0.0254, against 0.0166 (3-seed
  mean) for Model D and 0.0194 for served seed 42. The control stopped early: best epoch 3, 4 and 3 for seeds 42, 43, 44 (11,
  12, 11 epochs run), with a mixed clean-and-augmented validation EER of 0.1265, 0.1268, 0.1265. The early-stopping rule in this
  notebook (validation that mixes clean and augmented clips, as the addendum required) is the only difference from Model D's
  recipe that was identified; it may explain the weaker control, but that was not tested. Part of F2's apparent clean gain over
  the control (-0.0055) therefore reflects a weak control, not a gain over served Model D.
- **F2 on clean audio is about equal to served seed 42:** F2's clean EER is 0.0199 (seed-averaged scores) against 0.0194 for
  served seed 42 and 0.0166 for the Model D 3-seed mean. Different score averaging; no claim that F2 is better on clean audio.
- **Held-out codecs: ranking improved on only 1 of 4** (aac16). Not shown on opus12, mulaw8k, mp3_32.
- **The test is synthetic.** It is ASVspoof 2019 LA evaluation-partition clips run through ffmpeg codecs and a synthetic echo
  and noise model, on a speaker-disjoint split of the same dataset. It is not real phone calls, real rooms or another dataset.
  It does not say how F2 does on In-the-Wild or on a real phone network.
- The decision uses seed-averaged scores, so the three seeds are not three independent tests. Each condition uses a 4000-clip
  subset of the test set. The "genuine flagged" shares depend on each model's own clean-validation threshold; they are not
  the served thresholds.
- Not done and not decided: serving F2, recalibrating thresholds, changing the site. These wait for the owner.

**Local inference-only check on a few real clips (2026-10-17; not part of the results file, no training).** The three F2
heads and the served Model D head scored the owner's own recordings (3 genuine voice memos, 3 voice clones) and the 9 genuine
local clips (4 dataset clips, 5 LibriSpeech clips), at the served thresholds (high 0.6578, medium 0.0239 on spoof
probability), which were not recalibrated for F2. Aggregates only; nothing per clip is kept. These are a few real clips, not
a validated test.

| Group (n) | Model | Flagged high | Flagged medium or high | Mean bonafide logit | Mean spoof probability |
|---|---|---|---|---|---|
| Own genuine memos (3) | served D | 1.00 | 1.00 | -10.46 | 1.000 |
| | F2 seed 42 / 43 / 44 | 0.67 / 0.67 / 0.67 | 1.00 / 1.00 / 0.67 | -1.87 / -2.32 / +0.32 | 0.736 / 0.762 / 0.542 |
| | F2, mean of 3 seeds | 0.67 | 1.00 | -1.29 | 0.668 |
| Own voice clones (3) | served D | 1.00 | 1.00 | -15.08 | 1.000 |
| | F2 seed 42 / 43 / 44 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | -11.67 / -12.33 / -8.97 | 1.000 / 1.000 / 1.000 |
| | F2, mean of 3 seeds | 1.00 | 1.00 | -10.99 | 1.000 |
| 9 genuine local clips (9) | served D | 0.00 | 0.00 | +8.33 | 0.001 |
| | F2 seed 42 / 43 / 44 | 0.00 / 0.00 / 0.00 | 0.00 / 0.11 / 0.00 | +10.19 / +9.19 / +12.51 | 0.001 / 0.004 / 0.001 |
| | F2, mean of 3 seeds | 0.00 | 0.00 | +10.63 | 0.001 |

Plain reading: on the owner's three phone-recorded genuine memos F2 moves the mean score from strongly "fake" towards the
middle, but 2 of 3 memos are still in the high band; F2 still flags all three voice clones; and it does not flag the 9 genuine
clean clips. Three memos cannot support any rate.

## Frontend Revision 2: six worlds (2026-10-17)

Spec: `PHASE_2_FRONTEND_REVISION_2_IMMERSIVE.md` with its performance addendum (the addendum wins where they differ). This
revision changes how the site looks and moves. It does not change what the site claims: the honesty rules of the spec's
section 1 are enforced by tests (`frontend/src/test/worlds.test.tsx`) on every route.

**The six worlds and the 404.** Each page has one idea, one palette per theme, one display typeface, one 3D scene and one
motion character.

| Route | World | Idea | Concept chosen (the other considered) and why |
|---|---|---|---|
| `/` | The Mesh | trust is a thread between things | A thread mesh whose nodes re-arrange for each chapter (a particle galaxy with orbiting engines was the other). The re-arrangement doubles as the three findings. |
| `/graph` | The Ledger | a transaction is a bead on a copper thread | A loom with beads (a city of towers per time step). The time axis is the point of the page, and a ledger reads as accounting. |
| `/voice` | The Terrain | a voice, as a landscape | A spectrogram terrain drawn from the user's own clip (a rotating waveform ring). Frequency, time and loudness are the real structure, and the band-limit cliff becomes visible. |
| `/case` | The Balance | two scores, honestly weighed | A beam balance with three pans (a radar with two needles). The weights are the honest subject, and the empty third pan is the Visual slot. |
| `/evidence` | The Archive | what we checked, what we could not | A wall of drawers (a timeline tunnel). "Sealed" and "empty" become physical states. |
| `/about` | The Workbench | how to rebuild it yourself | An exploded blueprint that draws itself on (a floating card stack). |
| `*` | an unlinked node | | One node adrift, its thread snapped; the link back ties it. |

**Research, stated honestly.** The reference site (scrolltide.co) was opened and studied in a browser; four more sources
were read at summary level (two Codrops articles, an Awwwards listing, a WebGL roundup). That is five, not the eight the
spec asks for. One article could not be fetched (HTTP 403). Notes are kept outside the repository.

**Typefaces.** Four display families (Bricolage Grotesque for Home, Fraunces for Graph and Evidence, Unbounded for Voice,
Sofia Sans Extra Condensed for Case), plus Martian Mono, already used for code, which the Workbench also uses for its
headings. All are SIL OFL 1.1, variable, self-served, latin subset only (declared in `src/fonts.css`, which is why the
build emits one file per family). All six latin faces together weigh 234.1 KB; a page loads the body face and its own
display face (Home: 86.1 KB). Each face has a stand-in (Arial scaled to its measured width and line metrics) so the page
does not move when the real face arrives. Graph and Evidence share Fraunces; the Archive is told apart by its green,
kraft and drawers, not by a fifth family (the spec allows at most four).

**Dependencies (all permissive; `npm audit --omit=dev` reports 0 vulnerabilities).** `three` 0.186.1 (MIT), `@react-three/fiber`
9.8.2 (MIT; its peer range is React 19 below 19.4, the installed React is 19.3.0), `lenis` 1.3.27 (MIT), and the four
`@fontsource-variable` packages (OFL-1.1). **GSAP was rejected**: its licence ("Standard no charge") is not MIT, ISC,
BSD or Apache, which the spec requires. A 60-line ticker and easing module replaces it (`src/world/ticker.ts`). `drei`,
postprocessing and `zustand` were not needed; a small store with `useSyncExternalStore` does the job.

**Architecture.**
- One persistent WebGL canvas for the whole site, loaded after first paint. One animation loop (the ticker) drives smooth
  scroll and the canvas; nothing else owns a `requestAnimationFrame`. The canvas draws on demand (scroll, pointer, a
  theme change, an animating scene) and not at all while the tab is hidden.
- A page is cut into chapters; when a chapter crosses the middle of the screen the scene eases to it. A page hands its scene
  a small data object; the scene never reads the page. Scene code is loaded per world; no world loads another's.
- Everything is instanced or shader-driven: Home 1,200 / 3,000 / 6,000 nodes (low / medium / high) in one points draw and
  one lines draw; Graph beads as lit-sphere point sprites; the Voice terrain is one grid whose heights are one vertex
  attribute written per clip; the Archive is two instanced meshes and one dashed line list. No textures at all (texture
  memory is 0 in every world), no shadows, no reflections, no post-processing.
- A quality governor starts on the low tier and promotes only after three good windows in a row, against 60 fps (or 30 for
  scenes that ask for 30), demotes quickly on a dip, and never flip-flops. Weak devices and touch screens are capped at
  medium. Shaders are compiled with `compileAsync` before a scene shows, so a route change does not block on a compile.
- Fallbacks, each tested in a browser: reduced motion draws one still frame per world and crossfades between pages; data
  saving, no WebGL and a lost context show a drawn poster with the same text.

**Dark only (2026-10-17, owner decision after reviewing the build).** The light theme and its switch were removed: the site is dark in every world, and no light palette is defined (a test checks this). The Workbench changed from blue to the same dark graphite as the other pages, keeping its white drawing lines and monospaced headings. The two solid engine nodes on the Overview (Graph, Voice) were redrawn to match the hollow Visual node: a glowing core, a wireframe shell, two tilted rings and a halo, four draw calls each. Where this entry says "light and dark" above, only dark remains.

**Where the spec and the repository disagreed, and what was done.**
1. *The Overview keeps its case desk.* The spec moves the combined case desk to `/case` and keeps "a compact path to it" on
   the landing page. An existing test (one of the 37 that must keep passing) scores a case on `/`, so the Overview keeps a
   compact copy of the desk, and `/case` holds the full one with the placeholder-weights explanation and a register.
2. *The cases register is local to the visit.* The service's own `/cases` listing is blocked by the nginx proxy on purpose
   (it would list real transaction scores) and would break recording mode, so the register shows the cases scored in this
   visit, highest risk first, and says so.
3. *No real transaction scores in the worked example.* The spec's example (a graph score of 0.993 and a clean voice score
   giving 0.497, "medium") uses per-transaction scores that are not in the data module. The page gives the arithmetic with
   the extremes instead (highest and lowest possible scores average to the middle, which falls in the medium band), and
   says the band edges are placeholders too. `scripts/verify_serving.py` still prints the real example locally.
4. *Phase 2D.* The spec says the element about it must read "pre-registered, run pending" until the owner updates the data
   module. This task's brief fixed the wording as "reduced, not removed", so the Archive's envelope is sealed until you press
   it and then states the result with its caveats. The 14 figures it shows are new data-module entries (phase code P2D),
   each sourced to the table in "Phase 2D FULL results" above and checked by `scripts/check_frontend_numbers.py`.
5. *New data-module entries.* 28 entries were added, none changed: 14 for the audio-quality warning, the two compression
   tests, the live-versus-offline check and the audit; 14 for Phase 2D. All 110 entries match their sources.
6. *Stale entries, not touched.* `infra.pythonTests` (116) and `infra.frontendTests` (28) are older than the code (185 Python
   tests; 87 frontend tests after this revision). They are shown in the Overview's "For engineers" layer. They need the
   owner's instruction to change.
7. *The owner's open items.* The resume link and a paragraph in the owner's own words are marked `TODO(owner)` in
   `AboutPage.tsx` and shown to visitors as "Resume: coming soon". The GitHub and LinkedIn links already in the footer are
   used as they were.
8. *A bug found by measuring.* The first build put React into the 3D chunk (the bundler files shared code under whichever
   chunk asks first), so the first page downloaded 353 KB of JavaScript instead of 110 KB. React is now pinned to its own
   chunk and the 3D chunk is loaded only after first paint.

**What was measured, and what could not be.** Everything below was measured on the production build served by the Docker
nginx container (gzip on), in headless Edge with WebGL drawn in software (SwiftShader). Counts (draw calls, triangles,
geometries, textures, bytes) are exact. **Frame rates and frame times from a software renderer are not real GPU speed**:
they are used only to compare one world or one build with another. Real-GPU smoothness is unverified until the owner opens
the site on a laptop and a phone (the checklist is in the final report). Lighthouse on real hardware, screen readers,
Firefox, Safari and a real touch device were not run. Scripts: `scripts/check_worlds_browser.py` (browser checks),
`scripts/check_worlds_live.py` (live checks against the running stack), `scripts/bundle_sizes.mjs`,
`scripts/check_contrast.py`, `frontend/src/test/worlds.test.tsx` (route tests). Add `?perf=1` to any page for a readout
(and `&tier=low|medium|high` to pin the tier); nothing is sent anywhere.

**Sizes (gzip).** First load of any page, JavaScript: 109.9 KB (React 67.4 KB, the page shell 42.5 KB; budget 180 KB);
CSS 8.2 KB; fonts for the Overview 86.1 KB (body face 45.7 KB, Bricolage Grotesque 40.4 KB). Loaded after first paint and
shared by every world: the 3D chunk, 242.7 KB (budget 450 KB). One chunk per world scene: Home 3.6, Graph 3.6, Voice 3.2,
Case 2.0, Evidence 1.8, About 1.7, 404 0.9 KB (budget 60 KB); one chunk per page: Graph 7.3, Voice 7.4, Case 2.0, Evidence
4.1, About 2.0 KB. Smooth scroll (Lenis), 5.3 KB, is fetched after the page is up. Fonts, all six latin faces: 234.1 KB
(budget about 450 KB). No world loads another world's scene.

**Per tier (exact, from the renderer's own counters; texture memory is 0 bytes everywhere).**

| World | Low: draw calls / triangles | Medium | High | Budget (high / low) |
|---|---|---|---|---|
| Home | 6 / 360 (1,200 nodes) | 6 / 360 (3,000) | 6 / 360 (6,000) | 60 calls, 300,000 triangles / 30, 100,000 |
| Graph | 2 / 0 (600 beads) | 2 / 0 (1,500) | 2 / 0 (3,000) | met |
| Voice | 1 / 3,906 (64 x 32 grid) | 1 / 16,002 (128 x 64) | 1 / 64,770 (256 x 128) | met |
| Case | 11 / 2,232 | 11 / 2,232 | 11 / 2,232 | met |
| Evidence | 4 / 792 | 4 / 792 | 4 / 792 | met |
| About | 6 / 0 | 6 / 0 | 6 / 0 | met |
| 404 | 2 / 180 | 2 / 180 | 2 / 180 | met |

Points and lines are drawn without triangles, so Home, Graph and About show 0 or few triangles; the node counts are the
per-tier instance caps. Rendering stops when nothing changes: the Archive, About and the 404 draw only on scroll, pointer
or a click, and the others cap themselves at 30 frames a second where their motion is slow.

**Cold first visit, phone-sized (390 x 800), 4x CPU slowdown, 1.6 Mbit/s with 150 ms round trip, cache off.**

| Page | Largest contentful paint | Layout shift | Long tasks (ms) |
|---|---|---|---|
| Overview | 2.32 s (met, under 2.5) | 0 | 143, 369, 196, 145, 61, 142, 69 |
| Graph | 2.16 s | 0.034 | 145, 101, 70 |
| Voice | 2.09 s | 0.002 | 96, 96, 186 |
| Case | 3.18 s in this run, **2.41 s when run alone: borderline** | 0.001 | 50, 75, 50, 165 |
| Evidence | 1.66 s (the largest paint is the theme switch; the heading is revealed later) | 0 | 50, 84, 81, 52, 158 |
| About | 1.64 s (same) | 0 | 216, 83, 51, 71 |

Runs vary: the Overview measured 1.98 s to 2.32 s over three runs on the nginx container, and the Case page 3.18 s in the
full run above but 2.41 s when loaded on its own (target 2.5 s), so the Case page is borderline and the difference between
runs is larger than the difference between pages. In the isolated run the order was: page shell and React at about 1.4 s,
the page chunk at 1.8 s, first content at 1.84 s, the 3D chunk (242.7 KB) arriving at 3.7 s and the scene appearing at
about 4.1 s, so the scene never delays the text. **Long tasks.** On route changes under the same slowdown, tasks
over 50 ms still occur in every world (Graph 65 to 107 ms, Voice 68 to 234, Case 81 to 141, Evidence 64, About 68 to 73,
Overview 87 to 258): the budget of none over 50 ms is **missed**. They are page rendering and scene set-up, and software
shader compilation is part of it; none exceeds 260 ms. **Input response** (real clicks, 4x slowdown, the browser's own
event timing): 72 ms for the theme switch and 48 ms for the menu button, under the 100 ms target.

**Memory.** After visiting all six worlds twice and returning to the Overview, the renderer holds 5 geometries and 0
textures, the same as before (pass); the JavaScript heap was 10 MB.

**Quality governor** (software rendering, so only a comparison): left alone for 14 s the Overview settled on the high tier,
Graph on medium, Voice on low (the Voice grid is the heaviest scene).

**Fallbacks, all tested in a browser.** Reduced motion: one still frame per world, no curtain between pages, nothing
drawn while idle. Data saving: a poster and no canvas. No WebGL: a poster and the full text on all seven routes. A lost
WebGL context: the poster appears and the text stays.

**Checks that passed.** `npm run build` (type check, 87 frontend tests of which 37 are the earlier ones, production build),
the same inside the Docker build; axe-core 0 violations in 28 runs (7 routes, light and dark, plain and engineer layer);
every interactive element of every route reachable by the Tab key (7 routes, 2 layers); no horizontal overflow in 84
combinations of width, theme and layer; no console errors on any route; contrast: 16 palettes, 0 failures, including the
worst case behind a scrim or panel over a 3D scene (arithmetic on the palette, not measured pixels);
`scripts/check_frontend_numbers.py` 110 of 110; `scripts/verify_serving.py` all ok. nginx serves the hashed assets with
gzip and a one-year immutable cache, and the page itself with no-cache. Live checks against the running stack: a tone is
refused and the terrain collapses; band-limited speech is scored with the warning and the fog bank sits at the reported
cut-off; and in recording mode no transaction score, label or number reaches the balance, the neighbourhood beads or the
register.

**Distinctness test.** Each world's opening view and one mid-page view were rendered at 1440 x 900, in light and dark, and shrunk to one contact sheet at 25 percent (kept outside the repository). Judged honestly: every world can be told apart by shape, colour and composition alone, with the text unreadable. Home is a fine web of threads with two solid spheres and one hollow one; Graph is ruled rows of beads; Voice is a thermal landscape (contour lines on paper in light); Case is a beam balance in steel; Evidence is a wall of drawers in green and kraft; About is white lines on blue; the 404 is one node on an empty ground. The closest pair is Home and Graph in dark mode (warm points on a dark ground); they differ in structure (a web against rows) and in ground colour (navy against brown). No world was redesigned. Two weaknesses seen: in light mode the Graph beads read almost black rather than teal, and the Workbench drawing is sparse until it has drawn itself on.

**The three optimisations that mattered most.**
1. Keeping React out of the 3D chunk (found by reading the build, not by guessing): the first page went from 353 KB to
   110 KB of JavaScript, and the largest paint of the Overview from 3.7 s to about 2.1 to 2.3 s in the same slow conditions.
2. Metric-matched stand-ins for every font and a full-screen placeholder while a page loads: layout shift on a first
   visit fell from as much as 0.63 (the footer jumped when a lazy page arrived) to 0.034 or less.
3. Non-blocking work when a world opens (shaders compiled before the scene shows, the idle terrain's sound made in
   6 ms slices, grid keys as numbers): the Voice route's longest task fell from 603 ms to 234 ms.


## Open items (not yet built)

- `engines/graph/`: built for everything Phase 1D uses. The Phase 1–1C-only pieces (GAT, GIN,
  GNNExplainer analysis) still live only inline in notebook 01.
- Model weight files (`graphsage_weights.pt`, `graphsage_tuned_weights.pt`, `gat_weights.pt`,
  `gin_weights.pt`, `graphsage_wider_search_weights.pt`, `gat_tuned_weights.pt`,
  `xgboost_baseline.json`, `xgboost_tier1_tier2.json`) — written by the notebook to
  `artifacts/graph/` on Kaggle; still need to be downloaded from Kaggle's Output panel into this
  repo's `artifacts/graph/`.
- `project.md`'s Results Log / Day 1 status table — deferred until `project.md` exists.
