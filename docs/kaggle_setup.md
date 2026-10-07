# Kaggle Setup

## `notebooks/01_graph_baseline.ipynb` (Phase 1 / 1B / 1C)

1. Upload the notebook (Kaggle → Create → New Notebook → File → Upload Notebook).
2. Settings (right sidebar): **Accelerator = GPU** (T4 x1 or better), **Internet = on**.
3. Run All. Cell 1 `pip install`s `torch_geometric`, `xgboost`, `optuna`, and best-effort
   `pyg-lib` — no Kaggle dataset attachment needed, the Elliptic dataset downloads itself via
   `EllipticBitcoinDataset`'s own PyG-hosted URLs.
4. Outputs land under `./artifacts/graph/` and `./docs/` in the Kaggle working directory —
   download them from the Output panel into this repo's `artifacts/graph/` once the run finishes.

## `notebooks/02_graph_full.ipynb` (Phase 1D)

1. Before uploading, make sure the notebook is current: `python scripts/build_notebooks.py --check`
   (rebuild without `--check` if it says out of date). Edit `notebooks/src/02_graph_full.py` or
   `engines/graph/`, never the `.ipynb`.
2. Upload, then **Accelerator = GPU**, **Internet = on** — same as notebook 01. No dataset attachment
   and no other files needed; the `engines/graph` code is already inlined.
3. Run All. Expect roughly 1–5 hours: the expanded Optuna search dominates and is capped at 4 hours
   (`SEARCH_TIMEOUT_SEC`). Kaggle's 12-hour session limit leaves margin. If the session dies,
   `artifacts/graph/results_phase1d.json` still holds every task that finished.
4. Download `artifacts/graph/results_phase1d.json` and the `phase1d_*.pt` / `phase1d_*.json` model
   files into this repo's `artifacts/graph/`, then run
   `python scripts/build_graph_leaderboard.py --merge` locally to rebuild `docs/graph_leaderboard.md`
   and copy Phase 1D into `results.json`.

## `notebooks/03_voice_baseline.ipynb` (Phase 2, Voice Engine)

1. Rebuild/verify first: `python scripts/build_notebooks.py --check`. Edit
   `notebooks/src/03_voice_baseline.py` or `engines/voice/`, never the `.ipynb`.
2. Upload, then **Accelerator = GPU**, **Internet = on**. Internet is required: the notebook streams
   ASVspoof 2019 LA from HuggingFace rather than using an attached Kaggle dataset.
3. Cell 1 `pip install`s only `datasets` and `soundfile`. There is deliberately no `torchaudio` or
   `librosa` dependency — LFCC, the silence trim and the resampler are implemented in plain torch/numpy
   in `engines/voice/preprocess.py`, so a torchaudio version mismatch can't break the run.
4. Run All. Expect roughly 20–50 minutes: streaming ~6000 clips (~370 MB of the 4.39 GB parquet),
   featurizing them twice (trimmed and untrimmed), and training three small LCNNs.
   `N_CLIPS` at the top of the config cell is the knob for a bigger or smaller subset — whatever it is
   set to is the number that must be quoted with the results.
5. **Then click "Save Version → Save & Run All (Commit)".** "Run All" alone does not persist
   `/kaggle/working`, so the output files disappear when the session ends. This is what cost the Graph
   Engine run its artifacts; don't repeat it.
6. Download from the Output panel into this repo's `artifacts/voice/`: `results.json`,
   `det_curves.png`, `test_scores.npz`, and the three `lcnn_*.pt` weight files. Also download
   `docs/voice_data_exploration.md`, which the notebook writes with the real EDA numbers.

## `notebooks/04_voice_depth.ipynb` (Phase 2B, Voice Engine depth pass)

1. Verify first: `python scripts/build_notebooks.py --check`. Edit `notebooks/src/04_voice_depth.py`
   or `engines/voice/`, never the `.ipynb`.
2. Upload, then **Accelerator = GPU** (T4; with T4 x2 only the first GPU is used) and **Internet = on**.
   Internet is needed for the ASVspoof parquet (4.39 GB), the protocol file, the XLS-R checkpoint
   (~1.2 GB) and the In-the-Wild archive (read in place, not downloaded whole).
3. **Skip the interactive Run All — go straight to "Save Version → Save & Run All (Commit)".** A commit
   re-runs the notebook from scratch in a fresh session anyway, so running it interactively first would
   spend the GPU time twice. Watch progress in the version's log.
4. Expect roughly 2–3 hours: download ~5 min, one decode/featurize pass ~10 min, XLS-R feature
   extraction ~15 min, ~13 LCNN trainings, bootstrap CIs, In-the-Wild. Optional steps check a 10-hour
   budget and are recorded as `not_run` if skipped.
5. Disk: the feature cache in `/kaggle/working/cache` peaks at roughly 15–16 GB (under the 20 GB output
   limit) and is deleted at the end of a successful run; the ASVspoof parquet goes to the `datasets`
   cache outside `/kaggle/working`. If disk is tight, lower `SPOOF_PER_BONAFIDE` (never cut bonafide).
6. Download from the Output panel into `artifacts/voice/`: `results_phase2b.json`,
   `det_curves_phase2b.png`, `phase2b_test_scores.npz`, and the `phase2b_*.pt` weights. Then run
   `python scripts/build_voice_leaderboard.py` locally to rebuild `docs/voice_leaderboard.md`. Keep
   `results_phase2b.json` separate from Phase 2's `results.json`; the script refuses smoke-test files.

## `notebooks/05_visual_v1.ipynb` (Phase 3, Visual Engine Version 1)

This notebook runs in **stages**, and stops after the one named in `STAGE` (first code cell). The
default is `task0`, the pre-flight. No embedding is extracted until the Task 0 results have been read
and the `PLAN` values in the notebook have been decided from them.

### Task 0 (pre-flight and streaming-feasibility test)

1. Verify first: `python scripts/build_notebooks.py --check`. Edit `notebooks/src/05_visual_v1.py` or
   `engines/visual/`, never the `.ipynb`.
2. On Kaggle: **Create -> New Notebook -> File -> Import Notebook**, upload `notebooks/05_visual_v1.ipynb`.
3. In the right-hand panel:
   - **Accelerator = GPU** (T4; with T4 x2 only the first GPU is used) and **Internet = on**.
   - **Add Input** -> search for and add the **ArtiFact** dataset (`awsaf49/artifact-dataset`) and the
     **FFHQ thumbnails** dataset ("Flickr Faces 70k Thumbnails 128x128").
   - **Add-ons -> Secrets** -> switch **`HF_TOKEN`** on for this notebook. The notebook reads it from
     Kaggle Secrets, records only whether it is present, and never prints or saves its value.
   - Keep the notebook **private** (Share -> Private). Task 0 writes only aggregate counts, but the later
     stages write per-image metadata and embeddings to the output, which the datasets' terms do not
     allow in a public place.
4. Leave `STAGE` as it is (`task0`). **Skip the interactive Run All -- go straight to "Save Version ->
   Save & Run All (Commit)"**, and close any interactive session of the same notebook so it does not
   use the GPU quota in parallel.
5. What it does, in order, each step printing as it goes: environment and real free disk; OpenFake file
   listing and licenses; **real OpenFake counts read from the metadata columns only** (no image
   downloaded; a progress line every 25 shards); a **timed streamed sample** of about 2,400 images from
   two shards (downloaded, decoded, dropped); how many shards a per-family quota needs and the projected
   hours; what the ArtiFact and FFHQ folders really contain; CLIP ViT-L/14 load, embedding size,
   preprocessing check, determinism and speed; the projected embedding-cache size; whether the
   Deepfake-Eval-2024 access request has been approved.
6. A check that fails is recorded with its error and the others still run; the version does not fail.
   The run time is not known yet -- that is one of the things this run measures. The metadata scan stops
   by itself after 45 minutes and reports how many shards it reached.
7. Download from the Output panel into `artifacts/visual/`: `preflight_task0.json`,
   `openfake_model_counts.csv`, `openfake_shard_counts.csv` (all aggregate counts), and download the
   version's log. Report these back before changing `STAGE`.

### Later stages (`scan` -> `split` -> `extract` -> `heads`)

Not to be run until Task 0 has been reported and the `PLAN` decided. Each later stage writes its chunks
under `artifacts/visual/private/` in the notebook's output. To resume or continue in a new session, add
the previous version's output as an input (**Add Input -> Notebook Output Files -> Your Work**); the
notebook finds `/kaggle/input/*/artifacts/visual/private` by itself and skips every finished chunk.
Nothing under `artifacts/visual/private/` is ever copied into this repository.
