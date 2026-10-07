# Phase 2 status report (runbook `PHASE_2_MASTER_RUNBOOK.md`)

Plain-English record of each work package, newest entry last. "Verified" means it was run and the result
was seen. Anything not run says so.

## WP0 — Pre-flight (2026-10-08)

**Spec location.** The runbook and both Phase 2 specs are in `E:\TrustMesh\docs\`, as the runbook says.

**Stack.** The Docker stack was already up (Neo4j healthy, API, the old Streamlit frontend; the loader had
exited 0). `scripts/verify_serving.py` against it: all 11 checks passed, exit 0. `/health` reports
`serving_default: D`, weights `phase2c_D_ssl_mean_only_P1_s42.pt`, `fallback_active: false`.

**Memory (measured on this machine, Docker Desktop on WSL2, 7.39 GiB VM limit).**

| Service | `docker stats` idle | `docker stats` while scoring | Process resident memory (RSS) | Peak |
|---|---|---|---|---|
| API (XGBoost + GAT + XLS-R encoder + both voice heads loaded) | 744 MiB | up to 794 MiB (2 s samples) | 2,139 MB | 2,253 MB process peak; 2,162 MB cgroup peak |
| Neo4j (203,769 nodes, 234,355 edges; heap max 1 GB, page cache 1 GB) | 798 MiB | up to 1,022 MiB | not read | 1,392 MB cgroup peak; 867 MB process peak |
| Frontend (Streamlit, to be replaced) | 118 MiB | 118 MiB | not read | not read |
| Loader (one-shot, exits) | not measured this run | | | |

Important for hosting: **`docker stats` under-reports the API by a factor of about three.** The process really
holds about 2.1 GB (the 1.27 GB encoder weights plus the Elliptic feature matrix and libraries). Plan for
about 2.3 GB for the API and about 1.4 GB for Neo4j, so roughly 4 GB with headroom, not the ~1.5 GB that
`docker stats` suggests. The loader's peak was not measured; it loads the graph and XGBoost but not the encoder.

Disk: API image 3.39 GB, Neo4j image 986 MB, Neo4j data volume 1.25 GB, `data/` folder (Elliptic, encoder,
caches) about 2.0 GB.

**Third (visual) signal slot.** Not present. `serving/fusion.py::fuse` takes exactly two scores and
`/case/score` returns `graph`, `voice`, `fusion` only. WP3b is therefore needed.

**Tools found.** Node 24.16, npm 11.13, git 2.54, Docker 29.6. `TrustMesh_CODE` is not a git repository yet.

## WP1 — Model D documentation (2026-10-08)

- Re-checked `README.md`, `docs/design_decisions.md`, `PROJECT_LOG.md`, `serving/`, `scripts/`: no document claims
  Model C is the served model. Remaining mentions of C are the `?model=C` fallback and the history in issues #37 and #44.
- Fixed the docstring of `scripts/calibrate_voice_thresholds.py`: it no longer calls the thresholds "held-out". It now
  says they are calibrated in-sample on the test scores. Docstring only; re-running the script gives the same
  thresholds (D: 0.6578 high, 0.0239 medium).
- `TrustMesh_CODE/project.md` was not touched.
- Left alone on purpose: one API message in `serving/voice_service.py` says "no held-out test scores ... are available
  locally" for a model with no calibration file. It describes missing scores, not a rate, and it is program text, which
  WP1 said not to change.
- Tests: `tests/test_serving.py` 25 passed.

## WP2 — Terms and licences (2026-10-08)

Full table with source links: `docs/design_decisions.md`, section "Terms findings". Not legal advice.

| Item | Licence found | Read at |
|---|---|---|
| Elliptic Data Set | CC BY-NC-ND 4.0 | Kaggle's record for `ellipticco/elliptic-data-set` (the API record; the HTML page did not render for the reader tool) |
| ASVspoof 2019 LA | Open Data Commons Attribution License (ODC-By); covers the database, not the audio contents | Edinburgh DataShare item 10283/3336 and its `LICENSE_text.txt`; the Hugging Face copy states the same |
| In-the-Wild | Apache 2.0 | deepfake-total.com/in_the_wild |
| Schibsted Grotesk, Martian Mono | SIL OFL 1.1 | each project's `OFL.txt` on GitHub |

The frontend spec names two font families, not three. Both are open, so nothing was substituted.

**Owner decisions needed (Elliptic, CC BY-NC-ND 4.0):**
1. Whether to show real Elliptic transactions with their model scores in a public video, screenshot or deployment.
   The licence allows sharing the data non-commercially with attribution but says adapted material may be produced
   "but not Share[d]". Whether a per-transaction score is adapted material is not settled by the text. The build takes
   the conservative reading: nothing of that kind is in the repository or in saved screenshots.
2. Whether to ever publish the graph model weights (same question).
3. Whether a portfolio shown to employers counts as "NonCommercial".

Not verified: the terms of the VCTK corpus that ASVspoof's speech comes from (not read).
No stop condition applied: the spec already keeps per-sample predictions out of public material, so nothing it shows
publicly is clearly forbidden.

## WP3 — Security hardening (2026-10-08)

- **Settings from the environment.** Neo4j password, CORS origins, upload limits and the rate limit are environment
  variables with documented defaults in `.env.example`. `trustmesh-demo` remains only as the local default.
- **Deployment profile.** `docker-compose.deploy.yml` (used on top of `docker-compose.yml`) removes Neo4j's and the API's
  published ports, publishes only the frontend, and refuses to start unless `NEO4J_PASSWORD` and
  `TRUSTMESH_CORS_ORIGINS` are set. The frontend's nginx proxies `/api` and does not expose `/docs`, `/openapi.json` or
  the `/cases` listings.
- **CORS.** The API had no CORS policy before; it now allows only the configured origins.
- **Upload limits.** Audio uploads: content type must be audio (415 otherwise), at most 10 MB (413), at most 60 seconds
  (422), and anything libsndfile cannot decode is refused (422). All four are covered by unit tests.
- **Rate limit.** 30 scoring requests (POST) per minute per client address, in memory. It protects one small machine
  from a runaway loop; it is not a defence against a determined attacker. Tested.
- **Cases table.** Backed up to `data/backups/cases_2026-10-08_before_cleanup.sqlite`, then removed 12 rows that
  `scripts/verify_serving.py` had written (case ids 26 to 34, 35, 46, 56: node 154134 and the eight local demo clips).
  The 25 rows written by `scripts/seed_cases.py` were kept. Note: every run of `verify_serving.py` adds such rows again.
- Verification of the running deployment profile is recorded under WP4/WP8 below, because it needs the new frontend image.

## WP3b — Optional visual signal slot (2026-10-08)

WP0 found no slot, so one was added. `serving/fusion.py::fuse` takes an optional `visual_score` (default `None`) and
`w_visual`; `/case/score` returns `"visual": null` and `fusion.signals_used`; `/health` reports the visual engine as not
connected. With the slot empty the combined score and weights are exactly as before (tested for equality). The 0.5/0.5
weights are unchanged and still labelled placeholders. No Visual Engine code was imported or changed.
Tests: slot empty gives identical output; slot filled with a dummy value (named as a test-only input) gives the
three-way weighted average. `tests/test_serving.py`: 33 passed.

Two small additive API fields were also added for the frontend (a deliberate, recorded exception to "keep the API
untouched"): `/graph/score` returns the transaction's four graph statistics (`graph_features`), and `/voice/score`
returns the trimmed silence (`lead_silence_s`, `trail_silence_s`). Nothing existing changed.
