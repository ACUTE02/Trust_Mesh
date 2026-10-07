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

## WP4 — React frontend (2026-10-08)

- Built in the order the spec asks: the data module first (68 entries, each checked against its source by
  `scripts/check_frontend_numbers.py`), then the two-scores instrument (0.7730 random, 0.5115 time split, difference
  0.2616), then the pages, then the desks wired to the real API.
- Streamlit is gone: `frontend/app.py` and `Dockerfile.frontend` deleted (both were created earlier in this project), the compose service replaced, README and docs updated. The site is on port 8080; 8501 is unused.
- `npm run build` is clean: type check, 16 tests, production build. The tests include the build-time rule that a
  rendered number must come from the data module or the live API, the instrument, the case card with empty fields,
  and `ModelInUse` for a D response, a fallback-to-C response and a response with no model.
- The frontend container runs in Docker Compose (nginx, 93 MB image, 13 MB of memory) and proxies `/api`.
- **Clicked through in the browser against the real API, all done:**
  1. Overview: instrument switched to the time split; the status table showed "Serving now: Voice model D
     (phase2c_D_ssl_mean_only_P1_s42.pt)" read from `/health`.
  2. Case desk, graph only: graph score shown, voice and combined rows read "empty", visual row dashed.
  3. Case desk, combined: graph score, voice score from Model D, combined score, with the fixed
     "no measured accuracy" note.
  4. Voice desk: clip loaded, waveform and spectrogram drawn, scored, trimmed silence shaded, thresholds and model shown.
  5. Graph desk: score, the four graph statistics, neighbourhood drawing and table from Neo4j, illicit-nearby list,
     the secondary GAT explanation.
  The clip in steps 3 and 4 was a synthetic tone made for the test and put into the page's file input by a script.
  **Not done:** the microphone button, a real speech recording, the shortest-path form in the browser (the endpoint
  itself passes in `verify_serving.py`), and the API-down state in a real browser (it is covered by the unit tests).
- **Deployment profile verified** (the WP3 item that needed this image): with `docker-compose.deploy.yml`, host ports
  7474, 7687 and 8000 refused connections, the site answered on the public port, a Neo4j-backed query worked through
  `/api`, `/api/docs` returned 404, and a foreign origin got no CORS header. Tested with the existing volume's
  password; a changed password on an existing volume was not tested.

## WP5 — Critique and polish (2026-10-08)

| Check | Result |
|---|---|
| Screenshots at 390, 768, 1440 px, both themes, three pages | 18 captured and reviewed |
| Against the spec's "discarded" list | None of the discarded patterns present |
| Generic defaults named and changed | The bordered theme-toggle button; the four-column tables that overflowed a phone |
| One decorative element removed | The rule under the header |
| Contrast, all pairs, both themes | Pass: 34 of 34 (`scripts/check_contrast.py`) |
| No horizontal page scroll | Pass: 18 of 18 captures |
| Keyboard operation of the instrument | Native radio group; unit-tested. **Not tabbed through by hand** |
| Keyboard operation of the demos | Native controls. **Not verified by hand** |
| Visible focus | In the stylesheet for every focusable element. **Not screenshotted** |
| `prefers-reduced-motion` | Bar transition disabled in CSS. **Not verified with the OS setting on** |
| Text or table equivalent for every chart | Yes (see `docs/design_decisions.md`) |
| Screen reader | **Not tested** |
| No dataset clip, per-sample prediction or weight in the saved screenshots | Pass: empty states only; `check_no_dataset_outputs.py` passes. One slip found and fixed first: the example menu named a transaction's dataset label |

Screenshots are saved in `docs/frontend_screenshots/` (18 files, 8.8 MB).

## WP6 — Demo documents and hosting analysis (2026-10-08)

- `docs/demo_runsheet.md`: what must not be on screen, inputs to prepare, start and warm-up, the two-minute script,
  hosting analysis, pre-deploy checklist, teardown checklist, and a list of the steps that were not tested.
- Hosting, in one line: record from the local Docker stack (Option A). The stack needs about 4 GB of memory and
  15 GB of disk, so free single-container web tiers cannot hold it; a 4 to 8 GB virtual machine, a tunnel from this
  machine, or static hosting of the frontend alone can. **No provider page was read; every provider-specific limit
  is marked unverified in the runsheet.**
- README: results table, architecture diagram, one-command run, honest limits, datasets and citations from WP2,
  one screenshot.
- Could not test: whether a person who has never seen the project can follow the runsheet from a clean machine.
  The start, warm-up, check and deployment-profile steps were run here; HTTPS, a real host, a tunnel and the
  teardown commands on a remote host were not.

## WP7 — Repository hygiene (2026-10-08)

- `git init` in `TrustMesh_CODE` (branch `main`), five local commits, working tree clean, **no remote, nothing pushed**.
- The pre-commit hook (`scripts/check_no_dataset_outputs.py`) is installed and was tested: a planted
  `artifacts/voice/planted_test_weights.pt`, force-added, was refused with "FORBIDDEN ... model weights trained on a
  dataset". The planted file was then removed. The check was extended to cover the Graph and Voice engines.
- Excluded from git, and why:

| Excluded | Why |
|---|---|
| `data/` (Elliptic, the encoder, demo clips, caches, `cases.sqlite`, backups) | Dataset files, dataset audio, per-sample predictions |
| `artifacts/**/*.pt`, `artifacts/graph/*xgboost*.json` | Trained weights (Elliptic is CC BY-NC-ND; publishing weights is an open owner decision) |
| `artifacts/**/*.npz`, the extracted `phase2c_*` folders, `_to_delete/` | Per-sample test scores and extracted weights |
| `artifacts/graph/results.json`, `results_phase1c_partial.json` | They hold one transaction's prediction and explanation example. **Consequence:** on a fresh clone `scripts/check_frontend_numbers.py` cannot check the entries sourced from that file. The owner may decide to commit it |
| `*.wav`, `*.flac`, `*.mp3`, `*.ogg`, `*.sqlite` | Audio clips; the cases table |
| `.env` (not `.env.example`) | Local settings and secrets |
| `node_modules/`, `frontend/dist/`, `.venv/` | Installed packages and build output |

- Committed on purpose: aggregate results (`results_phase1d.json`, the voice result files, threshold files, plots),
  code, specs, docs and the 18 empty-state screenshots. The notebooks have no cell outputs.
- For the owner to review before any push: `project.md` (the stale log) is committed as it stands; the commits use
  the name and email from the owner's account as the author.

## WP8 — Final verification (2026-10-08)

| Check | Result |
|---|---|
| Python unit tests (`pytest tests`) | 108 passed |
| Serving tests alone | 33 passed |
| Frontend tests (inside `npm run build`) | 16 passed; build clean |
| `pyflakes serving scripts tests/test_serving.py` | clean |
| `scripts/build_notebooks.py --check` | up to date |
| `scripts/verify_serving.py` against the Docker stack with the new frontend container | 11 of 11 ok, exit 0 |
| Start from an empty Neo4j volume in Docker (separate compose project) | Loader filled 203,769 nodes and 234,355 edges in 113 s; verify 11 of 11 |
| Every number on the three pages against its source | 68 of 68 match (`scripts/check_frontend_numbers.py`), and the page tests confirm no other number is rendered |
| `scripts/check_no_dataset_outputs.py --all` | 138 tracked files, none forbidden |
| Start from a clean machine (empty `data/`) | **Not done.** Re-downloading Elliptic and the 1.27 GB encoder was impractical on this connection (the encoder took about an hour earlier) |

Known failure left as found: none. Known mismatch left open: issue #45 (two runs disagree on the GAT-tuned and GIN
figures; the site shows the log's and says so).

## WP9 — Hand-off (2026-10-08)

**Files changed or added in this run** (all inside `TrustMesh_CODE` unless a full path is given):
- Added: `frontend/` (React site, replacing the Streamlit file), `docker-compose.deploy.yml`, `.env.example`,
  `docs/PHASE_2_STATUS_REPORT.md`, `docs/demo_runsheet.md`, `docs/frontend_screenshots/` (18 images),
  `scripts/check_frontend_numbers.py`, `scripts/check_contrast.py`, `scripts/capture_screenshots.py`, the `.git` folder.
- Changed: `serving/api.py`, `serving/config.py`, `serving/fusion.py`, `serving/graph_service.py`,
  `serving/voice_service.py`, `tests/test_serving.py`, `scripts/calibrate_voice_thresholds.py` (docstring),
  `scripts/check_no_dataset_outputs.py`, `scripts/verify_serving.py` (lists all clips), `docker-compose.yml`,
  `.gitignore`, `requirements-dev.txt`, `README.md`, `docs/design_decisions.md`, and `E:\TrustMesh\PROJECT_LOG.md`.
- Deleted: `frontend/app.py`, `Dockerfile.frontend` (the Streamlit page and its image).
- Local state changed: `data/cases.sqlite` (test rows removed; backup in `data/backups/`), Docker images rebuilt.
- Not touched: `project.md`, any Visual Engine code, any model, the extracted folders and `_to_delete/` in
  `artifacts/voice/`.

**Memory table:** see WP0. In short: API about 2.1 GB, Neo4j up to 1.4 GB, frontend 13 MB.

**Terms findings that need an owner decision:** the three Elliptic questions under WP2.

**Open and unverified items:**
1. Clean-machine start not tested.
2. No real deployment, HTTPS, tunnel or static-only hosting tested; provider limits unverified.
3. Microphone recording and real speech through the new frontend not tested.
4. Keyboard-only use, screen reader and reduced motion not verified in a live browser.
5. Issue #45: which run's GAT-tuned and GIN figures are canonical.
6. `artifacts/graph/results.json` is kept out of git; the number check is therefore complete only on this machine.
7. The terms of the VCTK corpus behind ASVspoof were not read.
8. `verify_serving.py` writes test rows into the cases table each time it runs.

**Owner-only list, with exact next steps:**
1. **Decide the Elliptic questions** (`docs/design_decisions.md`, "Terms findings"). Until then keep real
   transaction scores out of anything public.
2. **Record the video** from the local stack: `docs/demo_runsheet.md`, sections 1 to 4. Record your own voice clips first.
3. **Hosting:** choose Option A (no hosting) or one of the options in the runsheet's section 5; if you deploy, follow
   sections 6 and 7 and keep Neo4j's ports closed.
4. **Review, then push:** in `TrustMesh_CODE` run `git log --stat` and `git ls-files`, check nothing listed should
   stay private, add your remote and push. Nothing has been pushed.
5. **Spec defaults** you may overrule: the combined case desk is on the landing page; no email or phone is shown.
6. **Sync the second log copy** (`project.md` is stale and was left alone, as instructed).
7. Anything on Kaggle, including a seeded re-run if you want issue #45 settled.
