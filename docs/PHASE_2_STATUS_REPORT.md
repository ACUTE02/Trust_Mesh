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

## Revision 1 — easy language first, technical detail on request (2026-10-08)

Spec: `PHASE_2_FRONTEND_REVISION_1.md` (parent `docs/` folder). Test counts after this revision:

- Python unit tests (`pytest tests`): 109 passed
- Frontend tests (inside `npm run build`): 26 passed
- `scripts/verify_serving.py` against the Docker stack: 11 of 11 ok

**What changed**
- **Two layers on all three pages.** The default layer is plain language. A "Show the detail for engineers" button
  (one per page, remembered for the browser session) opens every "For engineers" block on that page: protocol, exact
  model names, full leaderboards, intervals, terse limits and a glossary.
- **New headlines and captions** in the style the revision asks for. Every score now has a line beside it saying what
  it means and whether higher or lower is better. Metric and model names (PR-AUC, EER, XGBoost and so on) no longer
  appear in the default layer; a test fails the build if one does.
- **Landing page shortened and reordered:** instrument, "What TrustMesh does" (three lines), two engine cards that are
  deliberately not twins, the combined case desk with its fixed caveat, an honest status list (measured /
  demonstration only / designed, not built), then "For engineers": a hand-built system diagram, a footprint table, a
  verification strip and an eight-row decisions table. The old "What the evidence shows" and "What was cut" sections
  were removed from this page because they repeated the engine pages.
- **Plain figures are derived entries.** "About 1.7 in 100", "0.75" and six more are registered in the data module with
  the entry they come from and the formula; `scripts/check_frontend_numbers.py` recomputes each one (81 entries, all
  match). The page test was tightened: it now also rejects any one-decimal number and any "N in 100" that is not from
  the data module. Verified by planting "About 3.25 in 100" on the landing page: the build failed, and passed again
  once it was removed.
- **Technical view on each desk** (off by default): every request the page made, its status, the round-trip time
  measured in the browser, the raw response, the model and weights file the service reports (the seed is read from
  the file name), and on the graph desk the Cypher text. The Cypher comes from a new read-only endpoint,
  `/graph/query/cypher`, which returns the same query functions the store runs (one new Python test).
- **Fixes from the screenshots:** the legend no longer touches "Datasets and citations" (a rule and space between);
  the disabled "Score this clip" button was a faded button and is now a dashed outline with full-strength secondary
  text (6.0:1 in light, 8.2:1 in dark against the page) plus the words "Add a recording first."
- **Found and fixed while testing:** a page opened while the service was still loading its models showed "service not
  running" until reloaded, because `/health` was asked only once. It now retries for about 75 seconds.
- **New guide:** `docs/HOW_TO_TEST_MYSELF.md`.
- `start.bat` and `stop.bat` (added on the owner's request before this revision) are now committed.
- No model code, no training, no push, no deploy. The only serving change is the read-only Cypher-text endpoint.

**Checks run**
- Number check: 81 of 81. Build: clean, 26 frontend tests. Python: 109 passed. `verify_serving.py`: 11 of 11.
- Clicked through in the browser against the Docker stack, with the synthetic tone:
  the combined desk (graph only, then graph plus recording, Technical view shown and hidden); the graph desk
  (score, neighbours, Technical view showing five requests and the Cypher, a clear error for transaction 999999, the
  engineer toggle); the voice desk (score, levels, Technical view, the disabled button's new style).
- 18 empty-state screenshots re-captured at 390, 768 and 1440 px in both themes: no horizontal overflow in any, and
  `check_no_dataset_outputs.py` passes on them. They show the default (plain) layer only.

**Captions I could not fully simplify without a new claim, or that break a rule**
- The revision's examples say "calls". The data are recordings, not phone calls, so the site says "recordings".
- The voice score's meaning line has a 16-word sentence ("How often it gets a recording wrong, when missed fakes and
  false alarms are equally common."). Shorter versions dropped the "equally common" condition, which is the definition.
- The hero keeps the word "split", as the revision's fixed caption requires, and explains both kinds beneath it.
- "A large ready-made speech model with a small add-on" is as plain as the chosen model gets without saying how it
  was pre-trained, which the project log does not record.
- "Voice model D" still appears in the default layer where the page reports which model the service is running; it
  comes from the service and is not a headline. The names "Graph Engine", "Voice Engine" and "Visual Engine" remain
  in the navigation and the status list.
- A few explanatory sentences in the default layer run to 16 to 20 words (the camouflage explanation, the levels
  caveat, two limits). They were read aloud for jargon; none was found, but they were not all cut to 15 words.
- No reading-level tool was run. "About grade 8" was judged by reading, not measured.
- Left out rather than invented: a plain figure for the camouflage result (it is a share caught, already plain), and
  any statement of how the speech model was pre-trained.

**Not tested**
- The microphone button and any real speech recording (only a synthetic tone was scored).
- Keyboard-only use, screen readers, and reduced motion with the system setting on.
- The engineer layer and the Technical view at 390 and 768 px in a real browser (the unit tests render them; the
  screenshots show the default layer only). The system diagram scrolls sideways inside its own box on a phone by design;
  that was not looked at on a phone.
- The retry of `/health` was seen to be needed (the page was caught in the "down" state), but the fixed build was not
  re-observed recovering during a restart.
- The steps in `docs/HOW_TO_TEST_MYSELF.md` that need a person: recording, a text-to-speech clip, a phone.
- A clean-machine start, any deployment, and everything else listed as untested under WP8 and WP9.

## Self-test — ST1 to ST10 (2026-10-08)

Runbook: `PHASE_2_SELF_TEST_RUNBOOK.md` plus the owner's extra step ST10. Full results in easy language:
`docs/SELF_TEST_REPORT.md`. Test counts after this run:

- Python unit tests (`pytest tests`): 116 passed
- Frontend tests (inside `npm run build`): 28 passed
- `scripts/verify_serving.py` against the Docker stack: 12 of 12 ok

| Step | Result |
|---|---|
| ST1 headline numbers | Reproduced. Graph 0.7474 (difference 0.00003). Voice per seed 0.0194, 0.0152, 0.0152; mean 0.0166, spread 0.0019. Tolerance 0.0005. No stop condition. |
| ST2 live against offline | Graph: 200 test nodes, maximum difference 0. Voice: **not run**, the evaluation audio is not local. |
| ST3 thresholds out of sample | 200 half-splits: genuine flagged high 1.1 in 100 (in-sample 1.0); fakes flagged medium or high 99.5 in 100 (in-sample 99.5). An estimate from one dataset. Served thresholds unchanged; nothing added to the site. |
| ST4 seed-42 interval | EER 0.0194, 95% interval 0.0159 to 0.0214 (1,000 stratified resamples). |
| ST5 fail-safe | 12 cases. Two real faults found and fixed (silence was scored "low risk"; a 0.1 s clip got a confident score). All pass now. Known limit: no check that a sound is speech. |
| ST6 browser | Every control reached by Tab on all pages; all three desks run by keyboard; reduced motion turns the animation off; axe-core 0 violations in 12 runs; no overflow in 36 combinations; microphone flow works with a fake device (a tone, not speech). |
| ST7 clean checkout | Two real faults found and fixed: `frontend/src/data/` was never committed (the root `data/` ignore rule hid it), so a fresh clone could not build the site; and missing files produced a traceback with the site never starting. Now a plain list is printed and the site still opens. API image rebuild in the clone timed out (about 1 GB of packages); the existing image built from the same code was used. |
| ST8 clean-ups | `verify_serving.py` leaves no rows (`record=false`, checked by the script and a unit test); issue #45 investigated, report only; sanitised `results.json` copy written and left uncommitted (79 of 81 entries check against it); recording mode added and tested. |
| ST9 report | `docs/SELF_TEST_REPORT.md`. |
| ST10 owner's clips | 3 pairs. **0 of 3 genuine recordings passed; all 3 were flagged high. 3 of 3 clones were flagged high.** The model did not separate them. A 4-clip diagnostic (genuine dataset recordings re-encoded as low-bitrate AAC) moved 3 of 4 out of "low". A small demo check, not an accuracy measurement. Per-clip data is outside the repository. |

**Serving-layer changes made in this run (no model code):** silent audio refused; very short audio flagged
(`low_confidence`, `warnings`); plain messages for empty and unreadable files; `record=false` on the three scoring
endpoints; `recording_mode` in `/health`; `serving/preflight.py` and its use in the loader and API; nginx resolves the
API per request and the frontend no longer waits for the API; the instrument no longer depends on an animation frame.
**Repository change:** `.gitignore` `data/` became `/data/`; `frontend/src/data/` is now tracked.
**Installed:** `imageio-ffmpeg` and `axe-core`, both inside the project (`.venv` and `frontend/node_modules`). Nothing system-wide.
**Created outside the repository:** `E:\TrustMesh\my_voice_tests\_converted_wav\` (WAV copies), `MY_VOICE_RESULTS.md` and
`my_voice_results.json` next to them. The clean-clone folder and its images were removed after the test.

**Not run:** the voice half of ST2; a first start with real downloads; real speech through the microphone; screen
readers; a real phone; the shortest-path box by keyboard and in recording mode; anything needing training.

## Why do genuine phone recordings score as fake? (2026-10-08)

Question from the owner after the self-test: all 3 genuine voice-memo recordings were flagged as fake, while the
In-the-Wild check gave an error rate of 0.0800. Script: `scripts/diagnose_phone_audio.py`. This was a diagnosis only:
no training, no model change, no threshold change, nothing pushed or deployed.

**Every number below is a total or an average.** The owner's recordings are 3 genuine clips and 3 clones; the
research-dataset clips are the 4 genuine and 4 fake clips already on this machine. These are very small samples. They
show which factors matter in these clips. They do not measure how often the model fails.

The scoring in this test used the same code and weights as the service. It reproduced the service's score for a clip
exactly (difference 0), so this is not a bug in the service.

### What the owner's recordings are

- The 3 genuine recordings are voice-memo files: AAC at 32 kb/s, already 16 kHz, mono. The 3 clones are MP3 at
  128 kb/s, 44.1 kHz.
- They are 7 to 10 seconds long (average 8.7 s genuine, 7.7 s clones). The research clips are about 3 seconds.

### Check 1: level, rate, length and trimming

| | Owner, genuine (3) | Owner, clones (3) | Dataset, genuine (4) | Dataset, fake (4) |
|---|---|---|---|---|
| Loudest point (dBFS) | -8.3 | -5.8 | 0.0 | 0.0 |
| Average level of the speech (dBFS) | -29.3 | -25.2 | -16.5 | -16.9 |
| Quietest background (dBFS) | -67.2 | -51.9 | -61.4 | -58.3 |
| Average length (s) | 8.7 | 7.7 | 3.1 | 2.7 |
| Length after trimming silence (s) | 7.6 | 7.3 | 1.7 | 1.8 |
| Share of the sound energy kept by trimming | 99.99% | 100% | 100% | 100% |
| Share of the energy above 4 kHz | 0.2% | 0.2% | 1.6% | 1.5% |

- **Trimming did not remove the speech.** It cut about 13% of the time, mostly silence, and kept 99.99% of the sound.
- The owner's genuine recordings are quieter than the research clips (about 12 dB lower on average), and have very
  little energy above 4 kHz. That is normal for a 32 kb/s voice memo.
- Their background is quiet, not noisy.

### Check 2: the genuine recordings in several versions

| Version of the 3 genuine recordings | Flagged high | Passed as low risk | Average raw score* |
|---|---|---|---|
| Straight resample to 16 kHz (the normal route) | 3 | 0 | -10.5 |
| Resampled the way the service does it (simple interpolation) | 3 | 0 | -10.5 |
| Made as loud as possible without clipping (peak -1 dBFS) | 3 | 0 | -10.4 |
| Set to the average level of the research clips | 3 | 0 | -10.6 |
| Trimmed more strictly (20 dB) | 3 | 0 | -9.2 |
| Trimmed less strictly (40, 50, 60 dB) | 3 | 0 | -10.8, -10.9, -10.3 |
| No trimming, first 4 seconds | 3 | 0 | -10.1 |
| No trimming, middle 4 seconds | 3 | 0 | -9.8 |

*The raw score is the number the model produces before it is turned into a 0-to-1 score. Higher means more genuine.
Genuine research clips average about +8 on this scale. Fake research clips average about -14.

- **Nothing in this table changes the outcome.** Level, trimming, and how the clip is resampled are not the cause.
  The model normalises loudness itself, so the level test was expected to change nothing, and on the research clips
  it changed the raw score by at most 0.2.
- The owner's genuine recordings were already 16 kHz, so there was no real resampling to go wrong.
- The 3 clones scored lower (more fake) than their genuine twins in all 3 pairs, by about 4.6 on the raw scale. So the
  model did see some difference. But both sit far below the warning level.

### Check 3: the site's own microphone path

The browser's microphone cannot be used in an unattended test. Instead the browser's fake microphone played the owner's
genuine recording, and the site recorded it, converted it, and sent it, exactly as it does for a real microphone.
All 3 came out as "high risk" (average raw score -11.5, against -10.5 for the straight route). The 3 genuine recordings
and the 3 clones, uploaded as files through the site, were all flagged high too (6 of 6). The browser route differs from
the straight route by about 1 point on the raw scale. So **the site's recording and conversion is not the cause.**
This is not real speech in a real room: it is the real recording played into the browser.

### Check 4: what do codecs, noise, level and rooms do to genuine research clips?

The 4 genuine and 4 fake research clips were damaged one way at a time and scored again. Counts are out of 4.

| What was done to the genuine clips | Stayed "low risk" | Became "medium" | Became "high" | Average raw score |
|---|---|---|---|---|
| Nothing (the original clips) | 4 | 0 | 0 | +7.8 |
| Resampled through 48 kHz and back, no codec | 3 | 1 | 0 | +6.1 |
| MP3 128 kb/s | 2 | 2 | 0 | +3.7 |
| AAC 128 kb/s | 4 | 0 | 0 | +7.1 |
| **AAC 32 kb/s (voice-memo quality)** | **1** | 1 | **2** | -1.8 |
| AAC 32 kb/s, after a 48 kHz step like a phone | 0 | 0 | 4 | -6.3 |
| Opus 64 kb/s | 3 | 1 | 0 | +6.4 |
| Opus 24 kb/s | 2 | 2 | 0 | +3.8 |
| Quieter by 20 / 40 / 60 dB | 4 / 3 / 2 | 0 / 1 / 2 | 0 | +8.0 / +7.1 / +4.6 |
| Louder by 12 dB, clipped | 3 | 1 | 0 | +4.7 |
| White noise (30 / 20 / 10 dB signal-to-noise) | 4 each | 0 | 0 | +8.8 / +10.3 / +8.2 |
| **Room echo (reverb, 0.4 s)** | **0** | 0 | **4** | -15.9 |

- **The fake clips stayed "high risk" in nearly every case** (at most 1 of 4 slipped, under heavy white noise).
  Whatever damages a clip pushes the model towards "fake". It almost never pushes a fake towards "genuine".
- **Heavy compression matters.** At voice-memo quality (AAC 32 kb/s), only 1 of 4 genuine research clips stayed low
  risk. At 128 kb/s, 4 of 4 did. Low bitrates and extra resampling steps both hurt.
- **Room echo hurt most.** With a simple simulated room echo, none of the 4 genuine clips passed, and their average
  raw score fell to -15.9, as low as a typical fake.
- **Background noise did not hurt** genuine clips. Level only hurt at extreme quietness.
- The model is brittle even with no codec: a plain trip through 48 kHz and back moved 1 clip of 4 out of "low".

### Check 5: how different are the owner's recordings, overall?

- On the raw scale, the owner's genuine recordings average -10.5 (range -12.4 to -7.8). Genuine clips in the research
  test set (2,165 of them) mostly sit between +2 and +13. **None of them** looked less genuine than the owner's
  recordings. About 25% of the research test set's fake clips (8,992) looked less fake than the owner's genuine ones.
- The owner's clones average -15.1, close to the typical research fake (-14.5).
- Inside the model, the owner's genuine recordings and the owner's clones look as similar to each other (average
  similarity 0.979) as the genuine recordings look to one another (0.979). That is more than genuine research clips look
  to one another (0.971). So the model mostly sees "this speaker, this recording" and much less "real or cloned".
- Seen from the middle of the research genuine clips, and from the middle of the research fake clips, both of the
  owner's groups sit slightly nearer the fake side (3 of 3 each). This is only 3 clips against 4, and the differences
  are small. It is a hint, not a result.
- The research clips that were damaged with AAC stayed nearer the genuine side in this view (0 of 4 nearer the fake
  side), even though the model flagged many of them. So this view does not explain the model's decision. It only
  describes how different the recordings are.

### So what explains the failure?

**Ruled out** (tested, and the failure stays the same): loudness, trimming, resampling, background noise level,
clipping, and the site's browser path. It is also not a bug in the service.

**Shown to hurt genuine speech, on research clips:** heavy compression (AAC at 32 kb/s, the voice-memo setting) and
room echo. Both push genuine clips towards "fake" and neither pushes fake clips towards "genuine".

**Not established:**
- **Which of these, or what else, causes the owner's result.** Heavy AAC compression on research clips brought the
  average raw score from +7.8 to -1.8. The owner's recordings average -10.5. So compression alone, as tested here,
  explains **only part** of the gap. Room echo, the phone's microphone, and the speaker (a different voice and accent
  from the research recordings) are all untested for the owner's voice, and could explain the rest. The owner's
  recordings only exist in compressed form, so these cannot be told apart from the files we have.
- **How the In-the-Wild result fits.** The error rate of 0.0800 is the error at the best warning level for that test
  set, not at the warning levels the site uses. It tells us the model ranks fakes below genuine clips fairly well on
  that set. It does not tell us how many of its genuine clips would be flagged at the site's levels. The In-the-Wild
  audio is not on this machine, so that was not checked. The owner's pairs also ranked the clone below the real
  recording in 3 of 3 cases, so a ranking signal exists here too. That is a hint from 3 pairs, not a result.

**Warning levels were not changed**, as asked. Changing them would hide this problem, not explain it.

### What would settle it (needs the owner, not training)

Record the same sentences once in a **lossless** mode (a recorder app set to WAV), once close to the microphone in a
quiet small room, and once in an echoey room. Compress the quiet WAV to AAC at 32 kb/s ourselves. Then the same script
can separate compression, room, and microphone/speaker, with the same speaker each time.

### Not run

- Real microphone and real room: the browser test used a fake microphone playing the recording.
- An uncompressed recording of the owner's voice: none exists, so compression could not be separated from the room,
  the phone, and the speaker.
- In-the-Wild audio: not on this machine, not downloaded.
- More research clips: only the 4 genuine and 4 fake clips on this machine were used. Each count above is out of 4 or
  out of 3. A larger sample could change the picture.
- Other speakers, other phones, other recording apps.
- Anything that needs training, such as training with compressed audio (the unrun Model F idea).

Per-clip detail stays in `E:\TrustMesh\my_voice_tests\PHONE_AUDIO_DIAGNOSIS_PRIVATE.json` and the `_diagnosis` folder
next to the recordings (outside the repository). Totals are in `data/self_test/phone_audio.json` (git-ignored).
