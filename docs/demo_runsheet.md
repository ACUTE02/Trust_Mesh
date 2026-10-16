# TrustMesh demo runsheet

For the owner: how to show the project, record the walkthrough, optionally put it online for a short time, and take
it down again. Written 2026-10-08. Steps marked **(not tested)** were written from the code and the compose files but
not executed; everything else was run on the owner's Windows machine with Docker Desktop.

## 0. The short version

**Option A, recommended: record the video from the local Docker stack. No hosting at all.**
The video looks exactly the same as one recorded from a hosted copy, because it is the same containers. It costs
nothing, exposes nothing, and needs no teardown. Deployment adds value in only one case: you want to hand someone a
**live link** they can click while it is up. If you do not need that, stop at Option A.

## 1. What must not appear on screen (in any public video, screenshot or post)

1. **No dataset clip and no dataset audio.** Use your own voice recording for the voice desk. Do not use the files
   in `data/demo_clips/` (they are ASVspoof clips, kept locally for verification only).
2. **Real Elliptic transactions with their scores are your decision, not a default.** Elliptic's licence is
   CC BY-NC-ND 4.0: sharing the data non-commercially with attribution is allowed, sharing "adapted material" is not,
   and the text does not settle whether a model's score for a transaction is adapted material
   (`docs/design_decisions.md`, "Terms findings"; not legal advice). The safe recording shows the findings pages and
   the voice desk, and either leaves the graph desk and the case desk unscored or shows them only in a private,
   live demo. If you decide to show them, show the citation too (it is in the footer of every page).
3. No model weights, no files from `artifacts/` or `data/`, no terminal showing `.env`.
4. No Neo4j browser with the password visible, and no password typed on camera.
5. Do not describe the combined score as accurate, validated or tested. The words on the page are the claim.
6. Do not describe the Visual Engine as working. It has no results.
7. Do not present 0.0097 or 0.0047 as scores the model achieved. They are the shortcut.
8. Do not compare the voice numbers with published ASVspoof numbers.

## 2. Inputs to prepare

- **Your own voice clips:** two or three short recordings (3 to 10 seconds each) of you speaking, saved as WAV or MP3.
  Up to 60 seconds and 10 MB each. You can also record in the page with the microphone button.
- Optional: a synthetic clip you generated yourself (a text-to-speech sample you have the right to show), to show
  the voice desk reacting to a non-human voice. A clip like this was not tested during the build; a pure tone was,
  and it scored as high risk.
- **Example cases:** the case desk and the graph desk offer three example transaction numbers. They show nothing
  until scored (see rule 2 above before scoring them on camera).

## 3. Start, warm up, check (Option A, local)

1. Start Docker Desktop and wait until it says it is running.
2. In `E:\TrustMesh\TrustMesh_CODE`: `docker compose up -d --build`. The first build downloads about 1 GB of
   packages; later starts take about a minute. The loader fills Neo4j on the first start (about 80 seconds) and exits.
3. Double-clicking `start.bat` does steps 1 to 3 for you. Wait until the "Voice check" card on
   http://localhost:8080 shows "Running now: Voice model D" (the API loads the encoder for 20 to 60 seconds).
4. Check: `.venv\Scripts\python.exe scripts\verify_serving.py` should print only `[ok]` lines.
   (It writes test rows into the cases table; that table is not shown on the site.)
5. **Warm-up:** on the Voice Engine page, score one of your clips once before recording. The first scoring call
   after a start took about 10 seconds on this machine; later ones about 2 seconds.
6. Choose the theme you want to record in (the link in the top right). Set the browser zoom to 100%.

If something fails: `docker compose ps` (every service should be "Up", the loader "Exited (0)");
`docker compose logs api --tail 50`. If Docker Desktop's engine has stopped by itself (it did twice during the
build): `docker desktop start`.

## 4. The demo script (about three minutes)

Say only what is on the page; every sentence below is backed by the project log. The site has six pages, each with its
own look and its own 3D scene (revision 2, 2026-10-17); the same yellow tags, solid / hatched / dashed marks and the
plain-language-first layers run through all of them. Start on the Overview in the theme you want to record. Scroll
slowly: the scene behind the text changes with each section, and a section is only on screen for about a screen's height.

1. **Overview, "The Mesh" (30 s).** The hero names the two checks and shows the third as a dashed row: "Designed, not
   built. No results yet." Scroll through the three findings. "The same model scored 0.7730 on a random test and
   0.5115 when tested fairly on later transactions" (the nodes split into a past sheet and a future sheet). "The first
   voice model got 1 in 100 wrong while it could hear the silence at the start, and 24 in 100 once it was removed: the
   low number was a shortcut." "Compressed phone audio is a known weakness; the warning catches MP3 and telephone-band
   audio, and caught none of the AAC and Opus copies."
2. **Click a yellow tag (10 s).** "Every number carries the run it came from and the file."
3. **Graph Engine, "The Ledger" (30 s).** "A transaction is a bead on a thread." Scroll to the two sheets (the leak),
   then the leaderboard: "a simple model that counts connections beats the best graph model, 0.7482 against 0.5913."
   Graph desk: only if you decided rule 2 above allows it.
4. **Voice Engine, "The Terrain" (60 s).** In the first panel record or choose your own clip: the landscape behind the
   page is your voice, drawn in the browser before anything is sent. **Score this clip**: a light runs across it. Scroll
   to "What phone audio does to speech": the cliff, the small table, and the honest sentence that this model has mostly
   seen clean audio. "Two ways the site says no": press the two buttons to show the collapse and the fog bank. Then the
   hatched rows of "Where the cheat was hiding".
5. **Case desk, "The Balance" (25 s).** Score a case (rule 2 again). "The weights are placeholders: each score counts for
   half, nobody tested that, and the combined score has no measured accuracy." Show the dashed list of what would make
   it real.
6. **Evidence, "The Archive" (30 s).** Pull two exhibits (each pulls its drawer in the wall). Show the dashed empty
   drawers: "what we could not test." Break the seal on Phase 2D: "training on squeezed copies reduced the problem; it
   did not remove it, and the control was weaker than the served model."
7. **About, "The Workbench" (10 s).** The drawing goes back together as you scroll; the steps to run it yourself.
8. **Close.** Switch the theme once to show the same page in the other palette. For a technical audience, press "Show the
   detail for engineers" on any page first.

Recording mode (`TRUSTMESH_RECORDING_MODE=1` on the API) hides real transaction numbers, labels and scores in every
view, including the 3D scenes: the balance gets no transaction score, the neighbourhood beads are not coloured by a
label, and the cases register shows no scores.

If the 3D looks slow while recording, add `?perf=1` to the address to see the frame rate and quality tier, or record
with reduced motion on (the pages show one still frame per world and crossfade between pages).

## 5. Hosting analysis (only if you want a live link)

Measured on this machine (see `docs/PHASE_2_STATUS_REPORT.md`, WP0):

| Service | Memory it really uses | Notes |
|---|---|---|
| API | about 2.1 GB resident, 2.25 GB peak | `docker stats` shows only about 0.75 GB; do not size from that |
| Neo4j | about 0.8 GB idle, 1.4 GB peak | heap capped at 1 GB, page cache 1 GB |
| Frontend (nginx) | under 0.1 GB (not measured separately for the nginx image) | static files |
| Loader | not measured | runs once and exits; it loads the graph and XGBoost, not the encoder |

Disk: API image 3.4 GB, Neo4j image 1 GB, Neo4j data 1.25 GB, plus `data/` (2 GB: Elliptic, the encoder) and
`artifacts/`. Plan for a machine with **at least 4 GB of memory (8 GB is comfortable) and 15 GB of disk**, able to run
Docker Compose with three long-running containers and one persistent volume. Scoring a clip takes about 2 seconds on
a laptop CPU; no GPU is needed.

What that means by kind of hosting. **Every provider-specific limit below is unverified**: no provider page was read
for this analysis, and free tiers change. Check the provider's own pricing page before choosing.

| Kind of hosting | Fits? | Why |
|---|---|---|
| A virtual machine with 4 to 8 GB RAM where you run `docker compose` yourself | **Yes** | This is exactly what the stack is. Any provider's small general-purpose VM of that size works. Whether a given free tier offers that much memory is unverified. |
| Free "web service" tiers of platform hosts (typically one container, well under 1 GB RAM, sleeping when idle) | **No** | The API alone needs about 2.1 GB and there are three services plus a volume. Unverified per provider, but the memory gap is large. |
| Static hosting (pages-style hosting) | **Frontend only** | The three pages and all findings would load; the three desks would show their "service didn't answer" state. This is a legitimate, zero-risk public version: it shows every finding and no per-sample prediction. Not tested as a deployment. |
| Model-demo hosts built for a single app (notebook or space style) | **Not as built** | They usually run one process; Neo4j would have to be dropped or moved to a hosted graph database. That is new work, not a deployment of this stack. Unverified. |
| Managed Neo4j free tier plus a separate API host | **Possible, more work** | 203,769 nodes and 234,355 relationships; whether that fits a given free tier's limits is unverified. The API still needs about 2.1 GB somewhere. |
| A tunnel from your own machine (the stack stays local, a tunnel gives a temporary public URL) | **Yes, simplest live link** | Nothing is deployed; the link dies when you close the tunnel. Use the deployment compose file so only the frontend port is tunnelled. Tunnel products and their terms are unverified. |

Recommendation: Option A for the video. If you want a public page that stays up, publish the **static frontend
only**. If you want a live link for a day, use a tunnel or a small VM with the deployment profile below.

## 6. Pre-deploy checklist (only for a deployment reachable from the internet)

1. Decide rule 2 of section 1 first: a public deployment lets any visitor score any Elliptic transaction. If you are
   not comfortable with that under CC BY-NC-ND, do not deploy the API; publish the static frontend only.
2. Copy `.env.example` to `.env` on the host. Set `NEO4J_PASSWORD` to a new long random value (not
   `trustmesh-demo`). Set `TRUSTMESH_CORS_ORIGINS` to the exact public origin, for example `https://demo.example.org`.
   Never commit `.env` (it is git-ignored).
3. Start with both compose files: `docker compose -f docker-compose.yml -f docker-compose.deploy.yml up -d --build`.
   This publishes **only the frontend** (port `PUBLIC_PORT`, default 80). Neo4j's ports 7474 and 7687 and the API's
   port 8000 are not published. Verified locally: with this profile the three ports refuse connections from the
   host, the site answers, and the API works through `/api`.
4. If the Neo4j volume was first created with the old password, the new one will not apply (Neo4j sets the password
   only when the database is created). On a fresh host this does not arise. Otherwise remove the volume
   (`docker volume rm trustmesh_code_neo4j-data`) and let the loader refill it. **(not tested with a changed password)**
5. Put HTTPS in front (the host's proxy or the platform's). The compose file serves plain HTTP. **(not tested)**
   Microphone recording in the browser only works on HTTPS or on localhost.
6. Confirm from another network: `https://your-host/` loads; `https://your-host:7474` and `:7687` and `:8000` do not
   answer; `https://your-host/api/docs` returns 404.
7. The API allows 30 scoring requests per minute per visitor address and 10 MB / 60 seconds of audio per upload.
   This is a demo-grade limit kept in memory, not protection against a determined attacker. Do not leave the
   deployment up unattended for long.
8. Copy `data/` (Elliptic, the encoder) and `artifacts/` (weights) to the host by a private channel. They must not
   go through a public repository.
9. No secrets in the repository: check `git status` shows no `.env`.

## 7. Teardown checklist

1. `docker compose -f docker-compose.yml -f docker-compose.deploy.yml down --volumes` on the host (removes the
   containers, the network, and the Neo4j and cache volumes).
2. Delete the copied `data/` and `artifacts/` folders and the `.env` file from the host.
3. Delete the images (`docker image rm trustmesh_code-api trustmesh_code-frontend neo4j:5-community`) or delete the
   whole machine.
4. Delete the VM or service, its disk, any snapshot or backup, any static IP, and any firewall rule you opened.
5. Remove the DNS record or domain mapping, and any TLS certificate issued for it.
6. Close the tunnel if you used one, and revoke its token.
7. Change nothing locally: the local stack still uses its own default password and is bound to 127.0.0.1.
8. Check from another network that the URL no longer answers.

## 8. After teardown: showing it live

`docker compose up -d` in the repository brings the same site back on http://localhost:8080 in about a minute, for
an interview or a screen share. Use section 3's warm-up first.

## 9. Steps that were not tested

- HTTPS in front of the frontend, and microphone recording on a non-localhost origin.
- A deployment on any real host or tunnel (the deployment profile was verified on this machine only).
- Changing the Neo4j password on an existing volume.
- Static-only hosting of the frontend.
- A start from a completely empty machine (no `data/`, no images): see the final verification in the status report.
- The demo script was not timed with a person speaking.
