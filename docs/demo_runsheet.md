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

## 4. The demo script (about two minutes)

Say only what is on the page; every sentence below is backed by the project log.

1. **Overview, the instrument (20 s).** "This is one graph neural network scored two ways." Leave it on *Random
   split*: 0.7730. Click **Time split**: 0.5115. "Same model, same settings. A random split trains and tests on the
   same period; a time split tests on later transactions. The random split overstated it by 0.26."
2. **The two cards under "What each check scored" (15 s).** "Two engines are measured. On the graph, plain XGBoost with three cheap graph statistics
   scores 0.7482; the best graph neural network, after more than 175 tuning trials, scores 0.5913. The voice model
   scores an equal error rate of 0.0166 on my own split of ASVspoof, which is not the official protocol. The third
   engine is designed and not built: it has no results, so it gets a dashed row."
3. **Click a yellow tag (10 s).** "Every number carries the run it came from and the file."
4. **Voice Engine page, the instrument (25 s).** "The first voice model scored 0.0097. It was cheating: it listened to the silence."
   Click **Silence removed**: 0.2380. "Bonafide clips in this dataset start with more silence. So I trim silence
   before training and before scoring." Scroll to the ladder: "these hatched rows are the shortcut, kept on the page
   so nobody mistakes them for results."
5. **Generalisation (20 s).** "With four attack systems held out of training, the plain CNN's error rose by 0.24. The
   frozen self-supervised models moved by about 0.001, which is not significant. On a different dataset the CNN
   collapses and these hold at 0.08 to 0.10, on a small sample of 200 clips."
6. **Voice desk (20 s).** Choose your own clip, **Score this clip**. "My recording, converted in the browser, scored
   by Model D. The shaded part is silence the scorer trimmed. The thresholds were set on the test scores, so this is
   a demonstration of the mechanism, not a calibrated alarm."
7. **Graph Engine page (15 s).** Show the leaderboard and the three checks. Mention the limits list at the bottom.
   (Graph desk: only if you decided rule 2 allows it. If so: enter an example, **Score this transaction**, and say
   "the score is XGBoost's; the neighbourhood is from Neo4j and decided nothing".)
8. **Close (10 s).** Overview, "What is tested, and what is not": "Two checks are measured. The combined warning is a
   demonstration with no measured accuracy, because no dataset has the same case in both. The visual check is
   designed and not built." For a technical audience, press "Show the detail for engineers" first: the exact terms,
   the system diagram and the decisions table are there.

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
