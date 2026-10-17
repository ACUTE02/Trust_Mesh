# 🕸️ TrustMesh

**A fraud-triage workbench that combines a transaction-graph model and a voice-deepfake model, and is explicit about how far each result can be trusted.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-wav2vec2%20head-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Graph%20engine-189AB4)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20store-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![React](https://img.shields.io/badge/React-Three.js%20frontend-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-195%20Python%20%2B%2097%20frontend-brightgreen)](#-testing)

> This project reports what was measured, on which data, with which caveats. Where something is a mechanism and not a validated result, it says so.

<!-- Add a demo video or GIF here once recorded. Record with TRUSTMESH_RECORDING_MODE=1. -->

---

## 📖 Overview

TrustMesh is a portfolio project built around one idea: a fraud analyst looks at a *case*, and a case can have more than one kind of evidence. The system has three engines and a fusion step.

| Engine | Status | What it does |
|---|---|---|
| **Graph** | Built | Scores Bitcoin transactions as likely illicit or licit, from node features plus neighbourhood features. |
| **Voice** | Built | Scores an uploaded audio clip for the risk that it is synthetic or cloned speech. |
| **Visual** | **Not built** | Designed only. Paused, no results. The site says "Designed, not built. No results yet." |
| **Fusion** | Mechanism only | Combines the available engine scores into one case view. It uses placeholder 0.5 / 0.5 weights and has **no validated accuracy**. |

A browser frontend (six 3D "worlds" and a case desk) sits on top of a FastAPI service and a Neo4j graph store, all started with Docker Compose.

## ✨ Features

- 🕸️ **Graph engine**: XGBoost on the Elliptic Bitcoin dataset using the original 165 features plus neighbourhood aggregates, evaluated with a time-based split.
- 🎙️ **Voice engine**: a frozen `wav2vec2-xls-r-300m` front end with mean pooling and a small trained head (133,402 parameters).
- 🛡️ **Audio guard**: a heuristic that refuses non-speech (tone, noise, silence) and warns on band-limited audio. It is not a validated detector.
- 🔀 **Optional codec-robust voice model (F2)**: opt-in only, trained with codec-augmented copies. The default model stays Model D.
- 🗂️ **Case desk**: groups engine outputs for one case and shows each score with its caveat.
- 🎬 **Recording mode**: `TRUSTMESH_RECORDING_MODE=1` hides real transaction numbers, labels and scores on the graph desks so a video can show the flow without publishing them.
- 🧪 **Tests**: 195 Python tests and 97 frontend tests, plus a serving verification script.

## 📊 Measured Results

All numbers below are from the project's own experiments. Read the caveat column.

| What | Result | Caveat |
|---|---|---|
| Graph engine, PR-AUC (XGBoost, 3 seeds) | **0.7482 ± 0.0030** | Elliptic, time-based split. The served model reproduces **0.7474**. The two numbers are different things and should not be mixed. |
| Voice engine, EER (Model D, 3 seeds) | **0.0166 ± 0.0019** | ASVspoof 2019 LA. The served model is seed 42 only, EER about 0.0194. |
| Voice risk thresholds | high 0.6578, medium 0.0239 | **In-sample**: chosen on the same data they are reported on. |
| Codec robustness (F2 vs control) | Genuine clips flagged: 0.5982 → 0.0596 (AAC 32) | Synthetic ffmpeg re-encodes of ASVspoof clips. The control is weaker than D. Ranking improved on only 1 of 4 held-out codecs. F2 is still weak on echo, AAC 16 and Opus 12. |
| Fusion | none | Placeholder weights. No accuracy claimed. |

**Known limits**

- Voice results come from a research dataset, not real phone calls or real microphones. A real browser-microphone recording of a genuine speaker was scored as high risk during testing, and the cause is not settled.
- The audio guard does not catch every re-encode (for example AAC 32 and Opus 24 pass through it).
- Test sets for the real-audio checks were very small.
- Frontend smoothness on a real GPU has not been verified. Two performance budgets were missed (see `docs/design_decisions.md`).

## 🏗️ Architecture

```
 Browser (React + Three.js)
        │  /api (nginx proxy)
        ▼
 FastAPI serving layer ──► Graph service (XGBoost)  ──► Neo4j (case/graph store)
        │
        ├──────────────► Voice service (wav2vec2 + head, optional F2)
        │                     └─ audio guard (heuristic)
        └──────────────► Fusion (placeholder weights)
```

## 📁 Project Structure

```
TrustMesh_CODE/
├── engines/
│   ├── graph/        # features, XGBoost baseline, GNN experiments, evaluation
│   ├── voice/        # SSL front end, head, codec augmentation, evaluation
│   └── visual/       # designed only, no results
├── serving/          # FastAPI app, graph/voice services, audio guard, fusion, Neo4j store
├── frontend/         # React + Three.js site, nginx image
├── notebooks/        # Kaggle notebooks (all training ran on Kaggle)
├── scripts/          # verification, calibration, self-test and check scripts
├── tests/            # Python tests
├── docs/             # design decisions, leaderboards, data exploration
├── artifacts/        # small result files and thresholds only
├── docker-compose.yml
└── start.bat / stop.bat
```

## 🛠️ Tech Stack

`Python 3.11` · `PyTorch` · `XGBoost` · `PyTorch Geometric` · `Hugging Face Transformers` · `FastAPI` · `Neo4j` · `React` · `Three.js (react-three-fiber)` · `Vite` · `Docker Compose` · `pytest` · `Vitest`

## ⚠️ Model weights are not included

Trained weights, per-sample scores and dataset-derived outputs are **not** in this repository, because the Elliptic dataset's licence terms for sharing derived models have not been settled. A fresh clone therefore builds and runs the site, but the scoring desks will report that the scoring service did not answer until you train the models yourself (notebooks in `notebooks/`, run on Kaggle) and place the weight files under `artifacts/`.

## 🚀 Local Deployment

Requires Docker Desktop.

```bash
git clone https://github.com/ACUTE02/Trust_Mesh.git
cd Trust_Mesh
cp .env.example .env        # then edit it; set your own NEO4J_PASSWORD
docker compose up -d
```

- Site: `http://localhost:8080`
- API: `http://localhost:8000` (`/health` shows which models are loaded)
- Neo4j browser: `http://localhost:7474`

On Windows, `start.bat` starts Docker if needed, brings the stack up and opens the site; `stop.bat` stops it. All ports are bound to `127.0.0.1`.

The frontend build runs `tsc --noEmit && vitest run && vite build`, so a failing frontend test fails the image build.

## 🧪 Testing

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt     # CPU-only
.venv/bin/python -m pytest tests/
cd frontend && npm ci && npm test
```

`scripts/verify_serving.py` checks a running stack end to end (it needs the weights).

## 🙏 Data and Credits

- **Elliptic Bitcoin Dataset** (graph engine). Elliptic, licensed **CC BY-NC-ND 4.0**. This project uses it for non-commercial research and portfolio work and does not redistribute the data.
  Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., Leiserson, C. E. *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* KDD Anomaly Detection in Finance Workshop, 2019. arXiv:1908.02591.
- **ASVspoof 2019 (Logical Access)** (voice engine). Not redistributed here.
  Wang, X. et al. *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.* Computer Speech & Language, 2020.
- **wav2vec 2.0 XLS-R 300M** (frozen voice front end). Babu, A. et al. *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.* 2021.

## ⚖️ Responsible Use

TrustMesh is a research and portfolio demonstration. It is not a production fraud system and must not be used to make decisions about real people or real money.

## 👤 Author

**Ayushmaan Gupta**, B.Tech CSE (AI & ML), VIT Bhopal.

<div align="center">

**🕸️ TrustMesh**

*Fraud triage that says how far to trust it.*

</div>
