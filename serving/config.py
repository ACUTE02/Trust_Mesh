"""Paths and service settings, overridable by environment variable (docker-compose sets them)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("TRUSTMESH_DATA_DIR", ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("TRUSTMESH_ARTIFACTS_DIR", ROOT / "artifacts"))

ELLIPTIC_ROOT = DATA_DIR / "elliptic"
SERVING_CACHE = DATA_DIR / "serving"  # derived and regenerable: Tier-2 features, all-node scores
DEMO_CLIPS_DIR = Path(os.environ.get("TRUSTMESH_CLIPS_DIR", DATA_DIR / "demo_clips"))
CASES_DB = Path(os.environ.get("TRUSTMESH_CASES_DB", DATA_DIR / "cases.sqlite"))

XGB_WEIGHTS = ARTIFACTS_DIR / "graph" / "xgboost_tier1_tier2.json"
GAT_WEIGHTS = ARTIFACTS_DIR / "graph" / "gat_weights.pt"
GRAPH_THRESHOLDS = ARTIFACTS_DIR / "serving" / "graph_thresholds.json"
VOICE_THRESHOLDS = ARTIFACTS_DIR / "serving" / "voice_thresholds.json"

# A local copy under data/models/ wins over the Hub id (faster start, works offline, shared with the
# container through the ./data mount).
_LOCAL_SSL = DATA_DIR / "models" / "wav2vec2-xls-r-300m"
SSL_MODEL = os.environ.get(
    "TRUSTMESH_SSL_MODEL",
    str(_LOCAL_SSL) if (_LOCAL_SSL / "config.json").exists() else "facebook/wav2vec2-xls-r-300m")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
# Local demo credential for a database only exposed on localhost; not a secret.
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "trustmesh-demo")

# Fusion defaults (placeholder, NOT fit to labeled joint data -- see docs/design_decisions.md).
FUSION_W_GRAPH = float(os.environ.get("TRUSTMESH_W_GRAPH", 0.5))
FUSION_W_VOICE = float(os.environ.get("TRUSTMESH_W_VOICE", 0.5))

# --- Deployment hardening (all overridable; see .env.example) -----------------------------------
# Browser origins allowed to call the API directly. The Docker frontend proxies /api itself, so a
# deployment can set this to its one public origin; the default covers local development only.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "TRUSTMESH_CORS_ORIGINS", "http://localhost:8080,http://localhost:5173").split(",") if o.strip()]
MAX_UPLOAD_BYTES = int(os.environ.get("TRUSTMESH_MAX_UPLOAD_MB", 10)) * 1024 * 1024
MAX_AUDIO_SECONDS = float(os.environ.get("TRUSTMESH_MAX_AUDIO_SECONDS", 60))
# Scoring requests (POST) allowed per client address per minute; 0 disables the limit.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("TRUSTMESH_RATE_LIMIT_PER_MINUTE", 30))
# Set to 1 only behind a reverse proxy you control (the bundled nginx): the client address is then
# taken from X-Forwarded-For instead of the socket.
TRUST_PROXY = os.environ.get("TRUSTMESH_TRUST_PROXY", "0") == "1"
# Recording mode: the site masks real transaction numbers, labels and scores on the graph desks, so a video
# can show the flow without publishing per-transaction results. The API itself is unchanged. Default off.
RECORDING_MODE = os.environ.get("TRUSTMESH_RECORDING_MODE", "0") == "1"
