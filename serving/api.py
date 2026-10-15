"""FastAPI app: graph scoring, voice scoring, graph-database queries and the fused case endpoint.

Run:  uvicorn serving.api:app --port 8000
Services are injectable through `create_app` so the endpoints can be tested without loading models.
"""
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import config
from .cases import CaseStore
from .fusion import fuse

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
RATE_LIMIT_MAX_CLIENTS = 1000  # above this many tracked addresses, idle ones are evicted


def create_app(graph=None, voice=None, neo4j=None, cases=None, load_on_startup=True):
    state = {"graph": graph, "voice": voice, "neo4j": neo4j, "cases": cases}

    @asynccontextmanager
    async def lifespan(app):
        if load_on_startup:
            from .graph_service import GraphService
            from .neo4j_store import Neo4jStore
            from .voice_service import VoiceService
            state["cases"] = state["cases"] or CaseStore(config.CASES_DB)
            from .preflight import report
            text, blocking = report()
            if text:
                print(text, flush=True)
            if blocking and (state["graph"] is None or state["voice"] is None):
                raise SystemExit("TrustMesh API not started: required files are missing (listed above).")
            if state["graph"] is None:
                state["graph"] = GraphService()
                state["graph"].load()
            if state["voice"] is None:
                state["voice"] = VoiceService()
                try:
                    state["voice"].load()
                except Exception as exc:  # the graph half of the API stays usable
                    state["voice"].error = f"{type(exc).__name__}: {exc}"
            if state["neo4j"] is None:
                state["neo4j"] = Neo4jStore()
        elif state["cases"] is None:
            state["cases"] = CaseStore(":memory:")
        yield
        if load_on_startup and state["neo4j"] is not None:
            state["neo4j"].close()

    app = FastAPI(title="TrustMesh", version="1.0", lifespan=lifespan,
                  description="Graph risk (XGBoost Tier-1+2) + voice spoof risk (frozen XLS-R head), fused.")

    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    # Demo-grade rate limit on the scoring POSTs (the expensive calls): a sliding one-minute window per
    # client address, kept in memory. It protects one small box from a runaway loop, nothing more.
    hits = defaultdict(deque)
    app.state.rate_limit_hits = hits  # exposed for tests

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        limit = config.RATE_LIMIT_PER_MINUTE
        if limit and request.method == "POST":
            client = request.client.host if request.client else "unknown"
            if config.TRUST_PROXY and request.headers.get("x-forwarded-for"):
                client = request.headers["x-forwarded-for"].split(",")[0].strip()
            now = time.monotonic()
            if len(hits) > RATE_LIMIT_MAX_CLIENTS:  # forget clients idle for over a minute
                for key in [k for k, w in hits.items() if not w or now - w[-1] > 60]:
                    del hits[key]
            window = hits[client]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= limit:
                return JSONResponse({"detail": f"Too many scoring requests. The limit is {limit} per minute; "
                                               "wait a minute and try again."}, status_code=429,
                                    headers={"Retry-After": "60"})
            window.append(now)
        return await call_next(request)

    # -- helpers ---------------------------------------------------------------------------------
    def graph_svc():
        if state["graph"] is None or not state["graph"].ready:
            raise HTTPException(503, "graph engine not loaded")
        return state["graph"]

    def voice_svc():
        v = state["voice"]
        if v is None or not v.ready:
            raise HTTPException(503, f"voice engine not loaded: {getattr(v, 'error', None) or 'not initialised'}")
        return v

    def db():
        if state["neo4j"] is None:
            raise HTTPException(503, "Neo4j is not configured")
        return state["neo4j"]

    def run_query(fn, *args, **kwargs):
        from neo4j.exceptions import DriverError, Neo4jError
        try:
            return fn(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except (DriverError, Neo4jError, OSError) as exc:
            raise HTTPException(503, f"Neo4j query failed (is the database up and loaded?): {exc}")

    def known_node(node_id):
        try:
            graph_svc().check_node(node_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc.args[0]))

    async def read_audio(file, clip):
        """Audio from an upload or from a named demo clip. Returns (bytes, display name)."""
        if (file is None) == (clip is None):
            raise HTTPException(422, "send exactly one of `file` (upload) or `clip` (demo clip name)")
        if file is not None:
            ctype = (file.content_type or "").lower()
            suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
            if not (ctype.startswith("audio/") or (ctype in ("", "application/octet-stream") and suffix in AUDIO_SUFFIXES)):
                raise HTTPException(415, f"That is not an audio file (content type {ctype or 'missing'}). "
                                         "Send WAV, FLAC, OGG or MP3.")
            blob = await file.read(config.MAX_UPLOAD_BYTES + 1)
            if len(blob) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"Audio file is larger than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            return blob, file.filename or "upload"
        try:
            return voice_svc().clip_path(clip).read_bytes(), clip
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    def score_voice(blob, model):
        try:
            return voice_svc().score_bytes(blob, model)
        except KeyError as exc:
            raise HTTPException(422, str(exc.args[0]))
        except (ValueError, RuntimeError) as exc:  # undecodable / too-short audio
            raise HTTPException(422, f"Could not score this audio: {exc}.")

    # -- meta ------------------------------------------------------------------------------------
    @app.get("/health")
    def health():
        g, v, n = state["graph"], state["voice"], state["neo4j"]
        neo = {"connected": False}
        if n is not None:
            try:
                neo = {"connected": True, **n.stats()}
            except Exception as exc:
                neo = {"connected": False, "error": f"{type(exc).__name__}"}
        return {
            "graph": {"ready": bool(g and g.ready), "model": "xgboost_tier1_tier2",
                      "test_pr_auc_check": g.verification if g and g.ready else None},
            "voice": {"ready": bool(v and v.ready), **(v.info() if v and v.ready else {"error": getattr(v, "error", None)})},
            "neo4j": neo, "cases": state["cases"].count(),
            "visual": {"connected": False, "note": "Designed, not built. No results yet."},
            "recording_mode": config.RECORDING_MODE,
            "limits": {"max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
                       "max_audio_seconds": config.MAX_AUDIO_SECONDS,
                       "scoring_requests_per_minute": config.RATE_LIMIT_PER_MINUTE},
        }

    # -- graph -----------------------------------------------------------------------------------
    @app.get("/graph/score/{node_id}")
    def graph_score(node_id: int, record: bool = Query(True, description="false = do not write a row to the cases table")):
        known_node(node_id)
        result = graph_svc().score(node_id)
        if record:
            state["cases"].record_graph(node_id, result["score"], result["verdict"])
        return result

    @app.get("/graph/explain/{node_id}")
    def graph_explain(node_id: int):
        known_node(node_id)
        return graph_svc().explain(node_id)

    @app.get("/graph/query/cypher")
    def q_cypher(hops: int = Query(2, ge=1, le=4)):
        """The exact Cypher text the query endpoints run (for the site's technical view). No database call."""
        from .neo4j_store import cypher_text
        return cypher_text(neighbor_hops=hops)

    @app.get("/graph/query/neighbors/{node_id}")
    def q_neighbors(node_id: int, hops: int = Query(2, ge=1, le=4), limit: int = Query(100, ge=1, le=500)):
        known_node(node_id)
        rows = run_query(db().neighbors_within, node_id, hops, limit)
        return {"node_id": node_id, "hops": hops, "count": len(rows), "neighbors": rows}

    @app.get("/graph/query/illicit/{node_id}")
    def q_illicit(node_id: int, hops: int = Query(2, ge=1, le=4)):
        known_node(node_id)
        rows = run_query(db().illicit_within, node_id, hops)
        return {"node_id": node_id, "hops": hops, "known_illicit_within_hops": rows}

    @app.get("/graph/query/path")
    def q_path(source: int, target: int, max_hops: int = Query(8, ge=1, le=12)):
        known_node(source)
        known_node(target)
        path = run_query(db().shortest_path, source, target, max_hops)
        return {"source": source, "target": target, "connected": path is not None, "path": path}

    @app.get("/graph/query/flagged-neighbors/{node_id}")
    def q_flagged(node_id: int):
        known_node(node_id)
        g = graph_svc()
        return run_query(db().flagged_neighbor_counts, node_id, g.thresholds["medium"]) | {
            "flag_threshold": g.thresholds["medium"]}

    # -- voice -----------------------------------------------------------------------------------
    @app.get("/voice/clips")
    def voice_clips():
        return {"clips": voice_svc().list_clips()}

    @app.post("/voice/score")
    async def voice_score(file: UploadFile | None = File(None), clip: str | None = Form(None),
                          model: str | None = Query(None, description="'D' (default) or 'C' (fallback)"),
                          record: bool = Query(True, description="false = do not write a row to the cases table")):
        blob, name = await read_audio(file, clip)
        # CPU-bound inference runs in a worker thread so it does not block the event loop (other requests,
        # including /health, would otherwise wait for every clip to finish).
        result = await run_in_threadpool(score_voice, blob, model)
        if record:
            state["cases"].record_voice(name, result["score"], result["verdict"], result["model"])
        return {"clip": name, **result}

    # -- fused case ------------------------------------------------------------------------------
    @app.post("/case/score")
    async def case_score(node_id: int = Form(...), file: UploadFile | None = File(None),
                         clip: str | None = Form(None), model: str | None = Query(None),
                         w_graph: float = Query(config.FUSION_W_GRAPH, ge=0),
                         w_voice: float = Query(config.FUSION_W_VOICE, ge=0),
                         record: bool = Query(True, description="false = do not write a row to the cases table")):
        known_node(node_id)
        blob, name = await read_audio(file, clip)
        graph = graph_svc().score(node_id)
        voice = await run_in_threadpool(score_voice, blob, model)
        try:
            fused = fuse(graph["score"], voice["score"], w_graph, w_voice)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        case_id = state["cases"].record_combined(node_id, name, graph, voice, voice["model"], fused) if record else None
        # `visual` is the reserved third-signal slot: no Visual Engine is connected, so it is always null.
        return {"case_id": case_id, "graph": graph, "voice": {"clip": name, **voice}, "visual": None,
                "fusion": fused}

    # -- cases -----------------------------------------------------------------------------------
    @app.get("/cases/flagged")
    def flagged(limit: int = Query(50, ge=1, le=500)):
        return {"cases": state["cases"].flagged(limit)}

    @app.get("/cases")
    def all_cases(limit: int = Query(200, ge=1, le=1000)):
        return {"cases": state["cases"].all_cases(limit)}

    return app


app = create_app()
