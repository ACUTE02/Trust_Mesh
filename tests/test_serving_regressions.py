"""Regression tests for serving bugs found in the 2026-10-14 final audit. Each test reproduces the original
failure; see interview_prep/final_audit/TrustMesh_Bug_Fix_Log.md.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_serving_regressions.py
"""
import asyncio
import io
import time

import httpx
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import serving.api as api_module
from serving import config
from serving.api import create_app
from serving.cases import CaseStore
from serving.fusion import fuse
from serving.graph_service import REPORTED_SEED42_PR_AUC, GraphService

from test_serving import StubGraph, StubNeo4j, tiny_voice_service, wav_bytes


@pytest.fixture(autouse=True)
def hermetic_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_THRESHOLDS", tmp_path / "no_such_voice_thresholds.json")


class SlowVoice:
    """Voice stub whose scoring blocks its thread, like real CPU inference."""
    ready = True

    def info(self):
        return {}

    def score_bytes(self, blob, model=None):
        time.sleep(1.0)
        return {"score": 0.2, "verdict": "low", "model": "D", "fallback_used": False}


class FixedVoice(SlowVoice):
    def score_bytes(self, blob, model=None):
        return {"score": 0.2, "verdict": "low", "model": "D", "fallback_used": False}


# BF-01: CPU-bound voice inference inside `async def` endpoints blocked the event loop.
def test_health_answers_while_a_voice_clip_is_being_scored():
    app = create_app(graph=StubGraph(), voice=SlowVoice(), neo4j=StubNeo4j(), cases=CaseStore(":memory:"),
                     load_on_startup=False)

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            t0 = time.perf_counter()

            async def health():
                await asyncio.sleep(0.1)
                r = await c.get("/health")
                return r.status_code, time.perf_counter() - t0

            async def score():
                r = await c.post("/voice/score", params={"record": "false"},
                                 files={"file": ("a.wav", wav_bytes(), "audio/wav")})
                return r.status_code

            return await asyncio.gather(health(), score())

    (health_status, health_elapsed), score_status = asyncio.run(run())
    assert health_status == 200 and score_status == 200
    assert health_elapsed < 0.8, f"/health waited {health_elapsed:.2f}s for a 1.0s scoring call"


# BF-02: an infinite fusion weight produced a NaN combined score, wrote a corrupt case row, then HTTP 500.
@pytest.mark.parametrize("weights", [(float("inf"), 0.5), (0.5, float("inf")), (float("nan"), 0.5)])
def test_fuse_rejects_non_finite_weights(weights):
    with pytest.raises(ValueError):
        fuse(0.5, 0.5, *weights)


def test_case_score_with_infinite_weight_is_refused_and_writes_nothing():
    cases = CaseStore(":memory:")
    app = create_app(graph=StubGraph(), voice=FixedVoice(), neo4j=StubNeo4j(), cases=cases, load_on_startup=False)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/case/score", params={"w_graph": "inf"}, data={"node_id": "7"},
                   files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 422
    assert cases.count() == 0


# BF-03: NaN / inf samples in a float WAV were scored instead of refused.
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_voice_refuses_non_finite_samples(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(config, "DEMO_CLIPS_DIR", tmp_path)
    sr = 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
    x[100:200] = bad
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    app = create_app(graph=StubGraph(), voice=tiny_voice_service(tmp_path), neo4j=StubNeo4j(),
                     cases=CaseStore(":memory:"), load_on_startup=False)
    with TestClient(app) as c:
        r = c.post("/voice/score", files={"file": ("bad.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 422
    assert "invalid sample values" in r.json()["detail"]


def test_valid_float_wav_is_still_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_CLIPS_DIR", tmp_path)
    sr = 16000
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
    app = create_app(graph=StubGraph(), voice=tiny_voice_service(tmp_path), neo4j=StubNeo4j(),
                     cases=CaseStore(":memory:"), load_on_startup=False)
    with TestClient(app) as c:
        r = c.post("/voice/score", params={"record": "false"}, files={"file": ("ok.wav", buf.getvalue(), "audio/wav")})
    assert r.status_code == 200 and 0.0 <= r.json()["score"] <= 1.0


# BF-04: the graph startup check reported the PR-AUC but never refused mismatching weights.
def test_graph_verification_mismatch_refuses_to_serve():
    svc = GraphService()
    svc.verification = {"test_pr_auc": REPORTED_SEED42_PR_AUC - 0.01}
    with pytest.raises(RuntimeError, match="do not reproduce"):
        svc._check_verification()


def test_graph_verification_within_tolerance_passes():
    svc = GraphService()
    svc.verification = {"test_pr_auc": REPORTED_SEED42_PR_AUC + 1e-6}
    svc._check_verification()


def test_graph_verification_nan_is_refused():
    svc = GraphService()
    svc.verification = {"test_pr_auc": float("nan")}
    with pytest.raises(RuntimeError):
        svc._check_verification()


# BF-05: the in-memory rate-limit store kept one entry per client address forever.
def test_rate_limit_store_evicts_idle_clients(monkeypatch):
    monkeypatch.setattr(api_module, "RATE_LIMIT_MAX_CLIENTS", 3)
    monkeypatch.setattr(config, "TRUST_PROXY", True)
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 30)
    clock = [1000.0]
    monkeypatch.setattr(api_module.time, "monotonic", lambda: clock[0])
    app = create_app(graph=StubGraph(), voice=FixedVoice(), neo4j=StubNeo4j(), cases=CaseStore(":memory:"),
                     load_on_startup=False)
    with TestClient(app) as c:
        for i in range(5):
            c.post("/voice/score", params={"record": "false"}, headers={"x-forwarded-for": f"10.0.0.{i}"},
                   files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert len(app.state.rate_limit_hits) == 5   # all active within the minute: nothing evicted
        clock[0] += 120                              # two minutes later, one new client arrives
        c.post("/voice/score", params={"record": "false"}, headers={"x-forwarded-for": "10.0.0.99"},
               files={"file": ("a.wav", wav_bytes(), "audio/wav")})
        assert set(app.state.rate_limit_hits) == {"10.0.0.99"}
