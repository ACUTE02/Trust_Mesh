"""CPU tests for the serving layer (docs/PHASE_2_INFRA_AND_DEMO.md). Fusion arithmetic, the cases
table, voice pre-processing / polarity / fallback, and every API endpoint against stub services.
No reported numbers: real-model checks live in scripts/verify_serving.py.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_serving.py
"""
import io

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from serving import config
from serving.api import create_app
from serving.cases import CaseStore
from serving.fusion import COMBINED_HIGH, COMBINED_MEDIUM, fuse
from serving.graph_service import GraphService, compute_thresholds
from serving.voice_service import VoiceService, prepare_clip


@pytest.fixture(autouse=True)
def hermetic_thresholds(tmp_path, monkeypatch):
    """Tests must not pick up the real calibrated threshold files under artifacts/serving."""
    monkeypatch.setattr(config, "VOICE_THRESHOLDS", tmp_path / "no_such_voice_thresholds.json")


# --- fusion ---------------------------------------------------------------------------------------------
def test_fuse_is_the_weighted_average_and_normalises_weights():
    r = fuse(0.8, 0.2, w_graph=3, w_voice=1)
    assert r["weights"] == {"graph": 0.75, "voice": 0.25}
    assert r["combined_score"] == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)


def test_fuse_default_equal_weights_and_bands():
    assert fuse(0.2, 0.2)["combined_verdict"] == "low"
    assert fuse(COMBINED_MEDIUM, COMBINED_MEDIUM)["combined_verdict"] == "medium"
    assert fuse(COMBINED_HIGH, COMBINED_HIGH)["combined_verdict"] == "high"
    assert "placeholders" in fuse(0.5, 0.5)["caveat"]


@pytest.mark.parametrize("args", [(1.2, 0.5), (0.5, -0.1), (0.5, 0.5, 0, 0), (0.5, 0.5, -1, 1)])
def test_fuse_rejects_bad_input(args):
    with pytest.raises(ValueError):
        fuse(*args)


# --- cases table ----------------------------------------------------------------------------------------
def test_graph_only_case_has_null_voice_and_is_updated_not_duplicated():
    store = CaseStore(":memory:")
    store.record_graph(5, 0.9, "high")
    store.record_graph(5, 0.95, "high")
    rows = store.all_cases()
    assert len(rows) == 1
    assert rows[0]["graph_score"] == 0.95
    assert rows[0]["voice_score"] is None and rows[0]["combined_score"] is None


def test_voice_only_case_has_null_node_and_combined_is_a_new_row():
    store = CaseStore(":memory:")
    store.record_voice("a.wav", 0.7, "high", "C")
    fused = fuse(0.9, 0.7)
    store.record_combined(5, "a.wav", {"score": 0.9, "verdict": "high"}, {"score": 0.7, "verdict": "high"}, "C", fused)
    store.record_combined(5, "a.wav", {"score": 0.9, "verdict": "high"}, {"score": 0.7, "verdict": "high"}, "C", fused)
    rows = {r["case_id"]: r for r in store.all_cases()}
    assert len(rows) == 3
    voice_only = next(r for r in rows.values() if r["node_id"] is None)
    assert voice_only["graph_score"] is None and voice_only["combined_score"] is None


def test_flagged_view_sorted_by_risk_and_excludes_low():
    store = CaseStore(":memory:")
    store.record_graph(1, 0.10, "low")
    store.record_graph(2, 0.60, "high")
    store.record_graph(3, 0.20, "medium")
    store.record_voice("v.wav", 0.95, "high", "C")
    flagged = store.flagged()
    assert [r["risk_score"] for r in flagged] == [0.95, 0.60, 0.20]
    assert all(r["risk_verdict"] in ("medium", "high") for r in flagged)


# --- graph thresholds -----------------------------------------------------------------------------------
def test_threshold_rule_uses_only_validation_period_licit_nodes():
    n = 2000
    rng = np.random.default_rng(0)
    scores = rng.random(n).astype(np.float32)
    labels = np.zeros(n, dtype=int)
    time_step = np.full(n, 33)
    labels[:100] = 1          # illicit: must not influence thresholds
    scores[:100] = 5.0
    time_step[1500:] = 10     # train period: ignored
    t = compute_thresholds(scores, labels, time_step)
    val_licit = scores[(labels == 0) & (time_step == 33)]
    assert (val_licit >= t["high"]).mean() == pytest.approx(0.01, abs=0.005)
    assert (val_licit >= t["medium"]).mean() == pytest.approx(0.10, abs=0.01)
    assert t["medium"] < t["high"]


# --- voice ----------------------------------------------------------------------------------------------
def tiny_voice_service(tmp_path, with_d=False):
    """VoiceService wired to a tiny random encoder and random heads (shape/behaviour tests only)."""
    from engines.voice.model import SSLHead
    from engines.voice.ssl import SSLEncoder, tiny_wav2vec2_for_smoke

    folder = tmp_path / "voice"
    folder.mkdir()
    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    torch.manual_seed(0)
    torch.save(SSLHead(encoder.n_layers, encoder.feature_dim).state_dict(), folder / "phase2b_C_ssl_P1_s42.pt")
    if with_d:
        torch.save(SSLHead(encoder.n_layers, encoder.hidden_size).state_dict(),
                   folder / "phase2c_D_ssl_mean_only_P1_s42.pt")
    svc = VoiceService(artifacts_dir=tmp_path)
    svc.load(encoder=encoder)
    return svc


def wav_bytes(seconds=2.0, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def test_prepare_clip_is_fixed_length_int16_grid():
    clip, stats = prepare_clip(np.random.default_rng(0).normal(size=24000).astype(np.float32) * 0.1, 24000)
    assert clip.shape == (64000,) and clip.dtype == np.float32
    assert np.allclose(clip * 32767.0, np.round(clip * 32767.0), atol=1e-3)  # int16 round-trip applied


def test_prepare_clip_rejects_tiny_audio():
    with pytest.raises(ValueError):
        prepare_clip(np.zeros(100, dtype=np.float32), 16000)


def test_voice_service_falls_back_to_c_when_d_weights_missing(tmp_path):
    svc = tiny_voice_service(tmp_path)
    assert svc.default_model == "C" and svc.info()["fallback_active"]
    result = svc.score_bytes(wav_bytes())
    assert result["model"] == "C" and result["fallback_used"] is True and result["model_requested"] == "D"
    assert 0.0 <= result["score"] <= 1.0


def test_voice_service_uses_d_when_present_and_spoof_prob_is_complement_of_bonafide(tmp_path):
    svc = tiny_voice_service(tmp_path, with_d=True)
    assert svc.default_model == "D" and not svc.info()["fallback_active"]
    d = svc.score_bytes(wav_bytes())
    assert d["model"] == "D" and not d["fallback_used"]
    assert d["score"] == pytest.approx(1.0 / (1.0 + np.exp(d["bonafide_logit"])))
    assert svc.score_bytes(wav_bytes(), model="C")["model"] == "C"


def test_voice_service_unknown_model_and_uncalibrated_thresholds(tmp_path):
    svc = tiny_voice_service(tmp_path)
    with pytest.raises(KeyError):
        svc.score_bytes(wav_bytes(), model="Z")
    assert "uncalibrated" in svc.score_bytes(wav_bytes())["thresholds"]["source"]


def test_clip_path_refuses_traversal(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "ok.wav").write_bytes(wav_bytes())
    (tmp_path / "secret.wav").write_bytes(wav_bytes())
    monkeypatch.setattr(config, "DEMO_CLIPS_DIR", clips)
    svc = VoiceService(artifacts_dir=tmp_path)
    assert svc.clip_path("ok.wav").name == "ok.wav"
    for bad in ("../secret.wav", "missing.wav", ".."):
        with pytest.raises(FileNotFoundError):
            svc.clip_path(bad)


# --- API (stub graph, real tiny voice service, stub neo4j) ------------------------------------------------
class StubGraph:
    ready = True
    thresholds = {"medium": 0.1, "high": 0.6}
    verification = {"test_pr_auc": 0.0}
    verification = {"test_pr_auc": 0.0}

    def check_node(self, node_id):
        if not 0 <= node_id < 100:
            raise KeyError(f"node_id {node_id} out of range")

    def score(self, node_id):
        self.check_node(node_id)
        s = 0.9 if node_id == 7 else 0.05
        return {"node_id": node_id, "score": s, "verdict": "high" if s > 0.6 else "low", "model": "xgboost_tier1_tier2"}

    def explain(self, node_id):
        return {"node_id": node_id, "available": True, "label": "secondary model"}


class StubNeo4j:
    def neighbors_within(self, node_id, hops, limit):
        if hops > 4:
            raise ValueError("hops must be between 1 and 4")
        return [{"node_id": 9, "hops": 1}]

    def illicit_within(self, node_id, hops):
        return []

    def shortest_path(self, a, b, max_hops):
        return None

    def flagged_neighbor_counts(self, node_id, thr):
        return {"node_id": node_id, "neighbors": 3, "flagged_neighbors": 1, "known_illicit_neighbors": 0}

    def stats(self):
        return {"nodes": 100, "edges": 200}


@pytest.fixture
def client(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "demo.wav").write_bytes(wav_bytes())
    monkeypatch.setattr(config, "DEMO_CLIPS_DIR", clips)
    app = create_app(graph=StubGraph(), voice=tiny_voice_service(tmp_path), neo4j=StubNeo4j(),
                     cases=CaseStore(":memory:"), load_on_startup=False)
    with TestClient(app) as c:
        yield c


def test_graph_score_and_unknown_node(client):
    assert client.get("/graph/score/7").json()["verdict"] == "high"
    assert client.get("/graph/score/500").status_code == 404


def test_graph_score_creates_graph_only_case(client):
    client.get("/graph/score/7")
    cases = client.get("/cases").json()["cases"]
    assert len(cases) == 1 and cases[0]["voice_score"] is None and cases[0]["combined_score"] is None


def test_graph_explain_and_queries(client):
    assert client.get("/graph/explain/7").json()["available"] is True
    assert client.get("/graph/query/neighbors/7?hops=2").json()["count"] == 1
    assert client.get("/graph/query/neighbors/7?hops=9").status_code == 422
    assert client.get("/graph/query/illicit/7").status_code == 200
    assert client.get("/graph/query/path?source=1&target=2").json()["connected"] is False
    assert client.get("/graph/query/flagged-neighbors/7").json()["flagged_neighbors"] == 1


def test_voice_score_by_demo_clip_by_upload_and_validation(client):
    r = client.post("/voice/score", data={"clip": "demo.wav"})
    assert r.status_code == 200 and r.json()["model"] == "C" and r.json()["fallback_used"] is True
    up = client.post("/voice/score", files={"file": ("x.wav", wav_bytes(), "audio/wav")})
    assert up.status_code == 200
    assert client.post("/voice/score").status_code == 422                                  # neither
    assert client.post("/voice/score", data={"clip": "nope.wav"}).status_code == 404
    assert client.post("/voice/score", data={"clip": "demo.wav"}, params={"model": "Z"}).status_code == 422
    bad = client.post("/voice/score", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert bad.status_code == 422


def test_case_score_returns_fusion_with_both_components_and_logs_it(client):
    r = client.post("/case/score", data={"node_id": 7, "clip": "demo.wav"})
    assert r.status_code == 200
    body = r.json()
    assert body["graph"]["score"] == 0.9 and "score" in body["voice"]
    expected = 0.5 * 0.9 + 0.5 * body["voice"]["score"]
    assert body["fusion"]["combined_score"] == pytest.approx(expected)
    combined = [c for c in client.get("/cases").json()["cases"] if c["combined_score"] is not None]
    assert len(combined) == 1 and combined[0]["node_id"] == 7


def test_case_score_custom_weights_and_errors(client):
    r = client.post("/case/score", data={"node_id": 7, "clip": "demo.wav"}, params={"w_graph": 1, "w_voice": 0})
    assert r.json()["fusion"]["combined_score"] == pytest.approx(0.9)
    assert client.post("/case/score", data={"node_id": 7, "clip": "demo.wav"},
                       params={"w_graph": 0, "w_voice": 0}).status_code == 422
    assert client.post("/case/score", data={"node_id": 999, "clip": "demo.wav"}).status_code == 404


def test_flagged_cases_endpoint_sorted(client):
    client.get("/graph/score/7")
    client.get("/graph/score/8")  # low: excluded
    flagged = client.get("/cases/flagged").json()["cases"]
    assert [c["node_id"] for c in flagged] == [7]


def test_health_reports_voice_fallback(client):
    h = client.get("/health").json()
    assert h["voice"]["fallback_active"] is True and h["neo4j"]["connected"] is True


def test_graph_service_attention_label_says_it_is_not_the_score():
    """The explanation text must never present GAT as the scorer (Phase 1D finding)."""
    import inspect
    assert "NOT the source of the risk score" in inspect.getsource(GraphService.explain)


# --- WP3b: optional third (visual) signal slot ---------------------------------------------------------
def test_fuse_with_empty_visual_slot_is_identical_to_two_signal_fusion():
    two = fuse(0.8, 0.2, 3, 1)
    for empty in (fuse(0.8, 0.2, 3, 1, visual_score=None), fuse(0.8, 0.2, 3, 1, visual_score=None, w_visual=5)):
        assert empty == two
    assert two["signals_used"] == ["graph", "voice"] and "visual" not in two["weights"]
    assert two["combined_score"] == pytest.approx(0.75 * 0.8 + 0.25 * 0.2)


def test_fuse_with_dummy_visual_value_TEST_ONLY_no_visual_model_exists():
    """The 0.4 below is a made-up test input. There is no Visual Engine result anywhere in the project."""
    r = fuse(0.9, 0.3, 1, 1, visual_score=0.4, w_visual=2)
    assert r["weights"] == {"graph": 0.25, "voice": 0.25, "visual": 0.5}
    assert r["combined_score"] == pytest.approx(0.25 * 0.9 + 0.25 * 0.3 + 0.5 * 0.4)
    with pytest.raises(ValueError):
        fuse(0.9, 0.3, visual_score=1.4, w_visual=1)


def test_case_score_reports_visual_slot_as_not_connected(client):
    body = client.post("/case/score", data={"node_id": 7, "clip": "demo.wav"}).json()
    assert body["visual"] is None and body["fusion"]["signals_used"] == ["graph", "voice"]
    assert client.get("/health").json()["visual"]["connected"] is False


# --- WP3: upload limits, rate limit, CORS --------------------------------------------------------------
def test_upload_rejects_non_audio_content_type(client):
    r = client.post("/voice/score", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and "not an audio file" in r.json()["detail"]
    exe = client.post("/voice/score", files={"file": ("x.exe", wav_bytes(), "application/octet-stream")})
    assert exe.status_code == 415


def test_upload_rejects_oversized_and_overlong_audio(client, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    big = client.post("/voice/score", files={"file": ("x.wav", wav_bytes(), "audio/wav")})
    assert big.status_code == 413
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(config, "MAX_AUDIO_SECONDS", 1.0)
    long = client.post("/voice/score", files={"file": ("x.wav", wav_bytes(seconds=2.0), "audio/wav")})
    assert long.status_code == 422 and "limit is 1 s" in long.json()["detail"]


def test_rate_limit_on_scoring_posts_only(client, monkeypatch):
    monkeypatch.setattr(config, "RATE_LIMIT_PER_MINUTE", 3)
    codes = [client.post("/voice/score", data={"clip": "demo.wav"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert client.get("/graph/score/7").status_code == 200  # reads are not limited


def test_cors_allows_only_configured_origins(client):
    ok = client.get("/health", headers={"Origin": config.CORS_ORIGINS[0]})
    assert ok.headers.get("access-control-allow-origin") == config.CORS_ORIGINS[0]
    other = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_voice_response_reports_trimmed_silence(client):
    body = client.post("/voice/score", data={"clip": "demo.wav"}).json()
    assert {"lead_silence_s", "trail_silence_s", "scored_seconds"} <= set(body)


def test_cypher_text_endpoint_returns_the_queries_the_store_runs(client):
    from serving.neo4j_store import Q_FLAGGED, q_neighbors
    body = client.get("/graph/query/cypher?hops=2").json()
    assert set(body) == {"neighbors", "illicit", "path", "flagged_neighbors"}
    assert body["neighbors"].split() == q_neighbors(2).split()
    assert body["flagged_neighbors"].split() == Q_FLAGGED.split()
    assert "SENDS_TO*1..2" in body["neighbors"] and "SENDS_TO*1..3" in body["illicit"]


# --- Self-test fixes (2026-10-08): silence, very short audio, record=false, recording mode -----------------
def silent_wav(seconds=2.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def test_silent_recording_is_refused_not_scored(client):
    r = client.post("/voice/score", files={"file": ("s.wav", silent_wav(), "audio/wav")})
    assert r.status_code == 422 and "silent" in r.json()["detail"]


def test_empty_and_unreadable_files_get_a_plain_message(client):
    empty = client.post("/voice/score", files={"file": ("e.wav", b"", "audio/wav")})
    assert empty.status_code == 422 and "the file is empty" in empty.json()["detail"]
    junk = client.post("/voice/score", files={"file": ("j.wav", b"not audio at all", "audio/wav")})
    assert junk.status_code == 422 and "BytesIO" not in junk.json()["detail"]


def test_very_short_audio_is_scored_but_flagged_low_confidence(client):
    short = client.post("/voice/score", files={"file": ("x.wav", wav_bytes(seconds=0.3), "audio/wav")}).json()
    assert short["low_confidence"] is True and "too little for a reliable result" in short["warnings"][0]
    normal = client.post("/voice/score", files={"file": ("x.wav", wav_bytes(seconds=3.0), "audio/wav")}).json()
    assert normal["low_confidence"] is False and normal["warnings"] == []


def test_record_false_scores_without_writing_to_the_cases_table(client):
    assert client.get("/graph/score/7", params={"record": "false"}).status_code == 200
    assert client.post("/voice/score", data={"clip": "demo.wav"}, params={"record": "false"}).status_code == 200
    body = client.post("/case/score", data={"node_id": 7, "clip": "demo.wav"}, params={"record": "false"}).json()
    assert body["case_id"] is None and "combined_score" in body["fusion"]
    assert client.get("/cases").json()["cases"] == []
    client.get("/graph/score/7")
    assert len(client.get("/cases").json()["cases"]) == 1


def test_health_reports_recording_mode(client, monkeypatch):
    assert client.get("/health").json()["recording_mode"] is False
    monkeypatch.setattr(config, "RECORDING_MODE", True)
    assert client.get("/health").json()["recording_mode"] is True


def test_preflight_lists_missing_files_in_plain_words(tmp_path, monkeypatch):
    from serving import preflight
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "ELLIPTIC_ROOT", tmp_path / "data" / "elliptic")
    text, blocking = preflight.report()
    assert blocking is True
    for needle in ("REQUIRED: artifacts/graph/xgboost_tier1_tier2.json", "REQUIRED: artifacts/voice/phase2c_D_ssl_mean_only_P1_s42.pt",
                   "optional: data/models/wav2vec2-xls-r-300m/pytorch_model.bin", "Files you must supply"):
        assert needle in text
    (tmp_path / "artifacts" / "graph").mkdir(parents=True)
    (tmp_path / "artifacts" / "graph" / "xgboost_tier1_tier2.json").write_text("{}")
    (tmp_path / "artifacts" / "graph" / "gat_weights.pt").write_bytes(b"x")
    graph_text, graph_blocking = preflight.report("graph")
    assert graph_blocking is False and "Elliptic dataset" in graph_text


def test_preflight_is_silent_when_everything_is_present(tmp_path, monkeypatch):
    from serving import preflight
    monkeypatch.setattr(preflight, "missing", lambda group=None: [])
    assert preflight.report() == ("", False)
