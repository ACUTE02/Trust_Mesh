"""F2 (the Phase 2D codec-augmented head) is an opt-in voice model. D stays the default everywhere; /case/score
never uses F2; without the private F2 weights the service starts and F2 is reported as unavailable.

Tiny random heads only: these tests check behaviour, not accuracy.
Run: .venv\\Scripts\\python.exe -m pytest tests/test_voice_f2_optin.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from engines.voice.model import SSLHead
from engines.voice.ssl import SSLEncoder, tiny_wav2vec2_for_smoke
from serving import config
from serving.api import create_app
from serving.cases import CaseStore
from serving.voice_service import F2_LOGIT_THRESHOLD, F2_THRESHOLD, VoiceService

from test_serving import StubGraph, StubNeo4j, wav_bytes


@pytest.fixture(autouse=True)
def hermetic(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_THRESHOLDS", tmp_path / "no_such_voice_thresholds.json")
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "demo.wav").write_bytes(wav_bytes())
    monkeypatch.setattr(config, "DEMO_CLIPS_DIR", clips)


def make_service(tmp_path, with_f2=True, only_two_f2=False):
    folder = tmp_path / "voice"
    folder.mkdir(exist_ok=True)
    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    torch.manual_seed(0)
    torch.save(SSLHead(encoder.n_layers, encoder.hidden_size).state_dict(), folder / "phase2c_D_ssl_mean_only_P1_s42.pt")
    if with_f2:
        for seed in (42, 43) if only_two_f2 else (42, 43, 44):
            torch.manual_seed(seed)  # different seeds give different heads
            head = SSLHead(encoder.n_layers, encoder.hidden_size)
            torch.save({"state_dict": head.state_dict(), "n_layers": encoder.n_layers, "hidden": encoder.hidden_size,
                        "mean_only": True}, folder / f"phase2d_F2_s{seed}.pt")
    svc = VoiceService(artifacts_dir=tmp_path)
    svc.load(encoder=encoder)
    return svc


def make_client(svc):
    app = create_app(graph=StubGraph(), voice=svc, neo4j=StubNeo4j(), cases=CaseStore(":memory:"), load_on_startup=False)
    return TestClient(app)


def test_default_is_d_even_when_f2_is_installed(tmp_path):
    svc = make_service(tmp_path)
    assert svc.default_model == "D"
    r = svc.score_bytes(wav_bytes())
    assert r["model"] == "D" and r["model_label"] == "D (seed 42)" and not r["fallback_used"]


def test_f2_scores_when_asked_and_is_the_mean_of_three_seed_logits(tmp_path):
    svc = make_service(tmp_path)
    r = svc.score_bytes(wav_bytes(), model="F2")
    assert r["model"] == "F2" and "42, 43 and 44" in r["model_label"] and r["fallback_used"] is False
    # the encoder is the same, so the F2 logit must be the mean of the three single-head logits
    from serving.voice_service import MeanLogitHeads
    assert isinstance(svc.heads["F2"], MeanLogitHeads) and len(svc.heads["F2"].heads) == 3
    assert r["score"] == pytest.approx(1.0 / (1.0 + math.exp(r["bonafide_logit"])))
    from serving.voice_service import prepare_clip
    from test_serving import wav_bytes as wb
    import io
    import soundfile as sf
    wave, sr = sf.read(io.BytesIO(wb()), dtype="float32")
    clip, _ = prepare_clip(wave, sr)
    with torch.no_grad():
        pooled = svc.encoder.mean_only(svc.encoder.pooled(torch.from_numpy(clip).unsqueeze(0)))
        singles = [float(h(pooled)[0]) for h in svc.heads["F2"].heads]
    assert r["bonafide_logit"] == pytest.approx(float(np.mean(singles)), abs=1e-5)
    assert len(set(round(s, 6) for s in singles)) == 3  # three genuinely different heads


def test_f2_threshold_is_the_recorded_clean_validation_threshold_mapped_exactly():
    assert F2_LOGIT_THRESHOLD == pytest.approx(0.19378728093336556)
    assert F2_THRESHOLD["high"] == pytest.approx(1.0 / (1.0 + math.exp(F2_LOGIT_THRESHOLD)))
    assert F2_THRESHOLD["single_band"] is True and "not recalibrated" in F2_THRESHOLD["source"].lower()
    results = Path(config.ARTIFACTS_DIR) / "voice" / "results_phase2d.json"
    if results.exists():  # the private results file is the source of the number
        recorded = json.loads(results.read_text())["evaluation"]["thresholds_clean_val_eer"]["F2"]
        assert recorded == pytest.approx(F2_LOGIT_THRESHOLD, abs=1e-12)


def test_f2_has_one_band_so_no_medium_verdict(tmp_path):
    svc = make_service(tmp_path)
    t = F2_THRESHOLD["high"]
    assert svc.verdict(t + 1e-6, "F2") == "high" and svc.verdict(t - 1e-6, "F2") == "low"
    assert svc.verdict(0.3, "D") in ("low", "medium", "high")
    r = svc.score_bytes(wav_bytes(), model="F2")
    assert r["verdict"] in ("low", "high") and r["thresholds"]["single_band"] is True


def test_f2_applies_the_same_audio_guard_as_d(tmp_path, monkeypatch):
    from serving import audio_guard
    svc = make_service(tmp_path)
    monkeypatch.setattr(config, "AUDIO_GUARD", True)
    for model in (None, "F2"):  # a steady tone is not speech: refused whichever head would score it
        with pytest.raises(audio_guard.NotSpeechError):
            svc.score_bytes(wav_bytes(), model=model)


def test_missing_f2_weights_the_service_starts_and_f2_is_unavailable(tmp_path):
    svc = make_service(tmp_path, with_f2=False)
    assert svc.ready and svc.default_model == "D"
    info = svc.info()
    assert "F2" not in info["loaded"] and info["opt_in"] == {"F2": False} and info["weights"]["F2"] is None
    with pytest.raises(KeyError) as err:
        svc.score_bytes(wav_bytes(), model="F2")
    assert "not available on this server" in str(err.value) and "private" in str(err.value)


def test_two_of_three_f2_files_is_not_enough(tmp_path):
    svc = make_service(tmp_path, only_two_f2=True)
    assert svc.info()["opt_in"] == {"F2": False}


def test_health_lists_loaded_models_and_the_default(tmp_path):
    with make_client(make_service(tmp_path)) as c:
        v = c.get("/health").json()["voice"]
        assert v["serving_default"] == "D" and v["loaded"] == ["D", "F2"] and v["opt_in"] == {"F2": True}
        assert "F2" in v["model_labels"]
    (tmp_path / "none").mkdir()
    with make_client(make_service(tmp_path / "none", with_f2=False)) as c:
        v = c.get("/health").json()["voice"]
        assert v["serving_default"] == "D" and v["loaded"] == ["D"] and v["opt_in"] == {"F2": False}


def test_voice_score_endpoint_default_f2_and_unavailable(tmp_path):
    with make_client(make_service(tmp_path)) as c:
        d = c.post("/voice/score", data={"clip": "demo.wav"}, params={"record": "false"})
        assert d.status_code == 200 and d.json()["model"] == "D"
        f = c.post("/voice/score", data={"clip": "demo.wav"}, params={"model": "F2", "record": "false"})
        assert f.status_code == 200 and f.json()["model"] == "F2" and "42, 43 and 44" in f.json()["model_label"]
    (tmp_path / "b").mkdir()
    with make_client(make_service(tmp_path / "b", with_f2=False)) as c:
        r = c.post("/voice/score", data={"clip": "demo.wav"}, params={"model": "F2", "record": "false"})
        assert r.status_code == 422 and "not available on this server" in r.json()["detail"]
        assert "Traceback" not in r.text


def test_case_score_ignores_an_f2_request_and_uses_d(tmp_path):
    with make_client(make_service(tmp_path)) as c:
        r = c.post("/case/score", data={"node_id": 7, "clip": "demo.wav"}, params={"model": "F2", "record": "false"})
        assert r.status_code == 200
        assert r.json()["voice"]["model"] == "D" and r.json()["fusion"]["signals_used"] == ["graph", "voice"]
        plain = c.post("/case/score", data={"node_id": 7, "clip": "demo.wav"}, params={"record": "false"}).json()
        assert plain["voice"]["score"] == pytest.approx(r.json()["voice"]["score"])
