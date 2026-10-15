"""Tests for the audio-quality guard (serving/audio_guard.py, wired into serving/voice_service.py).

Each service-level test turns the guard on explicitly (tests/conftest.py leaves it off for the older tests, which
use tones as stand-in audio). On the code before the guard these tests fail: the module and the AUDIO_GUARD
setting do not exist, and a tone or noise got an ordinary score.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_audio_guard.py
"""
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import audio_fixtures as fx
from engines.voice.preprocess import trim_silence
from serving import audio_guard, config
from serving.api import create_app
from serving.cases import CaseStore
from test_serving import StubGraph, StubNeo4j, tiny_voice_service

DEMO_CLIPS = Path(__file__).resolve().parents[1] / "data" / "demo_clips"


@pytest.fixture(autouse=True)
def hermetic_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VOICE_THRESHOLDS", tmp_path / "no_such_voice_thresholds.json")


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(config, "AUDIO_GUARD", True)


def wav(samples, sr=fx.SR):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def checked(samples):
    trimmed, _ = trim_silence(np.asarray(samples, dtype=np.float32), fx.SR, 30.0)
    return audio_guard.check(trimmed)


# --- the measures -------------------------------------------------------------------------------------------
def test_speech_like_audio_is_speech_and_not_low_quality():
    r = checked(fx.speech_like())
    assert r["speech_like"] is True
    assert r["low_quality_audio"] is False and r["reasons"] == []
    assert r["spectral_movement"] > audio_guard.SPEECH_MOVEMENT_MIN * 2  # a wide margin, not an edge case


@pytest.mark.parametrize("name,samples", [("tone", fx.tone()), ("tone 1 kHz", fx.tone(freq=1000)),
                                          ("white noise", fx.white_noise()), ("quiet noise", fx.white_noise(level=0.02))])
def test_steady_sounds_are_not_speech(name, samples):
    r = checked(samples)
    assert r["speech_like"] is False, name
    assert r["low_quality_audio"] is False  # not speech: nothing is said about its quality


def test_band_limited_speech_is_flagged_low_quality_with_reasons():
    r = checked(fx.telephone_band(fx.speech_like()))
    assert r["speech_like"] is True and r["low_quality_audio"] is True
    assert "very little sound above 5 kHz" in r["reasons"]
    assert "sound cut off below 7.6 kHz" in r["reasons"]


def test_a_file_recorded_at_8_khz_is_flagged_even_though_up_sampling_hides_the_band_limit(tmp_path, guard_on):
    """Up-sampling 8 kHz audio by interpolation fills 4 to 8 kHz with mirror images, so the spectral rules cannot see
    the missing band. The file's own sample rate is what gives it away."""
    from engines.voice.preprocess import resample_linear
    speech_8k = fx.speech_like()[::2].copy()  # crude decimation: 8 kHz
    assert checked(resample_linear(speech_8k, 8000)[0])["low_quality_audio"] is False  # the spectral rules alone miss it
    result = tiny_voice_service(tmp_path).score_bytes(wav(speech_8k, sr=8000))
    assert result["low_quality_audio"] is True
    assert any("sample rate" in reason for reason in result["audio_quality"]["reasons"])


def test_thresholds_are_reported_with_the_measures():
    r = checked(fx.speech_like())
    assert r["thresholds"] == {"spectral_movement_min": 0.30, "hf_to_mid_min_db": -55.0, "cutoff_min_hz": 7600.0,
                               "min_native_rate_hz": 16000}


def test_a_clip_with_under_50_ms_of_loud_sound_cannot_be_speech():
    r = audio_guard.check(fx.speech_like()[:300])  # 19 ms: shorter than one analysis frame
    assert r["analysed"] is False and r["speech_like"] is False


# --- the service ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("samples", [fx.tone(), fx.white_noise()])
def test_service_refuses_non_speech_with_a_plain_message(tmp_path, guard_on, samples):
    svc = tiny_voice_service(tmp_path)
    with pytest.raises(audio_guard.NotSpeechError, match="does not look like speech"):
        svc.score_bytes(wav(samples))


def test_service_scores_speech_like_audio_with_no_warning(tmp_path, guard_on):
    result = tiny_voice_service(tmp_path).score_bytes(wav(fx.speech_like()))
    assert result["low_quality_audio"] is False and result["warnings"] == [] and result["low_confidence"] is False
    assert result["audio_quality"]["speech_like"] is True


def test_band_limited_speech_is_scored_with_a_warning_and_the_score_is_unchanged(tmp_path, monkeypatch):
    svc = tiny_voice_service(tmp_path)
    blob = wav(fx.telephone_band(fx.speech_like()))
    monkeypatch.setattr(config, "AUDIO_GUARD", False)
    without_guard = svc.score_bytes(blob)
    monkeypatch.setattr(config, "AUDIO_GUARD", True)
    with_guard = svc.score_bytes(blob)
    assert with_guard["low_quality_audio"] is True and with_guard["low_confidence"] is True
    assert audio_guard.LOW_QUALITY_MESSAGE in with_guard["warnings"]
    # the guard warns, it never changes the answer or the risk band
    assert with_guard["score"] == without_guard["score"]
    assert with_guard["verdict"] == without_guard["verdict"]
    assert with_guard["bonafide_logit"] == without_guard["bonafide_logit"]


def test_guard_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUDIO_GUARD", False)
    result = tiny_voice_service(tmp_path).score_bytes(wav(fx.tone()))
    assert 0.0 <= result["score"] <= 1.0 and result["audio_quality"] is None and result["low_quality_audio"] is False


def test_warning_text_matches_the_wording_the_site_promises():
    assert audio_guard.LOW_QUALITY_MESSAGE == (
        "This audio looks phone-quality or heavily compressed. This model was trained on clean recordings, "
        "so the result is unreliable.")


# --- the API -------------------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, guard_on):
    app = create_app(graph=StubGraph(), voice=tiny_voice_service(tmp_path), neo4j=StubNeo4j(),
                     cases=CaseStore(":memory:"), load_on_startup=False)
    with TestClient(app) as c:
        yield c


def test_api_answers_422_with_the_plain_message_for_a_tone(client):
    r = client.post("/voice/score", files={"file": ("t.wav", wav(fx.tone()), "audio/wav")})
    assert r.status_code == 422
    assert r.json()["detail"].startswith("This does not look like speech")
    assert "score" not in r.json()


def test_api_returns_the_warning_and_the_measures_for_band_limited_speech(client):
    r = client.post("/voice/score", files={"file": ("p.wav", wav(fx.telephone_band(fx.speech_like())), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["low_quality_audio"] is True
    assert audio_guard.LOW_QUALITY_MESSAGE in body["warnings"]
    assert body["audio_quality"]["thresholds"]["cutoff_min_hz"] == 7600.0
    assert body["audio_quality"]["reasons"]


def test_api_refuses_a_tone_in_a_combined_case_too(client):
    r = client.post("/case/score", data={"node_id": 7}, files={"file": ("t.wav", wav(fx.tone()), "audio/wav")})
    assert r.status_code == 422 and "does not look like speech" in r.json()["detail"]


# --- real local clips: the guard must not reject genuine speech -------------------------------------------------
@pytest.mark.skipif(not list(DEMO_CLIPS.glob("*.wav")), reason="local demo clips are not present (they are not in git)")
def test_real_dataset_clips_are_all_treated_as_speech_and_none_as_low_quality():
    clips = sorted(DEMO_CLIPS.glob("*.wav"))
    for path in clips:
        r = checked(sf.read(path, dtype="float32")[0])
        assert r["speech_like"], path.name
        assert not r["low_quality_audio"], path.name
