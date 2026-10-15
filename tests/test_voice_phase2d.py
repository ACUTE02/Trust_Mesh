"""Tests for Phase 2D machinery (engines/voice/codec.py, engines/voice/robust.py, notebook 06's constants).

Run: .venv\\Scripts\\python.exe -m pytest tests/test_voice_phase2d.py
The full notebook is exercised separately by tests/smoke_notebook_06.py (slow: real ffmpeg, tiny encoder).
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from engines.voice import codec, robust
from engines.voice.data import BONAFIDE, SPOOF, speaker_disjoint_split
from engines.voice.evaluate import bootstrap_eer_difference, compute_eer

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def speech_like(seconds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    carrier = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 700, 1500, 3000, 5200), 1))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t))
    return (0.25 * carrier * envelope + 0.01 * rng.normal(size=len(t))).astype(np.float32)


def band_energy(wave, lo, hi):
    spec = np.abs(np.fft.rfft(wave)) ** 2
    freqs = np.fft.rfftfreq(len(wave), 1 / SR)
    return float(spec[(freqs >= lo) & (freqs < hi)].sum())


@pytest.fixture(scope="module")
def ffmpeg():
    try:
        return codec.find_ffmpeg()
    except codec.FFmpegMissing as exc:
        pytest.skip(str(exc))


# --- the named conditions -----------------------------------------------------------------------------------
def test_condition_names_are_the_pre_registered_ones():
    assert codec.TRAINING_MENU == ("aac32", "aac32_via48k", "mp3_64", "opus24", "echo", "noise_lowsnr")
    assert codec.HELD_OUT == ("aac16", "opus12", "mulaw8k", "mp3_32")
    assert not set(codec.TRAINING_MENU) & set(codec.HELD_OUT)


@pytest.mark.parametrize("name", [n for n in codec.ALL_CONDITIONS if n not in codec.SYNTHETIC])
def test_every_real_codec_round_trips_to_a_16k_float_clip_that_differs(ffmpeg, name):
    wave = speech_like()
    out = codec.apply_condition(wave, name, np.random.default_rng(0), ffmpeg)
    assert out.dtype == np.float32 and out.ndim == 1
    assert abs(len(out) - len(wave)) < 0.2 * SR            # codec delay/padding only
    n = min(len(out), len(wave))
    assert not np.allclose(out[:n], wave[:n], atol=1e-3)   # it really was processed
    assert np.isfinite(out).all() and np.abs(out).max() <= 1.5


def test_telephony_condition_removes_everything_above_4_khz(ffmpeg):
    out = codec.apply_condition(speech_like(), "mulaw8k", np.random.default_rng(0), ffmpeg)
    assert band_energy(out, 4500, 8000) < 1e-3 * band_energy(out, 100, 3400)


def test_echo_adds_a_tail_and_keeps_the_level():
    wave = np.concatenate([speech_like(1.0), np.zeros(SR, dtype=np.float32)])   # speech then silence
    out = codec.apply_condition(wave, "echo", np.random.default_rng(1))
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(np.sqrt(np.mean(wave ** 2)), rel=0.05)
    assert np.sqrt(np.mean(out[SR + 2000:SR + 8000] ** 2)) > 1e-3   # reverberation continues after the speech stops


def test_noise_condition_stays_within_the_stated_snr_range():
    wave = speech_like(4.0)
    for seed in range(8):
        out = codec.apply_condition(wave, "noise_lowsnr", np.random.default_rng(seed))
        noise = out - wave
        snr = 10 * np.log10(np.mean(wave.astype(float) ** 2) / np.mean(noise.astype(float) ** 2))
        assert codec.NOISE_SNR_DB[0] - 1.0 <= snr <= codec.NOISE_SNR_DB[1] + 1.0


def test_synthetic_conditions_are_reproducible_for_a_clip_and_differ_between_clips():
    wave = speech_like()
    a = codec.apply_condition(wave, "echo", codec.condition_rng(42, 7, "echo"))
    b = codec.apply_condition(wave, "echo", codec.condition_rng(42, 7, "echo"))
    c = codec.apply_condition(wave, "echo", codec.condition_rng(42, 8, "echo"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_unknown_condition_is_an_error():
    with pytest.raises(KeyError):
        codec.apply_condition(speech_like(), "gsm", np.random.default_rng(0), "ffmpeg")


# --- assignment and resolution ------------------------------------------------------------------------------
def test_assignment_is_distinct_per_clip_and_fixed_by_seed():
    menu = list(codec.TRAINING_MENU)
    a = codec.assign_conditions(500, menu, seed=42)
    assert a.shape == (500, 3)
    assert all(len(set(row)) == 3 for row in a)
    assert np.array_equal(a, codec.assign_conditions(500, menu, seed=42))
    assert not np.array_equal(a, codec.assign_conditions(500, menu, seed=43))
    assert set(a.ravel()) == set(range(len(menu)))           # every condition is used


def test_assignment_with_too_small_a_menu_is_refused():
    with pytest.raises(ValueError, match="3 are needed"):
        codec.assign_conditions(10, ["aac32", "mp3_64"])


def test_a_missing_codec_is_substituted_openly_and_recorded():
    def probe(name):
        if name.startswith("opus"):
            raise RuntimeError("Unknown encoder 'libopus'")
    r = codec.resolve_conditions("ffmpeg", probe=probe)
    assert r["substituted"] == {"opus24": "aac24_for_opus24", "opus12": "aac12_for_opus12"}
    assert "aac24_for_opus24" in r["training"] and "aac12_for_opus12" in r["held_out"]
    assert "opus24" in r["unavailable"] and "libopus" in r["unavailable"]["opus24"]


def test_a_codec_with_no_substitute_is_dropped_and_recorded():
    def probe(name):
        if name in ("mp3_64", "mp3_32"):
            raise RuntimeError("no encoder")
    r = codec.resolve_conditions("ffmpeg", probe=probe)
    assert "mp3_64" not in r["training"] and "mp3_32" not in r["held_out"] and "mp3_64" in r["unavailable"]


def test_ffmpeg_is_found_here_or_through_the_package(ffmpeg):
    assert Path(ffmpeg).exists() or ffmpeg == "ffmpeg"


# --- disk and cost ------------------------------------------------------------------------------------------
def test_bytes_per_version_matches_the_addendum():
    assert robust.bytes_per_version(25, 1024) == 51200


def test_version_counts():
    c = robust.version_counts(n_clips=100, n_trainval=70, n_eval_clips=30, n_eval_conditions=10, per_clip=3)
    assert c == {"clean": 100, "augmented": 210, "evaluation": 300, "total": 610}


def test_disk_check_stops_with_a_plain_message(tmp_path):
    with pytest.raises(robust.DiskBudgetError, match="Not enough free disk space"):
        robust.check_disk(tmp_path, required_bytes=10 ** 18)
    assert robust.check_disk(tmp_path, required_bytes=1000)["ok"] is True


def test_cost_projection_is_a_straight_line():
    p = robust.project_cost(measured_seconds=100.0, measured_versions=200, measured_bytes=2_000_000, total_versions=20000)
    assert p["projected_seconds"] == pytest.approx(10000.0) and p["projected_cache_gb"] == pytest.approx(0.2)


# --- chunking and resume ------------------------------------------------------------------------------------
class StubEncoder:
    """Deterministic stand-in for the frozen encoder: a pooled vector that depends only on the waveform."""
    hidden_size, n_layers = 8, 3

    def pooled(self, waves):
        m = waves.mean(dim=1, keepdim=True)
        s = waves.std(dim=1, keepdim=True)
        base = torch.cat([m, s, waves[:, :6]], dim=1)                # [b, 8]
        return torch.stack([base * (k + 1) for k in range(self.n_layers)], dim=1).repeat(1, 1, 2)


def make_clips(n):
    return {p: (speech_like(1.0, seed=p), SR) for p in range(n)}


def extract(out_dir, get_clip, positions, **kw):
    trainval = {p for p in positions if p % 3 != 0}
    evalset = {p for p in positions if p % 3 == 0}
    return robust.extract_features(
        get_clip, list(positions), StubEncoder(), SR, out_dir, trainval=trainval,
        aug_conditions=["echo", "noise_lowsnr", "echo", "noise_lowsnr"],
        aug_assign={p: [0, 1, 3] for p in trainval}, eval_positions=evalset,
        eval_conditions=["echo", "noise_lowsnr"], chunk_clips=4, batch_size=3, workers=2, log=lambda *_: None, **kw)


def test_a_finished_run_has_every_version_and_no_temp_files(tmp_path):
    clips = make_clips(10)
    stats = extract(tmp_path, lambda p: clips[p], range(10))
    assert stats["chunks_done"] == 3 and not list(tmp_path.glob("*.tmp.npz"))
    loaded = robust.load_chunks([tmp_path])
    assert sorted(loaded["clean"]["positions"]) == list(range(10))
    n_tv = sum(p % 3 != 0 for p in range(10))
    assert len(loaded["aug"]["positions"]) == 3 * n_tv
    assert len(loaded["eval"]["positions"]) == 2 * sum(p % 3 == 0 for p in range(10))
    assert loaded["clean"]["data"].shape == (10, 3, 8) and loaded["clean"]["data"].dtype == np.float16
    assert set(loaded["eval"]["cond"]) == {0, 1}


def test_a_crash_midway_resumes_without_redoing_finished_chunks_and_matches_a_clean_run(tmp_path):
    clips = make_clips(10)
    extract(tmp_path / "reference", lambda p: clips[p], range(10))

    def broken(p):
        if p == 8:
            raise OSError("session killed")
        return clips[p]

    with pytest.raises(OSError):
        extract(tmp_path / "resumed", broken, range(10))
    assert len(list((tmp_path / "resumed").glob("chunk_*.npz"))) == 2
    assert not list((tmp_path / "resumed").glob("*.tmp.npz"))

    stats = extract(tmp_path / "resumed", lambda p: clips[p], range(10))
    assert stats["chunks_reused"] == 2 and stats["chunks_done"] == 1
    a, b = robust.load_chunks([tmp_path / "reference"]), robust.load_chunks([tmp_path / "resumed"])
    for kind in ("clean", "aug", "eval"):
        assert np.array_equal(a[kind]["positions"], b[kind]["positions"])
        assert np.array_equal(a[kind]["data"], b[kind]["data"])


def test_a_time_limit_stops_cleanly_at_a_chunk_boundary(tmp_path):
    clips = make_clips(10)
    stats = extract(tmp_path, lambda p: clips[p], range(10), deadline=0.0)
    assert stats["stopped_early"] is True and stats["chunks_done"] == 0
    stats = extract(tmp_path, lambda p: clips[p], range(10))
    assert stats["chunks_done"] == 3 and stats["stopped_early"] is False


def test_chunks_from_an_earlier_session_are_reused(tmp_path):
    clips = make_clips(10)
    extract(tmp_path / "old", lambda p: clips[p], range(10))
    stats = extract(tmp_path / "new", lambda p: clips[p], range(10), resume_dirs=[tmp_path / "old"])
    assert stats["chunks_done"] == 0 and stats["chunks_reused"] == 3
    assert len(robust.load_chunks([tmp_path / "new", tmp_path / "old"])["clean"]["positions"]) == 10


def test_stored_features_hold_no_waveforms(tmp_path):
    clips = make_clips(4)
    extract(tmp_path, lambda p: clips[p], range(4))
    with np.load(next(tmp_path.glob("chunk_*.npz"))) as z:
        for name in z.files:
            assert z[name].dtype in (np.float16, np.int64, np.int16)
            assert z[name].ndim == 1 or z[name].shape[1:] == (3, 8)   # [n] index arrays or [n, layers, hidden]


# --- the split is Model D's split ---------------------------------------------------------------------------
def test_notebook_06_uses_notebook_04_split_settings():
    def constants(name):
        text = (ROOT / "notebooks" / "src" / name).read_text(encoding="utf-8")
        return {
            "spoof_per_bonafide": re.search(r"^SPOOF_PER_BONAFIDE = (\d+)", text, re.M).group(1),
            "fractions": re.search(r"^TEST_FRACTION, VAL_FRACTION = ([\d., ]+)", text, re.M).group(1).strip(),
            "seed": re.search(r"^SEED = (\d+)", text, re.M).group(1),
            "seeds": re.search(r"^SEEDS = (\[.*?\])", text, re.M).group(1),
            "ssl_model": re.search(r'^SSL_MODEL = "(.*?)"', text, re.M).group(1),
            "lr_patience": re.search(r"^SSL_LR, SSL_PATIENCE = (.*)$", text, re.M).group(1),
            "split_call": re.search(r"^trainval, test = (.*)$", text, re.M).group(1),
            "sub_call": re.search(r"^train, val = (.*)$", text, re.M).group(1),
        }
    assert constants("04_voice_depth.py") == constants("06_voice_codec_robust.py")


def test_split_is_deterministic_and_speaker_disjoint():
    rng = np.random.default_rng(0)
    clips = [{"speaker_id": f"S{i % 23}", "label": int(rng.random() < 0.3)} for i in range(600)]
    a = speaker_disjoint_split(clips, 0.3, 42)
    assert a == speaker_disjoint_split(clips, 0.3, 42)
    assert not {clips[i]["speaker_id"] for i in a[0]} & {clips[i]["speaker_id"] for i in a[1]}


def test_expected_split_matches_the_recorded_model_d_results_when_present():
    path = ROOT / "artifacts" / "voice" / "results_phase2b.json"
    if not path.exists():
        pytest.skip("results_phase2b.json is not present locally")
    recorded = json.loads(path.read_text(encoding="utf-8"))["splits"]["P1"]
    text = (ROOT / "notebooks" / "src" / "06_voice_codec_robust.py").read_text(encoding="utf-8")
    expected = json.loads(re.search(r"^EXPECTED_P1 = (\{.*?\})\n", text, re.M | re.S).group(1).replace("'", '"'))
    for part in ("train", "val", "test"):
        assert expected[part] == {"bonafide": recorded[part]["bonafide"], "spoof": recorded[part]["spoof"]}


# --- the comparison -----------------------------------------------------------------------------------------
def scored_set(n=400, seed=0, gap=2.0):
    rng = np.random.default_rng(seed)
    labels = np.array([BONAFIDE] * (n // 4) + [SPOOF] * (3 * n // 4))
    return labels, rng.normal(size=n) + gap * (labels == BONAFIDE)


def test_clustered_bootstrap_sees_a_clearly_better_model():
    labels, good = scored_set(gap=3.0)
    _, bad = scored_set(gap=0.5, seed=0)
    diff = robust.clustered_paired_bootstrap(good, bad, labels, np.arange(len(labels)), n_resamples=200, seed=1)
    assert diff["difference"] < 0 and diff["excludes_zero"] and diff["ci_high"] < 0
    assert robust.criterion_status(diff) == {"not_worse": True, "better": True}


def test_identical_models_do_not_exclude_zero_and_are_not_better():
    labels, s = scored_set()
    diff = robust.clustered_paired_bootstrap(s, s, labels, np.arange(len(labels)), n_resamples=100, seed=1)
    assert diff["difference"] == 0 and not diff["excludes_zero"]
    assert robust.criterion_status(diff) == {"not_worse": True, "better": False}


def test_a_worse_model_fails_the_not_worse_check():
    labels, good = scored_set(gap=3.0)
    _, bad = scored_set(gap=0.5)
    diff = robust.clustered_paired_bootstrap(bad, good, labels, np.arange(len(labels)), n_resamples=200, seed=1)
    assert robust.criterion_status(diff) == {"not_worse": False, "better": False}


def test_clustered_interval_is_wider_than_treating_copies_as_independent():
    labels, a = scored_set(n=200, gap=1.0)
    _, b = scored_set(n=200, gap=1.3, seed=5)
    copies = 5
    la, sa, sb = np.tile(labels, copies), np.tile(a, copies), np.tile(b, copies)
    groups = np.tile(np.arange(len(labels)), copies)
    naive = bootstrap_eer_difference(sa, la, sb, la, n_resamples=200, seed=2, paired=True)
    clustered = robust.clustered_paired_bootstrap(sa, sb, la, groups, n_resamples=200, seed=2)
    assert (clustered["ci_high"] - clustered["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_share_flagged_counts_only_genuine_clips_below_the_threshold():
    labels = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([2.0, 1.0, -1.0, -2.0, -3.0, 5.0])
    assert robust.share_flagged(scores, labels, threshold=0.0) == 0.5


def test_flagged_share_threshold_comes_from_the_clean_eer():
    labels, s = scored_set()
    result = compute_eer(s, labels)
    assert robust.share_flagged(s, labels, result["threshold"]) == pytest.approx(result["frr"], abs=0.02)
