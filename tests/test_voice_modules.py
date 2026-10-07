"""CPU unit tests for engines/voice on tiny synthetic audio. No real data, no reported numbers.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_voice_modules.py
"""
import numpy as np
import pytest
import torch

from engines.voice.data import (class_counts, make_synthetic_asvspoof, pos_weight, random_split,
                                speaker_disjoint_split, split_report)
from engines.voice.evaluate import (compute_eer, det_curve_points, duration_leakage_probe,
                                    evaluate_scores, leakage_verdict, per_class_accuracy)
from engines.voice.model import LCNN, predict_scores, train_lcnn
from engines.voice.preprocess import VoicePreprocessor, fix_length, trim_silence


def test_eer_perfect_and_chance_separation():
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert compute_eer(np.array([9.0, 8.0, 7.0, 1.0, 2.0, 3.0]), labels)["eer"] == 0.0
    # Perfectly inverted scores: every threshold is wrong, so EER is 1.0.
    assert compute_eer(np.array([1.0, 2.0, 3.0, 9.0, 8.0, 7.0]), labels)["eer"] == 1.0
    # A known-EER case: one bonafide and one spoof sit on the wrong side -> 0.25 each way.
    labels8 = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    scores8 = np.array([9.0, 8.0, 7.0, 1.5, 6.0, 3.0, 2.0, 1.0])
    assert abs(compute_eer(scores8, labels8)["eer"] - 0.25) < 1e-9


def test_eer_matches_bruteforce_threshold_sweep():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=400)
    scores = rng.normal(labels * 1.2, 1.0)
    eer = compute_eer(scores, labels)["eer"]
    grid = np.linspace(scores.min() - 1, scores.max() + 1, 20001)
    far = [(scores[labels == 0] > t).mean() for t in grid]
    frr = [(scores[labels == 1] <= t).mean() for t in grid]
    brute = min((f + r) / 2 for f, r in zip(far, frr) if abs(f - r) < 0.01)
    assert abs(eer - brute) < 0.01, f"interpolated EER {eer} vs sweep {brute}"


def test_eer_single_class_is_nan_not_crash():
    assert np.isnan(compute_eer(np.array([1.0, 2.0]), np.array([1, 1]))["eer"])


def test_trim_silence_removes_padding_and_measures_it():
    sr = 16000
    speech = 0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)  # 1.0 s
    padded = np.concatenate([np.zeros(sr // 2, np.float32), speech, np.zeros(sr, np.float32)])
    trimmed, stats = trim_silence(padded, sr)
    assert abs(stats["lead_silence_s"] - 0.5) < 0.05 and abs(stats["trail_silence_s"] - 1.0) < 0.05
    assert abs(stats["speech_s"] - 1.0) < 0.1 and abs(len(trimmed) / sr - 1.0) < 0.1
    assert abs(stats["total_s"] - 2.5) < 1e-6
    # An already-tight clip should barely change.
    _, tight = trim_silence(speech, sr)
    assert tight["lead_silence_s"] < 0.05 and tight["trail_silence_s"] < 0.05


def test_fix_length_crops_and_repeat_pads():
    assert len(fix_length(np.ones(100, np.float32), 40)) == 40
    padded = fix_length(np.arange(10, dtype=np.float32), 25)
    assert len(padded) == 25 and padded[10] == 0.0 and padded[24] == 4.0  # repeat, not zero-pad
    assert np.all(fix_length(np.zeros(0, np.float32), 8) == 0)


def test_preprocessor_shapes_and_determinism():
    pre = VoicePreprocessor(clip_seconds=1.0)
    clip = make_synthetic_asvspoof(4)[0]
    feats, stats = pre(clip["waveform"], clip["sample_rate"])
    assert feats.shape == (pre.n_features, pre.n_frames) == (60, pre.n_frames)
    assert torch.isfinite(feats).all()
    again, _ = pre(clip["waveform"], clip["sample_rate"])
    assert torch.equal(feats, again)
    assert set(stats) >= {"total_s", "lead_silence_s", "trail_silence_s", "speech_s"}

    untrimmed = VoicePreprocessor(clip_seconds=1.0, trim=False)
    _, raw_stats = untrimmed(clip["waveform"], clip["sample_rate"])
    assert raw_stats["lead_silence_s"] == 0.0  # not measured when trimming is off


def test_speaker_disjoint_split_shares_no_speaker():
    clips = make_synthetic_asvspoof(120)
    train_idx, test_idx = speaker_disjoint_split(clips, test_fraction=0.3)
    report = split_report(clips, train_idx, test_idx)
    assert report["shared_speakers"] == 0
    assert set(train_idx).isdisjoint(test_idx)
    assert len(train_idx) + len(test_idx) == len(clips)
    assert 0.1 < len(test_idx) / len(clips) < 0.5

    rand_train, rand_test = random_split(clips, test_fraction=0.3)
    assert len(rand_test) == 36 and set(rand_train).isdisjoint(rand_test)
    # The point of the comparison: a random split does share speakers.
    assert split_report(clips, rand_train, rand_test)["shared_speakers"] > 0


def test_class_counts_and_pos_weight():
    clips = make_synthetic_asvspoof(100)
    counts = class_counts(clips)
    assert counts["bonafide"] + counts["spoof"] == 100 and counts["bonafide"] < counts["spoof"]
    assert abs(pos_weight(clips) - counts["spoof"] / counts["bonafide"]) < 1e-9


def test_duration_probe_finds_planted_leakage():
    """Synthetic clips are built with spoof-longer silence, so the probe must flag it, and
    trimming must remove it -- the same before/after check the notebook runs on real data."""
    clips = make_synthetic_asvspoof(160)
    labels = [c["label"] for c in clips]
    untrimmed = VoicePreprocessor(clip_seconds=1.0, trim=False)
    trimmer = VoicePreprocessor(clip_seconds=1.0, trim=True)

    raw_stats = [{"total_s": len(c["waveform"]) / c["sample_rate"], "lead_silence_s": 0.0,
                  "trail_silence_s": 0.0, "speech_s": len(c["waveform"]) / c["sample_rate"],
                  "silence_ratio": 0.0} for c in clips]
    before = duration_leakage_probe(raw_stats, labels)
    assert before["total_s"]["eer"] < 0.35, "planted duration leakage not detected"
    assert leakage_verdict(before)["level"] in ("strong", "mild")

    trimmed_stats = [trimmer(c["waveform"], c["sample_rate"])[1] for c in clips]
    after = duration_leakage_probe([{**s, "total_s": s["speech_s"]} for s in trimmed_stats], labels)
    assert after["total_s"]["eer"] > before["total_s"]["eer"], "trimming did not reduce duration leakage"
    assert untrimmed(clips[0]["waveform"], clips[0]["sample_rate"])[0].shape[0] == 60


def test_lcnn_trains_and_scores():
    torch.manual_seed(0)
    clips = make_synthetic_asvspoof(48)
    pre = VoicePreprocessor(clip_seconds=1.0)
    x, y, _ = pre.featurize_clips(clips, progress_every=None)
    assert x.shape[1:] == (1, pre.n_features, pre.n_frames)

    model = LCNN(n_features=pre.n_features, n_frames=pre.n_frames, channels=(8, 8, 12, 12), fc_dim=16)
    fit = train_lcnn(model, x, y, pos_weight(clips), torch.device("cpu"), epochs=2, batch_size=8,
                     x_val=x, y_val=y, patience=None, verbose=False)
    assert len(fit["history"]) == 2 and np.isfinite(fit["history"][-1]["train_loss"])
    assert 0.0 <= fit["best_val_eer"] <= 1.0

    scores = predict_scores(model, x, torch.device("cpu"))
    assert scores.shape == (len(clips),) and np.isfinite(scores).all()
    report = evaluate_scores(scores, y.numpy())
    assert 0.0 <= report["eer"] <= 1.0
    assert len(report["det_curve"]["far"]) == len(report["det_curve"]["frr"])
    acc = report["per_class_at_eer_threshold"]
    assert acc["n_bonafide"] + acc["n_spoof"] == len(clips)


def test_decode_audio_reads_chained_zip_url(tmp_path):
    """Regression test for the In-the-Wild failure on Kaggle: that dataset ships as one zip, so the
    audio path arrives as a chained fsspec URL ("zip://member::hf://.../x.zip"), which libsndfile
    can't open. A local "zip://member::file://..." URL goes through exactly the same branch."""
    sf = pytest.importorskip("soundfile")
    import zipfile

    from engines.voice.data import _decode_audio

    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr // 4) / sr)).astype(np.float32)
    wav_path = tmp_path / "clip.wav"
    sf.write(wav_path, tone, sr)
    zip_path = tmp_path / "release.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(wav_path, arcname="release/clip.wav")

    url = f"zip://release/clip.wav::{zip_path.resolve().as_uri()}"
    wave, got_sr = _decode_audio({"path": url, "bytes": None})
    assert got_sr == sr and len(wave) == len(tone)
    assert np.allclose(wave, tone, atol=1e-4)

    # Plain local paths and in-memory bytes still work.
    assert len(_decode_audio({"path": str(wav_path), "bytes": None})[0]) == len(tone)
    assert len(_decode_audio({"path": None, "bytes": wav_path.read_bytes()})[0]) == len(tone)


def test_per_class_accuracy_and_det_endpoints():
    labels = np.array([1, 1, 0, 0])
    acc = per_class_accuracy(np.array([1.0, 1.0, -1.0, -1.0]), labels, threshold=0.0)
    assert acc["bonafide_accuracy"] == 1.0 and acc["spoof_accuracy"] == 1.0
    det = det_curve_points(np.array([1.0, 0.9, -1.0, -0.8]), labels)
    assert det["far"][0] == 0.0 and det["frr"][0] == 1.0  # strictest threshold rejects everything
