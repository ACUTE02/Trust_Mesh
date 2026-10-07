"""CPU tests for the Phase 2B voice additions. Synthetic data only; no reported numbers.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_voice_phase2b.py
"""
import numpy as np
import pytest
import torch

from engines.voice.data import (BONAFIDE, SPOOF, ZipAudioArchive, choose_held_out_attacks, drop_attacks,
                                keep_only_attacks, load_archive_clips_balanced, load_eval_protocol,
                                make_synthetic_asvspoof, select_subset, verify_protocol,
                                write_synthetic_archive, write_synthetic_protocol)
from engines.voice.evaluate import (bootstrap_ci_at_counts, bootstrap_eer_ci, bootstrap_eer_difference,
                                    compute_eer, per_attack_eer, speaker_leakage_call)
from engines.voice.features import (INT16_SCALE, build_feature_cache, lfcc_array_source, quantize,
                                    wave_lfcc_source)
from engines.voice.model import LCNN, SSLHead, score_model, train_model
from engines.voice.preprocess import (LFCCExtractor, VoicePreprocessor, ablate_silence, augment_waveforms,
                                      silence_window, spec_augment, trim_silence)


def _scores(labels, separation, rng):
    return rng.normal(labels * separation, 1.0)


# --- Pre-flight fix 1: the leakage rule must use a CI, not a sign ------------------------------------
def test_leakage_rule_not_detectable_when_ci_includes_zero():
    """Two test sets drawn from the same distribution: a small point difference must NOT be called
    leakage -- this is exactly the Phase 2 case (+0.0009 labelled 'speaker leakage')."""
    rng = np.random.default_rng(1)
    la, lb = rng.integers(0, 2, 600), rng.integers(0, 2, 600)
    diff = bootstrap_eer_difference(_scores(la, 1.0, rng), la, _scores(lb, 1.0, rng), lb, n_resamples=300, seed=0)
    assert diff["ci_low"] < 0 < diff["ci_high"] and not diff["excludes_zero"]
    call = speaker_leakage_call(diff)
    assert call["verdict"] == "not_detectable"
    assert call["text"].startswith("no leakage detectable at this sample size")


def test_leakage_rule_detected_when_ci_excludes_zero():
    """Speaker-disjoint (a) much worse than random split (b) -> leakage detected."""
    rng = np.random.default_rng(2)
    la, lb = rng.integers(0, 2, 600), rng.integers(0, 2, 600)
    diff = bootstrap_eer_difference(_scores(la, 0.5, rng), la, _scores(lb, 3.0, rng), lb, n_resamples=300, seed=0)
    assert diff["difference"] > 0 and diff["excludes_zero"]
    assert speaker_leakage_call(diff)["verdict"] == "leakage_detected"


def test_leakage_rule_flags_random_split_significantly_worse():
    rng = np.random.default_rng(3)
    la, lb = rng.integers(0, 2, 600), rng.integers(0, 2, 600)
    diff = bootstrap_eer_difference(_scores(la, 3.0, rng), la, _scores(lb, 0.5, rng), lb, n_resamples=300, seed=0)
    assert speaker_leakage_call(diff)["verdict"] == "random_split_worse"


def test_paired_bootstrap_requires_same_clips_and_is_tighter():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 2, 800)
    a = _scores(labels, 1.5, rng)
    b = a + rng.normal(0, 0.05, len(a))  # nearly identical model on the same clips
    paired = bootstrap_eer_difference(a, labels, b, labels, n_resamples=300, seed=0, paired=True)
    unpaired = bootstrap_eer_difference(a, labels, b, labels, n_resamples=300, seed=0, paired=False)
    assert (paired["ci_high"] - paired["ci_low"]) < (unpaired["ci_high"] - unpaired["ci_low"])
    with pytest.raises(ValueError):
        bootstrap_eer_difference(a, labels, b[:-1], labels[:-1], paired=True)


def test_bootstrap_ci_brackets_point_and_shrinks_with_more_data():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, 4000)
    scores = _scores(labels, 1.5, rng)
    full = bootstrap_eer_ci(scores, labels, n_resamples=300, seed=0)
    assert full["ci_low"] <= full["eer"] <= full["ci_high"]
    small = bootstrap_ci_at_counts(scores, labels, n_bonafide=100, n_spoof=400, n_resamples=300, seed=0)
    assert small["ci_width"] > full["ci_width"]


def test_per_attack_eer_isolates_each_attack():
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    attacks = np.array(["bonafide"] * 3 + ["A07", "A07", "A08", "A08"])
    scores = np.array([5.0, 6.0, 7.0, 0.0, 1.0, 9.0, 8.0])  # A07 easy, A08 fools the model
    out = per_attack_eer(scores, labels, attacks)
    assert out["A07"]["eer"] == 0.0 and out["A08"]["eer"] > 0.5
    assert out["A07"]["n_spoof"] == 2


# --- Task 1 / Task 2: subset and protocol -----------------------------------------------------------
def test_select_subset_keeps_every_bonafide_and_caps_spoof():
    labels = np.array([BONAFIDE] * 10 + [SPOOF] * 200)
    idx = select_subset(labels, spoof_per_bonafide=4, seed=0)
    assert (labels[idx] == BONAFIDE).sum() == 10 and (labels[idx] == SPOOF).sum() == 40
    assert len(set(idx.tolist())) == len(idx)
    capped = select_subset(np.array([BONAFIDE] * 10 + [SPOOF] * 5), spoof_per_bonafide=4)
    assert (np.array([BONAFIDE] * 10 + [SPOOF] * 5)[capped] == SPOOF).sum() == 5  # never more spoof than exist


def test_protocol_roundtrip_and_verification(tmp_path):
    clips = make_synthetic_asvspoof(60)
    path = write_synthetic_protocol(clips, tmp_path / "proto.txt")
    protocol, _ = load_eval_protocol(path=path)
    utt = [c["utterance_id"] for c in clips]
    labels = [c["label"] for c in clips]
    speakers = [c["speaker_id"] for c in clips]
    ok = verify_protocol(protocol, utt, labels, speakers)
    assert ok["usable"] and ok["label_agreement"] == 60
    assert protocol[utt[1]]["attack"] == clips[1]["attack"] and protocol[utt[0]]["attack"] == "bonafide"

    flipped = list(labels)
    flipped[3] = BONAFIDE if flipped[3] == SPOOF else SPOOF
    assert not verify_protocol(protocol, utt, flipped, speakers)["usable"], "a label mismatch must fail verification"
    assert not verify_protocol(protocol, utt + ["MISSING"], labels + [SPOOF], speakers + ["X"])["usable"]


def test_attack_disjoint_split_holds_attacks_out():
    clips = make_synthetic_asvspoof(120)
    attacks = [c["attack"] for c in clips]
    held = choose_held_out_attacks(attacks, 2, seed=0)
    assert len(held) == 2 and "bonafide" not in held
    everything = list(range(len(clips)))
    train = drop_attacks(everything, attacks, held)
    test = keep_only_attacks(everything, attacks, held)
    assert not ({attacks[i] for i in train} & set(held))
    assert {attacks[i] for i in test} == set(held) | {"bonafide"}
    bonafide = {i for i in everything if attacks[i] == "bonafide"}
    assert bonafide <= set(train) and bonafide <= set(test)


# --- Preprocessing: batched LFCC, ablations, augmentation -------------------------------------------
def test_batched_lfcc_matches_single_clip_path():
    clips = make_synthetic_asvspoof(3)
    pre = VoicePreprocessor(clip_seconds=1.0)
    waves = [pre.prepare_waveform(c["waveform"], c["sample_rate"])[0] for c in clips]
    single = torch.stack([pre.lfcc(w) for w in waves])
    batched = pre.extractor(torch.from_numpy(np.stack(waves)))
    assert torch.allclose(single, batched, atol=1e-4)


def test_silence_ablations_remove_what_they_claim():
    sr = 16000
    speech = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    room_tone = (0.003 * np.random.default_rng(0).standard_normal(sr // 2)).astype(np.float32)
    wave = np.concatenate([room_tone, speech, room_tone[: sr // 4]])
    _, stats = trim_silence(wave, sr)

    zeros_orig = ablate_silence(wave, sr, "zeros_original_duration", stats)
    assert len(zeros_orig) == len(wave)  # duration kept...
    assert np.all(zeros_orig[: stats["start_sample"]] == 0)  # ...content removed
    fixed_a = ablate_silence(wave, sr, "zeros_fixed", stats, fixed_seconds=0.5)
    assert len(fixed_a) == stats["end_sample"] - stats["start_sample"] + sr  # 0.5 s each side
    noisy = ablate_silence(wave, sr, "noise_fixed", stats, fixed_seconds=0.5, noise_dbfs=-60, rng=np.random.default_rng(1))
    assert abs(20 * np.log10(np.sqrt(np.mean(noisy[: sr // 2] ** 2))) + 60) < 1.0

    window = silence_window(wave, stats, 3200)
    assert window is not None and len(window) == 3200 and np.allclose(window, wave[:3200])
    assert silence_window(speech, trim_silence(speech, sr)[1], 3200) is None  # no silence -> excluded
    with pytest.raises(ValueError):
        ablate_silence(wave, sr, "nonsense", stats)


def test_augmentation_changes_training_inputs_only():
    extractor = LFCCExtractor()
    waves = np.stack([to16 for to16 in (np.sin(np.linspace(0, 200, 16000)) * 0.3 * INT16_SCALE).astype(np.int16)[None].repeat(4, 0)])
    fetch = wave_lfcc_source(waves, extractor, torch.device("cpu"), augment=True, seed=0)
    clean_a, clean_b = fetch([0, 1], False), fetch([0, 1], False)
    assert torch.equal(clean_a, clean_b), "evaluation inputs must be deterministic and unaugmented"
    augmented = fetch([0, 1], True)
    assert not torch.allclose(augmented, clean_a), "training inputs should be augmented"
    plain = wave_lfcc_source(waves, extractor, torch.device("cpu"), augment=False)
    assert torch.equal(plain([0, 1], True), clean_a), "augment=False must ignore the training flag"

    g = torch.Generator().manual_seed(0)
    w = torch.randn(3, 16000) * 0.1
    assert augment_waveforms(w, g).shape == w.shape
    masked = spec_augment(g, n_freq_masks=1, max_freq_width=5)(torch.randn(3, 70, 50))
    assert masked.shape == (3, 70, 50)


# --- Feature cache ----------------------------------------------------------------------------------
def test_feature_cache_builds_consistently_and_reuses(tmp_path):
    clips = make_synthetic_asvspoof(24)
    extractor = LFCCExtractor()
    kwargs = dict(n_clips=len(clips), cache_dir=tmp_path, extractor=extractor, device=torch.device("cpu"),
                  clip_samples=16000, silence_samples=1600, batch_size=7, config_key="t", progress_every=None)
    cache = build_feature_cache(iter(clips), **kwargs)
    assert cache["wave_trimmed"].shape == (24, 16000)
    assert cache["lfcc"]["untrimmed"].shape[0] == 24 and len(cache["stats"]) == 24
    assert cache["has_silence_window"].any()

    # Cached untrimmed LFCC == LFCC computed now from the same (quantized) waveform.
    pre = VoicePreprocessor(clip_seconds=1.0, trim=False)
    fixed, _ = pre.prepare_waveform(clips[5]["waveform"], clips[5]["sample_rate"])
    fresh = pre.lfcc(quantize(fixed)).numpy()
    assert np.allclose(cache["lfcc"]["untrimmed"][5].astype(np.float32), fresh, atol=0.05)

    # The waveform source reproduces the trimmed pipeline (on the same int16-quantized signal -- every
    # cached variant is quantized identically, so comparisons between variants stay like-for-like).
    trimmer = VoicePreprocessor(clip_seconds=1.0)
    trimmed_wave, _ = trimmer.prepare_waveform(clips[5]["waveform"], clips[5]["sample_rate"])
    via_source = wave_lfcc_source(cache["wave_trimmed"], extractor, torch.device("cpu"))([5], False)[0, 0]
    assert torch.allclose(via_source, trimmer.lfcc(quantize(trimmed_wave)), atol=1e-3)

    reused = build_feature_cache(iter([]), **kwargs)  # empty iterator: must not need to decode again
    assert np.array_equal(np.asarray(reused["wave_trimmed"]), np.asarray(cache["wave_trimmed"]))


def test_train_model_on_memmap_sources(tmp_path):
    torch.manual_seed(0)
    clips = make_synthetic_asvspoof(48)
    extractor = LFCCExtractor()
    cache = build_feature_cache(iter(clips), n_clips=48, cache_dir=tmp_path, extractor=extractor,
                                device=torch.device("cpu"), clip_samples=16000, silence_samples=1600,
                                config_key="m", progress_every=None)
    labels = np.array([c["label"] for c in clips])
    train, val = np.arange(0, 36), np.arange(36, 48)
    for fetch in (wave_lfcc_source(cache["wave_trimmed"], extractor, torch.device("cpu"), augment=True),
                  lfcc_array_source(cache["lfcc"]["untrimmed"], torch.device("cpu"))):
        model = LCNN(n_features=60, n_frames=extractor.n_frames(16000), channels=(8, 8, 12, 12), fc_dim=16)
        fit = train_model(model, fetch, labels, train, 3.0, torch.device("cpu"), val_idx=val, epochs=2,
                          batch_size=8, patience=None, verbose=False)
        assert fit["validation_usable"] and len(fit["history"]) == 2
        assert np.isfinite(score_model(model, fetch, val)).all()


# --- In-the-Wild archive loader (local zip, same fsspec code path as hf://) -------------------------
def test_archive_loader_reads_labels_from_meta_csv_and_records_failures(tmp_path):
    pytest.importorskip("soundfile")
    zip_path = tmp_path / "release_in_the_wild.zip"
    write_synthetic_archive(zip_path, n_real=4, n_fake=5)
    archive = ZipAudioArchive(str(zip_path), protocol="file")
    clips, report = load_archive_clips_balanced(archive, n_per_class=3, seed=0)
    assert report["label_column"] == "label" and report["file_column"] == "file"
    assert report["loaded_bonafide"] == 3 and report["loaded_spoof"] == 3
    assert {c["raw_label"] for c in clips} == {"bona-fide", "spoof"}
    assert all(c["waveform"].dtype == np.float32 and c["sample_rate"] == 16000 for c in clips)

    clips_all, report_all = load_archive_clips_balanced(archive, n_per_class=10, seed=0)
    assert report_all["failed"] == 1 and "broken.wav" in report_all["failures"][0]["member"]
    assert report_all["loaded_spoof"] == 5  # every loadable spoof, the broken one recorded, not hidden


# --- SSL front-end ----------------------------------------------------------------------------------
def test_ssl_encoder_and_head_on_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, build_ssl_cache, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(0).standard_normal((5, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)
    feats = build_ssl_cache(encoder, waves, tmp_path / "ssl.npy", batch_size=2, progress_every=None)
    assert feats.shape == (5, encoder.n_layers, encoder.feature_dim) == (5, 3, 64)
    assert np.isfinite(np.asarray(feats, dtype=np.float32)).all()

    head = SSLHead(encoder.n_layers, encoder.feature_dim, hidden=8)
    out = head(torch.from_numpy(np.asarray(feats, dtype=np.float32)))
    assert out.shape == (5,)
    weights = head.layer_weights()
    assert len(weights) == 3 and abs(sum(weights) - 1) < 1e-6


def test_compute_eer_unchanged_by_refactor():
    assert compute_eer(np.array([9.0, 8.0, 1.0, 2.0]), np.array([1, 1, 0, 0]))["eer"] == 0.0
    assert bootstrap_eer_ci(np.array([9.0, 8.0, 1.0, 2.0] * 10), np.array([1, 1, 0, 0] * 10), n_resamples=20)["eer"] == 0.0
