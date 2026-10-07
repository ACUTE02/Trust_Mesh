"""CPU tests for the Phase 2C voice additions (docs/PHASE_2C_VOICE_SSL_DEPTH.md). Synthetic data only;
no reported numbers.

Run: .venv\\Scripts\\python.exe -m pytest tests/test_voice_phase2c.py
"""
import numpy as np
import pytest
import torch

from engines.voice.features import (INT16_SCALE, array_mean_only_source, array_source, wave_ssl_source,
                                    wave_ssl_sequence_source)
from engines.voice.model import SSLHead, SSLSeqHead, score_model, train_model
from engines.voice.preprocess import augment_waveforms, simulate_low_bitrate


# --- Task 0 pre-flight: Model C's own pooling ---------------------------------------------------------
def test_pooled_is_mean_std_concat_and_mean_only_slices_it():
    """The pre-flight finding this phase is built on: `pooled()` already concatenates mean and std, so
    `mean_only()` must be an exact slice of it, not a recomputation."""
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = torch.randn(4, 16000) * 0.1
    pooled = encoder.pooled(waves)
    assert pooled.shape == (4, encoder.n_layers, 2 * encoder.hidden_size)
    mean_half, std_half = pooled[..., :encoder.hidden_size], pooled[..., encoder.hidden_size:]
    assert torch.equal(encoder.mean_only(pooled), mean_half)
    assert not torch.allclose(mean_half, std_half)  # sanity: the two halves aren't accidentally identical


# --- Task 1: Model D (mean-only ablation) reuses the cache, no new encoder pass ------------------------
def test_array_mean_only_source_matches_manual_slice_and_costs_no_recompute(tmp_path):
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, build_ssl_cache, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(0).standard_normal((6, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)
    pooled = build_ssl_cache(encoder, waves, tmp_path / "ssl.npy", batch_size=2, progress_every=None)

    fetch = array_mean_only_source(pooled, encoder.hidden_size, torch.device("cpu"))
    x = fetch([0, 2, 4], False)
    assert x.shape == (3, encoder.n_layers, encoder.hidden_size)
    manual = torch.from_numpy(np.asarray(pooled[[0, 2, 4]], dtype=np.float32))[..., :encoder.hidden_size]
    assert torch.equal(x, manual)

    # Model D is just an SSLHead sized for the mean-only half -- trains and scores like any SSLHead.
    model = SSLHead(encoder.n_layers, encoder.hidden_size, hidden=8)
    labels = np.array([0, 1, 0, 1, 0, 1])
    fit = train_model(model, fetch, labels, np.array([0, 1, 2, 3]), 1.0, torch.device("cpu"),
                      val_idx=np.array([4, 5]), epochs=1, batch_size=2, patience=None, verbose=False)
    assert fit["epochs_run"] == 1
    assert np.isfinite(score_model(model, fetch, [0, 1])).all()


# --- Task 2: Model E (BiLSTM over one SSL layer's per-frame sequence) ----------------------------------
def test_ssl_sequence_cache_shape_and_reuse(tmp_path):
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, build_ssl_sequence_cache, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(1).standard_normal((5, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)
    seq = build_ssl_sequence_cache(encoder, waves, layer=1, cache_path=tmp_path / "seq.npy",
                                   batch_size=2, progress_every=None)
    assert seq.shape[0] == 5 and seq.shape[2] == encoder.hidden_size
    assert np.isfinite(np.asarray(seq, dtype=np.float32)).all()

    reused = build_ssl_sequence_cache(encoder, waves, layer=1, cache_path=tmp_path / "seq.npy",
                                      batch_size=2, progress_every=None)
    assert np.array_equal(np.asarray(reused), np.asarray(seq))


def test_ssl_seq_head_trains_on_the_sequence_cache_and_reports_param_count():
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, build_ssl_sequence_cache, tiny_wav2vec2_for_smoke
    import tempfile
    from pathlib import Path

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(2).standard_normal((8, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)
    tmp = Path(tempfile.mkdtemp())
    seq = build_ssl_sequence_cache(encoder, waves, layer=0, cache_path=tmp / "seq.npy", batch_size=4,
                                   progress_every=None)
    fetch = array_source(seq, torch.device("cpu"))

    model = SSLSeqHead(encoder.hidden_size, lstm_hidden=4, hidden=8)
    labels = np.array([0, 1] * 4)
    fit = train_model(model, fetch, labels, np.arange(6), 1.0, torch.device("cpu"), val_idx=np.array([6, 7]),
                      epochs=1, batch_size=3, patience=None, verbose=False)
    assert fit["epochs_run"] == 1
    scores = score_model(model, fetch, [0, 1, 2])
    assert np.isfinite(scores).all() and scores.shape == (3,)
    assert model.n_params() > 0
    # No per-layer weighting mechanism on a single-layer input -- SSLSeqHead has no layer_weights method.
    assert not hasattr(model, "layer_weights")


# --- Task 2, disk fix (issue #25): Model E reads its sequence on the fly, never a disk cache -----------
def test_wave_ssl_sequence_source_matches_direct_encoder_call_and_writes_no_cache_file(tmp_path):
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(4).standard_normal((5, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)

    fetch = wave_ssl_sequence_source(waves, encoder, layer=1, device=torch.device("cpu"))
    out = fetch([0, 2], False)
    direct = encoder.sequence(torch.from_numpy(waves[[0, 2]].astype(np.float32) / INT16_SCALE), 1)
    assert torch.allclose(out, direct)
    assert out.shape[0] == 2 and out.shape[2] == encoder.hidden_size
    # No files created anywhere -- this is the whole point of the on-the-fly path (issue #25).
    assert list(tmp_path.iterdir()) == []

    model = SSLSeqHead(encoder.hidden_size, lstm_hidden=4, hidden=8)
    scores = score_model(model, fetch, [0, 1, 2])
    assert np.isfinite(scores).all() and scores.shape == (3,)


# --- Task 3: Model F -- augmentation must happen before the frozen encoder's forward pass --------------
def test_wave_ssl_source_augments_only_when_training_and_is_deterministic_at_eval():
    pytest.importorskip("transformers")
    from engines.voice.ssl import SSLEncoder, tiny_wav2vec2_for_smoke

    encoder = SSLEncoder(model=tiny_wav2vec2_for_smoke(), device=torch.device("cpu"), fp16=False)
    waves = (np.random.default_rng(3).standard_normal((4, 16000)) * 0.1 * INT16_SCALE).astype(np.int16)

    fetch = wave_ssl_source(waves, encoder, torch.device("cpu"), augment=True, seed=0, codec_sim=True)
    eval_a, eval_b = fetch([0, 1], False), fetch([0, 1], False)
    assert torch.equal(eval_a, eval_b), "eval-time features must be deterministic and unaugmented"
    train_a = fetch([0, 1], True)
    assert not torch.allclose(train_a, eval_a), "training-time features should be augmented"
    assert train_a.shape == eval_a.shape == (2, encoder.n_layers, encoder.feature_dim)

    plain = wave_ssl_source(waves, encoder, torch.device("cpu"), augment=False)
    assert torch.equal(plain([0, 1], True), eval_a), "augment=False must ignore the training flag"


def test_simulate_low_bitrate_shape_range_and_codec_sim_flag():
    g = torch.Generator().manual_seed(0)
    waves = torch.randn(6, 16000) * 0.1
    out = simulate_low_bitrate(waves, g, p_apply=1.0)  # force every clip through the proxy codec
    assert out.shape == waves.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, waves), "the low-bitrate proxy should audibly change every clip when p_apply=1"

    g2 = torch.Generator().manual_seed(0)
    with_codec = augment_waveforms(waves.clone(), g2, codec_sim=True)
    g3 = torch.Generator().manual_seed(0)
    without_codec = augment_waveforms(waves.clone(), g3, codec_sim=False)
    assert not torch.allclose(with_codec, without_codec), "codec_sim=True must change the augmented output"
