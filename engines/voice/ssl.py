"""Frozen self-supervised speech encoder (wav2vec 2.0 / XLS-R family) as a front-end (Phase 2B Run C).

The encoder is frozen and run once over every clip. Each layer's frame sequence is pooled to its
mean and standard deviation and cached, so the classifier (`model.SSLHead`, a learned softmax weight
per layer + a small MLP) trains in seconds and extra seeds cost almost nothing. Fine-tuning the
encoder is deliberately not done in this phase.
"""
from pathlib import Path

import numpy as np
import torch

from .features import INT16_SCALE


class SSLEncoder:
    def __init__(self, model_name=None, device=torch.device("cpu"), fp16=True, model=None):
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.half = bool(fp16) and device.type == "cuda"
        self.model = model.to(device).eval()
        if self.half:
            self.model.half()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model_name = model_name or type(model).__name__
        self.n_layers = self.model.config.num_hidden_layers + 1  # + the CNN feature-projection output
        self.hidden_size = self.model.config.hidden_size

    @property
    def feature_dim(self):
        return 2 * self.hidden_size  # mean and std pooling

    def _normalize(self, waves):
        x = waves.to(self.device, torch.float32)
        return (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-7)

    @torch.no_grad()
    def pooled(self, waves):
        """[batch, samples] float waveforms -> [batch, layers, 2*hidden] pooled features.

        Inputs are normalized to zero mean / unit variance per clip, which is what the wav2vec 2.0 /
        XLS-R feature extractors do (`do_normalize=True`).
        """
        x = self._normalize(waves)
        out = self.model(x.half() if self.half else x, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=1).float()  # [batch, layers, frames, hidden]
        return torch.cat([hidden.mean(dim=2), hidden.std(dim=2)], dim=-1)

    def mean_only(self, pooled_features):
        """[*, 2*hidden] concatenated mean+std pooling -> [*, hidden] mean-only half (Phase 2C Model D:
        an ablation of `pooled`'s std half, not a new encoder pass -- see `build_ssl_cache`'s cache)."""
        return pooled_features[..., :self.hidden_size]

    @torch.no_grad()
    def sequence(self, waves, layer):
        """[batch, samples] -> [batch, frames, hidden] per-frame hidden states of one layer index
        (0 = the CNN feature-projection output, same indexing as `hidden_states`). Phase 2C Model E
        needs the full frame sequence, not the pooled summary `pooled` gives; caching every layer's
        full sequence is infeasible (see the notebook's cost note), so only one layer is kept."""
        x = self._normalize(waves)
        out = self.model(x.half() if self.half else x, output_hidden_states=True)
        return out.hidden_states[layer].float()


def build_ssl_cache(encoder, waves_int16, cache_path, batch_size=16, progress_every=2000):
    """Pooled features for every clip in `waves_int16` -> float16 memmap [N, layers, 2*hidden].
    Reuses an existing file of the right shape (e.g. after a kernel restart)."""
    from numpy.lib.format import open_memmap

    cache_path = Path(cache_path)
    shape = (len(waves_int16), encoder.n_layers, encoder.feature_dim)
    done_marker = cache_path.with_suffix(".done")
    if cache_path.exists() and done_marker.exists():
        existing = np.load(cache_path, mmap_mode="r")
        if existing.shape == shape:
            print(f"reusing SSL feature cache {cache_path}")
            return existing
    out = open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
    for start in range(0, len(waves_int16), batch_size):
        batch = torch.from_numpy(np.asarray(waves_int16[start:start + batch_size], dtype=np.float32) / INT16_SCALE)
        out[start:start + len(batch)] = encoder.pooled(batch).cpu().numpy().astype(np.float16)
        done = start + len(batch)
        if progress_every and (done // batch_size) % max(1, progress_every // batch_size) == 0:
            print(f"  SSL features {done}/{len(waves_int16)}")
    out.flush()
    done_marker.write_text("ok", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def build_ssl_sequence_cache(encoder, waves_int16, layer, cache_path, batch_size=16, progress_every=2000):
    """Per-frame hidden states of one layer for every clip -> float16 memmap [N, frames, hidden]
    (Phase 2C Model E). Frame count is fixed because every input clip is the same fixed sample count.
    Reuses an existing file of the right shape, same as `build_ssl_cache`."""
    from numpy.lib.format import open_memmap

    cache_path = Path(cache_path)
    done_marker = cache_path.with_suffix(".done")
    with torch.no_grad():
        probe = encoder.sequence(torch.from_numpy(np.asarray(waves_int16[:1], dtype=np.float32) / INT16_SCALE), layer)
    n_frames = probe.shape[1]
    shape = (len(waves_int16), n_frames, encoder.hidden_size)
    if cache_path.exists() and done_marker.exists():
        existing = np.load(cache_path, mmap_mode="r")
        if existing.shape == shape:
            print(f"reusing SSL sequence cache {cache_path}")
            return existing
    out = open_memmap(cache_path, mode="w+", dtype=np.float16, shape=shape)
    for start in range(0, len(waves_int16), batch_size):
        batch = torch.from_numpy(np.asarray(waves_int16[start:start + batch_size], dtype=np.float32) / INT16_SCALE)
        out[start:start + len(batch)] = encoder.sequence(batch, layer).cpu().numpy().astype(np.float16)
        done = start + len(batch)
        if progress_every and (done // batch_size) % max(1, progress_every // batch_size) == 0:
            print(f"  SSL sequence features {done}/{len(waves_int16)}")
    out.flush()
    done_marker.write_text("ok", encoding="utf-8")
    return np.load(cache_path, mmap_mode="r")


def tiny_wav2vec2_for_smoke():
    """A randomly initialised, tiny wav2vec 2.0 -- exercises the real transformers code path in the
    local smoke test without downloading a checkpoint. Never used for reported numbers."""
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
    torch.manual_seed(0)
    return Wav2Vec2Model(config)
