"""Voice scoring: frozen wav2vec2 XLS-R 300M + a small trained head (Phase 2B/2C).

Model D (mean-only pooling, 133,402 head params, EER 0.0166) is the intended default. Model C
(mean+std pooling, 266,522 params) is the documented fallback and is what is served whenever D's
weights file is not present -- the response always says which model actually produced the score.

The preprocessing mirrors how the training cache was built: resample to 16 kHz, energy-trim leading
and trailing silence, crop / repeat-pad to 4 s, round-trip through int16, then the encoder's own
per-clip normalisation and per-layer mean (+ std) pooling.
"""
import io
import json
import threading
import time

import numpy as np
import soundfile as sf
import torch

from engines.voice.features import quantize
from engines.voice.model import SSLHead
from engines.voice.preprocess import TARGET_SR, fix_length, resample_linear, trim_silence
from engines.voice.ssl import SSLEncoder

from . import config

CLIP_SECONDS = 4.0
MODELS = {  # name -> (weights glob, preferred file, pooling)
    "D": ("phase2c_D_ssl_mean_only*.pt", "phase2c_D_ssl_mean_only_P1_s42.pt", "mean"),
    "C": ("phase2b_C_ssl_P1_s*.pt", "phase2b_C_ssl_P1_s42.pt", "mean+std"),
}
UNCALIBRATED = {"high": 0.5, "medium": 0.2,
                "source": "uncalibrated default (no held-out test scores for this model are available locally)"}
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
# Input checks added after the self-test (2026-10-08). They guard the service, they are not part of the model.
SILENCE_PEAK = 1e-4       # below this peak amplitude (about -80 dBFS) the recording is treated as silent
MIN_RELIABLE_SOUND_S = 1.0  # less sound than this after trimming: scored, but flagged as low confidence


def decode_audio(blob, max_seconds=None):
    """Decode with libsndfile. Anything it cannot parse as audio raises (the API turns that into a 422).
    The duration is read from the header first, so an over-long file is refused before it is decoded."""
    if not blob:
        raise ValueError("the file is empty")
    try:
        handle = sf.SoundFile(io.BytesIO(blob))
    except sf.LibsndfileError:
        raise ValueError("the file is not audio this service can read (use WAV, FLAC, OGG or MP3)") from None
    with handle as f:
        seconds = f.frames / f.samplerate
        if max_seconds is not None and seconds > max_seconds:
            raise ValueError(f"audio is {seconds:.0f} s long; the limit is {max_seconds:.0f} s")
    wave, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    return wave, sr


def prepare_clip(wave, sr):
    """Raw waveform -> the fixed 4 s, 16 kHz, int16-quantised clip the SSL encoder was trained on."""
    wave, sr = resample_linear(np.asarray(wave, dtype=np.float32), sr, TARGET_SR)
    if len(wave) < int(0.1 * sr):
        raise ValueError("audio clip is shorter than 0.1 s")
    trimmed, stats = trim_silence(wave, sr, top_db=30.0)
    return quantize(fix_length(trimmed, int(TARGET_SR * CLIP_SECONDS))), stats


def find_weights(name, artifacts_dir=None):
    glob, preferred, _ = MODELS[name]
    folder = (artifacts_dir or config.ARTIFACTS_DIR) / "voice"
    if (folder / preferred).exists():
        return folder / preferred
    found = sorted(folder.glob(glob))
    return found[0] if found else None


class VoiceService:
    def __init__(self, artifacts_dir=None):
        self._lock = threading.Lock()
        self.artifacts_dir = artifacts_dir
        self.encoder = None
        self.heads = {}
        self.thresholds = {}
        self.available = {n: find_weights(n, artifacts_dir) for n in MODELS}
        self.default_model = "D" if self.available["D"] else ("C" if self.available["C"] else None)
        self.requested_default = "D"
        self.ready = False
        self.error = None

    def load(self, encoder=None):
        """Loads the SSL encoder (~1.2 GB download on first run) and every available head."""
        t0 = time.time()
        if self.default_model is None:
            raise FileNotFoundError("no voice head weights found (neither Model D nor Model C)")
        self.encoder = encoder or SSLEncoder(config.SSL_MODEL, device=torch.device("cpu"), fp16=False)
        for name, path in self.available.items():
            if path is None:
                continue
            feat_dim = self.encoder.hidden_size * (1 if MODELS[name][2] == "mean" else 2)
            head = SSLHead(self.encoder.n_layers, feat_dim)
            head.load_state_dict(torch.load(path, map_location="cpu"))
            self.heads[name] = head.eval()
        if config.VOICE_THRESHOLDS.exists():
            self.thresholds = json.loads(config.VOICE_THRESHOLDS.read_text())
        self.ready = True
        self.load_seconds = round(time.time() - t0, 1)

    def info(self):
        return {"requested_default": self.requested_default, "serving_default": self.default_model,
                "weights": {n: (p.name if p else None) for n, p in self.available.items()},
                "fallback_active": self.default_model != self.requested_default,
                "thresholds": {n: self.thresholds.get(n, UNCALIBRATED) for n in self.heads}}

    def _threshold(self, name):
        return self.thresholds.get(name, UNCALIBRATED)

    def verdict(self, spoof_prob, name):
        t = self._threshold(name)
        return "high" if spoof_prob >= t["high"] else "medium" if spoof_prob >= t["medium"] else "low"

    @torch.no_grad()
    def score_waveform(self, wave, sr, model=None):
        name = model or self.default_model
        if name not in self.heads:
            raise KeyError(f"voice model {name!r} is not loaded (loaded: {sorted(self.heads)})")
        # A float WAV can carry NaN/inf samples; they made trimming report 0 s of sound and the head return a
        # meaningless score. Refuse them (the API turns this into a 422). Valid audio is untouched.
        if len(wave) and not np.all(np.isfinite(wave)):
            raise ValueError("the recording contains invalid sample values (NaN or infinity)")
        if len(wave) == 0 or float(np.max(np.abs(wave))) < SILENCE_PEAK:
            raise ValueError("the recording is silent, so there is nothing to score")
        clip, stats = prepare_clip(wave, sr)
        warnings = []
        if stats["speech_s"] < MIN_RELIABLE_SOUND_S:
            warnings.append(f"Only {stats['speech_s']:.1f} s of sound after trimming silence. "
                            f"That is too little for a reliable result (at least {MIN_RELIABLE_SOUND_S:.0f} s is needed).")
        with self._lock:
            pooled = self.encoder.pooled(torch.from_numpy(clip).unsqueeze(0))  # [1, layers, 2*hidden]
            feats = self.encoder.mean_only(pooled) if MODELS[name][2] == "mean" else pooled
            bonafide_logit = float(self.heads[name](feats)[0])
        # Training label convention: 1 = bonafide (the EER target class), so the head's logit is
        # evidence for *bonafide*; the risk score is the complementary spoof probability.
        spoof_prob = float(1.0 / (1.0 + np.exp(bonafide_logit)))
        threshold = self._threshold(name)
        return {
            "score": spoof_prob, "verdict": self.verdict(spoof_prob, name),
            "low_confidence": bool(warnings), "warnings": warnings,
            "model": name, "model_requested": model or self.requested_default,
            "fallback_used": name != (model or self.requested_default),
            "bonafide_logit": bonafide_logit,
            "thresholds": {"medium": threshold["medium"], "high": threshold["high"],
                           "source": threshold.get("source", "")},
            "duration_s": round(len(wave) / sr, 2), "speech_s": round(stats["speech_s"], 2),
            "lead_silence_s": round(stats["lead_silence_s"], 3), "trail_silence_s": round(stats["trail_silence_s"], 3),
            "scored_seconds": CLIP_SECONDS,
        }

    def score_bytes(self, blob, model=None):
        wave, sr = decode_audio(blob, config.MAX_AUDIO_SECONDS)
        return self.score_waveform(wave, sr, model)

    def list_clips(self):
        folder = config.DEMO_CLIPS_DIR
        if not folder.exists():
            return []
        manifest = {}
        if (folder / "manifest.json").exists():
            manifest = {m["file"]: m for m in json.loads((folder / "manifest.json").read_text())}
        return [{"clip": p.name, **{k: v for k, v in manifest.get(p.name, {}).items() if k != "file"}}
                for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_SUFFIXES]

    def clip_path(self, clip):
        """Resolve a demo-clip name, refusing anything that would escape the demo-clips folder."""
        path = (config.DEMO_CLIPS_DIR / clip).resolve()
        if path.parent != config.DEMO_CLIPS_DIR.resolve() or not path.is_file():
            raise FileNotFoundError(f"no demo clip named {clip!r}")
        return path
