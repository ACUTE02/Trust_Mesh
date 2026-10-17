"""Voice scoring: frozen wav2vec2 XLS-R 300M + a small trained head (Phase 2B/2C).

Model D (mean-only pooling, 133,402 head params, EER 0.0166) is the intended default. Model C
(mean+std pooling, 266,522 params) is the documented fallback and is what is served whenever D's
weights file is not present -- the response always says which model actually produced the score.

F2 (Phase 2D, trained on clean plus codec-processed copies) is an opt-in extra, reachable only with
?model=F2. It is the mean bonafide logit of its three seed heads (42, 43, 44), which is the object the
Phase 2D result describes. Its weights are private and git-ignored; without them the service starts
as usual and F2 is reported as unavailable. It is never the default and never used by /case/score.

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

from . import audio_guard, config

CLIP_SECONDS = 4.0
MODELS = {  # name -> (weights glob, preferred file, pooling)
    "D": ("phase2c_D_ssl_mean_only*.pt", "phase2c_D_ssl_mean_only_P1_s42.pt", "mean"),
    "C": ("phase2b_C_ssl_P1_s*.pt", "phase2b_C_ssl_P1_s42.pt", "mean+std"),
    "F2": ("phase2d_F2_s{seed}.pt", None, "mean"),  # three files, one per seed; all three are needed
}
F2_SEEDS = (42, 43, 44)
OPT_IN_MODELS = ("F2",)  # never the default, never used by /case/score
MODEL_LABELS = {
    "D": "D (seed 42)",
    "C": "C (seed 42)",
    "F2": "F2 (mean of seeds 42, 43 and 44)",
}
# F2 has no calibration of its own. Its per-clip test scores were not saved, so D's method (bands set on test
# scores) cannot be repeated. What exists is one recorded number: the threshold at the EER on the clean validation
# clips, in artifacts/voice/results_phase2d.json, evaluation.thresholds_clean_val_eer.F2. It is on the scale of the
# seed-averaged bonafide logit (a clip below it is flagged), so it maps exactly to a spoof probability:
# 1 / (1 + exp(0.19378728093336556)). One recorded point means one band: high or low, no medium band.
F2_LOGIT_THRESHOLD = 0.19378728093336556
F2_THRESHOLD = {
    "high": float(1.0 / (1.0 + np.exp(F2_LOGIT_THRESHOLD))), "single_band": True,
    "source": "the clean-validation EER threshold recorded for F2 in results_phase2d.json (seed-averaged score), "
              "mapped to a spoof probability. One operating point, so there is no medium band. Not recalibrated "
              "on test scores and not tuned on any local clips.",
}
F2_THRESHOLD["medium"] = F2_THRESHOLD["high"]
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


def prepare_clip(wave, sr, want_trimmed=False):
    """Raw waveform -> the fixed 4 s, 16 kHz, int16-quantised clip the SSL encoder was trained on.
    With `want_trimmed` the trimmed (not yet cropped or padded) waveform is returned as a third value."""
    wave, sr = resample_linear(np.asarray(wave, dtype=np.float32), sr, TARGET_SR)
    if len(wave) < int(0.1 * sr):
        raise ValueError("audio clip is shorter than 0.1 s")
    trimmed, stats = trim_silence(wave, sr, top_db=30.0)
    clip = quantize(fix_length(trimmed, int(TARGET_SR * CLIP_SECONDS)))
    return (clip, stats, trimmed) if want_trimmed else (clip, stats)


class MeanLogitHeads(torch.nn.Module):
    """Several heads on the same pooled features; the output is the mean of their bonafide logits."""

    def __init__(self, heads):
        super().__init__()
        self.heads = torch.nn.ModuleList(heads)

    def forward(self, x):
        return torch.stack([h(x) for h in self.heads]).mean(dim=0)


def find_weights(name, artifacts_dir=None):
    glob, preferred, _ = MODELS[name]
    folder = (artifacts_dir or config.ARTIFACTS_DIR) / "voice"
    if name == "F2":  # all three seed heads or nothing: a partial set would not be the evaluated model
        paths = [folder / glob.format(seed=s) for s in F2_SEEDS]
        return paths if all(p.exists() for p in paths) else None
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
            if name == "F2":
                members = []
                for p in path:
                    blob = torch.load(p, map_location="cpu")
                    member = SSLHead(blob["n_layers"], blob["hidden"])
                    member.load_state_dict(blob["state_dict"])
                    members.append(member.eval())
                self.heads[name] = MeanLogitHeads(members).eval()
                continue
            head = SSLHead(self.encoder.n_layers, feat_dim)
            head.load_state_dict(torch.load(path, map_location="cpu"))
            self.heads[name] = head.eval()
        if config.VOICE_THRESHOLDS.exists():
            self.thresholds = json.loads(config.VOICE_THRESHOLDS.read_text())
        self.ready = True
        self.load_seconds = round(time.time() - t0, 1)

    def info(self):
        return {"requested_default": self.requested_default, "serving_default": self.default_model,
                "weights": {n: ((p.name if not isinstance(p, list) else ", ".join(q.name for q in p)) if p else None)
                            for n, p in self.available.items()},
                "loaded": sorted(self.heads), "opt_in": {n: n in self.heads for n in OPT_IN_MODELS},
                "model_labels": {n: MODEL_LABELS[n] for n in self.heads},
                "fallback_active": self.default_model != self.requested_default,
                "thresholds": {n: self._threshold(n) for n in self.heads}}

    def _threshold(self, name):
        if name == "F2":
            return F2_THRESHOLD
        return self.thresholds.get(name, UNCALIBRATED)

    def verdict(self, spoof_prob, name):
        t = self._threshold(name)
        if t.get("single_band"):
            return "high" if spoof_prob >= t["high"] else "low"
        return "high" if spoof_prob >= t["high"] else "medium" if spoof_prob >= t["medium"] else "low"

    @torch.no_grad()
    def score_waveform(self, wave, sr, model=None):
        name = model or self.default_model
        if name not in self.heads:
            if name in OPT_IN_MODELS:
                raise KeyError(f"voice model {name} is not available on this server: its weights are private and are "
                               f"not installed. Model {self.default_model} is the default (loaded: {sorted(self.heads)})")
            raise KeyError(f"voice model {name!r} is not loaded (loaded: {sorted(self.heads)})")
        # A float WAV can carry NaN/inf samples; they made trimming report 0 s of sound and the head return a
        # meaningless score. Refuse them (the API turns this into a 422). Valid audio is untouched.
        if len(wave) and not np.all(np.isfinite(wave)):
            raise ValueError("the recording contains invalid sample values (NaN or infinity)")
        if len(wave) == 0 or float(np.max(np.abs(wave))) < SILENCE_PEAK:
            raise ValueError("the recording is silent, so there is nothing to score")
        clip, stats, trimmed = prepare_clip(wave, sr, want_trimmed=True)
        warnings = []
        quality = None
        if config.AUDIO_GUARD:
            # Runs before the model and never changes the score: a clip that is not speech is refused, and
            # phone-quality or band-limited speech is scored but flagged as unreliable.
            quality = audio_guard.check(trimmed, native_rate=sr)
            if not quality["speech_like"]:
                raise audio_guard.NotSpeechError(audio_guard.NOT_SPEECH_MESSAGE)
            if quality["low_quality_audio"]:
                warnings.append(audio_guard.LOW_QUALITY_MESSAGE)
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
            "low_quality_audio": bool(quality and quality["low_quality_audio"]), "audio_quality": quality,
            "model": name, "model_label": MODEL_LABELS[name], "model_requested": model or self.requested_default,
            "fallback_used": name != (model or self.requested_default),
            "bonafide_logit": bonafide_logit,
            "thresholds": {"medium": threshold["medium"], "high": threshold["high"],
                           "source": threshold.get("source", ""), "single_band": bool(threshold.get("single_band"))},
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
