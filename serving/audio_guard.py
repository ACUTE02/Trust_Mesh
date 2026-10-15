"""Audio-quality guard: two simple, explainable checks that run before the voice model sees a clip.

The model was trained on clean studio recordings. Two kinds of input make its answer meaningless, and
neither is something the model can report about itself:

1. Sound that is not speech (a steady tone, hiss, music-like drone). The model still returns a number.
2. Speech that was recorded or compressed so that its high frequencies are gone (phone audio, voice memos,
   telephone codecs). The model tends to call such genuine speech "fake".

The checks do NOT change the score or the risk thresholds. The first one refuses, the second adds a warning.
Both are heuristics fixed once from a handful of local clips (see scripts/calibrate_audio_guard.py and
docs/design_decisions.md, "Audio-quality guard"). They are not validated detectors.

Everything here works on the 16 kHz mono clip after leading and trailing silence has been trimmed.

Measures
--------
spectral movement (speech check)
    Speech keeps changing the shape of its sound: vowels, consonants and pauses move the spectral centroid
    around. A steady tone or a hiss does not. Measured as the standard deviation of the spectral centroid
    (100 to 7500 Hz) over the loud frames, divided by its mean. 25 ms frames, 10 ms hop; loud frames are those
    within 25 dB of the loudest.
high-frequency level (band-limit check, rule 1)
    Median over the loud frames of the energy at 5000 to 7900 Hz relative to the energy at 300 to 3400 Hz, in dB.
    Phone-band and phone-recorded speech has far less sound up there than studio speech.
cut-off frequency (band-limit check, rule 2)
    The highest frequency at which the average spectrum of the loud frames is within 55 dB of its level at
    300 to 3400 Hz. A hard low-pass (telephone codecs, some MP3 encoders) shows up as a cut-off well below 8 kHz.
recording sample rate (band-limit check, rule 3)
    A file recorded below 16 kHz (8 kHz telephony is the common case) cannot hold anything above half its sample
    rate. The service up-samples it by simple interpolation, which fills the missing band with mirror images, so
    rules 1 and 2 would not see it. The file's own rate is used instead.
"""
import numpy as np

SAMPLE_RATE = 16000
FRAME = 400               # 25 ms
HOP = 160                 # 10 ms
ACTIVE_WITHIN_DB = 25.0   # a frame is "loud" if it is within this many dB of the loudest frame
MIN_ACTIVE_FRAMES = 5     # 50 ms of loud sound; fewer than this cannot be a word

# Thresholds. Each was chosen once, from the local clips listed in docs/design_decisions.md, and not tuned after.
SPEECH_MOVEMENT_MIN = 0.30      # real speech >= 0.477 (lowest of 84 clean, compressed and owner clips); synthetic
                                # non-speech <= 0.159 (a frequency sweep, 0.538, is the known exception)
HF_TO_MID_MIN_DB = -55.0        # 13 clean clips: -52.0 to -28.9; the threshold is 3 dB below the lowest
CUTOFF_MIN_HZ = 7600.0          # 13 clean clips: 7880 to 8000 Hz; the threshold is 280 Hz below the lowest
CUTOFF_DROP_DB = 55.0
MIN_NATIVE_RATE_HZ = 16000      # below this the recording itself has no sound above 8 kHz

NOT_SPEECH_MESSAGE = (
    "This does not look like speech, so nothing was scored. The sound does not change the way speech does "
    "(a steady tone, hiss or very short sound looks like this). Record a few seconds of someone talking."
)
LOW_QUALITY_MESSAGE = (
    "This audio looks phone-quality or heavily compressed. This model was trained on clean recordings, "
    "so the result is unreliable."
)


class NotSpeechError(ValueError):
    """The clip has no speech-like sound. The API turns this into a 422 with the message below."""


def _loud_frames(x):
    """Power spectra of the loud frames, and the frequency of each bin. Returns (S, freqs) or (None, freqs)."""
    freqs = np.fft.rfftfreq(FRAME, 1.0 / SAMPLE_RATE)
    if len(x) < FRAME:
        return None, freqs
    windows = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP] * np.hanning(FRAME)
    power = np.abs(np.fft.rfft(windows, axis=1)) ** 2
    level_db = 10.0 * np.log10(power.sum(axis=1) + 1e-12)
    loud = level_db > level_db.max() - ACTIVE_WITHIN_DB
    return power[loud], freqs


def analyse(wave):
    """The three measures for one trimmed 16 kHz mono clip (float samples). Pure NumPy, deterministic."""
    x = np.asarray(wave, dtype=np.float64)
    power, freqs = _loud_frames(x)
    if power is None or len(power) < MIN_ACTIVE_FRAMES:
        return {"analysed": False, "loud_frames": 0 if power is None else int(len(power)),
                "spectral_movement": 0.0, "hf_to_mid_db": None, "cutoff_hz": None}
    band = (freqs >= 100) & (freqs <= 7500)
    centroid = (power[:, band] * freqs[band]).sum(axis=1) / (power[:, band].sum(axis=1) + 1e-12)
    movement = float(np.std(centroid) / (np.mean(centroid) + 1e-9))

    mid = power[:, (freqs >= 300) & (freqs <= 3400)].sum(axis=1)
    high = power[:, (freqs >= 5000) & (freqs <= 7900)].sum(axis=1)
    hf_to_mid_db = float(np.median(10.0 * np.log10((high + 1e-12) / (mid + 1e-12))))

    mean_db = 10.0 * np.log10(power.mean(axis=0) + 1e-12)
    reference = np.median(mean_db[(freqs >= 300) & (freqs <= 3400)])
    above = np.flatnonzero(mean_db > reference - CUTOFF_DROP_DB)
    cutoff_hz = float(freqs[above.max()]) if len(above) else 0.0
    return {"analysed": True, "loud_frames": int(len(power)), "spectral_movement": movement,
            "hf_to_mid_db": hf_to_mid_db, "cutoff_hz": cutoff_hz}


def check(wave, native_rate=None):
    """Run the guard on a trimmed clip. `native_rate` is the sample rate the file was recorded at (before the
    service resampled it to 16 kHz). Returns the measures, the thresholds and the two verdicts."""
    m = analyse(wave)
    speech_like = bool(m["analysed"] and m["spectral_movement"] >= SPEECH_MOVEMENT_MIN)
    reasons = []
    if speech_like and native_rate is not None and native_rate < MIN_NATIVE_RATE_HZ:
        reasons.append(f"recorded at a low sample rate ({int(native_rate)} Hz)")
    if m["analysed"]:
        if m["hf_to_mid_db"] < HF_TO_MID_MIN_DB:
            reasons.append("very little sound above 5 kHz")
        if m["cutoff_hz"] < CUTOFF_MIN_HZ:
            reasons.append("sound cut off below 7.6 kHz")
    return {**m, "speech_like": speech_like, "low_quality_audio": speech_like and bool(reasons), "reasons": reasons,
            "native_rate_hz": int(native_rate) if native_rate is not None else None,
            "thresholds": {"spectral_movement_min": SPEECH_MOVEMENT_MIN, "hf_to_mid_min_db": HF_TO_MID_MIN_DB,
                           "cutoff_min_hz": CUTOFF_MIN_HZ, "min_native_rate_hz": MIN_NATIVE_RATE_HZ}}
