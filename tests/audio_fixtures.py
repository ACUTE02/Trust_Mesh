"""Synthetic audio for the guard tests. Nothing here comes from a dataset.

`speech_like` is a crude formant synthesiser: a voiced pulse train with a gliding pitch, shaped by three
resonances that move from vowel to vowel, with short fricative bursts and pauses between "syllables". It is
not intelligible speech. It is built so that, like speech, the shape of its spectrum keeps changing and it has
sound at high frequencies, which is what the two guard checks look at.
"""
import numpy as np
from scipy.signal import butter, lfilter, sosfilt

SR = 16000

VOWELS = [(700, 1200, 2600), (300, 2300, 3000), (350, 800, 2400), (550, 1700, 2500), (450, 1050, 2700)]


def _resonator(freq, bandwidth):
    r = np.exp(-np.pi * bandwidth / SR)
    theta = 2 * np.pi * freq / SR
    return [1.0 - r], [1.0, -2.0 * r * np.cos(theta), r * r]


def speech_like(seconds=3.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    syllable = 0.22
    for k in range(int(seconds / (syllable + 0.06))):
        n = int(syllable * SR)
        f0 = 100 + 40 * np.sin(2 * np.pi * (k % 5) / 5) + np.linspace(0, 12, n)
        phase = np.cumsum(f0 / SR)
        pulses = (np.diff(np.floor(phase), prepend=0) > 0).astype(np.float64)
        voiced = pulses
        for freq, bw in zip(VOWELS[k % len(VOWELS)], (90, 110, 160)):
            b, a = _resonator(freq, bw)
            voiced = lfilter(b, a, voiced)
        voiced = voiced / (np.abs(voiced).max() + 1e-9) + rng.normal(0, 0.04, n)  # a little aspiration noise
        voiced *= np.hanning(n)
        # a short fricative (an "s"-like hiss, high-passed) at the start of every other syllable
        burst = np.zeros(n)
        if k % 2 == 0:
            hiss = rng.normal(0, 1, int(0.07 * SR))
            sos = butter(4, 4500, btype="highpass", fs=SR, output="sos")
            burst[: len(hiss)] = sosfilt(sos, hiss) * np.hanning(len(hiss)) * 0.35
        out.append(voiced / (np.abs(voiced).max() + 1e-9) * 0.5 + burst)
        out.append(np.zeros(int(0.06 * SR)))
    wave = np.concatenate(out)
    wave = wave / (np.abs(wave).max() + 1e-9) * 0.6
    return wave.astype(np.float32)


def telephone_band(wave, cutoff_hz=3400):
    """The same audio with everything above `cutoff_hz` removed: a stand-in for phone-band speech."""
    sos = butter(8, cutoff_hz, btype="lowpass", fs=SR, output="sos")
    return sosfilt(sos, wave).astype(np.float32)


def white_noise(seconds=3.0, level=0.3, seed=1):
    return np.clip(np.random.default_rng(seed).normal(0, level, int(seconds * SR)), -1, 1).astype(np.float32)


def tone(seconds=3.0, freq=170.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
