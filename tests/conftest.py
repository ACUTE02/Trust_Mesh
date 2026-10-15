"""Shared test setup.

The earlier serving tests feed the service pure tones and noise as stand-in audio. The audio guard
(serving/audio_guard.py) correctly refuses those, so it is switched off by default here and switched on
explicitly in tests/test_audio_guard.py, which is what tests the guard.
"""
import pytest

from serving import config


@pytest.fixture(autouse=True)
def audio_guard_off_unless_a_test_turns_it_on(monkeypatch):
    monkeypatch.setattr(config, "AUDIO_GUARD", False)
