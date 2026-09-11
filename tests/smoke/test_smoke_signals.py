"""Smoke tests for signal processing pipeline."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def eeg_signal():
    """Synthetic EEG-like signal: alpha + beta + noise."""
    sr = 256
    t = np.arange(4 * sr) / sr
    signal = (
        5 * np.sin(2 * np.pi * 10 * t)    # alpha
        + 2 * np.sin(2 * np.pi * 22 * t)  # beta
        + np.random.normal(0, 1, len(t))
    )
    return signal, sr


@pytest.mark.smoke
def test_signal_pipeline_load(eeg_signal):
    from wavqwise import SignalPipeline
    signal, sr = eeg_signal
    sig = SignalPipeline()
    sig._data = signal
    sig._sample_rate = sr
    assert sig._data is not None
    assert len(sig._data) == 4 * sr


@pytest.mark.smoke
def test_signal_filter(eeg_signal):
    from wavqwise import SignalPipeline
    signal, sr = eeg_signal
    sig = SignalPipeline()
    sig._data = signal.copy()
    sig._sample_rate = sr
    sig.filter(low=1, high=50)
    assert len(sig._data) == len(signal)


@pytest.mark.smoke
def test_signal_band_extraction(eeg_signal):
    from wavqwise import SignalPipeline
    signal, sr = eeg_signal
    sig = SignalPipeline()
    sig._data = signal
    sig._sample_rate = sr
    result = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
    assert "alpha" in result.bands
    assert "beta" in result.bands
    assert result.bands["alpha"] > result.bands["delta"]  # Alpha dominant


@pytest.mark.smoke
def test_signal_event_detection(eeg_signal):
    from wavqwise import SignalPipeline
    signal, sr = eeg_signal
    sig = SignalPipeline()
    sig._data = signal
    sig._sample_rate = sr
    result = sig.detect_events(threshold=3.0)
    assert isinstance(result.events, list)
