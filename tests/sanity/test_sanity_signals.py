"""Sanity tests for signal processing correctness."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.sanity
def test_alpha_dominant_in_alpha_signal():
    """Pure 10Hz signal should have dominant alpha band."""
    from wavqwise import SignalPipeline
    sr = 256
    t = np.arange(4 * sr) / sr
    signal = 10 * np.sin(2 * np.pi * 10 * t)  # Pure 10Hz alpha
    sig = SignalPipeline()
    sig._data = signal
    sig._sample_rate = sr
    result = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
    assert result.bands["alpha"] > result.bands["beta"]
    assert result.bands["alpha"] > result.bands["theta"]
    assert result.bands["alpha"] > result.bands["delta"]


@pytest.mark.sanity
def test_beta_dominant_in_beta_signal():
    """Pure 20Hz signal should have dominant beta band."""
    from wavqwise import SignalPipeline
    sr = 256
    t = np.arange(4 * sr) / sr
    signal = 10 * np.sin(2 * np.pi * 20 * t)  # Pure 20Hz beta
    sig = SignalPipeline()
    sig._data = signal
    sig._sample_rate = sr
    result = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
    assert result.bands["beta"] > result.bands["alpha"]
    assert result.bands["beta"] > result.bands["theta"]


@pytest.mark.sanity
def test_filter_removes_high_freq():
    """Band-pass 1-30 Hz should remove 50Hz component."""
    from wavqwise import SignalPipeline
    sr = 256
    t = np.arange(4 * sr) / sr
    signal = 5 * np.sin(2 * np.pi * 10 * t) + 5 * np.sin(2 * np.pi * 50 * t)
    sig = SignalPipeline()
    sig._data = signal.copy()
    sig._sample_rate = sr
    sig.filter(low=1, high=30)
    # After filtering, gamma (30-50) should be near zero
    result = sig.extract_bands(["alpha", "gamma"])
    assert result.bands["alpha"] > result.bands["gamma"] * 5
