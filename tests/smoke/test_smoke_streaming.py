"""Smoke tests for real-time streaming engine."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def stream_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    values = 50 + 10 * np.sin(2 * np.pi * np.arange(100) / 24) + np.random.normal(0, 1, 100)
    return pd.DataFrame({"timestamp": dates, "value": values})


@pytest.mark.smoke
def test_stream_creation(stream_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream(model="ema", anomaly_method="zscore")
    assert stream is not None
    assert stream.stats["points_processed"] == 0


@pytest.mark.smoke
def test_stream_push(stream_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream(model="ema", forecast_every=5)
    for i in range(10):
        stream.push({"timestamp": f"2024-01-05 {i:02d}:00:00", "value": 50 + np.random.normal(0, 1)})
    assert stream.stats["points_processed"] == 10


@pytest.mark.smoke
def test_stream_detects_anomaly(stream_data):
    from wavqwise import WavqPipeline
    alerts = []
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream(model="ema", on_anomaly=lambda e: alerts.append(e), anomaly_threshold=2.5)
    stream.push({"timestamp": "2024-01-05 00:00:00", "value": 200})  # Obvious anomaly
    assert len(alerts) > 0
    assert alerts[0].has_anomaly


@pytest.mark.smoke
def test_stream_updates_forecast(stream_data):
    from wavqwise import WavqPipeline
    forecasts = []
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream(model="ema", on_forecast=lambda f: forecasts.append(f), forecast_every=3)
    for i in range(6):
        stream.push({"timestamp": f"2024-01-05 {i:02d}:00:00", "value": 50})
    assert len(forecasts) >= 1


@pytest.mark.smoke
def test_stream_spike_not_masked(stream_data):
    """Regression: a spike must NOT define its own baseline.

    Before the fix, new data was appended to the window BEFORE the
    detector was fitted, so the spike contaminated the baseline and
    looked 'normal'.  After the fix the detector fits on a snapshot
    taken BEFORE the new row is appended.
    """
    from wavqwise import WavqPipeline
    alerts = []
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream(
        model="ema",
        on_anomaly=lambda e: alerts.append(e),
        anomaly_threshold=2.5,
    )
    # Push several normal points first to stabilise baseline
    for i in range(20):
        stream.push({"timestamp": f"2024-01-05 {i:02d}:00:00",
                      "value": 50 + np.random.normal(0, 0.5)})
    normal_alerts = len(alerts)

    # Now push a massive spike -- must be caught
    stream.push({"timestamp": "2024-01-06 00:00:00", "value": 500})
    assert len(alerts) > normal_alerts, (
        "Spike at 500 was not detected -- detector may be fitting on contaminated window"
    )


@pytest.mark.smoke
def test_stream_cold_start_no_alerts(stream_data):
    """No anomaly alerts during cold-start (window too small)."""
    from wavqwise import WavqPipeline
    alerts = []
    p = WavqPipeline()
    # Start with very small historical dataset (below min_window threshold)
    tiny = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "value": [50.0, 50.1, 49.9, 50.2, 50.0],
    })
    p.load(tiny, target="value", time="timestamp")
    p.forecast(horizon=1, model="ema")
    stream = p.stream(
        model="ema",
        on_anomaly=lambda e: alerts.append(e),
        anomaly_threshold=2.0,
        window_size=100,
    )
    # Push a few points -- window (5 + 5 = 10) still below min_window (20)
    for i in range(5):
        stream.push({"timestamp": f"2024-01-01 {i + 5:02d}:00:00", "value": 999})
    assert len(alerts) == 0, "Should not alert during cold-start"


@pytest.mark.smoke
def test_stream_summary(stream_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream()
    summary = stream.summary()
    assert "points" in summary
    assert "anomalies" in summary
