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
def test_stream_summary(stream_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(stream_data, target="value", time="timestamp")
    p.forecast(horizon=10, model="ema")
    stream = p.stream()
    summary = stream.summary()
    assert "points" in summary
    assert "anomalies" in summary
