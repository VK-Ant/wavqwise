"""Sanity tests for forecaster output shapes."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.sanity
def test_moving_average_shape(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=14, model="moving_average")
    assert result.forecast.shape[0] == 14
    assert "value_lower" in result.forecast.columns
    assert "value_upper" in result.forecast.columns

@pytest.mark.sanity
def test_ema_shape(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=7, model="ema")
    assert result.forecast.shape[0] == 7

@pytest.mark.sanity
def test_naive_shape(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=10, model="naive")
    assert result.forecast.shape[0] == 10

@pytest.mark.sanity
def test_forecast_values_reasonable(sample_data):
    from wavqwise import WavqPipeline
    import numpy as np
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=10, model="moving_average")
    last_val = sample_data["value"].iloc[-1]
    pred_mean = result.forecast["value"].mean()
    assert abs(pred_mean - last_val) < 50  # Reasonable range
