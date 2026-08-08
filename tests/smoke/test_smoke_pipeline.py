"""Smoke tests: Does the pipeline run without crashing?"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.smoke
def test_pipeline_load_and_forecast(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=10, model="moving_average")
    assert len(result.forecast) == 10
    assert "value" in result.forecast.columns

@pytest.mark.smoke
def test_pipeline_csv_load(sample_csv):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_csv, target="value", time="date")
    result = p.forecast(horizon=5, model="moving_average")
    assert len(result.forecast) == 5

@pytest.mark.smoke
def test_pipeline_ema(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    result = p.forecast(horizon=10, model="ema")
    assert len(result.forecast) == 10

@pytest.mark.smoke
def test_pipeline_incremental(sample_data):
    import pandas as pd
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    train = sample_data.iloc[:150]
    new = sample_data.iloc[150:160]
    p.load(train, target="value", time="date")
    p.forecast(horizon=5, model="moving_average")
    p.update(new)
    result = p.forecast(horizon=5)
    assert len(result.forecast) == 5

@pytest.mark.smoke
def test_available_models():
    from wavqwise import WavqPipeline
    models = WavqPipeline.available_models()
    assert "moving_average" in models
    assert "arima" in models

@pytest.mark.smoke
def test_pipeline_auto_detect(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data)
    assert p._target is not None
    assert p._time_col is not None
