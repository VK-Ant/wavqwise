"""Integration tests: Full end-to-end pipeline flows."""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def full_dataset():
    """Larger realistic dataset for integration tests."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(100, 180, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 5, n)
    values = trend + seasonal + noise
    return pd.DataFrame({"timestamp": dates, "sales": values})


@pytest.mark.integration
def test_full_pipeline_csv_roundtrip(full_dataset, tmp_path):
    """Load CSV -> forecast -> save -> reload -> forecast again."""
    from wavqwise import WavqPipeline

    csv_path = tmp_path / "data.csv"
    full_dataset.to_csv(csv_path, index=False)

    # First run
    p1 = WavqPipeline()
    p1.load(str(csv_path), target="sales", time="timestamp")
    r1 = p1.forecast(horizon=14, model="moving_average")
    assert len(r1.forecast) == 14

    # Save pipeline
    save_path = str(tmp_path / "pipeline.joblib")
    p1.save(save_path)

    # Reload
    p2 = WavqPipeline.load_pipeline(save_path)
    p2.load(str(csv_path), target="sales", time="timestamp")
    r2 = p2.forecast(horizon=14)
    assert len(r2.forecast) == 14

    # Both pipelines should produce same model forecasts
    assert r1.model_name == r2.model_name


@pytest.mark.integration
def test_multi_update_pipeline(full_dataset):
    """Load -> forecast -> update 5 times -> forecast."""
    from wavqwise import WavqPipeline

    # Initial load with first 300 rows
    p = WavqPipeline()
    p.load(full_dataset.iloc[:300], target="sales", time="timestamp")
    p.forecast(horizon=10, model="ema")

    # 5 incremental updates, 20 rows each
    for i in range(5):
        start = 300 + i * 20
        end = start + 20
        new_data = full_dataset.iloc[start:end]
        p.update(new_data)

    result = p.forecast(horizon=10)
    assert len(result.forecast) == 10
    assert p._incremental.update_count == 5


@pytest.mark.integration
def test_model_switch_mid_pipeline(full_dataset):
    """Load -> forecast with SMA -> switch to EMA -> forecast again."""
    from wavqwise import WavqPipeline

    p = WavqPipeline()
    p.load(full_dataset, target="sales", time="timestamp")

    r1 = p.forecast(horizon=10, model="moving_average")
    assert r1.model_name == "moving_average"

    r2 = p.forecast(horizon=10, model="ema")
    assert r2.model_name == "ema"

    r3 = p.forecast(horizon=10, model="naive")
    assert r3.model_name == "naive"

    # All should produce valid forecasts
    for r in [r1, r2, r3]:
        assert len(r.forecast) == 10
        assert not r.forecast["sales"].isnull().any()


@pytest.mark.integration
def test_auto_model_selection(full_dataset):
    """Auto model selection should pick a model and forecast."""
    from wavqwise import WavqPipeline

    p = WavqPipeline()
    p.load(full_dataset, target="sales", time="timestamp")
    result = p.forecast(horizon=14, model="auto")
    assert len(result.forecast) == 14
    assert "auto(" in result.model_name


@pytest.mark.integration
def test_compare_and_pick_best(full_dataset):
    """Model comparison should rank models by MAE."""
    from wavqwise import WavqPipeline

    p = WavqPipeline()
    p.load(full_dataset, target="sales", time="timestamp")
    comparison = p.compare_models(
        models=["moving_average", "ema", "naive"],
        horizon=14,
    )
    assert len(comparison) == 3
    assert comparison["MAE"].iloc[0] <= comparison["MAE"].iloc[-1]  # Sorted


@pytest.mark.integration
def test_forecast_result_summary_and_repr(full_dataset):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(full_dataset, target="sales", time="timestamp")
    result = p.forecast(horizon=10, model="moving_average")

    summary = result.summary()
    assert "moving_average" in summary
    assert "Horizon: 10" in summary

    repr_str = repr(result)
    assert "ForecastResult" in repr_str

    df = result.to_dataframe()
    assert len(df) == 10


@pytest.mark.integration
def test_pipeline_info_output(full_dataset):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(full_dataset, target="sales", time="timestamp")
    p.forecast(horizon=5, model="ema")

    info = p.info()
    assert "Runtime:" in info
    assert "Data loaded: True" in info
    assert "sales" in info
    assert "ema" in info
    assert "Fitted: True" in info
