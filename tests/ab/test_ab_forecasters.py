"""A/B tests for forecaster comparison and ensemble."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def seasonal_data():
    """Data with clear weekly seasonality — seasonal models should win."""
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    trend = np.linspace(100, 130, 365)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(365) / 7)
    noise = np.random.normal(0, 2, 365)
    return pd.DataFrame({"date": dates, "value": trend + seasonal + noise})


@pytest.mark.ab
def test_seasonal_naive_beats_naive_on_seasonal(seasonal_data):
    """Seasonal naive should outperform naive on seasonal data."""
    from wavqwise import WavqPipeline
    from wavqwise.evaluation.metrics import Metrics

    data = seasonal_data
    n = len(data)
    horizon = 14
    train = data.iloc[:n - horizon]
    test = data.iloc[n - horizon:]

    # Naive
    p1 = WavqPipeline()
    p1.load(train, target="value", time="date")
    f1 = p1.forecast(horizon=horizon, model="naive")

    # Seasonal Naive
    p2 = WavqPipeline()
    p2.load(train, target="value", time="date")
    f2 = p2.forecast(horizon=horizon, model="seasonal_naive")

    mae_naive = Metrics.mae(test["value"].values, f1.forecast["value"].values)
    mae_seasonal = Metrics.mae(test["value"].values, f2.forecast["value"].values)

    assert mae_seasonal < mae_naive  # Seasonal should win


@pytest.mark.ab
def test_ensemble_not_worse_than_worst(sample_data):
    """Ensemble should not be worse than the worst individual model."""
    from wavqwise import WavqPipeline
    from wavqwise.evaluation.metrics import Metrics

    data = sample_data
    horizon = 10
    train = data.iloc[:-horizon]
    test = data.iloc[-horizon:]

    # Individual models
    maes = {}
    for model in ["moving_average", "ema", "naive"]:
        p = WavqPipeline()
        p.load(train, target="value", time="date")
        f = p.forecast(horizon=horizon, model=model)
        maes[model] = Metrics.mae(test["value"].values, f.forecast["value"].values[:len(test)])

    worst_mae = max(maes.values())

    # Ensemble
    p = WavqPipeline()
    p.load(train, target="value", time="date")
    f_ens = p.forecast(horizon=horizon, model=["moving_average", "ema", "naive"])
    mae_ens = Metrics.mae(test["value"].values, f_ens.forecast["value"].values[:len(test)])

    assert mae_ens <= worst_mae * 1.1  # Should not be much worse than worst


@pytest.mark.ab
def test_ema_adapts_faster_than_sma(sample_data):
    """EMA with short span should track recent values more closely than SMA with longer window."""
    from wavqwise.forecasters.traditional.moving_average import (
        MovingAverageForecaster, EMAForecaster,
    )

    data = sample_data.copy()
    # Add sudden jump in last 20 points
    data.loc[data.index[-20:], "value"] += 50

    sma = MovingAverageForecaster(window=30)
    sma.fit(data, target="value", time_col="date")
    pred_sma = sma.predict(horizon=5)

    ema = EMAForecaster(span=5)
    ema.fit(data, target="value", time_col="date")
    pred_ema = ema.predict(horizon=5)

    last_val = data["value"].iloc[-1]
    diff_sma = abs(pred_sma["value"].iloc[0] - last_val)
    diff_ema = abs(pred_ema["value"].iloc[0] - last_val)
    # Short EMA should be closer than long SMA to the jumped level
    assert diff_ema < diff_sma
