"""A/B tests: Model comparison."""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.ab
def test_compare_models(sample_data):
    from wavqwise import WavqPipeline
    p = WavqPipeline()
    p.load(sample_data, target="value", time="date")
    comparison = p.compare_models(models=["moving_average", "ema", "naive"], horizon=10)
    assert len(comparison) == 3
    assert "MAE" in comparison.columns

@pytest.mark.ab
def test_ab_incremental_vs_full(sample_data):
    from wavqwise import WavqPipeline
    from wavqwise.evaluation.metrics import Metrics
    train = sample_data.iloc[:150]
    test = sample_data.iloc[180:190]
    new = sample_data.iloc[150:180]

    # Full retrain
    p1 = WavqPipeline()
    p1.load(sample_data.iloc[:180], target="value", time="date")
    f1 = p1.forecast(horizon=10, model="moving_average")

    # Incremental
    p2 = WavqPipeline()
    p2.load(train, target="value", time="date")
    p2.forecast(horizon=10, model="moving_average")
    p2.update(new)
    f2 = p2.forecast(horizon=10, model="moving_average")

    mae_full = Metrics.mae(test["value"].values, f1.forecast["value"].values[:len(test)])
    mae_incr = Metrics.mae(test["value"].values, f2.forecast["value"].values[:len(test)])

    # Incremental should be within 50% of full retrain
    assert mae_incr < mae_full * 1.5
