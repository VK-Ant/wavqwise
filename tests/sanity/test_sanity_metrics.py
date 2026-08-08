"""Sanity tests: Are outputs correct?"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.sanity
def test_mae_correct():
    from wavqwise.evaluation.metrics import Metrics
    assert abs(Metrics.mae([1, 2, 3], [1, 2, 4]) - 1/3) < 1e-6

@pytest.mark.sanity
def test_rmse_correct():
    from wavqwise.evaluation.metrics import Metrics
    assert abs(Metrics.rmse([1, 2, 3], [1, 2, 3]) - 0.0) < 1e-6

@pytest.mark.sanity
def test_mape_correct():
    from wavqwise.evaluation.metrics import Metrics
    assert Metrics.mape([100, 200], [110, 190]) == pytest.approx(7.5, abs=0.1)

@pytest.mark.sanity
def test_all_metrics():
    from wavqwise.evaluation.metrics import Metrics
    result = Metrics.all_metrics([1,2,3,4,5], [1.1,2.1,2.9,4.2,4.8])
    assert "MAE" in result
    assert "RMSE" in result
    assert "MAPE" in result
