"""Sanity tests for anomaly detection correctness."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.sanity
def test_zscore_detects_obvious_anomaly():
    """A value 10x the std should always be flagged."""
    from wavqwise.anomaly.statistical import ZScoreDetector
    values = np.random.normal(50, 5, 1000)
    values[500] = 200  # Obvious outlier
    data = pd.DataFrame({"value": values})
    d = ZScoreDetector(threshold=3.0)
    d.fit(data, target="value")
    result = d.detect(data)
    assert result.iloc[500]["is_anomaly"] == True
    assert result.iloc[500]["anomaly_score"] > 3.0


@pytest.mark.sanity
def test_iqr_detects_obvious_anomaly():
    from wavqwise.anomaly.statistical import IQRDetector
    values = np.random.normal(50, 5, 1000)
    values[100] = -100  # Way below
    data = pd.DataFrame({"value": values})
    d = IQRDetector(multiplier=1.5)
    d.fit(data, target="value")
    result = d.detect(data)
    assert result.iloc[100]["is_anomaly"] == True


@pytest.mark.sanity
def test_anomaly_severity_ordering():
    """Higher scores should get higher severity."""
    from wavqwise.anomaly.statistical import ZScoreDetector
    values = np.random.normal(50, 5, 500)
    values[100] = 200   # Very high
    values[200] = 120   # Moderately high
    data = pd.DataFrame({"value": values})
    d = ZScoreDetector(threshold=3.0)
    d.fit(data, target="value")
    result = d.detect(data)
    assert result.iloc[100]["anomaly_score"] > result.iloc[200]["anomaly_score"]


@pytest.mark.sanity
def test_no_false_positives_on_clean_data():
    """Clean gaussian data should have very few anomalies."""
    from wavqwise.anomaly.statistical import ZScoreDetector
    np.random.seed(42)
    values = np.random.normal(50, 5, 10000)
    data = pd.DataFrame({"value": values})
    d = ZScoreDetector(threshold=3.0)
    d.fit(data, target="value")
    result = d.detect(data)
    anomaly_rate = result["is_anomaly"].mean()
    assert anomaly_rate < 0.01  # Less than 1% false positives


@pytest.mark.sanity
def test_anomaly_pipeline_summary():
    from wavqwise import AnomalyPipeline
    values = np.random.normal(50, 5, 500)
    values[50] = 200
    data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=500, freq="D"),
        "value": values,
    })
    p = AnomalyPipeline()
    p.load(data, target="value", time="date")
    result = p.detect(method="zscore")
    summary = result.summary()
    assert "Anomalies:" in summary
    assert "zscore" in summary
