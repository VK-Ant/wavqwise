"""Smoke tests for anomaly detection."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

@pytest.mark.smoke
def test_anomaly_zscore(anomaly_data):
    from wavqwise import AnomalyPipeline
    p = AnomalyPipeline()
    p.load(anomaly_data, target="value", time="date")
    result = p.detect(method="zscore")
    assert len(result.anomalies) > 0
    assert "is_anomaly" in result.data.columns

@pytest.mark.smoke
def test_anomaly_iqr(anomaly_data):
    from wavqwise import AnomalyPipeline
    p = AnomalyPipeline()
    p.load(anomaly_data, target="value", time="date")
    result = p.detect(method="iqr")
    assert len(result.anomalies) > 0
