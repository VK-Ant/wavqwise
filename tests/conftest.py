"""Shared test fixtures."""
import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    values = 100 + np.cumsum(np.random.normal(0, 1, 200))
    return pd.DataFrame({"date": dates, "value": values})

@pytest.fixture
def sample_csv(tmp_path, sample_data):
    path = tmp_path / "test_data.csv"
    sample_data.to_csv(path, index=False)
    return str(path)

@pytest.fixture
def anomaly_data():
    dates = pd.date_range("2024-01-01", periods=500, freq="D")
    values = np.random.normal(50, 5, 500)
    values[100] = 200  # Obvious anomaly
    values[300] = -50   # Obvious anomaly
    return pd.DataFrame({"date": dates, "value": values})
