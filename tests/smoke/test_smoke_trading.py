"""Smoke tests for trading indicators."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def stock_data():
    """Synthetic OHLCV data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    return pd.DataFrame({
        "Date": dates,
        "Open": close + np.random.uniform(-0.5, 0.5, n),
        "High": close + np.random.uniform(0, 2, n),
        "Low": close - np.random.uniform(0, 2, n),
        "Close": close,
        "Volume": np.random.randint(100000, 1000000, n),
    })


@pytest.mark.smoke
def test_rsi_indicator(stock_data):
    from wavqwise.trading.indicators.momentum import RSIIndicator
    result = RSIIndicator(14).compute(stock_data)
    assert "RSI" in result.columns
    rsi_valid = result["RSI"].dropna()
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()


@pytest.mark.smoke
def test_macd_indicator(stock_data):
    from wavqwise.trading.indicators.trend import MACDIndicator
    result = MACDIndicator().compute(stock_data)
    assert "MACD" in result.columns
    assert "MACD_signal" in result.columns
    assert "MACD_hist" in result.columns


@pytest.mark.smoke
def test_bollinger_bands(stock_data):
    from wavqwise.trading.indicators.volatility import BollingerBandsIndicator
    result = BollingerBandsIndicator(20, 2).compute(stock_data)
    assert "BB_upper" in result.columns
    assert "BB_lower" in result.columns
    assert "BB_middle" in result.columns
    valid = result.dropna()
    assert (valid["BB_upper"] >= valid["BB_middle"]).all()
    assert (valid["BB_lower"] <= valid["BB_middle"]).all()


@pytest.mark.smoke
def test_sma_indicator(stock_data):
    from wavqwise.trading.indicators.trend import SMAIndicator
    result = SMAIndicator(20).compute(stock_data)
    assert "SMA_20" in result.columns


@pytest.mark.smoke
def test_stochastic_indicator(stock_data):
    from wavqwise.trading.indicators.momentum import StochasticIndicator
    result = StochasticIndicator(14).compute(stock_data)
    assert "%K" in result.columns
    assert "%D" in result.columns
