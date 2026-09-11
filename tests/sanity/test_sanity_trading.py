"""Sanity tests for trading indicator correctness."""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.fixture
def rising_stock():
    """Steadily rising stock — RSI should be high."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(100, 200, n)
    return pd.DataFrame({
        "Date": dates, "Open": close - 0.5, "High": close + 1,
        "Low": close - 1, "Close": close,
        "Volume": np.full(n, 1000000),
    })


@pytest.fixture
def falling_stock():
    """Steadily falling stock — RSI should be low."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(200, 100, n)
    return pd.DataFrame({
        "Date": dates, "Open": close + 0.5, "High": close + 1,
        "Low": close - 1, "Close": close,
        "Volume": np.full(n, 1000000),
    })


@pytest.mark.sanity
def test_rsi_high_for_rising(rising_stock):
    from wavqwise.trading.indicators.momentum import RSIIndicator
    result = RSIIndicator(14).compute(rising_stock)
    last_rsi = result["RSI"].iloc[-1]
    assert last_rsi > 70  # Overbought for steady rise


@pytest.mark.sanity
def test_rsi_low_for_falling(falling_stock):
    from wavqwise.trading.indicators.momentum import RSIIndicator
    result = RSIIndicator(14).compute(falling_stock)
    last_rsi = result["RSI"].iloc[-1]
    assert last_rsi < 30  # Oversold for steady fall


@pytest.mark.sanity
def test_bollinger_contains_price(rising_stock):
    """Most prices should fall within Bollinger Bands."""
    from wavqwise.trading.indicators.volatility import BollingerBandsIndicator
    result = BollingerBandsIndicator(20, 2).compute(rising_stock)
    valid = result.dropna()
    within = ((valid["Close"] >= valid["BB_lower"]) &
              (valid["Close"] <= valid["BB_upper"])).mean()
    assert within > 0.90  # At least 90% within bands


@pytest.mark.sanity
def test_macd_positive_for_rising(rising_stock):
    from wavqwise.trading.indicators.trend import MACDIndicator
    result = MACDIndicator().compute(rising_stock)
    last_macd = result["MACD"].iloc[-1]
    assert last_macd > 0  # Bullish
