"""Trend indicators."""
import pandas as pd
from wavqwise.core.base import BaseIndicator

class SMAIndicator(BaseIndicator):
    def __init__(self, period=20, **kwargs): self.period = period
    def compute(self, data, **kwargs):
        data[f"SMA_{self.period}"] = data["Close"].rolling(self.period).mean()
        return data

class EMAIndicator(BaseIndicator):
    def __init__(self, period=20, **kwargs): self.period = period
    def compute(self, data, **kwargs):
        data[f"EMA_{self.period}"] = data["Close"].ewm(span=self.period).mean()
        return data

class MACDIndicator(BaseIndicator):
    def compute(self, data, **kwargs):
        exp1 = data["Close"].ewm(span=12).mean()
        exp2 = data["Close"].ewm(span=26).mean()
        data["MACD"] = exp1 - exp2
        data["MACD_signal"] = data["MACD"].ewm(span=9).mean()
        data["MACD_hist"] = data["MACD"] - data["MACD_signal"]
        return data
