"""Volatility indicators."""
import pandas as pd
from wavqwise.core.base import BaseIndicator

class BollingerBandsIndicator(BaseIndicator):
    def __init__(self, period=20, std_dev=2, **kwargs): self.period = period; self.std_dev = std_dev
    def compute(self, data, **kwargs):
        sma = data["Close"].rolling(self.period).mean()
        std = data["Close"].rolling(self.period).std()
        data["BB_upper"] = sma + self.std_dev * std
        data["BB_middle"] = sma
        data["BB_lower"] = sma - self.std_dev * std
        return data

class ATRIndicator(BaseIndicator):
    def __init__(self, period=14, **kwargs): self.period = period
    def compute(self, data, **kwargs):
        import numpy as np
        h_l = data["High"] - data["Low"]
        h_pc = abs(data["High"] - data["Close"].shift(1))
        l_pc = abs(data["Low"] - data["Close"].shift(1))
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        data["ATR"] = tr.rolling(self.period).mean()
        return data
