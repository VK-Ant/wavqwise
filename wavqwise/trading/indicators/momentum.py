"""Momentum indicators."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseIndicator

class RSIIndicator(BaseIndicator):
    def __init__(self, period=14, **kwargs): self.period = period
    def compute(self, data, **kwargs):
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()
        rs = gain / loss.replace(0, 1e-8)
        data["RSI"] = 100 - (100 / (1 + rs))
        return data

class StochasticIndicator(BaseIndicator):
    def __init__(self, period=14, **kwargs): self.period = period
    def compute(self, data, **kwargs):
        low_min = data["Low"].rolling(self.period).min()
        high_max = data["High"].rolling(self.period).max()
        data["%K"] = (data["Close"] - low_min) / (high_max - low_min + 1e-8) * 100
        data["%D"] = data["%K"].rolling(3).mean()
        return data
