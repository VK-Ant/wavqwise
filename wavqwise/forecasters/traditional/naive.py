"""Naive and Seasonal Naive baselines."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster

class NaiveForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last = None
        self._std = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._last = data[target].iloc[-1]
        self._std = data[target].diff().std()
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._last_date = data[time_col].max()
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        z = 1.96 if confidence_level == 0.95 else 1.645
        dates = pd.date_range(self._last_date, periods=horizon+1, freq=self._freq)[1:]
        margins = z * self._std * np.sqrt(np.arange(1, horizon+1))
        return pd.DataFrame({
            self._time_col: dates,
            self._target: self._last,
            f"{self._target}_lower": self._last - margins,
            f"{self._target}_upper": self._last + margins,
        })

    def update(self, new_data, **kwargs):
        self._last = new_data[self._target].iloc[-1]

class SeasonalNaiveForecaster(BaseForecaster):
    def __init__(self, season_length=7, **kwargs):
        super().__init__(**kwargs)
        self.season_length = season_length
        self._seasonal = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._seasonal = data[target].tail(self.season_length).values
        self._std = data[target].diff().std()
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._last_date = data[time_col].max()
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        z = 1.96
        dates = pd.date_range(self._last_date, periods=horizon+1, freq=self._freq)[1:]
        preds = np.tile(self._seasonal, (horizon // self.season_length) + 1)[:horizon]
        margins = z * self._std * np.sqrt(np.arange(1, horizon+1))
        return pd.DataFrame({
            self._time_col: dates, self._target: preds,
            f"{self._target}_lower": preds - margins,
            f"{self._target}_upper": preds + margins,
        })
