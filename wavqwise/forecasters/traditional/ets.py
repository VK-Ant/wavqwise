"""Exponential Smoothing (ETS) forecaster."""
import pandas as pd
from wavqwise.core.base import BaseForecaster


class ETSForecaster(BaseForecaster):
    def __init__(self, seasonal_periods=None, trend="add", seasonal="add", **kwargs):
        super().__init__(**kwargs)
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self._result = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self._target = target
        self._time_col = time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        series = data.set_index(time_col)[target].asfreq(self._freq).ffill()
        sp = self.seasonal_periods or self._detect_seasonal_period(series)
        model = ExponentialSmoothing(
            series, trend=self.trend,
            seasonal=self.seasonal if sp and sp > 1 else None,
            seasonal_periods=sp if sp and sp > 1 else None,
        )
        self._result = model.fit(optimized=True)
        self._fitted = True

    def _detect_seasonal_period(self, series):
        freq = self._freq
        period_map = {"H": 24, "D": 7, "W": 52, "M": 12, "MS": 12, "Q": 4, "QS": 4}
        for k, v in period_map.items():
            if freq and k in freq:
                return v
        return None

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        import numpy as np
        pred = self._result.forecast(horizon)
        residuals = self._result.resid
        std = residuals.std()
        z = 1.96 if confidence_level == 0.95 else 1.645
        return pd.DataFrame({
            self._time_col: pred.index,
            self._target: pred.values,
            f"{self._target}_lower": pred.values - z * std * np.sqrt(np.arange(1, horizon+1)),
            f"{self._target}_upper": pred.values + z * std * np.sqrt(np.arange(1, horizon+1)),
        })
