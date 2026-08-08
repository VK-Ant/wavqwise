"""SARIMA forecaster."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster


class SARIMAForecaster(BaseForecaster):
    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        self._target = target
        self._time_col = time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        series = data.set_index(time_col)[target].asfreq(self._freq).ffill()
        model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
        self._result = model.fit(disp=False)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        forecast = self._result.get_forecast(steps=horizon, alpha=1-confidence_level)
        mean = forecast.predicted_mean
        ci = forecast.conf_int()
        return pd.DataFrame({
            self._time_col: mean.index,
            self._target: mean.values,
            f"{self._target}_lower": ci.iloc[:, 0].values,
            f"{self._target}_upper": ci.iloc[:, 1].values,
        })

    def update(self, new_data, **kwargs):
        self._result = self._result.append(new_data[self._target].values)


class AutoSARIMAForecaster(SARIMAForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
