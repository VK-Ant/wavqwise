"""ARIMA and AutoARIMA forecasters."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    def __init__(self, order=(1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self._model = None
        self._result = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data: pd.DataFrame, target="value", time_col="timestamp", **kwargs):
        from statsmodels.tsa.arima.model import ARIMA
        self._target = target
        self._time_col = time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        series = data.set_index(time_col)[target].asfreq(self._freq)
        series = series.ffill()
        self._model = ARIMA(series, order=self.order)
        self._result = self._model.fit()
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs) -> pd.DataFrame:
        forecast = self._result.get_forecast(steps=horizon, alpha=1 - confidence_level)
        mean = forecast.predicted_mean
        ci = forecast.conf_int()
        return pd.DataFrame({
            self._time_col: mean.index,
            self._target: mean.values,
            f"{self._target}_lower": ci.iloc[:, 0].values,
            f"{self._target}_upper": ci.iloc[:, 1].values,
        })

    def update(self, new_data: pd.DataFrame, **kwargs):
        series = new_data[self._target].values
        self._result = self._result.append(series)


class AutoARIMAForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data: pd.DataFrame, target="value", time_col="timestamp", **kwargs):
        try:
            from statsforecast.models import AutoARIMA as SFAutoARIMA
            from statsforecast import StatsForecast
            self._target = target
            self._time_col = time_col
            self._freq = pd.infer_freq(data[time_col]) or "D"
            df = data[[time_col, target]].copy()
            df.columns = ["ds", "y"]
            df["unique_id"] = "series_1"
            self._sf = StatsForecast(models=[SFAutoARIMA()], freq=self._freq)
            self._sf.fit(df)
            self._fitted = True
        except ImportError:
            raise ImportError("pip install wavqwise[traditional]")

    def predict(self, horizon=30, confidence_level=0.95, **kwargs) -> pd.DataFrame:
        level = [int(confidence_level * 100)]
        pred = self._sf.predict(h=horizon, level=level)
        pred = pred.reset_index()
        lvl = level[0]
        result = pd.DataFrame({
            self._time_col: pred["ds"].values,
            self._target: pred["AutoARIMA"].values,
            f"{self._target}_lower": pred.get(f"AutoARIMA-lo-{lvl}", pred["AutoARIMA"]).values,
            f"{self._target}_upper": pred.get(f"AutoARIMA-hi-{lvl}", pred["AutoARIMA"]).values,
        })
        return result
