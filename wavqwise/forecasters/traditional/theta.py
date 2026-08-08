"""Theta method forecaster."""
import pandas as pd
from wavqwise.core.base import BaseForecaster

class ThetaForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result = None
        self._target = None
        self._time_col = None
        self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        from statsmodels.tsa.forecasting.theta import ThetaModel
        self._target = target
        self._time_col = time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        series = data.set_index(time_col)[target].asfreq(self._freq).ffill()
        model = ThetaModel(series)
        self._result = model.fit()
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        pred = self._result.forecast(horizon)
        pi = self._result.prediction_intervals(horizon, alpha=1-confidence_level)
        return pd.DataFrame({
            self._time_col: pred.index,
            self._target: pred.values,
            f"{self._target}_lower": pi.iloc[:, 0].values,
            f"{self._target}_upper": pi.iloc[:, 1].values,
        })
