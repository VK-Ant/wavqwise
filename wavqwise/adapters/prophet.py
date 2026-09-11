"""Adapter for Meta Prophet."""
import pandas as pd
from wavqwise.core.base import BaseForecaster

class ProphetAdapter(BaseForecaster):
    """Wrap Meta Prophet.

    Usage:
        from wavqwise.adapters import ProphetAdapter
        pipeline.forecast(model=ProphetAdapter())
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._target = self._time_col = self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("pip install prophet")
        self._target, self._time_col = target, time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        df = data[[time_col, target]].rename(columns={time_col: "ds", target: "y"})
        self._model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        self._model.fit(df)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq)
        pred = self._model.predict(future).tail(horizon)
        return pd.DataFrame({
            self._time_col: pred["ds"].values,
            self._target: pred["yhat"].values,
            f"{self._target}_lower": pred["yhat_lower"].values,
            f"{self._target}_upper": pred["yhat_upper"].values,
        })
