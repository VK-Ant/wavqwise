"""Adapter for Darts (unit8co/darts) time-series library."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster

class DartsAdapter(BaseForecaster):
    """Wrap any Darts model.

    Usage:
        from wavqwise.adapters import DartsAdapter
        pipeline.forecast(model=DartsAdapter("ExponentialSmoothing"))
        pipeline.forecast(model=DartsAdapter("NBEATSModel"))
    """
    def __init__(self, model_name="ExponentialSmoothing", model_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self._model = None
        self._target = self._time_col = self._freq = None
        self._series = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from darts import TimeSeries
            import darts.models as dm
        except ImportError:
            raise ImportError("pip install darts")
        self._target, self._time_col = target, time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._series = TimeSeries.from_dataframe(data, time_col, target, freq=self._freq)
        model_class = getattr(dm, self.model_name)
        self._model = model_class(**self.model_kwargs)
        self._model.fit(self._series)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        pred = self._model.predict(horizon)
        df = pred.pd_dataframe().reset_index()
        df.columns = [self._time_col, self._target]
        std = self._series.pd_dataframe().std().iloc[0]
        z = 1.96
        df[f"{self._target}_lower"] = df[self._target] - z * std
        df[f"{self._target}_upper"] = df[self._target] + z * std
        return df
