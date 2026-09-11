"""Adapter for Nixtla StatsForecast / NeuralForecast / MLForecast."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster

class NixtlaAdapter(BaseForecaster):
    """Wrap any Nixtla model.

    Usage:
        from wavqwise.adapters import NixtlaAdapter
        pipeline.forecast(model=NixtlaAdapter("AutoARIMA"))
        pipeline.forecast(model=NixtlaAdapter("AutoETS"))
        pipeline.forecast(model=NixtlaAdapter("CES"))
    """
    def __init__(self, model_name="AutoARIMA", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._sf = None
        self._target = self._time_col = self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from statsforecast import StatsForecast
            from statsforecast import models as sfm
        except ImportError:
            raise ImportError("pip install statsforecast")
        self._target, self._time_col = target, time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        model_class = getattr(sfm, self.model_name)
        df = data[[time_col, target]].copy()
        df.columns = ["ds", "y"]; df["unique_id"] = "series_1"
        self._sf = StatsForecast(models=[model_class()], freq=self._freq)
        self._sf.fit(df)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        level = [int(confidence_level * 100)]
        pred = self._sf.predict(h=horizon, level=level).reset_index()
        lvl = level[0]
        col = self.model_name
        return pd.DataFrame({
            self._time_col: pred["ds"].values,
            self._target: pred[col].values,
            f"{self._target}_lower": pred.get(f"{col}-lo-{lvl}", pred[col]).values,
            f"{self._target}_upper": pred.get(f"{col}-hi-{lvl}", pred[col]).values,
        })
