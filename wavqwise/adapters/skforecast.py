"""Adapter for skforecast library."""
from wavqwise.core.base import BaseForecaster

class SkforecastAdapter(BaseForecaster):
    """Wrap skforecast models.

    Usage:
        from wavqwise.adapters import SkforecastAdapter
        pipeline.forecast(model=SkforecastAdapter(regressor=LGBMRegressor()))
    """
    def __init__(self, regressor=None, lags=14, **kwargs):
        super().__init__(**kwargs)
        self.regressor = regressor
        self.lags = lags
        self._model = None
        self._target = self._time_col = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from skforecast.ForecasterAutoreg import ForecasterAutoreg
        except ImportError:
            raise ImportError("pip install skforecast")
        import pandas as pd
        self._target, self._time_col = target, time_col
        if self.regressor is None:
            from sklearn.ensemble import RandomForestRegressor
            self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self._model = ForecasterAutoreg(regressor=self.regressor, lags=self.lags)
        series = data.set_index(time_col)[target]
        series.index = pd.DatetimeIndex(series.index, freq=pd.infer_freq(series.index) or "D")
        self._model.fit(y=series)
        self._fitted = True
        self._last_date = data[time_col].max()
        self._freq = pd.infer_freq(data[time_col]) or "D"

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        import pandas as pd, numpy as np
        preds = self._model.predict(steps=horizon)
        dates = pd.date_range(self._last_date, periods=horizon+1, freq=self._freq)[1:]
        std = preds.std()
        return pd.DataFrame({
            self._time_col: dates, self._target: preds.values,
            f"{self._target}_lower": preds.values - 1.96*std,
            f"{self._target}_upper": preds.values + 1.96*std,
        })
