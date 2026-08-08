"""Sklearn-compatible model wrapper for time-series."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster
from wavqwise.features.auto_features import AutoFeatureEngineer


class SklearnForecaster(BaseForecaster):
    """Wraps any sklearn model for time-series forecasting."""
    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._fe = AutoFeatureEngineer()
        self._target = None
        self._time_col = None
        self._freq = None
        self._data = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._data = data.copy()
        df = self._fe.generate(data, target, time_col)
        feature_cols = [c for c in df.columns if c not in [target, time_col]]
        X = df[feature_cols].values
        y = df[target].values
        self._feature_cols = feature_cols
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        last_date = self._data[self._time_col].max()
        dates = pd.date_range(last_date, periods=horizon+1, freq=self._freq)[1:]
        preds = []
        current_data = self._data.copy()
        for date in dates:
            new_row = {self._time_col: date, self._target: current_data[self._target].iloc[-1]}
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            df = self._fe.generate(current_data, self._target, self._time_col)
            X = df[self._feature_cols].values[-1:]
            pred = self._model.predict(X)[0]
            current_data.iloc[-1, current_data.columns.get_loc(self._target)] = pred
            preds.append(pred)
        preds = np.array(preds)
        std = self._data[self._target].std() * 0.1
        z = 1.96
        return pd.DataFrame({
            self._time_col: dates, self._target: preds,
            f"{self._target}_lower": preds - z * std * np.sqrt(np.arange(1, horizon+1)),
            f"{self._target}_upper": preds + z * std * np.sqrt(np.arange(1, horizon+1)),
        })

class RandomForestForecaster(SklearnForecaster):
    def __init__(self, n_estimators=100, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        super().__init__(model=RandomForestRegressor(n_estimators=n_estimators, random_state=42), **kwargs)
