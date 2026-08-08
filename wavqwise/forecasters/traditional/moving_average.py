"""Moving Average forecasters - SMA and EMA."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseForecaster


class MovingAverageForecaster(BaseForecaster):
    """Simple Moving Average forecaster."""

    def __init__(self, window: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self._last_values = None
        self._target = None
        self._time_col = None
        self._last_date = None
        self._freq = None

    def fit(self, data: pd.DataFrame, target: str = "value",
            time_col: str = "timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._last_values = data[target].tail(self.window).values
        self._last_date = data[time_col].max()
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._fitted = True

    def predict(self, horizon: int = 30, confidence_level: float = 0.95, **kwargs) -> pd.DataFrame:
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        predictions = []
        values = list(self._last_values)
        std = np.std(values) if len(values) > 1 else 0

        z = 1.96 if confidence_level == 0.95 else 1.645
        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]

        for i in range(horizon):
            pred = np.mean(values[-self.window:])
            margin = z * std * np.sqrt(1 + i / self.window)
            predictions.append({
                self._time_col: future_dates[i],
                self._target: pred,
                f"{self._target}_lower": pred - margin,
                f"{self._target}_upper": pred + margin,
            })
            values.append(pred)

        return pd.DataFrame(predictions)

    def update(self, new_data: pd.DataFrame, **kwargs):
        new_values = new_data[self._target].values
        self._last_values = np.concatenate([self._last_values, new_values])[-self.window:]
        self._last_date = new_data[self._time_col].max()


class EMAForecaster(BaseForecaster):
    """Exponential Moving Average forecaster."""

    def __init__(self, span: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.span = span
        self._ema_value = None
        self._target = None
        self._time_col = None
        self._last_date = None
        self._freq = None
        self._std = None

    def fit(self, data: pd.DataFrame, target: str = "value",
            time_col: str = "timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._ema_value = data[target].ewm(span=self.span).mean().iloc[-1]
        self._std = data[target].std()
        self._last_date = data[time_col].max()
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._fitted = True

    def predict(self, horizon: int = 30, confidence_level: float = 0.95, **kwargs) -> pd.DataFrame:
        z = 1.96 if confidence_level == 0.95 else 1.645
        future_dates = pd.date_range(self._last_date, periods=horizon + 1, freq=self._freq)[1:]
        predictions = []
        for i in range(horizon):
            margin = z * self._std * np.sqrt(1 + i / self.span)
            predictions.append({
                self._time_col: future_dates[i],
                self._target: self._ema_value,
                f"{self._target}_lower": self._ema_value - margin,
                f"{self._target}_upper": self._ema_value + margin,
            })
        return pd.DataFrame(predictions)

    def update(self, new_data: pd.DataFrame, **kwargs):
        alpha = 2.0 / (self.span + 1)
        for val in new_data[self._target].values:
            self._ema_value = alpha * val + (1 - alpha) * self._ema_value
        self._last_date = new_data[self._time_col].max()
