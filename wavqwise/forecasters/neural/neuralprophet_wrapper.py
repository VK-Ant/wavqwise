"""NeuralProphet wrapper."""
import pandas as pd
from wavqwise.core.base import BaseForecaster

class NeuralProphetForecaster(BaseForecaster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._target = self._time_col = self._freq = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from neuralprophet import NeuralProphet
        except ImportError:
            raise ImportError("pip install wavqwise[neural]")
        self._target, self._time_col = target, time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        df = data[[time_col, target]].rename(columns={time_col: "ds", target: "y"})
        self._model = NeuralProphet(**kwargs)
        self._model.fit(df, freq=self._freq)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        future = self._model.make_future_dataframe(self._model.data_params, periods=horizon)
        pred = self._model.predict(future)
        return pd.DataFrame({
            self._time_col: pred["ds"].values[-horizon:],
            self._target: pred["yhat1"].values[-horizon:],
            f"{self._target}_lower": pred["yhat1"].values[-horizon:] * 0.9,
            f"{self._target}_upper": pred["yhat1"].values[-horizon:] * 1.1,
        })
