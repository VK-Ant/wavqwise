"""Amazon Chronos foundation model wrapper."""
import pandas as pd
import numpy as np
from wavqwise.core.base import BaseForecaster

class ChronosForecaster(BaseForecaster):
    def __init__(self, model_id="amazon/chronos-t5-small", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self._pipeline = None
        self._target = self._time_col = self._freq = self._data = None

    def fit(self, data, target="value", time_col="timestamp", **kwargs):
        try:
            from chronos import ChronosPipeline
            import torch
        except ImportError:
            raise ImportError("pip install wavqwise[foundation]")
        self._target, self._time_col = target, time_col
        self._freq = pd.infer_freq(data[time_col]) or "D"
        self._data = data
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._pipeline = ChronosPipeline.from_pretrained(self.model_id, device_map=device)
        self._fitted = True

    def predict(self, horizon=30, confidence_level=0.95, **kwargs):
        import torch
        context = torch.tensor(self._data[self._target].values, dtype=torch.float32)
        forecast = self._pipeline.predict(context, horizon)
        median = np.median(forecast.numpy(), axis=1)[0]
        lo = np.percentile(forecast.numpy(), ((1-confidence_level)/2)*100, axis=1)[0]
        hi = np.percentile(forecast.numpy(), (1-(1-confidence_level)/2)*100, axis=1)[0]
        dates = pd.date_range(self._data[self._time_col].max(), periods=horizon+1, freq=self._freq)[1:]
        return pd.DataFrame({
            self._time_col: dates, self._target: median,
            f"{self._target}_lower": lo, f"{self._target}_upper": hi,
        })
