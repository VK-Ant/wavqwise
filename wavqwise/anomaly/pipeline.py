"""Anomaly detection pipeline."""
import pandas as pd
from wavqwise.core.registry import Registry
from wavqwise.loaders.auto_loader import AutoLoader

class AnomalyResult:
    def __init__(self, data, target, time_col, method):
        self.data = data
        self.target = target
        self.time_col = time_col
        self.method = method
        self.anomalies = data[data["is_anomaly"]]

    def plot(self, engine="matplotlib", show_severity=False, **kwargs):
        from wavqwise.visualization.anomaly_plot import AnomalyPlot
        plotter = AnomalyPlot(engine=engine)
        return plotter.plot(self.data, self.target, self.time_col, show_severity=show_severity)

    def summary(self):
        total = len(self.data)
        n_anomalies = len(self.anomalies)
        return f"Anomalies: {n_anomalies}/{total} ({n_anomalies/total*100:.1f}%) | Method: {self.method}"

    def __repr__(self):
        return f"AnomalyResult(anomalies={len(self.anomalies)}, method='{self.method}')"


class AnomalyPipeline:
    def __init__(self):
        self._data = None
        self._target = None
        self._time_col = None

    def load(self, source, target=None, time=None, **kwargs):
        if isinstance(source, pd.DataFrame):
            self._data = source.copy()
        else:
            self._data = AutoLoader().load(source, **kwargs)
        self._target = target or self._data.select_dtypes("number").columns[0]
        self._time_col = time or self._data.columns[0]
        if not pd.api.types.is_datetime64_any_dtype(self._data[self._time_col]):
            self._data[self._time_col] = pd.to_datetime(self._data[self._time_col])
        return self

    def detect(self, method="zscore", **kwargs):
        detector = Registry.get_anomaly_detector(method)
        detector.fit(self._data, target=self._target)
        result = detector.detect(self._data)
        return AnomalyResult(result, self._target, self._time_col, method)
