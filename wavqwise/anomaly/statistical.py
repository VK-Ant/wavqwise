"""Statistical anomaly detectors."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseAnomalyDetector


class ZScoreDetector(BaseAnomalyDetector):
    def __init__(self, threshold=3.0, **kwargs):
        self.threshold = threshold
        self._mean = self._std = self._target = None

    def fit(self, data, target="value", **kwargs):
        self._target = target
        self._mean = data[target].mean()
        self._std = data[target].std()

    def detect(self, data=None, **kwargs):
        vals = data[self._target]
        scores = np.abs((vals - self._mean) / max(self._std, 1e-8))
        is_anomaly = scores > self.threshold
        severity = np.where(scores > self.threshold * 2, "critical",
                  np.where(scores > self.threshold * 1.5, "high",
                  np.where(scores > self.threshold, "medium", "low")))
        result = data.copy()
        result["is_anomaly"] = is_anomaly
        result["anomaly_score"] = scores
        result["severity"] = severity
        return result


class IQRDetector(BaseAnomalyDetector):
    def __init__(self, multiplier=1.5, **kwargs):
        self.multiplier = multiplier
        self._q1 = self._q3 = self._iqr = self._target = None

    def fit(self, data, target="value", **kwargs):
        self._target = target
        self._q1 = data[target].quantile(0.25)
        self._q3 = data[target].quantile(0.75)
        self._iqr = self._q3 - self._q1

    def detect(self, data=None, **kwargs):
        vals = data[self._target]
        lower = self._q1 - self.multiplier * self._iqr
        upper = self._q3 + self.multiplier * self._iqr
        is_anomaly = (vals < lower) | (vals > upper)
        scores = np.where(vals < lower, (lower - vals) / max(self._iqr, 1e-8),
                 np.where(vals > upper, (vals - upper) / max(self._iqr, 1e-8), 0))
        result = data.copy()
        result["is_anomaly"] = is_anomaly
        result["anomaly_score"] = scores
        result["severity"] = np.where(scores > 3, "critical", np.where(scores > 2, "high", np.where(scores > 1, "medium", "low")))
        return result
