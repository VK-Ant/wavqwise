"""Isolation Forest anomaly detector."""
import numpy as np
import pandas as pd
from wavqwise.core.base import BaseAnomalyDetector

class IsolationForestDetector(BaseAnomalyDetector):
    def __init__(self, contamination=0.05, **kwargs):
        self.contamination = contamination
        self._model = None
        self._target = None

    def fit(self, data, target="value", **kwargs):
        from sklearn.ensemble import IsolationForest
        self._target = target
        self._model = IsolationForest(contamination=self.contamination, random_state=42)
        self._model.fit(data[[target]])

    def detect(self, data=None, **kwargs):
        scores = self._model.decision_function(data[[self._target]])
        preds = self._model.predict(data[[self._target]])
        result = data.copy()
        result["is_anomaly"] = preds == -1
        result["anomaly_score"] = -scores
        result["severity"] = np.where(-scores > 0.3, "critical", np.where(-scores > 0.2, "high", np.where(-scores > 0.1, "medium", "low")))
        return result
