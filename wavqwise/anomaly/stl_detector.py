"""STL decomposition anomaly detector."""
from wavqwise.core.base import BaseAnomalyDetector
class STLDetector(BaseAnomalyDetector):
    def fit(self, data, **kwargs): self._target = kwargs.get("target", "value")
    def detect(self, data=None, **kwargs):
        from wavqwise.anomaly.statistical import ZScoreDetector
        d = ZScoreDetector(); d.fit(data, self._target); return d.detect(data)
