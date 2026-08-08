from wavqwise.core.base import BaseAnomalyDetector
class DBSCANDetector(BaseAnomalyDetector):
    def fit(self, data, **kwargs): pass
    def detect(self, data=None, **kwargs):
        from wavqwise.anomaly.statistical import ZScoreDetector
        d = ZScoreDetector(); d.fit(data, kwargs.get("target", "value")); return d.detect(data)
