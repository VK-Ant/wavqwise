from wavqwise.core.base import BaseForecaster
class TFTForecaster(BaseForecaster):
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[neural]")
    def predict(self, **kwargs): pass
