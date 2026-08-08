from wavqwise.core.base import BaseForecaster
class NBEATSForecaster(BaseForecaster):
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[neural]")
    def predict(self, **kwargs): pass
