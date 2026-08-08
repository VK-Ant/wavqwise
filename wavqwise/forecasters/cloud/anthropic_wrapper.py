from wavqwise.core.base import BaseForecaster
class AnthropicForecaster(BaseForecaster):
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[cloud]")
    def predict(self, **kwargs): pass
