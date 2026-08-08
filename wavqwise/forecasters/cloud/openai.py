from wavqwise.core.base import BaseForecaster
class OpenAIForecaster(BaseForecaster):
    def __init__(self, model_name="gpt-4o", **kwargs):
        super().__init__(**kwargs)
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[cloud]")
    def predict(self, **kwargs): pass
