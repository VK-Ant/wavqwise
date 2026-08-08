from wavqwise.core.base import BaseForecaster
class OllamaForecaster(BaseForecaster):
    def __init__(self, model_name="llama3", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[cloud]")
    def predict(self, **kwargs): pass
