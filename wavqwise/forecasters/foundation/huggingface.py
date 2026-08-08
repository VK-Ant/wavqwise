from wavqwise.core.base import BaseForecaster
class HuggingFaceForecaster(BaseForecaster):
    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
    def fit(self, data, **kwargs): raise ImportError("pip install wavqwise[foundation]")
    def predict(self, **kwargs): pass
