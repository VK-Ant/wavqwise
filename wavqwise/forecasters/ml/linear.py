"""Linear forecasters."""
from wavqwise.forecasters.ml.sklearn_wrapper import SklearnForecaster

class RidgeForecaster(SklearnForecaster):
    def __init__(self, alpha=1.0, **kwargs):
        from sklearn.linear_model import Ridge
        super().__init__(model=Ridge(alpha=alpha), **kwargs)

class LassoForecaster(SklearnForecaster):
    def __init__(self, alpha=1.0, **kwargs):
        from sklearn.linear_model import Lasso
        super().__init__(model=Lasso(alpha=alpha), **kwargs)

class ElasticNetForecaster(SklearnForecaster):
    def __init__(self, alpha=1.0, l1_ratio=0.5, **kwargs):
        from sklearn.linear_model import ElasticNet
        super().__init__(model=ElasticNet(alpha=alpha, l1_ratio=l1_ratio), **kwargs)
