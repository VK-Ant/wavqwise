"""Gradient boosting forecasters: XGBoost, LightGBM, CatBoost."""
from wavqwise.forecasters.ml.sklearn_wrapper import SklearnForecaster

class XGBoostForecaster(SklearnForecaster):
    def __init__(self, n_estimators=100, **kwargs):
        try:
            from xgboost import XGBRegressor
            super().__init__(model=XGBRegressor(n_estimators=n_estimators, random_state=42, verbosity=0), **kwargs)
        except ImportError:
            raise ImportError("pip install wavqwise[ml]")

class LightGBMForecaster(SklearnForecaster):
    def __init__(self, n_estimators=100, **kwargs):
        try:
            from lightgbm import LGBMRegressor
            super().__init__(model=LGBMRegressor(n_estimators=n_estimators, random_state=42, verbose=-1), **kwargs)
        except ImportError:
            raise ImportError("pip install wavqwise[ml]")

class CatBoostForecaster(SklearnForecaster):
    def __init__(self, n_estimators=100, **kwargs):
        try:
            from catboost import CatBoostRegressor
            super().__init__(model=CatBoostRegressor(iterations=n_estimators, random_seed=42, verbose=0), **kwargs)
        except ImportError:
            raise ImportError("pip install catboost")
