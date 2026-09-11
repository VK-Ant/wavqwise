"""
Universal Model Adapter
========================
Plug ANY external model or PyPI package into WavqWise.

3 ways to bring your own model:

1. Register any class:
   WavqPipeline.register("my_model", MyModelClass)

2. Pass instance directly:
   pipeline.forecast(model=my_trained_sklearn_model)

3. Use adapters for popular libraries:
   from wavqwise.adapters import NixtlaAdapter, DartsAdapter
   pipeline.forecast(model=NixtlaAdapter("AutoARIMA"))

Your class just needs fit(data, target, time_col) and predict(horizon).
Or use ModelAdapter to wrap anything.
"""
import numpy as np
import pandas as pd
from typing import Any, Callable, Optional
from wavqwise.core.base import BaseForecaster


class ModelAdapter(BaseForecaster):
    """Wrap ANY external model into WavqWise's BaseForecaster interface.

    Usage:
        from wavqwise.core.adapter import ModelAdapter

        # Wrap a sklearn model
        adapter = ModelAdapter.from_sklearn(my_model)

        # Wrap a custom fit/predict pair
        adapter = ModelAdapter.from_functions(fit_fn, predict_fn)

        # Wrap a PyPI package model
        adapter = ModelAdapter.from_class(SomeModelClass, init_kwargs={...})
    """

    def __init__(self, model: Any = None, fit_fn: Optional[Callable] = None,
                 predict_fn: Optional[Callable] = None, name: str = "custom", **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._fit_fn = fit_fn
        self._predict_fn = predict_fn
        self._name = name
        self._target = None
        self._time_col = None
        self._data = None
        self._freq = None

    @classmethod
    def from_sklearn(cls, model, name: str = "sklearn_custom"):
        """Wrap any sklearn-compatible estimator."""
        from wavqwise.forecasters.ml.sklearn_wrapper import SklearnForecaster
        wrapper = SklearnForecaster(model=model)
        wrapper._name = name
        return wrapper

    @classmethod
    def from_functions(cls, fit_fn: Callable, predict_fn: Callable, name: str = "custom"):
        """Wrap fit/predict function pair.

        fit_fn(data: pd.DataFrame, target: str, time_col: str) -> Any
        predict_fn(model: Any, horizon: int) -> pd.DataFrame
        """
        return cls(fit_fn=fit_fn, predict_fn=predict_fn, name=name)

    @classmethod
    def from_class(cls, model_class, init_kwargs: dict = None, name: str = "custom"):
        """Wrap any model class. Must have fit() and predict() methods."""
        model = model_class(**(init_kwargs or {}))
        return cls(model=model, name=name)

    def fit(self, data: pd.DataFrame, target: str = "value",
            time_col: str = "timestamp", **kwargs):
        self._target = target
        self._time_col = time_col
        self._data = data.copy()
        self._freq = pd.infer_freq(data[time_col]) or "D"

        if self._fit_fn:
            self._model = self._fit_fn(data, target, time_col)
        elif self._model and hasattr(self._model, "fit"):
            # Try common fit signatures
            try:
                self._model.fit(data, target=target, time_col=time_col)
            except TypeError:
                try:
                    self._model.fit(data[[target]])
                except Exception:
                    X = data[[target]].values
                    self._model.fit(X)

        self._fitted = True

    def predict(self, horizon: int = 30, confidence_level: float = 0.95,
                **kwargs) -> pd.DataFrame:
        if self._predict_fn:
            result = self._predict_fn(self._model, horizon)
            if isinstance(result, pd.DataFrame):
                return result

        last_date = self._data[self._time_col].max()
        dates = pd.date_range(last_date, periods=horizon + 1, freq=self._freq)[1:]

        if self._model and hasattr(self._model, "predict"):
            try:
                preds = self._model.predict(horizon)
                if isinstance(preds, pd.DataFrame):
                    return preds
                preds = np.array(preds).flatten()[:horizon]
            except Exception:
                preds = np.full(horizon, self._data[self._target].iloc[-1])
        else:
            preds = np.full(horizon, self._data[self._target].iloc[-1])

        std = self._data[self._target].std()
        z = 1.96
        return pd.DataFrame({
            self._time_col: dates[:len(preds)],
            self._target: preds,
            f"{self._target}_lower": preds - z * std,
            f"{self._target}_upper": preds + z * std,
        })
