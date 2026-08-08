"""Base interfaces for all pluggable components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


class BaseForecaster(ABC):
    """Base interface for all forecasting backends.
    Every model - ARIMA, XGBoost, Chronos, Ollama - implements this.
    User never sees this class. WavqPipeline uses it internally.
    """

    def __init__(self, **kwargs):
        self._fitted = False
        self._params = kwargs

    @abstractmethod
    def fit(self, data: pd.DataFrame, target: str = "value",
            time_col: str = "timestamp", **kwargs):
        """Initial training on historical data."""
        pass

    @abstractmethod
    def predict(self, horizon: int = 30,
                confidence_level: float = 0.95, **kwargs) -> pd.DataFrame:
        """Generate forecast. Returns DataFrame with columns:
        [time_col, target, target_lower, target_upper]
        """
        pass

    def update(self, new_data: pd.DataFrame, **kwargs):
        """Incremental update. Override for true incremental learning.
        Default: raises NotImplementedError (triggers full refit in IncrementalEngine).
        """
        raise NotImplementedError("This model does not support incremental update.")

    def save(self, path: str):
        """Persist trained state."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        """Load previously trained state."""
        import joblib
        return joblib.load(path)


class BasePreprocessor(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame, target: str, time_col: str,
                  **kwargs) -> pd.DataFrame:
        pass

    def fit_transform(self, data: pd.DataFrame, target: str, time_col: str,
                      **kwargs) -> pd.DataFrame:
        return self.transform(data, target, time_col, **kwargs)


class BaseFeatureEngineer(ABC):
    @abstractmethod
    def generate(self, data: pd.DataFrame, target: str, time_col: str,
                 **kwargs) -> pd.DataFrame:
        pass


class BaseAnomalyDetector(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: str, **kwargs):
        pass

    @abstractmethod
    def detect(self, data: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """Returns DataFrame with columns: [timestamp, value, is_anomaly, severity, score]"""
        pass


class BaseAlertHandler(ABC):
    @abstractmethod
    def send(self, anomaly: Dict[str, Any], **kwargs):
        pass


class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        pass


class BaseDBConnector(ABC):
    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def save(self, data: pd.DataFrame, **kwargs):
        pass


class BaseIndicator(ABC):
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        pass


class BaseSignalProcessor(ABC):
    @abstractmethod
    def process(self, data: np.ndarray, sample_rate: float, **kwargs) -> Dict[str, Any]:
        pass


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray, **kwargs) -> Dict[str, float]:
        pass


class BaseVisualizer(ABC):
    @abstractmethod
    def plot(self, **kwargs):
        pass
