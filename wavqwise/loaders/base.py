"""Base loader interface."""
from abc import ABC, abstractmethod
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        pass
