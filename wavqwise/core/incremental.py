"""
Incremental training engine.
Models update with new data WITHOUT full retraining.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class IncrementalEngine:
    """Manages incremental model updates.

    Strategy varies by model type:
    - Statistical (ARIMA, ETS): append observations, update parameters
    - ML (XGBoost, RF): warm-start or partial_fit
    - Neural: fine-tune on new batch
    - Foundation: zero-shot, no training needed
    - Cloud/LLM: prompt with latest data
    """

    def __init__(self, window_size: Optional[int] = None):
        self._history = None
        self._window_size = window_size
        self._update_count = 0

    @property
    def history(self) -> Optional[pd.DataFrame]:
        return self._history

    @property
    def update_count(self) -> int:
        return self._update_count

    def initialize(self, data: pd.DataFrame) -> pd.DataFrame:
        self._history = data.copy()
        self._update_count = 0
        return self._history

    def update(self, new_data: pd.DataFrame, forecaster: Any) -> pd.DataFrame:
        if self._history is None:
            raise ValueError("Engine not initialized. Call initialize() first.")

        self._history = pd.concat([self._history, new_data], ignore_index=True)

        if self._window_size and len(self._history) > self._window_size:
            self._history = self._history.tail(self._window_size).reset_index(drop=True)

        if hasattr(forecaster, "update"):
            try:
                forecaster.update(new_data)
                logger.info(
                    f"Incremental update #{self._update_count + 1}: "
                    f"{len(new_data)} new observations"
                )
            except NotImplementedError:
                logger.info("Model does not support incremental update, refitting on window")
                forecaster.fit(self._history)
        else:
            forecaster.fit(self._history)

        self._update_count += 1
        return self._history

    def get_training_data(self) -> pd.DataFrame:
        if self._history is None:
            raise ValueError("Engine not initialized.")
        return self._history.copy()
