"""
Real-Time Streaming Engine
===========================
Continuous data ingestion, live forecasting, real-time anomaly alerts.

Usage:
    from wavqwise import WavqPipeline

    pipeline = WavqPipeline()
    pipeline.load(historical_data, target="temperature", time="timestamp")
    pipeline.forecast(horizon=30, model="ema")

    # Start real-time monitoring
    stream = pipeline.stream(
        model="ema",
        anomaly_method="zscore",
        window_size=500,
        on_anomaly=my_alert_function,
        on_forecast=my_dashboard_update,
    )

    # Feed data points as they arrive
    stream.push({"timestamp": "2025-01-01 10:00:00", "temperature": 72.3})
    stream.push(new_dataframe_batch)

    # Or connect to a live source
    stream.connect_csv("live_sensor.csv", poll_interval=5)
    stream.connect_callback(my_data_generator)
"""

import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StreamEvent:
    """Single streaming event with data + metadata."""
    def __init__(self, data: pd.DataFrame, forecast: Optional[pd.DataFrame] = None,
                 anomalies: Optional[pd.DataFrame] = None,
                 timestamp: Optional[datetime] = None):
        self.data = data
        self.forecast = forecast
        self.anomalies = anomalies
        self.timestamp = timestamp or datetime.now()
        self.has_anomaly = anomalies is not None and len(anomalies) > 0

    def __repr__(self):
        n_anom = len(self.anomalies) if self.anomalies is not None else 0
        return f"StreamEvent(rows={len(self.data)}, anomalies={n_anom})"


class StreamConfig:
    """Configuration for streaming pipeline."""
    def __init__(self, model: str = "ema", anomaly_method: str = "zscore",
                 window_size: int = 500, forecast_horizon: int = 30,
                 anomaly_threshold: float = 3.0, refit_every: int = 100,
                 forecast_every: int = 10):
        self.model = model
        self.anomaly_method = anomaly_method
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.anomaly_threshold = anomaly_threshold
        self.refit_every = refit_every
        self.forecast_every = forecast_every


class StreamingEngine:
    """Real-time streaming forecasting and anomaly detection.

    Core loop:
    1. Receive data point(s)
    2. Append to sliding window
    3. Run anomaly detection on new point(s)
    4. If anomaly → fire on_anomaly callback
    5. Every N points → update forecast
    6. Fire on_forecast callback with new prediction
    """

    def __init__(self, pipeline, config: StreamConfig,
                 on_anomaly: Optional[Callable] = None,
                 on_forecast: Optional[Callable] = None,
                 on_data: Optional[Callable] = None):

        self._pipeline = pipeline
        self._config = config
        self._on_anomaly = on_anomaly or self._default_anomaly_handler
        self._on_forecast = on_forecast
        self._on_data = on_data

        self._window: Optional[pd.DataFrame] = None
        self._target = pipeline._target
        self._time_col = pipeline._time_col
        self._count = 0
        self._anomaly_count = 0
        self._running = False
        self._thread = None

        # Initialize from pipeline data
        if pipeline._data is not None:
            self._window = pipeline._data.tail(config.window_size).copy()

        # Anomaly detector
        from wavqwise.anomaly.statistical import ZScoreDetector, IQRDetector
        detectors = {"zscore": ZScoreDetector, "iqr": IQRDetector}
        detector_cls = detectors.get(config.anomaly_method, ZScoreDetector)
        self._detector = detector_cls(threshold=config.anomaly_threshold)
        if self._window is not None and len(self._window) > 10:
            self._detector.fit(self._window, target=self._target)

        self._latest_forecast = None

    def push(self, data: Union[dict, pd.DataFrame, pd.Series]):
        """Push new data point(s) into the stream.

        Args:
            data: dict, DataFrame, or Series with new observation(s)
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = pd.DataFrame([data])

        if self._time_col in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[self._time_col]):
                data[self._time_col] = pd.to_datetime(data[self._time_col])

        # Append to window
        if self._window is None:
            self._window = data.copy()
        else:
            self._window = pd.concat([self._window, data], ignore_index=True)

        # Trim to window size
        if len(self._window) > self._config.window_size:
            self._window = self._window.tail(self._config.window_size).reset_index(drop=True)

        self._count += len(data)

        # Anomaly detection on new points
        anomalies = self._check_anomalies(data)

        # Fire data callback
        if self._on_data:
            self._on_data(StreamEvent(data, anomalies=anomalies))

        # Periodic forecast update
        if self._count % self._config.forecast_every == 0:
            self._update_forecast()

        # Periodic refit
        if self._count % self._config.refit_every == 0:
            self._refit()

    def push_batch(self, data: pd.DataFrame):
        """Push a batch of data points."""
        for _, row in data.iterrows():
            self.push(row)

    def _check_anomalies(self, new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Run anomaly detection on new data points."""
        if self._target not in new_data.columns:
            return None

        try:
            if len(self._window) > 20:
                self._detector.fit(self._window, target=self._target)
            result = self._detector.detect(new_data)
            anomalies = result[result["is_anomaly"]]

            if len(anomalies) > 0:
                self._anomaly_count += len(anomalies)
                event = StreamEvent(new_data, anomalies=anomalies)
                self._on_anomaly(event)
                return anomalies
        except Exception as e:
            logger.debug(f"Anomaly check failed: {e}")

        return None

    def _update_forecast(self):
        """Generate updated forecast from current window."""
        if self._window is None or len(self._window) < 10:
            return

        try:
            self._pipeline._data = self._window.copy()
            self._pipeline._processed_data = None
            self._pipeline._is_fitted = False
            result = self._pipeline.forecast(
                horizon=self._config.forecast_horizon,
                model=self._config.model,
            )
            self._latest_forecast = result

            if self._on_forecast:
                self._on_forecast(result)
        except Exception as e:
            logger.debug(f"Forecast update failed: {e}")

    def _refit(self):
        """Refit model on current window."""
        if self._window is not None and len(self._window) > 20:
            self._pipeline._is_fitted = False
            logger.info(f"Refit at count={self._count}, window={len(self._window)}")

    def _default_anomaly_handler(self, event: StreamEvent):
        """Default anomaly alert: print to console."""
        for _, row in event.anomalies.iterrows():
            severity = row.get("severity", "unknown")
            score = row.get("anomaly_score", 0)
            value = row.get(self._target, "?")
            print(f"[ALERT] Anomaly detected: {self._target}={value}, "
                  f"severity={severity}, score={score:.2f}")

    def connect_callback(self, data_generator: Callable, interval: float = 1.0):
        """Connect to a data generator function.

        data_generator() should return a dict or DataFrame with new data.
        Called every `interval` seconds.
        """
        self._running = True

        def _loop():
            while self._running:
                try:
                    data = data_generator()
                    if data is not None:
                        self.push(data)
                except Exception as e:
                    logger.error(f"Generator error: {e}")
                time.sleep(interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        logger.info(f"Streaming started (interval={interval}s)")

    def connect_csv(self, path: str, poll_interval: float = 5.0):
        """Watch a CSV file for new rows (tail -f style)."""
        self._running = True
        last_size = 0

        def _poll():
            nonlocal last_size
            while self._running:
                try:
                    df = pd.read_csv(path)
                    if len(df) > last_size:
                        new_rows = df.iloc[last_size:]
                        if self._time_col in new_rows.columns:
                            new_rows[self._time_col] = pd.to_datetime(new_rows[self._time_col])
                        self.push_batch(new_rows)
                        last_size = len(df)
                except Exception as e:
                    logger.debug(f"CSV poll error: {e}")
                time.sleep(poll_interval)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()
        logger.info(f"Watching {path} (interval={poll_interval}s)")

    def stop(self):
        """Stop streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Streaming stopped")

    @property
    def latest_forecast(self):
        return self._latest_forecast

    @property
    def stats(self) -> dict:
        return {
            "points_processed": self._count,
            "anomalies_detected": self._anomaly_count,
            "window_size": len(self._window) if self._window is not None else 0,
            "model": self._config.model,
            "anomaly_method": self._config.anomaly_method,
            "running": self._running,
        }

    def summary(self) -> str:
        s = self.stats
        return (
            f"Stream: {s['points_processed']} points | "
            f"{s['anomalies_detected']} anomalies | "
            f"window={s['window_size']} | "
            f"model={s['model']} | "
            f"{'LIVE' if s['running'] else 'stopped'}"
        )
