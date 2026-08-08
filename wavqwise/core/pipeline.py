"""
WavqPipeline - The ONLY class users need.

from wavqwise import WavqPipeline

pipeline = WavqPipeline()
pipeline.load("data.csv", target="sales", time="date")
forecast = pipeline.forecast(horizon=30, model="arima")
forecast.plot()
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from wavqwise.core.config import PipelineConfig
from wavqwise.core.incremental import IncrementalEngine
from wavqwise.core.registry import Registry
from wavqwise.loaders.base import BaseLoader
from wavqwise.loaders.auto_loader import AutoLoader
from wavqwise.preprocessors.auto_preprocessor import AutoPreprocessor
from wavqwise.features.auto_features import AutoFeatureEngineer
from wavqwise.evaluation.metrics import Metrics
from wavqwise.visualization.forecast_plot import ForecastPlot

logger = logging.getLogger(__name__)


class ForecastResult:
    """Container for forecast results with built-in plotting."""

    def __init__(self, forecast_df: pd.DataFrame, history: pd.DataFrame,
                 model_name: str, metrics: Optional[Dict] = None,
                 target: str = "value", time_col: str = "timestamp"):
        self.forecast = forecast_df
        self.history = history
        self.model_name = model_name
        self.metrics = metrics or {}
        self.target = target
        self.time_col = time_col

    def plot(self, engine: str = "matplotlib", show_confidence: bool = True,
             show_components: bool = False, style: str = "default", **kwargs):
        plotter = ForecastPlot(engine=engine, style=style)
        return plotter.plot(
            history=self.history,
            forecast=self.forecast,
            target=self.target,
            time_col=self.time_col,
            model_name=self.model_name,
            show_confidence=show_confidence,
            show_components=show_components,
            **kwargs,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return self.forecast.copy()

    def summary(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Horizon: {len(self.forecast)} steps",
        ]
        if self.metrics:
            for k, v in self.metrics.items():
                lines.append(f"{k}: {v:.4f}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"ForecastResult(model='{self.model_name}', "
            f"horizon={len(self.forecast)}, "
            f"metrics={self.metrics})"
        )


class WavqPipeline:
    """Main pipeline. One import, one class, any model.

    Usage:
        pipeline = WavqPipeline()
        pipeline.load("data.csv", target="revenue", time="date")
        forecast = pipeline.forecast(horizon=30, model="arima")
        forecast.plot()

        # Incremental update
        pipeline.update(new_data)
        forecast = pipeline.forecast(horizon=30)
    """

    def __init__(self, config: Optional[PipelineConfig] = None,
                 device: Optional[str] = None, **kwargs):
        self.config = config or PipelineConfig(**kwargs)
        self._data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None
        self._time_col: Optional[str] = None
        self._frequency: Optional[str] = None
        self._forecaster: Optional[Any] = None

        # Auto-detect runtime (GPU/ONNX/TensorRT/CPU)
        try:
            from wavqwise.runtime.engine import RuntimeEngine
            self._runtime = RuntimeEngine(preferred=device)
        except Exception:
            self._runtime = None
        self._model_name: Optional[str] = None
        self._incremental = IncrementalEngine(window_size=self.config.extra.get("window_size"))
        self._preprocessor = AutoPreprocessor()
        self._feature_engineer = AutoFeatureEngineer()
        self._loader = AutoLoader()
        self._is_fitted = False

    # === Loading ===

    def load(self, source: Union[str, pd.DataFrame], target: Optional[str] = None,
             time: Optional[str] = None, **kwargs) -> "WavqPipeline":
        self._target = target or self.config.target
        self._time_col = time or self.config.time_col

        if isinstance(source, pd.DataFrame):
            self._data = source.copy()
        elif isinstance(source, str):
            # Database connection string
            if "://" in source:
                self._data = self._load_from_database(source, **kwargs)
            else:
                # File path
                self._data = self._loader.load(source, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Auto-detect columns if not specified
        self._auto_detect_columns()

        # Auto-detect frequency
        self._frequency = self._detect_frequency()

        # Initialize incremental engine
        self._incremental.initialize(self._data)

        if self.config.verbose:
            logger.info(
                f"Loaded {len(self._data)} rows | "
                f"target='{self._target}' | time='{self._time_col}' | "
                f"frequency='{self._frequency}'"
            )

        return self

    def _load_from_database(self, connection_string: str, **kwargs) -> pd.DataFrame:
        scheme = connection_string.split("://")[0].lower()
        db_type_map = {
            "sqlite": "sqlite",
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "mysql": "mysql",
            "mongodb": "mongodb",
            "timescaledb": "timescaledb",
            "influxdb": "influxdb",
        }
        db_type = db_type_map.get(scheme)
        if db_type is None:
            raise ValueError(f"Unsupported database: {scheme}")

        connector = Registry.get_db_connector(db_type, connection_string=connection_string)
        return connector.load(**kwargs)

    def _auto_detect_columns(self):
        if self._data is None:
            return

        # Auto-detect time column
        if self._time_col is None:
            date_cols = self._data.select_dtypes(include=["datetime64"]).columns
            if len(date_cols) > 0:
                self._time_col = date_cols[0]
            else:
                for col in self._data.columns:
                    try:
                        pd.to_datetime(self._data[col])
                        self._time_col = col
                        break
                    except (ValueError, TypeError):
                        continue

            if self._time_col is None:
                self._time_col = self._data.columns[0]

        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self._data[self._time_col]):
            self._data[self._time_col] = pd.to_datetime(self._data[self._time_col])

        # Auto-detect target column
        if self._target is None:
            numeric_cols = self._data.select_dtypes(include=[np.number]).columns
            non_time = [c for c in numeric_cols if c != self._time_col]
            if non_time:
                self._target = non_time[0]
            else:
                raise ValueError("No numeric target column found. Specify target= parameter.")

        # Sort by time
        self._data = self._data.sort_values(self._time_col).reset_index(drop=True)

    def _detect_frequency(self) -> str:
        if self._data is None or self._time_col is None:
            return "unknown"
        try:
            freq = pd.infer_freq(self._data[self._time_col])
            return freq or "unknown"
        except (ValueError, TypeError):
            return "unknown"

    # === Preprocessing ===

    def preprocess(self, **kwargs) -> "WavqPipeline":
        if self._data is None:
            raise ValueError("No data loaded. Call .load() first.")
        self._processed_data = self._preprocessor.transform(
            self._data.copy(), target=self._target, time_col=self._time_col, **kwargs
        )
        return self

    # === Forecasting ===

    def forecast(self, horizon: Optional[int] = None,
                 model: Optional[str] = None,
                 confidence_level: Optional[float] = None,
                 **kwargs) -> ForecastResult:
        if self._data is None:
            raise ValueError("No data loaded. Call .load() first.")

        horizon = horizon or self.config.horizon
        model_name = model or self.config.model
        confidence = confidence_level or self.config.confidence_level

        # Auto preprocess if needed
        if self._processed_data is None and self.config.auto_preprocess:
            self.preprocess()

        data = self._processed_data if self._processed_data is not None else self._data

        # Handle model selection
        if model_name == "auto":
            return self._auto_forecast(data, horizon, confidence, **kwargs)

        # Handle ensemble
        if isinstance(model_name, list):
            return self._ensemble_forecast(data, model_name, horizon, confidence, **kwargs)

        # Get or create forecaster
        if self._forecaster is None or self._model_name != model_name:
            if isinstance(model_name, str):
                self._forecaster = Registry.get_forecaster(model_name, **kwargs)
            else:
                # Custom forecaster instance passed directly
                self._forecaster = model_name
            self._model_name = str(model_name)
            self._is_fitted = False

        # Fit if needed
        if not self._is_fitted:
            self._forecaster.fit(
                data, target=self._target, time_col=self._time_col
            )
            self._is_fitted = True

        # Predict
        forecast_df = self._forecaster.predict(
            horizon=horizon, confidence_level=confidence
        )

        # Evaluate on held-out data if possible
        metrics = self._evaluate_if_possible(data, horizon)

        return ForecastResult(
            forecast_df=forecast_df,
            history=data,
            model_name=self._model_name,
            metrics=metrics,
            target=self._target,
            time_col=self._time_col,
        )

    def _auto_forecast(self, data, horizon, confidence, **kwargs) -> ForecastResult:
        candidates = ["moving_average", "arima", "ets", "holtwinters"]
        best_model = None
        best_score = float("inf")
        best_result = None

        for name in candidates:
            try:
                forecaster = Registry.get_forecaster(name)
                forecaster.fit(data, target=self._target, time_col=self._time_col)
                score = self._quick_eval(forecaster, data, horizon)
                if score < best_score:
                    best_score = score
                    best_model = name
                    self._forecaster = forecaster
                    self._model_name = name
                    self._is_fitted = True
            except Exception as e:
                logger.debug(f"Auto-select: {name} failed: {e}")
                continue

        if best_model is None:
            raise RuntimeError("Auto model selection failed. Try specifying a model manually.")

        logger.info(f"Auto-selected: {best_model} (score: {best_score:.4f})")

        forecast_df = self._forecaster.predict(
            horizon=horizon, confidence_level=confidence
        )
        return ForecastResult(
            forecast_df=forecast_df,
            history=data,
            model_name=f"auto({best_model})",
            metrics={"auto_score": best_score},
            target=self._target,
            time_col=self._time_col,
        )

    def _ensemble_forecast(self, data, model_names, horizon, confidence,
                           ensemble_method="weighted", **kwargs) -> ForecastResult:
        forecasts = []
        weights = []

        for name in model_names:
            try:
                forecaster = Registry.get_forecaster(name)
                forecaster.fit(data, target=self._target, time_col=self._time_col)
                pred = forecaster.predict(horizon=horizon, confidence_level=confidence)
                score = self._quick_eval(forecaster, data, min(horizon, len(data) // 5))
                forecasts.append(pred)
                weights.append(1.0 / max(score, 1e-8))
            except Exception as e:
                logger.warning(f"Ensemble: {name} failed: {e}")

        if not forecasts:
            raise RuntimeError("All ensemble models failed.")

        # Weighted average
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        combined = forecasts[0].copy()
        combined[self._target] = sum(
            w * f[self._target].values for w, f in zip(weights, forecasts)
        )

        return ForecastResult(
            forecast_df=combined,
            history=data,
            model_name=f"ensemble({','.join(model_names)})",
            target=self._target,
            time_col=self._time_col,
        )

    def _quick_eval(self, forecaster, data, horizon) -> float:
        n = len(data)
        if n < horizon + 10:
            return float("inf")
        train = data.iloc[:-horizon]
        test = data.iloc[-horizon:]
        forecaster.fit(train, target=self._target, time_col=self._time_col)
        pred = forecaster.predict(horizon=horizon)
        actual = test[self._target].values[:len(pred)]
        predicted = pred[self._target].values[:len(actual)]
        return Metrics.mae(actual, predicted)

    def _evaluate_if_possible(self, data, horizon) -> Dict:
        return {}

    # === Incremental Update ===

    def update(self, new_data: Union[pd.DataFrame, str], **kwargs) -> "WavqPipeline":
        if isinstance(new_data, str):
            new_data = self._loader.load(new_data, **kwargs)

        if self._time_col and self._time_col in new_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(new_data[self._time_col]):
                new_data[self._time_col] = pd.to_datetime(new_data[self._time_col])

        self._data = self._incremental.update(new_data, self._forecaster)
        self._processed_data = None  # Re-process on next forecast

        if self.config.verbose:
            logger.info(
                f"Incremental update #{self._incremental.update_count}: "
                f"+{len(new_data)} rows, total={len(self._data)}"
            )

        return self

    # === Model Comparison ===

    def compare_models(self, models: Optional[List[str]] = None,
                       horizon: Optional[int] = None) -> pd.DataFrame:
        models = models or ["moving_average", "arima", "ets", "holtwinters"]
        horizon = horizon or self.config.horizon
        data = self._processed_data if self._processed_data is not None else self._data

        results = []
        for name in models:
            try:
                forecaster = Registry.get_forecaster(name)
                mae = self._quick_eval(forecaster, data, horizon)
                results.append({"model": name, "MAE": mae})
            except Exception as e:
                results.append({"model": name, "MAE": float("nan"), "error": str(e)})

        return pd.DataFrame(results).sort_values("MAE")

    # === Visualization ===

    def plot(self, **kwargs):
        if self._data is None:
            raise ValueError("No data loaded.")
        plotter = ForecastPlot(engine=self.config.plot_engine, style=self.config.plot_style)
        return plotter.plot_series(
            self._data, target=self._target, time_col=self._time_col, **kwargs
        )

    def dashboard(self, output: str = "dashboard.html", **kwargs):
        from wavqwise.visualization.dashboard import DashboardBuilder
        builder = DashboardBuilder()
        builder.build(self._data, target=self._target, time_col=self._time_col,
                      output=output, **kwargs)

    # === Export ===

    def save(self, path: str):
        import joblib
        state = {
            "config": self.config,
            "forecaster": self._forecaster,
            "model_name": self._model_name,
            "target": self._target,
            "time_col": self._time_col,
            "frequency": self._frequency,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(state, path)
        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load_pipeline(cls, path: str) -> "WavqPipeline":
        import joblib
        state = joblib.load(path)
        pipeline = cls(config=state["config"])
        pipeline._forecaster = state["forecaster"]
        pipeline._model_name = state["model_name"]
        pipeline._target = state["target"]
        pipeline._time_col = state["time_col"]
        pipeline._frequency = state["frequency"]
        pipeline._is_fitted = state["is_fitted"]
        return pipeline

    # === Info ===

    @staticmethod
    def available_models() -> List[str]:
        return Registry.list_forecasters()

    def info(self) -> str:
        runtime_str = self._runtime.device if self._runtime else "cpu"
        lines = [
            "WavqPipeline Status:",
            f"  Runtime: {runtime_str}",
            f"  Data loaded: {self._data is not None}",
            f"  Rows: {len(self._data) if self._data is not None else 0}",
            f"  Target: {self._target}",
            f"  Time column: {self._time_col}",
            f"  Frequency: {self._frequency}",
            f"  Model: {self._model_name}",
            f"  Fitted: {self._is_fitted}",
            f"  Updates: {self._incremental.update_count}",
        ]
        return "\n".join(lines)

    def runtime_info(self) -> str:
        """Full hardware/runtime report."""
        if self._runtime:
            return self._runtime.summary()
        return "Runtime: CPU (no GPU/ONNX detected)"

    def __repr__(self):
        return f"WavqPipeline(model='{self._model_name}', fitted={self._is_fitted})"
