"""
Registry - Pluggable backend discovery and management.
User never imports models directly. Registry handles everything.
"""

from typing import Any, Dict, Optional, Type
import importlib
import logging

logger = logging.getLogger(__name__)


class Registry:
    """Central registry for all pluggable backends.

    User specifies model="arima" or model="chronos" as a string.
    Registry resolves it to the correct class, handles imports,
    and returns an initialized instance.
    """

    _forecasters: Dict[str, Dict[str, Any]] = {}
    _anomaly_detectors: Dict[str, Dict[str, Any]] = {}
    _preprocessors: Dict[str, Dict[str, Any]] = {}
    _loaders: Dict[str, Dict[str, Any]] = {}
    _indicators: Dict[str, Dict[str, Any]] = {}
    _db_connectors: Dict[str, Dict[str, Any]] = {}
    _visualizers: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_forecaster(cls, name: str, module_path: str, class_name: str,
                            requires: Optional[str] = None):
        cls._forecasters[name.lower()] = {
            "module": module_path,
            "class": class_name,
            "requires": requires,
        }

    @classmethod
    def register_anomaly_detector(cls, name: str, module_path: str, class_name: str,
                                   requires: Optional[str] = None):
        cls._anomaly_detectors[name.lower()] = {
            "module": module_path,
            "class": class_name,
            "requires": requires,
        }

    @classmethod
    def register_indicator(cls, name: str, module_path: str, class_name: str,
                           requires: Optional[str] = None):
        cls._indicators[name.lower()] = {
            "module": module_path,
            "class": class_name,
            "requires": requires,
        }

    @classmethod
    def register_db_connector(cls, name: str, module_path: str, class_name: str,
                               requires: Optional[str] = None):
        cls._db_connectors[name.lower()] = {
            "module": module_path,
            "class": class_name,
            "requires": requires,
        }

    @classmethod
    def get_forecaster(cls, name: str, **kwargs) -> Any:
        return cls._resolve("forecaster", cls._forecasters, name, **kwargs)

    @classmethod
    def get_anomaly_detector(cls, name: str, **kwargs) -> Any:
        return cls._resolve("anomaly_detector", cls._anomaly_detectors, name, **kwargs)

    @classmethod
    def get_indicator(cls, name: str, **kwargs) -> Any:
        return cls._resolve("indicator", cls._indicators, name, **kwargs)

    @classmethod
    def get_db_connector(cls, name: str, **kwargs) -> Any:
        return cls._resolve("db_connector", cls._db_connectors, name, **kwargs)

    @classmethod
    def _resolve(cls, kind: str, registry: Dict, name: str, **kwargs) -> Any:
        name_lower = name.lower()

        # Handle HuggingFace models: "huggingface:model/name"
        if name_lower.startswith("huggingface:"):
            model_id = name.split(":", 1)[1]
            entry = registry.get("huggingface")
            if entry is None:
                raise ValueError(
                    f"HuggingFace {kind} backend not registered. "
                    f"Install: pip install wavqwise[foundation]"
                )
            return cls._instantiate(entry, model_id=model_id, **kwargs)

        # Handle Ollama models: "ollama:model_name"
        if name_lower.startswith("ollama:"):
            model_name = name.split(":", 1)[1]
            entry = registry.get("ollama")
            if entry is None:
                raise ValueError(
                    f"Ollama {kind} backend not registered. "
                    f"Install: pip install wavqwise[cloud]"
                )
            return cls._instantiate(entry, model_name=model_name, **kwargs)

        # Handle OpenAI models: "openai:model_name"
        if name_lower.startswith("openai:"):
            model_name = name.split(":", 1)[1]
            entry = registry.get("openai")
            if entry is None:
                raise ValueError(
                    f"OpenAI {kind} backend not registered. "
                    f"Install: pip install wavqwise[cloud]"
                )
            return cls._instantiate(entry, model_name=model_name, **kwargs)

        # Direct lookup
        if name_lower not in registry:
            available = list(registry.keys())
            raise ValueError(
                f"Unknown {kind}: '{name}'. Available: {available}. "
                f"Or use a custom instance implementing Base{kind.title().replace('_', '')}."
            )

        entry = registry[name_lower]
        return cls._instantiate(entry, **kwargs)

    @classmethod
    def _instantiate(cls, entry: Dict, **kwargs) -> Any:
        requires = entry.get("requires")
        module_path = entry["module"]
        class_name = entry["class"]

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            install_hint = f"pip install wavqwise[{requires}]" if requires else str(e)
            raise ImportError(
                f"Could not import {module_path}. Install required: {install_hint}"
            ) from e

        klass = getattr(module, class_name)
        return klass(**kwargs)

    @classmethod
    def list_forecasters(cls) -> list:
        return sorted(cls._forecasters.keys())

    @classmethod
    def list_anomaly_detectors(cls) -> list:
        return sorted(cls._anomaly_detectors.keys())

    @classmethod
    def list_indicators(cls) -> list:
        return sorted(cls._indicators.keys())

    @classmethod
    def list_db_connectors(cls) -> list:
        return sorted(cls._db_connectors.keys())


# === Auto-register all built-in backends ===

# Traditional forecasters
_TRADITIONAL = [
    ("moving_average", "wavqwise.forecasters.traditional.moving_average", "MovingAverageForecaster"),
    ("sma", "wavqwise.forecasters.traditional.moving_average", "MovingAverageForecaster"),
    ("ema", "wavqwise.forecasters.traditional.moving_average", "EMAForecaster"),
    ("arima", "wavqwise.forecasters.traditional.arima", "ARIMAForecaster"),
    ("auto_arima", "wavqwise.forecasters.traditional.arima", "AutoARIMAForecaster"),
    ("sarima", "wavqwise.forecasters.traditional.sarima", "SARIMAForecaster"),
    ("auto_sarima", "wavqwise.forecasters.traditional.sarima", "AutoSARIMAForecaster"),
    ("ets", "wavqwise.forecasters.traditional.ets", "ETSForecaster"),
    ("holtwinters", "wavqwise.forecasters.traditional.holtwinters", "HoltWintersForecaster"),
    ("theta", "wavqwise.forecasters.traditional.theta", "ThetaForecaster"),
    ("ces", "wavqwise.forecasters.traditional.ces", "CESForecaster"),
    ("croston", "wavqwise.forecasters.traditional.croston", "CrostonForecaster"),
    ("naive", "wavqwise.forecasters.traditional.naive", "NaiveForecaster"),
    ("seasonal_naive", "wavqwise.forecasters.traditional.naive", "SeasonalNaiveForecaster"),
]

for name, mod, cls_name in _TRADITIONAL:
    Registry.register_forecaster(name, mod, cls_name, requires="traditional")

# ML forecasters
_ML = [
    ("xgboost", "wavqwise.forecasters.ml.gradient_boost", "XGBoostForecaster"),
    ("lightgbm", "wavqwise.forecasters.ml.gradient_boost", "LightGBMForecaster"),
    ("catboost", "wavqwise.forecasters.ml.gradient_boost", "CatBoostForecaster"),
    ("random_forest", "wavqwise.forecasters.ml.sklearn_wrapper", "RandomForestForecaster"),
    ("ridge", "wavqwise.forecasters.ml.linear", "RidgeForecaster"),
    ("lasso", "wavqwise.forecasters.ml.linear", "LassoForecaster"),
    ("elasticnet", "wavqwise.forecasters.ml.linear", "ElasticNetForecaster"),
]

for name, mod, cls_name in _ML:
    Registry.register_forecaster(name, mod, cls_name, requires="ml")

# Neural forecasters
_NEURAL = [
    ("neuralprophet", "wavqwise.forecasters.neural.neuralprophet_wrapper", "NeuralProphetForecaster"),
    ("nbeats", "wavqwise.forecasters.neural.nbeats", "NBEATSForecaster"),
    ("tft", "wavqwise.forecasters.neural.tft", "TFTForecaster"),
]

for name, mod, cls_name in _NEURAL:
    Registry.register_forecaster(name, mod, cls_name, requires="neural")

# Foundation model forecasters
_FOUNDATION = [
    ("chronos", "wavqwise.forecasters.foundation.chronos", "ChronosForecaster"),
    ("timesfm", "wavqwise.forecasters.foundation.timesfm", "TimesFMForecaster"),
    ("lagllama", "wavqwise.forecasters.foundation.lagllama", "LagLlamaForecaster"),
    ("moirai", "wavqwise.forecasters.foundation.moirai", "MoiraiForecaster"),
    ("huggingface", "wavqwise.forecasters.foundation.huggingface", "HuggingFaceForecaster"),
]

for name, mod, cls_name in _FOUNDATION:
    Registry.register_forecaster(name, mod, cls_name, requires="foundation")

# Cloud forecasters
_CLOUD = [
    ("timegpt", "wavqwise.forecasters.cloud.timegpt", "TimeGPTForecaster"),
    ("ollama", "wavqwise.forecasters.cloud.ollama", "OllamaForecaster"),
    ("openai", "wavqwise.forecasters.cloud.openai", "OpenAIForecaster"),
    ("anthropic", "wavqwise.forecasters.cloud.anthropic_wrapper", "AnthropicForecaster"),
]

for name, mod, cls_name in _CLOUD:
    Registry.register_forecaster(name, mod, cls_name, requires="cloud")

# Anomaly detectors
_ANOMALY = [
    ("zscore", "wavqwise.anomaly.statistical", "ZScoreDetector"),
    ("iqr", "wavqwise.anomaly.statistical", "IQRDetector"),
    ("isolation_forest", "wavqwise.anomaly.isolation_forest", "IsolationForestDetector"),
    ("stl", "wavqwise.anomaly.stl_detector", "STLDetector"),
    ("dbscan", "wavqwise.anomaly.dbscan_detector", "DBSCANDetector"),
]

for name, mod, cls_name in _ANOMALY:
    Registry.register_anomaly_detector(name, mod, cls_name)

# Database connectors
_DB = [
    ("sqlite", "wavqwise.database.sqlite", "SQLiteConnector"),
    ("postgresql", "wavqwise.database.postgres", "PostgreSQLConnector"),
    ("mysql", "wavqwise.database.mysql", "MySQLConnector"),
    ("mongodb", "wavqwise.database.mongodb", "MongoDBConnector"),
    ("timescaledb", "wavqwise.database.timescaledb", "TimescaleDBConnector"),
    ("influxdb", "wavqwise.database.influxdb", "InfluxDBConnector"),
]

for name, mod, cls_name in _DB:
    Registry.register_db_connector(name, mod, cls_name, requires="database")

# Trading indicators
_INDICATORS = [
    ("rsi", "wavqwise.trading.indicators.momentum", "RSIIndicator"),
    ("macd", "wavqwise.trading.indicators.trend", "MACDIndicator"),
    ("bollinger", "wavqwise.trading.indicators.volatility", "BollingerBandsIndicator"),
    ("sma_indicator", "wavqwise.trading.indicators.trend", "SMAIndicator"),
    ("ema_indicator", "wavqwise.trading.indicators.trend", "EMAIndicator"),
    ("vwap", "wavqwise.trading.indicators.volume", "VWAPIndicator"),
    ("atr", "wavqwise.trading.indicators.volatility", "ATRIndicator"),
    ("stochastic", "wavqwise.trading.indicators.momentum", "StochasticIndicator"),
]

for name, mod, cls_name in _INDICATORS:
    Registry.register_indicator(name, mod, cls_name, requires="trading")
