"""
WavqWise - Pluggable Temporal Intelligence
Sense. Forecast. Alert.

Any model. Any signal. Five lines to forecast.
"""

__version__ = "0.1.5"
__author__ = "VK-Ant (Venkatkumar Rajan)"
__tagline__ = "Sense. Forecast. Alert."

from wavqwise.core.pipeline import WavqPipeline
from wavqwise.anomaly.pipeline import AnomalyPipeline
from wavqwise.trading.pipeline import TradingPipeline
from wavqwise.signals.pipeline import SignalPipeline
from wavqwise.weather.pipeline import WeatherPipeline
from wavqwise.ecosystem.bridge import EcosystemBridge
from wavqwise.core.registry import Registry

__all__ = [
    "WavqPipeline",
    "AnomalyPipeline",
    "TradingPipeline",
    "SignalPipeline",
    "WeatherPipeline",
    "EcosystemBridge",
    "Registry",
    "__version__",
]
