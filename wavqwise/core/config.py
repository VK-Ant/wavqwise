"""Pipeline configuration management."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PipelineConfig:
    """Configuration for WavqPipeline."""

    model: str = "moving_average"
    horizon: int = 30
    target: Optional[str] = None
    time_col: Optional[str] = None
    frequency: Optional[str] = None  # Auto-detected if None
    confidence_level: float = 0.95
    auto_preprocess: bool = True
    auto_features: bool = True
    incremental: bool = True
    plot_engine: str = "matplotlib"  # "matplotlib" or "plotly"
    plot_style: str = "default"
    verbose: bool = True
    random_state: int = 42
    n_jobs: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyConfig:
    """Configuration for AnomalyPipeline."""

    method: str = "auto"  # "zscore", "iqr", "isolation_forest", "stl", "auto"
    sensitivity: float = 0.95
    window_size: Optional[int] = None
    streaming: bool = False
    alert_handlers: List[str] = field(default_factory=list)
    severity_levels: Dict[str, float] = field(default_factory=lambda: {
        "low": 2.0,
        "medium": 3.0,
        "high": 4.0,
        "critical": 5.0,
    })


@dataclass
class TradingConfig:
    """Configuration for TradingPipeline."""

    indicators: List[str] = field(default_factory=list)
    strategy: str = "momentum"
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    position_size: float = 1.0
