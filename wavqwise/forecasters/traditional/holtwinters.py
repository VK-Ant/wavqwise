"""Holt-Winters forecaster."""
from wavqwise.forecasters.traditional.ets import ETSForecaster

class HoltWintersForecaster(ETSForecaster):
    """Holt-Winters (same as ETS with additive trend + seasonal)."""
    def __init__(self, seasonal_periods=None, **kwargs):
        super().__init__(seasonal_periods=seasonal_periods, trend="add", seasonal="add", **kwargs)
