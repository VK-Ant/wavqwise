"""
Auto Feature Engineering for Time-Series
==========================================
Comprehensive, pluggable feature generation for temporal data.

Categories:
  - Lag features (simple + multi-horizon)
  - Rolling / sliding-window statistics (mean, std, min, max, median, skew, kurt)
  - Exponentially weighted statistics (EWM mean, std, var)
  - Rate of change / momentum
  - Differencing (first & second order)
  - Percentage change
  - Calendar / datetime decomposition
  - Cyclical encoding (sin/cos for periodic features)
  - Fourier seasonal features
  - Trend features (linear, quadratic)
  - Interaction features (ratio, product of rolling stats)
  - Volatility features (rolling z-score, Bollinger-like bands)

Usage:
    from wavqwise.features import AutoFeatureEngineer

    fe = AutoFeatureEngineer()
    enriched = fe.generate(df, target="value", time_col="timestamp")
    print(f"Created {len(enriched.columns) - len(df.columns)} new features")
"""

import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any


class AutoFeatureEngineer:
    """Automatic feature engineering for time-series data.

    Generates a rich set of temporal features from a single target column
    and optional datetime index.  All features are backward-looking so they
    are safe for forecasting (no data leakage).

    Parameters
    ----------
    lag_periods : list[int]
        Lag offsets to create.  Default ``[1, 2, 3, 7, 14, 30]``.
    rolling_windows : list[int]
        Window sizes for rolling statistics.  Default ``[7, 14, 30, 60]``.
    ewm_spans : list[int]
        Spans for exponentially weighted stats.  Default ``[7, 14, 30]``.
    fourier_order : int
        Number of sin/cos pairs for seasonal Fourier features.  Default ``3``.
    drop_na : bool
        Whether to drop rows with NaN introduced by lagging.  Default ``True``.
    feature_groups : list[str] | None
        Subset of groups to generate.  ``None`` → all.
        Valid groups: ``"lags"``, ``"rolling"``, ``"ewm"``, ``"calendar"``,
        ``"cyclical"``, ``"fourier"``, ``"trend"``, ``"diff"``,
        ``"pct_change"``, ``"rate_of_change"``, ``"volatility"``,
        ``"interaction"``.
    """

    VALID_GROUPS = {
        "lags", "rolling", "ewm", "calendar", "cyclical", "fourier",
        "trend", "diff", "pct_change", "rate_of_change", "volatility",
        "interaction",
    }

    def __init__(
        self,
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        ewm_spans: Optional[List[int]] = None,
        fourier_order: int = 3,
        drop_na: bool = True,
        feature_groups: Optional[List[str]] = None,
    ):
        self.lag_periods = lag_periods or [1, 2, 3, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30, 60]
        self.ewm_spans = ewm_spans or [7, 14, 30]
        self.fourier_order = fourier_order
        self.drop_na = drop_na

        if feature_groups is not None:
            unknown = set(feature_groups) - self.VALID_GROUPS
            if unknown:
                raise ValueError(f"Unknown feature groups: {unknown}")
            self.feature_groups = set(feature_groups)
        else:
            self.feature_groups = self.VALID_GROUPS.copy()

        # Track generated feature names
        self._generated_features: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        data: pd.DataFrame,
        target: str,
        time_col: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate all configured feature groups.

        Parameters
        ----------
        data : DataFrame
            Input time-series data.
        target : str
            Name of the target column.
        time_col : str
            Name of the datetime column.

        Returns
        -------
        DataFrame
            Original data enriched with new feature columns.
        """
        df = data.copy()
        self._generated_features = []
        n = len(df)

        # Ensure datetime
        if time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Suppress pandas fragmentation warning (we build columns incrementally)
        warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

        # Dispatch each feature group
        dispatch = {
            "lags": self._add_lags,
            "rolling": self._add_rolling,
            "ewm": self._add_ewm,
            "calendar": self._add_calendar,
            "cyclical": self._add_cyclical,
            "fourier": self._add_fourier,
            "trend": self._add_trend,
            "diff": self._add_diff,
            "pct_change": self._add_pct_change,
            "rate_of_change": self._add_rate_of_change,
            "volatility": self._add_volatility,
            "interaction": self._add_interaction,
        }

        for group in self.feature_groups:
            fn = dispatch.get(group)
            if fn:
                df = fn(df, target, time_col, n)

        if self.drop_na:
            df = df.dropna().reset_index(drop=True)

        return df

    def list_features(self) -> List[str]:
        """Return names of features generated in the last ``generate()`` call."""
        return list(self._generated_features)

    def feature_report(self) -> Dict[str, Any]:
        """Summary of the last feature generation run."""
        return {
            "total_features": len(self._generated_features),
            "groups_used": sorted(self.feature_groups),
            "feature_names": self.list_features(),
        }

    # ------------------------------------------------------------------
    # Feature group implementations
    # ------------------------------------------------------------------

    def _track(self, name: str):
        """Register a generated feature name."""
        self._generated_features.append(name)

    def _add_lags(self, df, target, time_col, n):
        """Lag features: shifted values of the target."""
        for lag in self.lag_periods:
            if lag < n:
                col = f"lag_{lag}"
                df[col] = df[target].shift(lag)
                self._track(col)
        return df

    def _add_rolling(self, df, target, time_col, n):
        """Rolling window statistics: mean, std, min, max, median, skew, kurtosis."""
        for w in self.rolling_windows:
            if w < n:
                r = df[target].rolling(w)

                col_mean = f"rolling_mean_{w}"
                df[col_mean] = r.mean()
                self._track(col_mean)

                col_std = f"rolling_std_{w}"
                df[col_std] = r.std()
                self._track(col_std)

                col_min = f"rolling_min_{w}"
                df[col_min] = r.min()
                self._track(col_min)

                col_max = f"rolling_max_{w}"
                df[col_max] = r.max()
                self._track(col_max)

                col_med = f"rolling_median_{w}"
                df[col_med] = r.median()
                self._track(col_med)

                # Skew & kurtosis need enough data points
                if w >= 3:
                    col_skew = f"rolling_skew_{w}"
                    df[col_skew] = r.skew()
                    self._track(col_skew)

                if w >= 4:
                    col_kurt = f"rolling_kurt_{w}"
                    df[col_kurt] = r.kurt()
                    self._track(col_kurt)

        return df

    def _add_ewm(self, df, target, time_col, n):
        """Exponentially weighted moving statistics."""
        for span in self.ewm_spans:
            if span < n:
                ewm = df[target].ewm(span=span)

                col_mean = f"ewm_mean_{span}"
                df[col_mean] = ewm.mean()
                self._track(col_mean)

                col_std = f"ewm_std_{span}"
                df[col_std] = ewm.std()
                self._track(col_std)

                col_var = f"ewm_var_{span}"
                df[col_var] = ewm.var()
                self._track(col_var)

        return df

    def _add_calendar(self, df, target, time_col, n):
        """Calendar / datetime decomposition features."""
        if time_col not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            return df

        dt = df[time_col].dt

        mappings = {
            "hour": dt.hour,
            "day_of_week": dt.dayofweek,
            "day_of_month": dt.day,
            "day_of_year": dt.dayofyear,
            "week_of_year": dt.isocalendar().week.astype(int),
            "month": dt.month,
            "quarter": dt.quarter,
            "is_weekend": (dt.dayofweek >= 5).astype(int),
            "is_month_start": dt.is_month_start.astype(int),
            "is_month_end": dt.is_month_end.astype(int),
            "is_quarter_start": dt.is_quarter_start.astype(int),
            "is_quarter_end": dt.is_quarter_end.astype(int),
        }

        for col_name, values in mappings.items():
            df[col_name] = values
            self._track(col_name)

        return df

    def _add_cyclical(self, df, target, time_col, n):
        """Cyclical sin/cos encoding for periodic calendar features.

        Encodes hour-of-day, day-of-week, day-of-year, and month as
        sin/cos pairs so the model sees that Dec 31 ↔ Jan 1 are neighbours.
        """
        if time_col not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            return df

        dt = df[time_col].dt
        cycles = {
            "hour": (dt.hour, 24),
            "dayofweek": (dt.dayofweek, 7),
            "dayofyear": (dt.dayofyear, 365),
            "month": (dt.month, 12),
        }

        for name, (values, period) in cycles.items():
            sin_col = f"{name}_sin"
            cos_col = f"{name}_cos"
            df[sin_col] = np.sin(2 * np.pi * values / period)
            df[cos_col] = np.cos(2 * np.pi * values / period)
            self._track(sin_col)
            self._track(cos_col)

        return df

    def _add_fourier(self, df, target, time_col, n):
        """Fourier seasonal features: multiple sin/cos harmonics.

        Captures complex seasonal patterns by projecting the time index
        onto ``fourier_order`` pairs of sinusoids.
        """
        t = np.arange(n, dtype=float)
        for k in range(1, self.fourier_order + 1):
            sin_col = f"fourier_sin_{k}"
            cos_col = f"fourier_cos_{k}"
            df[sin_col] = np.sin(2 * np.pi * k * t / n)
            df[cos_col] = np.cos(2 * np.pi * k * t / n)
            self._track(sin_col)
            self._track(cos_col)

        return df

    def _add_trend(self, df, target, time_col, n):
        """Linear and quadratic trend features."""
        t = np.arange(n, dtype=float)

        col_lin = "trend_linear"
        df[col_lin] = t
        self._track(col_lin)

        col_quad = "trend_quadratic"
        df[col_quad] = t ** 2
        self._track(col_quad)

        # Normalised trend (0-1) for scale-invariant models
        col_norm = "trend_normalised"
        df[col_norm] = t / max(n - 1, 1)
        self._track(col_norm)

        return df

    def _add_diff(self, df, target, time_col, n):
        """First and second order differencing."""
        col_d1 = "diff_1"
        df[col_d1] = df[target].diff(1)
        self._track(col_d1)

        col_d2 = "diff_2"
        df[col_d2] = df[target].diff(2)
        self._track(col_d2)

        # Seasonal differencing (lag-7)
        if n > 7:
            col_d7 = "diff_seasonal_7"
            df[col_d7] = df[target].diff(7)
            self._track(col_d7)

        return df

    def _add_pct_change(self, df, target, time_col, n):
        """Percentage change features at multiple horizons."""
        for period in [1, 7, 14]:
            if period < n:
                col = f"pct_change_{period}"
                df[col] = df[target].pct_change(period)
                self._track(col)
        return df

    def _add_rate_of_change(self, df, target, time_col, n):
        """Rate of change / momentum indicators.

        ROC = (current - past) / past × 100
        Momentum = current - past
        """
        for period in [7, 14, 30]:
            if period < n:
                past = df[target].shift(period)

                roc_col = f"roc_{period}"
                df[roc_col] = ((df[target] - past) / past.replace(0, np.nan)) * 100
                self._track(roc_col)

                mom_col = f"momentum_{period}"
                df[mom_col] = df[target] - past
                self._track(mom_col)

        return df

    def _add_volatility(self, df, target, time_col, n):
        """Volatility features: rolling z-score and Bollinger-like bands."""
        for w in self.rolling_windows:
            if w < n:
                r_mean = df[target].rolling(w).mean()
                r_std = df[target].rolling(w).std()

                # Rolling z-score
                z_col = f"rolling_zscore_{w}"
                df[z_col] = (df[target] - r_mean) / r_std.replace(0, np.nan)
                self._track(z_col)

                # Upper / lower bands (±2σ)
                upper_col = f"band_upper_{w}"
                lower_col = f"band_lower_{w}"
                df[upper_col] = r_mean + 2 * r_std
                df[lower_col] = r_mean - 2 * r_std
                self._track(upper_col)
                self._track(lower_col)

                # Band width (normalised volatility)
                bw_col = f"band_width_{w}"
                df[bw_col] = (4 * r_std) / r_mean.replace(0, np.nan)
                self._track(bw_col)

        return df

    def _add_interaction(self, df, target, time_col, n):
        """Interaction features between rolling windows.

        Creates ratios and products of rolling statistics at different
        scales to capture cross-scale dynamics.
        """
        windows = sorted([w for w in self.rolling_windows if w < n])
        if len(windows) < 2:
            return df

        # Short / long rolling-mean ratio (golden cross / death cross style)
        short_w = windows[0]
        long_w = windows[-1]

        short_mean = df[target].rolling(short_w).mean()
        long_mean = df[target].rolling(long_w).mean()

        ratio_col = f"mean_ratio_{short_w}_{long_w}"
        df[ratio_col] = short_mean / long_mean.replace(0, np.nan)
        self._track(ratio_col)

        crossover_col = f"mean_crossover_{short_w}_{long_w}"
        df[crossover_col] = (short_mean - long_mean).apply(np.sign)
        self._track(crossover_col)

        # Std ratio: short-term vol vs long-term vol
        if len(windows) >= 2:
            short_std = df[target].rolling(short_w).std()
            long_std = df[target].rolling(long_w).std()

            std_ratio_col = f"std_ratio_{short_w}_{long_w}"
            df[std_ratio_col] = short_std / long_std.replace(0, np.nan)
            self._track(std_ratio_col)

        # Distance from rolling mean (multiple scales)
        for w in windows:
            r_mean = df[target].rolling(w).mean()
            dist_col = f"dist_from_mean_{w}"
            df[dist_col] = df[target] - r_mean
            self._track(dist_col)

        return df
