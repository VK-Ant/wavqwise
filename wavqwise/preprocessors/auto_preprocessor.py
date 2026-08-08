"""Auto preprocessor - handles missing values, outliers, normalization."""
import numpy as np
import pandas as pd


class AutoPreprocessor:
    def __init__(self):
        self._scaler_params = {}

    def transform(self, data: pd.DataFrame, target: str, time_col: str, **kwargs) -> pd.DataFrame:
        df = data.copy()

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)

        # Handle missing values in target
        if df[target].isnull().any():
            null_pct = df[target].isnull().mean()
            if null_pct < 0.3:
                df[target] = df[target].interpolate(method="linear")
                df[target] = df[target].ffill().bfill()
            else:
                df = df.dropna(subset=[target]).reset_index(drop=True)

        # Handle outliers (clip at 3 sigma)
        if kwargs.get("clip_outliers", True):
            mean = df[target].mean()
            std = df[target].std()
            if std > 0:
                lower = mean - 3 * std
                upper = mean + 3 * std
                df[target] = df[target].clip(lower, upper)

        return df
