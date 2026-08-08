"""Auto feature engineering for time-series."""
import numpy as np
import pandas as pd


class AutoFeatureEngineer:
    def generate(self, data: pd.DataFrame, target: str, time_col: str, **kwargs) -> pd.DataFrame:
        df = data.copy()

        # Lag features
        for lag in [1, 7, 14, 30]:
            if lag < len(df):
                df[f"lag_{lag}"] = df[target].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            if window < len(df):
                df[f"rolling_mean_{window}"] = df[target].rolling(window).mean()
                df[f"rolling_std_{window}"] = df[target].rolling(window).std()

        # Calendar features
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df["day_of_week"] = df[time_col].dt.dayofweek
            df["month"] = df[time_col].dt.month
            df["day_of_year"] = df[time_col].dt.dayofyear
            df["is_weekend"] = (df[time_col].dt.dayofweek >= 5).astype(int)

        # Trend
        df["trend"] = np.arange(len(df))

        # Drop NaN rows from features
        df = df.dropna().reset_index(drop=True)

        return df
