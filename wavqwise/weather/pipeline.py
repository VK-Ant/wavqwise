"""WeatherPipeline - Weather forecasting with WavqWise."""
import pandas as pd
import numpy as np
from typing import Optional, List
from wavqwise.core.pipeline import WavqPipeline, ForecastResult
from wavqwise.weather.loader import WeatherLoader
from wavqwise.weather.models import WeatherIndicators

class WeatherPipeline:
    def __init__(self, device=None):
        self._loader = WeatherLoader()
        self._pipeline = WavqPipeline(device=device)
        self._data = None
        self._city = None

    def load_city(self, city, days=365, frequency="daily"):
        self._city = city
        self._data = self._loader.load_historical(city, days=days, frequency=frequency)
        self._add_indicators()
        return self

    def load_data(self, data):
        self._data = data.copy()
        self._add_indicators()
        return self

    def _add_indicators(self):
        if self._data is None: return
        df = self._data
        if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
            df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
        if "temperature_2m_mean" in df.columns and "relative_humidity_2m_mean" in df.columns:
            try:
                df["heat_index"] = WeatherIndicators.heat_index(df["temperature_2m_mean"], df["relative_humidity_2m_mean"])
                df["dew_point"] = WeatherIndicators.dew_point(df["temperature_2m_mean"], df["relative_humidity_2m_mean"])
            except: pass
        if "temperature_2m_mean" in df.columns and "windspeed_10m_max" in df.columns:
            try: df["wind_chill"] = WeatherIndicators.wind_chill(df["temperature_2m_mean"], df["windspeed_10m_max"])
            except: pass
        self._data = df

    def forecast(self, target="temperature_2m_mean", horizon=14, model="ema", **kwargs):
        if self._data is None: raise ValueError("No data. Call load_city() first.")
        if model.lower() in ("graphcast","gencast"):
            from wavqwise.weather.models import GraphCastForecaster
            f = GraphCastForecaster(); f.fit(self._data, target=target, time_col="date")
            return ForecastResult(f.predict(horizon=horizon), self._data, "graphcast", target=target, time_col="date")
        elif model.lower() == "aurora":
            from wavqwise.weather.models import AuroraForecaster
            f = AuroraForecaster(); f.fit(self._data, target=target, time_col="date")
            return ForecastResult(f.predict(horizon=horizon), self._data, "aurora", target=target, time_col="date")
        self._pipeline.load(self._data, target=target, time="date")
        return self._pipeline.forecast(horizon=horizon, model=model, **kwargs)

    def compare_models(self, target="temperature_2m_mean", models=None, horizon=14):
        if self._data is None: raise ValueError("No data.")
        models = models or ["moving_average","ema","naive","seasonal_naive"]
        self._pipeline.load(self._data, target=target, time="date")
        return self._pipeline.compare_models(models=models, horizon=horizon)

    def fetch_api_forecast(self, days=14):
        if self._city is None: raise ValueError("No city set.")
        return self._loader.load_forecast(self._city, days=days)

    def summary(self):
        if self._data is None: return "No data loaded."
        df = self._data
        lines = [f"Weather: {self._city or 'Custom'}", f"  Period: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}", f"  Days: {len(df)}"]
        for col in ["temperature_2m_mean","temperature_2m_max","precipitation_sum","windspeed_10m_max"]:
            if col in df.columns:
                lines.append(f"  {col}: avg={df[col].mean():.1f}, range={df[col].min():.1f}-{df[col].max():.1f}")
        return "\n".join(lines)

    @property
    def data(self): return self._data
