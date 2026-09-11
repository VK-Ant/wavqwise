"""
WavqWise Demo: Weather Forecasting (REAL DATA)
================================================
Data: Open-Meteo API - real historical weather (free, no API key)
Cities: Chennai, Tokyo, New York (configurable)

Shows:
  1. Load real weather data for any city
  2. Weather indicators (heat index, wind chill, dew point)
  3. Multi-variable forecast (temperature, precipitation, wind)
  4. Model comparison on weather data
  5. Multi-city comparison
  6. Visualization with weather-specific charts

Requirements: pip install wavqwise requests

For foundation weather models:
  pip install graphcast   # Google DeepMind GraphCast
  pip install microsoft-aurora  # Microsoft Aurora
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavqwise import WeatherPipeline
from wavqwise.weather.loader import WeatherLoader
from wavqwise.weather.models import WeatherIndicators


# === Run Demo ===
print("=" * 60)
print("WavqWise Weather Forecasting - REAL DATA")
print("Sense. Forecast. Alert.")
print("=" * 60)

CITY = "chennai"  # Change: tokyo, london, new york, mumbai, etc.

# 1. Load real weather data
print(f"\n[1/6] Loading weather data for {CITY.title()}...")
weather = WeatherPipeline()
try:
    weather.load_city(CITY, days=365)
    print(weather.summary())
except Exception as e:
    print(f"  API call failed (needs network): {e}")
    print("  Generating synthetic weather data for demo...")

    # Fallback: generate realistic weather data
    np.random.seed(42)
    days = 365
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    t = np.arange(days)

    # Seasonal temperature (Chennai-like: 25-38 C)
    temp_mean = 30 + 4 * np.sin(2 * np.pi * t / 365 - np.pi/3) + np.random.normal(0, 1.5, days)
    temp_max = temp_mean + np.random.uniform(3, 8, days)
    temp_min = temp_mean - np.random.uniform(3, 6, days)

    # Precipitation (monsoon pattern)
    precip = np.maximum(0, np.random.exponential(2, days) * (1 + 3 * np.sin(2 * np.pi * t / 365 - np.pi/2)))

    # Wind
    wind = 10 + 5 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, days)
    wind = np.maximum(0, wind)

    # Humidity
    humidity = 65 + 15 * np.sin(2 * np.pi * t / 365 - np.pi/4) + np.random.normal(0, 5, days)
    humidity = np.clip(humidity, 30, 100)

    data = pd.DataFrame({
        "date": dates,
        "temperature_2m_mean": temp_mean,
        "temperature_2m_max": temp_max,
        "temperature_2m_min": temp_min,
        "precipitation_sum": precip,
        "windspeed_10m_max": wind,
        "relative_humidity_2m_mean": humidity,
        "city": CITY.title(),
    })
    weather.load_data(data)
    print(weather.summary())

df = weather.data

# 2. Weather indicators
print(f"\n[2/6] Computing weather indicators...")
if "heat_index" in df.columns:
    print(f"  Heat index: avg={df['heat_index'].mean():.1f}C, max={df['heat_index'].max():.1f}C")
if "dew_point" in df.columns:
    print(f"  Dew point: avg={df['dew_point'].mean():.1f}C")
if "wind_chill" in df.columns:
    print(f"  Wind chill: min={df['wind_chill'].min():.1f}C")
if "temp_range" in df.columns:
    print(f"  Temp range: avg={df['temp_range'].mean():.1f}C")

# 3. Forecast temperature
print(f"\n[3/6] Forecasting temperature (30 days)...")
for model_name in ["moving_average", "ema", "naive"]:
    result = weather.forecast(target="temperature_2m_mean", horizon=30, model=model_name)
    day1 = result.forecast["temperature_2m_mean"].iloc[0]
    day30 = result.forecast["temperature_2m_mean"].iloc[-1]
    print(f"  {model_name}: day1={day1:.1f}C, day30={day30:.1f}C")

# 4. Model comparison
print(f"\n[4/6] Model comparison (14-day horizon)...")
comparison = weather.compare_models(
    target="temperature_2m_mean",
    models=["moving_average", "ema", "naive", "seasonal_naive"],
    horizon=14,
)
print(comparison.to_string(index=False))
best_model = comparison.iloc[0]["model"]
print(f"  Best: {best_model}")

# 5. Best model forecast
print(f"\n[5/6] Generating forecast with {best_model}...")
forecast = weather.forecast(target="temperature_2m_mean", horizon=30, model=best_model)

# 6. Visualization
print(f"\n[6/6] Generating charts...")
fig, axes = plt.subplots(4, 1, figsize=(16, 18), gridspec_kw={"height_ratios": [3, 1.5, 1.5, 1.5]})
fig.suptitle(f"WavqWise Weather Analysis: {CITY.title()} (Real Data)", fontsize=16, fontweight="bold")

# Temperature + forecast
ax = axes[0]
ax.plot(df["date"], df["temperature_2m_mean"], color="#dc2626", linewidth=1, label="Mean temp", alpha=0.8)
if "temperature_2m_max" in df.columns:
    ax.fill_between(df["date"], df["temperature_2m_min"], df["temperature_2m_max"],
                    alpha=0.15, color="#dc2626", label="Min-Max range")
if "heat_index" in df.columns:
    ax.plot(df["date"], df["heat_index"], color="#f59e0b", linewidth=0.8, alpha=0.6, label="Heat index")

# Forecast overlay
fc = forecast.forecast
ax.plot(fc["date"], fc["temperature_2m_mean"], color="#059669", linewidth=2.5, linestyle="--",
        label=f"Forecast ({best_model})")
ax.fill_between(fc["date"], fc["temperature_2m_mean_lower"], fc["temperature_2m_mean_upper"],
                alpha=0.15, color="#059669")

ax.set_ylabel("Temperature (C)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Precipitation
ax = axes[1]
if "precipitation_sum" in df.columns:
    ax.bar(df["date"], df["precipitation_sum"], color="#2563eb", alpha=0.6, width=1)
ax.set_ylabel("Precipitation (mm)")
ax.grid(True, alpha=0.3)

# Wind speed
ax = axes[2]
if "windspeed_10m_max" in df.columns:
    ax.plot(df["date"], df["windspeed_10m_max"], color="#7c3aed", linewidth=0.8)
    ax.fill_between(df["date"], 0, df["windspeed_10m_max"], alpha=0.1, color="#7c3aed")
ax.set_ylabel("Wind (km/h)")
ax.grid(True, alpha=0.3)

# Humidity
ax = axes[3]
if "relative_humidity_2m_mean" in df.columns:
    ax.plot(df["date"], df["relative_humidity_2m_mean"], color="#0d9488", linewidth=0.8)
    ax.fill_between(df["date"], 0, df["relative_humidity_2m_mean"], alpha=0.1, color="#0d9488")
ax.set_ylabel("Humidity (%)")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("demos/weather_forecast_results.png", dpi=150, bbox_inches="tight")
print(f"  Plot saved: demos/weather_forecast_results.png")

# Available weather models
print(f"\nWeather foundation models available:")
print(f"  graphcast  - Google DeepMind (pip install graphcast)")
print(f"  aurora     - Microsoft (pip install microsoft-aurora)")
print(f"  pangu      - Huawei Pangu-Weather")
print(f"  fourcastnet - NVIDIA FourCastNet")
print(f"\nDone.")
