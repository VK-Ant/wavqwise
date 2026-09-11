"""
WavqWise Use Case: Dam Water Level Monitoring & Flood Early Warning
=====================================================================
Real-world application of WavqWise for:
  - Dam water level forecasting
  - Flood risk prediction
  - Rainfall-driven inflow forecasting
  - Real-time anomaly alerts (sudden rise/overflow)
  - OpenStreetMap visualization of dam network

This demo simulates a network of Tamil Nadu dams with realistic
monsoon-driven water level patterns.

Dams simulated (real coordinates):
  - Mettur Dam (Salem) - Cauvery River
  - Vaigai Dam (Theni) - Vaigai River
  - Mullaperiyar Dam (Idukki) - Periyar River
  - Bhavanisagar Dam (Erode) - Bhavani River
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from wavqwise import WavqPipeline
from wavqwise.anomaly.pipeline import AnomalyPipeline
from wavqwise.visualization.map_viz import WeatherMap


# === Tamil Nadu Dam Network (real coordinates) ===
DAMS = {
    "Mettur Dam": {
        "lat": 11.7939, "lon": 77.8017,
        "river": "Cauvery", "capacity_ft": 120,
        "base_level": 60, "seasonal_amp": 35,
    },
    "Vaigai Dam": {
        "lat": 10.0112, "lon": 77.5494,
        "river": "Vaigai", "capacity_ft": 71,
        "base_level": 30, "seasonal_amp": 25,
    },
    "Mullaperiyar Dam": {
        "lat": 9.5322, "lon": 77.1387,
        "river": "Periyar", "capacity_ft": 142,
        "base_level": 100, "seasonal_amp": 30,
    },
    "Bhavanisagar Dam": {
        "lat": 11.4450, "lon": 77.0803,
        "river": "Bhavani", "capacity_ft": 105,
        "base_level": 45, "seasonal_amp": 40,
    },
}


def generate_dam_data(dam_info, days=365, seed=42):
    """Generate realistic dam water level data with monsoon patterns."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    t = np.arange(days)

    # Seasonal pattern (monsoon: Oct-Dec for Tamil Nadu)
    monsoon = dam_info["seasonal_amp"] * np.maximum(0, np.sin(2 * np.pi * (t - 240) / 365))
    base = dam_info["base_level"]

    # Rainfall-driven noise
    rainfall = np.random.exponential(3, days) * (1 + 2 * np.maximum(0, np.sin(2 * np.pi * (t - 240) / 365)))
    inflow = np.convolve(rainfall, np.exp(-np.arange(10) / 3), mode="same")

    # Water level
    level = base + monsoon + inflow * 0.3 + np.random.normal(0, 1.5, days)
    level = np.clip(level, 0, dam_info["capacity_ft"] * 1.05)

    return pd.DataFrame({
        "date": dates,
        "water_level_ft": level,
        "rainfall_mm": rainfall,
        "inflow_cusecs": inflow * 1000 + np.random.normal(0, 500, days),
        "outflow_cusecs": np.maximum(0, (level - base) * 200 + np.random.normal(0, 300, days)),
    })


# === Run Demo ===
print("=" * 60)
print("WavqWise: Dam Water Level Monitoring")
print("Flood Early Warning System")
print("Sense. Forecast. Alert.")
print("=" * 60)

# 1. Generate data for all dams
print("\n[1/6] Loading dam network data (Tamil Nadu)...")
dam_data = {}
for name, info in DAMS.items():
    seed = hash(name) % 1000
    dam_data[name] = generate_dam_data(info, days=365, seed=seed)
    latest = dam_data[name]["water_level_ft"].iloc[-1]
    pct = (latest / info["capacity_ft"]) * 100
    status = "CRITICAL" if pct > 90 else "HIGH" if pct > 75 else "NORMAL"
    print(f"  {name} ({info['river']}): {latest:.1f}/{info['capacity_ft']} ft ({pct:.0f}%) [{status}]")

# 2. Forecast each dam
print("\n[2/6] Forecasting water levels (30 days)...")
forecasts = {}
for name, data in dam_data.items():
    pipeline = WavqPipeline()
    pipeline.load(data, target="water_level_ft", time="date")
    result = pipeline.forecast(horizon=30, model="ema")
    forecasts[name] = result

    current = data["water_level_ft"].iloc[-1]
    predicted = result.forecast["water_level_ft"].iloc[-1]
    change = predicted - current
    print(f"  {name}: current={current:.1f}ft, 30d forecast={predicted:.1f}ft ({change:+.1f}ft)")

# 3. Anomaly detection
print("\n[3/6] Detecting anomalies (sudden level changes)...")
for name, data in dam_data.items():
    detector = AnomalyPipeline()
    detector.load(data, target="water_level_ft", time="date")
    result = detector.detect(method="zscore")
    n_anom = len(result.anomalies)
    if n_anom > 0:
        worst = result.anomalies.nlargest(1, "anomaly_score").iloc[0]
        print(f"  {name}: {n_anom} anomalies, worst: {worst['water_level_ft']:.1f}ft (score={worst['anomaly_score']:.2f})")
    else:
        print(f"  {name}: No anomalies")

# 4. Real-time streaming simulation
print("\n[4/6] Simulating real-time monitoring (Mettur Dam)...")
mettur = dam_data["Mettur Dam"]
pipeline = WavqPipeline()
pipeline.load(mettur.iloc[:350], target="water_level_ft", time="date")
pipeline.forecast(horizon=7, model="ema")

alerts = []
stream = pipeline.stream(
    model="ema",
    anomaly_method="zscore",
    window_size=200,
    on_anomaly=lambda e: alerts.append(e),
    anomaly_threshold=2.0,
    forecast_every=5,
)

# Simulate 15 days of incoming data including a flood event
for i in range(15):
    row = mettur.iloc[350 + i]
    value = row["water_level_ft"]
    if i == 10:
        value = DAMS["Mettur Dam"]["capacity_ft"] * 0.98  # Near overflow
    stream.push({"date": row["date"], "water_level_ft": value})

print(f"  Stream: {stream.summary()}")
print(f"  Flood alerts triggered: {len(alerts)}")

# 5. Model comparison for water level prediction
print("\n[5/6] Model comparison (Mettur Dam, 14-day)...")
pipeline2 = WavqPipeline()
pipeline2.load(mettur, target="water_level_ft", time="date")
comparison = pipeline2.compare_models(
    models=["moving_average", "ema", "naive", "seasonal_naive"],
    horizon=14,
)
print(comparison.to_string(index=False))

# 6. Map visualization
print("\n[6/6] Generating OpenStreetMap visualization...")
wmap = WeatherMap(center=(10.8, 77.8), zoom=8)

for name, info in DAMS.items():
    data = dam_data[name]
    current_level = data["water_level_ft"].iloc[-1]
    forecast_level = forecasts[name].forecast["water_level_ft"].iloc[-1]

    wmap.add_dam_marker(
        name=f"{name} ({info['river']})",
        lat=info["lat"], lon=info["lon"],
        water_level=current_level,
        capacity=info["capacity_ft"],
        forecast_level=forecast_level,
    )

map_path = wmap.save("demos/dam_monitoring_map.html")
print(f"  Map saved: {map_path}")

# 7. Visualization charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("WavqWise Dam Monitoring - Tamil Nadu Network", fontsize=16, fontweight="bold")

for idx, (name, info) in enumerate(DAMS.items()):
    ax = axes[idx // 2][idx % 2]
    data = dam_data[name]
    fc = forecasts[name].forecast

    ax.plot(data["date"], data["water_level_ft"], color="#2563eb", linewidth=1, label="Actual")
    ax.plot(fc["date"], fc["water_level_ft"], "--", color="#dc2626", linewidth=2, label="Forecast")
    ax.fill_between(fc["date"], fc["water_level_ft_lower"], fc["water_level_ft_upper"],
                    alpha=0.15, color="#dc2626")
    ax.axhline(info["capacity_ft"], color="#ef4444", linestyle=":", linewidth=1, label=f"Capacity ({info['capacity_ft']}ft)")
    ax.axhline(info["capacity_ft"] * 0.9, color="#f59e0b", linestyle=":", linewidth=0.8, label="90% alert")
    ax.set_title(f"{name} ({info['river']})")
    ax.set_ylabel("Water Level (ft)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("demos/dam_monitoring_charts.png", dpi=150, bbox_inches="tight")
print(f"  Charts saved: demos/dam_monitoring_charts.png")

print(f"\nUse cases demonstrated:")
print(f"  1. Dam water level forecasting (30-day)")
print(f"  2. Flood early warning (anomaly detection)")
print(f"  3. Real-time monitoring (streaming)")
print(f"  4. Multi-dam network visualization (OpenStreetMap)")
print(f"  5. Model comparison for best accuracy")
print(f"\nDone.")
