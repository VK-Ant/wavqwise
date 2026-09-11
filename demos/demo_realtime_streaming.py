"""
WavqWise Demo: Real-Time Streaming
====================================
Simulates a live IoT sensor feed with real-time anomaly detection
and continuous forecast updates.

This is what makes WavqWise different:
- Nixtla/Darts/Prophet: batch only, retrain from scratch
- WavqWise: continuous streaming + incremental update + live alerts
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from wavqwise import WavqPipeline


# === 1. Simulate historical sensor data ===
print("=" * 60)
print("WavqWise Real-Time Streaming Demo")
print("Sense. Forecast. Alert.")
print("=" * 60)

np.random.seed(42)
n = 200
dates = pd.date_range("2025-01-01", periods=n, freq="h")
temp = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 1, n)
historical = pd.DataFrame({"timestamp": dates, "temperature": temp})

print(f"\n[1/4] Loaded {len(historical)} hours of historical sensor data")

# === 2. Initialize pipeline with history ===
pipeline = WavqPipeline()
pipeline.load(historical, target="temperature", time="timestamp")
pipeline.forecast(horizon=24, model="ema")
print(f"[2/4] Pipeline initialized, model fitted")

# === 3. Start streaming ===
alerts = []

def on_anomaly(event):
    for _, row in event.anomalies.iterrows():
        alert = {
            "time": datetime.now().isoformat(),
            "value": row["temperature"],
            "severity": row.get("severity", "unknown"),
            "score": row.get("anomaly_score", 0),
        }
        alerts.append(alert)

forecast_updates = []

def on_forecast(result):
    forecast_updates.append({
        "time": datetime.now().isoformat(),
        "next_hour": result.forecast["temperature"].iloc[0],
        "next_24h_avg": result.forecast["temperature"].mean(),
    })

stream = pipeline.stream(
    model="ema",
    anomaly_method="zscore",
    window_size=200,
    forecast_horizon=24,
    on_anomaly=on_anomaly,
    on_forecast=on_forecast,
    forecast_every=5,
    anomaly_threshold=2.5,
)

print(f"[3/4] Stream started")

# === 4. Simulate 50 incoming data points ===
print(f"\n[4/4] Streaming 50 new sensor readings...\n")

last_time = historical["timestamp"].max()
for i in range(50):
    t = last_time + timedelta(hours=i + 1)
    value = 50 + 10 * np.sin(2 * np.pi * (n + i) / 24) + np.random.normal(0, 1)

    # Inject anomalies at specific points
    if i == 15:
        value = 95  # Spike
    elif i == 30:
        value = 10  # Drop

    stream.push({"timestamp": t, "temperature": value})

# === Results ===
print(f"\n{'=' * 60}")
print(f"STREAMING RESULTS")
print(f"{'=' * 60}")
print(f"\n{stream.summary()}")
print(f"\nAnomalies detected: {len(alerts)}")
for a in alerts:
    print(f"  temp={a['value']:.1f}, severity={a['severity']}, score={a['score']:.2f}")

print(f"\nForecast updates: {len(forecast_updates)}")
if forecast_updates:
    last = forecast_updates[-1]
    print(f"  Latest: next_hour={last['next_hour']:.1f}F, 24h_avg={last['next_24h_avg']:.1f}F")

stats = stream.stats
print(f"\nStats:")
print(f"  Points processed: {stats['points_processed']}")
print(f"  Window size: {stats['window_size']}")
print(f"  Model: {stats['model']}")
print(f"  Anomaly method: {stats['anomaly_method']}")
print(f"\nDone.")
