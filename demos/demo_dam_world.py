"""
WavqWise Demo: World Dam Monitoring & Flood Early Warning
==========================================================
OUTPUT: PNG charts + CLI tables + CSV export (no HTML)

Real data sources (all free, no API key):
  - India WRIS: india-wris.nrsc.gov.in (5000+ Indian dams)
  - USGS: waterservices.usgs.gov (1.5M+ US water sites)
  - Global Dam Watch: globaldamwatch.org (7000+ worldwide)
  - Open-Meteo: api.open-meteo.com (rainfall forecasts)

This demo simulates real dam data patterns. To use actual live data,
connect to the APIs listed above via WavqPipeline.load().

Demonstrates:
  1. Multi-dam network analysis (worldwide)
  2. Water level forecasting (30-day)
  3. Anomaly detection (flood early warning)
  4. Real-time streaming monitoring
  5. Model comparison
  6. PNG chart output (no HTML)
  7. CLI table output
  8. CSV/JSON data export
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from wavqwise import WavqPipeline
from wavqwise.anomaly.pipeline import AnomalyPipeline
from wavqwise.visualization.renderer import ResultRenderer, CLIPrinter, DataExporter
from wavqwise.weather.real_data import list_sources

# === World Dam Database (real coordinates, realistic levels) ===
WORLD_DAMS = [
    # India
    {"name": "Mettur Dam", "country": "India", "state": "Tamil Nadu", "river": "Cauvery",
     "lat": 11.7939, "lon": 77.8017, "capacity": 120, "base": 60, "amp": 35},
    {"name": "Vaigai Dam", "country": "India", "state": "Tamil Nadu", "river": "Vaigai",
     "lat": 10.0112, "lon": 77.5494, "capacity": 71, "base": 30, "amp": 25},
    {"name": "Mullaperiyar Dam", "country": "India", "state": "Kerala", "river": "Periyar",
     "lat": 9.5322, "lon": 77.1387, "capacity": 142, "base": 100, "amp": 30},
    {"name": "Bhakra Dam", "country": "India", "state": "Himachal", "river": "Sutlej",
     "lat": 31.4109, "lon": 76.4327, "capacity": 1685, "base": 1200, "amp": 300},
    {"name": "Hirakud Dam", "country": "India", "state": "Odisha", "river": "Mahanadi",
     "lat": 21.5181, "lon": 83.8719, "capacity": 630, "base": 400, "amp": 150},
    # USA
    {"name": "Hoover Dam", "country": "USA", "state": "Nevada", "river": "Colorado",
     "lat": 36.0161, "lon": -114.7377, "capacity": 1229, "base": 900, "amp": 200},
    {"name": "Glen Canyon Dam", "country": "USA", "state": "Arizona", "river": "Colorado",
     "lat": 36.9375, "lon": -111.4858, "capacity": 3700, "base": 3200, "amp": 300},
    {"name": "Oroville Dam", "country": "USA", "state": "California", "river": "Feather",
     "lat": 39.5386, "lon": -121.4858, "capacity": 900, "base": 600, "amp": 200},
    # China
    {"name": "Three Gorges Dam", "country": "China", "state": "Hubei", "river": "Yangtze",
     "lat": 30.8236, "lon": 111.0035, "capacity": 175, "base": 135, "amp": 25},
    # Brazil
    {"name": "Itaipu Dam", "country": "Brazil", "state": "Parana", "river": "Parana",
     "lat": -25.4084, "lon": -54.5894, "capacity": 220, "base": 180, "amp": 25},
    # Egypt
    {"name": "Aswan High Dam", "country": "Egypt", "state": "Aswan", "river": "Nile",
     "lat": 23.9708, "lon": 32.8781, "capacity": 183, "base": 155, "amp": 20},
    # Turkey
    {"name": "Ataturk Dam", "country": "Turkey", "state": "Sanliurfa", "river": "Euphrates",
     "lat": 37.4750, "lon": 38.3250, "capacity": 542, "base": 400, "amp": 80},
]


def generate_dam_data(dam, days=365):
    """Generate realistic dam water level with monsoon/seasonal patterns."""
    np.random.seed(hash(dam["name"]) % 10000)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    t = np.arange(days) / 365

    seasonal = dam["amp"] * np.sin(2 * np.pi * (t - 0.65))
    monsoon = dam["amp"] * 0.4 * np.maximum(0, np.sin(2 * np.pi * (t - 0.7)))
    rainfall = np.random.exponential(3, days) * (1 + 2 * np.maximum(0, np.sin(2 * np.pi * (t - 0.65))))
    inflow = np.convolve(rainfall, np.exp(-np.arange(10) / 3), mode="same")
    noise = np.random.normal(0, dam["capacity"] * 0.01, days)

    level = dam["base"] + seasonal + monsoon + inflow * 0.2 + noise
    level = np.clip(level, dam["capacity"] * 0.1, dam["capacity"] * 1.02)

    return pd.DataFrame({
        "date": dates,
        "water_level_ft": np.round(level, 1),
        "rainfall_mm": np.round(np.maximum(0, rainfall), 1),
        "inflow_cusecs": np.round(np.maximum(0, inflow * 1000 + np.random.normal(0, 500, days)), 0),
    })


# ============================================================
#  RUN DEMO
# ============================================================
CLIPrinter.header("WavqWise: World Dam Monitoring & Flood Early Warning")
print("  Sense. Forecast. Alert. | v0.1.5")
print("  Output: PNG charts + CLI tables + CSV export")

# --- Data Sources ---
CLIPrinter.subheader("DATA SOURCES (All Free, No API Key)")
list_sources()

# --- 1. Load Dam Data ---
CLIPrinter.subheader("1. LOADING WORLD DAM NETWORK")
CLIPrinter.data_source(
    "Simulated (patterns match India-WRIS & USGS real data)",
    "india-wris.nrsc.gov.in | waterservices.usgs.gov",
    "Replace with real API calls for production deployment"
)

dam_data = {}
dam_results = []
for dam in WORLD_DAMS:
    data = generate_dam_data(dam, days=365)
    dam_data[dam["name"]] = data
    latest = data["water_level_ft"].iloc[-1]
    pct = (latest / dam["capacity"]) * 100
    dam_results.append([dam["name"], dam["country"], dam["river"],
                        f"{latest:.0f}", f"{dam['capacity']}", f"{pct:.0f}%"])

CLIPrinter.table(
    ["Dam", "Country", "River", "Level(ft)", "Capacity", "Fill%"],
    dam_results,
    [22, 10, 12, 10, 10, 8],
)

# --- 2. Forecast ---
CLIPrinter.subheader("2. WATER LEVEL FORECASTS (30-day)")
forecasts = {}
forecast_rows = []
for dam in WORLD_DAMS:
    data = dam_data[dam["name"]]
    p = WavqPipeline()
    p.load(data, target="water_level_ft", time="date")
    result = p.forecast(horizon=30, model="ema")
    forecasts[dam["name"]] = result

    current = data["water_level_ft"].iloc[-1]
    predicted = result.forecast["water_level_ft"].iloc[-1]
    change = predicted - current
    fc_pct = (predicted / dam["capacity"]) * 100
    risk = "FLOOD RISK" if fc_pct > 90 else "WATCH" if fc_pct > 75 else "OK"
    forecast_rows.append([dam["name"], f"{current:.0f}", f"{predicted:.0f}",
                         f"{change:+.0f}", f"{fc_pct:.0f}%", risk])

CLIPrinter.table(
    ["Dam", "Now(ft)", "30d(ft)", "Change", "30d Fill%", "Risk"],
    forecast_rows,
    [22, 10, 10, 8, 10, 12],
)

# --- 3. Anomaly Detection ---
CLIPrinter.subheader("3. ANOMALY DETECTION (Flood Early Warning)")
anomaly_rows = []
for dam in WORLD_DAMS:
    data = dam_data[dam["name"]]
    det = AnomalyPipeline()
    det.load(data, target="water_level_ft", time="date")
    result = det.detect(method="zscore")
    n = len(result.anomalies)
    if n > 0:
        worst = result.anomalies.nlargest(1, "anomaly_score").iloc[0]
        anomaly_rows.append([dam["name"], str(n), f"{worst['anomaly_score']:.2f}", worst.get("severity", "?")])
    else:
        anomaly_rows.append([dam["name"], "0", "-", "CLEAN"])

CLIPrinter.table(
    ["Dam", "Anomalies", "Worst Score", "Severity"],
    anomaly_rows,
    [22, 12, 14, 12],
)

# --- 4. Real-Time Streaming (Mettur Dam) ---
CLIPrinter.subheader("4. REAL-TIME STREAMING (Mettur Dam)")
mettur = dam_data["Mettur Dam"]
p = WavqPipeline()
p.load(mettur.iloc[:350], target="water_level_ft", time="date")
p.forecast(horizon=7, model="ema")

alerts = []
stream = p.stream(model="ema", anomaly_method="zscore",
                   on_anomaly=lambda e: alerts.append(e), anomaly_threshold=2.0)

for i in range(15):
    row = mettur.iloc[350 + i] if 350 + i < len(mettur) else mettur.iloc[-1]
    value = row["water_level_ft"]
    if i == 10:
        value = WORLD_DAMS[0]["capacity"] * 0.98
    stream.push({"date": row["date"], "water_level_ft": value})

print(f"  {stream.summary()}")
for a in alerts:
    for _, r in a.anomalies.iterrows():
        print(f"  ALERT: level={r['water_level_ft']:.1f}ft, severity={r.get('severity','?')}, score={r.get('anomaly_score',0):.2f}")

# --- 5. Model Comparison ---
CLIPrinter.subheader("5. MODEL COMPARISON (Mettur Dam, 14-day)")
p2 = WavqPipeline()
p2.load(mettur, target="water_level_ft", time="date")
comparison = p2.compare_models(models=["moving_average", "ema", "naive", "seasonal_naive"], horizon=14)
comp_rows = [[r["model"], f"{r['MAE']:.4f}"] for _, r in comparison.iterrows()]
CLIPrinter.table(["Model", "MAE"], comp_rows, [18, 12])
print(f"  Best model: {comparison.iloc[0]['model']}")

# --- 6. PNG Chart Output ---
CLIPrinter.subheader("6. GENERATING PNG CHARTS")
os.makedirs("demos/output", exist_ok=True)

# Forecast chart for Mettur Dam
path = ResultRenderer.forecast_chart(
    history=mettur, forecast=forecasts["Mettur Dam"].forecast,
    target="water_level_ft", time_col="date", model_name="EMA",
    title="Mettur Dam (Cauvery) - Water Level Forecast",
    source="Data pattern: India-WRIS (india-wris.nrsc.gov.in)",
    save_path="demos/output/mettur_forecast.png",
)
print(f"  Saved: {path}")

# Dam network overview
dam_chart_data = []
for dam in WORLD_DAMS[:6]:
    data = dam_data[dam["name"]]
    fc = forecasts[dam["name"]].forecast["water_level_ft"].values
    dam_chart_data.append({
        "name": dam["name"],
        "capacity": dam["capacity"],
        "level": data["water_level_ft"].iloc[-1],
        "timeseries": data["water_level_ft"].values,
    })
path = ResultRenderer.dam_network_chart(
    dam_chart_data,
    forecasts={d["name"]: forecasts[d["name"]].forecast["water_level_ft"].values for d in WORLD_DAMS[:6]},
    title="WavqWise Dam Network - 6 Dams Overview",
    source="Data pattern: India-WRIS + USGS",
    save_path="demos/output/dam_network.png",
)
print(f"  Saved: {path}")

# Model comparison chart
path = ResultRenderer.comparison_chart(
    comparison, title="Model Comparison - Mettur Dam (14-day MAE)",
    save_path="demos/output/model_comparison.png",
)
print(f"  Saved: {path}")

# --- 7. CSV/JSON Export ---
CLIPrinter.subheader("7. DATA EXPORT")
DataExporter.to_csv(mettur, "demos/output/mettur_dam_levels.csv", source="india-wris.nrsc.gov.in")
print(f"  CSV: demos/output/mettur_dam_levels.csv")

fc_df = forecasts["Mettur Dam"].forecast
DataExporter.to_json(fc_df, "demos/output/mettur_forecast.json", source="WavqWise EMA forecast")
print(f"  JSON: demos/output/mettur_forecast.json")

# --- Summary ---
CLIPrinter.header("DEMO COMPLETE")
print(f"  Dams analyzed: {len(WORLD_DAMS)} ({len(set(d['country'] for d in WORLD_DAMS))} countries)")
print(f"  Forecasts: 30-day for each dam")
print(f"  Anomalies: Z-score detection with severity")
print(f"  Streaming: Real-time monitoring with alerts")
print(f"  Charts: demos/output/*.png")
print(f"  Data: demos/output/*.csv, *.json")
print(f"\n  To use REAL data, connect to:")
print(f"    India: india-wris.nrsc.gov.in")
print(f"    USA: waterservices.usgs.gov")
print(f"    Global: globaldamwatch.org")
print(f"    Weather: api.open-meteo.com")
