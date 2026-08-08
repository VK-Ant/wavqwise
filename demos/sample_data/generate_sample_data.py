"""Generate sample datasets for demos and tests."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sales_data(days=365, start="2024-01-01"):
    dates = pd.date_range(start, periods=days, freq="D")
    trend = np.linspace(100, 150, days)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(days) / 7)
    yearly = 30 * np.sin(2 * np.pi * np.arange(days) / 365)
    noise = np.random.normal(0, 5, days)
    sales = trend + seasonal + yearly + noise
    return pd.DataFrame({"date": dates, "sales": np.maximum(sales, 0)})


def generate_sensor_data(n=10000, freq="1min", anomaly_pct=0.02):
    dates = pd.date_range("2024-01-01", periods=n, freq=freq)
    temp = 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 1440) + np.random.normal(0, 1, n)
    # Inject anomalies
    n_anomalies = int(n * anomaly_pct)
    anomaly_idx = np.random.choice(n, n_anomalies, replace=False)
    temp[anomaly_idx] += np.random.choice([-1, 1], n_anomalies) * np.random.uniform(15, 30, n_anomalies)
    return pd.DataFrame({"timestamp": dates, "temperature": temp})


def generate_stock_data(days=500, start="2023-01-01"):
    dates = pd.date_range(start, periods=days, freq="B")
    price = 100
    prices = [price]
    for _ in range(days - 1):
        price *= 1 + np.random.normal(0.0003, 0.015)
        prices.append(price)
    prices = np.array(prices)
    return pd.DataFrame({
        "Date": dates, "Open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        "High": prices * (1 + np.random.uniform(0, 0.02, days)),
        "Low": prices * (1 - np.random.uniform(0, 0.02, days)),
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, days),
    })


def generate_eeg_data(seconds=30, sample_rate=256, channels=4):
    n = seconds * sample_rate
    t = np.arange(n) / sample_rate
    data = {"time": t}
    ch_names = ["Fp1", "Fp2", "C3", "C4"][:channels]
    for ch in ch_names:
        alpha = 5 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        beta = 2 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
        theta = 3 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
        noise = np.random.normal(0, 1, n)
        data[ch] = alpha + beta + theta + noise
    return pd.DataFrame(data)


if __name__ == "__main__":
    generate_sales_data().to_csv("sales_daily.csv", index=False)
    generate_sensor_data().to_csv("sensor_readings.csv", index=False)
    generate_stock_data().to_csv("stock_ohlcv.csv", index=False)
    generate_eeg_data().to_csv("eeg_sample.csv", index=False)
    print("Sample data generated")
