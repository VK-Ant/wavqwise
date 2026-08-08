"""
WavqWise Demo: Trading Forecast
================================
Forecast stock prices and generate trading signals using technical indicators.

Data: Realistic synthetic OHLCV data (no API key needed)
Shows:
  1. Load OHLCV stock data
  2. Add technical indicators (RSI, MACD, Bollinger Bands)
  3. Forecast close price with multiple models
  4. Compare model performance
  5. Simple momentum backtest
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavqwise import WavqPipeline
from wavqwise.trading.indicators.momentum import RSIIndicator
from wavqwise.trading.indicators.trend import MACDIndicator, SMAIndicator
from wavqwise.trading.indicators.volatility import BollingerBandsIndicator


# === 1. Generate realistic stock data ===
def generate_realistic_stock(ticker="WAVQ", days=500, seed=42):
    """Generate realistic stock OHLCV data with trends, volatility clusters, and volume."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="B")

    # Price with regime changes
    price = 100.0
    prices = [price]
    regime = 0  # 0=normal, 1=bull, 2=bear
    for i in range(days - 1):
        # Regime switching
        if np.random.random() < 0.02:
            regime = np.random.choice([0, 1, 2])

        drift = {0: 0.0002, 1: 0.001, 2: -0.0008}[regime]
        vol = {0: 0.015, 1: 0.012, 2: 0.022}[regime]

        # GARCH-like volatility clustering
        if i > 0 and abs(prices[-1] / prices[-2] - 1) > 0.02:
            vol *= 1.5

        ret = np.random.normal(drift, vol)
        price *= (1 + ret)
        prices.append(max(price, 1))

    prices = np.array(prices)

    # Generate OHLCV
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "High": prices * (1 + np.random.uniform(0.002, 0.025, days)),
        "Low": prices * (1 - np.random.uniform(0.002, 0.025, days)),
        "Close": prices,
        "Volume": (np.random.lognormal(15, 0.5, days)).astype(int),
    })

    # Ensure High >= Open,Close and Low <= Open,Close
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    return df


# === Run Demo ===
print("=" * 60)
print("WavqWise Trading Forecast Demo")
print("Sense. Forecast. Alert.")
print("=" * 60)

# Generate data
print("\n[1/6] Generating realistic stock data (WAVQ)...")
stock = generate_realistic_stock(ticker="WAVQ", days=500)
print(f"  Period: {stock['Date'].iloc[0].date()} to {stock['Date'].iloc[-1].date()}")
print(f"  Trading days: {len(stock)}")
print(f"  Price range: ${stock['Close'].min():.2f} - ${stock['Close'].max():.2f}")
print(f"  Total return: {(stock['Close'].iloc[-1]/stock['Close'].iloc[0]-1)*100:.1f}%")

# Save for reproducibility
stock.to_csv("demos/sample_data/stock_ohlcv.csv", index=False)

# Add technical indicators
print("\n[2/6] Adding technical indicators...")
stock = RSIIndicator(period=14).compute(stock)
stock = MACDIndicator().compute(stock)
stock = SMAIndicator(period=20).compute(stock)
stock = SMAIndicator(period=50).compute(stock)
stock = BollingerBandsIndicator(period=20, std_dev=2).compute(stock)

print(f"  RSI (14): latest = {stock['RSI'].iloc[-1]:.1f}")
print(f"  MACD: latest = {stock['MACD'].iloc[-1]:.4f}")
print(f"  SMA-20: ${stock['SMA_20'].iloc[-1]:.2f}")
print(f"  SMA-50: ${stock['SMA_50'].iloc[-1]:.2f}")
print(f"  Bollinger Width: ${(stock['BB_upper'].iloc[-1] - stock['BB_lower'].iloc[-1]):.2f}")

# Forecast with WavqPipeline
print("\n[3/6] Forecasting close price (30 trading days)...")
pipeline = WavqPipeline()
pipeline.load(stock, target="Close", time="Date")

# Moving Average forecast
result_sma = pipeline.forecast(horizon=30, model="moving_average")
print(f"  Moving Average: next day = ${result_sma.forecast['Close'].iloc[0]:.2f}")

# EMA forecast
result_ema = pipeline.forecast(horizon=30, model="ema")
print(f"  EMA: next day = ${result_ema.forecast['Close'].iloc[0]:.2f}")

# Naive forecast
result_naive = pipeline.forecast(horizon=30, model="naive")
print(f"  Naive: next day = ${result_naive.forecast['Close'].iloc[0]:.2f}")

# Compare models
print("\n[4/6] Model comparison (14-day horizon)...")
comparison = pipeline.compare_models(
    models=["moving_average", "ema", "naive", "seasonal_naive"],
    horizon=14
)
print(comparison.to_string(index=False))
best = comparison.iloc[0]["model"]
print(f"  Winner: {best}")

# Simple momentum signals
print("\n[5/6] Generating trading signals...")
stock_clean = stock.dropna().copy()
stock_clean["signal"] = 0
# Buy: RSI < 30 and price below lower Bollinger
stock_clean.loc[(stock_clean["RSI"] < 30) & (stock_clean["Close"] < stock_clean["BB_lower"]), "signal"] = 1
# Sell: RSI > 70 and price above upper Bollinger
stock_clean.loc[(stock_clean["RSI"] > 70) & (stock_clean["Close"] > stock_clean["BB_upper"]), "signal"] = -1
# SMA crossover
stock_clean.loc[stock_clean["SMA_20"] > stock_clean["SMA_50"], "trend"] = "bullish"
stock_clean.loc[stock_clean["SMA_20"] <= stock_clean["SMA_50"], "trend"] = "bearish"

n_buy = (stock_clean["signal"] == 1).sum()
n_sell = (stock_clean["signal"] == -1).sum()
print(f"  Buy signals: {n_buy}")
print(f"  Sell signals: {n_sell}")
print(f"  Current trend: {stock_clean['trend'].iloc[-1]}")

# Visualization
print("\n[6/6] Generating charts...")
fig, axes = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={"height_ratios": [3, 1, 1, 2]})
fig.suptitle("WavqWise Trading Analysis: WAVQ", fontsize=14, fontweight="bold")

# Price + Bollinger + SMA + Forecast
ax = axes[0]
ax.plot(stock_clean["Date"], stock_clean["Close"], color="#1e293b", linewidth=1.2, label="Close")
ax.plot(stock_clean["Date"], stock_clean["SMA_20"], color="#2563eb", linewidth=0.8, alpha=0.7, label="SMA-20")
ax.plot(stock_clean["Date"], stock_clean["SMA_50"], color="#dc2626", linewidth=0.8, alpha=0.7, label="SMA-50")
ax.fill_between(stock_clean["Date"], stock_clean["BB_lower"], stock_clean["BB_upper"],
                alpha=0.1, color="#6366f1", label="Bollinger Bands")
# Forecast
ax.plot(result_sma.forecast["Date"], result_sma.forecast["Close"],
        color="#059669", linewidth=2, linestyle="--", label="Forecast (SMA)")
ax.fill_between(result_sma.forecast["Date"],
                result_sma.forecast["Close_lower"], result_sma.forecast["Close_upper"],
                alpha=0.15, color="#059669")
# Buy/Sell markers
buys = stock_clean[stock_clean["signal"] == 1]
sells = stock_clean[stock_clean["signal"] == -1]
ax.scatter(buys["Date"], buys["Close"], marker="^", color="#059669", s=60, zorder=5, label="Buy")
ax.scatter(sells["Date"], sells["Close"], marker="v", color="#dc2626", s=60, zorder=5, label="Sell")
ax.set_ylabel("Price ($)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Volume
ax = axes[1]
colors = ["#059669" if stock_clean["Close"].iloc[i] >= stock_clean["Open"].iloc[i] else "#dc2626"
          for i in range(len(stock_clean))]
ax.bar(stock_clean["Date"], stock_clean["Volume"], color=colors, alpha=0.6, width=1)
ax.set_ylabel("Volume")
ax.grid(True, alpha=0.3)

# RSI
ax = axes[2]
ax.plot(stock_clean["Date"], stock_clean["RSI"], color="#7c3aed", linewidth=1)
ax.axhline(70, color="#dc2626", linestyle="--", alpha=0.5, linewidth=0.8)
ax.axhline(30, color="#059669", linestyle="--", alpha=0.5, linewidth=0.8)
ax.fill_between(stock_clean["Date"], 30, 70, alpha=0.05, color="#6366f1")
ax.set_ylabel("RSI")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# MACD
ax = axes[3]
ax.plot(stock_clean["Date"], stock_clean["MACD"], color="#2563eb", linewidth=1, label="MACD")
ax.plot(stock_clean["Date"], stock_clean["MACD_signal"], color="#dc2626", linewidth=1, label="Signal")
macd_hist = stock_clean["MACD_hist"]
colors_hist = ["#059669" if v >= 0 else "#dc2626" for v in macd_hist]
ax.bar(stock_clean["Date"], macd_hist, color=colors_hist, alpha=0.5, width=1)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("MACD")
ax.set_xlabel("Date")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("demos/trading_forecast_results.png", dpi=150, bbox_inches="tight")
print(f"  Plot saved: demos/trading_forecast_results.png")
print("\nDone.")
