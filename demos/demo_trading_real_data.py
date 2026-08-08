"""
WavqWise Demo: Trading Forecast (REAL DATA)
=============================================
Data: Real stock data from Yahoo Finance via yfinance
Tickers: AAPL, TSLA, MSFT (configurable)
Source: https://finance.yahoo.com

Shows:
  1. Download real OHLCV data
  2. Technical indicators (RSI, MACD, Bollinger, SMA crossover)
  3. Multi-model forecast comparison
  4. Incremental update with latest data
  5. Trading signal generation

Requirements: pip install wavqwise[trading]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wavqwise import WavqPipeline
from wavqwise.trading.indicators.momentum import RSIIndicator, StochasticIndicator
from wavqwise.trading.indicators.trend import MACDIndicator, SMAIndicator, EMAIndicator
from wavqwise.trading.indicators.volatility import BollingerBandsIndicator, ATRIndicator
from wavqwise.trading.indicators.volume import VWAPIndicator


def load_real_stock_data(ticker="AAPL", period="2y"):
    """Load REAL stock data from Yahoo Finance."""
    try:
        import yfinance as yf
        print(f"  Downloading {ticker} from Yahoo Finance...")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df = df.reset_index()

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]

        print(f"  Period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
        print(f"  Trading days: {len(df)}")
        print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
        print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
        total_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        print(f"  Total return: {total_ret:+.1f}%")
        return df

    except ImportError:
        print("  yfinance not installed. Install: pip install yfinance")
        raise
    except Exception as e:
        print(f"  Download failed: {e}")
        raise


def load_multi_stocks(tickers=("AAPL", "TSLA", "MSFT"), period="2y"):
    """Load multiple stocks for comparison."""
    try:
        import yfinance as yf
        stocks = {}
        for ticker in tickers:
            print(f"  Downloading {ticker}...")
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            df = df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == "" else col[0] for col in df.columns]
            stocks[ticker] = df
        return stocks
    except Exception as e:
        print(f"  Multi-stock download failed: {e}")
        raise


# === Run Demo ===
print("=" * 60)
print("WavqWise Trading Forecast - REAL DATA")
print("Sense. Forecast. Alert.")
print("=" * 60)

TICKER = "AAPL"

# 1. Load real data
print(f"\n[1/7] Loading real stock data ({TICKER})...")
stock = load_real_stock_data(TICKER, period="2y")

# 2. Add technical indicators
print(f"\n[2/7] Adding technical indicators...")
stock = RSIIndicator(14).compute(stock)
stock = MACDIndicator().compute(stock)
stock = SMAIndicator(20).compute(stock)
stock = SMAIndicator(50).compute(stock)
stock = SMAIndicator(200).compute(stock)
stock = EMAIndicator(12).compute(stock)
stock = EMAIndicator(26).compute(stock)
stock = BollingerBandsIndicator(20, 2).compute(stock)
stock = ATRIndicator(14).compute(stock)
stock = StochasticIndicator(14).compute(stock)

latest = stock.dropna().iloc[-1]
print(f"  RSI (14): {latest['RSI']:.1f} {'(Overbought)' if latest['RSI'] > 70 else '(Oversold)' if latest['RSI'] < 30 else '(Neutral)'}")
print(f"  MACD: {latest['MACD']:.4f} | Signal: {latest['MACD_signal']:.4f}")
print(f"  SMA-20: ${latest['SMA_20']:.2f} | SMA-50: ${latest['SMA_50']:.2f} | SMA-200: ${latest['SMA_200']:.2f}")
print(f"  Bollinger: ${latest['BB_lower']:.2f} - ${latest['BB_upper']:.2f}")
print(f"  ATR (14): ${latest['ATR']:.2f}")
print(f"  Stochastic %K: {latest['%K']:.1f} | %D: {latest['%D']:.1f}")

# Trend analysis
sma_trend = "Bullish" if latest["SMA_20"] > latest["SMA_50"] > latest["SMA_200"] else \
            "Bearish" if latest["SMA_20"] < latest["SMA_50"] < latest["SMA_200"] else "Mixed"
print(f"  Trend (SMA): {sma_trend}")

# 3. Forecast
print(f"\n[3/7] Forecasting close price (30 trading days)...")
pipeline = WavqPipeline()
pipeline.load(stock, target="Close", time="Date")

results = {}
for model_name in ["moving_average", "ema", "naive", "seasonal_naive"]:
    result = pipeline.forecast(horizon=30, model=model_name)
    results[model_name] = result
    print(f"  {model_name}: next day = ${result.forecast['Close'].iloc[0]:.2f}, "
          f"30d = ${result.forecast['Close'].iloc[-1]:.2f}")

# 4. Model comparison
print(f"\n[4/7] Model comparison (14-day backtest)...")
comparison = pipeline.compare_models(
    models=["moving_average", "ema", "naive", "seasonal_naive"],
    horizon=14
)
print(comparison.to_string(index=False))
best_model = comparison.iloc[0]["model"]
print(f"  Best model: {best_model}")

# 5. Incremental update demo
print(f"\n[5/7] Incremental update demo...")
train_data = stock.iloc[:-5]
new_data = stock.iloc[-5:]
p2 = WavqPipeline()
p2.load(train_data, target="Close", time="Date")
p2.forecast(horizon=5, model="moving_average")
p2.update(new_data)
updated_forecast = p2.forecast(horizon=5, model="moving_average")
print(f"  Updated with {len(new_data)} new trading days")
print(f"  Next 5 days forecast: ${updated_forecast.forecast['Close'].iloc[0]:.2f} - ${updated_forecast.forecast['Close'].iloc[-1]:.2f}")

# 6. Trading signals
print(f"\n[6/7] Generating trading signals...")
clean = stock.dropna().copy()
clean["signal"] = 0
# Buy: RSI < 30 and price below lower Bollinger
clean.loc[(clean["RSI"] < 30) & (clean["Close"] < clean["BB_lower"]), "signal"] = 1
# Sell: RSI > 70 and price above upper Bollinger
clean.loc[(clean["RSI"] > 70) & (clean["Close"] > clean["BB_upper"]), "signal"] = -1
# SMA golden/death cross
clean["golden_cross"] = (clean["SMA_50"] > clean["SMA_200"]) & (clean["SMA_50"].shift(1) <= clean["SMA_200"].shift(1))
clean["death_cross"] = (clean["SMA_50"] < clean["SMA_200"]) & (clean["SMA_50"].shift(1) >= clean["SMA_200"].shift(1))

n_buy = (clean["signal"] == 1).sum()
n_sell = (clean["signal"] == -1).sum()
n_golden = clean["golden_cross"].sum()
n_death = clean["death_cross"].sum()
print(f"  Buy signals (RSI+BB): {n_buy}")
print(f"  Sell signals (RSI+BB): {n_sell}")
print(f"  Golden crosses: {n_golden}")
print(f"  Death crosses: {n_death}")

# 7. Visualization
print(f"\n[7/7] Generating charts...")
best_forecast = results[best_model]

fig, axes = plt.subplots(5, 1, figsize=(16, 20), gridspec_kw={"height_ratios": [4, 1.2, 1.2, 1.2, 1.2]})
fig.suptitle(f"WavqWise Trading Analysis: {TICKER} (Real Data)", fontsize=16, fontweight="bold")

# Price + indicators + forecast
ax = axes[0]
ax.plot(clean["Date"], clean["Close"], color="#1e293b", linewidth=1.2, label="Close")
ax.plot(clean["Date"], clean["SMA_20"], color="#2563eb", linewidth=0.7, alpha=0.6, label="SMA-20")
ax.plot(clean["Date"], clean["SMA_50"], color="#dc2626", linewidth=0.7, alpha=0.6, label="SMA-50")
ax.plot(clean["Date"], clean["SMA_200"], color="#059669", linewidth=0.7, alpha=0.6, label="SMA-200")
ax.fill_between(clean["Date"], clean["BB_lower"], clean["BB_upper"], alpha=0.08, color="#6366f1", label="Bollinger")

# Forecast overlay
fc = best_forecast.forecast
ax.plot(fc["Date"], fc["Close"], color="#f59e0b", linewidth=2.5, linestyle="--", label=f"Forecast ({best_model})")
ax.fill_between(fc["Date"], fc["Close_lower"], fc["Close_upper"], alpha=0.15, color="#f59e0b")

# Buy/Sell markers
buys = clean[clean["signal"] == 1]
sells = clean[clean["signal"] == -1]
if len(buys) > 0:
    ax.scatter(buys["Date"], buys["Close"], marker="^", color="#059669", s=80, zorder=5, label="Buy signal")
if len(sells) > 0:
    ax.scatter(sells["Date"], sells["Close"], marker="v", color="#dc2626", s=80, zorder=5, label="Sell signal")

ax.set_ylabel("Price ($)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Volume
ax = axes[1]
colors = ["#059669" if clean["Close"].iloc[i] >= clean["Open"].iloc[i] else "#dc2626"
          for i in range(len(clean))]
ax.bar(clean["Date"], clean["Volume"], color=colors, alpha=0.5, width=1)
ax.set_ylabel("Volume")
ax.grid(True, alpha=0.3)

# RSI
ax = axes[2]
ax.plot(clean["Date"], clean["RSI"], color="#7c3aed", linewidth=1)
ax.axhline(70, color="#dc2626", linestyle="--", alpha=0.5)
ax.axhline(30, color="#059669", linestyle="--", alpha=0.5)
ax.fill_between(clean["Date"], 30, 70, alpha=0.05, color="#6366f1")
ax.set_ylabel("RSI (14)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# MACD
ax = axes[3]
ax.plot(clean["Date"], clean["MACD"], color="#2563eb", linewidth=1, label="MACD")
ax.plot(clean["Date"], clean["MACD_signal"], color="#dc2626", linewidth=1, label="Signal")
hist_colors = ["#059669" if v >= 0 else "#dc2626" for v in clean["MACD_hist"]]
ax.bar(clean["Date"], clean["MACD_hist"], color=hist_colors, alpha=0.5, width=1)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("MACD")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Stochastic
ax = axes[4]
ax.plot(clean["Date"], clean["%K"], color="#2563eb", linewidth=1, label="%K")
ax.plot(clean["Date"], clean["%D"], color="#dc2626", linewidth=1, label="%D")
ax.axhline(80, color="#dc2626", linestyle="--", alpha=0.5)
ax.axhline(20, color="#059669", linestyle="--", alpha=0.5)
ax.set_ylabel("Stochastic")
ax.set_ylim(0, 100)
ax.set_xlabel("Date")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("demos/trading_real_data_results.png", dpi=150, bbox_inches="tight")
print(f"  Plot saved: demos/trading_real_data_results.png")
print(f"\nDone. {TICKER} analysis complete.")
