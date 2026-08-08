"""WavqWise Forecasting Demo - Quick Start"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wavqwise import WavqPipeline

# === 1. Load data (5 lines to forecast) ===
pipeline = WavqPipeline()
pipeline.load("demos/sample_data/sales_daily.csv", target="sales", time="date")

# === 2. Forecast with Moving Average ===
print("=== Moving Average Forecast ===")
result = pipeline.forecast(horizon=30, model="moving_average")
print(result.summary())

# === 3. Try different model - just change the string ===
print("\n=== EMA Forecast ===")
result_ema = pipeline.forecast(horizon=30, model="ema")
print(result_ema.summary())

# === 4. Compare models ===
print("\n=== Model Comparison ===")
comparison = pipeline.compare_models(models=["moving_average", "ema", "naive"], horizon=14)
print(comparison.to_string(index=False))

# === 5. Incremental update ===
print("\n=== Incremental Update ===")
import pandas as pd
import numpy as np
new_data = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=7, freq="D"),
    "sales": np.random.normal(140, 10, 7),
})
pipeline.update(new_data)
result_updated = pipeline.forecast(horizon=14, model="moving_average")
print(f"After update: {result_updated.summary()}")

print("\nPipeline info:")
print(pipeline.info())
print("\nAvailable models:", WavqPipeline.available_models())
