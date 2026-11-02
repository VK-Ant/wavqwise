"""
WavqWise Demo: Auto Feature Engineering (v0.1.7)
==================================================
Demonstrates the AutoFeatureEngineer with 12 feature groups
generating 80+ features from a single target column.

Groups: lags, rolling, ewm, calendar, cyclical, fourier,
        trend, diff, pct_change, rate_of_change, volatility,
        interaction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from wavqwise.features import AutoFeatureEngineer


# === 1. Create synthetic time-series data ===
print("=" * 60)
print("WavqWise Auto Feature Engineering Demo")
print("12 feature groups · 80+ features · zero manual work")
print("=" * 60)

np.random.seed(42)
n = 365
dates = pd.date_range("2024-01-01", periods=n, freq="D")
trend = np.linspace(100, 150, n)
seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 3, n)
values = trend + seasonal + noise

df = pd.DataFrame({"timestamp": dates, "value": values})
print(f"\n[1/4] Generated {n} days of synthetic data (trend + seasonality + noise)")
print(f"       Columns: {list(df.columns)}")

# === 2. Generate ALL features ===
print(f"\n[2/4] Generating features (all 12 groups)...")
fe = AutoFeatureEngineer()
enriched = fe.generate(df, target="value", time_col="timestamp")
report = fe.feature_report()

print(f"       Original columns:  {len(df.columns)}")
print(f"       New features:      {report['total_features']}")
print(f"       Final columns:     {len(enriched.columns)}")
print(f"       Rows after dropna: {len(enriched)} (from {n})")

# === 3. Show feature groups ===
print(f"\n[3/4] Feature groups generated:")
groups = {}
for feat in report["feature_names"]:
    # Infer group from prefix
    if feat.startswith("lag_"):
        g = "Lags"
    elif feat.startswith("rolling_mean") or feat.startswith("rolling_std") or \
         feat.startswith("rolling_min") or feat.startswith("rolling_max") or \
         feat.startswith("rolling_med") or feat.startswith("rolling_skew") or \
         feat.startswith("rolling_kurt"):
        g = "Rolling"
    elif feat.startswith("ewm_"):
        g = "EWM"
    elif feat in ("hour", "day_of_week", "day_of_month", "day_of_year",
                  "week_of_year", "month", "quarter", "is_weekend",
                  "is_month_start", "is_month_end", "is_quarter_start",
                  "is_quarter_end"):
        g = "Calendar"
    elif feat.endswith("_sin") or feat.endswith("_cos"):
        if feat.startswith("fourier_"):
            g = "Fourier"
        else:
            g = "Cyclical"
    elif feat.startswith("trend_"):
        g = "Trend"
    elif feat.startswith("diff_"):
        g = "Differencing"
    elif feat.startswith("pct_change"):
        g = "Pct Change"
    elif feat.startswith("roc_") or feat.startswith("momentum_"):
        g = "Rate of Change"
    elif feat.startswith("rolling_zscore") or feat.startswith("band_"):
        g = "Volatility"
    elif feat.startswith("mean_ratio") or feat.startswith("mean_crossover") or \
         feat.startswith("std_ratio") or feat.startswith("dist_from_mean"):
        g = "Interaction"
    else:
        g = "Other"
    groups.setdefault(g, []).append(feat)

for g, feats in sorted(groups.items()):
    print(f"  {g:20s} : {len(feats):3d} features  ({', '.join(feats[:3])}{'...' if len(feats) > 3 else ''})")

# === 4. Selective feature groups ===
print(f"\n[4/4] Selective generation (lags + ewm + cyclical only)...")
fe_selective = AutoFeatureEngineer(
    feature_groups=["lags", "ewm", "cyclical"],
    lag_periods=[1, 7, 14],
    ewm_spans=[7, 14],
)
enriched_sel = fe_selective.generate(df, target="value", time_col="timestamp")
report_sel = fe_selective.feature_report()
print(f"       Features generated: {report_sel['total_features']}")
print(f"       Features: {report_sel['feature_names']}")

# === Show a sample of enriched data ===
print(f"\n{'=' * 60}")
print("Sample enriched row (first 10 features):")
print("=" * 60)
sample = enriched.iloc[0]
feat_cols = [c for c in enriched.columns if c not in ("timestamp", "value")]
for col in feat_cols[:10]:
    print(f"  {col:25s} = {sample[col]:>10.3f}")
print(f"  ... and {len(feat_cols) - 10} more features")

print(f"\nDone. {report['total_features']} features ready for forecasting / ML.")
