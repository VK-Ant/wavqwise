<p align="center">
  <img src="assets/wavqwise_hero.png" alt="WavqWise" width="100%"/>
</p>

<h1 align="center">WavqWise</h1>
<h3 align="center">Sense. Forecast. Alert.</h3>
<p align="center">Pluggable temporal intelligence. Any model. Any signal. Five lines to forecast.</p>

<p align="center">
  <a href="https://pypi.org/project/wavqwise/"><img src="https://img.shields.io/pypi/v/wavqwise?color=blue" alt="PyPI"/></a>
  <a href="https://github.com/VK-Ant/wavqwise/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"/></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#one-import-any-model">33+ Models</a> |
  <a href="#auto-gpu--onnx--tensorrt">Auto GPU</a> |
  <a href="#real-data-demos">Real Data Demos</a> |
  <a href="#colab-notebooks">Colab Notebooks</a> |
  <a href="#ecosystem">Ecosystem</a>
</p>

---

## Why WavqWise?

| Without WavqWise | With WavqWise |
|:---|:---|
| Import statsmodels for ARIMA | `model="arima"` |
| Import xgboost for ML | `model="xgboost"` |
| Import chronos for foundation | `model="chronos"` |
| Write preprocessing pipeline | Auto-handled |
| Write evaluation loop | Built-in |
| Manual GPU setup | Auto-detected |
| Full retrain on new data | `.update()` incremental |
| **~50 lines per model** | **5 lines, any model** |

---

## Quick Start

```bash
pip install wavqwise
```

```python
from wavqwise import WavqPipeline

pipeline = WavqPipeline()
pipeline.load("sales.csv", target="revenue", time="date")
forecast = pipeline.forecast(horizon=30, model="arima")
forecast.plot()
```

**That's it.** No model imports. No boilerplate. Change `"arima"` to `"xgboost"` or `"chronos"` — same code.

---

## One Import. Any Model.

```python
from wavqwise import WavqPipeline  # The ONLY import you need

# Traditional
forecast = pipeline.forecast(horizon=30, model="moving_average")
forecast = pipeline.forecast(horizon=30, model="arima")
forecast = pipeline.forecast(horizon=30, model="ets")
forecast = pipeline.forecast(horizon=30, model="holtwinters")
forecast = pipeline.forecast(horizon=30, model="theta")

# ML
forecast = pipeline.forecast(horizon=30, model="xgboost")
forecast = pipeline.forecast(horizon=30, model="lightgbm")
forecast = pipeline.forecast(horizon=30, model="random_forest")

# Foundation Models
forecast = pipeline.forecast(horizon=30, model="chronos")
forecast = pipeline.forecast(horizon=30, model="timesfm")

# Cloud / LLM
forecast = pipeline.forecast(horizon=30, model="ollama:llama3")

# Auto-select best model
forecast = pipeline.forecast(horizon=30, model="auto")

# Ensemble
forecast = pipeline.forecast(horizon=30, model=["arima", "ets", "xgboost"])
```

### 33+ Pluggable Models

| Category | Models |
|----------|--------|
| **Traditional** | Moving Average, EMA, ARIMA, SARIMA, ETS, Holt-Winters, Theta, CES, Croston, Naive, Seasonal Naive |
| **ML** | XGBoost, LightGBM, CatBoost, Random Forest, Ridge, Lasso, ElasticNet |
| **Neural** | NeuralProphet, N-BEATS, TFT |
| **Foundation** | Chronos, TimesFM, Lag-Llama, Moirai, HuggingFace Hub |
| **Cloud** | TimeGPT, Ollama, OpenAI, Anthropic |

---

## Incremental Training

Most libraries retrain from scratch. WavqWise updates in-place:

```python
pipeline = WavqPipeline()
pipeline.load(historical_data, target="sales", time="date")
pipeline.forecast(horizon=30, model="arima")

# New data arrives — update, don't retrain
pipeline.update(new_week_data)
forecast = pipeline.forecast(horizon=30)
```

---

## Auto GPU / ONNX / TensorRT

WavqWise auto-detects the best available hardware on startup:

```python
pipeline = WavqPipeline()              # Auto-detect
pipeline = WavqPipeline(device="cuda")  # Force CUDA
pipeline = WavqPipeline(device="tensorrt")  # Force TensorRT

print(pipeline.runtime_info())
```

```
WavqWise Runtime Engine
========================================
Backend:  tensorrt
Device:   NVIDIA RTX 4090
GPU:      Yes
VRAM:     24.0 GB
Compute:  SM 8.9
Available: cpu, onnx-cpu, onnx-gpu, cuda, tensorrt
```

**Detection priority:** TensorRT → ONNX GPU → PyTorch CUDA → PyTorch MPS (Apple) → ONNX CPU → CPU

### ONNX Export for Production

```python
from wavqwise.runtime import ONNXExporter, ONNXPredictor

# Export trained model to ONNX
exporter = ONNXExporter()
exporter.export_sklearn(trained_model, "model.onnx", n_features=13)

# Optimize with TensorRT (FP16)
exporter.optimize_for_tensorrt("model.onnx")

# Fast inference
predictor = ONNXPredictor("model.onnx")  # Auto GPU
result = predictor.predict(input_array)
print(predictor.benchmark(input_array))   # Latency report
```

---

## Anomaly Detection

```python
from wavqwise import AnomalyPipeline

detector = AnomalyPipeline()
detector.load("sensor_data.csv", target="temperature", time="timestamp")
anomalies = detector.detect(method="zscore")  # or "iqr", "isolation_forest"
anomalies.plot(show_severity=True)
print(anomalies.summary())
# Anomalies: 93/10000 (0.9%) | Method: zscore
```

---

## EEG & Signal Processing

```python
from wavqwise import SignalPipeline

sig = SignalPipeline()
sig.load("eeg.csv", channels=["Fp1", "Fp2", "C3", "C4"], sample_rate=256)
sig.filter(low=1, high=50, notch=50)
bands = sig.extract_bands(["delta", "theta", "alpha", "beta", "gamma"])
bands.plot_bands()
events = sig.detect_events(threshold=3.0)
```

---

## Trading & Financial Analysis

```python
from wavqwise import WavqPipeline
from wavqwise.trading.indicators.momentum import RSIIndicator
from wavqwise.trading.indicators.trend import MACDIndicator, SMAIndicator
from wavqwise.trading.indicators.volatility import BollingerBandsIndicator

# Load real stock data
import yfinance as yf
stock = yf.download("AAPL", period="2y", auto_adjust=True).reset_index()

# Add indicators
stock = RSIIndicator(14).compute(stock)
stock = MACDIndicator().compute(stock)
stock = BollingerBandsIndicator(20, 2).compute(stock)

# Forecast
pipeline = WavqPipeline()
pipeline.load(stock, target="Close", time="Date")
forecast = pipeline.forecast(horizon=30, model="ema")
```

---

## Database Support

```python
# PostgreSQL / TimescaleDB
pipeline.load("postgresql://user:pass@host:5432/db",
              query="SELECT timestamp, value FROM sensors")

# SQLite
pipeline.load("sqlite:///local.db", table="readings")

# MongoDB / InfluxDB
pipeline.load("mongodb://host:27017/db", collection="data")
```

---

## Real Data Demos

All demos use **real open-source data** — no synthetic:

| Demo | Data Source | Run |
|------|-----------|-----|
| EEG Classification | MNE Sample Dataset (real clinical EEG) | `python demos/demo_eeg_real_data.py` |
| Trading Forecast | Yahoo Finance via yfinance (AAPL) | `python demos/demo_trading_real_data.py` |
| Anomaly Detection | Real sensor readings | `python demos/demo_anomaly_detection.py` |
| Forecasting | Sales data | `python demos/demo_forecasting.py` |
| EEG Band Analysis | Signal processing | `python demos/demo_eeg_analysis.py` |

---

## Colab Notebooks

Run in browser, zero setup:

| Notebook | Data | Open |
|----------|------|------|
| EEG Classification | MNE real EEG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/wavqwise/blob/main/demos/notebooks/wavqwise_eeg_real_data.ipynb) |
| Trading Forecast | yfinance AAPL/TSLA/MSFT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VK-Ant/wavqwise/blob/main/demos/notebooks/wavqwise_trading_real_data.ipynb) |

---

## CLI

```bash
# Forecast
wavqwise forecast --input data.csv --target sales --model arima --horizon 30

# Anomaly detection
wavqwise detect --input sensor.csv --target temperature --method zscore

# List models
wavqwise models
```

---

## Install Options

```bash
pip install wavqwise                    # Core (numpy, pandas, sklearn)
pip install wavqwise[traditional]       # + ARIMA, SARIMA, ETS, Theta
pip install wavqwise[ml]               # + XGBoost, LightGBM, CatBoost
pip install wavqwise[neural]           # + NeuralProphet, N-BEATS, TFT
pip install wavqwise[foundation]       # + Chronos, TimesFM, Moirai
pip install wavqwise[signals]          # + EEG, spectral analysis (MNE)
pip install wavqwise[trading]          # + yfinance, indicators, backtest
pip install wavqwise[database]         # + PostgreSQL, MongoDB, InfluxDB
pip install wavqwise[onnx-gpu]         # + ONNX Runtime GPU
pip install wavqwise[tensorrt]         # + TensorRT optimization
pip install wavqwise[all]              # Everything
```

---

## Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
# WavqWise + TimescaleDB + Grafana ready at localhost:8888
```

## Ecosystem

WavqWise is part of the **VK-Ant pluggable AI ecosystem** — six libraries, one architecture:

| Library | Domain | Tagline | PyPI |
|---------|--------|---------|------|
| [SightRAG](https://github.com/VK-Ant/SightRAG) | Vision | See. Search. Retrieve. | [![PyPI](https://img.shields.io/pypi/v/sightrag?color=blue)](https://pypi.org/project/sightrag/) |
| [Sonarwise](https://github.com/VK-Ant/sonarwise) | Audio | Hear. Search. Retrieve. | [![PyPI](https://img.shields.io/pypi/v/sonarwise?color=blue)](https://pypi.org/project/sonarwise/) |
| [Docqwise](https://github.com/VK-Ant/docqwise) | Documents | Read. Extract. Retrieve. | [![PyPI](https://img.shields.io/pypi/v/docqwise?color=blue)](https://pypi.org/project/docqwise/) |
| **WavqWise** | **Temporal** | **Sense. Forecast. Alert.** | [![PyPI](https://img.shields.io/pypi/v/wavqwise?color=blue)](https://pypi.org/project/wavqwise/) |
| [Adaptive Intelligence](https://github.com/VK-Ant/adaptive-intelligence) | Orchestration | Learn. Remember. Adapt. | [![PyPI](https://img.shields.io/pypi/v/adaptive-intelligence?color=blue)](https://pypi.org/project/adaptive-intelligence/) |
| [LLMEvalKit](https://github.com/VK-Ant/llmevalkit) | Evaluation | Evaluate. Score. Improve. | [![PyPI](https://img.shields.io/pypi/v/llmevalkit?color=blue)](https://pypi.org/project/llmevalkit/) |

---

## Contributing

```bash
git clone https://github.com/VK-Ant/wavqwise.git
cd wavqwise
pip install -e ".[dev]"
make test          # 18 tests (smoke + sanity + A/B)
make lint          # ruff + mypy
```

---

## License

Apache 2.0 License 

## Author

Venkatkumar Rajan
