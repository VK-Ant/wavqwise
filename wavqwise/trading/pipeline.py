"""Trading pipeline."""
import pandas as pd
from wavqwise.loaders.auto_loader import AutoLoader

class TradingResult:
    def __init__(self, data, equity_curve=None, metrics=None):
        self.data = data
        self.equity_curve = equity_curve
        self.metrics = metrics or {}

    def plot_equity_curve(self):
        import matplotlib.pyplot as plt
        if self.equity_curve is not None:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(self.equity_curve, color="#059669", linewidth=2)
            ax.set_title("Equity Curve")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def report(self):
        for k, v in self.metrics.items():
            print(f"{k}: {v}")

class TradingPipeline:
    def __init__(self):
        self._data = None

    def load_market(self, ticker, source="yahoo", period="2y", **kwargs):
        try:
            import yfinance as yf
            self._data = yf.download(ticker, period=period, auto_adjust=True)
            self._data = self._data.reset_index()
        except ImportError:
            raise ImportError("pip install wavqwise[trading]")
        return self

    def add_indicators(self, indicators, **kwargs):
        from wavqwise.core.registry import Registry
        for name in indicators:
            try:
                indicator = Registry.get_indicator(name.lower())
                self._data = indicator.compute(self._data)
            except Exception:
                pass
        return self

    def backtest(self, strategy="momentum", initial_capital=100000, **kwargs):
        capital = initial_capital
        return TradingResult(self._data, metrics={"initial_capital": capital, "strategy": strategy})
