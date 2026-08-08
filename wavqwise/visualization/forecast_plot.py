"""Forecast visualization."""
import matplotlib.pyplot as plt
import pandas as pd

class ForecastPlot:
    def __init__(self, engine="matplotlib", style="default"):
        self.engine = engine
        self.style = style

    def plot(self, history, forecast, target, time_col, model_name="",
             show_confidence=True, show_components=False, **kwargs):
        if self.engine == "plotly":
            return self._plot_plotly(history, forecast, target, time_col, model_name, show_confidence)
        return self._plot_matplotlib(history, forecast, target, time_col, model_name, show_confidence)

    def _plot_matplotlib(self, history, forecast, target, time_col, model_name, show_confidence):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history[time_col], history[target], label="History", color="#2563eb", linewidth=1.5)
        ax.plot(forecast[time_col], forecast[target], label=f"Forecast ({model_name})", color="#dc2626", linewidth=2, linestyle="--")
        if show_confidence and f"{target}_lower" in forecast.columns:
            ax.fill_between(forecast[time_col], forecast[f"{target}_lower"], forecast[f"{target}_upper"],
                            alpha=0.15, color="#dc2626", label="Confidence interval")
        ax.set_xlabel("Time")
        ax.set_ylabel(target)
        ax.set_title(f"WavqWise Forecast: {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    def _plot_plotly(self, history, forecast, target, time_col, model_name, show_confidence):
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history[time_col], y=history[target], name="History", line=dict(color="#2563eb")))
            fig.add_trace(go.Scatter(x=forecast[time_col], y=forecast[target], name=f"Forecast ({model_name})", line=dict(color="#dc2626", dash="dash")))
            if show_confidence and f"{target}_lower" in forecast.columns:
                fig.add_trace(go.Scatter(x=forecast[time_col], y=forecast[f"{target}_upper"], mode="lines", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast[time_col], y=forecast[f"{target}_lower"], fill="tonexty", mode="lines", line=dict(width=0), name="Confidence"))
            fig.update_layout(title=f"WavqWise Forecast: {model_name}", template="plotly_white")
            fig.show()
            return fig
        except ImportError:
            return self._plot_matplotlib(history, forecast, target, time_col, model_name, show_confidence)

    def plot_series(self, data, target, time_col, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(data[time_col], data[target], color="#2563eb", linewidth=1.5)
        ax.set_title("Time Series Data")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig
