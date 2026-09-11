"""
WavqWise Output Renderer
==========================
Renders results as:
  1. PNG charts (matplotlib) - default, works everywhere
  2. CLI text tables - terminal friendly
  3. CSV/JSON export - programmatic access
  4. Data source attribution - user knows where data comes from

No HTML. No browser dependency. Just images and text.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


class ResultRenderer:
    """Unified output renderer for all WavqWise results."""

    # Color palette
    COLORS = {
        "primary": "#2563eb",
        "forecast": "#dc2626",
        "confidence": "#dc2626",
        "anomaly_low": "#fbbf24",
        "anomaly_medium": "#f97316",
        "anomaly_high": "#ef4444",
        "anomaly_critical": "#7f1d1d",
        "normal": "#059669",
        "grid": "#e2e8f0",
        "text": "#1e293b",
        "label": "#64748b",
    }

    @staticmethod
    def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame,
                       target: str, time_col: str, model_name: str = "",
                       title: str = None, source: str = None,
                       save_path: str = "forecast.png", dpi: int = 150,
                       figsize: tuple = (14, 6)) -> str:
        """Render forecast as PNG chart."""
        fig, ax = plt.subplots(figsize=figsize)

        # History
        ax.plot(history[time_col], history[target], color="#2563eb",
                linewidth=1.2, label="Historical data", zorder=3)

        # Forecast
        ax.plot(forecast[time_col], forecast[target], color="#dc2626",
                linewidth=2, linestyle="--", label=f"Forecast ({model_name})", zorder=4)

        # Confidence interval
        lower_col = f"{target}_lower"
        upper_col = f"{target}_upper"
        if lower_col in forecast.columns and upper_col in forecast.columns:
            ax.fill_between(forecast[time_col], forecast[lower_col], forecast[upper_col],
                          alpha=0.12, color="#dc2626", label="95% confidence", zorder=2)

        ax.set_title(title or f"WavqWise Forecast: {target}", fontsize=14, fontweight="bold", color="#1e293b")
        ax.set_xlabel("Date", fontsize=11, color="#64748b")
        ax.set_ylabel(target, fontsize=11, color="#64748b")
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, color="#e2e8f0")
        ax.tick_params(colors="#64748b")

        # Data source attribution
        if source:
            ax.text(0.99, 0.01, f"Data: {source}", transform=ax.transAxes,
                   fontsize=8, color="#94a3b8", ha="right", va="bottom")

        ax.text(0.01, 0.01, "WavqWise | Sense. Forecast. Alert.",
               transform=ax.transAxes, fontsize=8, color="#94a3b8", ha="left", va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return save_path

    @staticmethod
    def dam_network_chart(dams: List[Dict], forecasts: Dict = None,
                          title: str = "Dam Network Monitoring",
                          source: str = None,
                          save_path: str = "dam_network.png", dpi: int = 150) -> str:
        """Render dam network overview as a single PNG image."""
        n_dams = len(dams)
        cols = min(3, n_dams)
        rows = (n_dams + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(title, fontsize=16, fontweight="bold", color="#1e293b", y=0.98)

        for idx, dam in enumerate(dams):
            r, c = idx // cols, idx % cols
            ax = axes[r][c]

            name = dam["name"]
            capacity = dam["capacity"]
            level = dam["level"]
            pct = (level / capacity) * 100

            # Status color
            if pct >= 90: color, status = "#dc2626", "CRITICAL"
            elif pct >= 75: color, status = "#f59e0b", "HIGH"
            elif pct >= 50: color, status = "#059669", "NORMAL"
            else: color, status = "#6366f1", "LOW"

            # Time series if available
            if "timeseries" in dam and dam["timeseries"] is not None:
                ts = dam["timeseries"]
                ax.plot(range(len(ts)), ts, color="#2563eb", linewidth=1, alpha=0.8)
                ax.axhline(capacity, color="#ef4444", linewidth=1, linestyle=":", alpha=0.7)
                ax.axhline(capacity * 0.9, color="#f59e0b", linewidth=0.5, linestyle=":", alpha=0.5)

                # Forecast overlay
                if forecasts and name in forecasts:
                    fc = forecasts[name]
                    fc_len = len(fc)
                    fc_x = range(len(ts), len(ts) + fc_len)
                    ax.plot(fc_x, fc, color="#dc2626", linewidth=2, linestyle="--")

            # Water level bar
            bar_y = ax.get_ylim()[0]
            ax.set_title(f"{name}\n{level:.0f}/{capacity:.0f} ft ({pct:.0f}%) [{status}]",
                        fontsize=10, fontweight="bold", color=color)
            ax.set_xlabel("Days", fontsize=8, color="#94a3b8")
            ax.grid(True, alpha=0.2)

        # Hide empty subplots
        for idx in range(n_dams, rows * cols):
            r, c = idx // cols, idx % cols
            axes[r][c].set_visible(False)

        if source:
            fig.text(0.99, 0.01, f"Data: {source}", fontsize=8, color="#94a3b8", ha="right")
        fig.text(0.01, 0.01, "WavqWise | Sense. Forecast. Alert.", fontsize=8, color="#94a3b8")

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return save_path

    @staticmethod
    def anomaly_chart(data: pd.DataFrame, target: str, time_col: str,
                      title: str = None, source: str = None,
                      save_path: str = "anomalies.png", dpi: int = 150) -> str:
        """Render anomaly detection results as PNG."""
        fig, ax = plt.subplots(figsize=(14, 5))

        normal = data[~data["is_anomaly"]]
        anomalies = data[data["is_anomaly"]]

        ax.plot(data[time_col], data[target], color="#2563eb", linewidth=0.8, alpha=0.6, label="Normal")

        severity_colors = {"low": "#fbbf24", "medium": "#f97316", "high": "#ef4444", "critical": "#7f1d1d"}
        if "severity" in anomalies.columns:
            for sev, clr in severity_colors.items():
                mask = anomalies["severity"] == sev
                if mask.any():
                    ax.scatter(anomalies[mask][time_col], anomalies[mask][target],
                             color=clr, s=50, zorder=5, label=f"Anomaly ({sev})")
        else:
            ax.scatter(anomalies[time_col], anomalies[target], color="#dc2626", s=50, zorder=5, label="Anomaly")

        ax.set_title(title or f"WavqWise Anomaly Detection: {target}", fontsize=14, fontweight="bold", color="#1e293b")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        if source:
            ax.text(0.99, 0.01, f"Data: {source}", transform=ax.transAxes, fontsize=8, color="#94a3b8", ha="right", va="bottom")
        ax.text(0.01, 0.01, "WavqWise", transform=ax.transAxes, fontsize=8, color="#94a3b8", ha="left", va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return save_path

    @staticmethod
    def comparison_chart(comparison: pd.DataFrame, title: str = "Model Comparison",
                         save_path: str = "comparison.png", dpi: int = 150) -> str:
        """Render model comparison as bar chart PNG."""
        fig, ax = plt.subplots(figsize=(10, 5))

        models = comparison["model"].values
        maes = comparison["MAE"].values
        colors = ["#059669" if i == 0 else "#94a3b8" for i in range(len(models))]

        bars = ax.barh(range(len(models)), maes, color=colors, height=0.6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=11)
        ax.set_xlabel("MAE (lower is better)", fontsize=11, color="#64748b")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#1e293b")
        ax.grid(True, alpha=0.3, axis="x")

        for bar, mae in zip(bars, maes):
            ax.text(bar.get_width() + max(maes) * 0.02, bar.get_y() + bar.get_height()/2,
                   f"{mae:.4f}", va="center", fontsize=10, color="#1e293b")

        ax.text(0.01, -0.08, "WavqWise | Best model highlighted in green",
               transform=ax.transAxes, fontsize=8, color="#94a3b8")

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        return save_path


class CLIPrinter:
    """CLI-friendly text output with formatted tables."""

    @staticmethod
    def header(text: str, width: int = 60):
        print(f"\n{'=' * width}")
        print(f"  {text}")
        print(f"{'=' * width}")

    @staticmethod
    def subheader(text: str):
        print(f"\n--- {text} ---")

    @staticmethod
    def table(headers: list, rows: list, col_widths: list = None):
        """Print formatted ASCII table."""
        if not col_widths:
            col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) + 2
                         for i, h in enumerate(headers)]

        # Header
        header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
        sep_line = "  ".join("-" * w for w in col_widths)
        print(f"  {header_line}")
        print(f"  {sep_line}")

        # Rows
        for row in rows:
            line = "  ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
            print(f"  {line}")

    @staticmethod
    def dam_status(name: str, level: float, capacity: float, forecast: float = None):
        """Print dam status with ASCII progress bar."""
        pct = (level / capacity) * 100
        bar_width = 30
        filled = int(pct / 100 * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)

        if pct >= 90: status = "CRITICAL"
        elif pct >= 75: status = "HIGH"
        elif pct >= 50: status = "NORMAL"
        else: status = "LOW"

        print(f"  {name}")
        print(f"    Level: {level:.1f} / {capacity:.1f} ft ({pct:.0f}%)")
        print(f"    [{bar}] [{status}]")
        if forecast is not None:
            fc_pct = (forecast / capacity) * 100
            print(f"    Forecast (30d): {forecast:.1f} ft ({fc_pct:.0f}%)")

    @staticmethod
    def forecast_summary(result, n_rows: int = 5):
        """Print forecast result summary."""
        print(f"  Model: {result.model_name}")
        print(f"  Horizon: {len(result.forecast)} steps")
        if result.metrics:
            for k, v in result.metrics.items():
                print(f"  {k}: {v:.4f}")
        print(f"\n  First {n_rows} predictions:")
        print(result.forecast.head(n_rows).to_string(index=False))

    @staticmethod
    def data_source(name: str, url: str, description: str = ""):
        """Print data source attribution."""
        print(f"  Source: {name}")
        print(f"  URL: {url}")
        if description:
            print(f"  Info: {description}")

    @staticmethod
    def anomaly_summary(result, top_n: int = 5):
        """Print anomaly detection summary."""
        total = len(result.data)
        n_anom = len(result.anomalies)
        print(f"  Total points: {total}")
        print(f"  Anomalies: {n_anom} ({n_anom/total*100:.1f}%)")
        print(f"  Method: {result.method}")
        if n_anom > 0:
            print(f"\n  Top {min(top_n, n_anom)} anomalies:")
            top = result.anomalies.nlargest(top_n, "anomaly_score")
            cols = [c for c in ["anomaly_score", "severity"] if c in top.columns]
            print(top[cols].to_string(index=False))


class DataExporter:
    """Export results to CSV/JSON for programmatic access."""

    @staticmethod
    def to_csv(data: pd.DataFrame, path: str, source: str = None):
        """Export DataFrame to CSV with source metadata."""
        data.to_csv(path, index=False)
        if source:
            # Write source info as comment in a companion file
            with open(path + ".source", "w") as f:
                f.write(f"source: {source}\n")
                f.write(f"exported: {pd.Timestamp.now()}\n")
                f.write(f"rows: {len(data)}\n")
        return path

    @staticmethod
    def to_json(data: pd.DataFrame, path: str, source: str = None):
        """Export DataFrame to JSON with metadata."""
        output = {
            "data": data.to_dict(orient="records"),
            "metadata": {
                "source": source,
                "rows": len(data),
                "columns": list(data.columns),
                "exported": str(pd.Timestamp.now()),
                "generator": "WavqWise v0.1.5",
            },
        }
        import json
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        return path
