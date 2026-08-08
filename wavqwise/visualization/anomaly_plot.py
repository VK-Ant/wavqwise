"""Anomaly visualization."""
import matplotlib.pyplot as plt

class AnomalyPlot:
    def __init__(self, engine="matplotlib"):
        self.engine = engine

    def plot(self, data, target, time_col, show_severity=False, **kwargs):
        fig, ax = plt.subplots(figsize=(12, 5))
        normal = data[~data["is_anomaly"]]
        anomalies = data[data["is_anomaly"]]
        ax.plot(data[time_col], data[target], color="#2563eb", linewidth=1, alpha=0.7, label="Normal")
        severity_colors = {"low": "#fbbf24", "medium": "#f97316", "high": "#ef4444", "critical": "#7f1d1d"}
        if show_severity and "severity" in anomalies.columns:
            for sev, color in severity_colors.items():
                mask = anomalies["severity"] == sev
                if mask.any():
                    ax.scatter(anomalies[mask][time_col], anomalies[mask][target], color=color, s=40, label=f"Anomaly ({sev})", zorder=5)
        else:
            ax.scatter(anomalies[time_col], anomalies[target], color="#dc2626", s=40, label="Anomaly", zorder=5)
        ax.set_title("WavqWise Anomaly Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig
