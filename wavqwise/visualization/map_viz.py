"""
Map Visualization - OpenStreetMap via Folium
=============================================
Plot weather forecasts, anomalies, sensor data, water levels
on interactive OpenStreetMap.

Free. Open source. No API key.

Usage:
    from wavqwise.visualization.map_viz import WeatherMap

    wmap = WeatherMap()
    wmap.add_weather_station("Chennai", 13.08, 80.27, temp=32.5, humidity=78)
    wmap.add_anomaly_marker("Dam A", 10.5, 78.8, level=85, severity="critical")
    wmap.add_forecast_overlay(city_forecasts)
    wmap.save("weather_map.html")
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class WeatherMap:
    """Interactive map visualization using OpenStreetMap (via Folium)."""

    # Severity color mapping
    SEVERITY_COLORS = {
        "low": "#22c55e",       # green
        "medium": "#f59e0b",    # amber
        "high": "#ef4444",      # red
        "critical": "#7f1d1d",  # dark red
        "normal": "#3b82f6",    # blue
    }

    def __init__(self, center: Tuple[float, float] = (20.0, 78.0), zoom: int = 5):
        try:
            import folium
            from folium.plugins import HeatMap, MarkerCluster
        except ImportError:
            raise ImportError("pip install folium")

        self._folium = folium
        self._map = folium.Map(
            location=center, zoom_start=zoom,
            tiles="OpenStreetMap",
        )
        self._markers = []

    def add_weather_station(self, name: str, lat: float, lon: float,
                            temp: float = None, humidity: float = None,
                            wind: float = None, precip: float = None,
                            status: str = "normal"):
        """Add a weather station marker with popup data."""
        import folium

        popup_lines = [f"<b>{name}</b><br>"]
        if temp is not None:
            popup_lines.append(f"Temp: {temp:.1f} C<br>")
        if humidity is not None:
            popup_lines.append(f"Humidity: {humidity:.0f}%<br>")
        if wind is not None:
            popup_lines.append(f"Wind: {wind:.1f} km/h<br>")
        if precip is not None:
            popup_lines.append(f"Precip: {precip:.1f} mm<br>")
        popup_lines.append(f"Status: <b>{status.upper()}</b>")

        color = self.SEVERITY_COLORS.get(status, "#3b82f6")
        icon_map = {"normal": "cloud", "low": "info-sign", "medium": "warning-sign",
                     "high": "fire", "critical": "exclamation-sign"}

        folium.Marker(
            [lat, lon],
            popup=folium.Popup("".join(popup_lines), max_width=250),
            tooltip=f"{name}: {temp:.1f}C" if temp else name,
            icon=folium.Icon(color="blue" if status == "normal" else "red",
                            icon=icon_map.get(status, "cloud")),
        ).add_to(self._map)

    def add_dam_marker(self, name: str, lat: float, lon: float,
                       water_level: float, capacity: float,
                       status: str = "normal", forecast_level: float = None):
        """Add dam/reservoir marker with water level info."""
        import folium

        pct = (water_level / capacity * 100) if capacity > 0 else 0
        if pct > 90:
            status = "critical"
        elif pct > 75:
            status = "high"
        elif pct > 50:
            status = "medium"

        bar_color = self.SEVERITY_COLORS.get(status, "#3b82f6")

        popup_html = f"""
        <div style="font-family:sans-serif;min-width:200px">
            <b style="font-size:14px">{name}</b><br><br>
            <b>Water Level:</b> {water_level:.1f} ft<br>
            <b>Capacity:</b> {capacity:.1f} ft<br>
            <b>Fill:</b> {pct:.1f}%<br>
            <div style="background:#e5e7eb;border-radius:4px;height:16px;margin:6px 0">
                <div style="background:{bar_color};height:16px;border-radius:4px;width:{min(pct, 100):.0f}%;
                     text-align:center;color:white;font-size:11px;line-height:16px">{pct:.0f}%</div>
            </div>
            <b>Status:</b> <span style="color:{bar_color}">{status.upper()}</span><br>
        """
        if forecast_level is not None:
            fc_pct = (forecast_level / capacity * 100)
            popup_html += f"<b>Forecast (7d):</b> {forecast_level:.1f} ft ({fc_pct:.0f}%)<br>"
        popup_html += "</div>"

        icon_color = "green" if pct < 50 else "orange" if pct < 75 else "red"
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{name}: {pct:.0f}% full",
            icon=folium.Icon(color=icon_color, icon="tint"),
        ).add_to(self._map)

    def add_anomaly_marker(self, name: str, lat: float, lon: float,
                           value: float, variable: str = "value",
                           severity: str = "medium", score: float = 0):
        """Add anomaly alert marker."""
        import folium

        color = self.SEVERITY_COLORS.get(severity, "#f59e0b")
        popup = (f"<b>ANOMALY: {name}</b><br>"
                 f"{variable}: {value:.1f}<br>"
                 f"Severity: <b style='color:{color}'>{severity.upper()}</b><br>"
                 f"Score: {score:.2f}")

        folium.CircleMarker(
            [lat, lon], radius=12 + score * 3,
            color=color, fill=True, fill_color=color, fill_opacity=0.6,
            popup=folium.Popup(popup, max_width=250),
            tooltip=f"ALERT: {name}",
        ).add_to(self._map)

    def add_heatmap(self, data: pd.DataFrame, lat_col: str = "latitude",
                    lon_col: str = "longitude", value_col: str = "value",
                    radius: int = 15):
        """Add heatmap layer from DataFrame."""
        from folium.plugins import HeatMap

        heat_data = data[[lat_col, lon_col, value_col]].dropna().values.tolist()
        HeatMap(heat_data, radius=radius, blur=10, max_zoom=8).add_to(self._map)

    def add_forecast_line(self, locations: list, values: list,
                          label: str = "Forecast", color: str = "#dc2626"):
        """Draw forecast trend line between locations."""
        import folium
        folium.PolyLine(locations, weight=3, color=color,
                        tooltip=label, opacity=0.8).add_to(self._map)

    def add_multi_city_weather(self, weather_data: pd.DataFrame,
                               temp_col: str = "temperature_2m_mean"):
        """Add markers for multiple cities from weather DataFrame."""
        for _, row in weather_data.iterrows():
            if "latitude" in row and "longitude" in row:
                temp = row.get(temp_col, None)
                name = row.get("city", f"({row['latitude']:.2f}, {row['longitude']:.2f})")
                status = "normal"
                if temp and temp > 40:
                    status = "high"
                elif temp and temp > 35:
                    status = "medium"
                self.add_weather_station(
                    name, row["latitude"], row["longitude"],
                    temp=temp,
                    humidity=row.get("relative_humidity_2m_mean"),
                    wind=row.get("windspeed_10m_max"),
                    precip=row.get("precipitation_sum"),
                    status=status,
                )

    def save(self, path: str = "wavqwise_map.html"):
        """Save interactive map to HTML file."""
        self._map.save(path)
        return path

    def _repr_html_(self):
        """Render in Jupyter/Colab."""
        return self._map._repr_html_()
