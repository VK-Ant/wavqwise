"""Weather data loader via Open-Meteo API. Free. No API key. Real data."""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

CITIES = {
    "new york":(40.7128,-74.006),"london":(51.5074,-0.1278),"tokyo":(35.6762,139.6503),
    "chennai":(13.0827,80.2707),"madurai":(9.9252,78.1198),"mumbai":(19.076,72.8777),
    "bangalore":(12.9716,77.5946),"delhi":(28.6139,77.209),"san francisco":(37.7749,-122.4194),
    "los angeles":(34.0522,-118.2437),"chicago":(41.8781,-87.6298),"paris":(48.8566,2.3522),
    "berlin":(52.52,13.405),"sydney":(-33.8688,151.2093),"singapore":(1.3521,103.8198),
    "dubai":(25.2048,55.2708),"seoul":(37.5665,126.978),"beijing":(39.9042,116.4074),
    "toronto":(43.6532,-79.3832),"sao paulo":(-23.5505,-46.6333),
}

class WeatherLoader:
    BASE_URL = "https://api.open-meteo.com/v1"

    def _get_coords(self, city):
        key = city.lower().strip()
        if key in CITIES: return CITIES[key]
        raise ValueError(f"Unknown city: '{city}'. Available: {sorted(CITIES.keys())}")

    def load_historical(self, city, days=365, variables=None, frequency="daily"):
        lat, lon = self._get_coords(city)
        end = datetime.now() - timedelta(days=5)
        start = end - timedelta(days=days)
        return self.load_coords(lat, lon, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), variables, frequency, city)

    def load_forecast(self, city, days=16, variables=None, frequency="daily"):
        lat, lon = self._get_coords(city)
        return self._fetch_forecast(lat, lon, days, variables, frequency, city)

    def load_city(self, city, start, end, variables=None, frequency="daily"):
        lat, lon = self._get_coords(city)
        return self.load_coords(lat, lon, start, end, variables, frequency, city)

    def load_coords(self, lat, lon, start, end, variables=None, frequency="daily", city_name=""):
        try: import requests
        except ImportError: raise ImportError("pip install requests")
        if variables is None:
            variables = ["temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                         "precipitation_sum","windspeed_10m_max","relative_humidity_2m_mean","weathercode"]
        params = {"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,
                  frequency:",".join(variables),"timezone":"auto"}
        resp = requests.get(f"{self.BASE_URL}/archive", params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data.get("daily", data.get("hourly", {})))
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"]); df = df.drop(columns=["time"])
        if city_name: df["city"] = city_name
        return df

    def _fetch_forecast(self, lat, lon, days, variables, frequency, city):
        try: import requests
        except ImportError: raise ImportError("pip install requests")
        if variables is None:
            variables = ["temperature_2m_max","temperature_2m_min","precipitation_sum","windspeed_10m_max","weathercode"]
        params = {"latitude":lat,"longitude":lon,"forecast_days":min(days,16),
                  "daily":",".join(variables),"timezone":"auto"}
        resp = requests.get(f"{self.BASE_URL}/forecast", params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data.get("daily",{}))
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"]); df = df.drop(columns=["time"])
        df["city"] = city
        return df

    def load_multi_city(self, cities, days=365, frequency="daily"):
        frames = []
        for city in cities:
            try: frames.append(self.load_historical(city, days=days, frequency=frequency))
            except Exception as e: print(f"  Warning: {city}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def available_cities(): return sorted(CITIES.keys())
