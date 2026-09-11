"""
Real Data Sources - Free, Open, No API Key
============================================
Every data source here is free and open. User always knows where data comes from.

Sources:
  - Open-Meteo: Weather data (api.open-meteo.com) - No key needed
  - India WRIS: Dam water levels (india-wris.nrsc.gov.in)
  - USGS: US water data (waterservices.usgs.gov)
  - Global Dam Watch: Worldwide reservoirs (globaldamwatch.org)
  - CWC India: Central Water Commission (cwc.gov.in)
"""

DATA_SOURCES = {
    "open_meteo": {
        "name": "Open-Meteo",
        "url": "https://api.open-meteo.com",
        "type": "Weather",
        "key_required": False,
        "coverage": "Global",
        "description": "Free weather API. Historical + forecast. 80+ years of data.",
    },
    "india_wris": {
        "name": "India-WRIS (Water Resources Information System)",
        "url": "https://india-wris.nrsc.gov.in",
        "type": "Dam water levels",
        "key_required": False,
        "coverage": "India (5000+ dams)",
        "description": "Real-time dam water levels, reservoir storage, river discharge.",
    },
    "cwc_india": {
        "name": "Central Water Commission (CWC)",
        "url": "https://cwc.gov.in",
        "type": "Dam water levels",
        "key_required": False,
        "coverage": "India (91 major reservoirs)",
        "description": "Weekly reservoir storage bulletin. Free PDF/data.",
    },
    "usgs": {
        "name": "USGS National Water Information System",
        "url": "https://waterservices.usgs.gov",
        "type": "Water data",
        "key_required": False,
        "coverage": "USA (1.5M+ sites)",
        "description": "Real-time streamflow, water levels, quality. REST API.",
    },
    "global_dam_watch": {
        "name": "Global Dam Watch",
        "url": "https://www.globaldamwatch.org",
        "type": "Reservoir monitoring",
        "key_required": False,
        "coverage": "Global (7000+ reservoirs)",
        "description": "Satellite-based reservoir monitoring. Water surface area.",
    },
    "copernicus": {
        "name": "Copernicus Climate Data Store",
        "url": "https://cds.climate.copernicus.eu",
        "type": "Climate reanalysis",
        "key_required": True,
        "coverage": "Global",
        "description": "ERA5 reanalysis data. Free account required.",
    },
    "noaa": {
        "name": "NOAA Climate Data Online",
        "url": "https://www.ncdc.noaa.gov/cdo-web/api/v2",
        "type": "Climate data",
        "key_required": True,
        "coverage": "Global",
        "description": "Historical weather observations. Free API key.",
    },
}


def list_sources():
    """Print all available data sources."""
    print("\nWavqWise Real Data Sources (Free & Open)")
    print("=" * 60)
    for key, src in DATA_SOURCES.items():
        key_str = "No key needed" if not src["key_required"] else "Free key required"
        print(f"\n  {src['name']}")
        print(f"    URL: {src['url']}")
        print(f"    Type: {src['type']} | Coverage: {src['coverage']}")
        print(f"    Auth: {key_str}")
        print(f"    {src['description']}")


def get_source_info(source_key: str) -> dict:
    """Get source metadata for attribution."""
    return DATA_SOURCES.get(source_key, {"name": "Unknown", "url": ""})
