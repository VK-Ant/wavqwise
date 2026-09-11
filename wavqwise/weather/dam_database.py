"""
World Dam Database with Regional Filtering
============================================
Filter by any country, state, river, or capacity range.
Works worldwide. User picks their region.

Usage:
    from wavqwise.weather.dam_database import DamDB

    # Get all dams
    DamDB.list_countries()
    DamDB.list_states("India")

    # Filter
    tn_dams = DamDB.filter(country="India", state="Tamil Nadu")
    us_dams = DamDB.filter(country="USA", state="California")
    big_dams = DamDB.filter(min_capacity=500)
    nile_dams = DamDB.filter(river="Nile")
"""

from typing import List, Optional, Dict
import pandas as pd


# Real coordinates, real rivers, real capacities
_DAM_DATA = [
    # === INDIA ===
    # Tamil Nadu
    {"name":"Mettur Dam","country":"India","state":"Tamil Nadu","river":"Cauvery","lat":11.7939,"lon":77.8017,"capacity_ft":120},
    {"name":"Vaigai Dam","country":"India","state":"Tamil Nadu","river":"Vaigai","lat":10.0112,"lon":77.5494,"capacity_ft":71},
    {"name":"Bhavanisagar Dam","country":"India","state":"Tamil Nadu","river":"Bhavani","lat":11.445,"lon":77.0803,"capacity_ft":105},
    {"name":"Amaravathi Dam","country":"India","state":"Tamil Nadu","river":"Amaravathi","lat":10.4312,"lon":77.2831,"capacity_ft":90},
    {"name":"Sathanur Dam","country":"India","state":"Tamil Nadu","river":"Thenpennai","lat":12.2097,"lon":78.8983,"capacity_ft":119},
    {"name":"Krishnagiri Dam","country":"India","state":"Tamil Nadu","river":"Ponnaiyar","lat":12.5083,"lon":78.2206,"capacity_ft":53},
    {"name":"Papanasam Dam","country":"India","state":"Tamil Nadu","river":"Tamiraparani","lat":8.6951,"lon":77.3718,"capacity_ft":143},
    {"name":"Sholayar Dam","country":"India","state":"Tamil Nadu","river":"Sholayar","lat":10.3167,"lon":76.8833,"capacity_ft":165},
    # Kerala
    {"name":"Mullaperiyar Dam","country":"India","state":"Kerala","river":"Periyar","lat":9.5322,"lon":77.1387,"capacity_ft":142},
    {"name":"Idukki Dam","country":"India","state":"Kerala","river":"Periyar","lat":9.8456,"lon":76.9776,"capacity_ft":2403},
    {"name":"Banasura Sagar","country":"India","state":"Kerala","river":"Karamanathodu","lat":11.6717,"lon":76.0403,"capacity_ft":775},
    # Karnataka
    {"name":"KRS Dam","country":"India","state":"Karnataka","river":"Cauvery","lat":12.4248,"lon":76.5718,"capacity_ft":124},
    {"name":"Tungabhadra Dam","country":"India","state":"Karnataka","river":"Tungabhadra","lat":15.2674,"lon":76.3372,"capacity_ft":1633},
    {"name":"Linganamakki Dam","country":"India","state":"Karnataka","river":"Sharavathi","lat":14.1840,"lon":75.0385,"capacity_ft":1819},
    # Andhra / Telangana
    {"name":"Nagarjuna Sagar","country":"India","state":"Telangana","river":"Krishna","lat":16.5755,"lon":79.3125,"capacity_ft":590},
    {"name":"Srisailam Dam","country":"India","state":"Andhra Pradesh","river":"Krishna","lat":15.8498,"lon":78.8689,"capacity_ft":885},
    # Maharashtra
    {"name":"Koyna Dam","country":"India","state":"Maharashtra","river":"Koyna","lat":17.4000,"lon":73.7500,"capacity_ft":2167},
    {"name":"Jayakwadi Dam","country":"India","state":"Maharashtra","river":"Godavari","lat":19.5167,"lon":75.3833,"capacity_ft":1541},
    # North India
    {"name":"Bhakra Dam","country":"India","state":"Himachal Pradesh","river":"Sutlej","lat":31.4109,"lon":76.4327,"capacity_ft":1685},
    {"name":"Tehri Dam","country":"India","state":"Uttarakhand","river":"Bhagirathi","lat":30.3778,"lon":78.4806,"capacity_ft":830},
    {"name":"Hirakud Dam","country":"India","state":"Odisha","river":"Mahanadi","lat":21.5181,"lon":83.8719,"capacity_ft":630},
    {"name":"Sardar Sarovar","country":"India","state":"Gujarat","river":"Narmada","lat":21.8300,"lon":73.7500,"capacity_ft":456},
    # === USA ===
    {"name":"Hoover Dam","country":"USA","state":"Nevada","river":"Colorado","lat":36.0161,"lon":-114.7377,"capacity_ft":1229},
    {"name":"Glen Canyon Dam","country":"USA","state":"Arizona","river":"Colorado","lat":36.9375,"lon":-111.4858,"capacity_ft":3700},
    {"name":"Grand Coulee Dam","country":"USA","state":"Washington","river":"Columbia","lat":47.9654,"lon":-118.9818,"capacity_ft":1290},
    {"name":"Oroville Dam","country":"USA","state":"California","river":"Feather","lat":39.5386,"lon":-121.4858,"capacity_ft":900},
    {"name":"Shasta Dam","country":"USA","state":"California","river":"Sacramento","lat":40.7208,"lon":-122.4189,"capacity_ft":1067},
    {"name":"Fort Peck Dam","country":"USA","state":"Montana","river":"Missouri","lat":48.0036,"lon":-106.4181,"capacity_ft":2250},
    {"name":"Garrison Dam","country":"USA","state":"North Dakota","river":"Missouri","lat":47.5025,"lon":-101.4275,"capacity_ft":1854},
    {"name":"Dworshak Dam","country":"USA","state":"Idaho","river":"N. Fork Clearwater","lat":46.5158,"lon":-116.2967,"capacity_ft":1600},
    # === CHINA ===
    {"name":"Three Gorges Dam","country":"China","state":"Hubei","river":"Yangtze","lat":30.8236,"lon":111.0035,"capacity_ft":175},
    {"name":"Xiluodu Dam","country":"China","state":"Yunnan","river":"Jinsha","lat":28.2525,"lon":103.6469,"capacity_ft":600},
    {"name":"Xiangjiaba Dam","country":"China","state":"Yunnan","river":"Jinsha","lat":28.6311,"lon":104.3931,"capacity_ft":380},
    {"name":"Longtan Dam","country":"China","state":"Guangxi","river":"Hongshui","lat":24.8458,"lon":107.0458,"capacity_ft":375},
    # === BRAZIL ===
    {"name":"Itaipu Dam","country":"Brazil","state":"Parana","river":"Parana","lat":-25.4084,"lon":-54.5894,"capacity_ft":220},
    {"name":"Tucurui Dam","country":"Brazil","state":"Para","river":"Tocantins","lat":-3.8319,"lon":-49.7217,"capacity_ft":72},
    {"name":"Belo Monte Dam","country":"Brazil","state":"Para","river":"Xingu","lat":-3.1167,"lon":-51.7833,"capacity_ft":97},
    # === AFRICA ===
    {"name":"Aswan High Dam","country":"Egypt","state":"Aswan","river":"Nile","lat":23.9708,"lon":32.8781,"capacity_ft":183},
    {"name":"Kariba Dam","country":"Zimbabwe","state":"Mashonaland","river":"Zambezi","lat":-16.5214,"lon":28.7614,"capacity_ft":488},
    {"name":"Akosombo Dam","country":"Ghana","state":"Eastern","river":"Volta","lat":6.3008,"lon":-0.0539,"capacity_ft":278},
    {"name":"Gibe III Dam","country":"Ethiopia","state":"SNNPR","river":"Omo","lat":6.8333,"lon":36.7167,"capacity_ft":243},
    # === EUROPE ===
    {"name":"Grande Dixence","country":"Switzerland","state":"Valais","river":"Dixence","lat":46.0808,"lon":7.4053,"capacity_ft":2365},
    {"name":"Vajont Dam","country":"Italy","state":"Veneto","river":"Vajont","lat":46.2667,"lon":12.3333,"capacity_ft":262},
    # === TURKEY ===
    {"name":"Ataturk Dam","country":"Turkey","state":"Sanliurfa","river":"Euphrates","lat":37.475,"lon":38.325,"capacity_ft":542},
    {"name":"Ilisu Dam","country":"Turkey","state":"Mardin","river":"Tigris","lat":37.4500,"lon":41.7667,"capacity_ft":525},
    # === AUSTRALIA ===
    {"name":"Gordon Dam","country":"Australia","state":"Tasmania","river":"Gordon","lat":-42.7458,"lon":146.0542,"capacity_ft":140},
    {"name":"Warragamba Dam","country":"Australia","state":"NSW","river":"Warragamba","lat":-33.8819,"lon":150.6008,"capacity_ft":142},
    # === JAPAN ===
    {"name":"Kurobe Dam","country":"Japan","state":"Toyama","river":"Kurobe","lat":36.5667,"lon":137.6625,"capacity_ft":186},
    {"name":"Miyagase Dam","country":"Japan","state":"Kanagawa","river":"Nakatsu","lat":35.5167,"lon":139.2333,"capacity_ft":156},
]


class DamDB:
    """World dam database with regional filtering."""

    _data = pd.DataFrame(_DAM_DATA)

    @classmethod
    def all(cls) -> pd.DataFrame:
        return cls._data.copy()

    @classmethod
    def filter(cls, country: str = None, state: str = None,
               river: str = None, min_capacity: float = None,
               max_capacity: float = None) -> pd.DataFrame:
        df = cls._data.copy()
        if country:
            df = df[df["country"].str.lower() == country.lower()]
        if state:
            df = df[df["state"].str.lower() == state.lower()]
        if river:
            df = df[df["river"].str.lower().str.contains(river.lower())]
        if min_capacity:
            df = df[df["capacity_ft"] >= min_capacity]
        if max_capacity:
            df = df[df["capacity_ft"] <= max_capacity]
        return df.reset_index(drop=True)

    @classmethod
    def list_countries(cls) -> list:
        return sorted(cls._data["country"].unique().tolist())

    @classmethod
    def list_states(cls, country: str) -> list:
        df = cls._data[cls._data["country"].str.lower() == country.lower()]
        return sorted(df["state"].unique().tolist())

    @classmethod
    def list_rivers(cls, country: str = None) -> list:
        df = cls._data
        if country:
            df = df[df["country"].str.lower() == country.lower()]
        return sorted(df["river"].unique().tolist())

    @classmethod
    def search(cls, query: str) -> pd.DataFrame:
        q = query.lower()
        mask = (
            cls._data["name"].str.lower().str.contains(q) |
            cls._data["river"].str.lower().str.contains(q) |
            cls._data["state"].str.lower().str.contains(q) |
            cls._data["country"].str.lower().str.contains(q)
        )
        return cls._data[mask].reset_index(drop=True)

    @classmethod
    def stats(cls) -> dict:
        return {
            "total_dams": len(cls._data),
            "countries": len(cls._data["country"].unique()),
            "states": len(cls._data["state"].unique()),
            "rivers": len(cls._data["river"].unique()),
        }

    @classmethod
    def summary(cls):
        s = cls.stats()
        print(f"\nWavqWise Dam Database")
        print(f"  Total dams: {s['total_dams']}")
        print(f"  Countries: {s['countries']}")
        print(f"  States/Provinces: {s['states']}")
        print(f"  Rivers: {s['rivers']}")
        print(f"\n  Countries: {', '.join(cls.list_countries())}")
        for country in cls.list_countries():
            states = cls.list_states(country)
            n = len(cls.filter(country=country))
            print(f"    {country} ({n} dams): {', '.join(states)}")
