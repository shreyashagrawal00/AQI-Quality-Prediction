"""
state_rankings.py
-----------------
Maps cities to Indian states, aggregates AQI/HCHO/fire metrics by state,
and generates ranked tables for the State Rankings dashboard page.

Data sources:
  - City → State mapping from data/stations.csv (StationId, City, State)
  - Supplemented by a built-in lookup table for cities not in stations.csv
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ── fallback city→state lookup (for cities not in stations.csv) ────────────
_CITY_STATE: dict[str, str] = {
    "Ahmedabad": "Gujarat", "Aizawl": "Mizoram", "Amaravati": "Maharashtra",
    "Amritsar": "Punjab", "Bengaluru": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Brajrajnagar": "Odisha", "Chandigarh": "Chandigarh", "Chennai": "Tamil Nadu",
    "Coimbatore": "Tamil Nadu", "Delhi": "Delhi", "Ernakulam": "Kerala",
    "Gurugram": "Haryana", "Guwahati": "Assam", "Hyderabad": "Telangana",
    "Jaipur": "Rajasthan", "Jorapokhar": "Jharkhand", "Kochi": "Kerala",
    "Kolkata": "West Bengal", "Lucknow": "Uttar Pradesh", "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra", "Nashik": "Maharashtra", "Patna": "Bihar",
    "Pune": "Maharashtra", "Rajgir": "Bihar", "Shillong": "Meghalaya",
    "Talcher": "Odisha", "Thiruvananthapuram": "Kerala", "Visakhapatnam": "Andhra Pradesh",
    "Agra": "Uttar Pradesh", "Aurangabad": "Maharashtra", "Varanasi": "Uttar Pradesh",
    "Surat": "Gujarat", "Vadodara": "Gujarat", "Indore": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh", "Prayagraj": "Uttar Pradesh",
    "Faridabad": "Haryana", "Meerut": "Uttar Pradesh", "Rajkot": "Gujarat",
    "Kalyan-Dombivli": "Maharashtra", "Vasai-Virar": "Maharashtra",
    "Ludhiana": "Punjab", "Agartala": "Tripura", "Dehradun": "Uttarakhand",
    "Noida": "Uttar Pradesh", "Ghaziabad": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Jabalpur": "Madhya Pradesh", "Madurai": "Tamil Nadu", "Raipur": "Chhattisgarh",
    "Kota": "Rajasthan", "Ranchi": "Jharkhand", "Jodhpur": "Rajasthan",
    "Puducherry": "Puducherry", "Mysuru": "Karnataka",
}


def load_state_map(data_dir: str) -> dict[str, str]:
    """
    Returns city → state dict, merging stations.csv data with the fallback table.
    """
    mapping = dict(_CITY_STATE)
    stations_path = os.path.join(data_dir, "stations.csv")
    if os.path.exists(stations_path):
        try:
            st = pd.read_csv(stations_path, encoding="utf-8-sig")
            st.columns = [c.strip() for c in st.columns]
            if "City" in st.columns and "State" in st.columns:
                for _, row in st.dropna(subset=["City", "State"]).iterrows():
                    mapping[str(row["City"]).strip()] = str(row["State"]).strip()
        except Exception:
            pass
    return mapping


def add_state_column(df: pd.DataFrame, city_state_map: dict[str, str]) -> pd.DataFrame:
    """Add a 'State' column to any DataFrame that has a 'City' column."""
    df = df.copy()
    if "City" in df.columns:
        df["State"] = df["City"].map(city_state_map).fillna("Unknown")
    return df


def rank_states_by_aqi(
    df: pd.DataFrame,
    city_state_map: dict[str, str],
    metric: str = "mean",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Aggregate AQI by state and return ranked DataFrame.

    Returns columns: State, AQI_mean, AQI_median, AQI_max, n_records
    Sorted worst → best.
    """
    df = add_state_column(df, city_state_map)
    df = df[df["State"] != "Unknown"].dropna(subset=["AQI"])
    grp = df.groupby("State")["AQI"].agg(
        AQI_mean="mean",
        AQI_median="median",
        AQI_max="max",
        n_records="count",
    ).reset_index()
    grp = grp[grp["n_records"] >= 30]   # exclude states with too few records
    return grp.sort_values("AQI_mean", ascending=False).head(top_n).reset_index(drop=True)


def rank_states_by_hcho(
    hcho_grid_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Aggregate HCHO from a grid DataFrame by approximate state bounding boxes.
    hcho_grid_df must have: lat, lon, hcho columns.

    Uses a simplified lat/lon state lookup for the grid (not exact borders).
    Returns: State, mean_hcho, max_hcho, n_points — sorted worst → best.
    """
    df = hcho_grid_df.copy()
    df["State"] = df.apply(_latlon_to_state, axis=1)
    df = df[df["State"] != "Unknown"]
    if len(df) == 0:
        return pd.DataFrame(columns=["State", "mean_hcho", "max_hcho", "n_points"])
    grp = df.groupby("State")["hcho"].agg(
        mean_hcho="mean",
        max_hcho="max",
        n_points="count",
    ).reset_index()
    return grp.sort_values("mean_hcho", ascending=False).head(top_n).reset_index(drop=True)


def rank_states_by_fire(
    fire_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Count fire points and sum FRP by approximate state.
    fire_df must have: latitude, longitude, frp columns.
    """
    if len(fire_df) == 0:
        return pd.DataFrame(columns=["State", "fire_count", "total_frp"])
    df = fire_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    df["State"] = df.apply(_latlon_to_state, axis=1)
    df = df[df["State"] != "Unknown"]
    grp = df.groupby("State").agg(
        fire_count=("lat", "count"),
        total_frp=("frp", "sum"),
    ).reset_index()
    return grp.sort_values("fire_count", ascending=False).head(top_n).reset_index(drop=True)


def most_improved_states(
    df: pd.DataFrame,
    city_state_map: dict[str, str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Computes year-over-year AQI improvement (most negative delta = most improved).
    Requires a 'Year' column.
    """
    df = add_state_column(df, city_state_map)
    df = df[df["State"] != "Unknown"].dropna(subset=["AQI", "Year"])
    yearly = df.groupby(["State", "Year"])["AQI"].mean().reset_index()
    years = sorted(yearly["Year"].unique())
    if len(years) < 2:
        return pd.DataFrame(columns=["State", "AQI_latest", "AQI_prev", "delta", "pct_change"])

    latest, prev = years[-1], years[-2]
    lat_df = yearly[yearly["Year"] == latest].set_index("State")["AQI"].rename("AQI_latest")
    prv_df = yearly[yearly["Year"] == prev].set_index("State")["AQI"].rename("AQI_prev")
    merged = pd.concat([lat_df, prv_df], axis=1).dropna()
    merged["delta"] = merged["AQI_latest"] - merged["AQI_prev"]
    merged["pct_change"] = (merged["delta"] / merged["AQI_prev"] * 100).round(1)
    return merged.sort_values("delta").head(top_n).reset_index()


# ── simplified lat/lon → state lookup ─────────────────────────────────────
_STATE_BBOXES = [
    # (state, lat_min, lat_max, lon_min, lon_max)
    ("Jammu & Kashmir",    32.5, 36.5, 73.5, 80.5),
    ("Himachal Pradesh",   30.4, 33.2, 75.5, 79.0),
    ("Punjab",             29.5, 32.5, 73.8, 76.9),
    ("Haryana",            27.7, 30.9, 74.5, 77.6),
    ("Uttarakhand",        28.7, 31.5, 77.5, 81.1),
    ("Delhi",              28.4, 28.9, 76.8, 77.4),
    ("Uttar Pradesh",      23.9, 30.4, 77.1, 84.7),
    ("Bihar",              24.3, 27.5, 83.3, 88.3),
    ("Rajasthan",          23.0, 30.2, 69.5, 78.2),
    ("Gujarat",            20.1, 24.7, 68.2, 74.5),
    ("Madhya Pradesh",     21.1, 26.9, 74.0, 82.8),
    ("Chhattisgarh",       17.8, 24.1, 80.2, 84.4),
    ("Jharkhand",          21.9, 25.4, 83.3, 87.5),
    ("West Bengal",        21.5, 27.3, 85.9, 89.9),
    ("Maharashtra",        15.6, 22.1, 72.6, 80.9),
    ("Odisha",             17.8, 22.6, 81.3, 87.5),
    ("Telangana",          15.9, 19.9, 77.2, 81.3),
    ("Andhra Pradesh",     12.7, 19.9, 76.8, 84.8),
    ("Karnataka",          11.6, 18.5, 74.1, 78.6),
    ("Tamil Nadu",          8.1, 13.6, 76.3, 80.4),
    ("Kerala",              8.3, 12.8, 74.9, 77.4),
    ("Assam",              24.1, 28.2, 89.7, 96.0),
    ("Arunachal Pradesh",  26.7, 29.5, 91.6, 97.4),
    ("Meghalaya",          25.0, 26.1, 89.8, 92.8),
    ("Manipur",            23.8, 25.7, 93.0, 94.8),
    ("Nagaland",           25.2, 27.0, 93.3, 95.3),
    ("Mizoram",            21.9, 24.5, 92.3, 93.4),
    ("Tripura",            22.9, 24.5, 91.2, 92.3),
    ("Goa",                14.9, 15.8, 73.7, 74.3),
    ("Sikkim",             27.1, 28.1, 88.0, 88.9),
]


def _latlon_to_state(row) -> str:
    lat, lon = row.lat, row.lon
    for (state, lat_min, lat_max, lon_min, lon_max) in _STATE_BBOXES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return state
    return "Unknown"
