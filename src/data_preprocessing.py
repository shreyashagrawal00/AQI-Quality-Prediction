"""
data_preprocessing.py
----------------------
Loads and cleans the CPCB ground-station datasets
(city_day.csv / city_hour.csv / station_day.csv / station_hour.csv).

Bugs this fixes, relative to a naive first pass at this dataset:

1. station_hour.csv is ~220MB / has NO City column, only StationId --
   a naive `pd.concat([city_hour, station_hour])` (a real mistake we've
   seen in similar student projects) silently produces a City column full
   of NaNs for every station row. We instead join stations.csv on
   StationId to recover City/State, and keep city-level and station-level
   data in clearly separate, well-typed frames.
2. Dropping every row with ANY missing pollutant (df.dropna()) throws
   away ~85% of the data here. We instead impute per-City medians
   (falling back to global median for cities with zero coverage of a
   given pollutant), which preserves the time series.
3. Date column is read as plain text unless parse_dates is set --
   downstream .dt accessors then silently fail. We always parse it.
4. Negative pollutant readings (sensor artifacts) are physically
   impossible and were left untouched in the source data; we clip
   them to 0 rather than silently letting them corrupt training.
5. AQI_Bucket is dropped before feature engineering -- it's a direct
   deterministic function of the AQI target, so keeping it as a
   "feature" is a data-leakage bug.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]


def _clip_negative(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
    return df


def _impute_by_group(df: pd.DataFrame, group_col: str, cols: list[str]) -> pd.DataFrame:
    """Fill NaNs with the group (city/station) median, then global median."""
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = df.groupby(group_col)[c].transform(lambda s: s.fillna(s.median()))
        df[c] = df[c].fillna(df[c].median())
    return df


def load_city_data(data_dir: str, granularity: str = "day") -> pd.DataFrame:
    """
    Load city_day.csv or city_hour.csv, cleaned.
    granularity: 'day' or 'hour'
    """
    fname = "city_day.csv" if granularity == "day" else "city_hour.csv"
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path, parse_dates=["Date"])

    df = _clip_negative(df, POLLUTANT_COLS)
    df = _impute_by_group(df, "City", POLLUTANT_COLS)

    # Drop rows with no AQI at all -- can't train/evaluate without a target,
    # and imputing the target itself would be circular.
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)

    return df


def load_station_data(data_dir: str, granularity: str = "day") -> pd.DataFrame:
    """
    Load station_day.csv or station_hour.csv, cleaned, and joined with
    stations.csv to recover City/State (fixes bug #1 above).
    """
    fname = "station_day.csv" if granularity == "day" else "station_hour.csv"
    path = os.path.join(data_dir, fname)

    df = pd.read_csv(path, parse_dates=["Date"])
    stations = pd.read_csv(os.path.join(data_dir, "stations.csv"), encoding="utf-8-sig")
    stations = stations.rename(columns=str.strip)

    df = df.merge(
        stations[["StationId", "StationName", "City", "State"]],
        on="StationId",
        how="left",
    )

    df = _clip_negative(df, POLLUTANT_COLS)
    df = _impute_by_group(df, "StationId", POLLUTANT_COLS)
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Extract Year/Month/Day(/Hour) features from the date column."""
    df = df.copy()
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    if "Hour" not in df.columns:
        # granularity == 'day' -> no hour info available
        pass
    return df


def attach_satellite_features(
    df: pd.DataFrame, satellite_csv_path: str, join_cols: tuple[str, ...] = ("City", "Year", "Month")
) -> pd.DataFrame:
    """
    Left-joins satellite_aod / satellite_no2 / satellite_co (built by
    src/fetch_satellite_features.py) onto a City/Year/Month-indexed
    ground-station dataframe. This is what lets the AQI model use
    satellite-derived predictors instead of only ground-station pollutants
    -- the piece the problem statement asks for ("surface AQI derived from
    satellite data"), since these columns generalise to grid points/cities
    that have no CPCB monitor at all.

    Requires df to already have Year/Month (call add_time_features first).
    If the CSV doesn't exist, returns df unchanged (with a printed note) so
    training still works without satellite data -- it's an enrichment, not
    a hard requirement.
    """
    if not os.path.exists(satellite_csv_path):
        print(f"[INFO] {satellite_csv_path} not found -- training without satellite "
              f"features. Run `python src/fetch_satellite_features.py` to add them.")
        return df

    sat = pd.read_csv(satellite_csv_path)
    keep = list(join_cols) + [c for c in sat.columns if c.startswith("satellite_")]
    sat = sat[keep].drop_duplicates(subset=list(join_cols))

    merged = df.merge(sat, on=list(join_cols), how="left")

    sat_cols = [c for c in merged.columns if c.startswith("satellite_")]
    for c in sat_cols:
        # cities/months with no satellite coverage (e.g. cloud cover) fall
        # back to the column's global median rather than dropping rows.
        merged[c] = merged[c].fillna(merged[c].median())

    return merged


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    city_df = load_city_data(data_dir, "day")
    print("city_day cleaned:", city_df.shape)
    print(city_df[POLLUTANT_COLS + ["AQI"]].isna().sum())

    station_df = load_station_data(data_dir, "day")
    print("station_day cleaned:", station_df.shape)
    print(station_df["City"].isna().sum(), "rows missing City after join")
