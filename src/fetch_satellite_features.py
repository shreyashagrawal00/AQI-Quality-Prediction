"""
fetch_satellite_features.py
----------------------------
One-off script: builds data/satellite_features.csv (City, Year, Month,
satellite_aod, satellite_no2, satellite_co) covering every city/month
present in city_day.csv, so train_model.py and the dashboard can merge
satellite features in without hitting Earth Engine on every run.

Usage:
    python src/fetch_satellite_features.py                # auto-detect EE, else simulate
    python src/fetch_satellite_features.py --live          # force live EE (errors if not auth'd)
    python src/fetch_satellite_features.py --simulate      # force placeholder values

Re-run this after `earthengine authenticate` to replace simulated values
with real Sentinel-5P/MODIS pixels -- nothing else needs to change,
train_model.py picks up data/satellite_features.csv automatically if present.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(__file__))
from city_coordinates import CITY_COORDINATES
from satellite_features import build_satellite_feature_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--out", default=None, help="output CSV path (default: <data-dir>/satellite_features.csv)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="force real Earth Engine calls")
    mode.add_argument("--simulate", action="store_true", help="force placeholder values (no EE needed)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(args.data_dir, "satellite_features.csv")
    city_day_path = os.path.join(args.data_dir, "city_day.csv")

    if not os.path.exists(city_day_path):
        raise SystemExit(f"Can't find {city_day_path} -- put the CPCB CSVs in {args.data_dir}/ first.")

    df = pd.read_csv(city_day_path, parse_dates=["Date"], usecols=["City", "Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    year_months = sorted(df[["Year", "Month"]].drop_duplicates().itertuples(index=False, name=None))

    known_cities = {c: coords for c, coords in CITY_COORDINATES.items() if c in set(df["City"])}
    missing = set(df["City"]) - set(known_cities)
    if missing:
        print(f"[WARN] no coordinates on file for: {sorted(missing)} -- skipping them. "
              f"Add them to src/city_coordinates.py if you need satellite features there.")

    use_live = True if args.live else (False if args.simulate else None)
    print(f"Fetching satellite features for {len(known_cities)} cities x "
          f"{len(year_months)} months ({len(known_cities) * len(year_months)} calls if live)...")

    table = build_satellite_feature_table(known_cities, year_months, use_live=use_live)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    table.to_csv(out_path, index=False)

    n_live = int((table["_source"] == "live").sum())
    n_sim = int((table["_source"] == "simulated").sum())
    print(f"Wrote {len(table)} rows to {out_path} ({n_live} live, {n_sim} simulated).")
    if n_sim and not args.simulate:
        print("[NOTE] Some/all rows are simulated placeholders -- run "
              "`earthengine authenticate` and re-run this script for real satellite data.")


if __name__ == "__main__":
    main()
