"""
train_satellite_aqi_model.py
------------------------------
Trains a SECOND, separate AQI model that uses ONLY satellite-derived
features (satellite_aod, satellite_no2, satellite_co, Month) -- no
ground-station pollutant readings at all.

Why a second model instead of just reusing aqi_model.pkl: the main
model (train_model.py) needs PM2.5/PM10/NO2/... as inputs, which only
exist at CPCB station cities. That model genuinely can't predict AQI
at an arbitrary point in India with no monitor. This one can, because
its only inputs (AOD/NO2/CO columns) come from satellites with
nationwide coverage -- it's what actually powers the "Surface AQI Map"
grid view in the dashboard.

It's trained on the SAME label (city-month mean AQI from city_day.csv)
but restricted to the satellite feature set, so accuracy will be lower
than the full ground-station model -- that's expected and worth stating
in a demo: it trades some accuracy for being usable anywhere, which is
the whole point of the "surface AQI from satellite data" ask.

Usage:
    python src/fetch_satellite_features.py     # if not already run
    python src/train_satellite_aqi_model.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ["satellite_aod", "satellite_no2", "satellite_co", "Month"]
TARGET = "AQI"


def _get_model():
    if _HAS_XGB:
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )


def train(data_dir: str):
    city_day_path = os.path.join(data_dir, "city_day.csv")
    satellite_path = os.path.join(data_dir, "satellite_features.csv")
    if not os.path.exists(satellite_path):
        raise SystemExit(
            f"{satellite_path} not found -- run "
            "`python src/fetch_satellite_features.py` first."
        )

    ground = pd.read_csv(city_day_path, parse_dates=["Date"])
    ground["Year"] = ground["Date"].dt.year
    ground["Month"] = ground["Date"].dt.month
    # City-month mean AQI -- matches the resolution satellite_features.csv is built at
    monthly_aqi = (
        ground.dropna(subset=["AQI"])
        .groupby(["City", "Year", "Month"], as_index=False)["AQI"].mean()
    )

    sat = pd.read_csv(satellite_path)
    df = monthly_aqi.merge(sat, on=["City", "Year", "Month"], how="inner")

    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} City/Year/Month rows matched between city_day.csv and "
            "satellite_features.csv -- check that fetch_satellite_features.py ran "
            "over the same date range as your ground data."
        )

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = _get_model()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "XGBRegressor" if _HAS_XGB else "GradientBoostingRegressor",
        "features": FEATURES,
        "note": "Trained on satellite features ONLY (no ground pollutant "
                "readings) so it can estimate AQI anywhere with satellite "
                "coverage, not just at CPCB station cities. Expect lower "
                "accuracy than models/aqi_model.pkl.",
        "satellite_source": df["_source"].mode().iat[0] if "_source" in df.columns else "unknown",
    }

    joblib.dump(model, os.path.join(MODEL_DIR, "satellite_aqi_model.pkl"))
    with open(os.path.join(MODEL_DIR, "satellite_aqi_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODEL_DIR, "satellite_aqi_columns.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    print("Satellite-only AQI model metrics:", json.dumps(metrics, indent=2))
    return model, metrics


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train(data_dir)