"""
feature_engineering.py
-----------------------
Turns cleaned pollutant/date data into a model-ready feature matrix.

Fixes vs. a naive approach:
- City/StationId are label-encoded (not one-hot), which keeps the
  feature space small and is what tree-based models like XGBoost
  handle natively/well -- one-hot on 26+ cities/1000+ stations would
  blow up dimensionality for no accuracy benefit in a tree model.
- The LabelEncoder is fit ONCE on training data and reused (via
  encoders.pkl) at inference time, otherwise a category unseen at
  fit-time crashes predict() in production -- a common bug.
- AQI_Bucket / AQI_category style columns are explicitly excluded
  from X (see data_preprocessing.py docstring: they leak the target).
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

BASE_FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
    "Year", "Month", "Day",
]

# Satellite-derived proxies (MODIS AOD, Sentinel-5P NO2/CO columns) --
# added to the feature set automatically by build_features() when present,
# so the model isn't limited to ground-station-only predictors. See
# src/satellite_features.py and src/fetch_satellite_features.py.
SATELLITE_FEATURES = ["satellite_aod", "satellite_no2", "satellite_co"]

TARGET = "AQI"


def build_features(
    df: pd.DataFrame,
    id_col: str = "City",
    hour_col: str | None = None,
    encoders: dict | None = None,
    fit_encoders: bool = True,
):
    """
    Returns (X, y, encoders).

    id_col: 'City' for city-level data, 'StationId' for station-level data.
    hour_col: pass 'Hour' if using *_hour.csv granularity.
    encoders: dict of {col: LabelEncoder}. Pass the dict returned from
              training when transforming new/inference data
              (fit_encoders=False).
    """
    df = df.copy()
    features = list(BASE_FEATURES)
    if hour_col and hour_col in df.columns:
        features.append(hour_col)
    # Include satellite features only if they were actually attached
    # (via attach_satellite_features) -- keeps this backwards-compatible
    # with data that has no satellite enrichment.
    features += [c for c in SATELLITE_FEATURES if c in df.columns]

    encoders = encoders or {}

    if id_col in df.columns:
        if fit_encoders:
            le = LabelEncoder()
            df[f"{id_col}_enc"] = le.fit_transform(df[id_col].astype(str))
            encoders[id_col] = le
        else:
            le = encoders[id_col]
            # unseen categories at inference -> map to a new 'unknown' bucket
            known = set(le.classes_)
            df[f"{id_col}_enc"] = df[id_col].astype(str).apply(
                lambda v: v if v in known else le.classes_[0]
            )
            df[f"{id_col}_enc"] = le.transform(df[f"{id_col}_enc"])
        features.append(f"{id_col}_enc")

    X = df[features]
    y = df[TARGET] if TARGET in df.columns else None
    return X, y, encoders


def save_encoders(encoders: dict, path: str):
    joblib.dump(encoders, path)


def load_encoders(path: str) -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    import os
    from data_preprocessing import load_city_data, add_time_features

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    df = load_city_data(data_dir, "day")
    df = add_time_features(df)
    X, y, enc = build_features(df, id_col="City")
    print(X.shape, y.shape)
    print(X.head())
