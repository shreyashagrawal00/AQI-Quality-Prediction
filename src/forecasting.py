"""
forecasting.py
--------------
7-day AQI forecasting using an XGBoost autoregressive model.

Approach:
  - Train on city_day.csv with lag features (t-1…t-7) + calendar features
  - Predict next 7 days recursively for any given city
  - Bootstrap confidence intervals (±1.5 × rolling std of residuals)

Run standalone:
    python src/forecasting.py
"""

from __future__ import annotations

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = False

from data_preprocessing import load_city_data, add_time_features
from feature_engineering import build_features, save_encoders

FORECAST_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "forecast_model.pkl"
)
FORECAST_CITIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "forecast_cities.json"
)

N_LAGS = 7


def _build_lag_features(df: pd.DataFrame, target: str = "AQI") -> pd.DataFrame:
    """Add lag columns (lag_1 … lag_7) and rolling stats."""
    df = df.copy().sort_values("Date")
    for lag in range(1, N_LAGS + 1):
        df[f"lag_{lag}"] = df.groupby("City")[target].shift(lag)
    df["rolling_mean_7"] = df.groupby("City")[target].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    df["rolling_std_7"] = df.groupby("City")[target].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).std().fillna(0)
    )
    return df.dropna(subset=[f"lag_{N_LAGS}"])


LAG_COLS = [f"lag_{i}" for i in range(1, N_LAGS + 1)] + ["rolling_mean_7", "rolling_std_7"]
CALENDAR_COLS = ["Year", "Month", "Day"]


def train_forecast_model(data_dir: str) -> None:
    """Train the autoregressive lag model and save to models/forecast_model.pkl."""
    df = load_city_data(data_dir, "day")
    df = add_time_features(df)

    # encode City as integer
    cities = sorted(df["City"].unique())
    city_to_id = {c: i for i, c in enumerate(cities)}
    df["city_id"] = df["City"].map(city_to_id)

    df = _build_lag_features(df)

    feat_cols = LAG_COLS + CALENDAR_COLS + ["city_id"]
    X = df[feat_cols].values
    y = df["AQI"].values

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    if _HAS_XGB:
        model = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42, n_jobs=-1,
            eval_metric="rmse", early_stopping_rounds=20,
        )
        from sklearn.model_selection import train_test_split as tts
        Xtr, Xv, ytr, yv = tts(X_train, y_train, test_size=0.1, random_state=42)
        model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    else:
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

    joblib.dump({"model": model, "city_to_id": city_to_id, "feat_cols": feat_cols},
                FORECAST_MODEL_PATH)
    with open(FORECAST_CITIES_PATH, "w") as f:
        json.dump(cities, f)
    print(f"Forecast model saved -> {FORECAST_MODEL_PATH}")


def load_forecast_model():
    if not os.path.exists(FORECAST_MODEL_PATH):
        return None
    return joblib.load(FORECAST_MODEL_PATH)


def forecast_city(
    city: str,
    city_day_df: pd.DataFrame,
    horizon: int = 7,
) -> pd.DataFrame | None:
    """
    Produce a `horizon`-day AQI forecast for `city`.

    Returns DataFrame: Date, predicted_aqi, lower_bound, upper_bound
    or None if the model isn't trained.
    """
    bundle = load_forecast_model()
    if bundle is None:
        return None

    model       = bundle["model"]
    city_to_id  = bundle["city_to_id"]

    if city not in city_to_id:
        return None

    city_id = city_to_id[city]
    hist = (
        city_day_df[city_day_df["City"] == city]
        .sort_values("Date")
        .dropna(subset=["AQI"])
        .tail(30)
        .copy()
    )
    if len(hist) < N_LAGS:
        return None

    recent_aqi = list(hist["AQI"].values)
    last_date  = hist["Date"].iloc[-1]

    # compute residual std from last 14 days for confidence band
    residual_std = float(np.std(np.diff(recent_aqi[-14:]))) if len(recent_aqi) >= 15 else 15.0

    preds = []
    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        lags = list(reversed(recent_aqi[-N_LAGS:]))  # lag_1 … lag_7
        roll_mean = float(np.mean(recent_aqi[-7:]))
        roll_std  = float(np.std(recent_aqi[-7:])) if len(recent_aqi) >= 7 else 0.0
        feat = lags + [roll_mean, roll_std, next_date.year, next_date.month, next_date.day, city_id]
        pred = float(model.predict(np.array([feat]))[0])
        pred = max(0.0, pred)

        half_width = 1.5 * residual_std * np.sqrt(step)  # uncertainty grows with horizon
        preds.append({
            "Date": next_date,
            "predicted_aqi": round(pred, 1),
            "lower_bound":   max(0.0, round(pred - half_width, 1)),
            "upper_bound":   round(pred + half_width, 1),
        })
        recent_aqi.append(pred)

    return pd.DataFrame(preds)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_forecast_model(data_dir)
