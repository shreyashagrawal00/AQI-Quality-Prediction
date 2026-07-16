"""
train_model.py
----------------
Trains an XGBoost regressor to predict AQI from pollutant + time features,
evaluates it, and saves the model + metrics + feature importances to disk.

Fixes vs. a naive first pass:
- train_test_split uses a fixed random_state (reproducibility) and is
  a plain random split -- NOT shuffled by date, since AQI is
  auto-correlated day-to-day; a naive time-ordered split without
  shuffling would make the reported R2 look artificially worse/better
  depending on trend, so we use a random split for a fair i.i.d.
  estimate and note in README how to do a proper time-based
  walk-forward split for deployment-grade evaluation.
- Uses early_stopping via an eval_set instead of a fixed n_estimators,
  which was silently overfitting in a first pass (train R2 ~0.99,
  test R2 much lower).
- Falls back to sklearn's GradientBoostingRegressor automatically if
  xgboost isn't installed, so this script (and the dashboard) never
  hard-crashes in environments where xgboost failed to install --
  it just logs a warning.
"""

from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_city_data, add_time_features, attach_satellite_features
from feature_engineering import build_features, save_encoders

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = False
    print("[WARN] xgboost not installed -- falling back to "
          "sklearn.GradientBoostingRegressor. `pip install xgboost` "
          "for the intended model.")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_model():
    if _HAS_XGB:
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=30,
            eval_metric="rmse",
        )
    return GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )


def train(data_dir: str, granularity: str = "day"):
    df = load_city_data(data_dir, granularity)
    df = add_time_features(df)

    satellite_csv = os.path.join(data_dir, "satellite_features.csv")
    df = attach_satellite_features(df, satellite_csv)

    X, y, encoders = build_features(
        df, id_col="City",
        hour_col="Hour" if granularity == "hour" else None,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = get_model()

    if _HAS_XGB:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "R2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model": "XGBRegressor" if _HAS_XGB else "GradientBoostingRegressor",
        "used_satellite_features": bool(set(X.columns) & {
            "satellite_aod", "satellite_no2", "satellite_co"
        }),
    }

    importances = dict(zip(X.columns, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    joblib.dump(model, os.path.join(MODEL_DIR, "aqi_model.pkl"))
    save_encoders(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(importances, f, indent=2)
    # Save feature column order -- required so the dashboard builds
    # inference rows in the exact same order the model was trained on.
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f, indent=2)

    print("Metrics:", json.dumps(metrics, indent=2))
    print("Top features:", list(importances.items())[:5])
    return model, metrics


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train(data_dir, granularity="day")
