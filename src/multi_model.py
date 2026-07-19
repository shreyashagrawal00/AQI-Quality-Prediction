"""
multi_model.py
--------------
Trains and compares multiple regression models for AQI prediction:
  - Random Forest (sklearn)
  - XGBoost
  - LightGBM
  - CatBoost

Each model is evaluated on the same train/test split. The best model
(lowest RMSE on the held-out test set) is saved as 'best_model.pkl'
so the dashboard can load it without hardcoding a model name.

Run:
    python src/multi_model.py

Outputs (all in models/):
    rf_model.pkl, xgb_model.pkl, lgbm_model.pkl, cat_model.pkl
    best_model.pkl  (symlink/copy of the winner)
    model_comparison.json
    validation_data.npz   (y_test, y_pred arrays for Validation page)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from data_preprocessing import load_city_data, add_time_features, attach_satellite_features
from feature_engineering import build_features, save_encoders

# ── optional heavy deps ────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("[WARN] xgboost not installed - XGB model skipped.")

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False
    print("[WARN] lightgbm not installed - LightGBM model skipped.")

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False
    print("[WARN] catboost not installed - CatBoost model skipped.")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _mape(y_true, y_pred) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true, y_pred, name: str) -> dict:
    return {
        "model": name,
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
        "MAPE": _mape(np.array(y_true), np.array(y_pred)),
    }


def _build_candidates() -> dict:
    """Return dict[name -> model_instance] for all installed libraries."""
    candidates = {}

    candidates["Random Forest"] = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=3,
        n_jobs=-1, random_state=42
    )

    if _HAS_XGB:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
            early_stopping_rounds=30, eval_metric="rmse",
        )

    if _HAS_LGBM:
        candidates["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1,
        )

    if _HAS_CAT:
        candidates["CatBoost"] = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6,
            random_seed=42, verbose=0,
        )

    return candidates


def train_all(data_dir: str, granularity: str = "day") -> dict:
    """Train all candidate models on the same split. Returns comparison dict."""
    print("Loading & preprocessing data ...")
    df = load_city_data(data_dir, granularity)
    df = add_time_features(df)
    satellite_csv = os.path.join(data_dir, "satellite_features.csv")
    df = attach_satellite_features(df, satellite_csv)

    X, y, encoders = build_features(df, id_col="City",
                                    hour_col="Hour" if granularity == "hour" else None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    feature_columns = list(X.columns)

    save_encoders(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    candidates = _build_candidates()
    comparison = []
    best_name, best_rmse, best_model = None, float("inf"), None
    best_preds = None

    for name, model in candidates.items():
        print(f"Training {name} ...")
        try:
            if name == "XGBoost":
                from sklearn.model_selection import train_test_split as tts
                Xtr, Xval, ytr, yval = tts(X_train, y_train, test_size=0.1, random_state=42)
                model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            m = _metrics(y_test, preds, name)
            m["n_train"] = int(len(X_train))
            m["n_test"]  = int(len(X_test))
            comparison.append(m)

            slug = name.lower().replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f"{slug}_model.pkl")
            joblib.dump(model, model_path)
            print(f"  -> RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}  saved {model_path}")

            if m["RMSE"] < best_rmse:
                best_rmse, best_name, best_model = m["RMSE"], name, model
                best_preds = preds
        except Exception as exc:
            print(f"  [ERROR] {name} failed: {exc}")

    if best_model is None:
        raise RuntimeError("No model trained successfully - check deps.")

    # ── save best model under canonical name ──────────────────────────────
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

    # ── feature importance (best model) ───────────────────────────────────
    try:
        imp = dict(zip(feature_columns, best_model.feature_importances_.tolist()))
        imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    except AttributeError:
        imp = {}
    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(imp, f, indent=2)

    # ── validation arrays for Validation page ─────────────────────────────
    np.savez(
        os.path.join(MODEL_DIR, "validation_data.npz"),
        y_test=np.array(y_test),
        y_pred=np.array(best_preds),
    )

    # ── individual metrics files (for backward compat with app.py) ─────────
    best_m = next(m for m in comparison if m["model"] == best_name)
    best_m["used_satellite_features"] = bool(
        set(feature_columns) & {"satellite_aod", "satellite_no2", "satellite_co"}
    )
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(best_m, f, indent=2)

    # ── model comparison JSON ──────────────────────────────────────────────
    result = {
        "best_model": best_name,
        "best_rmse": best_rmse,
        "feature_columns": feature_columns,
        "comparison": comparison,
    }
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Best model: {best_name} (RMSE={best_rmse:.2f})")
    print("   Saved to models/best_model.pkl")
    return result


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train_all(data_dir, granularity="day")
