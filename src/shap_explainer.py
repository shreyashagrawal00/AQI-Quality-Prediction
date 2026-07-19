"""
shap_explainer.py
-----------------
SHAP-based explainability wrappers for the AQI models.

Provides:
  - explain_prediction(model, X_row, feature_names) -> dict of feature contributions
  - global_shap_importance(model, X_sample, feature_names) -> DataFrame
  - shap_waterfall_data(model, X_row, feature_names) -> (base_value, contributions_df)

Requires: pip install shap
Falls back gracefully (returns empty/None) if SHAP is not installed.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HAS_SHAP = False
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    pass


def shap_available() -> bool:
    return _HAS_SHAP


def explain_prediction(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict[str, float] | None:
    """
    Compute SHAP values for a single prediction row.

    Returns dict {feature_name: shap_value} sorted by abs magnitude,
    or None if SHAP is not installed.
    """
    if not _HAS_SHAP:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_row)
        # shap_values can be 2D (n_rows × n_features) or 1D if already squeezed
        if hasattr(shap_vals, "shape") and shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = np.array(shap_vals).ravel()
        names = feature_names or list(X_row.columns)
        contrib = {n: float(v) for n, v in zip(names, vals)}
        return dict(sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True))
    except Exception:
        return None


def global_shap_importance(
    model,
    X_sample: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 500,
) -> pd.DataFrame | None:
    """
    Mean absolute SHAP values over a sample of rows.

    Returns DataFrame with columns ['Feature', 'SHAP_importance'] or None.
    """
    if not _HAS_SHAP:
        return None
    try:
        sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(sample)
        if hasattr(shap_vals, "shape") and shap_vals.ndim == 2:
            mean_abs = np.abs(shap_vals).mean(axis=0)
        else:
            mean_abs = np.abs(np.array(shap_vals)).mean(axis=0)
        names = feature_names or list(X_sample.columns)
        df = pd.DataFrame({"Feature": names, "SHAP_importance": mean_abs})
        return df.sort_values("SHAP_importance", ascending=False).reset_index(drop=True)
    except Exception:
        return None


def shap_waterfall_data(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str] | None = None,
    top_n: int = 12,
) -> tuple[float, pd.DataFrame] | tuple[None, None]:
    """
    Returns (base_value, contributions_df) for a waterfall chart.

    contributions_df columns: Feature, Value (feature raw value), SHAP (contribution)
    Sorted by abs(SHAP) descending, top_n rows kept.
    """
    if not _HAS_SHAP:
        return None, None
    try:
        explainer = shap.TreeExplainer(model)
        expl = explainer(X_row)
        base_value = float(expl.base_values.ravel()[0])
        shap_vals  = expl.values.ravel()
        names = feature_names or list(X_row.columns)
        raw_vals = X_row.values.ravel()

        df = pd.DataFrame({
            "Feature": names,
            "Value":   raw_vals,
            "SHAP":    shap_vals,
        })
        df["abs_SHAP"] = df["SHAP"].abs()
        df = df.sort_values("abs_SHAP", ascending=False).head(top_n).drop(columns="abs_SHAP")
        return base_value, df.reset_index(drop=True)
    except Exception:
        return None, None
