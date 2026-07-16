"""
hcho_hotspots.py
-----------------
Spatio-temporal HCHO (formaldehyde) hotspot detection over India from
Sentinel-5P TROPOMI, with a focus on biomass-burning periods (the
problem statement's actual ask -- not just a single-period map).

Two things this adds over a single static percentile map:
1. Real Earth Engine calls (get_hcho_mean_image / get_hcho_grid_timeseries),
   reused from the previous single-period version but now driving a proper
   time series.
2. A burning-season-vs-baseline ANOMALY view: for a grid of points over
   India, compare mean HCHO during known biomass-burning months (e.g.
   Oct-Nov post-monsoon paddy-stubble burning in Punjab/Haryana, or
   Apr-May pre-monsoon burning) against a baseline of the remaining
   months, and flag points with a large z-score as burning-linked
   hotspots (as opposed to a location that's just always high, e.g. a
   permanently industrial/urban area).

Falls back to simulate_hcho_grid()/simulate_hcho_timeseries() when Earth
Engine isn't authenticated, clearly labeled, so the dashboard and the
anomaly-detection logic can still be demoed end-to-end without credentials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from satellite_features import init_earth_engine  # reuse the same EE session/init logic

HCHO_COLLECTION = "COPERNICUS/S5P/OFFL/L3_HCHO"
HCHO_BAND = "tropospheric_HCHO_column_number_density"

# Known biomass-burning windows over India (month numbers). Post-monsoon
# stubble burning (Punjab/Haryana/western UP) peaks Oct-Nov; there's a
# smaller pre-monsoon agricultural-burning bump Apr-May in parts of
# central/eastern India. Adjust for a specific region/year as needed.
DEFAULT_BURNING_MONTHS = (10, 11)
DEFAULT_BASELINE_MONTHS = (1, 2, 3, 6, 7, 8, 12)


def india_grid(n_lat: int = 25, n_lon: int = 25) -> pd.DataFrame:
    """Regular lat/lon grid roughly covering the Indian mainland."""
    lats = np.linspace(8.0, 35.0, n_lat)
    lons = np.linspace(68.0, 97.0, n_lon)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    return pd.DataFrame({"lat": lat_mesh.ravel(), "lon": lon_mesh.ravel()})


# ---------------------------------------------------------------------------
# Live Earth Engine path
# ---------------------------------------------------------------------------
def get_hcho_mean_image(start_date: str, end_date: str, region):
    """Mean HCHO ee.Image over [start_date, end_date) clipped to region."""
    import ee
    if not init_earth_engine():
        raise RuntimeError("Earth Engine not initialised.")
    collection = (
        ee.ImageCollection(HCHO_COLLECTION)
        .filterDate(start_date, end_date)
        .filterBounds(region)
        .select(HCHO_BAND)
    )
    return collection.mean().clip(region)


def get_hcho_hotspot_mask(mean_image, region, percentile: float = 90):
    """Pixels above the given percentile for the period -- single-period hotspot mask."""
    import ee
    stats = mean_image.reduceRegion(
        reducer=ee.Reducer.percentile([percentile]),
        geometry=region, scale=5000, maxPixels=1e9,
    )
    threshold = stats.get(HCHO_BAND)
    return mean_image.gt(ee.Number(threshold))


def get_hcho_grid_timeseries(
    grid: pd.DataFrame, year: int, months: list[int]
) -> pd.DataFrame:
    """
    Live: monthly mean HCHO at each grid point for the given year/months.
    Returns columns: lat, lon, month, hcho.
    Uses ee.Image.sampleRegions in one call per month (fast) rather than
    one reduceRegion call per point (slow).
    """
    import ee
    if not init_earth_engine():
        raise RuntimeError("Earth Engine not initialised.")

    points = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([row.lon, row.lat]), {"lat": row.lat, "lon": row.lon})
        for row in grid.itertuples()
    ])

    rows = []
    for month in months:
        start = f"{year}-{month:02d}-01"
        end_month = month + 1 if month < 12 else 1
        end_year = year if month < 12 else year + 1
        end = f"{end_year}-{end_month:02d}-01"

        img = (
            ee.ImageCollection(HCHO_COLLECTION)
            .filterDate(start, end)
            .select(HCHO_BAND)
            .mean()
        )
        sampled = img.sampleRegions(collection=points, scale=5000, geometries=False).getInfo()
        for feat in sampled["features"]:
            props = feat["properties"]
            rows.append({
                "lat": props.get("lat"), "lon": props.get("lon"),
                "month": month, "hcho": props.get(HCHO_BAND),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulate fallback (clearly labeled placeholder, same shape as the live path)
# ---------------------------------------------------------------------------
def simulate_hcho_timeseries(grid: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """
    Placeholder time series with an intentional Oct/Nov bump concentrated
    over the Indo-Gangetic Plain (roughly lat 26-31, lon 74-83 -- Punjab/
    Haryana/western UP), so the anomaly-detection logic below has something
    realistic-shaped to find. NOT real satellite data.
    """
    rng = np.random.default_rng(42)
    rows = []
    for _, pt in grid.iterrows():
        igp_mask = (26 <= pt.lat <= 31) and (74 <= pt.lon <= 83)
        for month in months:
            base = rng.uniform(0.5, 1.3)
            burning_bump = 2.5 if (igp_mask and month in DEFAULT_BURNING_MONTHS) else 0.0
            noise = rng.normal(0, 0.15)
            rows.append({
                "lat": pt.lat, "lon": pt.lon, "month": month,
                "hcho": max(base + burning_bump + noise, 0.05),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly detection (shared by live and simulated data -- same function)
# ---------------------------------------------------------------------------
def compute_burning_anomalies(
    ts_df: pd.DataFrame,
    burning_months: tuple[int, ...] = DEFAULT_BURNING_MONTHS,
    baseline_months: tuple[int, ...] = DEFAULT_BASELINE_MONTHS,
    z_threshold: float = 1.5,
) -> pd.DataFrame:
    """
    For each grid point, compare mean HCHO during burning_months against
    the baseline_months mean/std, and flag points where the burning-season
    HCHO is anomalously high relative to that point's OWN baseline (so an
    always-high industrial pixel isn't mistaken for a burning hotspot).

    Returns one row per (lat, lon): baseline_mean, burning_mean, z_score,
    is_hotspot (z_score >= z_threshold).
    """
    burning = (
        ts_df[ts_df["month"].isin(burning_months)]
        .groupby(["lat", "lon"])["hcho"].mean()
        .rename("burning_mean")
    )
    baseline = (
        ts_df[ts_df["month"].isin(baseline_months)]
        .groupby(["lat", "lon"])["hcho"].agg(["mean", "std"])
        .rename(columns={"mean": "baseline_mean", "std": "baseline_std"})
    )

    out = baseline.join(burning, how="inner").reset_index()
    # guard against zero/near-zero std at points with near-constant baseline
    out["baseline_std"] = out["baseline_std"].replace(0, np.nan).fillna(out["baseline_std"].mean())
    out["z_score"] = (out["burning_mean"] - out["baseline_mean"]) / out["baseline_std"]
    out["is_hotspot"] = out["z_score"] >= z_threshold
    return out
