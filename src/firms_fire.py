"""
firms_fire.py
-------------
Fetches NASA FIRMS (Fire Information for Resource Management System)
active fire data for India from the public CSV endpoint.

Live path: Downloads VIIRS S-NPP NRT / MODIS C6.1 7-day fire data.
No API key is required for the public 7-day bounding-box CSV endpoint.

Simulate path: Returns a deterministic, plausible fire-point table that
still has the same schema and covers the main biomass-burning regions
(Punjab/Haryana in Oct-Nov, central/eastern India pre-monsoon Apr-May).

Usage:
    from firms_fire import get_fire_data, fire_density_grid

    fire_df = get_fire_data(days=7, satellite="VIIRS_SNPP_NRT")
    density  = fire_density_grid(fire_df, india_grid(25, 25))
"""

from __future__ import annotations

import datetime
import hashlib
import io
import warnings
from typing import Literal

import os
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")

# ── public FIRMS CSV endpoint (no key needed for bbox/7-day queries) ───────
_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
_INDIA_BBOX = "68.0,8.0,97.0,35.0"   # west,south,east,north

SATELLITE_OPTIONS: dict[str, str] = {
    "VIIRS_SNPP_NRT":  "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_NRT",
    "MODIS_NRT":       "MODIS_NRT",
}

# columns we guarantee in the returned DataFrame
FIRE_COLUMNS = [
    "latitude", "longitude", "brightness", "frp",
    "acq_date", "satellite", "confidence",
]


# ---------------------------------------------------------------------------
# Live download
# ---------------------------------------------------------------------------

def _download_firms(satellite: str = "VIIRS_SNPP_NRT", days: int = 7) -> pd.DataFrame | None:
    """
    Attempt to download FIRMS fire data for India bbox.
    Returns None on any network / parse error.
    """
    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        print("[WARN] FIRMS_MAP_KEY is missing from environment. Active fire data will fallback to simulation mode.")
        return None

    try:
        import requests
        days = min(max(int(days), 1), 5)
        map_key = str(map_key).strip().strip('"').strip("'")
        satellite = str(satellite).strip()
        url = f"{_FIRMS_BASE}/{map_key}/{satellite}/{_INDIA_BBOX}/{days}"

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200 or len(resp.text) < 50:
            print(f"[WARN] NASA FIRMS API returned status code {resp.status_code} or empty response.")
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        # normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        rename = {
            "bright_ti4": "brightness", "bright_t21": "brightness",
            "fire_radiative_power": "frp",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        # ensure required cols exist
        for col in ["latitude", "longitude", "frp", "acq_date"]:
            if col not in df.columns:
                print(f"[WARN] Required column {col} missing in FIRMS API response.")
                return None
        if "brightness" not in df.columns:
            df["brightness"] = np.nan
        if "confidence" not in df.columns:
            df["confidence"] = "nominal"
        df["satellite"] = satellite
        return df[FIRE_COLUMNS].dropna(subset=["latitude", "longitude"])
    except Exception as exc:
        print(f"[WARN] Live NASA FIRMS fetch failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Simulated fallback
# ---------------------------------------------------------------------------

def _simulate_fire_data(days: int = 7, reference_date: datetime.date | None = None) -> pd.DataFrame:
    """
    Deterministic placeholder fire points for India.
    Covers major burning regions with realistic seasonal weighting:
      - Punjab/Haryana (Oct–Nov stubble burning, lat ~30, lon ~75)
      - Central India / Odisha (Apr–May pre-monsoon, lat ~20–23, lon ~82–86)
      - Northeast / Assam (Feb–Mar, lat ~26, lon ~93)
    """
    if reference_date is None:
        reference_date = datetime.date.today()
    month = reference_date.month

    # season-weighted cluster centres
    clusters = []
    if month in (10, 11, 12, 1):   # post-monsoon stubble burning
        clusters += [
            (30.2, 75.2, 500, 3.5),   # Punjab
            (29.5, 76.5, 450, 3.0),   # Haryana
            (29.0, 80.0, 350, 2.0),   # Western UP
        ]
    if month in (3, 4, 5):          # pre-monsoon
        clusters += [
            (21.5, 83.5, 400, 4.0),   # Odisha/Chhattisgarh
            (22.0, 85.0, 380, 3.5),   # Jharkhand
            (18.0, 79.5, 300, 2.5),   # Telangana/Andhra
        ]
    if month in (2, 3):             # Northeast
        clusters += [
            (26.2, 93.5, 420, 3.0),   # Assam
            (25.5, 91.0, 390, 2.8),   # Meghalaya
        ]
    if not clusters:                # off-season — sparse background
        clusters = [
            (24.0, 80.0, 300, 1.5),
            (20.0, 85.0, 280, 1.2),
        ]

    seed = int(hashlib.md5(str(reference_date).encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    rows = []
    for (clat, clon, base_brightness, base_frp) in clusters:
        n = rng.integers(40, 120)
        lats = rng.normal(clat, 0.8, n)
        lons = rng.normal(clon, 0.8, n)
        brights = rng.normal(base_brightness, 30, n).clip(280, 700)
        frps = rng.exponential(base_frp, n).clip(0.1, 50)
        confs = rng.choice(["low", "nominal", "high"], n, p=[0.1, 0.6, 0.3])
        for i in range(n):
            rows.append({
                "latitude": float(lats[i]),
                "longitude": float(lons[i]),
                "brightness": float(brights[i]),
                "frp": float(frps[i]),
                "acq_date": str(reference_date - datetime.timedelta(days=int(rng.integers(0, days)))),
                "satellite": "SIMULATED",
                "confidence": confs[i],
            })

    df = pd.DataFrame(rows)
    # filter to India bbox
    df = df[(df.latitude >= 8) & (df.latitude <= 35) &
            (df.longitude >= 68) & (df.longitude <= 97)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fire_data(
    days: int = 7,
    satellite: str = "VIIRS_SNPP_NRT",
    reference_date: datetime.date | None = None,
    force_simulate: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Returns (fire_df, is_live).

    fire_df columns: latitude, longitude, brightness, frp, acq_date, satellite, confidence
    is_live: True if data came from FIRMS API, False if simulated.

    Parameters
    ----------
    days : int
        Rolling window of days (max 10 for free endpoint).
    satellite : str
        One of SATELLITE_OPTIONS keys.
    reference_date : date | None
        Used for simulated fallback date anchor (defaults to today).
    force_simulate : bool
        Skip live download attempt.
    """
    if not force_simulate:
        df = _download_firms(satellite, days)
        if df is not None and len(df) > 0:
            return df, True

    # fallback
    if reference_date is None:
        reference_date = datetime.date.today()
    return _simulate_fire_data(days, reference_date), False


def fire_density_grid(
    fire_df: pd.DataFrame,
    grid: pd.DataFrame,
    radius_deg: float = 1.0,
) -> pd.DataFrame:
    """
    For each grid point (lat, lon), count fire pixels within `radius_deg`
    and sum their FRP.

    Returns grid with added columns: fire_count, fire_frp_sum.
    """
    out = grid.copy()
    counts = np.zeros(len(out), dtype=int)
    frp_sums = np.zeros(len(out), dtype=float)

    if len(fire_df) == 0:
        out["fire_count"] = 0
        out["fire_frp_sum"] = 0.0
        return out

    fire_lats = fire_df["latitude"].values
    fire_lons = fire_df["longitude"].values
    fire_frps = fire_df["frp"].values

    for i, row in enumerate(out.itertuples()):
        dlat = fire_lats - row.lat
        dlon = fire_lons - row.lon
        dist = np.sqrt(dlat**2 + dlon**2)
        mask = dist <= radius_deg
        counts[i]   = int(mask.sum())
        frp_sums[i] = float(fire_frps[mask].sum())

    out["fire_count"]   = counts
    out["fire_frp_sum"] = frp_sums
    return out


def fire_statistics(fire_df: pd.DataFrame) -> dict:
    """Summary statistics for a fire DataFrame."""
    if len(fire_df) == 0:
        return {"count": 0, "total_frp": 0.0, "max_frp": 0.0,
                "mean_brightness": 0.0, "date_range": "N/A"}
    return {
        "count": int(len(fire_df)),
        "total_frp": float(fire_df["frp"].sum()),
        "max_frp":   float(fire_df["frp"].max()),
        "mean_brightness": float(fire_df["brightness"].mean()) if fire_df["brightness"].notna().any() else 0.0,
        "date_range": f"{fire_df['acq_date'].min()} → {fire_df['acq_date'].max()}",
    }
