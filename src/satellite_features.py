"""
satellite_features.py
----------------------
Satellite-derived pollution proxies (MODIS AOD, Sentinel-5P NO2 and CO
column densities) used as EXTRA model features so the AQI model can
estimate surface AQI in places without a CPCB ground station -- this is
the part of the problem statement ("surface AQI derived from satellite
data") that ground-station-only training does not address.

Two ways to run this:

1. Live mode (real satellite data): requires `earthengine-api` +
   Earth Engine credentials.

       pip install earthengine-api
       earthengine authenticate        # one-time, opens a browser
       # or, for headless/deployment use, a service account:
       #   export EE_SERVICE_ACCOUNT=name@project.iam.gserviceaccount.com
       #   export EE_SERVICE_ACCOUNT_KEY=/path/to/key.json

   Then `init_earth_engine()` succeeds and every function below pulls
   real MODIS/Sentinel-5P pixels.

2. Simulate mode: if Earth Engine isn't available/authenticated, every
   function that would normally hit the API falls back to
   `simulate_satellite_row()`, which returns clearly-labeled placeholder
   values (loosely correlated with month/city so the pipeline and the
   Streamlit UI behave sensibly in a demo). This is a fallback for
   development, NOT a substitute for the real thing in a submission --
   the README explains how to switch it on.

Data sources (all public, no cost, queried via Earth Engine):
- AOD:  MODIS/061/MCD19A2_GRANULES, band Optical_Depth_047
- NO2:  COPERNICUS/S5P/OFFL/L3_NO2, band tropospheric_NO2_column_number_density
- CO:   COPERNICUS/S5P/OFFL/L3_CO,  band CO_column_number_density
"""

from __future__ import annotations

import datetime as _dt
import os

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up a .env file in the project root, if present
except ImportError:
    pass  # python-dotenv not installed -- fall back to whatever's already in the shell env

SATELLITE_FEATURE_COLS = ["satellite_aod", "satellite_no2", "satellite_co"]

_EE_READY = False
_EE_LAST_ERROR: str | None = None
_ee = None  # lazily bound to the `ee` module once initialised


def init_earth_engine() -> bool:
    """
    Initialise Earth Engine once per process. Returns True on success,
    False if the library/credentials/project aren't available -- callers
    should treat False as "use simulate mode". Never raises; call
    get_ee_last_error() after a False return to see *why* it failed
    (common cause: Earth Engine now requires a linked Google Cloud
    project -- set EE_PROJECT=your-project-id).
    """
    global _EE_READY, _ee, _EE_LAST_ERROR
    if _EE_READY:
        return True
    try:
        import ee
    except ImportError:
        _EE_LAST_ERROR = "earthengine-api is not installed in this Python environment."
        return False

    try:
        service_account = os.environ.get("EE_SERVICE_ACCOUNT")
        key_path = os.environ.get("EE_SERVICE_ACCOUNT_KEY")
        project = os.environ.get("EE_PROJECT")  # required by Earth Engine since late 2024

        if service_account and key_path:
            credentials = ee.ServiceAccountCredentials(service_account, key_path)
            ee.Initialize(credentials, project=project) if project else ee.Initialize(credentials)
        elif project:
            ee.Initialize(project=project)  # uses `earthengine authenticate` cached credentials
        else:
            ee.Initialize()  # uses `earthengine authenticate` cached credentials
        _ee = ee
        _EE_READY = True
        _EE_LAST_ERROR = None
        return True
    except Exception as exc:
        # Not authenticated / no project linked / etc. -- callers fall back
        # to simulate mode. Keep the real reason so the UI can show it
        # instead of a generic "not authenticated" (common real cause as
        # of late 2024: Earth Engine requires a linked Cloud project --
        # set EE_PROJECT=your-project-id and retry).
        _EE_LAST_ERROR = str(exc)
        return False


def get_ee_last_error() -> str | None:
    """The exception message from the most recent failed init_earth_engine() call, if any."""
    return _EE_LAST_ERROR


def _monthly_window(date: _dt.date) -> tuple[str, str]:
    """[first_of_month, first_of_next_month) as ISO strings -- Sentinel-5P/MODIS
    coverage on any single day/point is patchy, so we average over the month."""
    start = date.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start.isoformat(), end.isoformat()


def get_satellite_features_for_point(
    lat: float, lon: float, date: _dt.date, buffer_km: float = 25.0
) -> dict:
    """
    Live Earth Engine query for one (lat, lon, month). Returns
    {"satellite_aod": float|nan, "satellite_no2": float|nan, "satellite_co": float|nan}.
    Raises RuntimeError if Earth Engine isn't initialised -- call
    init_earth_engine() first and check its return value.
    """
    if not _EE_READY:
        raise RuntimeError(
            "Earth Engine is not initialised. Call init_earth_engine() and "
            "check it returns True before calling this function, or use "
            "simulate_satellite_row() as a fallback."
        )
    ee = _ee
    start, end = _monthly_window(date)
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

    def _mean_band(collection_id: str, band: str) -> float:
        coll = (
            ee.ImageCollection(collection_id)
            .filterDate(start, end)
            .filterBounds(point)
            .select(band)
        )
        img = coll.mean()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=5000, maxPixels=1e9
        )
        val = stats.get(band).getInfo()
        return float(val) if val is not None else float("nan")

    return {
        "satellite_aod": _mean_band("MODIS/061/MCD19A2_GRANULES", "Optical_Depth_047"),
        "satellite_no2": _mean_band(
            "COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"
        ),
        "satellite_co": _mean_band(
            "COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density"
        ),
    }


def simulate_satellite_row(city: str, year: int, month: int, seed_salt: int = 0) -> dict:
    """
    Clearly-labeled placeholder used only when Earth Engine isn't available.
    Deterministic per (city, year, month) so repeated calls/caching are stable.
    Values are shaped to be plausible (AOD ~0.1-1.5, NO2/CO in typical Sentinel-5P
    ranges) with a winter/post-monsoon bump, NOT fit to real satellite pixels.
    """
    rng = np.random.default_rng(abs(hash((city, year, month, seed_salt))) % (2**32))
    seasonal = 1.6 if month in (10, 11, 12, 1) else 1.0  # winter/post-harvest bump
    return {
        "satellite_aod": float(rng.uniform(0.15, 0.6) * seasonal),
        "satellite_no2": float(rng.uniform(5e14, 4e15) * seasonal),
        "satellite_co": float(rng.uniform(0.02, 0.05) * seasonal),
    }


def simulate_satellite_grid_row(lat: float, lon: float, year: int, month: int) -> dict:
    """Same idea as simulate_satellite_row() but keyed by lat/lon instead of a
    city name, for building a map over arbitrary grid points (not just cities)."""
    rng = np.random.default_rng(abs(hash((round(lat, 2), round(lon, 2), year, month))) % (2**32))
    seasonal = 1.6 if month in (10, 11, 12, 1) else 1.0
    igp_boost = 1.4 if (26 <= lat <= 31 and 74 <= lon <= 83) else 1.0  # Indo-Gangetic Plain
    return {
        "satellite_aod": float(rng.uniform(0.15, 0.6) * seasonal * igp_boost),
        "satellite_no2": float(rng.uniform(5e14, 4e15) * seasonal * igp_boost),
        "satellite_co": float(rng.uniform(0.02, 0.05) * seasonal * igp_boost),
    }


def build_grid_satellite_table(
    grid: pd.DataFrame, year: int, month: int, use_live: bool | None = None
) -> pd.DataFrame:
    """
    Like build_satellite_feature_table but for an arbitrary lat/lon grid
    (e.g. hcho_hotspots.india_grid()) instead of named cities -- this is
    what powers a surface-AQI *map*, since most grid points have no CPCB
    station name to look up.
    """
    if use_live is None:
        use_live = init_earth_engine()
    elif use_live and not init_earth_engine():
        print("[WARN] Earth Engine requested but not available/authenticated -- "
              "falling back to simulate_satellite_grid_row().")
        use_live = False

    rows = []
    for row in grid.itertuples():
        if use_live:
            try:
                feats = get_satellite_features_for_point(row.lat, row.lon, _dt.date(year, month, 1))
                feats["_source"] = "live"
            except Exception as exc:
                print(f"[WARN] live fetch failed for ({row.lat:.2f},{row.lon:.2f}) "
                      f"{year}-{month:02d} ({exc}); using simulated value.")
                feats = simulate_satellite_grid_row(row.lat, row.lon, year, month)
                feats["_source"] = "simulated"
        else:
            feats = simulate_satellite_grid_row(row.lat, row.lon, year, month)
            feats["_source"] = "simulated"
        rows.append({"lat": row.lat, "lon": row.lon, **feats})
    return pd.DataFrame(rows)


def build_satellite_feature_table(
    city_coords: dict[str, tuple[float, float]],
    year_months: list[tuple[int, int]],
    use_live: bool | None = None,
) -> pd.DataFrame:
    """
    Builds one row per (City, Year, Month) with satellite_aod/no2/co.
    use_live=None -> auto-detect (tries init_earth_engine(), falls back to
    simulate mode with a printed warning). Pass True/False to force a mode.
    """
    if use_live is None:
        use_live = init_earth_engine()
    elif use_live:
        if not init_earth_engine():
            print("[WARN] Earth Engine requested but not available/authenticated -- "
                  "falling back to simulate_satellite_row(). See satellite_features.py "
                  "docstring for setup.")
            use_live = False

    rows = []
    for city, (lat, lon) in city_coords.items():
        for year, month in year_months:
            if use_live:
                try:
                    feats = get_satellite_features_for_point(
                        lat, lon, _dt.date(year, month, 1)
                    )
                except Exception as exc:  # network hiccup, empty collection, etc.
                    print(f"[WARN] live satellite fetch failed for {city} "
                          f"{year}-{month:02d} ({exc}); using simulated value.")
                    feats = simulate_satellite_row(city, year, month)
                    feats["_source"] = "simulated"
                else:
                    feats["_source"] = "live"
            else:
                feats = simulate_satellite_row(city, year, month)
                feats["_source"] = "simulated"
            rows.append({"City": city, "Year": year, "Month": month, **feats})

    return pd.DataFrame(rows)