# Surface AQI Estimation & HCHO Hotspot Detection

AI-powered system to predict AQI from CPCB ground-station data and
visualize Sentinel-5P HCHO hotspots over India, per the project guide.

## What was actually in the uploaded repo

The GitHub repo zip
(`Development-of-surface-AQI-Identification-of-HCHO-Hotspots-over-India-using-Satellite-Data_-main.zip`)
contained **only documentation** (`docs/*.md`) — most files were empty
stubs, and there was **no source code at all**. So there was nothing to
"fix" in the literal sense (no code ran, let alone had bugs); instead
this delivers the implementation the docs describe, end to end, and I
proactively avoided the specific bugs described below because they're
the ones most first-pass attempts at this exact dataset hit.

## Project structure

```
aqi_project/
├── app.py                     # Streamlit dashboard (6 pages, per the guide)
├── requirements.txt
├── data/                      # put city_day.csv, city_hour.csv, station_day.csv,
│                               # station_hour.csv, stations.csv here (from archive.zip)
│                               # satellite_features.csv is generated, not required upfront
├── src/
│   ├── aqi_utils.py             # CPCB AQI -> category/color/health-message
│   ├── data_preprocessing.py    # load + clean city/station CSVs; attach_satellite_features()
│   ├── feature_engineering.py   # build model-ready X/y (incl. optional satellite cols)
│   ├── train_model.py           # trains + evaluates + saves XGBoost model
│   ├── city_coordinates.py      # lat/lon centroids for CPCB cities (satellite queries need a point)
│   ├── satellite_features.py    # MODIS AOD + Sentinel-5P NO2/CO -> AQI predictors (live or simulated)
│   ├── fetch_satellite_features.py  # CLI: builds data/satellite_features.csv once
│   └── hcho_hotspots.py         # Sentinel-5P HCHO: single-period map + biomass-burning anomaly detection
└── models/                     # created by train_model.py: aqi_model.pkl,
                                 # encoders.pkl, metrics.json, feature_importance.json
```

## Setup

```bash
cd aqi_project
python3 -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. unzip your dataset into data/  (city_day.csv, city_hour.csv,
#    station_day.csv, station_hour.csv, stations.csv)

# 2. (optional but addresses the "surface AQI from satellite data" part
#    of the problem statement) build satellite AOD/NO2/CO features --
#    works with or without Earth Engine credentials (see below):
python src/fetch_satellite_features.py

# 3. train the model (writes models/aqi_model.pkl etc.; automatically
#    picks up data/satellite_features.csv if step 2 was run)
python src/train_model.py

# 4. launch the dashboard
streamlit run app.py
```

I validated the full pipeline (`data_preprocessing.py` →
`feature_engineering.py` → `train_model.py`) against your actual
`archive__1_.zip` data in a sandbox: it cleaned 24,850 city-day rows
with zero remaining NaNs and trained successfully
(R² ≈ 0.90 on a held-out test set, using a GradientBoosting fallback
since this sandbox has no internet access to install xgboost — your
own environment should install `xgboost` per `requirements.txt` for
the model the guide specifies).

## Bugs this implementation avoids (things a naive first pass hits on this exact dataset)

1. **Data leakage** — `AQI_Bucket` is a deterministic function of `AQI`.
   Using it as an input feature while predicting `AQI` leaks the
   target. It's excluded from `X` entirely; category is only ever
   derived *after* prediction, in `aqi_utils.aqi_to_category()`.
2. **Losing 85%+ of rows to `dropna()`** — pollutant columns like
   `Xylene`/`NH3` are missing in a large fraction of rows. Dropping any
   row with a missing pollutant discards most of the dataset.
   `data_preprocessing.py` instead imputes per-city medians (falling
   back to the global median), and only drops rows missing the
   **target** (`AQI`), since that can't be legitimately imputed.
3. **`station_hour.csv` has no `City` column** — it only has
   `StationId`. Concatenating it directly with `city_hour.csv` silently
   produces an all-NaN City column for every station row.
   `load_station_data()` joins `stations.csv` on `StationId` to recover
   `City`/`State` correctly.
4. **Negative pollutant readings** — sensor artifacts occasionally
   report negative concentrations, which are physically impossible;
   these are clipped to 0 before training.
5. **Unencoded categoricals reused inconsistently at inference** — the
   `LabelEncoder` for `City`/`StationId` is fit once during training and
   saved (`models/encoders.pkl`); the dashboard reuses the *same*
   encoder at inference time (with an unseen-category fallback) instead
   of re-fitting, which would silently scramble the numeric mapping.
6. **Overfitting from a fixed, large `n_estimators`** — `train_model.py`
   uses an eval set + early stopping (when xgboost is available) rather
   than a fixed tree count.
7. **Re-loading 65–220MB CSVs on every Streamlit interaction** —
   Streamlit reruns the whole script on every widget interaction;
   without `@st.cache_data`/`@st.cache_resource` the dashboard would
   reload and re-clean the full dataset every click. Both are applied
   in `app.py`.

## Satellite data: surface AQI predictors + HCHO hotspots

Two things needed real satellite data wired in, not just ground-station
data, to actually match the problem statement:

### 1. Satellite-derived AQI predictors (AOD, NO2, CO)

`src/satellite_features.py` pulls MODIS AOD (`MODIS/061/MCD19A2_GRANULES`)
and Sentinel-5P NO2/CO column densities via Earth Engine for each CPCB
city (`src/city_coordinates.py` gives each city a lat/lon centroid, since
the CPCB CSVs only have city names). These become extra model features
(`satellite_aod`, `satellite_no2`, `satellite_co`) — this is what lets the
model generalize beyond ground-monitored cities, since a grid point
anywhere in India has satellite coverage even where it has no CPCB station.

```bash
pip install earthengine-api
earthengine authenticate                 # one-time; or set
# export EE_SERVICE_ACCOUNT=...@...iam.gserviceaccount.com
# export EE_SERVICE_ACCOUNT_KEY=/path/to/key.json

python src/fetch_satellite_features.py    # writes data/satellite_features.csv
python src/train_model.py                 # auto-merges it in if present
```

Without Earth Engine installed/authenticated, `fetch_satellite_features.py`
still runs and writes `data/satellite_features.csv`, but with clearly
labeled *simulated* values (`_source` column says `simulated` vs `live`) so
the training/merge pipeline can be demoed end-to-end even without
credentials. **Re-run the script after authenticating** to replace those
with real pixels — nothing else needs to change.
`models/metrics.json` records `used_satellite_features: true/false`, and
the dashboard's Model Performance page surfaces this.

### 2. HCHO hotspots — single-period map + biomass-burning anomaly detection

The **HCHO Hotspot Map** page (`src/hcho_hotspots.py`) now has two views:

- **Single-period hotspot map** — mean Sentinel-5P HCHO over a chosen date
  range, same idea as before but wired to real data when Earth Engine is
  available.
- **Biomass-burning anomaly map** — the actual spatio-temporal ask in the
  problem statement: for a grid of points over India, it compares each
  point's mean HCHO during known burning months (default: Oct/Nov
  post-monsoon stubble burning) against that *same point's* baseline
  (non-burning months), and flags points with a z-score above a threshold.
  This distinguishes a genuine burning-linked hotspot from a
  permanently-elevated urban/industrial pixel.

Both views auto-detect Earth Engine: live if authenticated (same setup as
above), otherwise a clearly-labeled simulated grid with a realistic
Indo-Gangetic-Plain Oct/Nov bump baked in, so the anomaly logic and UI can
still be demoed without credentials. Re-authenticate and relaunch — no
code changes needed to switch to live data.

### Notes / things to check before presenting

- **city_coordinates.py has city centroids, not exact CPCB station
  coordinates** — fine for a 25km satellite sampling buffer, but say so if
  asked; add real station lat/lon if you have it for tighter accuracy.
- **The Oct/Nov default burning window is a starting point, not a fixed
  truth** — it's editable in the dashboard (`multiselect`), and you may
  want to add the Apr/May pre-monsoon window for certain regions/years.
- **`get_hcho_grid_timeseries` samples a 22×22 grid** (`india_grid()`),
  not a full-resolution raster — enough to show the pattern in a
  dashboard; increase `n_lat`/`n_lon` for a finer map if Earth Engine
  quota allows.

## Notes on `station_hour.csv`

At 220MB, loading the full hourly station file directly with
`pd.read_csv` works but is slow. `train_model.py` trains on
`city_day.csv` by default (fast, ~25k rows). To train on hourly/station
granularity, call `train(data_dir, granularity="hour")` — consider
adding `nrows=` or chunked reading in `data_preprocessing.py` if memory
is constrained.
