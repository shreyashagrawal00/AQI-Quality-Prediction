"""
Page 10 — About
System architecture, data sources, model documentation, folder structure.
"""

import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("ℹ️ About this Platform")
st.markdown(
    "**Satellite-based Surface AQI Estimation & HCHO Hotspot Detection Platform for India** — "
    "An AI-powered research platform combining satellite remote sensing, ground-station data, "
    "and machine learning."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏗️ Architecture", "📦 Data Sources", "🤖 ML Models", "📁 Folder Structure", "📚 References"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: System Architecture
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
### System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    SATELLITE DATA SOURCES                                ║
║                                                                          ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          ║
║  │  Sentinel-5P    │  │  MODIS / Terra  │  │  NASA FIRMS     │          ║
║  │  NO₂/SO₂/CO    │  │  AOD            │  │  Active Fire    │          ║
║  │  O₃/HCHO       │  │  Land Surface T │  │  VIIRS / MODIS  │          ║
║  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          ║
╚═══════════╪══════════════════════════════════════════╪═══════════════════╝
            │                                          │
            ▼                                          ▼
┌───────────────────────────┐            ┌─────────────────────────┐
│  Google Earth Engine      │            │  FIRMS Public API       │
│  (GEE) — Feature Extract  │            │  CSV Download           │
│  & Grid Sampling (0.25°)  │            │  (7-day rolling)        │
└───────────────┬───────────┘            └────────────┬────────────┘
                │                                     │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────┐
                  │  CPCB Ground Station Data  │
                  │  city_day.csv / stations   │
                  │  PM2.5, PM10, NO₂, AQI    │
                  └───────────────┬───────────┘
                                  │
                                  ▼
              ┌────────────────────────────────────┐
              │       DATA PREPROCESSING           │
              │  Imputation │ Clip Negatives        │
              │  Satellite Merge │ Time Features    │
              └────────────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │     MULTI-MODEL MACHINE LEARNING   │
              │                                    │
              │  ┌──────────┐  ┌──────────────┐   │
              │  │ XGBoost  │  │ LightGBM     │   │
              │  └──────────┘  └──────────────┘   │
              │  ┌──────────┐  ┌──────────────┐   │
              │  │ CatBoost │  │ Random Forest│   │
              │  └──────────┘  └──────────────┘   │
              │  → Auto-select best by RMSE        │
              │  → SHAP Explainability             │
              └────────────────────┬───────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
             AQI Prediction   HCHO Hotspot   Fire Analysis
             (0.25° grid)     (Z-score)      (NASA FIRMS)
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
              ┌────────────────────────────────────┐
              │    STREAMLIT INTERACTIVE DASHBOARD  │
              │                                    │
              │  10 Pages:                         │
              │  Home │ AQI Map │ Prediction       │
              │  HCHO │ Fire │ Trends │ Validation  │
              │  Analytics │ Rankings │ Report      │
              └────────────────────────────────────┘
```
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Data Sources
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    sources = pd.DataFrame({
        "Dataset": [
            "Sentinel-5P TROPOMI NO₂", "Sentinel-5P TROPOMI SO₂", "Sentinel-5P TROPOMI CO",
            "Sentinel-5P TROPOMI O₃", "Sentinel-5P TROPOMI HCHO",
            "MODIS MCD19A2 AOD", "MODIS MOD11A1 LST",
            "ERA5 Reanalysis", "CPCB city_day.csv", "CPCB stations.csv",
            "NASA FIRMS VIIRS S-NPP", "NASA FIRMS MODIS C6.1",
        ],
        "Provider": [
            "ESA/Copernicus","ESA/Copernicus","ESA/Copernicus","ESA/Copernicus","ESA/Copernicus",
            "NASA/USGS","NASA/USGS","ECMWF","CPCB India","CPCB India",
            "NASA EOSDIS","NASA EOSDIS",
        ],
        "GEE Collection / API": [
            "COPERNICUS/S5P/OFFL/L3_NO2","COPERNICUS/S5P/OFFL/L3_SO2",
            "COPERNICUS/S5P/OFFL/L3_CO","COPERNICUS/S5P/OFFL/L3_O3",
            "COPERNICUS/S5P/OFFL/L3_HCHO","MODIS/061/MCD19A2_GRANULES",
            "MODIS/061/MOD11A1","ECMWF/ERA5/DAILY",
            "Kaggle / data.gov.in","Kaggle / data.gov.in",
            "FIRMS API (free)","FIRMS API (free)",
        ],
        "Spatial Res.": [
            "3.5×5.5 km","3.5×5.5 km","7×3.5 km","3.5×5.5 km","3.5×5.5 km",
            "1 km","1 km","0.25°","Point","Point",
            "375 m","1 km",
        ],
        "Temporal Res.": [
            "Daily","Daily","Daily","Daily","Daily",
            "Daily","Daily","Daily (agr.)","Daily","Static",
            "NRT","NRT",
        ],
        "Key Variables": [
            "tropospheric_NO2_column_number_density","SO2_column_number_density_total",
            "CO_column_number_density","O3_column_number_density",
            "tropospheric_HCHO_column_number_density","Optical_Depth_047",
            "LST_Day_1km","temperature_2m, relative_humidity, u_component_of_wind",
            "PM2.5, PM10, NO, NO2, AQI, ...","StationId, City, State, Lat, Lon",
            "latitude, longitude, FRP, brightness","latitude, longitude, FRP, brightness",
        ],
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
### Machine Learning Pipeline

#### Feature Engineering
| Feature Group | Features |
|--------------|---------|
| **Ground Station** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene |
| **Satellite** | AOD (MODIS), NO₂ (Sentinel-5P), CO (Sentinel-5P) |
| **Temporal** | Year, Month, Day |
| **Spatial** | City encoding (label encoder) |

#### Models Trained
| Model | Hyperparameters | Library |
|-------|----------------|---------|
| **Random Forest** | n_estimators=200, max_depth=14 | scikit-learn |
| **XGBoost** | n_estimators=600, lr=0.05, max_depth=6, early stopping | xgboost |
| **LightGBM** | n_estimators=600, lr=0.05, num_leaves=63 | lightgbm |
| **CatBoost** | iterations=500, lr=0.05, depth=6 | catboost |

#### Model Selection
Auto-selects the model with lowest RMSE on the held-out 20% test split.
Saved as `models/best_model.pkl`.

#### Explainability
- **SHAP TreeExplainer**: Per-prediction waterfall charts
- **Global feature importance**: Mean absolute SHAP values over 500 samples
- Answers: *"Why was AQI predicted as X?"*

#### Forecasting
- 7-day horizon autoregressive XGBoost
- Features: lag₁…lag₇, rolling mean/std, month, day, city
- Confidence intervals widen with forecast horizon (bootstrap approach)
    """)

    # Show actual metrics if available
    import json
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        st.markdown("### Current Best Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", m.get("model", "Unknown"))
        c2.metric("MAE",  f"{m.get('MAE',0):.2f}")
        c3.metric("RMSE", f"{m.get('RMSE',0):.2f}")
        c4.metric("R²",   f"{m.get('R2',0):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Folder Structure
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
```
AQI2/
├── app.py                           # Home page (Streamlit entry point)
├── requirements.txt                 # All dependencies
├── .env                             # EE_PROJECT, API keys (git-ignored)
│
├── pages/                           # Streamlit multi-page app
│   ├── 1_🗺️_India_AQI_Map.py       # Satellite AQI heatmap over India
│   ├── 2_🔮_AQI_Prediction.py      # Per-city AQI prediction + SHAP
│   ├── 3_🌡️_HCHO_Hotspots.py      # HCHO hotspot detection & ranking
│   ├── 4_🔥_Biomass_Burning.py     # NASA FIRMS fire + HCHO overlay
│   ├── 5_📈_Trend_Analysis.py      # Temporal trend & seasonality
│   ├── 6_✅_Validation.py          # Model vs CPCB ground-truth
│   ├── 7_📊_Analytics.py           # Correlation matrix & statistics
│   ├── 8_🏆_State_Rankings.py      # State-level rankings
│   ├── 9_📄_Report.py              # PDF report generator
│   └── 10_ℹ️_About.py             # This page
│
├── src/                             # Backend Python modules
│   ├── aqi_utils.py                 # CPCB AQI category/color helpers
│   ├── city_coordinates.py          # City → (lat, lon) lookup
│   ├── data_preprocessing.py        # CSV loading, cleaning, imputation
│   ├── feature_engineering.py       # Feature matrix builder
│   ├── satellite_features.py        # GEE: MODIS AOD + S5P NO2/CO
│   ├── hcho_hotspots.py             # GEE: HCHO + burning anomaly
│   ├── multi_model.py               # Multi-model training (NEW)
│   ├── shap_explainer.py            # SHAP wrappers (NEW)
│   ├── firms_fire.py                # NASA FIRMS fire data (NEW)
│   ├── health_advisory.py           # Expanded health advisories (NEW)
│   ├── state_rankings.py            # State aggregation & rankings (NEW)
│   ├── report_generator.py          # PDF generation via fpdf2 (NEW)
│   ├── forecasting.py               # 7-day AQI forecast (NEW)
│   ├── train_model.py               # Legacy single-model trainer
│   ├── train_satellite_aqi_model.py # Satellite-only model trainer
│   └── fetch_satellite_features.py  # GEE feature fetch script
│
├── data/                            # Input datasets
│   ├── city_day.csv                 # CPCB daily AQI (primary)
│   ├── city_hour.csv                # CPCB hourly AQI
│   ├── station_day.csv              # Station-level daily
│   ├── station_hour.csv             # Station-level hourly
│   ├── stations.csv                 # Station metadata (City, State, Lat, Lon)
│   └── satellite_features.csv       # Pre-fetched GEE features (optional)
│
└── models/                          # Trained model artifacts
    ├── best_model.pkl               # Best model (auto-selected)
    ├── xgboost_model.pkl
    ├── random_forest_model.pkl
    ├── lightgbm_model.pkl
    ├── catboost_model.pkl
    ├── satellite_aqi_model.pkl      # Satellite-only AQI model
    ├── forecast_model.pkl           # 7-day forecast model
    ├── encoders.pkl                 # City label encoders
    ├── feature_columns.json
    ├── model_comparison.json        # All 4 models comparison
    ├── metrics.json                 # Best model metrics
    ├── feature_importance.json
    └── validation_data.npz          # y_test, y_pred for Validation page
```
    """)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: References
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
### References & Data Access

| Resource | URL |
|---------|-----|
| ISRO Problem Statement | [ISRO / MOSDAC AQI Challenge] |
| Sentinel-5P TROPOMI | https://sentinel.esa.int/web/sentinel/missions/sentinel-5p |
| MODIS AOD (MCD19A2) | https://lpdaac.usgs.gov/products/mcd19a2v061/ |
| Google Earth Engine | https://earthengine.google.com |
| NASA FIRMS API | https://firms.modaps.eosdis.nasa.gov/api/ |
| CPCB AQI Data | https://cpcb.nic.in / https://data.gov.in |
| ERA5 Reanalysis | https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5 |

### Python Libraries
- **scikit-learn** — Random Forest, preprocessing
- **XGBoost** — Gradient boosted trees
- **LightGBM** — Fast gradient boosting
- **CatBoost** — Categorical feature boosting
- **SHAP** — Shapley value explainability
- **earthengine-api** — Google Earth Engine Python API
- **Streamlit** — Interactive dashboard framework
- **Plotly** — Interactive charts and maps
- **fpdf2** — PDF report generation
- **pandas / numpy** — Data processing

### Citation
```
Platform: Satellite-based Surface AQI Estimation & HCHO Hotspot Detection
Data: CPCB India, ESA Copernicus, NASA MODIS, NASA FIRMS
ML Framework: XGBoost / LightGBM / CatBoost / scikit-learn
Visualization: Streamlit + Plotly
```
    """)
