# 🛰️ Satellite-based Surface AQI & HCHO Hotspot Analysis Platform (India)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-green.svg)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, research-grade, AI-driven platform that integrates **satellite remote sensing, active fire detection, CPCB ground-station data, and ensemble machine learning** to estimate surface Air Quality Index (AQI) and identify formaldehyde ($HCHO$) hotspots over the Indian landmass. 

This platform is specifically designed to address the **ISRO problem statement for satellite-derived surface AQI identification**, helping bridge the monitoring gap in regions lacking ground sensors.

---

## 📌 Table of Contents

1. [System Architecture](#-system-architecture)
2. [Scientific Methodologies](#-scientific-methodologies)
   - [Spatio-Temporal HCHO Anomaly Detection](#1-spatio-temporal-hcho-anomaly-detection)
   - [Autoregressive GBDT Forecasting](#2-autoregressive-gbdt-forecasting)
   - [Dual-Model Surface AQI Estimation](#3-dual-model-surface-aqi-estimation)
3. [Technology Stack](#-technology-stack)
4. [Directory Structure](#-directory-structure)
5. [Installation & Setup](#-installation--setup)
6. [Modes of Operation](#-modes-of-operation)
   - [Live Mode (Google Earth Engine & NASA FIRMS)](#1-live-mode)
   - [Simulated Mode (Deterministic Fallback)](#2-simulated-mode)
7. [Dashboard Pages Overview](#-dashboard-pages-overview)
8. [Data Ingestion & Cleaning Details](#-data-ingestion--cleaning-details)

---

## 🗺️ System Architecture

The following diagram illustrates the flow of data from remote sensing and ground station sources, through the preprocessing and ML modeling pipelines, to the interactive Streamlit user interface and PDF report exporter.

```mermaid
flowchart TD
    %% Data Sources
    subgraph Data Sources [Data Sources]
        A1[Sentinel-5P TROPOMI<br>tropospheric NO2, CO, HCHO]
        A2[MODIS Aqua/Terra<br>Optical Depth AOD]
        A3[NASA FIRMS API<br>Active Fire Brightness & FRP]
        A4[CPCB Ground Stations<br>PM2.5, PM10, SO2, O3, NO2, CO]
    end

    %% GEE Extraction
    subgraph GEE [Google Earth Engine Cloud]
        B1[Buffer Regions & Spatial Grid]
        B2[Image Collection Reducers]
    end

    %% Preprocessing
    subgraph Pipeline [Data Engineering & Modeling]
        C1[City Median Imputation &<br>Negative Value Clipping]
        C2[Feature Matrix Builder<br>Label Encoders]
        C3[Multi-Model GBDT Ensemble<br>RF | XGBoost | LightGBM | CatBoost]
        C4[Satellite-Only AQI Model<br>GBDT on AOD/NO2/CO]
        C5[Autoregressive Lag Model<br>lag_1 to lag_7 + Calendar Features]
    end

    %% Outputs & UI
    subgraph UI [Interactive Dashboard]
        D1[Streamlit Multi-page App]
        D2[Plotly & Mapbox Visuals]
        D3[SHAP Explainable AI Plots]
        D4[Automatic PDF Exporter<br>FPDF2 & Kaleido]
    end

    %% Connections
    A1 & A2 --> GEE
    GEE -->|Batch CSV / Live API| C2
    A3 -->|NASA CSV API| D2
    A4 -->|Ground Data CSVs| C1
    C1 --> C2
    C2 --> C3
    C2 --> C4
    C2 --> C5
    C3 -->|best_model.pkl| D1
    C4 -->|satellite_aqi_model.pkl| D1
    C5 -->|forecast_model.pkl| D1
    D1 --> D2 & D3 & D4
```

---

## 🔬 Scientific Methodologies

### 1. Spatio-Temporal HCHO Anomaly Detection
To distinguish seasonal biomass-burning plumes from persistent industrial pollution, the platform computes a pixel-wise spatio-temporal $Z$-score anomaly over a regular grid covering India:

$$\mu_{\text{burning}} = \frac{1}{|M_{\text{burning}}|} \sum_{m \in M_{\text{burning}}} HCHO_m$$

$$\mu_{\text{baseline}} = \frac{1}{|M_{\text{baseline}}|} \sum_{m \in M_{\text{baseline}}} HCHO_m$$

$$Z_i = \frac{\mu_{\text{burning}, i} - \mu_{\text{baseline}, i}}{\sigma_{\text{baseline}, i}}$$

Where:
- $M_{\text{burning}}$ defines the months of intense agricultural residue burning (typically October and November for post-monsoon crop residue burning in Punjab/Haryana; or April and May for pre-monsoon forest fires).
- $M_{\text{baseline}}$ represents the baseline months of minimal fire activity (e.g., monsoon and post-harvest winter months).
- $\sigma_{\text{baseline}, i}$ is the standard deviation of HCHO column densities at grid point $i$ during the baseline months.
- A grid point is flagged as a active **burning hotspot** if $Z_i \ge 1.5$.

### 2. Autoregressive GBDT Forecasting
The platform provides multi-day (1-day, 3-day, and 7-day) AQI forecasts for any monitored city. The forecasting system uses a recursive autoregressive Gradient Boosted Decision Tree (GBDT):

$$\widehat{AQI}_{t+k} = f\left(AQI_{t+k-1}, \dots, AQI_{t+k-7}, \text{Mean}(AQI_{t+k-7 \dots t+k-1}), \text{Std}(AQI_{t+k-7 \dots t+k-1}), \text{Year}, \text{Month}, \text{Day}, \text{CityId}\right)$$

Confidence intervals are calculated using bootstrap residuals over the previous 14-day rolling window, scaled by the square root of the step horizon to represent increasing uncertainty over time:

$$\text{Confidence Band} = \widehat{AQI}_{t+k} \pm 1.5 \times \sigma_{\text{residuals}} \times \sqrt{k}$$

### 3. Dual-Model Surface AQI Estimation
*   **Primary Ground-Satellite Model:** Trained on all CPCB ground pollutants ($PM_{2.5}, PM_{10}, SO_2,$ etc.) and calendar variables, and enriched with matched satellite proxies (MODIS AOD, Sentinel-5P $NO_2$ and $CO$ columns) at the city-month level. This achieves high prediction accuracy ($R^2 > 0.85$) at locations where monitors are active.
*   **Satellite-Only AQI Model:** To map surface AQI across the entire Indian mainland (including remote and unmonitored rural areas), a second model is trained *exclusively* on satellite observables (`satellite_aod`, `satellite_no2`, `satellite_co`) and calendar `Month`. It trades off some localized accuracy to gain 100% spatial coverage.

---

## 🛠️ Technology Stack

*   **User Interface:** [Streamlit](https://streamlit.io/) (with a customized dark-mode glassmorphic theme injected globally via [_theme.py](file:///s:/AQI-Quality-Prediction/src/_theme.py)).
*   **Geospatial Processing:** [Google Earth Engine API](https://earthengine.google.com/) (`earthengine-api`) for automated spatial polygon buffering and multitemporal satellite image reduction.
*   **Machine Learning:** 
    *   [XGBoost](https://xgboost.readthedocs.io/) & [LightGBM](https://lightgbm.readthedocs.io/) (Primary regressors).
    *   [CatBoost](https://catboost.ai/) (Alternative tree model, handles categorical features natively).
    *   [Scikit-learn](https://scikit-learn.org/) (Data splitting, metrics evaluation, and fallback Gradient Boosting Regressors).
*   **Explainable AI:** [SHAP (Shapley Additive exPlanations)](https://shap.readthedocs.io/) for local feature contribution waterfall charts and global feature importances.
*   **Active Fires:** [NASA FIRMS API](https://firms.modaps.eosdis.nasa.gov/api/area/) for downloading real-time MODIS and VIIRS active fire detections.
*   **Report Generation:** [FPDF2](https://pyfpdf.github.io/fpdf2/) for clean vector-drawn PDF layouts and [Kaleido](https://github.com/plotly/kaleido) for exporting Plotly interactive maps and charts to high-resolution PNG images.

---

## 📁 Directory Structure

```
AQI-Quality-Prediction/
│
├── app.py                           # Dashboard landing page / Home view
├── requirements.txt                 # Python project dependencies
├── .env                             # EE_PROJECT, FIRMS_MAP_KEY (git-ignored)
├── .env.example                     # Reference template for configuration
│
├── pages/                           # Streamlit Multi-Page dashboard
│   ├── 1_🗺️_India_AQI_Map.py       # India-wide predicted AQI surface map
│   ├── 2_🔮_AQI_Prediction.py      # Per-city ML prediction, SHAP waterfall, 7-day forecast
│   ├── 3_🌡️_HCHO_Hotspots.py      # Spatio-temporal GEE HCHO column density mapping
│   ├── 4_🔥_Biomass_Burning.py     # Overlays active NASA fire points on HCHO maps
│   ├── 5_📈_Trend_Analysis.py      # Aggregated time-series trend & seasonal analysis
│   ├── 6_✅_Validation.py          # Scatter plots, residuals, & errors vs CPCB ground-truth
│   ├── 7_📊_Analytics.py           # Correlation matrices & pollutant statistics
│   ├── 8_🏆_State_Rankings.py      # Aggregated air quality rankings by state
│   ├── 9_📄_Report.py              # Export configurable PDF reports of analysis
│   └── 10_ℹ️_About.py             # Methodology, data sources & scientific guides
│
├── src/                             # Backend pipeline modules
│   ├── _theme.py                    # CSS injector for Streamlit custom UI styling
│   ├── aqi_utils.py                 # CPCB AQI breakpoints, category, and coloring rules
│   ├── city_coordinates.py          # Coordinates lookup dictionary for 26+ major Indian cities
│   ├── data_preprocessing.py        # CPCB loading, city-median imputation, stations.csv joiner
│   ├── feature_engineering.py       # Encodes categorical features, creates train/test matrices
│   ├── fetch_satellite_features.py  # Script to batch download GEE parameters into local CSV
│   ├── satellite_features.py        # Live GEE interface for MODIS AOD and Sentinel-5P
│   ├── hcho_hotspots.py             # Spatio-temporal HCHO anomaly calculations
│   ├── firms_fire.py                # NASA FIRMS API handler and active fire density gridder
│   ├── forecasting.py               # Autoregressive lag model training and predictions
│   ├── health_advisory.py           # Clinical effects and advisories mapped to AQI categories
│   ├── multi_model.py               # Automated trainer comparing RF, XGBoost, LGBM, & CatBoost
│   └── train_satellite_aqi_model.py # Trainer for the satellite-only nationwide AQI model
│
├── data/                            # CPCB CSV datasets and cached satellite features
└── models/                          # Saved checkpoints (best_model.pkl, encoders.pkl, etc.)
```

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Environment
First, ensure you have Python 3.9 or higher installed. Navigate to the project root directory and install dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # On Windows

# Install required packages
pip install -r requirements.txt
```

### 2. Download Datasets
Place the official CPCB air quality csv files (obtainable from Kaggle or official portal) into the `data/` directory:
- `city_day.csv` (Essential)
- `stations.csv` (Essential)
- `station_day.csv` (Optional, for station-level views)
- `city_hour.csv` & `station_hour.csv` (Optional)

### 3. Generate Satellite Features
Run the satellite feature downloader to prepare the GEE cache file (`data/satellite_features.csv`). If you do not have Earth Engine set up yet, this will safely auto-simulate realistic data:

```bash
python src/fetch_satellite_features.py
```

### 4. Train Models
Run the multi-model trainer to fit Random Forest, XGBoost, LightGBM, and CatBoost models. The script will compare performance and auto-save the winner as `best_model.pkl`:

```bash
python src/multi_model.py
```

Next, train the satellite-only map regressor and the autoregressive forecasting model:

```bash
python src/train_satellite_aqi_model.py
python src/forecasting.py
```

### 5. Launch dashboard
```bash
streamlit run app.py
```

---

## 🔌 Modes of Operation

The platform dynamically adjusts between two execution modes depending on credentials and API configurations.

### 1. Live Mode
To fetch live, daily, monthly, or seasonal satellite columns directly from ESA and NASA, configure the following:

#### Google Earth Engine (GEE):
Authenticate your system to cache local credentials:
```bash
earthengine authenticate
```
Ensure you have a billing-enabled Google Cloud Project (required by GEE since late 2024). Add your project ID to a `.env` file in the root directory:
```env
EE_PROJECT="your-google-cloud-project-id"
```
Alternatively, set service account credentials for headless server deployments:
```env
EE_SERVICE_ACCOUNT="name@project.iam.gserviceaccount.com"
EE_SERVICE_ACCOUNT_KEY="path/to/key.json"
```

#### NASA FIRMS (Active Fire API):
Obtain a free MAP key from [NASA FIRMS API Portal](https://firms.modaps.eosdis.nasa.gov/api/area/) and append it to your `.env` file:
```env
FIRMS_MAP_KEY="your_nasa_firms_map_key_here"
```

### 2. Simulated Mode
If GEE credentials or `FIRMS_MAP_KEY` are not detected in the environment, the platform automatically enters a **simulated fallback mode**. 
- Functions generate deterministic, physically realistic spatial-temporal profiles based on latitude, longitude, and calendar seasonality.
- The Indo-Gangetic Plain (Punjab, Haryana, Delhi, Western UP) receives crop-residue stubble burning bumps in October-November.
- Central and Eastern India (Odisha, Chhattisgarh) receive fire events during pre-monsoon dry season (March–May).
- Allows all parts of the dashboard, ML inference pipelines, and PDF reporting scripts to run and print without errors.

---

## 🛰️ Dashboard Pages Overview

### 🏡 Home Page (`app.py`)
Provides a dashboard landing page displaying nationwide summary stats (monitored cities, historical dates, and current national averages). Contains an interactive tabs layout illustrating daily AQI trends, a ranking of the top 25 most polluted cities, a system architecture layout, and details about target dataset properties.

### 1. 🗺️ India Surface AQI Map (`1_🗺️_India_AQI_Map.py`)
Computes predicted surface AQI over a regular grid covering India using the **Satellite-Only model**. Generates an interactive Plotly Mapbox visualization. The user can adjust grid resolution (0.25° grid for quick rendering or 0.1° grid for high-resolution evaluation).

### 2. 🔮 City AQI & Forecasting (`2_🔮_AQI_Prediction.py`)
Interactive ML prediction console:
- Select any city and input pollutant concentrations to compute local predicted AQI.
- **Explainable AI (SHAP):** Renders a waterfall chart displaying the contribution of each pollutant feature relative to the dataset baseline.
- **7-Day Forecast:** Runs the recursive autoregressive model to plot the city's expected AQI over the next 7 days, complete with bootstrap uncertainty bands.

### 3. 🌡️ HCHO Hotspots (`3_🌡️_HCHO_Hotspots.py`)
Pulls tropospheric HCHO column densities via Sentinel-5P. Highlights locations where formaldehyde levels spike. Users can select temporal aggregations: daily, weekly, monthly, seasonal, or annual.

### 4. 🔥 Biomass Burning (`4_🔥_Biomass_Burning.py`)
Downloads NASA FIRMS active fire pixels (VIIRS or MODIS) for a specified day window and overlay-plots them alongside HCHO column densities. Helps scientists isolate HCHO hotspots triggered by agricultural stubble burning or forest fires versus chemical industrial emissions.

### 5. 📈 Trend & Seasonality (`5_📈_Trend_Analysis.py`)
Performs time-series analysis on CPCB ground data. Plots monthly seasonality curves, multi-year trends, and lets users compare pollutant histories between multiple cities simultaneously.

### 6. ✅ Validation Console (`6_✅_Validation.py`)
Provides model performance verification:
- Draws Scatter Plots of Actual vs. Predicted AQI with $\pm 20\%$ error bands.
- Residual plots showing prediction errors across different AQI magnitudes.
- Per-category error analysis tables illustrating metrics like MAE, RMSE, R², and MAPE for each CPCB category (Good, Moderate, Severe, etc.).

### 7. 📊 Analytics (`7_📊_Analytics.py`)
Computes correlation matrices (Pearson or Spearman) between all CPCB ground pollutants. Includes distributions, histogram plots, and basic scientific statistics (means, standard deviations, percentiles) for the dataset.

### 8. 🏆 State Rankings (`8_🏆_State_Rankings.py`)
Computes aggregated rankings by Indian state:
- Worst average AQI states.
- Cleanest air states.
- States with the highest active fire count and total Fire Radiative Power (FRP).
- Year-on-Year (YoY) improvements.

### 9. 📄 Export Report (`9_📄_Report.py`)
Generates customizable PDF reports. Users can select which sections to include (metrics tables, charts, maps), insert custom analyst notes, and download the resulting file as a print-ready vector PDF powered by `fpdf2`.

---

## 📦 Data Ingestion & Cleaning Details

CPCB datasets contain numerous anomalies that the platform resolves automatically during ingestion:
1. **Dimension Mismatches:** Station-level datasets do not contain a default `City` column. The platform joins them with `stations.csv` by `StationId` to recover city/state mapping.
2. **Missing Observations:** Rather than dropping missing rows (which would delete over 85% of CPCB data), the platform imputes values using city-specific medians, falling back to nationwide medians for coordinates with completely missing columns.
3. **Physical Impossibilities:** Negative pollutant readings (instrument calibration anomalies) are clipped to $0.0$ to prevent model corruption.
4. **Data Leakage:** Columns describing AQI classifications (such as `AQI_Bucket` or categorical levels) are dropped prior to feature engineering, as they represent direct deterministic functions of the target label.

---

*This platform provides a complete, scientific end-to-end implementation for estimating surface air quality parameters, enabling data-backed environmental monitoring across India.*
