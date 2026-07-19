# Satellite-based Surface AQI & HCHO Hotspot Analysis Platform for India

An advanced, research-grade platform that combines satellite remote sensing, ground-station data, active fire detections, and machine learning to estimate surface Air Quality Index (AQI) and identify formaldehyde (HCHO) hotspots over the Indian landmass.

This project is aligned with the ISRO problem statement for satellite-derived surface AQI identification.

---

## 🚀 Key Features

1. **India-wide Surface AQI Heatmap**
   - Configurable grid (0.25° fast / 0.1° high-res) across the Indian mainland.
   - Predictions generated using satellite features (MODIS AOD, Sentinel-5P NO₂/CO) to cover unmonitored regions.
   - Interactive zoom/pan map using Plotly Mapbox.

2. **NASA FIRMS Active Fire Integration**
   - Live downloads of active fire points (VIIRS/MODIS) within the India bounding box using NASA's public CSV API.
   - Real-time Fire Radiative Power (FRP) and daily fire count statistics.
   - Simultaneous HCHO + Fire activity overlay map to isolate biomass-burning hotspots.

3. **Formaldehyde (HCHO) Hotspot Analysis**
   - S5P TROPOMI HCHO column densities via Google Earth Engine.
   - Multi-resolution analysis: Daily, Weekly, Monthly, Seasonal, and Annual averages.
   - Spatio-temporal anomaly detection (Z-score, standard deviation, percentile thresholding) comparing burning seasons against baseline months.

4. **Explainable AI (SHAP)**
   - Integrated SHAP (Shapley Additive exPlanations) values.
   - Local waterfall charts: *"Why was this specific AQI predicted as 214?"*
   - Global feature importance analysis (mean absolute SHAP).

5. **Multi-Model Machine Learning**
   - Automated training and evaluation of: Random Forest, XGBoost, LightGBM, and CatBoost.
   - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score, and Mean Absolute Percentage Error (MAPE).
   - Auto-selection of the best performing model.

6. **Model Validation & Scientific Analytics**
   - CPCB actual vs. predicted AQI validation.
   - Scatter plots with ±20% bands, residuals, error distributions, and per-category error.
   - Scientific statistics (percentiles, bias) and Pearson correlation matrix for all pollutants.

7. **7-Day AQI Forecasting**
   - Autoregressive GBDT forecasting with lag features ($t-1$ to $t-7$) and rolling statistics.
   - 1-day, 3-day, and 7-day horizons for any city with bootstrap-derived uncertainty bands.

8. **State Rankings & Health Advisories**
   - State-level rankings for: Worst AQI, Cleanest Air, Top HCHO, Highest Fire, and YoY Improved States.
   - Detailed Indian CPCB health advisories with specific clinical effects and precautions.

9. **Automatic PDF Report Generator**
   - In-browser configurable PDF export powered by `fpdf2` and `kaleido`.
   - Embeds metrics, tables, and Plotly maps/charts.

---

## 📁 Directory Structure

```
AQI2/
├── app.py                           # Home page (Streamlit entry point)
├── requirements.txt                 # All dependencies
├── .env                             # EE_PROJECT, API keys (git-ignored)
│
├── pages/                           # Streamlit multi-page app
│   ├── 1_🗺️_India_AQI_Map.py       # Satellite AQI heatmap over India
│   ├── 2_🔮_AQI_Prediction.py      # Per-city AQI prediction + SHAP + Forecast
│   ├── 3_🌡️_HCHO_Hotspots.py      # HCHO hotspot detection & ranking
│   ├── 4_🔥_Biomass_Burning.py     # NASA FIRMS fire + HCHO overlay
│   ├── 5_📈_Trend_Analysis.py      # Temporal trend & seasonality
│   ├── 6_✅_Validation.py          # Model vs CPCB ground-truth
│   ├── 7_📊_Analytics.py           # Correlation matrix & statistics
│   ├── 8_🏆_State_Rankings.py      # State-level rankings
│   ├── 9_📄_Report.py              # PDF report generator
│   └── 10_ℹ️_About.py             # About page / system docs
│
├── src/                             # Backend Python modules
│   ├── aqi_utils.py                 # CPCB AQI category/color helpers
│   ├── city_coordinates.py          # City → (lat, lon) lookup
│   ├── data_preprocessing.py        # CSV loading, cleaning, imputation
│   ├── feature_engineering.py       # Feature matrix builder
│   ├── satellite_features.py        # GEE: MODIS AOD + S5P NO2/CO
│   ├── hcho_hotspots.py             # GEE: HCHO + burning anomaly
│   ├── multi_model.py               # Multi-model GBDT training
│   ├── shap_explainer.py            # SHAP TreeExplainer wrappers
│   ├── firms_fire.py                # NASA FIRMS active fire
│   ├── health_advisory.py           # Detailed health advisories
│   ├── state_rankings.py            # State aggregation & rankings
│   ├── report_generator.py          # PDF generation via fpdf2
│   ├── forecasting.py               # Autoregressive GBDT forecasting
│   ├── train_model.py               # Single-model GBDT trainer
│   ├── train_satellite_aqi_model.py # Satellite-only model trainer
│   └── fetch_satellite_features.py  # GEE feature fetch script
│
├── data/                            # CPCB Ground data & fetched features
└── models/                          # Trained model checkpoints
```

---

## 🛠️ Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download CPCB Ground Data**
   Place `city_day.csv` and `stations.csv` (from the CPCB Kaggle/official dataset) in the `data/` directory.

3. **Train Models**
   Run the multi-model GBDT training script:
   ```bash
   python src/multi_model.py
   ```
   This will train Random Forest, LightGBM, and CatBoost models, and save `best_model.pkl` along with validation data.

4. **Train Satellite-only and Forecast Models**
   ```bash
   python src/train_satellite_aqi_model.py
   python src/forecasting.py
   ```

5. **Authenticate Google Earth Engine (Optional but Recommended)**
   To pull live Sentinel-5P HCHO/pollutant maps and MODIS AOD:
   ```bash
   earthengine authenticate
   ```
   If using a modern billing-enabled Earth Engine project, configure `EE_PROJECT=your-project-id` in a `.env` file in the root directory.
   *Without authentication, the platform runs in **simulated fallback mode** using deterministic, realistic spatial-temporal profiles.*

6. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```
