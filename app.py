"""
app.py — Home Page
Satellite-based Surface AQI & HCHO Hotspot Analysis Platform for India
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_preprocessing import load_city_data, add_time_features
from health_advisory import aqi_to_category, health_message, get_advisory
from satellite_features import init_earth_engine, get_ee_last_error

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="India AQI & HCHO Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .metric-card {
    background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
    border-radius: 12px; padding: 1.2rem; color: white;
    box-shadow: 0 4px 15px rgba(30,58,138,0.3);
    text-align: center;
  }
  .metric-card h1 { font-size: 2rem; margin:0; font-weight:700; }
  .metric-card p  { margin:0; opacity:.8; font-size:.85rem; }
  .category-pill {
    display:inline-block; padding:0.3rem 1rem; border-radius:999px;
    font-weight:600; font-size:.9rem; margin:0.2rem;
  }
  .stTabs [data-baseweb="tab"] { font-size:0.95rem; font-weight:500; }
  .hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #1d4ed8 100%);
    border-radius:16px; padding:2.5rem 2rem; color:white; margin-bottom:1.5rem;
  }
  .hero-banner h1 { font-size:2.2rem; font-weight:700; margin-bottom:0.5rem; }
  .hero-banner p  { opacity:.85; font-size:1.05rem; max-width:700px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading ground station data…")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    df = add_time_features(df)
    df["Category"] = df["AQI"].apply(aqi_to_category)
    return df


@st.cache_resource(show_spinner=False)
def get_ee_status():
    return init_earth_engine()


@st.cache_data(show_spinner=False)
def get_model_comparison():
    path = os.path.join(MODEL_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # fallback to legacy metrics.json
    path2 = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path2):
        with open(path2) as f:
            m = json.load(f)
        return {"best_model": m.get("model", "XGBoost"), "comparison": [m]}
    return None


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Emblem_of_India.svg/120px-Emblem_of_India.svg.png", width=60)
    st.markdown("## 🛰️ AQI Platform")
    st.caption("Satellite-based Surface AQI & HCHO Hotspot Analysis")
    st.divider()

    ee_ready = get_ee_status()
    if ee_ready:
        st.success("🌍 Earth Engine: Connected")
    else:
        st.warning("🌍 Earth Engine: Offline (simulated data)")
        with st.expander("How to connect"):
            st.code("earthengine authenticate\n# set EE_PROJECT=your-project-id in .env")

    cmp = get_model_comparison()
    if cmp:
        st.info(f"🏆 Best model: **{cmp['best_model']}**")
    else:
        st.warning("⚠️ No model trained yet.\n\nRun:\n```\npython src/multi_model.py\n```")

    st.divider()
    st.caption("Navigate using the **Pages** menu above ↑")


# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🛰️ Satellite-based Surface AQI & HCHO Hotspot Analysis</h1>
  <p>
    AI-powered platform combining <strong>CPCB ground monitoring</strong>,
    <strong>Sentinel-5P satellite observations</strong>, and
    <strong>NASA FIRMS fire data</strong> to predict surface Air Quality Index
    and detect formaldehyde (HCHO) hotspots across India.
  </p>
</div>
""", unsafe_allow_html=True)


# ── Key metrics ──────────────────────────────────────────────────────────────
df = get_city_data()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="metric-card">
      <p>Cities Covered</p><h1>{df['City'].nunique()}</h1>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
      <p>Total Records</p><h1>{len(df)//1000}K</h1>
    </div>""", unsafe_allow_html=True)
with col3:
    yr_range = f"{df['Date'].min().year}–{df['Date'].max().year}"
    st.markdown(f"""<div class="metric-card">
      <p>Date Range</p><h1>{yr_range}</h1>
    </div>""", unsafe_allow_html=True)
with col4:
    avg_aqi = df['AQI'].mean()
    cat = aqi_to_category(avg_aqi)
    adv = get_advisory(avg_aqi)
    st.markdown(f"""<div class="metric-card">
      <p>Avg National AQI</p><h1>{avg_aqi:.0f}</h1>
      <p>{cat}</p>
    </div>""", unsafe_allow_html=True)
with col5:
    worst_city = df.groupby("City")["AQI"].mean().idxmax()
    st.markdown(f"""<div class="metric-card">
      <p>Most Polluted City</p><h1 style="font-size:1.2rem">{worst_city}</h1>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 National AQI Trend", "🏙️ City Overview", "🔬 System Architecture", "📦 Data Sources"])

with tab1:
    trend = df.groupby("Date", as_index=False)["AQI"].mean()
    fig = px.line(
        trend, x="Date", y="AQI",
        title="Average Daily AQI — All Cities (India)",
        color_discrete_sequence=["#3b82f6"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb"),
        xaxis=dict(gridcolor="#e5e7eb"),
        height=400,
    )
    # add background shading for AQI zones
    for low, high, color, label in [
        (0, 50, "rgba(0,228,0,0.08)", "Good"),
        (51, 100, "rgba(163,255,0,0.08)", "Satisfactory"),
        (101, 200, "rgba(255,255,0,0.08)", "Moderate"),
        (201, 300, "rgba(255,126,0,0.08)", "Poor"),
    ]:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0,
                      annotation_text=label, annotation_position="right")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    city_avg = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI", ascending=True)
    city_avg["Color"] = city_avg["AQI"].apply(lambda x: get_advisory(x)["color"])
    fig2 = px.bar(
        city_avg.tail(25), x="AQI", y="City", orientation="h",
        title="Average AQI by City (Top 25 Most Polluted)",
        color="AQI", color_continuous_scale="YlOrRd",
        height=650,
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### System Architecture")
    st.markdown("""
```
╔═══════════════════════════════════════════════════════════════╗
║              SATELLITE DATA SOURCES                           ║
║  Sentinel-5P (NO₂/SO₂/CO/O₃/HCHO) │ MODIS (AOD)             ║
║  ERA5 (Meteo)  │  NASA FIRMS (Fire) │  CPCB Ground Stations   ║
╚════════════════════════╤══════════════════════════════════════╝
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Google Earth Engine (GEE)   │
         │   Feature Extraction & Grid   │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Data Preprocessing & Merging │
         │  Imputation │ Normalization   │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Multi-Model Machine Learning │
         │  RF │ XGBoost │ LGBM │ CatBoost│
         │  SHAP Explainability          │
         └───────────────┬───────────────┘
                         │
                    ┌────┴────┐
                    │         │
                    ▼         ▼
             AQI Prediction  HCHO Hotspot
             (0.25° grid)    Detection
                    │         │
                    └────┬────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  Streamlit Interactive Dashboard│
         │  Maps │ Trends │ Rankings │ PDF│
         └───────────────────────────────┘
```
    """)

with tab4:
    sources = pd.DataFrame({
        "Dataset": ["Sentinel-5P TROPOMI", "MODIS MCD19A2", "ERA5 Reanalysis", "CPCB AQI Data",
                    "NASA FIRMS VIIRS", "Sentinel-2 MSI"],
        "Provider": ["ESA/Copernicus", "NASA/USGS", "ECMWF", "CPCB India",
                     "NASA EOSDIS", "ESA/Copernicus"],
        "Variables": ["NO₂, SO₂, CO, O₃, HCHO", "Aerosol Optical Depth", "Temp, RH, Wind, Pressure",
                      "PM2.5, PM10, NO₂, AQI", "Active Fire Points (FRP)", "NDVI, Land Cover"],
        "Spatial Res.": ["3.5 × 5.5 km", "1 km", "0.25°", "Point stations", "375 m (VIIRS)", "10 m"],
        "Temporal Res.": ["Daily", "Daily", "Hourly", "Daily/Hourly", "Near-real-time", "5 days"],
        "Access": ["Google Earth Engine", "Google Earth Engine", "Google Earth Engine",
                   "data.gov.in / Kaggle", "FIRMS API (free)", "Google Earth Engine"],
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

# ── AQI Category legend ───────────────────────────────────────────────────────
st.markdown("### AQI Category Reference (CPCB India)")
cols = st.columns(6)
for i, (cat, rng, color) in enumerate([
    ("Good", "0–50", "#00e400"), ("Satisfactory", "51–100", "#a3ff00"),
    ("Moderate", "101–200", "#ffff00"), ("Poor", "201–300", "#ff7e00"),
    ("Very Poor", "301–400", "#ff0000"), ("Severe", "401+", "#8f3f97"),
]):
    with cols[i]:
        st.markdown(
            f"<div style='background:{color}33;border:2px solid {color};border-radius:8px;"
            f"padding:0.6rem;text-align:center'>"
            f"<b style='color:{color}'>{cat}</b><br><small>{rng}</small></div>",
            unsafe_allow_html=True,
        )