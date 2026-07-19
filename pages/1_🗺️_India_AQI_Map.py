"""
Page 1 — India AQI Map (Satellite-derived)
Interactive heatmap of predicted AQI over India using satellite features.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from satellite_features import init_earth_engine, get_ee_last_error, build_grid_satellite_table
from hcho_hotspots import india_grid
from health_advisory import aqi_to_category, get_advisory
from state_rankings import _latlon_to_state

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="India AQI Map", page_icon="🗺️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_ee_status():
    return init_earth_engine()


@st.cache_resource(show_spinner="Loading satellite AQI model…")
def load_sat_model():
    import joblib, json
    mp  = os.path.join(MODEL_DIR, "best_model.pkl")
    mp2 = os.path.join(MODEL_DIR, "satellite_aqi_model.pkl")
    cp  = os.path.join(MODEL_DIR, "satellite_aqi_columns.json")
    if os.path.exists(mp2) and os.path.exists(cp):
        model = joblib.load(mp2)
        with open(cp) as f:
            cols = json.load(f)
        return model, cols
    return None, None


st.title("🗺️ India Surface AQI Map — Satellite-Derived")
st.markdown(
    "Predicted AQI across India using **satellite-only features** "
    "(MODIS AOD, Sentinel-5P NO₂/CO). Covers areas with no CPCB ground station."
)

model, feat_cols = load_sat_model()
if model is None:
    st.error("No satellite AQI model found. Run:\n```\npython src/fetch_satellite_features.py\npython src/train_satellite_aqi_model.py\n```")
    st.stop()

ee_ready = get_ee_status()

# ── Controls ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    year = st.number_input("Year", min_value=2018, max_value=2026, value=2020, key="aqi_map_yr")
with col2:
    month = st.selectbox("Month", list(range(1, 13)), index=9,
                         format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
                         key="aqi_map_mo")
with col3:
    resolution = st.radio("Grid Resolution", ["0.25° (Fast)", "0.1° (High-Res ⚠️)"], horizontal=True)
with col4:
    map_style = st.selectbox("Map Style", ["carto-positron", "open-street-map", "carto-darkmatter"])

if "High-Res" in resolution:
    st.warning("⚠️ 0.1° resolution generates ~77,000 grid points. Live EE mode will take several minutes. Recommended only in simulated mode.")
    n_lat, n_lon = 270, 290
else:
    n_lat, n_lon = 55, 58

with st.spinner("Generating AQI grid…"):
    grid = india_grid(n_lat=n_lat, n_lon=n_lon)
    sat_grid = build_grid_satellite_table(grid, int(year), int(month),
                                          use_live=None if ee_ready else False)

    X_grid = sat_grid[["satellite_aod", "satellite_no2", "satellite_co"]].copy()
    X_grid["Month"] = int(month)
    try:
        X_grid = X_grid[feat_cols]
    except KeyError:
        for c in feat_cols:
            if c not in X_grid.columns:
                X_grid[c] = 0
        X_grid = X_grid[feat_cols]

    sat_grid["predicted_aqi"] = np.clip(model.predict(X_grid), 0, 600)
    sat_grid["category"]      = sat_grid["predicted_aqi"].apply(aqi_to_category)
    sat_grid["color"]         = sat_grid["predicted_aqi"].apply(lambda x: get_advisory(x)["color"])

is_live = (sat_grid["_source"] == "live").all()
source_note = "🟢 Live Sentinel-5P + MODIS pixels" if is_live else "🟡 Simulated satellite data (EE not authenticated)"
st.caption(source_note)

# ── Summary Metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Mean AQI", f"{sat_grid['predicted_aqi'].mean():.0f}")
m2.metric("Max AQI",  f"{sat_grid['predicted_aqi'].max():.0f}")
m3.metric("Min AQI",  f"{sat_grid['predicted_aqi'].min():.0f}")
m4.metric("Grid Points", f"{len(sat_grid):,}")
worst_cat = aqi_to_category(sat_grid["predicted_aqi"].max())
m5.metric("Worst Category", worst_cat)

# ── Map ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌡️ Density Heatmap", "📊 Scatter Map"])

with tab1:
    fig = go.Figure(data=go.Densitymapbox(
        lat=sat_grid["lat"], lon=sat_grid["lon"], z=sat_grid["predicted_aqi"],
        radius=20, colorscale="YlOrRd", zmin=0, zmax=400,
        colorbar=dict(title="AQI", tickvals=[0,50,100,200,300,400],
                      ticktext=["0","50<br>Good","100<br>Satisf.","200<br>Mod","300<br>Poor","400<br>V.Poor"]),
        hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>AQI: %{z:.0f}<extra></extra>",
    ))
    fig.update_layout(
        mapbox_style=map_style,
        mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.8,
        height=600, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.scatter_mapbox(
        sat_grid, lat="lat", lon="lon",
        color="predicted_aqi", size_max=8,
        color_continuous_scale="YlOrRd",
        hover_data={"predicted_aqi": ":.0f", "category": True},
        zoom=3.8, center={"lat": 22, "lon": 82},
        height=600,
    )
    fig2.update_layout(mapbox_style=map_style, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig2, use_container_width=True)

# ── AQI Distribution ─────────────────────────────────────────────────────────
st.subheader("AQI Distribution across Grid")
cat_counts = sat_grid["category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]
cat_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
color_map = {c: get_advisory(c)["color"] for c in cat_order}
fig3 = px.bar(cat_counts, x="Category", y="Count", color="Category",
              color_discrete_map=color_map,
              title="Grid Points by AQI Category",
              category_orders={"Category": cat_order})
fig3.update_layout(showlegend=False, height=300,
                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

# ── Download ──────────────────────────────────────────────────────────────────
with st.expander("📥 Download Grid Data"):
    csv = sat_grid[["lat", "lon", "predicted_aqi", "category", "_source"]].to_csv(index=False)
    st.download_button("Download CSV", csv, "india_aqi_grid.csv", "text/csv")
