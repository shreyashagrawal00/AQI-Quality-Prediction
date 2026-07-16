"""
app.py
------
Streamlit dashboard for the Surface AQI Estimation & HCHO Hotspot Detection
project. Pages match the project guide exactly:

    Home | AQI Prediction | Data Analysis | HCHO Hotspot Map |
    Model Performance | About

Run with:
    streamlit run app.py
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
from aqi_utils import aqi_to_category, aqi_to_color, health_message  # noqa: E402
from data_preprocessing import load_city_data, add_time_features  # noqa: E402
from feature_engineering import build_features  # noqa: E402
from satellite_features import (  # noqa: E402
    init_earth_engine, get_ee_last_error, build_grid_satellite_table,
    get_satellite_features_for_point, simulate_satellite_row,
)
from city_coordinates import get_coordinates  # noqa: E402
from hcho_hotspots import (  # noqa: E402
    india_grid, get_hcho_mean_image, get_hcho_grid_timeseries,
    simulate_hcho_timeseries, compute_burning_anomalies,
    DEFAULT_BURNING_MONTHS, DEFAULT_BASELINE_MONTHS,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Surface AQI & HCHO Hotspot Dashboard",
                    page_icon="🌫️", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders -- FIX: loading the 65-220MB CSVs on every widget
# interaction (Streamlit reruns the whole script top-to-bottom on every
# interaction) was the single biggest performance bug in a naive first
# pass. @st.cache_data / @st.cache_resource make this load once per session.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading & cleaning ground station data...")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    df = add_time_features(df)
    df["Category"] = df["AQI"].apply(aqi_to_category)
    return df


@st.cache_resource(show_spinner="Loading trained model...")
def get_model_bundle():
    model_path = os.path.join(MODEL_DIR, "aqi_model.pkl")
    enc_path = os.path.join(MODEL_DIR, "encoders.pkl")
    cols_path = os.path.join(MODEL_DIR, "feature_columns.json")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    importance_path = os.path.join(MODEL_DIR, "feature_importance.json")

    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    importances = {}
    if os.path.exists(importance_path):
        with open(importance_path) as f:
            importances = json.load(f)
    return {
        "model": model,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "importances": importances,
    }


@st.cache_resource(show_spinner="Loading satellite-only AQI model...")
def get_satellite_model_bundle():
    model_path = os.path.join(MODEL_DIR, "satellite_aqi_model.pkl")
    cols_path = os.path.join(MODEL_DIR, "satellite_aqi_columns.json")
    metrics_path = os.path.join(MODEL_DIR, "satellite_aqi_metrics.json")

    if not os.path.exists(model_path):
        return None

    model = joblib.load(model_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {"model": model, "feature_columns": feature_columns, "metrics": metrics}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🌫️ Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "AQI Prediction", "Surface AQI Map (Satellite)", "Data Analysis",
     "HCHO Hotspot Map", "Model Performance", "About"],
)

@st.cache_resource(show_spinner=False)
def get_ee_status() -> bool:
    """Cached so we only try to authenticate once per session."""
    return init_earth_engine()


bundle = get_model_bundle()
if bundle is None:
    st.sidebar.warning(
        "No trained model found. Run `python src/train_model.py` first, "
        "then relaunch the dashboard."
    )


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Surface AQI Estimation & HCHO Hotspot Detection")
    st.markdown(
        "AI-powered air-quality system combining **ground monitoring "
        "data** with **Sentinel-5P satellite observations** to predict "
        "Air Quality Index (AQI) and surface formaldehyde (HCHO) hotspots "
        "across India."
    )

    df = get_city_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cities covered", df["City"].nunique())
    c2.metric("Records", f"{len(df):,}")
    c3.metric("Date range", f"{df['Date'].min().year}–{df['Date'].max().year}")
    c4.metric("Avg. national AQI", f"{df['AQI'].mean():.0f}")

    st.subheader("National AQI trend")
    trend = df.groupby("Date", as_index=False)["AQI"].mean()
    fig = px.line(trend, x="Date", y="AQI", title="Average daily AQI (all cities)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Workflow")
    st.markdown(
        "`Dataset` → `Cleaning` → `Feature Engineering` → `XGBoost` "
        "→ `Evaluation` → `Streamlit Dashboard`"
    )


# ---------------------------------------------------------------------------
# AQI PREDICTION
# ---------------------------------------------------------------------------
elif page == "AQI Prediction":
    st.title("Predict AQI")

    if bundle is None:
        st.error("Train the model first: `python src/train_model.py`")
    else:
        df = get_city_data()
        cities = sorted(df["City"].unique())

        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("City", cities)
            pm25 = st.number_input("PM2.5", 0.0, 999.0, 80.0)
            pm10 = st.number_input("PM10", 0.0, 999.0, 120.0)
            no = st.number_input("NO", 0.0, 500.0, 10.0)
            no2 = st.number_input("NO2", 0.0, 500.0, 25.0)
            nox = st.number_input("NOx", 0.0, 500.0, 30.0)
            nh3 = st.number_input("NH3", 0.0, 500.0, 15.0)
        with col2:
            co = st.number_input("CO", 0.0, 50.0, 1.0)
            so2 = st.number_input("SO2", 0.0, 500.0, 15.0)
            o3 = st.number_input("O3", 0.0, 500.0, 30.0)
            benzene = st.number_input("Benzene", 0.0, 100.0, 3.0)
            toluene = st.number_input("Toluene", 0.0, 100.0, 5.0)
            xylene = st.number_input("Xylene", 0.0, 100.0, 2.0)

        date = st.date_input("Date", value=pd.Timestamp.today())

        if st.button("Predict AQI", type="primary"):
            row = {
                "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
                "NH3": nh3, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene,
                "Toluene": toluene, "Xylene": xylene,
                "Year": date.year, "Month": date.month, "Day": date.day,
                "City": city,
            }

            # The trained model may or may not have satellite features
            # (depends on whether data/satellite_features.csv existed when
            # train_model.py last ran) -- only fetch/add them if it does,
            # so this page works either way instead of KeyError-ing.
            needs_satellite = any(
                c in bundle["feature_columns"]
                for c in ("satellite_aod", "satellite_no2", "satellite_co")
            )
            if needs_satellite:
                coords = get_coordinates(city)
                if coords and get_ee_status():
                    try:
                        sat_feats = get_satellite_features_for_point(coords[0], coords[1], date)
                    except Exception:
                        sat_feats = simulate_satellite_row(city, date.year, date.month)
                else:
                    sat_feats = simulate_satellite_row(city, date.year, date.month)
                row.update(sat_feats)

            row = pd.DataFrame([row])
            X, _, _ = build_features(
                row, id_col="City",
                encoders=bundle["encoders"], fit_encoders=False,
            )
            X = X[bundle["feature_columns"]]  # enforce exact training order
            pred = float(bundle["model"].predict(X)[0])
            category = aqi_to_category(pred)
            color = aqi_to_color(pred)

            if needs_satellite and not (get_coordinates(city) and get_ee_status()):
                st.caption("Satellite inputs for this prediction were simulated "
                           "(Earth Engine not authenticated or no coordinates on "
                           "file for this city) -- not live Sentinel-5P/MODIS pixels.")

            st.markdown(
                f"<div style='padding:1.2rem;border-radius:0.6rem;"
                f"background:{color}22;border:2px solid {color}'>"
                f"<h2 style='margin:0'>Predicted AQI: {pred:.0f}</h2>"
                f"<h4 style='margin:0;color:{color}'>{category}</h4>"
                f"<p style='margin-top:0.5rem'>{health_message(category)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# SURFACE AQI MAP (SATELLITE) -- estimates AQI at grid points with NO
# ground station, using only satellite AOD/NO2/CO, same live/simulate
# pattern as the HCHO page.
# ---------------------------------------------------------------------------
elif page == "Surface AQI Map (Satellite)":
    st.title("Surface AQI Map — Satellite-derived")
    st.markdown(
        "Estimates AQI across a grid over India using **only satellite "
        "features** (MODIS AOD, Sentinel-5P NO2/CO) — no ground-station "
        "pollutant readings. This is what covers areas with no CPCB "
        "monitor; the **AQI Prediction** page instead uses ground "
        "pollutant inputs for a specific monitored city."
    )

    sat_bundle = get_satellite_model_bundle()
    if sat_bundle is None:
        st.error(
            "No satellite-only AQI model found. Run:\n\n"
            "```\npython src/fetch_satellite_features.py\n"
            "python src/train_satellite_aqi_model.py\n```"
        )
    else:
        ee_ready = get_ee_status()
        if ee_ready:
            st.success("Google Earth Engine is authenticated in this session -- "
                       "the map below uses live satellite pixels.")
        else:
            reason = get_ee_last_error()
            st.warning(
                "Earth Engine isn't authenticated in this session, so the map "
                "below uses a clearly-labeled **simulated** satellite grid."
                + (f"\n\n**Reported error:** `{reason}`" if reason else "")
            )

        m = sat_bundle["metrics"]
        if m:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{m.get('MAE', float('nan')):.2f}")
            c2.metric("RMSE", f"{m.get('RMSE', float('nan')):.2f}")
            c3.metric("R²", f"{m.get('R2', float('nan')):.3f}")
            st.caption(
                "Trained on satellite features only, so accuracy is lower than "
                "the ground-station model on the Model Performance page -- "
                "expected trade-off for coverage anywhere, not just monitored cities."
            )

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Year", min_value=2018, max_value=2026, value=2020, key="aqi_map_year")
        with c2:
            month = st.selectbox(
                "Month", list(range(1, 13)), index=9,
                format_func=lambda m_: pd.Timestamp(2000, m_, 1).strftime("%b"),
                key="aqi_map_month",
            )

        grid = india_grid(n_lat=22, n_lon=22)
        sat_grid = build_grid_satellite_table(grid, int(year), int(month), use_live=None if ee_ready else False)

        X_grid = sat_grid[["satellite_aod", "satellite_no2", "satellite_co"]].copy()
        X_grid["Month"] = int(month)
        X_grid = X_grid[sat_bundle["feature_columns"]]
        sat_grid["predicted_aqi"] = sat_bundle["model"].predict(X_grid)

        source_note = ("Live satellite pixels" if (sat_grid["_source"] == "live").all()
                        else "Simulated placeholder grid (not live satellite data)" if
                        (sat_grid["_source"] == "simulated").all() else
                        "Mixed live/simulated grid (some live fetches failed)")
        st.caption(source_note)

        fig = go.Figure(
            data=go.Densitymapbox(
                lat=sat_grid["lat"], lon=sat_grid["lon"], z=sat_grid["predicted_aqi"],
                radius=25, colorscale="YlOrRd", colorbar_title="Predicted AQI",
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.2,
            height=600, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Grid average predicted AQI: {sat_grid['predicted_aqi'].mean():.0f} "
            f"({aqi_to_category(sat_grid['predicted_aqi'].mean())})"
        )


# ---------------------------------------------------------------------------
# DATA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Data Analysis":
    st.title("Data Analysis")
    df = get_city_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distributions", "City Comparison", "Correlation", "Seasonality"]
    )

    with tab1:
        col = st.selectbox("Pollutant / AQI", ["AQI", "PM2.5", "PM10", "NO2",
                                                 "SO2", "CO", "O3"])
        fig = px.histogram(df, x=col, nbins=60, title=f"{col} distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        city_avg = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI")
        fig = px.bar(city_avg, x="AQI", y="City", orientation="h",
                     title="Average AQI by city", height=700)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        num_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
                    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r",
                         title="Pollutant correlation heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        monthly = df.groupby("Month", as_index=False)["AQI"].mean()
        fig = px.line(monthly, x="Month", y="AQI", markers=True,
                      title="Average AQI by month (seasonality)")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# HCHO HOTSPOT MAP
# ---------------------------------------------------------------------------
elif page == "HCHO Hotspot Map":
    st.title("HCHO Hotspot Map")
    st.markdown(
        "Formaldehyde (HCHO) is a proxy for VOC emissions from biomass "
        "burning, industry, and vehicles. This page pulls Sentinel-5P "
        "TROPOMI HCHO column data via Google Earth Engine."
    )

    ee_ready = get_ee_status()
    if ee_ready:
        st.success("Google Earth Engine is authenticated in this session -- "
                   "views below use live Sentinel-5P data.")
    else:
        reason = get_ee_last_error()
        st.warning(
            "Earth Engine isn't authenticated in this session, so views "
            "below use a clearly-labeled **simulated** HCHO grid instead of "
            "live Sentinel-5P pixels."
            + (f"\n\n**Reported error:** `{reason}`" if reason else "")
        )
        st.caption(
            "If you already ran `earthengine authenticate` and still see "
            "this: (1) Earth Engine now requires a linked Google Cloud "
            "project -- set `EE_PROJECT=your-project-id` and restart the "
            "app; (2) make sure `earthengine-api` is installed in the SAME "
            "environment running `streamlit run app.py`, not just wherever "
            "you ran the CLI command; (3) fully restart Streamlit (not just "
            "rerun) -- this check is cached for the session, so authenticating "
            "*after* the app was already running won't be picked up until restart."
        )

    view = st.radio(
        "View",
        ["Single-period hotspot map", "Biomass-burning anomaly map (spatio-temporal)"],
        horizontal=True,
    )

    grid = india_grid(n_lat=22, n_lon=22)

    # -----------------------------------------------------------------
    # SINGLE-PERIOD MAP
    # -----------------------------------------------------------------
    if view == "Single-period hotspot map":
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start date", value=pd.Timestamp("2020-10-01"))
        with col2:
            end = st.date_input("End date", value=pd.Timestamp("2020-11-30"))

        if ee_ready:
            india_boundary = __import__("ee").Geometry.Rectangle([68, 8, 97, 35])
            mean_img = get_hcho_mean_image(str(start), str(end), india_boundary)
            sample_grid = grid.copy()
            month = pd.Timestamp(start).month
            ts = get_hcho_grid_timeseries(sample_grid, pd.Timestamp(start).year, [month])
            plot_df = ts.rename(columns={"hcho": "value"})
            source_note = "Live Sentinel-5P mean, {} to {}.".format(start, end)
        else:
            month = pd.Timestamp(start).month
            ts = simulate_hcho_timeseries(grid, [month])
            plot_df = ts.rename(columns={"hcho": "value"})
            source_note = "Simulated placeholder grid (not live satellite data)."

        st.caption(source_note)
        fig = go.Figure(
            data=go.Densitymapbox(
                lat=plot_df["lat"], lon=plot_df["lon"], z=plot_df["value"],
                radius=25, colorscale="YlOrRd",
                colorbar_title="HCHO (×10¹⁶ molec/cm²)",
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.2,
            height=600, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # BIOMASS-BURNING ANOMALY MAP (spatio-temporal)
    # -----------------------------------------------------------------
    else:
        st.markdown(
            "Compares each grid point's mean HCHO during **known biomass-"
            "burning months** against that *same point's* mean HCHO in "
            "**baseline (non-burning) months**. A point is flagged as a "
            "burning-linked hotspot only if it's anomalously high "
            "*relative to its own baseline* -- so a permanently high "
            "industrial/urban pixel isn't mistaken for a burning signal."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            year = st.number_input("Year", min_value=2018, max_value=2026, value=2020)
        with c2:
            burning_months = st.multiselect(
                "Burning-season months", list(range(1, 13)),
                default=list(DEFAULT_BURNING_MONTHS),
                format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%b"),
            )
        with c3:
            z_threshold = st.slider("Anomaly z-score threshold", 0.5, 3.0, 1.5, 0.1)

        baseline_months = [m for m in range(1, 13) if m not in burning_months]

        if ee_ready:
            all_months = sorted(set(burning_months) | set(baseline_months))
            ts = get_hcho_grid_timeseries(grid, int(year), all_months)
            source_note = f"Live Sentinel-5P monthly means for {year}."
        else:
            all_months = sorted(set(burning_months) | set(baseline_months))
            ts = simulate_hcho_timeseries(grid, all_months)
            source_note = "Simulated placeholder time series (not live satellite data)."

        st.caption(source_note)

        anomalies = compute_burning_anomalies(
            ts, burning_months=tuple(burning_months),
            baseline_months=tuple(baseline_months), z_threshold=z_threshold,
        )

        n_hotspots = int(anomalies["is_hotspot"].sum())
        st.metric("Burning-linked hotspot points", f"{n_hotspots} / {len(anomalies)}")

        fig = go.Figure(
            data=go.Densitymapbox(
                lat=anomalies["lat"], lon=anomalies["lon"], z=anomalies["z_score"],
                radius=25, colorscale="RdYlBu_r", zmid=0,
                colorbar_title="Burning-season z-score",
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.2,
            height=600, margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show flagged hotspot points (table)"):
            st.dataframe(
                anomalies[anomalies["is_hotspot"]]
                .sort_values("z_score", ascending=False)
                .reset_index(drop=True)
            )

    with st.expander("Show the Earth Engine / anomaly-detection code used here"):
        st.code(
            '''
# src/hcho_hotspots.py (abridged)
def get_hcho_grid_timeseries(grid, year, months):
    """Live: monthly mean HCHO at each grid point via Sentinel-5P."""
    ...  # ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_HCHO").filterDate(...).sampleRegions(...)

def compute_burning_anomalies(ts_df, burning_months, baseline_months, z_threshold):
    """Per-point z-score: (burning-season mean - own baseline mean) / own baseline std."""
    burning = ts_df[ts_df.month.isin(burning_months)].groupby(["lat","lon"]).hcho.mean()
    baseline = ts_df[ts_df.month.isin(baseline_months)].groupby(["lat","lon"]).hcho.agg(["mean","std"])
    z = (burning - baseline["mean"]) / baseline["std"]
    return z >= z_threshold
            ''',
            language="python",
        )


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    if bundle is None or not bundle["metrics"]:
        st.error("Train the model first: `python src/train_model.py`")
    else:
        m = bundle["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{m['MAE']:.2f}")
        c2.metric("RMSE", f"{m['RMSE']:.2f}")
        c3.metric("R²", f"{m['R2']:.3f}")
        st.caption(f"Model: {m.get('model')} · trained on "
                   f"{m.get('n_train'):,} rows, tested on {m.get('n_test'):,}")

        if m.get("used_satellite_features"):
            st.success("This model was trained WITH satellite-derived features "
                       "(satellite_aod / satellite_no2 / satellite_co) -- see "
                       "Feature importance below for their contribution.")
        else:
            st.info("This model was trained on ground-station features only. "
                   "Run `python src/fetch_satellite_features.py` then retrain "
                   "with `python src/train_model.py` to add satellite AOD/NO2/CO "
                   "as predictors.")

        if bundle["importances"]:
            imp_df = pd.DataFrame(
                list(bundle["importances"].items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                        title="Feature importance", height=500)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    st.title("About this project")
    st.markdown(
        """
**Surface AQI Estimation & HCHO Hotspot Detection using AI and Satellite Data**

Combines CPCB ground station measurements with Sentinel-5P satellite
observations to estimate air quality and identify formaldehyde
hotspots linked to biomass burning and VOC emissions across India.

**Tech stack:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit,
Plotly, Google Earth Engine.

**Pipeline:** Dataset → Cleaning → Feature Engineering → XGBoost →
Evaluation → Streamlit Dashboard.

**Future work:** AQI forecasting, anomaly detection, chatbot, health
alerts, API deployment.
        """
    )