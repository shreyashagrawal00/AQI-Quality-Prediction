"""
app.py
------
Streamlit dashboard for the Surface AQI Estimation & HCHO Hotspot Detection
project. Pages match the project guide exactly:

    Home | AQI Prediction | Data Analysis | HCHO Hotspot Map |
    Model Performance | About

Visual layer: custom CSS theme (Poppins font, gradient header, styled
metric/result cards, AQI-severity color coding) layered on top of the
same functionality as the original — no logic was changed, only how
it's presented.

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="Surface AQI & HCHO Hotspot Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME -- one CSS block, applied once, used everywhere below.
# Palette follows CPCB's own AQI severity colors so the visuals stay
# consistent with the domain (Good=green ... Severe=maroon), plus a
# neutral slate/indigo palette for chrome (sidebar, cards, headers).
# ---------------------------------------------------------------------------
PRIMARY = "#4F46E5"      # indigo — brand accent
PRIMARY_DARK = "#3730A3"
BG_SOFT = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT_MUTED = "#64748B"

AQI_COLORS = {
    "Good": "#22C55E",
    "Satisfactory": "#84CC16",
    "Moderate": "#F59E0B",
    "Poor": "#F97316",
    "Very Poor": "#EF4444",
    "Severe": "#991B1B",
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .main-header, .card-title {{
        font-family: 'Poppins', sans-serif;
    }}

    /* App background */
    .stApp {{
        background: {BG_SOFT};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #F1F5F9 !important;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: background 0.15s ease-in-out;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(255,255,255,0.20);
    }}

    /* Gradient page banner */
    .page-banner {{
        background: linear-gradient(120deg, {PRIMARY} 0%, #7C3AED 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }}
    .page-banner h1 {{
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: white !important;
    }}
    .page-banner p {{
        margin: 0.4rem 0 0 0;
        font-size: 1.02rem;
        opacity: 0.92;
    }}

    /* Generic content card */
    .content-card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.05);
        margin-bottom: 1.2rem;
    }}

    /* Stat / metric card */
    .stat-card {{
        background: {CARD_BG};
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(15, 23, 42, 0.05);
        text-align: left;
    }}
    .stat-card .stat-icon {{
        font-size: 1.6rem;
    }}
    .stat-card .stat-value {{
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-top: 0.2rem;
    }}
    .stat-card .stat-label {{
        color: {TEXT_MUTED};
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}

    /* AQI result card (dynamic color set inline per prediction) */
    .aqi-result {{
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        color: white;
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    }}
    .aqi-result h1 {{
        color: white !important;
        margin: 0;
        font-size: 2.4rem;
    }}
    .aqi-result h3 {{
        color: white !important;
        margin: 0.2rem 0 0.6rem 0;
        font-weight: 600;
        opacity: 0.95;
    }}
    .aqi-result p {{
        margin: 0;
        font-size: 0.98rem;
        opacity: 0.95;
    }}

    /* Section subheaders with icon */
    .section-title {{
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: #0F172A;
        margin: 0.4rem 0 0.8rem 0;
    }}

    /* Badge chips (About page tech stack) */
    .chip {{
        display: inline-block;
        background: {PRIMARY}18;
        color: {PRIMARY_DARK};
        border-radius: 999px;
        padding: 5px 14px;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 600;
    }}

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


def banner(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-banner">
        <h1>{icon} &nbsp;{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def stat_card(icon: str, value: str, label: str):
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">{icon}</div>
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders -- loading the 65-220MB CSVs on every widget interaction
# (Streamlit reruns the whole script top-to-bottom on every interaction)
# was the single biggest performance bug in a naive first pass.
# @st.cache_data / @st.cache_resource make this load once per session.
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


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "<h2 style='margin-bottom:0'>🌫️ AQI Intel</h2>"
    "<p style='opacity:0.85;margin-top:2px;font-size:0.85rem'>Satellite × Ground Air Quality</p>"
    "<hr style='border-color:rgba(255,255,255,0.2)'>",
    unsafe_allow_html=True,
)

NAV_OPTIONS = {
    "🏠  Home": "Home",
    "🔮  AQI Prediction": "AQI Prediction",
    "📊  Data Analysis": "Data Analysis",
    "🔥  HCHO Hotspot Map": "HCHO Hotspot Map",
    "📈  Model Performance": "Model Performance",
    "ℹ️  About": "About",
}
nav_choice = st.sidebar.radio("Go to", list(NAV_OPTIONS.keys()), label_visibility="collapsed")
page = NAV_OPTIONS[nav_choice]

bundle = get_model_bundle()
if bundle is None:
    st.sidebar.warning(
        "⚠️ No trained model found. Run `python src/train_model.py` first, "
        "then relaunch the dashboard."
    )

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.2)'>", unsafe_allow_html=True)
st.sidebar.caption("Built with Python · XGBoost · Sentinel-5P · CPCB")


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "Home":
    banner(
        "🌫️", "Surface AQI Estimation & HCHO Hotspot Detection",
        "AI-powered air-quality system combining ground monitoring data with "
        "Sentinel-5P satellite observations to predict Air Quality Index (AQI) "
        "and surface formaldehyde (HCHO) hotspots across India.",
    )

    df = get_city_data()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("🏙️", f"{df['City'].nunique()}", "Cities covered")
    with c2:
        stat_card("🗂️", f"{len(df):,}", "Records")
    with c3:
        stat_card("📅", f"{df['Date'].min().year}–{df['Date'].max().year}", "Date range")
    with c4:
        stat_card("🌡️", f"{df['AQI'].mean():.0f}", "Avg. national AQI")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 National AQI trend</div>', unsafe_allow_html=True)
    trend = df.groupby("Date", as_index=False)["AQI"].mean()
    fig = px.line(trend, x="Date", y="AQI")
    fig.update_traces(line_color=PRIMARY, line_width=2)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title=None, yaxis_title="AQI",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔗 Workflow</div>', unsafe_allow_html=True)
    steps = ["🗂️ Dataset", "🧹 Cleaning", "🧪 Feature Engineering",
             "🌲 XGBoost", "✅ Evaluation", "📊 Streamlit Dashboard"]
    cols = st.columns(len(steps))
    for c, step in zip(cols, steps):
        c.markdown(
            f"<div style='text-align:center;background:{PRIMARY}12;"
            f"border-radius:12px;padding:0.7rem 0.3rem;font-weight:600;"
            f"color:{PRIMARY_DARK};font-size:0.85rem'>{step}</div>",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AQI PREDICTION
# ---------------------------------------------------------------------------
elif page == "AQI Prediction":
    banner("🔮", "Predict AQI", "Enter pollutant readings for a city and date to estimate the Air Quality Index.")

    if bundle is None:
        st.error("Train the model first: `python src/train_model.py`")
    else:
        df = get_city_data()
        cities = sorted(df["City"].unique())

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧾 Input readings</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            city = st.selectbox("🏙️ City", cities)
            pm25 = st.number_input("💨 PM2.5", 0.0, 999.0, 80.0)
            pm10 = st.number_input("💨 PM10", 0.0, 999.0, 120.0)
            no = st.number_input("🧪 NO", 0.0, 500.0, 10.0)
            no2 = st.number_input("🧪 NO2", 0.0, 500.0, 25.0)
            nox = st.number_input("🧪 NOx", 0.0, 500.0, 30.0)
            nh3 = st.number_input("🧪 NH3", 0.0, 500.0, 15.0)
        with col2:
            co = st.number_input("🔥 CO", 0.0, 50.0, 1.0)
            so2 = st.number_input("🏭 SO2", 0.0, 500.0, 15.0)
            o3 = st.number_input("☀️ O3", 0.0, 500.0, 30.0)
            benzene = st.number_input("⚗️ Benzene", 0.0, 100.0, 3.0)
            toluene = st.number_input("⚗️ Toluene", 0.0, 100.0, 5.0)
            xylene = st.number_input("⚗️ Xylene", 0.0, 100.0, 2.0)

        date = st.date_input("📅 Date", value=pd.Timestamp.today())
        predict_clicked = st.button("🔮 Predict AQI", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked:
            row = pd.DataFrame([{
                "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
                "NH3": nh3, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene,
                "Toluene": toluene, "Xylene": xylene,
                "Year": date.year, "Month": date.month, "Day": date.day,
                "City": city,
            }])
            X, _, _ = build_features(
                row, id_col="City",
                encoders=bundle["encoders"], fit_encoders=False,
            )
            X = X[bundle["feature_columns"]]  # enforce exact training order
            pred = float(bundle["model"].predict(X)[0])
            category = aqi_to_category(pred)
            color = aqi_to_color(pred) or AQI_COLORS.get(category, PRIMARY)

            category_icons = {
                "Good": "🟢", "Satisfactory": "🟡", "Moderate": "🟠",
                "Poor": "🔶", "Very Poor": "🔴", "Severe": "⛔",
            }
            icon = category_icons.get(category, "🌫️")

            st.markdown(
                f"<div class='aqi-result' style='background:linear-gradient(120deg,{color} 0%,{color}CC 100%)'>"
                f"<h1>{icon} {pred:.0f}</h1>"
                f"<h3>{category}</h3>"
                f"<p>{health_message(category)}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# DATA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "Data Analysis":
    banner("📊", "Data Analysis", "Explore distributions, city comparisons, correlations, and seasonal patterns.")
    df = get_city_data()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📉 Distributions", "🏙️ City Comparison", "🔗 Correlation", "🍂 Seasonality"]
    )

    with tab1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        col = st.selectbox("Pollutant / AQI", ["AQI", "PM2.5", "PM10", "NO2",
                                                 "SO2", "CO", "O3"])
        fig = px.histogram(df, x=col, nbins=60, color_discrete_sequence=[PRIMARY])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        city_avg = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI")
        fig = px.bar(city_avg, x="AQI", y="City", orientation="h", height=700,
                     color="AQI", color_continuous_scale="RdYlGn_r")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        num_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
                    "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        monthly = df.groupby("Month", as_index=False)["AQI"].mean()
        fig = px.line(monthly, x="Month", y="AQI", markers=True)
        fig.update_traces(line_color=PRIMARY, marker_color=PRIMARY_DARK)
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HCHO HOTSPOT MAP
# ---------------------------------------------------------------------------
elif page == "HCHO Hotspot Map":
    banner("🔥", "HCHO Hotspot Map",
           "Formaldehyde (HCHO) is a proxy for VOC emissions from biomass "
           "burning, industry, and vehicles — rendered here from Sentinel-5P "
           "TROPOMI column data via Google Earth Engine.")

    st.info(
        "🛰️ Live Sentinel-5P retrieval requires a Google Earth Engine service "
        "account (`earthengine-api`) with credentials configured in this "
        "environment. See the **HCHO_hotspot** function below and "
        "`README.md` → 'Connecting real Sentinel-5P data' for setup."
    )

    with st.expander("🧑‍💻  Show the Earth Engine retrieval code used here"):
        st.code(
            '''
import ee

def get_hcho_hotspots(start_date, end_date, india_boundary):
    """Fetch mean Sentinel-5P HCHO column density over India."""
    collection = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_HCHO")
        .filterDate(start_date, end_date)
        .filterBounds(india_boundary)
        .select("tropospheric_HCHO_column_number_density")
    )
    mean_hcho = collection.mean().clip(india_boundary)

    # Hotspot = pixels above the 90th percentile for the period
    stats = mean_hcho.reduceRegion(
        reducer=ee.Reducer.percentile([90]),
        geometry=india_boundary, scale=5000, maxPixels=1e9,
    )
    threshold = stats.get("tropospheric_HCHO_column_number_density")
    hotspot_mask = mean_hcho.gt(ee.Number(threshold))
    return mean_hcho, hotspot_mask
            ''',
            language="python",
        )

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ Illustrative hotspot map (sample grid)</div>', unsafe_allow_html=True)
    st.caption(
        "No live Earth Engine credentials are configured in this session, "
        "so this view uses a coarse synthetic grid over India as a "
        "placeholder for the real Sentinel-5P layer -- swap in "
        "`get_hcho_hotspots()` output once GEE auth is set up."
    )

    rng = np.random.default_rng(42)
    lat_grid = np.linspace(8, 35, 30)
    lon_grid = np.linspace(68, 97, 30)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    # Fake elevated bands roughly over IGP (Indo-Gangetic Plain) latitudes
    base = rng.uniform(0.5, 2.0, lon_mesh.shape)
    igp_boost = np.exp(-((lat_mesh - 27) ** 2) / 20) * 3
    hcho = base + igp_boost

    fig = go.Figure(
        data=go.Densitymapbox(
            lat=lat_mesh.flatten(), lon=lon_mesh.flatten(), z=hcho.flatten(),
            radius=25, colorscale="YlOrRd",
            colorbar_title="HCHO (×10¹⁶ molec/cm²)",
        )
    )
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 22, "lon": 82},
        mapbox_zoom=3.2,
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    banner("📈", "Model Performance", "Evaluation metrics and feature importance for the trained AQI regression model.")

    if bundle is None or not bundle["metrics"]:
        st.error("Train the model first: `python src/train_model.py`")
    else:
        m = bundle["metrics"]
        c1, c2, c3 = st.columns(3)
        with c1:
            stat_card("📏", f"{m['MAE']:.2f}", "MAE")
        with c2:
            stat_card("📐", f"{m['RMSE']:.2f}", "RMSE")
        with c3:
            stat_card("🎯", f"{m['R2']:.3f}", "R²")

        st.caption(f"Model: {m.get('model')} · trained on "
                   f"{m.get('n_train'):,} rows, tested on {m.get('n_test'):,}")

        if bundle["importances"]:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🌟 Feature importance</div>', unsafe_allow_html=True)
            imp_df = pd.DataFrame(
                list(bundle["importances"].items()),
                columns=["Feature", "Importance"],
            ).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                        height=500, color="Importance", color_continuous_scale="Purples")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------------------
elif page == "About":
    banner("ℹ️", "About this project",
           "Surface AQI Estimation & HCHO Hotspot Detection using AI and Satellite Data.")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🛰️ What it does</div>', unsafe_allow_html=True)
    st.markdown(
        "Combines CPCB ground station measurements with Sentinel-5P satellite "
        "observations to estimate air quality and identify formaldehyde "
        "hotspots linked to biomass burning and VOC emissions across India."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧰 Tech stack</div>', unsafe_allow_html=True)
    tech = ["Python", "Pandas", "NumPy", "Scikit-learn", "XGBoost",
            "Streamlit", "Plotly", "Google Earth Engine"]
    chips_html = "".join(f"<span class='chip'>{t}</span>" for t in tech)
    st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔗 Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        "🗂️ Dataset → 🧹 Cleaning → 🧪 Feature Engineering → 🌲 XGBoost → "
        "✅ Evaluation → 📊 Streamlit Dashboard"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 Future work</div>', unsafe_allow_html=True)
    st.markdown(
        "📅 AQI forecasting · 🚨 Anomaly detection · 💬 Chatbot · "
        "🏥 Health alerts · 🌐 API deployment"
    )
    st.markdown('</div>', unsafe_allow_html=True)