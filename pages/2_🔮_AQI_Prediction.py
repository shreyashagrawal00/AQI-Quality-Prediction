"""
Page 2 — AQI Prediction with SHAP Explainability
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_preprocessing import load_city_data, add_time_features
from feature_engineering import build_features
from satellite_features import init_earth_engine, get_satellite_features_for_point, simulate_satellite_row
from city_coordinates import get_coordinates
from health_advisory import aqi_to_category, get_advisory
from shap_explainer import shap_available, shap_waterfall_data, explain_prediction

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="AQI Prediction", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading ground data…")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    return add_time_features(df)


@st.cache_resource(show_spinner=False)
def get_ee_status():
    return init_earth_engine()


def _load_model(model_name: str):
    slug_map = {
        "Best Model": "best_model",
        "XGBoost": "xgboost_model",
        "Random Forest": "random_forest_model",
        "LightGBM": "lightgbm_model",
        "CatBoost": "catboost_model",
    }
    slug = slug_map.get(model_name, "best_model")
    path = os.path.join(MODEL_DIR, f"{slug}.pkl")
    # fallback chain
    for p in [path, os.path.join(MODEL_DIR, "best_model.pkl"), os.path.join(MODEL_DIR, "aqi_model.pkl")]:
        if os.path.exists(p):
            return joblib.load(p)
    return None


@st.cache_resource(show_spinner="Loading model bundle…")
def get_bundle():
    enc_path  = os.path.join(MODEL_DIR, "encoders.pkl")
    cols_path = os.path.join(MODEL_DIR, "feature_columns.json")
    if not os.path.exists(enc_path):
        return None
    encoders = joblib.load(enc_path)
    with open(cols_path) as f:
        feature_columns = json.load(f)
    return {"encoders": encoders, "feature_columns": feature_columns}


st.title("🔮 AQI Prediction")
st.markdown("Enter ground pollutant readings to predict the Air Quality Index. "
            "Satellite features are auto-fetched where available.")

bundle = get_bundle()
if bundle is None:
    st.error("No trained model found. Run: `python src/multi_model.py`")
    st.stop()

df = get_city_data()
cities = sorted(df["City"].unique())

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Model Settings")
    model_choice = st.selectbox("Model", ["Best Model", "XGBoost", "Random Forest", "LightGBM", "CatBoost"])

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    city   = st.selectbox("City", cities)
    pm25   = st.number_input("PM2.5 (µg/m³)", 0.0, 999.0, 80.0, step=5.0)
    pm10   = st.number_input("PM10 (µg/m³)",  0.0, 999.0, 120.0, step=5.0)
    no     = st.number_input("NO (µg/m³)",    0.0, 500.0, 10.0)
    no2    = st.number_input("NO₂ (µg/m³)",   0.0, 500.0, 25.0)
with col2:
    nox    = st.number_input("NOx (µg/m³)",   0.0, 500.0, 30.0)
    nh3    = st.number_input("NH₃ (µg/m³)",   0.0, 500.0, 15.0)
    co     = st.number_input("CO (mg/m³)",     0.0, 50.0,  1.0)
    so2    = st.number_input("SO₂ (µg/m³)",   0.0, 500.0, 15.0)
    o3     = st.number_input("O₃ (µg/m³)",    0.0, 500.0, 30.0)
with col3:
    benzene = st.number_input("Benzene (µg/m³)", 0.0, 100.0, 3.0)
    toluene = st.number_input("Toluene (µg/m³)", 0.0, 100.0, 5.0)
    xylene  = st.number_input("Xylene (µg/m³)",  0.0, 100.0, 2.0)
    date    = st.date_input("Date", value=pd.Timestamp.today())

predict_btn = st.button("🚀 Predict AQI", type="primary", use_container_width=True)

if predict_btn:
    model = _load_model(model_choice)
    if model is None:
        st.error(f"Model '{model_choice}' not found. Run `python src/multi_model.py` first.")
        st.stop()

    row = {
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
        "NH3": nh3, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene,
        "Toluene": toluene, "Xylene": xylene,
        "Year": date.year, "Month": date.month, "Day": date.day,
        "City": city,
    }

    # Satellite feature enrichment
    needs_satellite = any(
        c in bundle["feature_columns"]
        for c in ("satellite_aod", "satellite_no2", "satellite_co")
    )
    sat_source = "none"
    if needs_satellite:
        coords = get_coordinates(city)
        if coords and get_ee_status():
            try:
                sat_feats  = get_satellite_features_for_point(coords[0], coords[1], date)
                sat_source = "live"
            except Exception:
                sat_feats  = simulate_satellite_row(city, date.year, date.month)
                sat_source = "simulated"
        else:
            sat_feats  = simulate_satellite_row(city, date.year, date.month)
            sat_source = "simulated"
        row.update(sat_feats)

    row_df = pd.DataFrame([row])
    X, _, _ = build_features(row_df, id_col="City",
                              encoders=bundle["encoders"], fit_encoders=False)
    for c in bundle["feature_columns"]:
        if c not in X.columns:
            X[c] = 0
    X = X[bundle["feature_columns"]]

    pred     = float(model.predict(X)[0])
    pred     = max(0.0, pred)
    category = aqi_to_category(pred)
    adv      = get_advisory(pred)

    st.markdown("---")

    # ── Result card ─────────────────────────────────────────────────────────
    res_col, adv_col = st.columns([1, 1])
    with res_col:
        color = adv["color"]
        bg    = adv["bg_color"]
        st.markdown(
            f"<div style='padding:1.5rem;border-radius:12px;background:{bg};"
            f"border:3px solid {color};text-align:center'>"
            f"<div style='font-size:3rem'>{adv['icon']}</div>"
            f"<h1 style='margin:0;color:{color}'>{pred:.0f}</h1>"
            f"<h3 style='color:{color};margin-top:0.2rem'>{category}</h3>"
            f"<p style='color:#555;margin-top:0.5rem'>{adv['effects'][0]}</p>"
            f"<small style='color:#888'>Model: {model_choice} "
            f"| Satellite: {sat_source}</small>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with adv_col:
        st.subheader(f"{adv['icon']} Health Advisory")
        st.markdown(f"**Who's at risk:** {adv['who_at_risk']}")
        st.markdown(f"**Outdoor activity:** {adv['outdoor_activity']}")
        st.markdown("**Health Effects:**")
        for eff in adv["effects"]:
            st.markdown(f"- {eff}")
        st.markdown("**Precautions:**")
        for pre in adv["precautions"]:
            st.markdown(f"- {pre}")

    # ── AQI Gauge chart ──────────────────────────────────────────────────────
    st.markdown("---")
    g_col, s_col = st.columns(2)
    with g_col:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred,
            title={"text": "Predicted AQI", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 500], "tickwidth": 1},
                "bar":  {"color": color, "thickness": 0.3},
                "steps": [
                    {"range": [0, 50],   "color": "rgba(0, 228, 0, 0.2)"},
                    {"range": [50, 100], "color": "rgba(163, 255, 0, 0.2)"},
                    {"range": [100, 200],"color": "rgba(255, 255, 0, 0.2)"},
                    {"range": [200, 300],"color": "rgba(255, 126, 0, 0.2)"},
                    {"range": [300, 400],"color": "rgba(255, 0, 0, 0.2)"},
                    {"range": [400, 500],"color": "rgba(143, 63, 151, 0.2)"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": pred},
            },
            number={"font": {"color": color, "size": 48}},
        ))
        fig_gauge.update_layout(height=320, margin=dict(l=20,r=20,t=40,b=0),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP Explainability ───────────────────────────────────────────────────
    with s_col:
        st.subheader("🔬 Why this AQI? (SHAP)")
        if not shap_available():
            st.info("Install SHAP for explainability: `pip install shap`")
        else:
            with st.spinner("Computing SHAP values…"):
                base_val, contrib_df = shap_waterfall_data(
                    model, X, feature_names=bundle["feature_columns"]
                )
            if contrib_df is not None:
                contrib_df["Direction"] = contrib_df["SHAP"].apply(lambda v: "Increases AQI ↑" if v > 0 else "Decreases AQI ↓")
                fig_shap = px.bar(
                    contrib_df, x="SHAP", y="Feature", orientation="h",
                    color="Direction",
                    color_discrete_map={"Increases AQI ↑": "#ef4444", "Decreases AQI ↓": "#22c55e"},
                    title=f"SHAP contributions (base: {base_val:.1f})",
                    height=320,
                )
                fig_shap.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True, legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning("SHAP explanation unavailable for this model type.")

    # ── 7-Day Forecast Section ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 7-Day AQI Forecast")
    from forecasting import forecast_city
    with st.spinner("Generating 7-day forecast..."):
        fc_df = forecast_city(city, df)
    if fc_df is not None:
        c_fc1, c_fc2 = st.columns([2, 1])
        with c_fc1:
            fig_fc = go.Figure()
            # Confidence intervals
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fc_df["Date"], fc_df["Date"].iloc[::-1]]),
                y=pd.concat([fc_df["upper_bound"], fc_df["lower_bound"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(59, 130, 246, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Confidence Interval",
            ))
            # Predicted line
            fig_fc.add_trace(go.Scatter(
                x=fc_df["Date"], y=fc_df["predicted_aqi"],
                mode="lines+markers",
                line=dict(color="#2563eb", width=3),
                marker=dict(size=8),
                name="Predicted AQI",
            ))
            fig_fc.update_layout(
                title=f"7-Day AQI Forecast for {city}",
                xaxis_title="Date",
                yaxis_title="AQI",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#e5e7eb"),
                xaxis=dict(gridcolor="#e5e7eb"),
                height=350,
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        with c_fc2:
            st.markdown(f"**7-Day Forecast values for {city}:**")
            # Format dataframe for display
            fc_disp = fc_df.copy()
            fc_disp["Date"] = fc_disp["Date"].dt.strftime("%Y-%m-%d")
            fc_disp = fc_disp.rename(columns={
                "predicted_aqi": "Predicted AQI",
                "lower_bound": "Min AQI",
                "upper_bound": "Max AQI"
            })
            st.dataframe(fc_disp, use_container_width=True, hide_index=True)
    else:
        st.info("Forecasting model or history not available for this city.")

