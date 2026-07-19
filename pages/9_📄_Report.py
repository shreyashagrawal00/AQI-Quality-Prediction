"""
Page 9 — Report Generator
Configurable PDF report with AQI maps, statistics, trends, and rankings.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_preprocessing import load_city_data, add_time_features
from health_advisory import aqi_to_category, get_advisory
from report_generator import generate_report, report_available, AQIReport
from state_rankings import load_state_map, rank_states_by_aqi

DATA_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="Report Generator", page_icon="📄", layout="wide")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from _theme import inject_dark_css, plotly_dark_layout
inject_dark_css()


@st.cache_data(show_spinner="Loading data…")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    return add_time_features(df)


@st.cache_data(show_spinner=False)
def get_state_map():
    return load_state_map(DATA_DIR)


st.title("📄 Report Generator")
st.markdown(
    "Generate a comprehensive **PDF report** with AQI statistics, trend charts, "
    "state rankings, and model performance. Download with one click."
)

if not report_available():
    st.error("📦 `fpdf2` is not installed. Run: `pip install fpdf2 kaleido`")
    st.stop()

df = get_city_data()
city_state_map = get_state_map()

# ── Report configurator ───────────────────────────────────────────────────────
st.subheader("⚙️ Configure Report")
col1, col2 = st.columns(2)

with col1:
    report_title = st.text_input("Report Title", "India Surface AQI & HCHO Analysis Report")
    year_sel     = st.selectbox("Analysis Year", sorted(df["Year"].unique(), reverse=True))
    incl_stats   = st.checkbox("Include Summary Statistics", value=True)
    incl_trend   = st.checkbox("Include National AQI Trend", value=True)

with col2:
    incl_city    = st.checkbox("Include City Rankings", value=True)
    incl_state   = st.checkbox("Include State Rankings", value=True)
    incl_legend  = st.checkbox("Include AQI Category Legend", value=True)
    incl_model   = st.checkbox("Include Model Performance", value=True)

st.markdown("---")

# ── Preview Section ────────────────────────────────────────────────────────────
st.subheader("👁️ Preview")

df_yr = df[df["Year"] == year_sel]
avg_aqi   = df_yr["AQI"].mean()
max_aqi   = df_yr["AQI"].max()
min_aqi   = df_yr["AQI"].min()
n_cities  = df_yr["City"].nunique()

summary_stats = {
    "Year": year_sel,
    "Cities": n_cities,
    "Avg AQI": f"{avg_aqi:.1f}",
    "Max AQI": f"{max_aqi:.0f}",
    "Min AQI": f"{min_aqi:.0f}",
    "Category": aqi_to_category(avg_aqi),
}

# preview metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Avg AQI", f"{avg_aqi:.1f}")
mc2.metric("Max AQI", f"{max_aqi:.0f}")
mc3.metric("Cities", n_cities)
mc4.metric("Category", aqi_to_category(avg_aqi))

# Build charts for preview and report
figs_for_report = []

if incl_trend:
    monthly = df_yr.groupby("Month")["AQI"].mean().reset_index()
    monthly["Month_Name"] = monthly["Month"].apply(lambda m: pd.Timestamp(2000, m, 1).strftime("%b"))
    fig_trend = px.line(monthly, x="Month_Name", y="AQI", markers=True,
                        title=f"Monthly Mean AQI — {year_sel}",
                        color_discrete_sequence=["#3b82f6"])
    fig_trend.update_layout(**plotly_dark_layout(height=350))
    st.plotly_chart(fig_trend, use_container_width=True)
    figs_for_report.append((fig_trend, f"Monthly Mean AQI — {year_sel}"))

if incl_city:
    city_avg = df_yr.groupby("City")["AQI"].mean().sort_values(ascending=False).head(15).reset_index()
    fig_city = px.bar(city_avg, x="AQI", y="City", orientation="h",
                      color="AQI", color_continuous_scale="YlOrRd",
                      title=f"Top 15 Most Polluted Cities — {year_sel}", height=400)
    fig_city.update_layout(**plotly_dark_layout())
    st.plotly_chart(fig_city, use_container_width=True)
    figs_for_report.append((fig_city, f"Top 15 Polluted Cities — {year_sel}"))

if incl_state:
    state_rank = rank_states_by_aqi(df_yr, city_state_map, top_n=15)
    if len(state_rank) > 0:
        fig_state = px.bar(state_rank, x="AQI_mean", y="State", orientation="h",
                           color="AQI_mean", color_continuous_scale="YlOrRd",
                           title=f"States by Mean AQI — {year_sel}", height=400)
        fig_state.update_layout(**plotly_dark_layout())
        st.plotly_chart(fig_state, use_container_width=True)
        figs_for_report.append((fig_state, f"State AQI Rankings — {year_sel}"))

tables_for_report = []
if incl_stats:
    stats_df = df_yr[["AQI","PM2.5","PM10","NO2","CO","SO2","O3"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95]
    ).T.round(2)
    tables_for_report.append((stats_df.reset_index().rename(columns={"index":"Pollutant"}),
                               f"Pollutant Statistics — {year_sel}"))

if incl_model:
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        model_df = pd.DataFrame([{
            "Model": m.get("model","N/A"), "MAE": round(m.get("MAE",0),2),
            "RMSE": round(m.get("RMSE",0),2), "R²": round(m.get("R2",0),4),
        }])
        tables_for_report.append((model_df, "Model Performance"))

# ── Generate PDF ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Generate & Download")

if st.button("🚀 Generate PDF Report", type="primary", use_container_width=True):
    with st.spinner("Generating PDF (may take 10–30 seconds for chart rendering)…"):
        try:
            pdf_bytes = generate_report(
                title=report_title,
                stats_dict=summary_stats,
                figures=figs_for_report,
                tables=tables_for_report,
                include_legend=incl_legend,
            )
            if pdf_bytes:
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"AQI_Report_{year_sel}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.error("Failed to generate report. Check that fpdf2 is installed.")
        except Exception as e:
            st.error(f"Error generating report: {e}")

st.info(
    "💡 **Tip:** Install `kaleido` for chart images in the PDF: `pip install kaleido`\n\n"
    "Without kaleido, the PDF will contain text placeholders instead of chart images."
)
