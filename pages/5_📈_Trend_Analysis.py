"""
Page 5 — Trend Analysis
AQI, HCHO, and fire temporal analysis with animations and year-over-year comparison.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from data_preprocessing import load_city_data, add_time_features
from health_advisory import aqi_to_category, get_advisory

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

st.set_page_config(page_title="Trend Analysis", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data…")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    return add_time_features(df)


st.title("📈 Trend Analysis")
st.markdown("Temporal analysis of AQI, pollutants, and seasonal patterns across India.")

df = get_city_data()
cities = sorted(df["City"].unique())

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Filter Settings")
    selected_cities = st.multiselect("Cities", cities, default=cities[:5])
    pollutant_options = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NOx", "NH3"]
    pollutant = st.selectbox("Pollutant / Metric", pollutant_options)
    agg_freq = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])
    year_range = st.slider("Year Range",
                           int(df["Year"].min()), int(df["Year"].max()),
                           (int(df["Year"].min()), int(df["Year"].max())))

# ── Filter ───────────────────────────────────────────────────────────────────
df_f = df.copy()
if selected_cities:
    df_f = df_f[df_f["City"].isin(selected_cities)]
df_f = df_f[(df_f["Year"] >= year_range[0]) & (df_f["Year"] <= year_range[1])]

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
freq = freq_map[agg_freq]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📅 Time Series", "📆 Monthly Seasonality", "📊 Yearly Trend", "🔄 Year Comparison", "🎬 Animation"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Time Series
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    if len(df_f) == 0:
        st.warning("No data for selected filters.")
    else:
        if selected_cities and len(selected_cities) <= 6:
            # per-city lines
            city_trend = df_f.groupby(["Date", "City"])[pollutant].mean().reset_index()
            fig = px.line(city_trend, x="Date", y=pollutant, color="City",
                          title=f"{pollutant} Over Time by City",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            # national average
            nat_trend = df_f.groupby("Date")[pollutant].mean().reset_index()
            fig = px.line(nat_trend, x="Date", y=pollutant,
                          title=f"National Average {pollutant} Over Time",
                          color_discrete_sequence=["#3b82f6"])

        if pollutant == "AQI":
            for low, high, color, label in [
                (0,50,"rgba(0,228,0,0.1)","Good"), (51,100,"rgba(163,255,0,0.08)","Satisfactory"),
                (101,200,"rgba(255,255,0,0.08)","Moderate"), (201,300,"rgba(255,126,0,0.08)","Poor"),
            ]:
                fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0,
                              annotation_text=label, annotation_position="right")

        fig.update_layout(height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(gridcolor="#e5e7eb"), xaxis=dict(gridcolor="#e5e7eb"))
        st.plotly_chart(fig, use_container_width=True)

        # Rolling average option
        show_roll = st.checkbox("Show 30-day rolling average")
        if show_roll and not (selected_cities and len(selected_cities) <= 6):
            nat_trend2 = df_f.groupby("Date")[pollutant].mean().reset_index()
            nat_trend2["rolling"] = nat_trend2[pollutant].rolling(30, center=True).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=nat_trend2["Date"], y=nat_trend2[pollutant],
                                      mode="lines", line=dict(color="#93c5fd", width=1), name="Daily"))
            fig2.add_trace(go.Scatter(x=nat_trend2["Date"], y=nat_trend2["rolling"],
                                      mode="lines", line=dict(color="#1d4ed8", width=2.5), name="30-day avg"))
            fig2.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Monthly Seasonality
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    monthly = df_f.groupby("Month")[pollutant].agg(["mean","std","median"]).reset_index()
    monthly["Month_Name"] = monthly["Month"].apply(lambda m: pd.Timestamp(2000, m, 1).strftime("%b"))

    fig_m = go.Figure()
    fig_m.add_trace(go.Bar(
        x=monthly["Month_Name"], y=monthly["mean"],
        error_y=dict(type="data", array=monthly["std"].fillna(0), visible=True),
        marker_color=monthly["mean"],
        marker_colorscale="YlOrRd",
        name="Mean ± Std",
    ))
    fig_m.add_trace(go.Scatter(
        x=monthly["Month_Name"], y=monthly["median"],
        mode="lines+markers", name="Median",
        line=dict(color="#1d4ed8", width=2.5, dash="dash"),
    ))
    fig_m.update_layout(
        title=f"Monthly Seasonality — {pollutant}",
        height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb"),
    )
    st.plotly_chart(fig_m, use_container_width=True)

    # Season box plot
    df_f["Season"] = df_f["Month"].map({
        12:"Winter", 1:"Winter", 2:"Winter",
        3:"Pre-Monsoon", 4:"Pre-Monsoon", 5:"Pre-Monsoon",
        6:"Monsoon", 7:"Monsoon", 8:"Monsoon",
        9:"Post-Monsoon", 10:"Post-Monsoon", 11:"Post-Monsoon",
    })
    fig_s = px.box(df_f.dropna(subset=[pollutant]),
                   x="Season", y=pollutant, color="Season",
                   title=f"Seasonal Distribution — {pollutant}",
                   category_orders={"Season": ["Winter","Pre-Monsoon","Monsoon","Post-Monsoon"]},
                   color_discrete_sequence=["#3b82f6","#f59e0b","#10b981","#ef4444"])
    fig_s.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_s, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Yearly Trend
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    yearly = df_f.groupby("Year")[pollutant].agg(["mean","median","max","min"]).reset_index()
    yearly.columns = ["Year","mean","median","max","min"]

    fig_y = go.Figure()
    fig_y.add_trace(go.Scatter(x=yearly["Year"], y=yearly["max"],
                               fill=None, mode="lines", line_color="#fca5a5", name="Max"))
    fig_y.add_trace(go.Scatter(x=yearly["Year"], y=yearly["min"],
                               fill="tonexty", mode="lines", line_color="#bbf7d0",
                               fillcolor="rgba(16,185,129,0.1)", name="Min"))
    fig_y.add_trace(go.Scatter(x=yearly["Year"], y=yearly["mean"],
                               mode="lines+markers", line=dict(color="#1d4ed8", width=3),
                               marker=dict(size=8), name="Mean"))
    fig_y.add_trace(go.Scatter(x=yearly["Year"], y=yearly["median"],
                               mode="lines", line=dict(color="#f59e0b", width=2, dash="dash"),
                               name="Median"))
    fig_y.update_layout(
        title=f"Year-over-Year Trend — {pollutant}",
        height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick=1, gridcolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb"),
    )
    st.plotly_chart(fig_y, use_container_width=True)
    st.dataframe(yearly.set_index("Year").style.background_gradient(subset=["mean"], cmap="YlOrRd"),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Year-over-Year Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    available_years = sorted(df_f["Year"].unique())
    if len(available_years) < 2:
        st.warning("Need at least 2 years of data for comparison.")
    else:
        c1, c2 = st.columns(2)
        yr_a = c1.selectbox("Year A", available_years, index=0)
        yr_b = c2.selectbox("Year B", available_years, index=min(1, len(available_years)-1))

        df_a = df_f[df_f["Year"] == yr_a].groupby("Month")[pollutant].mean().reset_index()
        df_b = df_f[df_f["Year"] == yr_b].groupby("Month")[pollutant].mean().reset_index()
        df_a["Month_Name"] = df_a["Month"].apply(lambda m: pd.Timestamp(2000,m,1).strftime("%b"))
        df_b["Month_Name"] = df_b["Month"].apply(lambda m: pd.Timestamp(2000,m,1).strftime("%b"))

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(x=df_a["Month_Name"], y=df_a[pollutant],
                                  name=str(yr_a), marker_color="#3b82f6", opacity=0.85))
        fig_cmp.add_trace(go.Bar(x=df_b["Month_Name"], y=df_b[pollutant],
                                  name=str(yr_b), marker_color="#f59e0b", opacity=0.85))
        fig_cmp.update_layout(
            barmode="group",
            title=f"Monthly {pollutant}: {yr_a} vs {yr_b}",
            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # delta table
        merged_cmp = df_a.merge(df_b, on="Month_Name", suffixes=(f"_{yr_a}", f"_{yr_b}"))
        merged_cmp["Δ"] = (merged_cmp[f"{pollutant}_{yr_b}"] - merged_cmp[f"{pollutant}_{yr_a}"]).round(1)
        merged_cmp["Δ%"] = (merged_cmp["Δ"] / merged_cmp[f"{pollutant}_{yr_a}"] * 100).round(1)
        st.dataframe(merged_cmp[["Month_Name", f"{pollutant}_{yr_a}", f"{pollutant}_{yr_b}", "Δ", "Δ%"]],
                     use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Animated Map
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("Animated city-level AQI by month across all years.")
    monthly_city = df_f.groupby(["Year","Month","City"])[pollutant].mean().reset_index()
    monthly_city["Month_Name"] = monthly_city["Month"].apply(lambda m: pd.Timestamp(2000,m,1).strftime("%b"))
    monthly_city["Period"] = monthly_city["Year"].astype(str) + "-" + monthly_city["Month_Name"]

    # need coordinates
    try:
        from city_coordinates import get_coordinates
        coords_list = []
        for c in monthly_city["City"].unique():
            coords = get_coordinates(c)
            if coords:
                coords_list.append({"City": c, "lat": coords[0], "lon": coords[1]})
        coords_df = pd.DataFrame(coords_list)
        anim_df = monthly_city.merge(coords_df, on="City", how="inner")
        if len(anim_df) > 0:
            fig_anim = px.scatter_mapbox(
                anim_df, lat="lat", lon="lon",
                color=pollutant, size=pollutant,
                animation_frame="Period",
                hover_name="City",
                color_continuous_scale="YlOrRd",
                size_max=25, zoom=3.5,
                center={"lat": 22, "lon": 82},
                height=550,
                title=f"Animated {pollutant} by City",
            )
            fig_anim.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_anim, use_container_width=True)
        else:
            st.info("No coordinate data available for animation.")
    except Exception as e:
        st.info(f"Animation unavailable: {e}")
