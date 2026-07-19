"""
Page 4 — Biomass Burning Analysis
Real NASA FIRMS fire data with HCHO correlation overlay.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from firms_fire import get_fire_data, fire_density_grid, fire_statistics, SATELLITE_OPTIONS
from hcho_hotspots import india_grid, simulate_hcho_timeseries
from satellite_features import init_earth_engine, get_ee_last_error
from hcho_hotspots import get_hcho_grid_timeseries
from _theme import inject_dark_css, plotly_dark_layout

st.set_page_config(page_title="Biomass Burning", page_icon="🔥", layout="wide")
inject_dark_css()


@st.cache_resource(show_spinner=False)
def get_ee_status():
    return init_earth_engine()


st.title("🔥 Biomass Burning Analysis")
st.markdown(
    "Active fire detection using **NASA FIRMS** (VIIRS / MODIS NRT) combined with "
    "**Sentinel-5P HCHO** data. Areas with simultaneous high HCHO and high fire activity "
    "are flagged as confirmed biomass-burning hotspots."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Fire Settings")
    satellite_choice = st.selectbox("Fire Satellite", list(SATELLITE_OPTIONS.keys()))
    days_back = st.slider("Days of fire data", 1, 10, 7)
    ref_date  = st.date_input("Reference Date", value=datetime.date.today())
    hcho_month = st.selectbox(
        "HCHO Month for overlay", list(range(1, 13)), index=ref_date.month - 1,
        format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B")
    )
    hcho_year = st.number_input("HCHO Year for overlay", 2018, 2026, ref_date.year)
    force_sim = st.checkbox("Force simulated fire data", value=False)
    hcho_thresh = st.slider("HCHO hotspot threshold (×10¹⁶)", 0.5, 3.0, 1.2, 0.1)

# ── Fetch fire data ───────────────────────────────────────────────────────────
with st.spinner("Fetching fire data from NASA FIRMS…"):
    fire_df, is_live_fire = get_fire_data(
        days=days_back,
        satellite=satellite_choice,
        reference_date=ref_date,
        force_simulate=force_sim,
    )

fire_src_note = f"🟢 Live NASA FIRMS ({satellite_choice})" if is_live_fire else "🟡 Simulated fire data"
st.caption(fire_src_note)

# ── Fire Statistics Metrics ───────────────────────────────────────────────────
stats = fire_statistics(fire_df)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Fire Points",   f"{stats['count']:,}")
m2.metric("Total FRP (MW)", f"{stats['total_frp']:.0f}")
m3.metric("Max FRP (MW)",   f"{stats['max_frp']:.1f}")
m4.metric("Mean Brightness (K)", f"{stats['mean_brightness']:.0f}" if stats['mean_brightness'] else "N/A")
m5.metric("Date Range", stats["date_range"])

st.markdown("---")

# ── Fetch HCHO for overlay ────────────────────────────────────────────────────
ee_ready = get_ee_status()
grid = india_grid(n_lat=25, n_lon=25)

with st.spinner("Fetching HCHO data…"):
    if ee_ready:
        try:
            hcho_ts = get_hcho_grid_timeseries(grid, int(hcho_year), [hcho_month])
        except Exception:
            hcho_ts = simulate_hcho_timeseries(grid, [hcho_month])
    else:
        hcho_ts = simulate_hcho_timeseries(grid, [hcho_month])

hcho_month_df = hcho_ts[hcho_ts["month"] == hcho_month].copy()
hcho_hotspot_df = hcho_month_df[hcho_month_df["hcho"] >= hcho_thresh]

# ── Main map tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Fire Map", "🌡️ Fire + HCHO Overlay", "📊 Statistics", "📋 Fire Data Table"])

with tab1:
    if len(fire_df) == 0:
        st.info("No fire data available for this period.")
    else:
        fig = go.Figure()
        # FRP-scaled markers
        max_frp = fire_df["frp"].max()
        fig.add_trace(go.Scattermapbox(
            lat=fire_df["latitude"],
            lon=fire_df["longitude"],
            mode="markers",
            marker=dict(
                size=(fire_df["frp"] / max_frp * 18 + 4).clip(4, 22),
                color=fire_df["frp"],
                colorscale="Hot",
                reversescale=True,
                colorbar=dict(title="FRP (MW)", x=1.02),
                opacity=0.8,
            ),
            text=fire_df.apply(lambda r: f"FRP: {r['frp']:.1f} MW<br>Date: {r['acq_date']}<br>Sat: {r['satellite']}", axis=1),
            hovertemplate="%{text}<extra></extra>",
            name="Fire Points",
        ))
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.8,
            height=580, margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            title=f"Active Fire Detections ({days_back}-day window)",
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(
        "Grid points with **simultaneously high HCHO** (≥ threshold) "
        "AND **fire activity** are confirmed biomass-burning hotspots."
    )

    # compute fire density on grid
    grid_density = fire_density_grid(fire_df, grid, radius_deg=0.5)
    grid_density = grid_density.merge(
        hcho_month_df[["lat","lon","hcho"]], on=["lat","lon"], how="left"
    ).fillna({"hcho": 0.0})
    grid_density["combined_hotspot"] = (
        (grid_density["hcho"] >= hcho_thresh) &
        (grid_density["fire_count"] >= 1)
    )

    n_combined = int(grid_density["combined_hotspot"].sum())

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("HCHO Hotspot Points", f"{len(hcho_hotspot_df)}")
    col_b.metric("Fire-Active Grid Points", f"{(grid_density['fire_count']>=1).sum()}")
    col_c.metric("🔴 Combined Hotspots", f"{n_combined}")

    fig2 = go.Figure()
    # HCHO base layer
    fig2.add_trace(go.Densitymapbox(
        lat=hcho_month_df["lat"], lon=hcho_month_df["lon"],
        z=hcho_month_df["hcho"],
        radius=18, colorscale="YlOrRd", opacity=0.45,
        name="HCHO Density", showscale=False,
    ))
    # fire points
    if len(fire_df) > 0:
        fig2.add_trace(go.Scattermapbox(
            lat=fire_df["latitude"], lon=fire_df["longitude"],
            mode="markers",
            marker=dict(size=7, color="orange", opacity=0.7),
            name="🟠 Fire Points",
        ))
    # combined hotspots
    combo = grid_density[grid_density["combined_hotspot"]]
    if len(combo) > 0:
        fig2.add_trace(go.Scattermapbox(
            lat=combo["lat"], lon=combo["lon"],
            mode="markers",
            marker=dict(size=14, color="red", symbol="circle", opacity=0.9),
            name="🔴 HCHO + Fire Hotspot",
        ))
    fig2.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.8,
        height=580, margin=dict(l=0,r=0,t=30,b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(26,26,46,0.9)", bordercolor="#2d2d44", font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Biomass Burning Hotspot: High HCHO + Active Fire",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    if len(fire_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            # FRP distribution
            fig3a = px.histogram(fire_df, x="frp", nbins=40,
                                 title="Fire Radiative Power (FRP) Distribution",
                                 color_discrete_sequence=["#f97316"])
            fig3a.update_layout(height=300, **plotly_dark_layout())
            st.plotly_chart(fig3a, use_container_width=True)

        with col2:
            # confidence breakdown
            conf_counts = fire_df["confidence"].value_counts().reset_index()
            conf_counts.columns = ["Confidence", "Count"]
            fig3b = px.pie(conf_counts, names="Confidence", values="Count",
                           title="Fire Detection Confidence",
                           color_discrete_sequence=px.colors.sequential.Oranges)
            fig3b.update_layout(height=300, **plotly_dark_layout())
            st.plotly_chart(fig3b, use_container_width=True)

        # Daily fire count
        fire_df_copy = fire_df.copy()
        fire_df_copy["acq_date"] = pd.to_datetime(fire_df_copy["acq_date"], errors="coerce")
        daily = fire_df_copy.groupby("acq_date").size().reset_index(name="count")
        fig3c = px.bar(daily, x="acq_date", y="count",
                       title="Daily Active Fire Count",
                       color_discrete_sequence=["#ef4444"])
        fig3c.update_layout(height=280, **plotly_dark_layout())
        st.plotly_chart(fig3c, use_container_width=True)

with tab4:
    if len(fire_df) == 0:
        st.info("No fire data to display.")
    else:
        display_df = fire_df.copy()
        display_df["frp"] = display_df["frp"].round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        csv = display_df.to_csv(index=False)
        st.download_button("📥 Download Fire Data CSV", csv, "fire_data.csv", "text/csv")
