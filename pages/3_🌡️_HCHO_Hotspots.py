"""
Page 3 — HCHO Hotspot Analysis
Enhanced hotspot detection with multiple temporal resolutions and detection methods.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from hcho_hotspots import (
    india_grid, get_hcho_grid_timeseries, simulate_hcho_timeseries,
    compute_burning_anomalies, DEFAULT_BURNING_MONTHS, DEFAULT_BASELINE_MONTHS,
)
from satellite_features import init_earth_engine, get_ee_last_error
from state_rankings import rank_states_by_hcho
from _theme import inject_dark_css, plotly_dark_layout

st.set_page_config(page_title="HCHO Hotspots", page_icon="🌡️", layout="wide")
inject_dark_css()


@st.cache_resource(show_spinner=False)
def get_ee_status():
    return init_earth_engine()


st.title("🌡️ HCHO Hotspot Analysis")
st.markdown(
    "**Formaldehyde (HCHO)** is a proxy for VOC emissions from biomass burning, industry, "
    "and vehicles. This page uses Sentinel-5P TROPOMI data to detect hotspots across India."
)

ee_ready = get_ee_status()
if ee_ready:
    st.success("🟢 Google Earth Engine connected — using live Sentinel-5P HCHO data.")
else:
    err = get_ee_last_error()
    st.warning("🟡 Earth Engine offline — using simulated HCHO data." + (f" _{err}_" if err else ""))

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ Analysis Settings")
    year = st.number_input("Year", 2018, 2026, 2020)
    detection_method = st.selectbox(
        "Hotspot Detection Method",
        ["Z-Score (Burning vs Baseline)", "Percentile Threshold", "Mean + StdDev"]
    )
    burning_months = st.multiselect(
        "Burning-Season Months", list(range(1, 13)),
        default=list(DEFAULT_BURNING_MONTHS),
        format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%b"),
    )
    z_threshold = st.slider("Z-Score / Sigma Threshold", 0.5, 3.0, 1.5, 0.1)
    percentile_thresh = st.slider("Percentile Threshold (if used)", 50, 99, 90)

baseline_months = [m for m in range(1, 13) if m not in burning_months]

# ── Temporal Resolution Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📅 Monthly", "🔥 Burning Anomaly", "📊 State Ranking", "ℹ️ Method Details"])

grid = india_grid(n_lat=25, n_lon=25)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Monthly HCHO map
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    month_sel = st.selectbox("Select Month", list(range(1, 13)),
                             format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B"),
                             key="hcho_month")
    with st.spinner("Fetching HCHO data…"):
        if ee_ready:
            ts = get_hcho_grid_timeseries(grid, int(year), [month_sel])
        else:
            ts = simulate_hcho_timeseries(grid, [month_sel])
    plot_df = ts[ts["month"] == month_sel].rename(columns={"hcho": "value"})

    if len(plot_df) == 0:
        st.warning("No data for selected month.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean HCHO", f"{plot_df['value'].mean():.3f}")
        c2.metric("Max HCHO",  f"{plot_df['value'].max():.3f}")
        c3.metric("Min HCHO",  f"{plot_df['value'].min():.3f}")
        c4.metric("Std Dev",   f"{plot_df['value'].std():.3f}")

        fig = go.Figure(go.Densitymapbox(
            lat=plot_df["lat"], lon=plot_df["lon"], z=plot_df["value"],
            radius=22, colorscale="YlOrRd",
            colorbar_title="HCHO (×10¹⁶ molec/cm²)",
            hovertemplate="Lat: %{lat:.2f}<br>Lon: %{lon:.2f}<br>HCHO: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.8,
            height=580, margin=dict(l=0,r=0,t=30,b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            title=f"HCHO Column Density — {pd.Timestamp(2000, month_sel, 1).strftime('%B')} {year}",
            title_font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Burning Anomaly Map
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        "Per-point z-score: *(burning-season HCHO mean − own baseline mean) / own baseline std*. "
        "Isolates biomass-burning signals from permanently-high industrial/urban areas."
    )
    all_months = sorted(set(burning_months) | set(baseline_months))

    with st.spinner("Computing burning anomalies…"):
        if ee_ready:
            ts_all = get_hcho_grid_timeseries(grid, int(year), all_months)
        else:
            ts_all = simulate_hcho_timeseries(grid, all_months)

        anomalies = compute_burning_anomalies(
            ts_all,
            burning_months=tuple(burning_months),
            baseline_months=tuple(baseline_months),
            z_threshold=z_threshold,
        )

    n_hot = int(anomalies["is_hotspot"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Grid Points", len(anomalies))
    k2.metric("🔴 Hotspot Points", n_hot)
    k3.metric("Hotspot %", f"{100*n_hot/max(len(anomalies),1):.1f}%")

    # detect non-hotspot points too for contrast
    non_hot = anomalies[~anomalies["is_hotspot"]]
    hot     = anomalies[anomalies["is_hotspot"]]

    fig2 = go.Figure()
    fig2.add_trace(go.Scattermapbox(
        lat=non_hot["lat"], lon=non_hot["lon"],
        mode="markers",
        marker=dict(size=6, color=non_hot["z_score"], colorscale="Blues",
                    cmin=anomalies["z_score"].min(), cmax=z_threshold - 0.01,
                    opacity=0.5),
        name="Normal",
        hovertemplate="z=%.2f<extra>Normal</extra>",
    ))
    fig2.add_trace(go.Scattermapbox(
        lat=hot["lat"], lon=hot["lon"],
        mode="markers",
        marker=dict(size=12, color="red", opacity=0.85, symbol="circle"),
        name=f"🔴 Hotspot (z≥{z_threshold})",
        hovertemplate="z=%.2f<extra>Hotspot</extra>",
    ))
    fig2.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": 22, "lon": 82}, mapbox_zoom=3.8,
        height=580, margin=dict(l=0,r=0,t=30,b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(26,26,46,0.9)", bordercolor="#2d2d44"),
        paper_bgcolor="rgba(0,0,0,0)",
        title=f"Biomass-Burning HCHO Anomalies — {year}",
        title_font=dict(color="#e2e8f0"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Top Hotspot Grid Points"):
        st.dataframe(
            anomalies[anomalies["is_hotspot"]]
            .sort_values("z_score", ascending=False)
            .reset_index(drop=True)
            .style.background_gradient(subset=["z_score"], cmap="Reds"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: State Ranking by HCHO
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    all_months_full = list(range(1, 13))
    with st.spinner("Aggregating HCHO by state…"):
        if ee_ready:
            ts_full = get_hcho_grid_timeseries(grid, int(year), all_months_full)
        else:
            ts_full = simulate_hcho_timeseries(grid, all_months_full)
        annual_mean = ts_full.groupby(["lat","lon"])["hcho"].mean().reset_index()
        state_rank  = rank_states_by_hcho(annual_mean)

    fig3 = px.bar(
        state_rank, x="mean_hcho", y="State", orientation="h",
        color="mean_hcho", color_continuous_scale="YlOrRd",
        title=f"States Ranked by Mean Annual HCHO — {year}",
        labels={"mean_hcho": "Mean HCHO (×10¹⁶ molec/cm²)"},
        height=500,
    )
    fig3.update_layout(**plotly_dark_layout())
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(state_rank, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Method Info
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
### Detection Methods

| Method | Formula | Best For |
|--------|---------|----------|
| **Z-Score (Burning vs Baseline)** | `z = (burning_mean − baseline_mean) / baseline_std` | Detecting seasonal anomalies independent of baseline pollution level |
| **Percentile Threshold** | Flag pixels above Nth percentile of all HCHO values | Simple spatial hotspot identification |
| **Mean + StdDev** | Flag pixels > (global_mean + k × global_std) | General outlier detection |

### Data Source
- **Sentinel-5P TROPOMI** HCHO column: `COPERNICUS/S5P/OFFL/L3_HCHO`
- Band: `tropospheric_HCHO_column_number_density`
- Spatial resolution: 3.5 × 5.5 km (resampled to 0.1° grid)
- Temporal: Monthly mean composites

### Key Regions
| Region | Burning Season | Cause |
|--------|---------------|-------|
| Punjab / Haryana | Oct–Nov | Post-monsoon paddy stubble burning |
| Central / Odisha | Apr–May | Pre-monsoon agricultural burning |
| Northeast (Assam) | Feb–Mar | Jhum (slash-and-burn) cultivation |
    """)
