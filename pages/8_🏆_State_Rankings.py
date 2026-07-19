"""
Page 8 — State Rankings
Worst AQI, cleanest states, top HCHO, highest fire activity, most improved.
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
from hcho_hotspots import india_grid, simulate_hcho_timeseries
from firms_fire import get_fire_data
from state_rankings import (
    load_state_map, rank_states_by_aqi, rank_states_by_hcho,
    rank_states_by_fire, most_improved_states,
)
from _theme import inject_dark_css, plotly_dark_layout

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

st.set_page_config(page_title="State Rankings", page_icon="🏆", layout="wide")
inject_dark_css()
# Extra rank card CSS on top of dark theme
st.markdown("""
<style>
.rank-card {
  background: linear-gradient(135deg, #1a1a2e, #16213e);
  border: 1px solid #2d2d44;
  border-radius: 14px; padding: 1rem; color: white;
  text-align: center;
  box-shadow: 0 4px 20px rgba(0,0,0,.5);
  margin-bottom: 0.5rem;
  position: relative; overflow: hidden;
}
.rank-card::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, #00d4ff, #3b82f6);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data…")
def get_city_data():
    df = load_city_data(DATA_DIR, "day")
    return add_time_features(df)


@st.cache_data(show_spinner=False)
def get_state_map():
    return load_state_map(DATA_DIR)


st.title("🏆 State Rankings")
st.markdown("Ranked tables and charts for Indian states across AQI, HCHO, fire activity, and improvement trends.")

df = get_city_data()
city_state_map = get_state_map()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Worst AQI", "🟢 Cleanest States", "🌡️ HCHO Hotspot", "🔥 Fire Activity", "📈 Most Improved"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Worst AQI States
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    top_n = st.slider("Show top N states", 5, 30, 15, key="worst_n")
    worst = rank_states_by_aqi(df, city_state_map, top_n=top_n)

    # Podium for top 3
    if len(worst) >= 3:
        p1, p2, p3 = st.columns(3)
        for col, rank, emoji in [(p2,0,"🥇"),(p1,1,"🥈"),(p3,2,"🥉")]:
            row = worst.iloc[rank]
            col.markdown(f"""<div class="rank-card">
              <div style="font-size:2rem">{emoji}</div>
              <b style="font-size:1.1rem">{row['State']}</b><br>
              <span style="font-size:1.5rem">{row['AQI_mean']:.0f}</span><br>
              <small>Mean AQI</small>
            </div>""", unsafe_allow_html=True)

    fig1 = px.bar(
        worst[::-1], x="AQI_mean", y="State", orientation="h",
        color="AQI_mean", color_continuous_scale="YlOrRd",
        text=worst[::-1]["AQI_mean"].round(0),
        title=f"Top {top_n} States by Mean AQI (Worst First)",
        labels={"AQI_mean": "Mean AQI"},
        height=max(350, top_n * 28),
    )
    fig1.update_layout(**plotly_dark_layout(height=max(350, top_n * 28)))
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(worst.style.background_gradient(subset=["AQI_mean"], cmap="YlOrRd"),
                 use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Cleanest States
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    all_ranked = rank_states_by_aqi(df, city_state_map, top_n=100)
    cleanest   = all_ranked.tail(15).iloc[::-1].reset_index(drop=True)

    fig2 = px.bar(
        cleanest, x="AQI_mean", y="State", orientation="h",
        color="AQI_mean", color_continuous_scale="Greens_r",
        text=cleanest["AQI_mean"].round(0),
        title="Cleanest States by Mean AQI",
        labels={"AQI_mean": "Mean AQI"},
        height=450,
    )
    fig2.update_layout(**plotly_dark_layout(height=450))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(cleanest, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: HCHO Hotspot States
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    hcho_year_sel = st.number_input("HCHO Year", 2018, 2026, 2020, key="hcho_rank_yr")
    grid = india_grid(n_lat=25, n_lon=25)
    with st.spinner("Aggregating HCHO by state…"):
        hcho_ts = simulate_hcho_timeseries(grid, list(range(1, 13)))
        annual_mean = hcho_ts.groupby(["lat","lon"])["hcho"].mean().reset_index()
        hcho_rank = rank_states_by_hcho(annual_mean)

    fig3 = px.bar(
        hcho_rank, x="mean_hcho", y="State", orientation="h",
        color="mean_hcho", color_continuous_scale="Oranges",
        text=hcho_rank["mean_hcho"].round(3),
        title=f"States Ranked by Mean HCHO (×10¹⁶ molec/cm²) — {hcho_year_sel}",
        height=450,
    )
    fig3.update_layout(**plotly_dark_layout(height=450))
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(hcho_rank, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Highest Fire Activity
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    with st.spinner("Loading fire data…"):
        fire_df, is_live = get_fire_data(days=7)
    src_note = "🟢 Live NASA FIRMS" if is_live else "🟡 Simulated fire data"
    st.caption(src_note)

    fire_rank = rank_states_by_fire(fire_df)
    if len(fire_rank) == 0:
        st.info("No fire data available.")
    else:
        fig4a = px.bar(
            fire_rank, x="fire_count", y="State", orientation="h",
            color="fire_count", color_continuous_scale="Hot_r",
            text=fire_rank["fire_count"],
            title="States Ranked by Fire Count (7-day window)",
            height=420,
        )
        fig4a.update_layout(**plotly_dark_layout(height=420))
        st.plotly_chart(fig4a, use_container_width=True)

        fig4b = px.bar(
            fire_rank, x="total_frp", y="State", orientation="h",
            color="total_frp", color_continuous_scale="Reds",
            text=fire_rank["total_frp"].round(0),
            title="States Ranked by Total FRP (MW)",
            height=420,
        )
        fig4b.update_layout(**plotly_dark_layout(height=420))
        st.plotly_chart(fig4b, use_container_width=True)
        st.dataframe(fire_rank, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Most Improved
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    improved = most_improved_states(df, city_state_map, top_n=15)
    if len(improved) == 0:
        st.info("Need at least 2 years of data to compute improvement.")
    else:
        # negative delta = improvement
        improved_pos = improved[improved["delta"] < 0].copy()
        improved_pos["improvement"] = -improved_pos["delta"]

        fig5 = px.bar(
            improved_pos, x="improvement", y="State", orientation="h",
            color="improvement", color_continuous_scale="Greens",
            text=improved_pos["improvement"].round(1),
            title="Most Improved States (Largest AQI Decrease YoY)",
            labels={"improvement": "AQI Improvement"},
            height=450,
        )
        fig5.update_layout(**plotly_dark_layout(height=450))
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(improved, use_container_width=True, hide_index=True)
