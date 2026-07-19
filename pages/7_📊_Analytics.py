"""
Page 7 — Analytics & Correlation Dashboard
Correlation matrix, scatter plots, scientific statistics.
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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

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


st.title("📊 Analytics & Correlation Dashboard")
st.markdown("Automatic correlation analysis, scatter plots, and scientific statistics for all pollutants.")

df = get_city_data()

# ── Pollutant selection ────────────────────────────────────────────────────────
ALL_POLLUTANTS = ["AQI", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
                  "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
avail_cols = [c for c in ALL_POLLUTANTS if c in df.columns]

tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Correlation Matrix", "📉 Scatter Plots", "📈 Scientific Statistics", "📋 Feature Importance"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("**Pearson correlation coefficients** between all pollutants and AQI.")
    sel_cols = st.multiselect("Select columns", avail_cols, default=avail_cols)

    if len(sel_cols) >= 2:
        corr = df[sel_cols].corr(method="pearson")

        fig1 = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Heatmap",
        )
        fig1.update_traces(textfont_size=10)
        fig1.update_layout(height=600, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig1, use_container_width=True)

        # Top correlations with AQI
        if "AQI" in sel_cols:
            aqi_corr = (
                corr["AQI"].drop("AQI")
                .reset_index()
                .rename(columns={"index": "Pollutant", "AQI": "Correlation with AQI"})
                .sort_values("Correlation with AQI", ascending=False)
            )
            aqi_corr["Correlation with AQI"] = aqi_corr["Correlation with AQI"].round(4)
            fig_bar = px.bar(
                aqi_corr, x="Pollutant", y="Correlation with AQI",
                color="Correlation with AQI", color_continuous_scale="RdBu_r",
                title="Pearson Correlation with AQI",
                text=aqi_corr["Correlation with AQI"],
            )
            fig_bar.add_hline(y=0, line_dash="dash", line_color="#374151")
            fig_bar.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(aqi_corr, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 columns.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Scatter Plots
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    col_a, col_b, col_c = st.columns(3)
    x_col = col_a.selectbox("X axis", avail_cols, index=avail_cols.index("NO2") if "NO2" in avail_cols else 0)
    y_col = col_b.selectbox("Y axis", avail_cols, index=0)
    color_col = col_c.selectbox("Color by", ["None"] + avail_cols, index=0)

    sample_n = st.slider("Sample size (for performance)", 1000, min(50000, len(df)), 5000, step=500)
    sample_df = df[avail_cols].dropna().sample(min(sample_n, len(df.dropna(subset=avail_cols))), random_state=42)

    c_arg = color_col if color_col != "None" else None
    fig2 = px.scatter(
        sample_df, x=x_col, y=y_col, color=c_arg,
        color_continuous_scale="YlOrRd" if c_arg else None,
        opacity=0.4, trendline="ols",
        title=f"{y_col} vs {x_col}",
        height=480,
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    # Pearson for this pair
    r = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
    n = df[[x_col, y_col]].dropna().shape[0]
    st.info(f"Pearson r({x_col}, {y_col}) = **{r:.4f}** (n={n:,})")

    # Quick AQI vs top pollutants
    st.markdown("### Quick AQI Scatter: Key Pollutants")
    quick_cols = [c for c in ["PM2.5","PM10","NO2","CO","SO2","O3"] if c in df.columns]
    ncols = 3
    rows_q = [quick_cols[i:i+ncols] for i in range(0, len(quick_cols), ncols)]
    for row_q in rows_q:
        cols_q = st.columns(ncols)
        for j, pc in enumerate(row_q):
            sub = df[["AQI", pc]].dropna().sample(min(2000, len(df.dropna(subset=["AQI",pc]))), random_state=42)
            r_q = sub.corr().iloc[0,1]
            fig_q = px.scatter(sub, x=pc, y="AQI", opacity=0.3, trendline="ols",
                               title=f"AQI vs {pc} (r={r_q:.3f})",
                               color_discrete_sequence=["#3b82f6"])
            fig_q.update_layout(height=280, margin=dict(l=10,r=10,t=40,b=10),
                                 plot_bgcolor="rgba(0,0,0,0)")
            cols_q[j].plotly_chart(fig_q, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Scientific Statistics
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("Descriptive statistics for every pollutant and AQI.")
    stats_df = df[avail_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95]).T
    stats_df.index.name = "Variable"
    stats_df = stats_df.rename(columns={"count":"N","mean":"Mean","std":"Std Dev",
                                         "min":"Min","5%":"P5","25%":"P25","50%":"Median",
                                         "75%":"P75","90%":"P90","95%":"P95","max":"Max"})
    stats_df = stats_df.round(3)
    st.dataframe(stats_df, use_container_width=True)

    # Violin plots for each pollutant
    st.subheader("Distribution Violin Plots")
    sel_violin = st.multiselect("Select variables", avail_cols, default=avail_cols[:5])
    if sel_violin:
        melt_df = df[sel_violin].melt(var_name="Pollutant", value_name="Value").dropna()
        fig3 = px.violin(melt_df, x="Pollutant", y="Value", box=True,
                         color="Pollutant", title="Pollutant Distributions",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=450, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    import json, os
    imp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "feature_importance.json")
    if os.path.exists(imp_path):
        with open(imp_path) as f:
            imp_data = json.load(f)
        imp_df = pd.DataFrame(list(imp_data.items()), columns=["Feature","Importance"])
        imp_df = imp_df.sort_values("Importance", ascending=True)
        imp_df["Importance %"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(2)

        fig4 = px.bar(imp_df.tail(20), x="Importance", y="Feature",
                      orientation="h", color="Importance",
                      color_continuous_scale="Blues",
                      title="Top 20 Feature Importances (Model)",
                      text=imp_df.tail(20)["Importance %"].apply(lambda v: f"{v:.1f}%"),
                      height=550)
        fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(imp_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance not available. Run `python src/multi_model.py` first.")
