"""
Page 6 — Model Validation
Compare CPCB ground-truth AQI against predicted AQI with scatter, residuals, and error distribution.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

st.set_page_config(page_title="Model Validation", page_icon="✅", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
</style>
""", unsafe_allow_html=True)


def _mape(y_true, y_pred):
    mask = np.array(y_true) != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask])) * 100)


@st.cache_data(show_spinner="Loading validation data…")
def load_validation():
    npz_path = os.path.join(MODEL_DIR, "validation_data.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data["y_test"], data["y_pred"]
    return None, None


@st.cache_data(show_spinner=False)
def load_comparison():
    path = os.path.join(MODEL_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


st.title("✅ Model Validation")
st.markdown(
    "Comparison of **CPCB ground-truth AQI** vs **model-predicted AQI** on the held-out test set."
)

y_test, y_pred = load_validation()
cmp_data = load_comparison()

if y_test is None:
    st.error(
        "No validation data found. Run:\n```\npython src/multi_model.py\n```\n"
        "to train models and generate validation_data.npz."
    )
    st.stop()

residuals  = y_pred - y_test
abs_errors = np.abs(residuals)

# ── Metrics Row ─────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = r2_score(y_test, y_pred)
mape = _mape(y_test, y_pred)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE",  f"{mae:.2f}",  help="Mean Absolute Error — lower is better")
c2.metric("RMSE", f"{rmse:.2f}", help="Root Mean Squared Error — lower is better")
c3.metric("R²",   f"{r2:.4f}",  help="Coefficient of determination — closer to 1 is better")
c4.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error — lower is better")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Scatter Plot", "📉 Residual Plot", "📊 Error Distribution", "📋 Model Comparison", "📈 Percentile Analysis"
])

with tab1:
    val_df = pd.DataFrame({"Actual AQI": y_test, "Predicted AQI": y_pred, "Residual": residuals})
    val_df["Abs Error"] = abs_errors
    val_df["Category"] = pd.cut(y_test, bins=[0,50,100,200,300,400,np.inf],
                                 labels=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"])

    fig1 = go.Figure()
    # perfect prediction line
    lo, hi = float(y_test.min()), float(y_test.max())
    fig1.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                              line=dict(color="#1d4ed8", dash="dash", width=2), name="Perfect Prediction"))
    # ±20% band
    fig1.add_trace(go.Scatter(x=[lo,hi], y=[lo*0.8, hi*0.8], mode="lines",
                              line=dict(color="#d1d5db", width=1), name="−20%", showlegend=True))
    fig1.add_trace(go.Scatter(x=[lo,hi], y=[lo*1.2, hi*1.2],
                              fill="tonexty", mode="lines",
                              line=dict(color="#d1d5db", width=1),
                              fillcolor="rgba(209,213,219,0.15)", name="+20%"))
    # scatter
    fig1.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(size=4, color=abs_errors, colorscale="YlOrRd",
                    colorbar=dict(title="Abs Error"), opacity=0.6),
        name="Test points",
        hovertemplate="Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>",
    ))
    fig1.update_layout(
        title=f"Actual vs Predicted AQI (n={len(y_test):,})",
        xaxis_title="Actual AQI (CPCB Ground Truth)",
        yaxis_title="Predicted AQI (Model)",
        height=520, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # within-n-AQI accuracy
    for tol in [25, 50, 75]:
        within = (abs_errors <= tol).mean() * 100
        st.metric(f"Within ±{tol} AQI", f"{within:.1f}%", label_visibility="visible")

with tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_test, y=residuals, mode="markers",
        marker=dict(size=4, color=residuals, colorscale="RdBu", cmid=0,
                    colorbar=dict(title="Residual"), opacity=0.5),
        hovertemplate="Actual: %{x:.0f}<br>Residual: %{y:.0f}<extra></extra>",
    ))
    fig2.add_hline(y=0, line_dash="dash", line_color="#374151", line_width=2)
    fig2.add_hline(y=mae,  line_dash="dot", line_color="#22c55e", annotation_text=f"+MAE={mae:.1f}")
    fig2.add_hline(y=-mae, line_dash="dot", line_color="#22c55e", annotation_text=f"−MAE={mae:.1f}")
    fig2.update_layout(
        title="Residual Plot (Predicted − Actual)",
        xaxis_title="Actual AQI", yaxis_title="Residual",
        height=450, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        fig3a = px.histogram(abs_errors, nbins=50,
                             title="Absolute Error Distribution",
                             labels={"value": "Absolute Error"},
                             color_discrete_sequence=["#3b82f6"])
        fig3a.add_vline(x=mae, line_dash="dash", line_color="#ef4444",
                        annotation_text=f"MAE={mae:.1f}")
        fig3a.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3a, use_container_width=True)

    with col_b:
        fig3b = px.histogram(residuals, nbins=50,
                             title="Residual Distribution",
                             labels={"value": "Residual (Pred − Actual)"},
                             color_discrete_sequence=["#8b5cf6"])
        fig3b.add_vline(x=0, line_dash="dash", line_color="#374151")
        fig3b.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig3b, use_container_width=True)

    # Scientific statistics
    st.subheader("Scientific Statistics (Errors)")
    stats_data = {
        "Metric": ["MAE", "RMSE", "R²", "MAPE (%)", "Median Abs Error",
                   "95th percentile error", "Max error", "Bias (mean residual)"],
        "Value": [
            f"{mae:.3f}", f"{rmse:.3f}", f"{r2:.4f}", f"{mape:.2f}",
            f"{np.median(abs_errors):.3f}", f"{np.percentile(abs_errors, 95):.3f}",
            f"{abs_errors.max():.3f}", f"{residuals.mean():.3f}",
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

with tab4:
    if cmp_data and "comparison" in cmp_data:
        cmp_df = pd.DataFrame(cmp_data["comparison"])
        display_cols = [c for c in ["model","MAE","RMSE","R2","MAPE","n_train","n_test"] if c in cmp_df.columns]
        cmp_df_disp = cmp_df[display_cols].copy()
        for col in ["MAE","RMSE","R2","MAPE"]:
            if col in cmp_df_disp.columns:
                cmp_df_disp[col] = cmp_df_disp[col].round(3)

        st.markdown(f"**Best model:** `{cmp_data.get('best_model','Unknown')}` "
                    f"(RMSE={cmp_data.get('best_rmse',0):.2f})")

        # highlight best
        st.dataframe(cmp_df_disp, use_container_width=True, hide_index=True)

        # comparison bar chart
        if "RMSE" in cmp_df.columns:
            fig4 = px.bar(cmp_df, x="model", y="RMSE", color="RMSE",
                          color_continuous_scale="RdYlGn_r",
                          title="Model Comparison by RMSE",
                          text=cmp_df["RMSE"].round(2))
            fig4.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run `python src/multi_model.py` for multi-model comparison.")

with tab5:
    bins = [0, 50, 100, 200, 300, 400, float("inf")]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    pred_cats  = pd.cut(y_pred, bins=bins, labels=labels)
    actual_cats = pd.cut(y_test, bins=bins, labels=labels)

    # per-category error
    cat_errors = pd.DataFrame({"Category": actual_cats, "AbsError": abs_errors})
    cat_err_grp = cat_errors.groupby("Category")["AbsError"].agg(["mean","median","count"]).reset_index()
    cat_err_grp.columns = ["AQI Category","Mean Abs Error","Median Abs Error","Count"]

    fig5 = px.bar(cat_err_grp, x="AQI Category", y="Mean Abs Error",
                  color="Mean Abs Error", color_continuous_scale="YlOrRd",
                  text=cat_err_grp["Mean Abs Error"].round(1),
                  title="Mean Absolute Error by AQI Category")
    fig5.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(cat_err_grp, use_container_width=True, hide_index=True)
