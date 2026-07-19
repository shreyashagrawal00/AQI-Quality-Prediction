"""
_theme.py — Shared dark-mode CSS injector for all pages.
Import and call inject_dark_css() at the top of each page.
"""

import streamlit as st

# ── Design tokens ────────────────────────────────────────────────────────────
BG_PRIMARY        = "#0a0a0f"
BG_SECONDARY      = "#12121a"
BG_CARD           = "#1a1a2e"
BG_CARD_HOVER     = "#16213e"
ACCENT_CYAN       = "#00d4ff"
ACCENT_BLUE       = "#3b82f6"
ACCENT_PURPLE     = "#a855f7"
ACCENT_GREEN      = "#10b981"
BORDER_COLOR      = "#2d2d44"
TEXT_PRIMARY      = "#e2e8f0"
TEXT_SECONDARY    = "#94a3b8"
TEXT_MUTED        = "#64748b"
GLOW_CYAN         = "rgba(0, 212, 255, 0.15)"
GLOW_BLUE         = "rgba(59, 130, 246, 0.15)"
GLOW_PURPLE       = "rgba(168, 85, 247, 0.15)"


DARK_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset & Base ──────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #0a0a0f !important;
    color: #e2e8f0 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #12121a; }
::-webkit-scrollbar-thumb { background: #2d2d44; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff55; }

/* ── Main content area ────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 1400px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #12121a 100%) !important;
    border-right: 1px solid #2d2d44 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem; }

/* ── Sidebar page navigation links (emoji page names) ─────────────────────── */
[data-testid="stSidebarNav"] a {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
    -webkit-text-fill-color: #94a3b8 !important;
}
[data-testid="stSidebarNav"] a:hover {
    color: #00d4ff !important;
    -webkit-text-fill-color: #00d4ff !important;
    background: rgba(0, 212, 255, 0.08) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] {
    color: #00d4ff !important;
    -webkit-text-fill-color: #00d4ff !important;
    background: rgba(0, 212, 255, 0.12) !important;
    border-left: 2px solid #00d4ff !important;
}
/* Ensure emoji characters inside nav links are always visible */
[data-testid="stSidebarNav"] span {
    -webkit-text-fill-color: initial !important;
    color: inherit !important;
}


/* ── Headers ──────────────────────────────────────────────────────────────── */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem !important;
}
/* Gradient only on non-emoji heading text via a wrapper span */
h1 .gradient-text {
    background: linear-gradient(135deg, #00d4ff, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2 { font-weight: 700 !important; color: #e2e8f0 !important; }
h3 { font-weight: 600 !important; color: #cbd5e1 !important; }

/* ── Streamlit metric widgets ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    border-color: #00d4ff55;
    box-shadow: 0 4px 25px rgba(0, 212, 255, 0.12);
    transform: translateY(-1px);
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-weight: 700 !important;
    font-size: 1.8rem !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #12121a;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #2d2d44;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e3a5f, #1a1a2e) !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff44 !important;
    box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.2rem;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #3b82f6 100%) !important;
    color: #0a0a0f !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 212, 255, 0.45) !important;
    filter: brightness(1.1);
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Input widgets ────────────────────────────────────────────────────────── */
.stSelectbox > div > div, .stNumberInput > div > div,
.stTextInput > div > div, .stDateInput > div > div {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: #00d4ff55 !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
}
.stMultiSelect > div > div {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 8px !important;
}

/* ── Slider ───────────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #00d4ff !important;
    box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
}

/* ── Dataframe / Table ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #2d2d44;
    border-radius: 10px;
    overflow: hidden;
    background: #12121a;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #12121a !important;
    border: 1px solid #2d2d44 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ── Alert / Info / Warning / Error ──────────────────────────────────────── */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid;
    background: #1a1a2e !important;
}
[data-testid="stAlert"][kind="info"] { border-color: #3b82f6 !important; }
[data-testid="stAlert"][kind="warning"] { border-color: #f59e0b !important; }
[data-testid="stAlert"][kind="error"] { border-color: #ef4444 !important; }
[data-testid="stAlert"][kind="success"] { border-color: #10b981 !important; }

/* ── Divider ──────────────────────────────────────────────────────────────── */
hr { border-color: #2d2d44 !important; margin: 1.5rem 0; }

/* ── Download button ──────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: #1a1a2e !important;
    color: #00d4ff !important;
    border: 1px solid #00d4ff44 !important;
    border-radius: 8px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #00d4ff11 !important;
    border-color: #00d4ff !important;
}

/* ── Radio buttons ────────────────────────────────────────────────────────── */
.stRadio > label { color: #94a3b8 !important; }

/* ── Caption / small text ─────────────────────────────────────────────────── */
.stCaption { color: #64748b !important; font-size: 0.78rem !important; }

/* ── Code blocks ──────────────────────────────────────────────────────────── */
code {
    background: #1a1a2e !important;
    color: #00d4ff !important;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important;
}
pre { background: #0d0d1a !important; border: 1px solid #2d2d44 !important; border-radius: 8px; }

/* ── Custom card components ───────────────────────────────────────────────── */
.glass-card {
    background: linear-gradient(135deg, rgba(26,26,46,0.9) 0%, rgba(22,33,62,0.9) 100%);
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
}
.glass-card:hover {
    border-color: rgba(0,212,255,0.3);
    box-shadow: 0 8px 40px rgba(0,212,255,0.1);
    transform: translateY(-2px);
}

/* ── Metric card ──────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2d2d44;
    border-radius: 14px;
    padding: 1.3rem 1rem;
    color: white;
    box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #3b82f6);
}
.metric-card:hover {
    border-color: rgba(0, 212, 255, 0.35);
    box-shadow: 0 8px 30px rgba(0, 212, 255, 0.15);
    transform: translateY(-3px);
}
.metric-card h1 {
    font-size: 2rem !important;
    margin: 0.2rem 0 !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
}
.metric-card p {
    margin: 0 !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}
.metric-card .sub {
    font-size: 0.75rem !important;
    color: #64748b !important;
}

/* ── Hero banner ──────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1729 40%, #0a1628 100%);
    border: 1px solid #2d2d44;
    border-radius: 20px;
    padding: 2.5rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(59,130,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #3b82f6, #a855f7);
}
.hero-banner h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.5rem !important;
    color: #ffffff !important;
    text-shadow: 0 0 30px rgba(0, 212, 255, 0.4);
}
.hero-banner p {
    color: #94a3b8 !important;
    font-size: 1rem !important;
    max-width: 750px;
    line-height: 1.6;
}
.hero-banner .badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
    letter-spacing: 0.04em;
}
.badge-cyan { background: rgba(0,212,255,0.1); border-color: rgba(0,212,255,0.35); color: #00d4ff; }
.badge-blue { background: rgba(59,130,246,0.1); border-color: rgba(59,130,246,0.35); color: #3b82f6; }
.badge-purple { background: rgba(168,85,247,0.1); border-color: rgba(168,85,247,0.35); color: #a855f7; }
.badge-green { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.35); color: #10b981; }

/* ── Category pill ────────────────────────────────────────────────────────── */
.category-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.82rem;
    border: 1px solid currentColor;
    margin: 0.15rem;
}

/* ── Status indicator ─────────────────────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse-dot 2s ease-in-out infinite;
}
.status-dot.green { background: #10b981; box-shadow: 0 0 6px #10b981; }
.status-dot.yellow { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* ── Plotly chart backgrounds ─────────────────────────────────────────────── */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
"""


def inject_dark_css():
    """Inject the global dark-mode CSS into the current page."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def plotly_dark_layout(**kwargs):
    """Return a dict of dark-themed Plotly layout overrides."""
    base = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        title_font=dict(size=15, color="#e2e8f0"),
        xaxis=dict(
            gridcolor="#2d2d44",
            zerolinecolor="#2d2d44",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="#2d2d44",
            zerolinecolor="#2d2d44",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(26,26,46,0.9)",
            bordercolor="#2d2d44",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8"),
        ),
    )
    base.update(kwargs)
    return base
