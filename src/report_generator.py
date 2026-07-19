"""
report_generator.py
-------------------
Generates a multi-page PDF report of AQI, HCHO, and fire analysis.

Uses fpdf2 (lightweight, no LaTeX dependency).
Requires: pip install fpdf2 kaleido

Usage:
    from report_generator import generate_report
    pdf_bytes = generate_report(title="India AQI Report", sections=[...])
    # then: st.download_button("Download PDF", pdf_bytes, "report.pdf")
"""

from __future__ import annotations

import io
import datetime
import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HAS_FPDF = False
try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    pass

_HAS_KALEIDO = False
try:
    import plotly.io as pio
    _HAS_KALEIDO = True
except ImportError:
    pass


# ── CPCB colour helpers ────────────────────────────────────────────────────
def _aqi_color_rgb(aqi: float) -> tuple[int, int, int]:
    if aqi <= 50:    return (0, 228, 0)
    if aqi <= 100:   return (163, 255, 0)
    if aqi <= 200:   return (255, 255, 0)
    if aqi <= 300:   return (255, 126, 0)
    if aqi <= 400:   return (255, 0, 0)
    return (143, 63, 151)


def _fig_to_png_bytes(fig) -> bytes | None:
    """Convert a Plotly figure to PNG bytes using kaleido."""
    if not _HAS_KALEIDO:
        return None
    try:
        import plotly.io as pio
        buf = io.BytesIO()
        fig.write_image(buf, format="png", width=700, height=400)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


class AQIReport:
    """
    Builds a PDF report page-by-page.

    Usage:
        r = AQIReport("India AQI Report")
        r.add_section_header("National AQI Overview")
        r.add_stats_table(stats_df)
        r.add_figure(plotly_fig, caption="Monthly AQI trend")
        pdf_bytes = r.output()
    """

    # A4 dimensions in mm
    _W, _H = 210, 297
    _MARGIN = 15
    _LINE_H = 7

    def __init__(self, title: str = "AQI Analysis Report"):
        if not _HAS_FPDF:
            raise ImportError(
                "fpdf2 is not installed. Run: pip install fpdf2"
            )
        self.title = title
        self._pdf = FPDF()
        self._pdf.set_auto_page_break(auto=True, margin=20)
        self._pdf.add_page()
        self._pdf.set_margins(self._MARGIN, self._MARGIN, self._MARGIN)
        self._add_cover()

    # ── internal helpers ─────────────────────────────────────────────────

    def _add_cover(self):
        pdf = self._pdf
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(30, 58, 138)   # deep blue
        pdf.ln(40)
        pdf.multi_cell(0, 12, self.title, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(80, 80, 80)
        pdf.ln(6)
        pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, "Satellite-based Surface AQI & HCHO Hotspot Analysis Platform", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        # divider
        pdf.set_draw_color(30, 58, 138)
        pdf.set_line_width(0.8)
        pdf.line(self._MARGIN, pdf.get_y(), self._W - self._MARGIN, pdf.get_y())
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0)

    def _new_page(self):
        self._pdf.add_page()
        self._pdf.set_margins(self._MARGIN, self._MARGIN, self._MARGIN)

    # ── public API ───────────────────────────────────────────────────────

    def add_section_header(self, text: str):
        pdf = self._pdf
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 210, 240)
        pdf.set_line_width(0.3)
        pdf.line(self._MARGIN, pdf.get_y(), self._W - self._MARGIN, pdf.get_y())
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def add_paragraph(self, text: str):
        pdf = self._pdf
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, self._LINE_H, text)
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)

    def add_metric_row(self, metrics: dict[str, Any]):
        """Display key-value pairs in a single row."""
        pdf = self._pdf
        pdf.set_font("Helvetica", "B", 10)
        n = len(metrics)
        col_w = (self._W - 2 * self._MARGIN) / max(n, 1)
        for key, val in metrics.items():
            pdf.set_fill_color(240, 245, 255)
            pdf.rect(pdf.get_x(), pdf.get_y(), col_w - 2, 14, style="F")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(col_w - 2, 6, str(key), align="C")
            pdf.set_x(pdf.get_x() - (col_w - 2) + 2)
            pdf.set_y(pdf.get_y() + 6)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 58, 138)
            pdf.cell(col_w - 2, 7, str(val), align="C")
            pdf.set_xy(pdf.get_x() + 2, pdf.get_y() - 6)
        pdf.ln(18)
        pdf.set_text_color(0, 0, 0)

    def add_stats_table(self, df: pd.DataFrame, caption: str = ""):
        """Render a pandas DataFrame as a PDF table (max 8 columns shown)."""
        pdf = self._pdf
        df = df.copy().reset_index(drop=True)
        cols = list(df.columns)[:8]
        df = df[cols]

        if caption:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, caption, new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)

        col_w = (self._W - 2 * self._MARGIN) / len(cols)
        # header
        pdf.set_fill_color(30, 58, 138)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 8)
        for col in cols:
            pdf.cell(col_w, 7, str(col)[:16], border=1, fill=True, align="C")
        pdf.ln()
        # rows
        pdf.set_font("Helvetica", "", 8)
        for i, row in df.iterrows():
            if i % 2 == 0:
                pdf.set_fill_color(240, 245, 255)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(30, 30, 30)
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    cell_text = f"{val:.2f}"
                else:
                    cell_text = str(val)[:20]
                pdf.cell(col_w, 6, cell_text, border=1, fill=True, align="C")
            pdf.ln()
        pdf.ln(3)
        pdf.set_text_color(0, 0, 0)

    def add_figure(self, fig, caption: str = "", width_mm: float = 175):
        """Embed a Plotly figure as PNG image (requires kaleido)."""
        png_bytes = _fig_to_png_bytes(fig)
        if png_bytes is None:
            self.add_paragraph(f"[Chart: {caption} — install kaleido for inline images]")
            return
        img_buf = io.BytesIO(png_bytes)
        aspect = 400 / 700
        height_mm = width_mm * aspect
        if self._pdf.get_y() + height_mm > self._H - 25:
            self._new_page()
        self._pdf.image(img_buf, x=self._MARGIN, w=width_mm)
        if caption:
            self._pdf.set_font("Helvetica", "I", 9)
            self._pdf.set_text_color(100, 100, 100)
            self._pdf.cell(0, 5, caption, new_x="LMARGIN", new_y="NEXT")
            self._pdf.set_text_color(0, 0, 0)
        self._pdf.ln(3)

    def add_aqi_category_legend(self):
        """Print the CPCB AQI colour legend."""
        self.add_section_header("AQI Category Reference (CPCB India)")
        categories = [
            ("Good",         "0–50",   (0, 228, 0)),
            ("Satisfactory", "51–100", (163, 255, 0)),
            ("Moderate",     "101–200",(255, 255, 0)),
            ("Poor",         "201–300",(255, 126, 0)),
            ("Very Poor",    "301–400",(255, 0, 0)),
            ("Severe",       "401–500+",(143, 63, 151)),
        ]
        pdf = self._pdf
        for cat, rng, rgb in categories:
            pdf.set_fill_color(*rgb)
            pdf.rect(self._MARGIN, pdf.get_y(), 8, 6, style="F")
            pdf.set_x(self._MARGIN + 10)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(40, 6, cat)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, rng, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    def output(self) -> bytes:
        """Return the PDF as bytes."""
        return bytes(self._pdf.output())


def report_available() -> bool:
    return _HAS_FPDF


def generate_report(
    title: str,
    stats_dict: dict | None = None,
    figures: list[tuple] | None = None,  # list of (fig, caption)
    tables: list[tuple] | None = None,   # list of (df, caption)
    include_legend: bool = True,
) -> bytes | None:
    """
    High-level convenience wrapper.

    Parameters
    ----------
    title : str
    stats_dict : dict  — shown as metric cards on page 1
    figures : list of (plotly_fig, caption_str)
    tables  : list of (pd.DataFrame, caption_str)
    include_legend : bool

    Returns PDF bytes or None if fpdf2 not installed.
    """
    if not _HAS_FPDF:
        return None

    r = AQIReport(title)

    if stats_dict:
        r.add_section_header("Summary Statistics")
        r.add_metric_row(stats_dict)

    if include_legend:
        r.add_aqi_category_legend()

    if figures:
        r.add_section_header("Charts & Maps")
        for fig, cap in figures:
            r.add_figure(fig, cap)

    if tables:
        r.add_section_header("Data Tables")
        for df, cap in tables:
            r.add_stats_table(df, cap)

    return r.output()
