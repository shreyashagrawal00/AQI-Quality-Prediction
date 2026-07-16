"""
aqi_utils.py
------------
Utility functions for AQI category classification following the
Indian CPCB (Central Pollution Control Board) National AQI standard.

FIX: The original project docs only mention "AQI category generation"
without specifying logic. A common bug in similar projects is hand-rolled
bucket thresholds that don't match CPCB's official breakpoints, or reusing
AQI_Bucket (a label DERIVED FROM AQI) as an input FEATURE to the regressor,
which is a data-leakage bug (the model would implicitly "see" a
transformation of its own target). This module keeps category generation
fully separate and derived only from the model's predicted AQI value,
never used upstream as a feature.
"""

from __future__ import annotations

# CPCB AQI breakpoints -> (category, color)
_AQI_BREAKPOINTS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Satisfactory", "#a3ff00"),
    (101, 200, "Moderate", "#ffff00"),
    (201, 300, "Poor", "#ff7e00"),
    (301, 400, "Very Poor", "#ff0000"),
    (401, 500, "Severe", "#8f3f97"),
]


def aqi_to_category(aqi: float) -> str:
    """Map a numeric AQI value to its CPCB category label."""
    if aqi is None or (isinstance(aqi, float) and aqi != aqi):  # NaN check
        return "Unknown"
    aqi = max(0, float(aqi))
    for low, high, label, _ in _AQI_BREAKPOINTS:
        if low <= aqi <= high:
            return label
    return "Severe"  # anything above 500 is still Severe


def aqi_to_color(aqi: float) -> str:
    """Map a numeric AQI value to its CPCB category color (for dashboards/maps)."""
    if aqi is None or (isinstance(aqi, float) and aqi != aqi):
        return "#9e9e9e"
    aqi = max(0, float(aqi))
    for low, high, label, color in _AQI_BREAKPOINTS:
        if low <= aqi <= high:
            return color
    return "#8f3f97"


def health_message(category: str) -> str:
    """Short, human-readable health advisory for a given AQI category."""
    messages = {
        "Good": "Air quality is satisfactory; minimal health risk.",
        "Satisfactory": "Acceptable air quality; minor breathing discomfort "
                         "to sensitive people.",
        "Moderate": "Breathing discomfort to people with lung, asthma, "
                     "and heart disease.",
        "Poor": "Breathing discomfort to most people on prolonged exposure.",
        "Very Poor": "Respiratory illness on prolonged exposure. "
                      "Avoid outdoor activity.",
        "Severe": "Serious health effects even on healthy people. "
                   "Avoid all outdoor exertion.",
        "Unknown": "Insufficient data to assess air quality.",
    }
    return messages.get(category, messages["Unknown"])


if __name__ == "__main__":
    for val in [10, 75, 150, 250, 350, 450, None]:
        print(val, "->", aqi_to_category(val), aqi_to_color(val))
