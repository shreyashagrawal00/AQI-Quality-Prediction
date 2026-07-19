"""
health_advisory.py
------------------
Expanded health advisory system for Indian CPCB AQI categories.
Returns structured advisory dicts for rich dashboard display.
"""

from __future__ import annotations

# ── CPCB AQI breakpoints ────────────────────────────────────────────────────
_ADVISORIES: dict[str, dict] = {
    "Good": {
        "range": "0–50",
        "color": "#00e400",
        "bg_color": "#e8fce8",
        "icon": "😊",
        "effects": [
            "Minimal impact on health.",
            "Air quality is satisfactory for all groups.",
        ],
        "precautions": [
            "No special precautions needed.",
            "Good time for outdoor activities.",
        ],
        "who_at_risk": "None",
        "outdoor_activity": "✅ Safe for all",
    },
    "Satisfactory": {
        "range": "51–100",
        "color": "#a3ff00",
        "bg_color": "#f4ffe8",
        "icon": "🙂",
        "effects": [
            "Minor breathing discomfort to sensitive people.",
            "Slight irritation for people with respiratory conditions.",
        ],
        "precautions": [
            "Sensitive individuals should limit prolonged outdoor exertion.",
            "People with asthma should keep inhalers handy.",
        ],
        "who_at_risk": "Sensitive groups (asthma, elderly, children)",
        "outdoor_activity": "⚠️ Sensitive groups: limit exertion",
    },
    "Moderate": {
        "range": "101–200",
        "color": "#ffff00",
        "bg_color": "#ffffcc",
        "icon": "😐",
        "effects": [
            "Breathing discomfort to people with lung disease, asthma, and heart disease.",
            "Discomfort to children and elderly on prolonged exposure.",
        ],
        "precautions": [
            "People with heart or lung disease, elderly, and children should reduce prolonged or heavy exertion.",
            "Wear N95/N99 masks when outdoors.",
            "Keep windows closed in high pollution hours (morning & evening).",
        ],
        "who_at_risk": "Children, elderly, heart/lung disease patients",
        "outdoor_activity": "⚠️ Sensitive groups: avoid prolonged outdoor activity",
    },
    "Poor": {
        "range": "201–300",
        "color": "#ff7e00",
        "bg_color": "#fff0e0",
        "icon": "😷",
        "effects": [
            "Breathing discomfort to most people on prolonged exposure.",
            "Serious effects on people with heart or lung disease.",
            "Coughing, throat irritation, eye watering.",
        ],
        "precautions": [
            "Everyone should avoid prolonged outdoor exertion.",
            "Wear N95 masks outdoors.",
            "Keep indoor air clean with air purifiers if available.",
            "Stay hydrated — it helps clear airways.",
            "Avoid exercising outdoors.",
        ],
        "who_at_risk": "General public and sensitive groups",
        "outdoor_activity": "🚫 Avoid outdoor exertion for all",
    },
    "Very Poor": {
        "range": "301–400",
        "color": "#ff0000",
        "bg_color": "#ffe0e0",
        "icon": "🤢",
        "effects": [
            "Respiratory illness on prolonged exposure.",
            "Effects may begin to be felt by healthy people.",
            "Serious aggravation of heart or lung disease.",
            "Premature mortality in people with heart/lung disease.",
        ],
        "precautions": [
            "Stay indoors and keep activity levels low.",
            "Seal gaps in windows/doors to reduce indoor pollution.",
            "Use air purifiers with HEPA filters.",
            "Wear P100/N99 respirators if you must go out.",
            "Avoid cooking with solid fuels indoors.",
            "Seek medical advice if experiencing symptoms.",
        ],
        "who_at_risk": "General public — everyone at risk",
        "outdoor_activity": "🚫 Stay indoors. Avoid all outdoor activity.",
    },
    "Severe": {
        "range": "401–500+",
        "color": "#8f3f97",
        "bg_color": "#f0e0f8",
        "icon": "☠️",
        "effects": [
            "Serious health effects even on healthy individuals.",
            "Respiratory and cardiovascular complications.",
            "Risk of premature death for elderly and those with chronic disease.",
            "Significant aggravation of existing lung and heart conditions.",
        ],
        "precautions": [
            "Stay indoors at all times.",
            "Do NOT perform any outdoor physical activity.",
            "Wear P100 respirators if venturing outdoors is unavoidable.",
            "Seek immediate medical attention if breathing difficulty occurs.",
            "Schools and outdoor public gatherings should be cancelled.",
            "Run air purifiers continuously.",
            "Check on elderly neighbors and family.",
        ],
        "who_at_risk": "EVERYONE — including healthy adults",
        "outdoor_activity": "☠️ Emergency: Avoid all outdoor exposure",
    },
    "Unknown": {
        "range": "N/A",
        "color": "#9e9e9e",
        "bg_color": "#f5f5f5",
        "icon": "❓",
        "effects": ["Insufficient data to assess air quality."],
        "precautions": ["Monitor local air quality reports."],
        "who_at_risk": "Unknown",
        "outdoor_activity": "❓ Insufficient data",
    },
}

_AQI_BREAKPOINTS = [
    (0,   50,  "Good"),
    (51,  100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, float("inf"), "Severe"),
]


def aqi_to_category(aqi: float) -> str:
    if aqi is None or (isinstance(aqi, float) and aqi != aqi):
        return "Unknown"
    aqi = max(0.0, float(aqi))
    for low, high, label in _AQI_BREAKPOINTS:
        if low <= aqi <= high:
            return label
    return "Severe"


def get_advisory(aqi_or_category) -> dict:
    """
    Returns the full advisory dict for a given AQI value or category string.

    Dict keys: range, color, bg_color, icon, effects, precautions,
                who_at_risk, outdoor_activity
    """
    if isinstance(aqi_or_category, str):
        category = aqi_or_category
    else:
        category = aqi_to_category(aqi_or_category)
    return _ADVISORIES.get(category, _ADVISORIES["Unknown"])


def health_message(category: str) -> str:
    """Short single-line summary for backward compatibility with app.py."""
    adv = _ADVISORIES.get(category, _ADVISORIES["Unknown"])
    return adv["effects"][0] if adv["effects"] else "Unknown AQI."


def all_categories() -> list[str]:
    return ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
