"""
city_coordinates.py
--------------------
Approximate (lat, lon) centroids for the cities present in the CPCB
city_day.csv / city_hour.csv files.

Why this exists: the CPCB CSVs give you a City *name* only -- no
coordinates. To pull a satellite pixel/region (MODIS AOD, Sentinel-5P
NO2/CO/HCHO) for a city via Earth Engine, you need somewhere to point
the query. These are city-centre coordinates, good enough for a
~25km sampling buffer; they are NOT the exact CPCB monitoring-station
coordinates (that level of detail lives in stations.csv's StationId
metadata, which CPCB does not expose lat/lon for either).
"""

from __future__ import annotations

# lat, lon in decimal degrees (WGS84)
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "Ahmedabad": (23.0225, 72.5714),
    "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5730, 80.3572),
    "Amritsar": (31.6340, 74.8723),
    "Bengaluru": (12.9716, 77.5946),
    "Bhopal": (23.2599, 77.4126),
    "Brajrajnagar": (21.8167, 83.9167),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Delhi": (28.7041, 77.1025),
    "Ernakulam": (9.9816, 76.2999),
    "Gurugram": (28.4595, 77.0266),
    "Guwahati": (26.1445, 91.7362),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Jorapokhar": (23.7000, 86.4136),
    "Kochi": (9.9312, 76.2673),
    "Kolkata": (22.5726, 88.3639),
    "Lucknow": (26.8467, 80.9462),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
    "Shillong": (25.5788, 91.8933),
    "Silchar": (24.8333, 92.7789),
    "Talcher": (20.9500, 85.2167),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam": (17.6868, 83.2185),
}


def get_coordinates(city: str) -> tuple[float, float] | None:
    """Case/whitespace-tolerant lookup. Returns None if the city isn't known."""
    if city in CITY_COORDINATES:
        return CITY_COORDINATES[city]
    key = city.strip().title()
    return CITY_COORDINATES.get(key)
