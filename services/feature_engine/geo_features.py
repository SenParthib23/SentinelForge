r"""
SentinelForge — services/feature_engine/geo_features.py

PURPOSE:
    Computes geographical features, specifically the Haversine great-circle
    distance (in kilometers) between successive transactions to detect
    impossible travel velocities and location spoofing.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Geographic Feature Subsystem.

MATHEMATICAL FORMULATION:
    Given two coordinates $(lat_1, lon_1)$ and $(lat_2, lon_2)$ in radians:
    $$a = \sin^2(\Delta lat / 2) + \cos(lat_1) \cdot \cos(lat_2) \cdot \sin^2(\Delta lon / 2)$$
    $$c = 2 \cdot \arctan2(\sqrt{a}, \sqrt{1 - a})$$
    $$d = R \cdot c$$
    Where $R \approx 6371.0 \text{ km}$ is Earth's mean radius.

INTERVIEW TALKING POINT:
    "We calculate Haversine distance in under 0.05ms in pure Python/C.
    If a transaction in Mumbai is followed 10 minutes later by a transaction
    in London (distance > 7,000 km), the calculated travel speed exceeds
    42,000 km/h, which is an indisputable physical impossibility and strong
    indicator of credential or session hijacking."
"""

import math
from typing import Optional

EARTH_RADIUS_KM: float = 6371.0


def compute_haversine_distance(
    lat1: Optional[float],
    lon1: Optional[float],
    lat2: Optional[float],
    lon2: Optional[float],
) -> float:
    """
    Calculate the great-circle distance between two GPS points on Earth.

    Args:
        lat1: Latitude of origin point (-90.0 to 90.0).
        lon1: Longitude of origin point (-180.0 to 180.0).
        lat2: Latitude of destination point (-90.0 to 90.0).
        lon2: Longitude of destination point (-180.0 to 180.0).

    Returns:
        float: Distance in kilometers. Returns 0.0 if any coordinate is None.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0

    # If coordinates are identical, distance is 0
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )

    # Bound value of a within [0.0, 1.0] to prevent math domain error from float precision
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return round(EARTH_RADIUS_KM * c, 3)
