"""Station encoding, hop count, and route heading utilities."""

from __future__ import annotations

import math

from config import STATION_ORDER, STATION_BY_NAME


def encode_station(name: str) -> int:
    """Encode station name to ordinal (0=Napindan east, 8=Escolta west)."""
    return STATION_ORDER.get(name, -1)


def compute_hop_count(departure: str, arrival: str) -> int:
    """Number of station hops between departure and arrival."""
    dep = STATION_ORDER.get(departure, 0)
    arr = STATION_ORDER.get(arrival, 0)
    return abs(arr - dep)


def encode_direction(direction: str) -> int:
    """0 = downstream (westward), 1 = upstream (eastward)."""
    return 1 if direction == "upstream" else 0


def compute_route_heading(departure: str, arrival: str) -> float:
    """Compute bearing (degrees) from departure to arrival station.

    Returns 0-360 where 0=North, 90=East, 180=South, 270=West.
    """
    dep = STATION_BY_NAME.get(departure)
    arr = STATION_BY_NAME.get(arrival)
    if dep is None or arr is None:
        return 0.0

    lat1 = math.radians(dep["lat"])
    lon1 = math.radians(dep["lon"])
    lat2 = math.radians(arr["lat"])
    lon2 = math.radians(arr["lon"])

    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y)) % 360
    return bearing


def resolve_wind_component(
    wind_speed_kn: float,
    wind_direction_deg: float,
    route_heading_deg: float,
) -> float:
    """Resolve wind along route direction.

    Positive = headwind (against travel direction), Negative = tailwind.

    wind_direction_deg: meteorological convention — direction wind comes FROM.
    route_heading_deg: direction ferry is traveling TO.
    """
    if wind_speed_kn is None or wind_direction_deg is None:
        return 0.0
    angle_diff = math.radians(wind_direction_deg - route_heading_deg)
    return wind_speed_kn * math.cos(angle_diff)


def resolve_wind_cross_component(
    wind_speed_kn: float,
    wind_direction_deg: float,
    route_heading_deg: float,
) -> float:
    """Resolve wind perpendicular to route direction (cross-wind).

    Positive = wind pushing ferry to starboard.
    """
    if wind_speed_kn is None or wind_direction_deg is None:
        return 0.0
    angle_diff = math.radians(wind_direction_deg - route_heading_deg)
    return wind_speed_kn * math.sin(angle_diff)


def resolve_current_component(
    current_velocity_ms: float,
    current_direction_deg: float,
    route_heading_deg: float,
) -> float:
    """Resolve ocean current along route direction.

    Positive = current flowing WITH travel direction (assists ferry).
    Negative = current flowing AGAINST travel (increases consumption).

    current_direction_deg: oceanographic convention — direction current flows TO.
    route_heading_deg: direction ferry is traveling TO.
    """
    if current_velocity_ms is None or current_direction_deg is None:
        return 0.0
    # Current direction = where current flows TO, same convention as route heading
    # cos(0) = 1 when current aligns with travel → positive = assisting
    angle_diff = math.radians(current_direction_deg - route_heading_deg)
    return current_velocity_ms * math.cos(angle_diff)


def resolve_wave_component(
    wave_height_m: float,
    wave_direction_deg: float,
    route_heading_deg: float,
) -> float:
    """Resolve wave impact along route direction.

    Positive = waves coming FROM ahead (head seas, more resistance).
    Negative = following seas.

    wave_direction_deg: direction waves come FROM (meteorological convention).
    """
    if wave_height_m is None or wave_direction_deg is None:
        return 0.0
    angle_diff = math.radians(wave_direction_deg - route_heading_deg)
    return wave_height_m * math.cos(angle_diff)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two GPS points in kilometres."""
    R = 6371.0  # Earth radius in km
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def station_pair_distance_km(departure: str, arrival: str) -> float:
    """Straight-line distance between two stations in km."""
    dep = STATION_BY_NAME.get(departure)
    arr = STATION_BY_NAME.get(arrival)
    if dep is None or arr is None:
        return 0.0
    return haversine_km(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
