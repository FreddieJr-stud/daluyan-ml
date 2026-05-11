"""NASA POWER supplemental weather features.

Provides solar radiation and specific humidity from NASA POWER hourly API
(MERRA-2 reanalysis, ~55km resolution) for the Pasig River ferry route.

The data is pre-fetched and cached as JSON. Features are matched by
date + hour to the nearest available observation.

Parameters:
    ALLSKY_SFC_SW_DWN: All-sky surface shortwave downward irradiance (W/m2)
    QV2M: Specific humidity at 2 meters (g/kg)

Usage:
    from features.nasa_power import get_nasa_power_features
    feats = get_nasa_power_features(departure_time)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_CACHE_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "nasa_power_hourly.json"
_DATA: dict[str, dict[str, float]] | None = None


def _load_data() -> dict[str, dict[str, float]]:
    """Load and parse the NASA POWER JSON cache."""
    global _DATA
    if _DATA is not None:
        return _DATA

    with open(_CACHE_PATH) as f:
        raw = json.load(f)

    params = raw["properties"]["parameter"]
    _DATA = {
        "solar": params["ALLSKY_SFC_SW_DWN"],
        "humidity": params["QV2M"],
    }
    return _DATA


def get_nasa_power_features(dep_time: datetime) -> dict[str, float]:
    """Get NASA POWER features for a departure time.

    Args:
        dep_time: Departure datetime (Philippine time, UTC+8).

    Returns:
        Dict with solar_radiation_wm2 and specific_humidity_gkg.
    """
    data = _load_data()

    # Convert Philippine time to UTC (subtract 8 hours)
    utc_hour = (dep_time.hour - 8) % 24
    # If subtracting 8 crosses midnight, adjust date
    if dep_time.hour < 8:
        from datetime import timedelta
        utc_date = dep_time.date() - timedelta(days=1)
    else:
        utc_date = dep_time.date()

    # NASA POWER key format: YYYYMMDDHH
    key = f"{utc_date.strftime('%Y%m%d')}{utc_hour:02d}"

    solar = data["solar"].get(key, -999.0)
    humidity = data["humidity"].get(key, -999.0)

    # Handle missing/nighttime solar (-999 or negative)
    if solar < 0:
        solar = 0.0

    # Handle missing humidity
    if humidity < 0:
        humidity = 17.5  # approximate mean for the region

    return {
        "solar_radiation_wm2": solar,
        "specific_humidity_gkg": humidity,
    }
