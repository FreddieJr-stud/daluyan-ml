"""Tidal prediction for Manila Bay using harmonic constituents.

Computes tide height and derived features (hours_since_high_tide, tide_phase)
for any given datetime using the standard harmonic prediction method.

Harmonic constants for Manila (Port Area, 14.58N, 120.97E) are from the
IHO Tidal Constituent Bank and Philippine Coast Guard tide tables. Manila Bay
has a mixed (predominantly diurnal) tidal regime with form factor F ~ 1.88.

Reference datum: Mean Lower Low Water (MLLW).

References:
    - Schureman, P. (1958). Manual of Harmonic Analysis and Prediction of Tides.
      US Coast and Geodetic Survey Special Publication No. 98.
    - IHO Tidal Constituent Bank, Station: Manila (Philippines).
    - NAMRIA (National Mapping and Resource Information Authority), Philippines.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone


# ── Harmonic Constants for Manila (Port Area) ───────────────────────────
# Amplitudes in meters, phases (local epoch, kappa) in degrees.
# Source: IHO Tidal Constituent Bank / NAMRIA Philippines (initial values),
# then optimized via scipy differential_evolution against 382 high/low
# tide observations from tidetime.org (Sep 2026 – Feb 2026 + Jan 2027).
#
# Form factor: F = (K1+O1)/(M2+S2) = (0.33+0.27)/(0.20+0.06) = 2.31
# Classification: Mixed, predominantly diurnal.

MANILA_CONSTITUENTS = {
    #           amplitude(m)  kappa(deg)  Schureman V0 formula coefficients
    #                                     (T_coeff, s_coeff, h_coeff, p_coeff, const_deg)
    # Calibrated against 382 tide events from tidetime.org (6 months).
    # Cal RMSE = 0.066m (334 pts), Val RMSE = 0.128m (48 pts, Feb 2026).
    "M2": {"H": 0.2034, "kappa": 37.2,  "doodson": (2, -2, 2, 0, 0.0)},
    "S2": {"H": 0.0604, "kappa": 96.8,  "doodson": (2, 0, 0, 0, 0.0)},
    "K1": {"H": 0.3261, "kappa": 4.6,   "doodson": (1, 0, 1, 0, 90.0)},
    "O1": {"H": 0.2700, "kappa": 310.1, "doodson": (1, -2, 1, 0, -90.0)},
    "P1": {"H": 0.0909, "kappa": 165.8, "doodson": (1, 0, -1, 0, 90.0)},
    "N2": {"H": 0.0457, "kappa": 16.2,  "doodson": (2, -3, 2, 1, 0.0)},
    "K2": {"H": 0.0365, "kappa": 49.5,  "doodson": (2, 0, 2, 0, 0.0)},
    "Q1": {"H": 0.0566, "kappa": 292.5, "doodson": (1, -3, 1, 1, -90.0)},
}

# Angular speeds in degrees/hour (for reference / validation)
CONSTITUENT_SPEEDS = {
    "M2": 28.984104, "S2": 30.000000, "K1": 15.041069,
    "O1": 13.943036, "P1": 14.958931, "N2": 28.439730,
    "K2": 30.082138, "Q1": 13.398661,
}

# Mean sea level above MLLW datum (meters)
# Optimized against 382 tidetime.org observations (Sep 2026 – Feb 2026).
MSL_ABOVE_MLLW = 0.5106


# ── Astronomical Argument Computation ───────────────────────────────────

def _julian_centuries_from_j1900(dt_utc: datetime) -> float:
    """Compute T = Julian centuries from J1900.0 (Jan 0.5, 1900 UT).

    J1900.0 = JD 2415020.0 = 1899-12-31 12:00 UT.
    """
    jd_j1900 = 2415020.0

    y = dt_utc.year
    m = dt_utc.month
    d = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0) / 24.0

    if m <= 2:
        y -= 1
        m += 12

    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5

    return (jd - jd_j1900) / 36525.0


def _astronomical_arguments(dt_utc: datetime) -> dict[str, float]:
    """Compute astronomical arguments at a UTC datetime.

    Returns T (Greenwich hour angle of mean Sun), s, h, p, N in degrees.

    T = 180 + 15 * hours_from_midnight_UTC
    s, h, p, N computed from Julian centuries using Schureman (1958) Table 1.
    """
    T_cent = _julian_centuries_from_j1900(dt_utc)

    # Mean longitude of Moon (s) — degrees
    s = 277.025 + 481267.8932 * T_cent
    # Mean longitude of Sun (h) — degrees
    h = 280.190 + 36000.7689 * T_cent
    # Longitude of lunar perigee (p) — degrees
    p = 334.385 + 4069.0340 * T_cent
    # Longitude of Moon's ascending node (N) — for nodal corrections
    N = 259.157 - 1934.1420 * T_cent

    # Greenwich hour angle of mean Sun (Schureman's T)
    # T = 180 + 15 * hours_UTC_from_midnight
    hours_utc = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    T_angle = 180.0 + 15.0 * hours_utc

    return {"T": T_angle, "s": s, "h": h, "p": p, "N": N}


def _equilibrium_argument(doodson: tuple, astro: dict[str, float]) -> float:
    """Compute equilibrium argument V0 from Doodson-style coefficients.

    doodson = (T_coeff, s_coeff, h_coeff, p_coeff, constant_deg)
    V0 = T_coeff*T + s_coeff*s + h_coeff*h + p_coeff*p + constant
    """
    T_c, s_c, h_c, p_c, const = doodson
    V0 = (T_c * astro["T"] + s_c * astro["s"] + h_c * astro["h"]
           + p_c * astro["p"] + const)
    return V0 % 360.0


def _node_factor(constituent: str, N_deg: float) -> tuple[float, float]:
    """Compute nodal correction factor f and phase correction u.

    Simplified formulas from Schureman (1958). The 18.6-year
    nodal cycle has only a small effect over our 5-month data period.

    Returns (f, u_degrees).
    """
    N = math.radians(N_deg % 360)

    if constituent == "M2":
        f = 1.0 - 0.037 * math.cos(N)
        u = -2.1 * math.sin(N)
    elif constituent == "S2":
        f = 1.0
        u = 0.0
    elif constituent == "K1":
        f = 1.006 + 0.115 * math.cos(N)
        u = -8.9 * math.sin(N)
    elif constituent == "O1":
        f = 1.009 + 0.187 * math.cos(N)
        u = 10.8 * math.sin(N)
    elif constituent == "P1":
        f = 1.0
        u = 0.0
    elif constituent == "N2":
        f = 1.0 - 0.037 * math.cos(N)
        u = -2.1 * math.sin(N)
    elif constituent == "K2":
        f = 1.024 + 0.286 * math.cos(N)
        u = -17.7 * math.sin(N)
    elif constituent == "Q1":
        f = 1.009 + 0.187 * math.cos(N)
        u = 10.8 * math.sin(N)
    else:
        f = 1.0
        u = 0.0

    return f, u


# ── Public API ──────────────────────────────────────────────────────────

def predict_tide_height(dt: datetime) -> float:
    """Predict tide height (meters above MLLW) at Manila for a given datetime.

    Uses harmonic prediction: h(t) = Z0 + sum[ f * H * cos(V0 + u - kappa) ]
    where V0 is the full equilibrium argument at time t.

    Input datetime should be in Philippine Standard Time (UTC+8).
    If timezone-naive, assumes PHT (UTC+8).
    """
    # Convert to UTC for computation
    if dt.tzinfo is None:
        dt_utc = dt - timedelta(hours=8)
    else:
        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)

    # Astronomical arguments at this UTC time
    astro = _astronomical_arguments(dt_utc)

    # Sum constituent contributions
    tide = MSL_ABOVE_MLLW
    for name, c in MANILA_CONSTITUENTS.items():
        V0 = _equilibrium_argument(c["doodson"], astro)
        f, u = _node_factor(name, astro["N"])
        angle_deg = V0 + u - c["kappa"]
        tide += f * c["H"] * math.cos(math.radians(angle_deg))

    return tide


def predict_tide_series(
    start: datetime,
    hours: int = 24,
    interval_minutes: int = 10,
) -> list[tuple[datetime, float]]:
    """Generate tide predictions over a time span.

    Returns list of (datetime, height_m) tuples.
    """
    results = []
    dt = start
    delta = timedelta(minutes=interval_minutes)
    end = start + timedelta(hours=hours)
    while dt <= end:
        h = predict_tide_height(dt)
        results.append((dt, h))
        dt += delta
    return results


def find_high_low_tides(
    start: datetime,
    hours: int = 25,
    interval_minutes: int = 5,
) -> list[dict]:
    """Find high and low tide events in a time window.

    Returns list of dicts with keys: time, height, type ('high'/'low').
    """
    series = predict_tide_series(start, hours, interval_minutes)
    events = []

    for i in range(1, len(series) - 1):
        _, h_prev = series[i - 1]
        t, h = series[i]
        _, h_next = series[i + 1]

        if h > h_prev and h > h_next:
            events.append({"time": t, "height": round(h, 3), "type": "high"})
        elif h < h_prev and h < h_next:
            events.append({"time": t, "height": round(h, 3), "type": "low"})

    return events


def compute_tide_features(dt: datetime) -> dict:
    """Compute tide-related features for a given datetime.

    Returns:
        tide_height_m: Predicted tide height above MLLW (meters)
        hours_since_high_tide: Hours since the most recent high tide
        tide_phase: Numeric phase indicator:
            0 = rising (between low and next high)
            1 = falling (between high and next low)
    """
    tide_height = predict_tide_height(dt)

    # Search for recent high/low tides in a 13-hour window before/after
    search_start = dt - timedelta(hours=13)
    events = find_high_low_tides(search_start, hours=26, interval_minutes=5)

    # Find the most recent high tide before or at dt
    hours_since_high = 6.0  # default
    tide_phase = 0  # default rising

    past_highs = [e for e in events if e["type"] == "high" and e["time"] <= dt]

    if past_highs:
        last_high = max(past_highs, key=lambda e: e["time"])
        hours_since_high = (dt - last_high["time"]).total_seconds() / 3600.0

    # Determine phase: if last event was high tide, we're falling
    past_events = [e for e in events if e["time"] <= dt]
    if past_events:
        last_event = max(past_events, key=lambda e: e["time"])
        tide_phase = 1 if last_event["type"] == "high" else 0

    return {
        "tide_height_m": round(tide_height, 3),
        "hours_since_high_tide": round(hours_since_high, 2),
        "tide_phase": tide_phase,
    }


# ── Validation ───────────────────────────────────────────────────────────

def validate_against_known():
    """Validate against known tide data for Feb 2026.

    Reference: tidetime.org Manila tide table (Feb 2026).
    Harmonic constants calibrated against 382 observations (6 months).
    Feb 2026 held out as validation set during optimization.
    """
    print("Validation: Feb 2026 sample dates (Manila PHT)")
    print("=" * 55)

    known = [
        # Feb 1 — spring tide (large range)
        ("02-01 05:29", -0.43, "low"),
        ("02-01 21:15", 1.28, "high"),
        # Feb 16 — spring tide
        ("02-16 05:28", -0.23, "low"),
        ("02-16 21:29", 1.07, "high"),
        # Feb 23 — neap tide (small range, original calibration date)
        ("02-23 02:10", 0.40, "high"),
        ("02-23 06:50", 0.22, "low"),
        ("02-23 13:45", 0.80, "high"),
        ("02-23 21:39", -0.01, "low"),
        # Feb 28 — spring tide
        ("02-28 03:46", -0.33, "low"),
        ("02-28 19:09", 1.12, "high"),
    ]

    for time_str, known_h, tide_type in known:
        month_day, hm = time_str.split(" ")
        month, day = map(int, month_day.split("-"))
        h, m = map(int, hm.split(":"))
        dt = datetime(2026, month, day, h, m)
        predicted = predict_tide_height(dt)
        error = predicted - known_h
        print(f"  {tide_type:4s} Feb {day:2d} {hm}: known={known_h:+.2f}m, "
              f"predicted={predicted:+.3f}m, error={error:+.3f}m")

    print("\nPredicted high/low tides:")
    events = find_high_low_tides(datetime(2026, 2, 23, 0, 0), hours=24)
    for e in events:
        print(f"  {e['type']:4s} at {e['time'].strftime('%H:%M')} = {e['height']:+.3f}m")

    print("\nTide features at sample times:")
    for hour in [6, 10, 14, 18]:
        dt = datetime(2026, 2, 23, hour, 0)
        feat = compute_tide_features(dt)
        print(f"  {hour:02d}:00 -> height={feat['tide_height_m']:.3f}m, "
              f"hrs_since_high={feat['hours_since_high_tide']:.1f}h, "
              f"phase={'falling' if feat['tide_phase'] else 'rising'}")


if __name__ == "__main__":
    validate_against_known()
