"""Extract trip-level features from DuckDB for Model 1A training.

Joins segments + statistics + ridership + weather and computes derived
features.  Saves the result to the feature store as a Parquet file.

Usage:
    python -m data.extract_trip_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import polars as pl

# Allow running as `python -m data.extract_trip_features` from Models/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DUCKDB_PATH,
    FEATURE_STORE_DIR,
    MAX_PASSENGERS,
    TRIP_FEATURES,
    TRIP_TARGET,
)
from features.station_graph import (
    compute_hop_count,
    compute_route_heading,
    encode_direction,
    encode_station,
    resolve_current_component,
    resolve_wave_component,
    resolve_wind_component,
    resolve_wind_cross_component,
)
from features.nasa_power import get_nasa_power_features
from features.tide import compute_tide_features
from features.water_physics import compute_water_physics_features


def _compute_ittc_features(speed_kn: float, sst_c: float) -> dict[str, float]:
    """Compute ITTC water physics features for a trip segment."""
    return compute_water_physics_features(speed_kn, sst_c)


# ── Quality filters ────────────────────────────────────────────────────────
MIN_DISTANCE_KM = 0.1
MIN_DURATION_S = 30
MAX_SOC_DELTA = 50.0  # remove clearly broken segments
def _query_raw_trips(db: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Fetch the base join of segments + stats + ridership."""
    sql = """
        SELECT
            s.segment_id,
            s.departure_station,
            s.arrival_station,
            s.direction,
            s.departure_time,
            st.distance_km,
            st.avg_speed,
            st.duration_seconds,
            st.first_soc,
            st.last_soc,
            st.first_hv_capacity,
            st.last_hv_capacity,
            sr.passengers_on_board,
            sr.passenger_load_ratio
        FROM segments s
        JOIN segment_statistics st ON s.segment_id = st.segment_id
        LEFT JOIN segment_ridership sr ON s.segment_id = sr.segment_id
        WHERE s.departure_time IS NOT NULL
        ORDER BY s.departure_time
    """
    df = db.execute(sql).fetchdf()
    return pl.from_pandas(df)


def _match_weather(
    db: duckdb.DuckDBPyConnection,
    date_str: str,
    hour: int,
) -> dict:
    """Find weather row closest to the given date + hour."""
    rows = db.execute(
        "SELECT * FROM weather_hourly WHERE date = ? ORDER BY time",
        [date_str],
    ).fetchall()
    if not rows:
        return {}

    cols = [d[0] for d in db.description]

    # Find row whose hour is closest to the departure hour
    best = None
    best_diff = 999
    for row in rows:
        row_dict = dict(zip(cols, row))
        # time column looks like '2025-09-30T14:00'
        try:
            row_hour = int(row_dict["time"].split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        diff = abs(row_hour - hour)
        if diff < best_diff:
            best_diff = diff
            best = row_dict
    return best or {}


def extract(verbose: bool = True) -> pl.DataFrame:
    """Build the complete trip-level feature DataFrame."""
    db = duckdb.connect(DUCKDB_PATH, read_only=True)

    # Step 1: raw join
    raw = _query_raw_trips(db)
    if verbose:
        print(f"Raw segments from DuckDB: {len(raw)}")

    # Step 2: quality filters
    filtered = raw.filter(
        (pl.col("direction") != "unknown")
        & (pl.col("departure_station") != "unknown")
        & (pl.col("arrival_station") != "unknown")
        & (pl.col("distance_km") >= MIN_DISTANCE_KM)
        & (pl.col("duration_seconds") >= MIN_DURATION_S)
        & ((pl.col("first_soc") - pl.col("last_soc")) >= 0)
        & ((pl.col("first_soc") - pl.col("last_soc")) <= MAX_SOC_DELTA)
    )
    if verbose:
        print(f"After quality filters: {len(filtered)}")

    # Step 3: compute median passengers for imputation
    pob_col = filtered["passengers_on_board"].drop_nulls()
    median_pob = int(pob_col.median()) if len(pob_col) > 0 else 15

    # Step 4: build feature rows
    records: list[dict] = []
    for row in filtered.iter_rows(named=True):
        dep = row["departure_station"]
        arr = row["arrival_station"]
        direction = row["direction"]
        dep_time = row["departure_time"]

        # Temporal
        hour = dep_time.hour
        date_str = dep_time.strftime("%Y-%m-%d")

        # Station encoding
        dep_enc = encode_station(dep)
        arr_enc = encode_station(arr)
        hops = compute_hop_count(dep, arr)
        dir_enc = encode_direction(direction)

        # Route heading for wind resolution
        route_heading = compute_route_heading(dep, arr)

        # Weather
        wx = _match_weather(db, date_str, hour)

        wind_speed = wx.get("wind_speed_kn") or 0.0
        wind_dir = wx.get("wind_direction_deg") or 0.0
        wind_comp = resolve_wind_component(wind_speed, wind_dir, route_heading)
        wind_cross = resolve_wind_cross_component(wind_speed, wind_dir, route_heading)

        # v4: Ocean current & wave direction
        current_vel = wx.get("ocean_current_velocity_ms") or 0.0
        current_dir = wx.get("ocean_current_direction_deg") or 0.0
        current_comp = resolve_current_component(current_vel, current_dir, route_heading)

        wave_dir_deg = wx.get("wave_direction_deg") or 0.0
        wave_height = wx.get("wave_height_m") or 0.0
        wave_comp = resolve_wave_component(wave_height, wave_dir_deg, route_heading)
        wave_period = wx.get("wave_period_s") or 0.0

        # Passengers — impute nulls with median
        pob = row["passengers_on_board"]
        if pob is None:
            pob = median_pob
        load_ratio = pob / MAX_PASSENGERS

        # Tide features at departure time
        tide = compute_tide_features(dep_time)

        # Speed
        avg_speed = row["avg_speed"] or 0.0

        # Target
        soc_delta = row["first_soc"] - row["last_soc"]

        records.append(
            {
                # Metadata (not features, but kept for splitting / analysis)
                "segment_id": row["segment_id"],
                "departure_time": dep_time,
                "departure_station": dep,
                "arrival_station": arr,
                "direction": direction,
                # Features
                "departure_station_encoded": dep_enc,
                "arrival_station_encoded": arr_enc,
                "hop_count": hops,
                "direction_encoded": dir_enc,
                "route_distance_km": row["distance_km"],
                "start_soc": row["first_soc"],
                "start_hv_capacity": row["first_hv_capacity"],
                "target_speed": avg_speed,
                "target_speed_squared": avg_speed**2,
                "passengers_on_board": pob,
                "passenger_load_ratio": load_ratio,
                "temperature_c": wx.get("temperature_c") or 0.0,
                "relative_humidity": wx.get("relative_humidity") or 0.0,
                "wind_speed_kn": wind_speed,
                "wind_direction_deg": wind_dir,
                "wind_component_along_route": wind_comp,
                "precipitation_mm": wx.get("precipitation_mm") or 0.0,
                "wave_height_m": wx.get("wave_height_m") or 0.0,
                "hour_of_day": hour,
                "is_peak_hour": int(7 <= hour <= 9 or 16 <= hour <= 19),
                # v2 interaction features
                "speed_distance_interaction": avg_speed * row["distance_km"],
                "energy_rate_proxy": (avg_speed ** 2) * hops,
                "wind_cross_component": wind_cross,
                "soc_per_km_capacity": row["first_soc"] / (row["distance_km"] + 0.01),
                # v3 tide features
                "tide_height_m": tide["tide_height_m"],
                "hours_since_high_tide": tide["hours_since_high_tide"],
                "tide_phase": tide["tide_phase"],
                # v4 ocean current & wave features
                "current_component_along_route": current_comp,
                "current_velocity_ms": current_vel,
                "wave_component_along_route": wave_comp,
                "wave_period_s": wave_period,
                # v5 pressure, wind gusts, day-of-week
                "pressure_msl_hpa": wx.get("pressure_msl_hpa") or 1013.25,
                "wind_gusts_kn": wx.get("wind_gusts_kn") or 0.0,
                "is_weekend": int(dep_time.weekday() >= 5),
                # v5-sst: sea surface temperature
                "sea_surface_temperature_c": wx.get("sea_surface_temperature_c") or 28.3,
                # v10 ITTC water physics features
                **_compute_ittc_features(avg_speed, wx.get("sea_surface_temperature_c") or 28.3),
                # Exp C: NASA POWER supplemental features
                **get_nasa_power_features(dep_time),
                # Target
                TRIP_TARGET: soc_delta,
            }
        )

    db.close()

    df = pl.DataFrame(records)

    if verbose:
        print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"Median passengers (imputed): {median_pob}")
        print(f"SOC delta — mean: {df[TRIP_TARGET].mean():.2f}%, "
              f"std: {df[TRIP_TARGET].std():.2f}%, "
              f"min: {df[TRIP_TARGET].min():.1f}%, "
              f"max: {df[TRIP_TARGET].max():.1f}%")
        dir_groups = df.group_by("direction").agg(
            pl.col(TRIP_TARGET).mean().alias("mean_soc_delta"),
            pl.col(TRIP_TARGET).count().alias("count"),
        )
        print("\nPer-direction breakdown:")
        for row in dir_groups.iter_rows(named=True):
            print(f"  {row['direction']}: mean={row['mean_soc_delta']:.2f}%, n={row['count']}")
        print(f"\nFeature columns: {TRIP_FEATURES}")

    # Save to feature store
    out_path = FEATURE_STORE_DIR / "trip_features.parquet"
    df.write_parquet(out_path)
    if verbose:
        print(f"\nSaved to: {out_path}")

    return df


if __name__ == "__main__":
    extract()
