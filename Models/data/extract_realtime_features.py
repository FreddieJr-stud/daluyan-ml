"""Extract 1Hz real-time features from trip parquets for Model 1B.

Reads each trip segment parquet, computes rolling window features,
trip progress metrics, and subsamples to every 10 seconds.

Usage:
    python -m data.extract_realtime_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DUCKDB_PATH,
    FEATURE_STORE_DIR,
    MAX_PASSENGERS,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
)
from features.station_graph import (
    compute_hop_count,
    compute_route_heading,
    encode_direction,
    encode_station,
    resolve_wind_component,
)
from features.nasa_power import get_nasa_power_features
from features.tide import compute_tide_features
from features.water_physics import (
    compute_water_physics_features,
    friction_coefficient as ittc_friction_coefficient,
    kinematic_viscosity as ittc_kinematic_viscosity,
    seawater_density as ittc_seawater_density,
)

SUBSAMPLE_INTERVAL_S = 10
EPS = 0.01  # avoid div-by-zero
def _query_segment_context(db: duckdb.DuckDBPyConnection) -> dict[str, dict]:
    """Load segment metadata, stats, ridership, weather for all segments."""
    rows = db.execute("""
        SELECT
            s.segment_id,
            s.departure_station,
            s.arrival_station,
            s.direction,
            s.departure_time,
            s.parquet_path,
            st.distance_km,
            st.first_soc,
            st.last_soc,
            sr.passengers_on_board
        FROM segments s
        JOIN segment_statistics st ON s.segment_id = st.segment_id
        LEFT JOIN segment_ridership sr ON s.segment_id = sr.segment_id
        WHERE s.direction != 'unknown'
          AND s.departure_station != 'unknown'
          AND s.arrival_station != 'unknown'
          AND st.distance_km >= 0.1
          AND st.duration_seconds >= 30
          AND (st.first_soc - st.last_soc) >= 0
          AND (st.first_soc - st.last_soc) <= 50
        ORDER BY s.departure_time
    """).fetchall()

    cols = [d[0] for d in db.description]
    segments = {}
    for row in rows:
        d = dict(zip(cols, row))
        segments[d["segment_id"]] = d
    return segments


def _get_weather(db: duckdb.DuckDBPyConnection, date_str: str, hour: int) -> dict:
    """Fetch weather nearest to date + hour."""
    rows = db.execute(
        "SELECT * FROM weather_hourly WHERE date = ? ORDER BY time",
        [date_str],
    ).fetchall()
    if not rows:
        return {}
    cols = [d[0] for d in db.description]
    best, best_diff = None, 999
    for row in rows:
        rd = dict(zip(cols, row))
        try:
            rh = int(rd["time"].split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        diff = abs(rh - hour)
        if diff < best_diff:
            best_diff = diff
            best = rd
    return best or {}


def _process_segment(
    seg_info: dict,
    weather: dict,
    median_pob: int,
    route_priors: dict | None = None,
) -> pl.DataFrame | None:
    """Process one segment parquet into real-time feature rows."""
    parquet_path = seg_info["parquet_path"]
    if not Path(parquet_path).exists():
        return None

    df = pl.read_parquet(parquet_path)
    if len(df) < 30:  # need at least 30 rows for rolling features
        return None

    # Sort by timestamp
    df = df.sort("timestamp")

    last_soc = seg_info["last_soc"]
    total_distance_km = seg_info["distance_km"]
    dep_time = df["timestamp"][0]

    # Trip progress: compute cumulative distance from trip_nm
    trip_nm = df["trip_nm"].to_numpy()
    trip_km = (trip_nm - trip_nm[0]) * 1.852
    total_trip_km = max(trip_km[-1], total_distance_km, EPS)

    # Time elapsed
    timestamps = df["timestamp"].to_list()
    elapsed_s = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])

    # Rolling features using Polars
    soc = df["soc_percent"]
    power = df["motor_power_combined"]
    speed = df["speed_over_ground"]

    features_df = df.with_columns([
        # Current state
        pl.col("soc_percent").alias("current_soc"),
        pl.col("speed_over_ground").alias("current_speed"),
        pl.col("motor_power_combined").alias("current_motor_power"),
        pl.col("battery_power").alias("current_battery_power"),
        pl.col("heading").alias("current_heading"),
        (pl.col("speed_over_ground") ** 2).alias("speed_squared"),

        # Trip progress
        pl.Series("distance_remaining_km", np.clip(total_trip_km - trip_km, 0, None)),
        pl.Series("trip_progress_fraction", np.clip(trip_km / total_trip_km, 0, 1)),
        pl.Series("elapsed_time_s", elapsed_s),

        # Rolling windows — SOC rate (%/min)
        (soc.diff(n=30) / (30 / 60.0)).alias("soc_rate_30s"),
        (soc.diff(n=60) / (60 / 60.0)).alias("soc_rate_60s"),
        (soc.diff(n=120) / (120 / 60.0)).alias("soc_rate_120s"),

        # Rolling averages
        power.rolling_mean(window_size=30).alias("avg_power_30s"),
        power.rolling_mean(window_size=60).alias("avg_power_60s"),
        speed.rolling_mean(window_size=30).alias("avg_speed_30s"),
        speed.rolling_mean(window_size=60).alias("avg_speed_60s"),

        # Physics
        (pl.col("motor_power_combined") / (pl.col("speed_over_ground") + EPS)).alias("power_speed_ratio"),

        # v2 stability features
        soc.diff(n=1).rolling_std(window_size=60).alias("soc_rate_std_60s"),
        power.rolling_std(window_size=60).alias("power_std_60s"),
        speed.diff(n=30).alias("speed_acceleration_30s"),

        # v6 trip-so-far consumption features (Exp 8)
        pl.Series("soc_consumed_so_far", seg_info["first_soc"] - soc.to_numpy()),
        pl.Series("empirical_soc_per_km", (seg_info["first_soc"] - soc.to_numpy()) / (trip_km + EPS)),
        pl.Series("empirical_power_per_km", power.to_numpy().cumsum() / (trip_km + EPS)),

        # v3 current proxy feature (telemetry-derived)
        # Note: power_speed_ratio (v1) already captures power/speed;
        # rpm_per_speed adds independent propulsion efficiency information
        (((pl.col("port_rpm") + pl.col("stbd_rpm")) / 2.0)
         / (pl.col("speed_over_ground") + EPS)).alias("rpm_per_speed"),

        # Target
        pl.lit(last_soc).alias("_last_soc"),
    ])

    # Route context (constant for the trip)
    dep_stn = seg_info["departure_station"]
    arr_stn = seg_info["arrival_station"]
    direction = seg_info["direction"]
    route_heading = compute_route_heading(dep_stn, arr_stn)

    pob = seg_info["passengers_on_board"]
    if pob is None:
        pob = median_pob

    wind_speed = weather.get("wind_speed_kn") or 0.0
    wind_dir = weather.get("wind_direction_deg") or 0.0
    wind_comp = resolve_wind_component(wind_speed, wind_dir, route_heading)
    sst = weather.get("sea_surface_temperature_c") or 28.3

    # v3: Tide features at departure time (tide changes slowly vs trip duration)
    tide = compute_tide_features(dep_time)

    features_df = features_df.with_columns([
        pl.lit(encode_station(dep_stn)).alias("departure_station_encoded"),
        pl.lit(encode_station(arr_stn)).alias("arrival_station_encoded"),
        pl.lit(encode_direction(direction)).alias("direction_encoded"),
        pl.lit(compute_hop_count(dep_stn, arr_stn)).alias("hop_count"),
        pl.lit(float(pob)).alias("passengers_on_board"),
        pl.lit(weather.get("temperature_c") or 0.0).alias("temperature_c"),
        pl.lit(wind_speed).alias("wind_speed_kn"),
        pl.lit(wind_comp).alias("wind_component_along_route"),
        pl.lit(weather.get("wave_height_m") or 0.0).alias("wave_height_m"),
        # v3 tide features
        pl.lit(tide["tide_height_m"]).alias("tide_height_m"),
        pl.lit(tide["hours_since_high_tide"]).alias("hours_since_high_tide"),
        pl.lit(float(tide["tide_phase"])).alias("tide_phase"),
        # v5 pressure, wind gusts, day-of-week
        pl.lit(weather.get("pressure_msl_hpa") or 1013.25).alias("pressure_msl_hpa"),
        pl.lit(weather.get("wind_gusts_kn") or 0.0).alias("wind_gusts_kn"),
        pl.lit(float(int(dep_time.weekday() >= 5))).alias("is_weekend"),
        # v10 ITTC water physics (SST-derived constants, per-trip)
        pl.lit(ittc_seawater_density(sst)).alias("water_density"),
        pl.lit(ittc_kinematic_viscosity(sst) * 1e6).alias("kinematic_viscosity"),
        # Exp C: NASA POWER supplemental features (constant per-trip)
        pl.lit(get_nasa_power_features(dep_time)["solar_radiation_wm2"]).alias("solar_radiation_wm2"),
        pl.lit(get_nasa_power_features(dep_time)["specific_humidity_gkg"]).alias("specific_humidity_gkg"),
        # v6 route prior (Exp 9): average SOC consumed on this route from training data
        pl.lit(
            route_priors.get(
                f"{encode_station(dep_stn)}_{encode_station(arr_stn)}", 5.0
            ) if route_priors else 5.0
        ).alias("route_avg_soc_delta"),
    ])

    # v10 ITTC: per-row friction coefficient from avg_speed_60s + SST
    # CF = 0.075 / (log10(Re) - 2)^2 where Re = V*L/nu
    _nu = ittc_kinematic_viscosity(sst)
    _L = 20.0  # vessel waterline length estimate (m)
    features_df = features_df.with_columns(
        (pl.col("avg_speed_60s") * 0.514444 * _L / _nu)
        .map_batches(
            lambda s: s.to_numpy().clip(min=1000),
            return_dtype=pl.Float64,
        )
        .alias("_re")
    )
    import math
    features_df = features_df.with_columns(
        pl.when(pl.col("_re") > 1000)
        .then(0.075 / (pl.col("_re").log(base=10) - 2.0).pow(2) * 1e3)
        .otherwise(0.0)
        .alias("friction_coefficient")
    ).drop("_re")

    # Compute target: soc_remaining_delta = current_soc - soc_at_trip_end
    features_df = features_df.with_columns(
        (pl.col("current_soc") - pl.col("_last_soc")).alias(REALTIME_TARGET)
    )

    # Add metadata for splitting
    features_df = features_df.with_columns([
        pl.lit(seg_info["segment_id"]).alias("segment_id"),
        pl.col("timestamp").alias("departure_time"),
    ])

    # Drop warmup rows (first 120s have incomplete rolling features)
    features_df = features_df.slice(120)

    if len(features_df) == 0:
        return None

    # Subsample every SUBSAMPLE_INTERVAL_S seconds
    features_df = features_df.gather_every(SUBSAMPLE_INTERVAL_S)

    # Select only the columns we need (+ experimental features for Exp 8-9)
    experimental_features = [
        "soc_consumed_so_far",
        "empirical_soc_per_km",
        "empirical_power_per_km",
        "route_avg_soc_delta",
        "passengers_on_board",
    ]
    keep_cols = REALTIME_FEATURES + experimental_features + [
        REALTIME_TARGET,
        "segment_id",
        "departure_time",
        "trip_progress_fraction",
    ]
    # Remove duplicates (trip_progress_fraction is already in features)
    keep_cols = list(dict.fromkeys(keep_cols))

    return features_df.select(keep_cols)


def _compute_route_priors(segments: dict[str, dict]) -> dict[str, float]:
    """Compute average SOC delta per route from training-set segments only.

    Returns dict mapping "{dep_encoded}_{arr_encoded}" → mean soc_delta.
    """
    from datetime import datetime as _dt
    train_cutoff = _dt.fromisoformat(TRAIN_END)

    from collections import defaultdict
    route_deltas: dict[str, list[float]] = defaultdict(list)
    for seg in segments.values():
        if seg["departure_time"] > train_cutoff:
            continue
        dep_enc = encode_station(seg["departure_station"])
        arr_enc = encode_station(seg["arrival_station"])
        soc_delta = seg["first_soc"] - seg["last_soc"]
        route_deltas[f"{dep_enc}_{arr_enc}"].append(soc_delta)

    return {k: float(np.mean(v)) for k, v in route_deltas.items()}


def extract(verbose: bool = True) -> pl.DataFrame:
    """Build the real-time feature DataFrame from all trip parquets."""
    db = duckdb.connect(DUCKDB_PATH, read_only=True)
    segments = _query_segment_context(db)

    if verbose:
        print(f"Valid segments: {len(segments)}")

    # Compute median passengers for imputation
    pob_vals = [s["passengers_on_board"] for s in segments.values() if s["passengers_on_board"] is not None]
    median_pob = int(np.median(pob_vals)) if pob_vals else 15

    # Compute route priors from training data (Exp 9)
    route_priors = _compute_route_priors(segments)
    if verbose:
        print(f"Route priors: {len(route_priors)} routes from training data")

    all_dfs: list[pl.DataFrame] = []
    skipped = 0

    for i, (seg_id, seg_info) in enumerate(segments.items()):
        dep_time = seg_info["departure_time"]
        date_str = dep_time.strftime("%Y-%m-%d")
        hour = dep_time.hour
        weather = _get_weather(db, date_str, hour)

        result = _process_segment(seg_info, weather, median_pob, route_priors)
        if result is not None and len(result) > 0:
            all_dfs.append(result)
        else:
            skipped += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(segments)} segments...")

    db.close()

    if not all_dfs:
        print("ERROR: No data extracted!")
        sys.exit(1)

    df = pl.concat(all_dfs, how="diagonal_relaxed")

    # Fill any remaining nulls from rolling features with 0
    df = df.fill_null(0.0)
    df = df.fill_nan(0.0)

    if verbose:
        print(f"\nFinal dataset: {len(df)} rows from {len(all_dfs)} segments (skipped {skipped})")
        print(f"Target (soc_remaining_delta) stats:")
        target = df[REALTIME_TARGET]
        print(f"  mean={target.mean():.2f}%, std={target.std():.2f}%, "
              f"min={target.min():.1f}%, max={target.max():.1f}%")

    # Save
    out_path = FEATURE_STORE_DIR / "realtime_features.parquet"
    df.write_parquet(out_path)
    if verbose:
        print(f"Saved to: {out_path}")

    return df


if __name__ == "__main__":
    extract()
