"""Export train/val/test splits for partner model development (RF, LSTM).

Exports:
  - Model 1A trip features (CSV + Parquet)
  - Model 1B realtime features (CSV + Parquet)
  - Model 1B raw 1Hz trip parquets (for LSTM)
  - Model 2 anomaly features (CSV + Parquet)
  - Feature description CSVs per model

Usage:
    python -m data.export_splits
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANOMALY_FEATURES,
    ANOMALY_TARGETS,
    ARTIFACTS_DIR,
    DUCKDB_PATH,
    FEATURE_STORE_DIR,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_TARGET,
    VAL_END,
)

EXPORT_DIR = ARTIFACTS_DIR / "exported_splits"


# ── Temporal Split ─────────────────────────────────────────────────────

def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split using config dates."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def _split_label(dep_time: datetime) -> str:
    """Return split label for a departure_time."""
    if dep_time <= datetime.fromisoformat(TRAIN_END):
        return "train"
    elif dep_time <= datetime.fromisoformat(VAL_END):
        return "val"
    else:
        return "test"


# ── Feature Descriptions ──────────────────────────────────────────────

TRIP_FEATURE_DESCRIPTIONS = {
    "departure_station_encoded": ("int", "Departure station ordinal (0=Napindan, 8=Escolta)", "encoding", ""),
    "arrival_station_encoded": ("int", "Arrival station ordinal (0=Napindan, 8=Escolta)", "encoding", ""),
    "hop_count": ("int", "Number of station hops between departure and arrival", "derived", ""),
    "direction_encoded": ("int", "Travel direction: 0=downstream (west), 1=upstream (east)", "encoding", ""),
    "route_distance_km": ("float64", "GPS distance of the trip", "telemetry", "km"),
    "start_soc": ("float64", "Battery state of charge at trip start", "telemetry", "%"),
    "start_hv_capacity": ("float64", "High-voltage battery capacity at trip start", "telemetry", "kWh"),
    "target_speed": ("float64", "Average speed over ground during trip", "telemetry", "kn"),
    "target_speed_squared": ("float64", "Average speed squared (drag proxy)", "derived", "kn^2"),
    "passengers_on_board": ("int", "Number of passengers (median-imputed if missing)", "ridership", ""),
    "passenger_load_ratio": ("float64", "Passengers / max capacity (40)", "derived", ""),
    "temperature_c": ("float64", "Ambient temperature at departure hour", "weather", "C"),
    "relative_humidity": ("float64", "Relative humidity at departure hour", "weather", "%"),
    "wind_speed_kn": ("float64", "Wind speed at departure hour", "weather", "kn"),
    "wind_direction_deg": ("float64", "Wind direction (meteorological, FROM)", "weather", "deg"),
    "wind_component_along_route": ("float64", "Wind resolved along route heading (+ve = headwind)", "derived", "kn"),
    "precipitation_mm": ("float64", "Precipitation at departure hour", "weather", "mm"),
    "wave_height_m": ("float64", "Wave height at departure hour", "weather", "m"),
    "hour_of_day": ("int", "Hour of departure (0-23)", "derived", ""),
    "is_peak_hour": ("int", "1 if 7-9am or 4-7pm, else 0", "derived", ""),
    "speed_distance_interaction": ("float64", "target_speed * route_distance_km", "derived", "kn*km"),
    "energy_rate_proxy": ("float64", "target_speed_squared * hop_count (drag * distance proxy)", "derived", "kn^2"),
    "wind_cross_component": ("float64", "Wind perpendicular to route heading", "derived", "kn"),
    "soc_per_km_capacity": ("float64", "start_soc / (route_distance_km + 0.01)", "derived", "%/km"),
    "tide_height_m": ("float64", "Predicted tide height above MLLW at departure", "derived", "m"),
    "hours_since_high_tide": ("float64", "Hours elapsed since most recent high tide", "derived", "h"),
    "tide_phase": ("int", "0=rising (flood), 1=falling (ebb)", "derived", ""),
    "current_component_along_route": ("float64", "Ocean current projected along route (+ve = with travel)", "weather", "m/s"),
    "current_velocity_ms": ("float64", "Raw ocean current speed", "weather", "m/s"),
    "wave_component_along_route": ("float64", "Wave height * cos(wave_dir - route_heading)", "derived", "m"),
    "wave_period_s": ("float64", "Wave period (longer = larger swells)", "weather", "s"),
    "soc_delta": ("float64", "TARGET: SOC consumed during trip (start - end)", "telemetry", "%"),
}

REALTIME_FEATURE_DESCRIPTIONS = {
    "current_soc": ("float64", "Battery SOC at this timestep", "telemetry", "%"),
    "current_speed": ("float64", "Speed over ground at this timestep", "telemetry", "kn"),
    "current_motor_power": ("float64", "Combined motor power at this timestep", "telemetry", "kW"),
    "current_battery_power": ("float64", "Battery power at this timestep", "telemetry", "kW"),
    "current_heading": ("float64", "Vessel heading at this timestep", "telemetry", "deg"),
    "speed_squared": ("float64", "Speed over ground squared", "derived", "kn^2"),
    "distance_remaining_km": ("float64", "Distance remaining to arrival station", "derived", "km"),
    "trip_progress_fraction": ("float64", "Fraction of trip completed (0-1)", "derived", ""),
    "elapsed_time_s": ("float64", "Seconds elapsed since trip start", "derived", "s"),
    "soc_rate_30s": ("float64", "SOC change rate over last 30s", "derived", "%/min"),
    "soc_rate_60s": ("float64", "SOC change rate over last 60s", "derived", "%/min"),
    "soc_rate_120s": ("float64", "SOC change rate over last 120s", "derived", "%/min"),
    "avg_power_30s": ("float64", "Rolling mean motor power (30s window)", "derived", "kW"),
    "avg_power_60s": ("float64", "Rolling mean motor power (60s window)", "derived", "kW"),
    "avg_speed_30s": ("float64", "Rolling mean speed (30s window)", "derived", "kn"),
    "avg_speed_60s": ("float64", "Rolling mean speed (60s window)", "derived", "kn"),
    "departure_station_encoded": ("int", "Departure station ordinal (0=Napindan, 8=Escolta)", "encoding", ""),
    "arrival_station_encoded": ("int", "Arrival station ordinal (0=Napindan, 8=Escolta)", "encoding", ""),
    "direction_encoded": ("int", "Travel direction: 0=downstream, 1=upstream", "encoding", ""),
    "hop_count": ("int", "Number of station hops between departure and arrival", "derived", ""),
    "passengers_on_board": ("int", "Number of passengers (median-imputed if missing)", "ridership", ""),
    "temperature_c": ("float64", "Ambient temperature at departure hour", "weather", "C"),
    "wind_speed_kn": ("float64", "Wind speed at departure hour", "weather", "kn"),
    "wind_component_along_route": ("float64", "Wind resolved along route heading", "derived", "kn"),
    "wave_height_m": ("float64", "Wave height at departure hour", "weather", "m"),
    "power_speed_ratio": ("float64", "motor_power / (speed + eps)", "derived", "kW/kn"),
    "soc_rate_std_60s": ("float64", "SOC rate standard deviation over 60s window", "derived", "%/min"),
    "power_std_60s": ("float64", "Motor power standard deviation over 60s", "derived", "kW"),
    "speed_acceleration_30s": ("float64", "Speed change over last 30s", "derived", "kn"),
    "rpm_per_speed": ("float64", "avg(port_rpm, stbd_rpm) / (speed + eps) — propulsion efficiency proxy", "derived", "rpm/kn"),
    "tide_height_m": ("float64", "Predicted tide height above MLLW", "derived", "m"),
    "hours_since_high_tide": ("float64", "Hours elapsed since most recent high tide", "derived", "h"),
    "tide_phase": ("int", "0=rising (flood), 1=falling (ebb)", "derived", ""),
    "soc_remaining_delta": ("float64", "TARGET: current_soc - soc_at_trip_end", "telemetry", "%"),
}

ANOMALY_FEATURE_DESCRIPTIONS = {
    "port_motor_power": ("float64", "Port motor power output", "telemetry", "kW"),
    "stbd_motor_power": ("float64", "Starboard motor power output", "telemetry", "kW"),
    "motor_power_combined": ("float64", "Total motor power (port + starboard)", "telemetry", "kW"),
    "port_rpm": ("float64", "Port motor RPM", "telemetry", "rpm"),
    "stbd_rpm": ("float64", "Starboard motor RPM", "telemetry", "rpm"),
    "speed_over_ground": ("float64", "Speed over ground", "telemetry", "kn"),
    "battery_power": ("float64", "Battery power output", "telemetry", "kW"),
    "port_stbd_power_ratio": ("float64", "Port power / (port + starboard + eps)", "derived", ""),
    "port_stbd_power_diff": ("float64", "abs(port_power - starboard_power)", "derived", "kW"),
    "port_stbd_rpm_ratio": ("float64", "Port RPM / (port + starboard + eps)", "derived", ""),
    "port_stbd_rpm_diff": ("float64", "abs(port_rpm - starboard_rpm)", "derived", "rpm"),
    "port_power_per_rpm": ("float64", "Port motor power / (port RPM + eps)", "derived", "kW/rpm"),
    "stbd_power_per_rpm": ("float64", "Starboard motor power / (starboard RPM + eps)", "derived", "kW/rpm"),
    "combined_power_per_speed": ("float64", "Combined power / (speed + eps)", "derived", "kW/kn"),
    "power_speed_squared_ratio": ("float64", "Combined power / (speed^2 + eps)", "derived", "kW/kn^2"),
}

RAW_1HZ_FEATURE_DESCRIPTIONS = {
    "timestamp": ("datetime", "UTC timestamp at 1Hz", "telemetry", ""),
    "soc_percent": ("float64", "Battery state of charge", "telemetry", "%"),
    "speed_over_ground": ("float64", "GPS speed over ground", "telemetry", "kn"),
    "heading": ("float64", "Vessel heading", "telemetry", "deg"),
    "motor_power_combined": ("float64", "Total motor power (port + starboard)", "telemetry", "kW"),
    "battery_power": ("float64", "Battery power output", "telemetry", "kW"),
    "port_motor_power": ("float64", "Port motor power", "telemetry", "kW"),
    "stbd_motor_power": ("float64", "Starboard motor power", "telemetry", "kW"),
    "port_rpm": ("float64", "Port motor RPM", "telemetry", "rpm"),
    "stbd_rpm": ("float64", "Starboard motor RPM", "telemetry", "rpm"),
    "trip_nm": ("float64", "Cumulative trip distance (nautical miles odometer)", "telemetry", "nm"),
    "latitude": ("float64", "GPS latitude", "telemetry", "deg"),
    "longitude": ("float64", "GPS longitude", "telemetry", "deg"),
}


def _write_feature_descriptions(
    out_dir: Path,
    descriptions: dict[str, tuple[str, str, str, str]],
    feature_list: list[str] | None = None,
) -> None:
    """Write feature_descriptions.csv for a model directory."""
    rows = []
    keys = feature_list if feature_list else list(descriptions.keys())
    for name in keys:
        if name in descriptions:
            dtype, desc, source, unit = descriptions[name]
        else:
            dtype, desc, source, unit = "float64", "", "unknown", ""
        rows.append({
            "feature_name": name,
            "dtype": dtype,
            "description": desc,
            "source": source,
            "unit": unit,
        })

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "feature_descriptions.csv")


# ── Export Functions ───────────────────────────────────────────────────

def _save_split(df: pl.DataFrame, out_dir: Path, split_name: str) -> None:
    """Save a split as both CSV and Parquet."""
    df.write_csv(out_dir / f"{split_name}.csv")
    df.write_parquet(out_dir / f"{split_name}.parquet")


def _print_split_summary(
    name: str,
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
    target_col: str | None,
) -> None:
    """Print summary stats for a model's splits."""
    print(f"\n  {name}:")
    print(f"    Train: {len(train):,} rows")
    print(f"    Val:   {len(val):,} rows")
    print(f"    Test:  {len(test):,} rows")
    if target_col and target_col in train.columns:
        for label, split in [("Train", train), ("Val", val), ("Test", test)]:
            t = split[target_col]
            print(f"    {label} target ({target_col}): "
                  f"mean={t.mean():.3f}, std={t.std():.3f}, "
                  f"min={t.min():.2f}, max={t.max():.2f}")


def export_trip(verbose: bool = True) -> None:
    """Export Model 1A trip-level splits."""
    out_dir = EXPORT_DIR / "model_1a_trip"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train, val, test = temporal_split(df)

    # Keep features + target + useful metadata
    keep_cols = TRIP_FEATURES + [TRIP_TARGET, "segment_id", "departure_time",
                                  "departure_station", "arrival_station", "direction"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        _save_split(split_df.select(keep_cols), out_dir, split_name)

    _write_feature_descriptions(out_dir, TRIP_FEATURE_DESCRIPTIONS, TRIP_FEATURES + [TRIP_TARGET])

    if verbose:
        n_feat = len(TRIP_FEATURES)
        print(f"    Features: {n_feat} + target")
        _print_split_summary("Model 1A (trip)", train, val, test, TRIP_TARGET)


def export_realtime(verbose: bool = True) -> None:
    """Export Model 1B realtime splits."""
    out_dir = EXPORT_DIR / "model_1b_realtime"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train, val, test = temporal_split(df)

    keep_cols = REALTIME_FEATURES + [REALTIME_TARGET, "segment_id", "departure_time"]
    # trip_progress_fraction is already in REALTIME_FEATURES
    keep_cols = list(dict.fromkeys(keep_cols))
    keep_cols = [c for c in keep_cols if c in df.columns]

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        _save_split(split_df.select(keep_cols), out_dir, split_name)

    _write_feature_descriptions(out_dir, REALTIME_FEATURE_DESCRIPTIONS,
                                REALTIME_FEATURES + [REALTIME_TARGET])

    if verbose:
        n_feat = len(REALTIME_FEATURES)
        print(f"    Features: {n_feat} + target")
        _print_split_summary("Model 1B (realtime)", train, val, test, REALTIME_TARGET)


def export_raw_1hz(verbose: bool = True) -> None:
    """Export raw 1Hz trip parquets into train/val/test folders for LSTM."""
    out_dir = EXPORT_DIR / "model_1b_raw_1hz"
    for split in ("train", "val", "test"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    # Query segment metadata from DuckDB
    db = duckdb.connect(DUCKDB_PATH, read_only=True)
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
            st.last_soc
        FROM segments s
        JOIN segment_statistics st ON s.segment_id = st.segment_id
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
    db.close()

    metadata_rows = []
    counts = {"train": 0, "val": 0, "test": 0}

    for row in rows:
        seg = dict(zip(cols, row))
        src_path = Path(seg["parquet_path"])
        if not src_path.exists():
            continue

        split = _split_label(seg["departure_time"])
        dest_name = f"{seg['segment_id']}.parquet"
        dest_path = out_dir / split / dest_name

        shutil.copy2(str(src_path), str(dest_path))
        counts[split] += 1

        soc_delta = seg["first_soc"] - seg["last_soc"]
        metadata_rows.append({
            "segment_id": seg["segment_id"],
            "departure_station": seg["departure_station"],
            "arrival_station": seg["arrival_station"],
            "direction": seg["direction"],
            "departure_time": seg["departure_time"],
            "split": split,
            "parquet_path": f"{split}/{dest_name}",
            "distance_km": seg["distance_km"],
            "first_soc": seg["first_soc"],
            "last_soc": seg["last_soc"],
            "soc_delta": soc_delta,
        })

    meta_df = pl.DataFrame(metadata_rows)
    meta_df.write_csv(out_dir / "segment_metadata.csv")

    _write_feature_descriptions(out_dir, RAW_1HZ_FEATURE_DESCRIPTIONS)

    if verbose:
        print(f"\n  Model 1B Raw 1Hz (LSTM):")
        for split, count in counts.items():
            print(f"    {split}: {count} trip segments")
        print(f"    Metadata: {len(metadata_rows)} rows")


def export_anomaly(verbose: bool = True) -> None:
    """Export Model 2 anomaly detection splits."""
    out_dir = EXPORT_DIR / "model_2_anomaly"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(FEATURE_STORE_DIR / "anomaly_features.parquet")
    train, val, test = temporal_split(df)

    keep_cols = ANOMALY_FEATURES + ["segment_id", "departure_time"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        _save_split(split_df.select(keep_cols), out_dir, split_name)

    # For anomaly, include reconstruction targets in description
    all_features_desc = dict(ANOMALY_FEATURE_DESCRIPTIONS)
    _write_feature_descriptions(out_dir, all_features_desc, ANOMALY_FEATURES)

    if verbose:
        n_feat = len(ANOMALY_FEATURES)
        print(f"    Features: {n_feat}")
        print(f"    Reconstruction targets: {ANOMALY_TARGETS}")
        _print_split_summary("Model 2 (anomaly)", train, val, test, target_col=None)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Exporting Train/Val/Test Splits")
    print(f"  Split dates: train <= {TRAIN_END} | val <= {VAL_END} | test after")
    print(f"  Output: {EXPORT_DIR}")
    print("=" * 60)

    export_trip()
    export_realtime()
    export_raw_1hz()
    export_anomaly()

    print("\n" + "=" * 60)
    print("  Export complete!")
    print(f"  Output directory: {EXPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
