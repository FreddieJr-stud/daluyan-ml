"""Extract BMS cell-level features for battery anomaly detection (Model 3).

Reads ALL raw CSVs (both _off and operational), extracts BMS data from
OneAries devices (port IP_3, stbd IP_4), computes cell/pack features,
and saves to feature store parquet files.

Outputs:
    artifacts/feature_store/bms_charging_features.parquet
    artifacts/feature_store/bms_operational_features.parquet

Usage:
    python -m data.extract_bms_features
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BATTERY_FEATURES,
    BATTERY_RAW_FEATURES,
    DEVICE_PORT_BMS,
    DEVICE_STBD_BMS,
    DUCKDB_PATH,
    FEATURE_STORE_DIR,
    RAW_CSV_DIR,
)

SUBSAMPLE_INTERVAL_S = 5
EPS = 0.01

# BMS JSON keys → feature names (common 34-field format)
_BMS_FIELD_MAP = {
    "gCellBalance": "cell_v_spread",
    "gMaxCellTemperature": "cell_t_max",
    "gMinCellTemperature": "cell_t_min",
    "gPackVoltage": "pack_voltage",
    "gCurrent": "pack_current",
    "gStateOfCharge": "soc",
    "gStateOfHealth": "soh",
    "gPower": "power",
    "gAverageTemperature": "avg_temperature",
    "gEnergyRemaining": "energy_remaining",
}

# Optional 115-field keys — extract when available, prefer over 34-field
_BMS_FIELD_115_OVERRIDES = {
    "vCellMax": "_vcell_max",
    "vCellMin": "_vcell_min",
    "tCellMax": "cell_t_max",      # overrides gMaxCellTemperature
    "tCellMin": "cell_t_min",      # overrides gMinCellTemperature
    "iPack": "pack_current",       # overrides gCurrent
    "socUser": "soc",              # overrides gStateOfCharge
    "sohUser": "soh",              # overrides gStateOfHealth
    "rIsolation": "isolation_r",
    "tCoolantIn": "coolant_in",
    "tCoolantOut": "coolant_out",
    "tAmbient": "ambient_t",
    "tShunt": "shunt_t",
    "vStack": "pack_voltage",      # overrides gPackVoltage
}


def _parse_json(raw: str) -> dict | None:
    """Parse a double-escaped JSON string from response_raw."""
    try:
        if not raw:
            return None
        s = raw.strip()
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1].replace('""', '"')
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_bms_fields(data: dict, prefix: str) -> dict:
    """Extract BMS fields from parsed JSON, with prefix (port/stbd).

    Uses 34-field common keys, then overrides with 115-field keys when available.
    """
    result = {}

    # Base extraction from 34-field format
    for json_key, feat_name in _BMS_FIELD_MAP.items():
        val = data.get(json_key)
        if val is not None:
            result[f"{prefix}_{feat_name}"] = float(val)

    # Override with 115-field precision when available
    for json_key, feat_name in _BMS_FIELD_115_OVERRIDES.items():
        val = data.get(json_key)
        if val is not None:
            result[f"{prefix}_{feat_name}"] = float(val)

    # If 115-field has vCellMax/vCellMin, compute cell_v_spread from those
    vcell_max = data.get("vCellMax")
    vcell_min = data.get("vCellMin")
    if vcell_max is not None and vcell_min is not None:
        result[f"{prefix}_cell_v_spread"] = float(vcell_max) - float(vcell_min)

    return result


def _read_bms_from_csv(filepath: Path) -> list[dict]:
    """Read BMS rows from a raw CSV file.

    Returns list of dicts with timestamp + port/stbd fields,
    grouped by second (pairing port/stbd readings at same timestamp).
    """
    # Read line-by-line for efficiency (files can be >1M rows, ~4% are BMS)
    rows_by_second: dict[str, dict] = {}

    # Match both 34-field and 115-field (_DEVICE suffix) BMS devices.
    # The 115-field device has vCellMax/vCellMin which override the stale
    # gCellBalance field (reports 0.000V since ~Feb 3, 2026).
    port_devs = {DEVICE_PORT_BMS, DEVICE_PORT_BMS + "_DEVICE"}
    stbd_devs = {DEVICE_STBD_BMS, DEVICE_STBD_BMS + "_DEVICE"}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        f.readline()  # skip header
        for line in f:
            if not line[:5].startswith("20"):
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                continue
            ts, dev, raw = parts[0], parts[1], parts[2].rstrip("\n\r")

            if dev in port_devs:
                prefix = "port"
            elif dev in stbd_devs:
                prefix = "stbd"
            else:
                continue

            data = _parse_json(raw)
            if data is None:
                continue

            # Truncate to second for port/stbd pairing
            ts_second = ts[:19]  # "2025/10/28 11:12:18"

            if ts_second not in rows_by_second:
                rows_by_second[ts_second] = {"timestamp_str": ts_second}

            rows_by_second[ts_second].update(_extract_bms_fields(data, prefix))

    return list(rows_by_second.values())


def _compute_derived_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived battery features."""
    return df.with_columns([
        # Cell temperature spread per side
        (pl.col("port_cell_t_max") - pl.col("port_cell_t_min"))
            .alias("port_cell_t_spread"),
        (pl.col("stbd_cell_t_max") - pl.col("stbd_cell_t_min"))
            .alias("stbd_cell_t_spread"),
        # Combined spreads (max of both sides)
        pl.max_horizontal("port_cell_v_spread", "stbd_cell_v_spread")
            .alias("cell_v_spread_combined"),
        # Pack imbalance
        (pl.col("port_soc") - pl.col("stbd_soc")).abs()
            .alias("port_stbd_soc_diff"),
        (pl.col("port_pack_voltage") - pl.col("stbd_pack_voltage")).abs()
            .alias("port_stbd_v_diff"),
        (pl.col("port_power") - pl.col("stbd_power")).abs()
            .alias("port_stbd_power_diff"),
    ]).with_columns([
        # Combined temp spread (needs port/stbd_cell_t_spread computed first)
        pl.max_horizontal("port_cell_t_spread", "stbd_cell_t_spread")
            .alias("cell_t_spread_combined"),
    ])


def _get_segment_windows(db: duckdb.DuckDBPyConnection) -> list[dict]:
    """Get trip segment time windows for operational data filtering."""
    rows = db.execute("""
        SELECT
            s.segment_id,
            s.departure_time,
            s.arrival_time,
            s.direction
        FROM segments s
        JOIN segment_statistics st ON s.segment_id = st.segment_id
        WHERE s.direction != 'unknown'
          AND st.distance_km >= 0.1
          AND st.duration_seconds >= 30
        ORDER BY s.departure_time
    """).fetchall()
    return [
        {
            "segment_id": r[0],
            "departure_time": r[1],
            "arrival_time": r[2],
            "direction": r[3],
        }
        for r in rows
    ]


def extract(verbose: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract BMS features from all raw CSVs.

    Returns:
        (charging_df, operational_df) — both with BATTERY_FEATURES columns
    """
    start = time.time()

    # Discover CSV files
    all_csvs = sorted(RAW_CSV_DIR.glob("*.csv"))
    off_csvs = [f for f in all_csvs if "_off" in f.name]
    op_csvs = [f for f in all_csvs if "_off" not in f.name]

    if verbose:
        print(f"Found {len(all_csvs)} raw CSVs ({len(off_csvs)} off, {len(op_csvs)} operational)")

    # Get segment windows for operational data filtering
    db = duckdb.connect(DUCKDB_PATH, read_only=True)
    segments = _get_segment_windows(db)
    db.close()
    if verbose:
        print(f"Loaded {len(segments)} segment windows for operational filtering")

    # ── Extract charging data (from _off files) ──
    if verbose:
        print("\n--- Charging BMS Data ---")
    charging_rows: list[dict] = []
    for i, csv_path in enumerate(off_csvs):
        rows = _read_bms_from_csv(csv_path)
        # Add date for splitting
        date_str = csv_path.stem[:8]  # "20250930"
        for r in rows:
            r["date"] = date_str
        charging_rows.extend(rows)
        if verbose and (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(off_csvs)} off files ({len(charging_rows)} rows)...")

    if verbose:
        print(f"  Total charging rows (pre-filter): {len(charging_rows)}")

    # ── Extract operational data (from non-off files) ──
    if verbose:
        print("\n--- Operational BMS Data ---")

    # Pre-index segments by date for fast lookup (avoids O(rows * segments) search)
    segments_by_date: dict[str, list[dict]] = {}
    for seg in segments:
        dep = seg["departure_time"]
        if isinstance(dep, str):
            dep = datetime.fromisoformat(dep)
            seg["departure_time"] = dep
        arr = seg["arrival_time"]
        if isinstance(arr, str):
            arr = datetime.fromisoformat(arr)
            seg["arrival_time"] = arr
        date_key = dep.strftime("%Y%m%d")
        segments_by_date.setdefault(date_key, []).append(seg)

    operational_rows: list[dict] = []
    for i, csv_path in enumerate(op_csvs):
        rows = _read_bms_from_csv(csv_path)
        if not rows:
            continue

        date_str = csv_path.stem[:8]
        # Only check segments from this date (typically 4-10 per day)
        day_segments = segments_by_date.get(date_str, [])
        if not day_segments:
            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(op_csvs)} op files "
                      f"({len(operational_rows)} matched rows)...")
            continue

        for r in rows:
            r["date"] = date_str
            try:
                ts = datetime.strptime(r["timestamp_str"], "%Y/%m/%d %H:%M:%S")
            except ValueError:
                continue

            for seg in day_segments:
                if seg["departure_time"] <= ts <= seg["arrival_time"]:
                    r["segment_id"] = seg["segment_id"]
                    r["departure_time"] = seg["departure_time"].isoformat()
                    operational_rows.append(r)
                    break

        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(op_csvs)} op files "
                  f"({len(operational_rows)} matched rows)...")

    if verbose:
        print(f"  Total operational rows (matched to segments): {len(operational_rows)}")

    # ── Build DataFrames ──
    charging_df = _build_dataframe(charging_rows, mode="charging", verbose=verbose)
    operational_df = _build_dataframe(operational_rows, mode="operational", verbose=verbose)

    elapsed = time.time() - start
    if verbose:
        print(f"\nTotal extraction time: {elapsed/60:.1f} min")

    return charging_df, operational_df


def _build_dataframe(rows: list[dict], mode: str, verbose: bool = True) -> pl.DataFrame:
    """Build a polars DataFrame from extracted rows, compute derived features, subsample."""
    if not rows:
        print(f"  WARNING: No {mode} BMS data found!")
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # Parse timestamp
    if "timestamp_str" in df.columns:
        df = df.with_columns(
            pl.col("timestamp_str")
            .str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S")
            .alias("timestamp")
        )

    # Ensure all raw feature columns exist (fill missing with null)
    for feat in BATTERY_RAW_FEATURES:
        if feat not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(feat))

    # Drop rows where essential BMS fields are all null (incomplete port/stbd pairing)
    essential = ["port_pack_voltage", "stbd_pack_voltage", "port_soc", "stbd_soc"]
    existing_essential = [c for c in essential if c in df.columns]
    if existing_essential:
        df = df.filter(
            pl.any_horizontal([pl.col(c).is_not_null() for c in existing_essential])
        )

    # Fill null/nan
    for feat in BATTERY_RAW_FEATURES:
        if feat in df.columns:
            df = df.with_columns(pl.col(feat).fill_null(0.0).fill_nan(0.0))

    if len(df) < 10:
        print(f"  WARNING: Only {len(df)} {mode} rows after filtering!")
        return pl.DataFrame()

    # Compute derived features
    df = _compute_derived_features(df)

    # Subsample every SUBSAMPLE_INTERVAL_S seconds
    n_before = len(df)
    df = df.gather_every(SUBSAMPLE_INTERVAL_S)

    # Add departure_time for temporal splitting
    if mode == "charging":
        # For charging, use date to create a pseudo departure_time
        if "date" in df.columns:
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Date, "%Y%m%d")
                .cast(pl.Datetime)
                .alias("departure_time")
            )
    elif mode == "operational":
        # departure_time already added during segment matching (may include microseconds)
        if "departure_time" in df.columns:
            df = df.with_columns(
                pl.col("departure_time")
                .cast(pl.Utf8)
                .str.to_datetime(strict=False)
                .alias("departure_time")
            )

    # Select final columns
    keep_cols = [c for c in BATTERY_FEATURES if c in df.columns]
    meta_cols = ["timestamp", "departure_time"]
    if mode == "operational" and "segment_id" in df.columns:
        meta_cols.append("segment_id")
    if mode == "charging" and "date" in df.columns:
        meta_cols.append("date")

    available_cols = [c for c in keep_cols + meta_cols if c in df.columns]
    df = df.select(available_cols)

    if verbose:
        print(f"\n  {mode.title()} dataset: {len(df)} rows (subsampled from {n_before})")
        for feat in BATTERY_RAW_FEATURES[:6]:
            if feat in df.columns:
                col = df[feat].drop_nulls()
                if len(col) > 0:
                    print(f"    {feat}: mean={col.mean():.3f}, std={col.std():.3f}, "
                          f"min={col.min():.3f}, max={col.max():.3f}")

    # Save
    out_path = FEATURE_STORE_DIR / f"bms_{mode}_features.parquet"
    df.write_parquet(out_path)
    if verbose:
        print(f"  Saved to: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")

    return df


if __name__ == "__main__":
    charging_df, operational_df = extract()
