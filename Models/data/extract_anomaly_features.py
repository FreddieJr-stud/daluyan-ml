"""Extract motor behavior features for anomaly detection (Model 2).

Reads trip segment parquets, filters to active transit (speed >= 0.5 kn),
computes motor symmetry and efficiency ratios, subsamples every 5 seconds.

Usage:
    python -m data.extract_anomaly_features
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANOMALY_FEATURES,
    ANOMALY_RAW_FEATURES,
    DUCKDB_PATH,
    FEATURE_STORE_DIR,
)

SUBSAMPLE_INTERVAL_S = 5
EPS = 0.01
SPEED_THRESHOLD_KN = 0.5


def _get_valid_parquet_paths(db: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Return (segment_id, parquet_path) for valid segments."""
    rows = db.execute("""
        SELECT s.segment_id, s.parquet_path, s.departure_time
        FROM segments s
        JOIN segment_statistics st ON s.segment_id = st.segment_id
        WHERE s.direction != 'unknown'
          AND st.distance_km >= 0.1
          AND st.duration_seconds >= 30
        ORDER BY s.departure_time
    """).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def _process_parquet(parquet_path: str) -> pl.DataFrame | None:
    """Extract motor features from one trip parquet."""
    path = Path(parquet_path)
    if not path.exists():
        return None

    df = pl.read_parquet(path)

    # Filter to active transit only
    df = df.filter(pl.col("speed_over_ground") >= SPEED_THRESHOLD_KN)
    if len(df) < 10:
        return None

    # Select raw features
    result = df.select([
        pl.col("port_motor_power"),
        pl.col("stbd_motor_power"),
        pl.col("motor_power_combined"),
        pl.col("port_rpm"),
        pl.col("stbd_rpm"),
        pl.col("speed_over_ground"),
        pl.col("battery_power"),
        pl.col("timestamp"),
    ])

    # Compute derived features
    result = result.with_columns([
        # Power symmetry
        (pl.col("port_motor_power") / (pl.col("port_motor_power") + pl.col("stbd_motor_power") + EPS))
            .alias("port_stbd_power_ratio"),
        (pl.col("port_motor_power") - pl.col("stbd_motor_power")).abs()
            .alias("port_stbd_power_diff"),
        # RPM symmetry
        (pl.col("port_rpm") / (pl.col("port_rpm") + pl.col("stbd_rpm") + EPS))
            .alias("port_stbd_rpm_ratio"),
        (pl.col("port_rpm") - pl.col("stbd_rpm")).abs()
            .alias("port_stbd_rpm_diff"),
        # Efficiency
        (pl.col("port_motor_power") / (pl.col("port_rpm") + EPS))
            .alias("port_power_per_rpm"),
        (pl.col("stbd_motor_power") / (pl.col("stbd_rpm") + EPS))
            .alias("stbd_power_per_rpm"),
        (pl.col("motor_power_combined") / (pl.col("speed_over_ground") + EPS))
            .alias("combined_power_per_speed"),
        (pl.col("motor_power_combined") / (pl.col("speed_over_ground") ** 2 + EPS))
            .alias("power_speed_squared_ratio"),
    ])

    # Subsample
    result = result.gather_every(SUBSAMPLE_INTERVAL_S)

    return result.select(ANOMALY_FEATURES + ["timestamp"])


def extract(verbose: bool = True) -> pl.DataFrame:
    """Build the anomaly detection feature DataFrame."""
    db = duckdb.connect(DUCKDB_PATH, read_only=True)
    segments = _get_valid_parquet_paths(db)
    db.close()

    if verbose:
        print(f"Valid segments: {len(segments)}")

    all_dfs: list[pl.DataFrame] = []
    skipped = 0
    # Track departure_time for temporal splitting
    seg_times: list[pl.DataFrame] = []

    for i, (seg_id, parquet_path, dep_time) in enumerate(segments):
        result = _process_parquet(parquet_path)
        if result is not None and len(result) > 0:
            # Add segment_id and departure_time for splitting later
            result = result.with_columns([
                pl.lit(seg_id).alias("segment_id"),
                pl.lit(dep_time).alias("departure_time"),
            ])
            all_dfs.append(result)
        else:
            skipped += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(segments)} segments...")

    if not all_dfs:
        print("ERROR: No data extracted!")
        sys.exit(1)

    df = pl.concat(all_dfs, how="diagonal_relaxed")
    df = df.fill_null(0.0).fill_nan(0.0)

    if verbose:
        print(f"\nFinal dataset: {len(df)} rows from {len(all_dfs)} segments (skipped {skipped})")
        for feat in ANOMALY_RAW_FEATURES[:4]:
            col = df[feat]
            print(f"  {feat}: mean={col.mean():.2f}, std={col.std():.2f}, "
                  f"min={col.min():.2f}, max={col.max():.2f}")

    out_path = FEATURE_STORE_DIR / "anomaly_features.parquet"
    df.write_parquet(out_path)
    if verbose:
        print(f"\nSaved to: {out_path}")

    return df


if __name__ == "__main__":
    extract()
