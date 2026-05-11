"""Backfill sea_surface_temperature_c into weather_hourly table.

Standalone script — no backend (app.*) imports. Reads distinct dates from
DuckDB, calls Open-Meteo Marine API for SST, and UPDATEs existing rows.

Usage:
    python -m data.backfill_sst            # write to DB
    python -m data.backfill_sst --dry-run  # preview without writing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DUCKDB_PATH

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
LATITUDE = 14.5873
LONGITUDE = 121.0107
TIMEZONE = "Asia/Manila"


def _fetch_sst(date: str) -> dict[str, float]:
    """Fetch hourly SST from Open-Meteo for a single date.

    Returns mapping of ISO time -> SST value (°C).
    """
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            MARINE_API_URL,
            params={
                "latitude": LATITUDE,
                "longitude": LONGITUDE,
                "start_date": date,
                "end_date": date,
                "timezone": TIMEZONE,
                "hourly": "sea_surface_temperature",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    sst_vals = hourly.get("sea_surface_temperature", [])

    result = {}
    for t, v in zip(times, sst_vals):
        if v is not None:
            result[t] = v
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill SST into weather_hourly")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    db = duckdb.connect(DUCKDB_PATH, read_only=args.dry_run)

    # Ensure column exists
    if not args.dry_run:
        try:
            db.execute("ALTER TABLE weather_hourly ADD COLUMN sea_surface_temperature_c DOUBLE")
            print("Added sea_surface_temperature_c column")
        except Exception:
            pass  # column already exists

    # Get all dates
    dates = [
        row[0]
        for row in db.execute(
            "SELECT DISTINCT date FROM weather_hourly ORDER BY date"
        ).fetchall()
    ]
    print(f"Found {len(dates)} dates in weather_hourly")

    total_updated = 0
    all_sst = []

    for i, date in enumerate(dates):
        try:
            sst_map = _fetch_sst(date)
        except Exception as e:
            print(f"  [{i+1}/{len(dates)}] {date}: API error — {e}")
            continue

        if not sst_map:
            print(f"  [{i+1}/{len(dates)}] {date}: no SST data")
            continue

        sst_values = list(sst_map.values())
        all_sst.extend(sst_values)

        if args.dry_run:
            print(
                f"  [{i+1}/{len(dates)}] {date}: {len(sst_map)} hours, "
                f"SST {min(sst_values):.1f}–{max(sst_values):.1f}°C"
            )
        else:
            for time_str, sst_val in sst_map.items():
                db.execute(
                    "UPDATE weather_hourly SET sea_surface_temperature_c = ? "
                    "WHERE date = ? AND time = ?",
                    [sst_val, date, time_str],
                )
            count = db.execute(
                "SELECT COUNT(*) FROM weather_hourly WHERE date = ? "
                "AND sea_surface_temperature_c IS NOT NULL",
                [date],
            ).fetchone()[0]
            total_updated += count
            print(
                f"  [{i+1}/{len(dates)}] {date}: {count} rows updated, "
                f"SST {min(sst_values):.1f}–{max(sst_values):.1f}°C"
            )

    # Summary
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Dates processed: {len(dates)}")
    if not args.dry_run:
        print(f"  Rows with SST: {total_updated}")
    if all_sst:
        print(f"  SST range: {min(all_sst):.1f}–{max(all_sst):.1f}°C")
        print(f"  SST mean:  {sum(all_sst)/len(all_sst):.2f}°C")

    db.close()


if __name__ == "__main__":
    main()
