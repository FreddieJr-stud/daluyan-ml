"""Compute ETA baselines from historical DuckDB segment data.

Run offline on the development machine whenever data is updated.
Output: rpi5_bundle/config/eta_baselines.json

Usage:
    python -m data.compute_eta_baselines
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb

DUCKDB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "Daluyan_V2" / "backend" / "data" / "daluyan.duckdb"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "rpi5_bundle" / "config" / "eta_baselines.json"
)

# Station cumulative km (mirrors STATIONS in dashboard_backend.py)
STATION_KM = {
    "Napindan": 0.0, "Guadalupe": 6.5, "Hulo": 8.3,
    "Valenzuela": 9.8, "Lambingan": 11.2, "Sta Ana": 12.7,
    "PUP": 14.0, "Quinta": 15.6, "Escolta": 16.0,
}

KM_TO_NM = 1 / 1.852


def main() -> None:
    if not DUCKDB_PATH.exists():
        print(f"ERROR: DuckDB not found at {DUCKDB_PATH}")
        return

    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)

    # --- 1. Segment travel times ---
    rows = con.execute("""
        SELECT
            departure_station, arrival_station, direction,
            COUNT(*) AS n,
            MEDIAN(EXTRACT(EPOCH FROM (arrival_time - departure_time))) AS median_time_s
        FROM segments
        WHERE departure_station IS NOT NULL
          AND arrival_station IS NOT NULL
          AND direction IN ('upstream', 'downstream')
          AND EXTRACT(EPOCH FROM (arrival_time - departure_time)) BETWEEN 30 AND 7200
        GROUP BY departure_station, arrival_station, direction
    """).fetchall()

    segment_baselines: dict[str, dict] = {}
    for dep, arr, direction, n, median_s in rows:
        km_dep = STATION_KM.get(dep)
        km_arr = STATION_KM.get(arr)
        if km_dep is not None and km_arr is not None:
            distance_km = abs(km_arr - km_dep)
            distance_nm = distance_km * KM_TO_NM
            hours = median_s / 3600
            avg_speed_kn = distance_nm / hours if hours > 0 else 5.0
        else:
            distance_km = 0.0
            avg_speed_kn = 5.0

        key = f"{dep}|{arr}|{direction}"
        segment_baselines[key] = {
            "time_s": round(median_s, 1),
            "distance_km": round(distance_km, 2),
            "avg_speed_kn": round(avg_speed_kn, 2),
            "count": n,
        }

    # --- 2. Dwell times (consecutive segments at same station) ---
    dwell_rows = con.execute("""
        SELECT
            s1.arrival_station AS station,
            s1.direction,
            COUNT(*) AS n,
            MEDIAN(EXTRACT(EPOCH FROM (s2.departure_time - s1.arrival_time)))
                AS median_dwell_s
        FROM segments s1
        JOIN segments s2
          ON s1.arrival_station = s2.departure_station
         AND CAST(s1.departure_time AS DATE) = CAST(s2.departure_time AS DATE)
         AND s2.departure_time > s1.arrival_time
         AND EXTRACT(EPOCH FROM (s2.departure_time - s1.arrival_time))
             BETWEEN 10 AND 600
         AND s1.direction = s2.direction
         AND s1.direction IN ('upstream', 'downstream')
        GROUP BY s1.arrival_station, s1.direction
    """).fetchall()

    dwell_baselines: dict[str, dict] = {}
    for station, direction, n, median_dwell in dwell_rows:
        key = f"{station}|{direction}"
        dwell_baselines[key] = {
            "dwell_s": round(median_dwell),
            "count": n,
        }

    con.close()

    # --- 3. Export ---
    output = {
        "segment_baselines": segment_baselines,
        "dwell_baselines": dwell_baselines,
        "generated_at": datetime.now().isoformat(),
        "source": str(DUCKDB_PATH),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print(f"ETA baselines saved to {OUTPUT_PATH}")
    print(f"  Segment baselines: {len(segment_baselines)} route-direction pairs")
    print(f"  Dwell baselines:   {len(dwell_baselines)} station-direction pairs")

    # Summary table
    print("\nSegment baselines:")
    for key, val in sorted(segment_baselines.items()):
        print(f"  {key:40s}  {val['time_s']:7.1f}s  "
              f"{val['avg_speed_kn']:5.2f}kn  (n={val['count']})")

    print("\nDwell baselines:")
    for key, val in sorted(dwell_baselines.items()):
        print(f"  {key:30s}  {val['dwell_s']:5.0f}s  (n={val['count']})")


if __name__ == "__main__":
    main()
