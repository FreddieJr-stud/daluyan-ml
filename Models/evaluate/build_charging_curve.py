"""Build empirical charging power-vs-SOC lookup curve from DuckDB telemetry.

Queries charging_telemetry, buckets by 1% SOC, computes median/p25/p75 power.
Output: artifacts/charging_curve.json

Usage:
    python -m evaluate.build_charging_curve
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR, DUCKDB_PATH, BATTERY_CAPACITY_KWH


def build_curve() -> dict:
    """Build SOC-to-charging-power lookup from historical telemetry."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)

    df = conn.execute("""
        SELECT
            CAST(FLOOR(soc_percent) AS INT) AS soc_bin,
            COUNT(*) AS n,
            MEDIAN(charger_power_kw) AS median_kw,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY charger_power_kw) AS p25_kw,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY charger_power_kw) AS p75_kw
        FROM charging_telemetry
        WHERE charger_power_kw > 10
          AND soc_percent IS NOT NULL
          AND soc_percent >= 20
          AND soc_percent < 100
        GROUP BY soc_bin
        ORDER BY soc_bin
    """).fetchdf()
    conn.close()

    curve = {
        "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
        "bins": df["soc_bin"].tolist(),
        "median_kw": [round(v, 2) for v in df["median_kw"].tolist()],
        "p25_kw": [round(v, 2) for v in df["p25_kw"].tolist()],
        "p75_kw": [round(v, 2) for v in df["p75_kw"].tolist()],
        "sample_counts": df["n"].tolist(),
    }

    out_path = ARTIFACTS_DIR / "charging_curve.json"
    with open(out_path, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"Saved charging curve ({len(curve['bins'])} bins) to {out_path}")

    # Also copy to rpi5_bundle
    rpi5_path = Path(__file__).resolve().parent.parent / "rpi5_bundle" / "charging_curve.json"
    with open(rpi5_path, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"Copied to {rpi5_path}")

    return curve


if __name__ == "__main__":
    curve = build_curve()
    print(f"\nSOC range: {curve['bins'][0]}% - {curve['bins'][-1]}%")
    print(f"Peak median power: {max(curve['median_kw']):.1f} kW")
    print(f"Min median power: {min(curve['median_kw']):.1f} kW")
    print(f"Total bins: {len(curve['bins'])}")
