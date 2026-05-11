"""Charging analysis for thesis — profile characterization and operational patterns.

Generates:
  - artifacts/progress_report/charging_analysis.png (6-panel figure)
  - artifacts/progress_report/charging_summary_stats.csv

Usage:
    python -m evaluate.analyze_charging
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ARTIFACTS_DIR, DUCKDB_PATH

REPORT_DIR = ARTIFACTS_DIR / "progress_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load charging sessions and telemetry from DuckDB."""
    conn = duckdb.connect(DUCKDB_PATH, read_only=True)
    sessions = conn.execute(
        "SELECT * FROM charging_sessions WHERE energy_added_kwh > 0"
    ).fetchdf()
    telemetry = conn.execute(
        "SELECT * FROM charging_telemetry WHERE charger_power_kw > 0 AND soc_percent IS NOT NULL"
    ).fetchdf()
    conn.close()
    return sessions, telemetry


def plot_analysis(sessions: pd.DataFrame, telemetry: pd.DataFrame, curve_path: Path) -> None:
    """Generate 6-panel thesis figure."""
    with open(curve_path) as f:
        curve = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("M/B Dalaray Charging Analysis", fontsize=16, fontweight="bold")

    # Panel 1: Power vs SOC curve with confidence band
    ax = axes[0, 0]
    bins = curve["bins"]
    ax.fill_between(bins, curve["p25_kw"], curve["p75_kw"],
                    alpha=0.3, color="green", label="P25-P75")
    ax.plot(bins, curve["median_kw"], color="green", linewidth=2, label="Median")
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("Charging Power (kW)")
    ax.set_title("Charging Power vs SOC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Duration vs starting SOC
    ax = axes[0, 1]
    ax.scatter(sessions["soc_start"], sessions["duration_minutes"],
               c="steelblue", alpha=0.7, edgecolors="navy", s=50)
    ax.set_xlabel("Starting SOC (%)")
    ax.set_ylabel("Duration (minutes)")
    ax.set_title("Charge Duration vs Starting SOC")
    ax.grid(True, alpha=0.3)

    # Panel 3: Energy per session histogram
    ax = axes[0, 2]
    ax.hist(sessions["energy_added_kwh"], bins=15,
            color="teal", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Energy Added (kWh)")
    ax.set_ylabel("Count")
    ax.set_title("Energy per Charging Session")
    ax.grid(True, alpha=0.3)

    # Panel 4: Time of day histogram
    ax = axes[1, 0]
    hours = pd.to_datetime(sessions["start_time"]).dt.hour
    ax.hist(hours, bins=range(0, 25), color="orange", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Count")
    ax.set_title("Charging Start Time Distribution")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    # Panel 5: Discharge depth (SOC at plug-in)
    ax = axes[1, 1]
    ax.hist(sessions["soc_start"], bins=15,
            color="salmon", edgecolor="black", alpha=0.8)
    ax.set_xlabel("SOC at Plug-in (%)")
    ax.set_ylabel("Count")
    ax.set_title("Discharge Depth Before Charging")
    ax.grid(True, alpha=0.3)

    # Panel 6: Weekday vs weekend
    ax = axes[1, 2]
    sessions = sessions.copy()
    sessions["date_parsed"] = pd.to_datetime(sessions["date"])
    sessions["is_weekend"] = sessions["date_parsed"].dt.dayofweek >= 5
    weekday = sessions[~sessions["is_weekend"]]
    weekend = sessions[sessions["is_weekend"]]
    labels = ["Weekday", "Weekend"]
    counts = [len(weekday), len(weekend)]
    avg_energy = [
        weekday["energy_added_kwh"].mean() if len(weekday) > 0 else 0,
        weekend["energy_added_kwh"].mean() if len(weekend) > 0 else 0,
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, counts, 0.35, label="Sessions", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, avg_energy, 0.35, label="Avg Energy (kWh)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Session Count")
    ax2.set_ylabel("Avg Energy (kWh)")
    ax.set_title("Weekday vs Weekend Charging")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = REPORT_DIR / "charging_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


def summary_stats(sessions: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics table."""
    cols = ["duration_minutes", "energy_added_kwh", "avg_power_kw",
            "peak_power_kw", "soc_start", "soc_end"]
    stats = sessions[cols].describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    stats = stats.round(2)
    out_path = REPORT_DIR / "charging_summary_stats.csv"
    stats.to_csv(out_path)
    print(f"Saved stats to {out_path}")
    print(stats)
    return stats


if __name__ == "__main__":
    sessions, telemetry = load_data()
    print(f"Loaded {len(sessions)} sessions, {len(telemetry)} telemetry rows")

    curve_path = ARTIFACTS_DIR / "charging_curve.json"
    if not curve_path.exists():
        print("ERROR: Run build_charging_curve.py first")
        sys.exit(1)

    plot_analysis(sessions, telemetry, curve_path)
    summary_stats(sessions)
    print("\nDone!")
