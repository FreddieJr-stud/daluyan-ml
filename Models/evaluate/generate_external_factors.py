"""
Generate individual binned bar charts showing effect of external factors on SOC.
Each chart controls for direction (upstream/downstream) and normalizes to SOC/km.

Output: artifacts/progress_report/ext_*.png

Usage:
    python -m evaluate.generate_external_factors
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
FEATURE_STORE = ARTIFACTS / "feature_store"
OUT = ARTIFACTS / "progress_report"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 200,
})

UP_COLOR = "#2563eb"
DN_COLOR = "#dc2626"


def _binned_bar_split(
    axes: tuple,
    df: pl.DataFrame,
    factor_col: str,
    bin_edges: list[float],
    bin_labels: list[str],
    xlabel: str,
    title: str,
    show_n: bool = True,
):
    """Plot separate bar charts for upstream/downstream with tight y-axes.

    axes: (ax_upstream, ax_downstream) — two side-by-side axes.
    Each direction gets its own y-range so small differences are visible.
    """
    ax_up, ax_dn = axes

    # Compute SOC per km
    df = df.with_columns(
        (pl.col("soc_delta") / pl.col("route_distance_km")).alias("soc_per_km")
    )

    # Assign bins
    df = df.with_columns(
        pl.when(pl.col(factor_col) <= bin_edges[0])
        .then(pl.lit(bin_labels[0]))
        .when(pl.col(factor_col) <= bin_edges[1])
        .then(pl.lit(bin_labels[1]))
        .otherwise(pl.lit(bin_labels[2]))
        .alias("bin")
    )

    x = np.arange(len(bin_labels))
    width = 0.55

    for ax, direction, color, dir_label in [
        (ax_up, 0, UP_COLOR, "Upstream"),
        (ax_dn, 1, DN_COLOR, "Downstream"),
    ]:
        means = []
        stds = []
        ns = []
        for bl in bin_labels:
            subset = df.filter(
                (pl.col("bin") == bl) & (pl.col("direction_encoded") == direction)
            )["soc_per_km"]
            vals = subset.to_numpy()
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            stds.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ns.append(len(vals))

        bars = ax.bar(
            x, means, width,
            yerr=stds, capsize=5,
            color=color, alpha=0.85, edgecolor="white", linewidth=1.2,
        )

        # Value + count labels
        for bar, m, se, n in zip(bars, means, stds, ns):
            y_pos = m + se + 0.01
            text = f"{m:.2f}"
            if show_n:
                text += f"\n(n={n})"
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos,
                text, ha="center", fontsize=9, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("SOC per km (%/km)", fontsize=10)
        ax.set_title(f"{dir_label}", fontweight="bold", fontsize=12, color=color)

        # Tight y-axis: pad 15% below min and 25% above max for labels
        min_val = min(m - s for m, s in zip(means, stds)) if means else 0
        max_val = max(m + s for m, s in zip(means, stds)) if means else 1
        data_range = max_val - min_val
        if data_range < 0.1:
            data_range = 0.3  # minimum visible range
        pad_bottom = data_range * 0.6
        pad_top = data_range * 1.2
        y_lo = max(0, min_val - pad_bottom)
        y_hi = max_val + pad_top
        ax.set_ylim(y_lo, y_hi)


def fig_passengers():
    """Physical Load / Number of Passengers effect on SOC/km."""
    tf = pl.read_parquet(FEATURE_STORE / "trip_features.parquet")
    real = tf.filter(pl.col("passengers_on_board") != 23)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Effect of Passenger Load on Energy Consumption\n(real ridership data only, n=165)",
        fontsize=16, fontweight="bold",
    )

    _binned_bar_split(
        (axes[0], axes[1]), real,
        factor_col="passengers_on_board",
        bin_edges=[10, 25],
        bin_labels=["Light\n(0–10 pax)", "Medium\n(11–25 pax)", "Heavy\n(26–40 pax)"],
        xlabel="Passenger Load",
        title="",
    )

    fig.text(
        0.5, 0.01,
        f"Ridership data available for {len(real)}/643 segments (25.7%). Remaining 478 use median imputation (23 pax).",
        ha="center", fontsize=9, fontstyle="italic", color="#6b7280",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(OUT / "ext_1_passengers.png", bbox_inches="tight")
    plt.close(fig)
    print("  [1/4] Passenger load effect")


def fig_tides():
    """Tidal effect on SOC/km."""
    tf = pl.read_parquet(FEATURE_STORE / "trip_features.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Effect of Tidal Conditions on Energy Consumption",
        fontsize=16, fontweight="bold",
    )

    _binned_bar_split(
        (axes[0], axes[1]), tf,
        factor_col="tide_height_m",
        bin_edges=[0.15, 0.55],
        bin_labels=["Low Tide\n(< 0.15 m)", "Mid Tide\n(0.15–0.55 m)", "High Tide\n(> 0.55 m)"],
        xlabel="Tidal Condition",
        title="",
    )

    fig.text(
        0.5, 0.01,
        "Tidal predictions from calibrated harmonic model (8 constituents, Val RMSE=0.128m). Range: -0.43m to 1.15m.",
        ha="center", fontsize=9, fontstyle="italic", color="#6b7280",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(OUT / "ext_2_tides.png", bbox_inches="tight")
    plt.close(fig)
    print("  [2/4] Tidal effect")


def fig_current_temp():
    """River current and temperature effect on SOC/km."""
    tf = pl.read_parquet(FEATURE_STORE / "trip_features.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Effect of River Current and Temperature on Energy Consumption",
        fontsize=16, fontweight="bold",
    )

    # Current: top row
    _binned_bar_split(
        (axes[0, 0], axes[0, 1]), tf,
        factor_col="current_component_along_route",
        bin_edges=[-0.05, 0.05],
        bin_labels=["Against\n(< -0.05 m/s)", "Slack\n(-0.05 to 0.05)", "With\n(> 0.05 m/s)"],
        xlabel="Current Relative to Travel Direction",
        title="",
    )
    axes[0, 0].set_title("Current — Upstream", fontweight="bold", fontsize=12, color=UP_COLOR)
    axes[0, 1].set_title("Current — Downstream", fontweight="bold", fontsize=12, color=DN_COLOR)

    # Temperature: bottom row
    _binned_bar_split(
        (axes[1, 0], axes[1, 1]), tf,
        factor_col="temperature_c",
        bin_edges=[26.0, 30.0],
        bin_labels=["Cool\n(< 26°C)", "Moderate\n(26–30°C)", "Hot\n(> 30°C)"],
        xlabel="Air Temperature",
        title="",
    )
    axes[1, 0].set_title("Temperature — Upstream", fontweight="bold", fontsize=12, color=UP_COLOR)
    axes[1, 1].set_title("Temperature — Downstream", fontweight="bold", fontsize=12, color=DN_COLOR)

    fig.text(
        0.5, 0.01,
        "Current: hourly Open-Meteo Marine API (projected along route heading). Temperature: hourly weather observation.",
        ha="center", fontsize=9, fontstyle="italic", color="#6b7280",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT / "ext_3_current_temperature.png", bbox_inches="tight")
    plt.close(fig)
    print("  [3/4] Current & temperature effect")


def fig_weather():
    """Wind effect on SOC/km."""
    tf = pl.read_parquet(FEATURE_STORE / "trip_features.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Effect of Weather Conditions (Wind) on Energy Consumption",
        fontsize=16, fontweight="bold",
    )

    # Wind direction: top row
    _binned_bar_split(
        (axes[0, 0], axes[0, 1]), tf,
        factor_col="wind_component_along_route",
        bin_edges=[-1.0, 1.0],
        bin_labels=["Headwind\n(< -1.0 kn)", "Calm\n(-1.0 to 1.0 kn)", "Tailwind\n(> 1.0 kn)"],
        xlabel="Wind Relative to Travel Direction",
        title="",
    )
    axes[0, 0].set_title("Wind Direction — Upstream", fontweight="bold", fontsize=12, color=UP_COLOR)
    axes[0, 1].set_title("Wind Direction — Downstream", fontweight="bold", fontsize=12, color=DN_COLOR)

    # Wind speed: bottom row
    _binned_bar_split(
        (axes[1, 0], axes[1, 1]), tf,
        factor_col="wind_speed_kn",
        bin_edges=[2.0, 5.0],
        bin_labels=["Light\n(< 2 kn)", "Moderate\n(2–5 kn)", "Strong\n(> 5 kn)"],
        xlabel="Wind Speed",
        title="",
    )
    axes[1, 0].set_title("Wind Speed — Upstream", fontweight="bold", fontsize=12, color=UP_COLOR)
    axes[1, 1].set_title("Wind Speed — Downstream", fontweight="bold", fontsize=12, color=DN_COLOR)

    fig.text(
        0.5, 0.01,
        "Wind data: hourly Open-Meteo observations. Wind component projected along route heading (+ve = tailwind).",
        ha="center", fontsize=9, fontstyle="italic", color="#6b7280",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(OUT / "ext_4_weather.png", bbox_inches="tight")
    plt.close(fig)
    print("  [4/4] Weather (wind) effect")


def main():
    print("Generating external factor visualizations...")
    print(f"Output: {OUT}\n")

    fig_passengers()
    fig_tides()
    fig_current_temp()
    fig_weather()

    print(f"\nDone! 4 images saved to {OUT}")


if __name__ == "__main__":
    main()
