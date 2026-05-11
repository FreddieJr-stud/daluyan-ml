"""Cross-Architecture Model Comparison for M/B Dalaray models.

Loads metadata from XGBoost, Random Forest, MLP/LSTM, and anomaly detection
models, then produces summary tables (printed + CSV) and bar charts (PNG)
for the thesis.

Usage:
    python -m evaluate.compare_models
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR


# ── Metadata Loading ─────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path.name} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def _safe_float(val) -> float:
    """Convert metric value to float (handles string MAPE values)."""
    if val is None:
        return float("nan")
    return float(val)


# ── Comparison Tables ────────────────────────────────────────────────────

def compare_trip_models() -> list[dict]:
    """Compare Model 1A across architectures."""
    print("\n" + "=" * 80)
    print("  MODEL 1A — Trip-Level SOC Prediction (Test Set)")
    print("=" * 80)

    entries = []

    # XGBoost
    meta = _load_json(ARTIFACTS_DIR / "soc_trip_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "XGBoost (v6)",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
            "Ensemble": f"{meta.get('n_ensemble_seeds', 1)} seeds",
        })

    # Random Forest
    meta = _load_json(ARTIFACTS_DIR / "rf_soc_trip_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "Random Forest",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
            "Ensemble": "1 model (bagged)",
        })

    # MLP
    meta = _load_json(ARTIFACTS_DIR / "mlp_soc_trip_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "MLP",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
            "Ensemble": "1 model",
        })

    if not entries:
        print("  No Model 1A results found.")
        return entries

    # Print table
    header = f"  {'Architecture':<20s} | {'MAE':>7s} | {'RMSE':>7s} | {'R2':>7s} | {'MAPE':>7s} | {'Trials':>7s} | {'Ensemble':<18s}"
    sep = f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*18}"
    print(header)
    print(sep)
    for e in entries:
        print(f"  {e['Architecture']:<20s} | {e['MAE']:>6.3f}% | {e['RMSE']:>6.3f}% | "
              f"{e['R2']:>7.4f} | {e['MAPE']:>6.1f}% | {str(e['Optuna Trials']):>7s} | {e['Ensemble']:<18s}")

    return entries


def compare_realtime_models() -> list[dict]:
    """Compare Model 1B across architectures."""
    print("\n" + "=" * 80)
    print("  MODEL 1B — Realtime SOC Range Prediction (Test Set)")
    print("=" * 80)

    entries = []

    # XGBoost
    meta = _load_json(ARTIFACTS_DIR / "soc_realtime_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "XGBoost (v6)",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
        })

    # Random Forest
    meta = _load_json(ARTIFACTS_DIR / "rf_soc_realtime_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "Random Forest",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
        })

    # LSTM
    meta = _load_json(ARTIFACTS_DIR / "lstm_soc_realtime_metadata.json")
    if meta:
        test = meta["metrics"]["Test"]
        entries.append({
            "Architecture": "LSTM",
            "MAE": _safe_float(test["MAE"]),
            "RMSE": _safe_float(test["RMSE"]),
            "R2": _safe_float(test["R2"]),
            "MAPE": _safe_float(test["MAPE"]),
            "Optuna Trials": meta.get("optuna_trials", "—"),
        })

    if not entries:
        print("  No Model 1B results found.")
        return entries

    # Print table
    header = f"  {'Architecture':<20s} | {'MAE':>7s} | {'RMSE':>7s} | {'R2':>7s} | {'MAPE':>7s} | {'Trials':>7s}"
    sep = f"  {'-'*20}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}"
    print(header)
    print(sep)
    for e in entries:
        print(f"  {e['Architecture']:<20s} | {e['MAE']:>6.3f}% | {e['RMSE']:>6.3f}% | "
              f"{e['R2']:>7.4f} | {e['MAPE']:>6.1f}% | {str(e['Optuna Trials']):>7s}")

    # Progress breakdown (if available)
    _print_progress_comparison()

    return entries


def _compute_xgb_progress_breakdown() -> dict:
    """Compute XGBoost per-progress MAE on fixed test set."""
    try:
        import xgboost as xgb
        import polars as pl
        from datetime import datetime

        from config import (FEATURE_STORE_DIR, REALTIME_FEATURES,
                            REALTIME_TARGET, VAL_END, ENSEMBLE_SEEDS)

        df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
        test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
        X_test = test.select(REALTIME_FEATURES).to_numpy().astype(np.float32)
        y_test = test[REALTIME_TARGET].to_numpy().astype(np.float32)
        progress = test["trip_progress_fraction"].to_numpy()

        preds_list = []
        for seed in ENSEMBLE_SEEDS:
            suffix = "" if seed == 42 else f"_seed{seed}"
            path = ARTIFACTS_DIR / f"soc_realtime_model{suffix}.json"
            if not path.exists():
                return {}
            model = xgb.XGBRegressor()
            model.load_model(str(path))
            preds_list.append(model.predict(X_test))
        y_pred = np.mean(preds_list, axis=0)

        breakdown = {}
        for lo, hi, label in [(0, 0.25, "0-25%"), (0.25, 0.50, "25-50%"),
                               (0.50, 0.75, "50-75%"), (0.75, 1.01, "75-101%")]:
            mask = (progress >= lo) & (progress < hi)
            if mask.sum() > 0:
                breakdown[label] = float(np.mean(np.abs(y_test[mask] - y_pred[mask])))
        return breakdown
    except Exception:
        return {}


def _print_progress_comparison():
    """Print per-progress breakdown for all Model 1B architectures."""
    print("\n  Per-Progress MAE Breakdown (Test Set):")

    rf_meta = _load_json(ARTIFACTS_DIR / "rf_soc_realtime_metadata.json")
    lstm_meta = _load_json(ARTIFACTS_DIR / "lstm_soc_realtime_metadata.json")

    bins = ["0-25%", "25-50%", "50-75%", "75-100%"]
    # Some training scripts use "75-101%" (from 1.01 upper bound)
    bin_aliases = {"75-100%": "75-101%"}

    xgb_progress = _compute_xgb_progress_breakdown()
    rf_progress = {}
    lstm_progress = {}

    if rf_meta and "progress_breakdown" in rf_meta.get("metrics", {}):
        rf_progress = rf_meta["metrics"]["progress_breakdown"]
    if lstm_meta and "progress_breakdown" in lstm_meta.get("metrics", {}):
        lstm_progress = lstm_meta["metrics"]["progress_breakdown"]

    if not rf_progress and not lstm_progress and not xgb_progress:
        return

    header = f"  {'Progress':<10s} | {'XGB MAE':>8s} | {'RF MAE':>8s} | {'LSTM MAE':>8s}"
    print(header)
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for b in bins:
        alias = bin_aliases.get(b, b)
        xgb_key = b if b in xgb_progress else alias
        rf_key = b if b in rf_progress else alias
        lstm_key = b if b in lstm_progress else alias
        xgb_val = f"{xgb_progress[xgb_key]:.3f}%" if xgb_key in xgb_progress else "—"
        rf_val = f"{rf_progress[rf_key]:.3f}%" if rf_key in rf_progress else "—"
        lstm_val = f"{lstm_progress[lstm_key]:.3f}%" if lstm_key in lstm_progress else "—"
        print(f"  {b:<10s} | {xgb_val:>8s} | {rf_val:>8s} | {lstm_val:>8s}")


# ── Bar Charts ───────────────────────────────────────────────────────────

def plot_mae_comparison(trip_entries: list[dict], rt_entries: list[dict]):
    """Create bar chart comparing MAE across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"XGBoost (v6)": "#2196F3", "Random Forest": "#4CAF50", "MLP": "#FF9800", "LSTM": "#FF9800"}

    # Model 1A
    ax = axes[0]
    if trip_entries:
        names = [e["Architecture"] for e in trip_entries]
        maes = [e["MAE"] for e in trip_entries]
        bars = ax.bar(names, maes, color=[colors.get(n, "#9E9E9E") for n in names],
                       edgecolor="black", linewidth=0.5)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mae:.3f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Test MAE (% SOC)")
        ax.set_title("Model 1A — Trip-Level SOC")
        ax.set_ylim(0, max(maes) * 1.3 if maes else 1)

    # Model 1B
    ax = axes[1]
    if rt_entries:
        names = [e["Architecture"] for e in rt_entries]
        maes = [e["MAE"] for e in rt_entries]
        bars = ax.bar(names, maes, color=[colors.get(n, "#9E9E9E") for n in names],
                       edgecolor="black", linewidth=0.5)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{mae:.3f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Test MAE (% SOC)")
        ax.set_title("Model 1B — Realtime SOC")
        ax.set_ylim(0, max(maes) * 1.3 if maes else 1)

    plt.suptitle("Architecture Comparison — Test MAE", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = ARTIFACTS_DIR / "architecture_comparison_mae.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Bar chart saved to: {save_path}")
    plt.close(fig)


def plot_r2_comparison(trip_entries: list[dict], rt_entries: list[dict]):
    """Create bar chart comparing R2 across architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"XGBoost (v6)": "#2196F3", "Random Forest": "#4CAF50", "MLP": "#FF9800", "LSTM": "#FF9800"}

    # Model 1A
    ax = axes[0]
    if trip_entries:
        names = [e["Architecture"] for e in trip_entries]
        r2s = [e["R2"] for e in trip_entries]
        bars = ax.bar(names, r2s, color=[colors.get(n, "#9E9E9E") for n in names],
                       edgecolor="black", linewidth=0.5)
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{r2:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Test R²")
        ax.set_title("Model 1A — Trip-Level SOC")
        ax.set_ylim(min(r2s) * 0.95 if r2s else 0, 1.0)

    # Model 1B
    ax = axes[1]
    if rt_entries:
        names = [e["Architecture"] for e in rt_entries]
        r2s = [e["R2"] for e in rt_entries]
        bars = ax.bar(names, r2s, color=[colors.get(n, "#9E9E9E") for n in names],
                       edgecolor="black", linewidth=0.5)
        for bar, r2 in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{r2:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Test R²")
        ax.set_title("Model 1B — Realtime SOC")
        ax.set_ylim(min(r2s) * 0.95 if r2s else 0, 1.0)

    plt.suptitle("Architecture Comparison — Test R²", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = ARTIFACTS_DIR / "architecture_comparison_r2.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  R² chart saved to: {save_path}")
    plt.close(fig)


# ── CSV Export ───────────────────────────────────────────────────────────

def save_comparison_csv(trip_entries: list[dict], rt_entries: list[dict]):
    """Save comparison tables as CSV."""
    import csv

    # Model 1A
    if trip_entries:
        path = ARTIFACTS_DIR / "comparison_model_1a.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Architecture", "MAE", "RMSE", "R2", "MAPE"])
            writer.writeheader()
            for e in trip_entries:
                writer.writerow({
                    "Architecture": e["Architecture"],
                    "MAE": f"{e['MAE']:.4f}",
                    "RMSE": f"{e['RMSE']:.4f}",
                    "R2": f"{e['R2']:.6f}",
                    "MAPE": f"{e['MAPE']:.2f}",
                })
        print(f"  CSV saved: {path.name}")

    # Model 1B
    if rt_entries:
        path = ARTIFACTS_DIR / "comparison_model_1b.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Architecture", "MAE", "RMSE", "R2", "MAPE"])
            writer.writeheader()
            for e in rt_entries:
                writer.writerow({
                    "Architecture": e["Architecture"],
                    "MAE": f"{e['MAE']:.4f}",
                    "RMSE": f"{e['RMSE']:.4f}",
                    "R2": f"{e['R2']:.6f}",
                    "MAPE": f"{e['MAPE']:.2f}",
                })
        print(f"  CSV saved: {path.name}")


# ── Model 2: Anomaly Detection Comparison ─────────────────────────────

def compare_anomaly_models() -> list[dict]:
    """Compare Model 2 (anomaly detection) across architectures."""
    print("\n" + "=" * 80)
    print("  MODEL 2 — Motor Anomaly Detection (Synthetic Injection Test)")
    print("=" * 80)

    entries = []

    # XGBoost (production)
    meta = _load_json(ARTIFACTS_DIR / "anomaly_metadata.json")
    if meta and "injection_test" in meta:
        inj = meta["injection_test"]
        entries.append({
            "Architecture": "XGBoost Recon.",
            "Detection Rate": _safe_float(inj["detection_rate"]),
            "FPR": _safe_float(inj["false_positive_rate"]),
            "Threshold": _safe_float(inj["threshold"]),
            "Optuna Trials": f"{meta.get('optuna_trials_per_target', '—')}/target",
            "Paradigm": "Reconstruction",
        })

    # RF Reconstruction
    meta = _load_json(ARTIFACTS_DIR / "rf_anomaly_metadata.json")
    if meta and "injection_test" in meta:
        inj = meta["injection_test"]
        entries.append({
            "Architecture": "RF Recon.",
            "Detection Rate": _safe_float(inj["detection_rate"]),
            "FPR": _safe_float(inj["false_positive_rate"]),
            "Threshold": _safe_float(inj["threshold"]),
            "Optuna Trials": f"{meta.get('optuna_trials_per_target', '—')}/target",
            "Paradigm": "Reconstruction",
        })

    # MLP Autoencoder
    meta = _load_json(ARTIFACTS_DIR / "ae_anomaly_metadata.json")
    if meta and "injection_test" in meta:
        inj = meta["injection_test"]
        entries.append({
            "Architecture": "MLP Autoencoder",
            "Detection Rate": _safe_float(inj["detection_rate"]),
            "FPR": _safe_float(inj["false_positive_rate"]),
            "Threshold": _safe_float(inj["threshold"]),
            "Optuna Trials": str(meta.get("optuna_trials", "—")),
            "Paradigm": "Autoencoder",
        })

    # Isolation Forest
    meta = _load_json(ARTIFACTS_DIR / "if_anomaly_metadata.json")
    if meta and "injection_test" in meta:
        inj = meta["injection_test"]
        entries.append({
            "Architecture": "Isolation Forest",
            "Detection Rate": _safe_float(inj["detection_rate"]),
            "FPR": _safe_float(inj["false_positive_rate"]),
            "Threshold": _safe_float(inj["threshold"]),
            "Optuna Trials": str(meta.get("optuna_trials", "—")),
            "Paradigm": "Direct Scoring",
        })

    if not entries:
        print("  No Model 2 results found.")
        return entries

    # Print table
    header = (f"  {'Architecture':<18s} | {'Detection':>10s} | {'FPR':>8s} | "
              f"{'Threshold':>10s} | {'Trials':>12s} | {'Paradigm':<16s}")
    sep = (f"  {'-'*18}-+-{'-'*10}-+-{'-'*8}-+-"
           f"{'-'*10}-+-{'-'*12}-+-{'-'*16}")
    print(header)
    print(sep)
    for e in entries:
        print(f"  {e['Architecture']:<18s} | {e['Detection Rate']:>9.1f}% | "
              f"{e['FPR']:>7.1f}% | {e['Threshold']:>10.4f} | "
              f"{e['Optuna Trials']:>12s} | {e['Paradigm']:<16s}")

    return entries


def plot_anomaly_comparison(entries: list[dict]):
    """Create grouped bar chart for anomaly detection comparison."""
    if not entries:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [e["Architecture"] for e in entries]
    colors_map = {
        "XGBoost Recon.": "#2196F3",
        "RF Recon.": "#4CAF50",
        "MLP Autoencoder": "#FF9800",
        "Isolation Forest": "#9C27B0",
    }
    bar_colors = [colors_map.get(n, "#9E9E9E") for n in names]

    # Detection Rate
    ax = axes[0]
    det_rates = [e["Detection Rate"] for e in entries]
    bars = ax.bar(names, det_rates, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, det_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Synthetic Injection Detection (port pwr x1.5)")
    ax.set_ylim(0, max(det_rates) * 1.15 if det_rates else 100)
    ax.tick_params(axis="x", rotation=15)

    # FPR
    ax = axes[1]
    fprs = [e["FPR"] for e in entries]
    bars = ax.bar(names, fprs, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("False Positive Rate (%)")
    ax.set_title("FPR at p99 Threshold")
    ax.set_ylim(0, max(fprs) * 1.5 if fprs else 5)
    ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model 2 — Anomaly Detection Architecture Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = ARTIFACTS_DIR / "anomaly_architecture_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Anomaly comparison chart saved to: {save_path}")
    plt.close(fig)


def save_anomaly_comparison_csv(entries: list[dict]):
    """Save anomaly comparison table as CSV."""
    import csv

    if not entries:
        return

    path = ARTIFACTS_DIR / "comparison_model_2_anomaly.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Architecture", "Detection Rate", "FPR", "Paradigm"],
        )
        writer.writeheader()
        for e in entries:
            writer.writerow({
                "Architecture": e["Architecture"],
                "Detection Rate": f"{e['Detection Rate']:.1f}",
                "FPR": f"{e['FPR']:.1f}",
                "Paradigm": e["Paradigm"],
            })
    print(f"  CSV saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  M/B Dalaray Digital Shadow — Architecture Comparison")
    print("=" * 80)

    trip_entries = compare_trip_models()
    rt_entries = compare_realtime_models()
    anomaly_entries = compare_anomaly_models()

    # Generate plots
    if trip_entries or rt_entries:
        print("\n  Generating SOC comparison plots...")
        plot_mae_comparison(trip_entries, rt_entries)
        plot_r2_comparison(trip_entries, rt_entries)
        save_comparison_csv(trip_entries, rt_entries)

    if anomaly_entries:
        print("\n  Generating anomaly comparison plots...")
        plot_anomaly_comparison(anomaly_entries)
        save_anomaly_comparison_csv(anomaly_entries)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    if trip_entries:
        best_1a = min(trip_entries, key=lambda e: e["MAE"])
        print(f"  Model 1A best: {best_1a['Architecture']} (MAE={best_1a['MAE']:.3f}%)")
    if rt_entries:
        best_1b = min(rt_entries, key=lambda e: e["MAE"])
        print(f"  Model 1B best: {best_1b['Architecture']} (MAE={best_1b['MAE']:.3f}%)")
    if anomaly_entries:
        best_2 = max(anomaly_entries, key=lambda e: e["Detection Rate"])
        print(f"  Model 2 best: {best_2['Architecture']} "
              f"(Detection={best_2['Detection Rate']:.1f}%, FPR={best_2['FPR']:.1f}%)")


if __name__ == "__main__":
    main()
