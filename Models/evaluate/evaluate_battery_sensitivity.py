"""Graduated severity sweep for battery anomaly models (3a/3b).

Tests detection sensitivity at multiple degradation levels, per-feature
and per-side, to find the minimum detectable anomaly.

Usage:
    python -m evaluate.evaluate_battery_sensitivity
    python -m evaluate.evaluate_battery_sensitivity --model charging
    python -m evaluate.evaluate_battery_sensitivity --model operational
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    BATTERY_FEATURES,
    BATTERY_TARGETS_CHARGING,
    BATTERY_TARGETS_OPERATIONAL,
    FEATURE_STORE_DIR,
    TRAIN_END,
    VAL_END,
)

N_INJECTIONS = 500
SEED = 42

# Severity multipliers relative to normal baseline (mean + std)
SEVERITY_MULTIPLIERS = [1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]

# Features to inject (grouped by type)
VOLTAGE_FEATURES = ["port_cell_v_spread", "stbd_cell_v_spread", "cell_v_spread_combined"]
THERMAL_FEATURES = ["port_cell_t_spread", "stbd_cell_t_spread", "cell_t_spread_combined"]
PORT_FEATURES = ["port_cell_v_spread", "port_cell_t_spread"]
STBD_FEATURES = ["stbd_cell_v_spread", "stbd_cell_t_spread"]

REPORT_DIR = ARTIFACTS_DIR / "progress_report"


def _temporal_split(df: pl.DataFrame):
    """Same temporal split as training."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def _load_models_and_data(mode: str):
    """Load trained models, thresholds, statistics, and test data."""
    prefix = f"battery_{mode}"
    targets = (
        BATTERY_TARGETS_CHARGING if mode == "charging"
        else BATTERY_TARGETS_OPERATIONAL
    )

    # Load models
    models = {}
    for target in targets:
        model_path = ARTIFACTS_DIR / f"{prefix}_{target}.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = xgb.Booster()
        model.load_model(str(model_path))
        models[target] = model

    # Load thresholds
    with open(ARTIFACTS_DIR / f"{prefix}_thresholds.json") as f:
        thresholds = json.load(f)

    # Load statistics (for normalization)
    with open(ARTIFACTS_DIR / f"{prefix}_statistics.json") as f:
        stats = json.load(f)

    # Load data and split
    parquet_path = FEATURE_STORE_DIR / f"bms_{mode}_features.parquet"
    df = pl.read_parquet(parquet_path)
    features = [f for f in BATTERY_FEATURES if f in df.columns]
    _, _, test_df = _temporal_split(df)

    test_data = test_df.select(features).to_numpy().astype(np.float32)

    # Score function
    feature_stds = stats["feature_stds"]

    def score_fn(data):
        scores = np.zeros(len(data))
        for target_col, model in models.items():
            predictor_cols = [c for c in features if c != target_col]
            target_idx = features.index(target_col)
            predictor_idxs = [features.index(c) for c in predictor_cols]
            X = data[:, predictor_idxs]
            y = data[:, target_idx]
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            pred = model.predict(dm)
            error = ((pred - y) / feature_stds[target_col]) ** 2
            scores += error
        return scores

    return models, thresholds, stats, test_data, features, score_fn


def _inject_features(
    normal_data: np.ndarray,
    feature_names: list[str],
    inject_features: list[str],
    severity_multiplier: float,
    feature_stats: dict,
    n_injections: int = N_INJECTIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Inject anomalies at a given severity level.

    Severity is expressed as a multiplier of (mean + 2*std) of the feature.
    This represents how many times above the normal operating range the
    injected value is.
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(normal_data), size=n_injections, replace=True)
    corrupted = normal_data[indices].copy()

    means = feature_stats["feature_means"]
    stds = feature_stats["feature_stds"]

    for feat in inject_features:
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)
        # Target value = severity_multiplier * (mean + 2*std)
        baseline = abs(means[feat]) + 2 * stds[feat]
        target_value = severity_multiplier * baseline
        # Add 10% noise around target
        corrupted[:, idx] = target_value + rng.normal(0, target_value * 0.1, n_injections)

    return corrupted


def _detection_rate(
    score_fn, corrupted: np.ndarray, threshold: float
) -> float:
    """Compute detection rate as % of corrupted samples above threshold."""
    scores = score_fn(corrupted)
    return float((scores > threshold).sum() / len(scores) * 100)


# ─── Test 1: Graduated Multi-Feature Severity ────────────────────────────


def test_graduated_severity(
    score_fn, test_data, features, stats, threshold, mode,
) -> list[dict]:
    """Sweep severity multipliers with all features injected simultaneously."""
    print(f"\n  ── Test 1: Graduated Multi-Feature Severity ──")

    all_inject = [f for f in VOLTAGE_FEATURES + THERMAL_FEATURES if f in features]
    results = []

    for mult in SEVERITY_MULTIPLIERS:
        corrupted = _inject_features(
            test_data, features, all_inject, mult, stats,
        )
        det = _detection_rate(score_fn, corrupted, threshold)
        results.append({
            "test": "graduated_multi",
            "mode": mode,
            "severity_multiplier": mult,
            "features_injected": "all_voltage+thermal",
            "detection_rate": det,
        })
        print(f"    {mult:5.1f}x  →  {det:6.1f}% detection")

    return results


# ─── Test 2: Single-Feature Injections ────────────────────────────────────


def test_single_feature(
    score_fn, test_data, features, stats, threshold, mode,
) -> list[dict]:
    """Inject only one feature at a time at varying severity."""
    print(f"\n  ── Test 2: Single-Feature Sensitivity ──")

    # Test each injectable feature individually
    injectable = [f for f in VOLTAGE_FEATURES + THERMAL_FEATURES if f in features]
    results = []

    for feat in injectable:
        print(f"    {feat}:")
        for mult in SEVERITY_MULTIPLIERS:
            corrupted = _inject_features(
                test_data, features, [feat], mult, stats,
            )
            det = _detection_rate(score_fn, corrupted, threshold)
            results.append({
                "test": "single_feature",
                "mode": mode,
                "severity_multiplier": mult,
                "features_injected": feat,
                "detection_rate": det,
            })
        # Print summary for this feature
        dets = [r["detection_rate"] for r in results if r["features_injected"] == feat]
        min_det_idx = next((i for i, d in enumerate(dets) if d >= 95), len(dets) - 1)
        print(f"      95% detection at: {SEVERITY_MULTIPLIERS[min_det_idx]}x" if dets[min_det_idx] >= 95
              else f"      Max detection: {max(dets):.1f}% at {SEVERITY_MULTIPLIERS[-1]}x")

    return results


# ─── Test 3: One-Sided Failures ───────────────────────────────────────────


def test_one_sided(
    score_fn, test_data, features, stats, threshold, mode,
) -> list[dict]:
    """Inject only port-side or starboard-side features."""
    print(f"\n  ── Test 3: One-Sided Failures ──")

    results = []
    for side_name, side_feats in [("port_only", PORT_FEATURES), ("stbd_only", STBD_FEATURES)]:
        available = [f for f in side_feats if f in features]
        if not available:
            continue
        print(f"    {side_name} ({available}):")
        for mult in SEVERITY_MULTIPLIERS:
            corrupted = _inject_features(
                test_data, features, available, mult, stats,
            )
            det = _detection_rate(score_fn, corrupted, threshold)
            results.append({
                "test": "one_sided",
                "mode": mode,
                "severity_multiplier": mult,
                "features_injected": side_name,
                "detection_rate": det,
            })
        dets = [r["detection_rate"] for r in results if r["features_injected"] == side_name]
        min_det_idx = next((i for i, d in enumerate(dets) if d >= 95), len(dets) - 1)
        print(f"      95% detection at: {SEVERITY_MULTIPLIERS[min_det_idx]}x" if dets[min_det_idx] >= 95
              else f"      Max detection: {max(dets):.1f}% at {SEVERITY_MULTIPLIERS[-1]}x")

    return results


# ─── Test 4: Voltage-Only vs Thermal-Only ─────────────────────────────────


def test_type_isolation(
    score_fn, test_data, features, stats, threshold, mode,
) -> list[dict]:
    """Inject only voltage features or only thermal features."""
    print(f"\n  ── Test 4: Voltage-Only vs Thermal-Only ──")

    results = []
    for type_name, type_feats in [("voltage_only", VOLTAGE_FEATURES), ("thermal_only", THERMAL_FEATURES)]:
        available = [f for f in type_feats if f in features]
        if not available:
            continue
        print(f"    {type_name} ({available}):")
        for mult in SEVERITY_MULTIPLIERS:
            corrupted = _inject_features(
                test_data, features, available, mult, stats,
            )
            det = _detection_rate(score_fn, corrupted, threshold)
            results.append({
                "test": "type_isolation",
                "mode": mode,
                "severity_multiplier": mult,
                "features_injected": type_name,
                "detection_rate": det,
            })
        dets = [r["detection_rate"] for r in results if r["features_injected"] == type_name]
        min_det_idx = next((i for i, d in enumerate(dets) if d >= 95), len(dets) - 1)
        print(f"      95% detection at: {SEVERITY_MULTIPLIERS[min_det_idx]}x" if dets[min_det_idx] >= 95
              else f"      Max detection: {max(dets):.1f}% at {SEVERITY_MULTIPLIERS[-1]}x")

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────


def plot_graduated_curve(results: list[dict], mode: str, save_path: Path):
    """Plot detection rate vs severity for multi-feature test."""
    multi = [r for r in results if r["test"] == "graduated_multi"]
    if not multi:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    mults = [r["severity_multiplier"] for r in multi]
    dets = [r["detection_rate"] for r in multi]

    ax.plot(mults, dets, "b-o", linewidth=2, markersize=8, label="All features")
    ax.axhline(95, color="r", linestyle="--", alpha=0.7, label="95% target")
    ax.axhline(90, color="orange", linestyle="--", alpha=0.5, label="90% target")
    ax.set_xlabel("Severity Multiplier (× normal baseline)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(f"Model 3{'a' if mode == 'charging' else 'b'}: "
                 f"Detection Rate vs Anomaly Severity ({mode.title()})", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_single_feature_curves(results: list[dict], mode: str, save_path: Path):
    """Plot per-feature detection sensitivity curves."""
    single = [r for r in results if r["test"] == "single_feature"]
    if not single:
        return

    features_tested = sorted(set(r["features_injected"] for r in single))

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(features_tested)))

    for feat, color in zip(features_tested, colors):
        feat_results = [r for r in single if r["features_injected"] == feat]
        mults = [r["severity_multiplier"] for r in feat_results]
        dets = [r["detection_rate"] for r in feat_results]
        label = feat.replace("_", " ").replace("cell ", "").replace("spread ", "spr ")
        ax.plot(mults, dets, "-o", color=color, linewidth=1.5, markersize=5, label=label)

    ax.axhline(95, color="r", linestyle="--", alpha=0.7, label="95% target")
    ax.set_xlabel("Severity Multiplier (× normal baseline)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(f"Model 3{'a' if mode == 'charging' else 'b'}: "
                 f"Per-Feature Detection Sensitivity ({mode.title()})", fontsize=13)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison(results: list[dict], mode: str, save_path: Path):
    """Plot one-sided + type-isolation comparison."""
    fig, ax = plt.subplots(figsize=(12, 7))

    test_types = [
        ("graduated_multi", "All features", "black", "-o", 3),
        ("one_sided", "port_only", "blue", "-s", 1.5),
        ("one_sided", "stbd_only", "cyan", "-s", 1.5),
        ("type_isolation", "voltage_only", "red", "-^", 1.5),
        ("type_isolation", "thermal_only", "orange", "-^", 1.5),
    ]

    for test_name, feat_filter, color, style, lw in test_types:
        subset = [r for r in results
                  if r["test"] == test_name and r["features_injected"] == feat_filter]
        if not subset:
            # For graduated_multi, features_injected is "all_voltage+thermal"
            if test_name == "graduated_multi":
                subset = [r for r in results if r["test"] == test_name]
            if not subset:
                continue
        mults = [r["severity_multiplier"] for r in subset]
        dets = [r["detection_rate"] for r in subset]
        label = feat_filter.replace("_", " ").title()
        ax.plot(mults, dets, style, color=color, linewidth=lw, markersize=6, label=label)

    ax.axhline(95, color="r", linestyle="--", alpha=0.5, label="95% target")
    ax.set_xlabel("Severity Multiplier (× normal baseline)", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(f"Model 3{'a' if mode == 'charging' else 'b'}: "
                 f"Detection Comparison ({mode.title()})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────


def evaluate_mode(mode: str) -> list[dict]:
    """Run full sensitivity evaluation for one battery model."""
    model_label = f"3{'a' if mode == 'charging' else 'b'}"
    print(f"\n{'=' * 60}")
    print(f"  MODEL {model_label}: Battery Sensitivity Evaluation ({mode.title()})")
    print(f"{'=' * 60}")

    models, thresholds, stats, test_data, features, score_fn = _load_models_and_data(mode)
    p99 = thresholds["p99"]

    print(f"  Test data: {len(test_data)} rows")
    print(f"  p99 threshold: {p99:.4f}")
    print(f"  Features: {len(features)}")

    # Normal baseline stats
    print(f"\n  Normal baselines (mean ± std):")
    for feat in VOLTAGE_FEATURES + THERMAL_FEATURES:
        if feat in stats["feature_means"]:
            m = stats["feature_means"][feat]
            s = stats["feature_stds"][feat]
            print(f"    {feat:30s}  {m:.4f} ± {s:.4f}")

    all_results = []
    all_results.extend(test_graduated_severity(score_fn, test_data, features, stats, p99, mode))
    all_results.extend(test_single_feature(score_fn, test_data, features, stats, p99, mode))
    all_results.extend(test_one_sided(score_fn, test_data, features, stats, p99, mode))
    all_results.extend(test_type_isolation(score_fn, test_data, features, stats, p99, mode))

    # Plots
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plot_graduated_curve(all_results, mode,
                         REPORT_DIR / f"battery_{mode}_graduated_severity.png")
    plot_single_feature_curves(all_results, mode,
                               REPORT_DIR / f"battery_{mode}_single_feature_sensitivity.png")
    plot_comparison(all_results, mode,
                    REPORT_DIR / f"battery_{mode}_detection_comparison.png")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Battery anomaly sensitivity evaluation")
    parser.add_argument(
        "--model",
        choices=["charging", "operational", "all"],
        default="all",
    )
    args = parser.parse_args()

    start = time.time()
    all_results = []

    if args.model in ("charging", "all"):
        all_results.extend(evaluate_mode("charging"))

    if args.model in ("operational", "all"):
        all_results.extend(evaluate_mode("operational"))

    # Save CSV
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "battery_sensitivity_results.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n  Results CSV: {csv_path}")

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Sensitivity evaluation complete ({elapsed:.1f}s)")
    print(f"  Results: {csv_path}")
    print(f"  Plots:   {REPORT_DIR}/battery_*_*.png")
    print(f"{'=' * 60}")

    # Print key findings
    for mode in ("charging", "operational"):
        mode_results = [r for r in all_results if r["mode"] == mode]
        if not mode_results:
            continue
        multi = [r for r in mode_results if r["test"] == "graduated_multi"]
        if multi:
            min95 = next(
                (r["severity_multiplier"] for r in multi if r["detection_rate"] >= 95),
                None,
            )
            label = f"3{'a' if mode == 'charging' else 'b'}"
            if min95:
                print(f"\n  Model {label}: 95% detection threshold = {min95}x normal")
            else:
                max_det = max(r["detection_rate"] for r in multi)
                print(f"\n  Model {label}: Max detection = {max_det:.1f}% "
                      f"(95% not reached at {SEVERITY_MULTIPLIERS[-1]}x)")


if __name__ == "__main__":
    main()
