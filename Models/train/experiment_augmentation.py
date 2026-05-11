"""Experiment: Data augmentation for Models 1A and 1B.

Model 1B (realtime): Currently no augmentation. Test Gaussian noise + C-Mixup.
Model 1A (trip): Currently 5x augmentation. Test 7x and 10x.

Uses saved hyperparameters (quick retrain style) for fair comparison.

Usage:
    python -m train.experiment_augmentation
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    ENSEMBLE_SEEDS,
    FEATURE_STORE_DIR,
    N_ENSEMBLE_SEEDS,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRIP_FEATURES,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    get_monotonic_constraint_tuple,
)
from data.augment import augment_trip_data, gaussian_noise, cmixup
from evaluate.evaluate_soc import compute_metrics
from train.retrain import (
    FIXED_TEST_BOUNDARY,
    compute_rolling_split_dates,
    get_fixed_test_set,
    load_production_metadata,
    rolling_temporal_split,
)
from train.train_v2 import df_to_dmatrix, ensemble_predict


# ── Helpers ──────────────────────────────────────────────────────────────


def train_ensemble(
    train_X: xgb.DMatrix,
    val_X: xgb.DMatrix,
    params: dict,
    n_estimators: int,
    seeds: list[int],
    verbose: bool = False,
) -> list[xgb.Booster]:
    """Train a multi-seed ensemble with early stopping."""
    models = []
    for seed in seeds:
        p = {**params, "seed": seed}
        model = xgb.train(
            p, train_X,
            num_boost_round=n_estimators,
            evals=[(val_X, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        if verbose:
            print(f"      Seed {seed}: {model.num_boosted_rounds()} trees")
        models.append(model)
    return models


def evaluate_realtime_by_progress(
    models: list[xgb.Booster],
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> dict:
    """Evaluate Model 1B by trip progress bins."""
    dmat, y_true = df_to_dmatrix(df, features, target)
    y_pred = ensemble_predict(models, dmat)

    overall = compute_metrics(y_true, y_pred)

    progress = df["trip_progress_fraction"].to_numpy() * 100  # 0-1 -> 0-100
    bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
    per_bin = {}
    for lo, hi in bins:
        mask = (progress >= lo) & (progress < (hi + 1 if hi == 100 else hi))
        if mask.sum() > 0:
            bin_mae = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
            per_bin[f"{lo}-{hi}%"] = {"MAE": bin_mae, "n": int(mask.sum())}

    return {"overall": overall, "per_progress": per_bin}


def augment_realtime(
    train_df: pl.DataFrame,
    features: list[str],
    target: str,
    gaussian_copies: int,
    noise_std: float,
    cmixup_alpha: float,
    cmixup_sigma: float,
    cmixup_n_synthetic: float,
) -> pl.DataFrame:
    """Apply Gaussian noise + C-Mixup to realtime data."""
    parts = [train_df]

    if gaussian_copies > 0:
        noise_df = gaussian_noise(
            train_df, features, target,
            n_copies=gaussian_copies, noise_std=noise_std,
        )
        parts.append(noise_df)

    if cmixup_n_synthetic > 0:
        mix_df = cmixup(
            train_df, features, target,
            alpha=cmixup_alpha, sigma=cmixup_sigma, n_synthetic=cmixup_n_synthetic,
        )
        parts.append(mix_df)

    combined = pl.concat(parts, how="diagonal_relaxed").fill_null(strategy="forward")
    return combined


# ── Model 1B Experiments ─────────────────────────────────────────────────


def run_1b_experiments() -> list[dict]:
    """Run augmentation experiments for Model 1B."""
    print("\n" + "=" * 70)
    print("  MODEL 1B: REALTIME SOC — AUGMENTATION EXPERIMENTS")
    print("=" * 70)

    # Load data
    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train_df, val_df, _ = rolling_temporal_split(df, train_end, val_end)
    fixed_test = get_fixed_test_set(df)

    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows | Fixed Test: {len(fixed_test):,} rows")

    # Load saved params
    meta = load_production_metadata("1b")
    saved_params = meta["best_params"]
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": saved_params["max_depth"],
        "learning_rate": saved_params["learning_rate"],
        "subsample": saved_params["subsample"],
        "colsample_bytree": saved_params["colsample_bytree"],
        "colsample_bylevel": saved_params["colsample_bylevel"],
        "min_child_weight": saved_params["min_child_weight"],
        "reg_alpha": saved_params["reg_alpha"],
        "reg_lambda": saved_params["reg_lambda"],
        "gamma": saved_params["gamma"],
    }
    n_est = saved_params["n_estimators"]

    experiments = [
        {
            "name": "Baseline (no augmentation)",
            "gaussian_copies": 0, "noise_std": 0,
            "cmixup_alpha": 0, "cmixup_sigma": 0, "cmixup_n_synthetic": 0,
        },
        {
            "name": "Gaussian 1x + C-Mixup 1x (3x total)",
            "gaussian_copies": 1, "noise_std": 0.02,
            "cmixup_alpha": 0.2, "cmixup_sigma": 2.0, "cmixup_n_synthetic": 1.0,
        },
        {
            "name": "Gaussian 2x + C-Mixup 1x (4x total)",
            "gaussian_copies": 2, "noise_std": 0.02,
            "cmixup_alpha": 0.2, "cmixup_sigma": 2.0, "cmixup_n_synthetic": 1.0,
        },
        {
            "name": "Gaussian 1x (noise_std=0.01, conservative)",
            "gaussian_copies": 1, "noise_std": 0.01,
            "cmixup_alpha": 0, "cmixup_sigma": 0, "cmixup_n_synthetic": 0,
        },
        {
            "name": "C-Mixup 1x only",
            "gaussian_copies": 0, "noise_std": 0,
            "cmixup_alpha": 0.2, "cmixup_sigma": 2.0, "cmixup_n_synthetic": 1.0,
        },
    ]

    results = []
    for exp in experiments:
        print(f"\n  --- {exp['name']} ---")
        t0 = time.time()

        if exp["gaussian_copies"] == 0 and exp["cmixup_n_synthetic"] == 0:
            train_data = train_df
        else:
            train_data = augment_realtime(
                train_df, REALTIME_FEATURES, REALTIME_TARGET,
                exp["gaussian_copies"], exp["noise_std"],
                exp["cmixup_alpha"], exp["cmixup_sigma"], exp["cmixup_n_synthetic"],
            )

        multiplier = len(train_data) / len(train_df)
        print(f"    Train rows: {len(train_df):,} -> {len(train_data):,} ({multiplier:.1f}x)")

        dtrain, _ = df_to_dmatrix(train_data, REALTIME_FEATURES, REALTIME_TARGET)
        dval, _ = df_to_dmatrix(val_df, REALTIME_FEATURES, REALTIME_TARGET)

        models = train_ensemble(dtrain, dval, params, n_est, ENSEMBLE_SEEDS, verbose=True)

        # Evaluate on fixed test
        metrics = evaluate_realtime_by_progress(
            models, fixed_test, REALTIME_FEATURES, REALTIME_TARGET,
        )
        elapsed = time.time() - t0

        print(f"    Fixed Test MAE: {metrics['overall']['MAE']:.4f}%  R2: {metrics['overall']['R2']:.4f}")
        for bin_name, bin_m in metrics["per_progress"].items():
            print(f"      {bin_name}: MAE={bin_m['MAE']:.4f}%  (n={bin_m['n']})")
        print(f"    Time: {elapsed:.1f}s")

        results.append({
            "name": exp["name"],
            "train_rows": len(train_data),
            "multiplier": round(multiplier, 1),
            "fixed_test_mae": metrics["overall"]["MAE"],
            "fixed_test_r2": metrics["overall"]["R2"],
            "per_progress": metrics["per_progress"],
            "elapsed_s": round(elapsed, 1),
        })

    return results


# ── Model 1A Experiments ─────────────────────────────────────────────────


def run_1a_experiments() -> list[dict]:
    """Run augmentation experiments for Model 1A."""
    print("\n" + "=" * 70)
    print("  MODEL 1A: TRIP SOC — AUGMENTATION EXPERIMENTS")
    print("=" * 70)

    # Load data
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train_df, val_df, _ = rolling_temporal_split(df, train_end, val_end)
    fixed_test = get_fixed_test_set(df)

    print(f"  Train: {len(train_df)} rows | Val: {len(val_df)} rows | Fixed Test: {len(fixed_test)} rows")

    # Load saved params
    meta = load_production_metadata("1a")
    saved_params = meta["best_params"]
    use_huber = meta.get("use_huber_loss", False)
    huber_slope = meta.get("huber_slope")
    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    params = {
        "objective": "reg:pseudohubererror" if use_huber else "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": saved_params["max_depth"],
        "learning_rate": saved_params["learning_rate"],
        "subsample": saved_params["subsample"],
        "colsample_bytree": saved_params["colsample_bytree"],
        "colsample_bylevel": saved_params["colsample_bylevel"],
        "min_child_weight": saved_params["min_child_weight"],
        "reg_alpha": saved_params["reg_alpha"],
        "reg_lambda": saved_params["reg_lambda"],
        "gamma": saved_params["gamma"],
        "monotone_constraints": mono_tuple,
    }
    if use_huber and huber_slope is not None:
        params["huber_slope"] = huber_slope
    n_est = saved_params["n_estimators"]

    experiments = [
        {
            "name": "Baseline (no augmentation)",
            "gaussian_copies": 0, "cmixup_n_synthetic": 0,
        },
        {
            "name": "Current (Gaussian 3x + C-Mixup 1x = 5x)",
            "gaussian_copies": 3, "cmixup_n_synthetic": 1.0,
        },
        {
            "name": "Gaussian 5x + C-Mixup 1x (7x)",
            "gaussian_copies": 5, "cmixup_n_synthetic": 1.0,
        },
        {
            "name": "Gaussian 7x + C-Mixup 2x (10x)",
            "gaussian_copies": 7, "cmixup_n_synthetic": 2.0,
        },
        {
            "name": "Gaussian 3x + C-Mixup 2x (6x)",
            "gaussian_copies": 3, "cmixup_n_synthetic": 2.0,
        },
    ]

    results = []
    for exp in experiments:
        print(f"\n  --- {exp['name']} ---")
        t0 = time.time()

        if exp["gaussian_copies"] == 0 and exp["cmixup_n_synthetic"] == 0:
            train_data = train_df
        else:
            parts = [train_df]
            if exp["gaussian_copies"] > 0:
                parts.append(gaussian_noise(
                    train_df, TRIP_FEATURES, TRIP_TARGET,
                    n_copies=exp["gaussian_copies"], noise_std=0.02,
                ))
            if exp["cmixup_n_synthetic"] > 0:
                parts.append(cmixup(
                    train_df, TRIP_FEATURES, TRIP_TARGET,
                    alpha=0.2, sigma=2.0, n_synthetic=exp["cmixup_n_synthetic"],
                ))
            train_data = pl.concat(parts, how="diagonal_relaxed").fill_null(strategy="forward")

        multiplier = len(train_data) / len(train_df)
        print(f"    Train rows: {len(train_df)} -> {len(train_data)} ({multiplier:.1f}x)")

        dtrain, _ = df_to_dmatrix(train_data, TRIP_FEATURES, TRIP_TARGET)
        dval, _ = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)

        models = train_ensemble(dtrain, dval, params, n_est, ENSEMBLE_SEEDS, verbose=True)

        # Evaluate
        dtest, y_test = df_to_dmatrix(fixed_test, TRIP_FEATURES, TRIP_TARGET)
        y_pred = ensemble_predict(models, dtest)
        metrics = compute_metrics(y_test, y_pred)
        elapsed = time.time() - t0

        print(f"    Fixed Test MAE: {metrics['MAE']:.4f}%  R2: {metrics['R2']:.4f}")
        print(f"    Time: {elapsed:.1f}s")

        results.append({
            "name": exp["name"],
            "train_rows": len(train_data),
            "multiplier": round(multiplier, 1),
            "fixed_test_mae": metrics["MAE"],
            "fixed_test_r2": metrics["R2"],
            "fixed_test_rmse": metrics["RMSE"],
            "elapsed_s": round(elapsed, 1),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  AUGMENTATION EXPERIMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    results_1b = run_1b_experiments()
    results_1a = run_1a_experiments()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — MODEL 1B (REALTIME)")
    print("=" * 70)

    baseline_1b = results_1b[0]["fixed_test_mae"]
    print(f"\n  {'Experiment':<45} {'Rows':>8} {'MAE':>8} {'Delta':>8} {'0-25%':>8} {'75-100%':>8}")
    print(f"  {'-'*90}")
    for r in results_1b:
        delta = (r["fixed_test_mae"] - baseline_1b) / baseline_1b * 100
        early = r["per_progress"].get("0-25%", {}).get("MAE", float("nan"))
        late = r["per_progress"].get("75-100%", {}).get("MAE", float("nan"))
        marker = " **" if r["fixed_test_mae"] < baseline_1b * 0.99 else ""
        print(f"  {r['name']:<45} {r['train_rows']:>8,} {r['fixed_test_mae']:>7.4f}% {delta:>+7.1f}% {early:>7.4f}% {late:>7.4f}%{marker}")

    print("\n" + "=" * 70)
    print("  SUMMARY — MODEL 1A (TRIP)")
    print("=" * 70)

    baseline_1a = results_1a[0]["fixed_test_mae"]
    print(f"\n  {'Experiment':<45} {'Rows':>8} {'MAE':>8} {'Delta':>8} {'R2':>8}")
    print(f"  {'-'*80}")
    for r in results_1a:
        delta = (r["fixed_test_mae"] - baseline_1a) / baseline_1a * 100
        marker = " **" if r["fixed_test_mae"] < baseline_1a * 0.99 else ""
        print(f"  {r['name']:<45} {r['train_rows']:>8,} {r['fixed_test_mae']:>7.4f}% {delta:>+7.1f}% {r['fixed_test_r2']:>7.4f}{marker}")

    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_1b": results_1b,
        "model_1a": results_1a,
    }
    report_path = ARTIFACTS_DIR / "experiment_augmentation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: {report_path}")


if __name__ == "__main__":
    main()
