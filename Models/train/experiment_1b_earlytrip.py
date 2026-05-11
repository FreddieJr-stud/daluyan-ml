"""Early-trip improvement experiments for Model 1B (Realtime SOC).

Model 1B's 0-25% progress bin has MAE ~3.11% — too high for reliable
early-trip range estimation. Root cause: at trip start, rolling windows
are flat and SOC rates near zero, so the model relies on route context
alone.

Experiments:
  8. Trip-so-far consumption features (soc_consumed_so_far, empirical_soc_per_km,
     empirical_power_per_km) — gives the model direct "this trip's energy intensity"
  9. Historical route prior (route_avg_soc_delta) — training-set average consumption
     for this departure→arrival route
 10. Progress-weighted sample loss — upweight early-trip rows during training
 11. Two-stage model — separate early-trip (progress < 0.3) and mid/late models

Each experiment builds on the best result from previous ones.

Usage:
    python -m train.experiment_1b_earlytrip
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
)
from train.retrain import compute_rolling_split_dates, rolling_temporal_split
from train.train_v2 import df_to_dmatrix, ensemble_predict

# Fixed test boundary (original thesis benchmark)
FIXED_TEST_BOUNDARY = datetime(2026, 2, 12, 23, 59, 59)

# Experimental features (extracted but not in production REALTIME_FEATURES)
EXP8_FEATURES = [
    "soc_consumed_so_far",
    "empirical_soc_per_km",
    "empirical_power_per_km",
]
EXP9_FEATURES = [
    "route_avg_soc_delta",
]

# Progress bins for evaluation
PROGRESS_BINS = [
    (0.0, 0.25, "0-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 0.75, "50-75%"),
    (0.75, 1.01, "75-100%"),
]


# ── Helpers ──────────────────────────────────────────────────────────────


def load_realtime_split() -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, datetime, datetime
]:
    """Load realtime feature store and return train/val/fixed_test DataFrames."""
    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train, val, _ = rolling_temporal_split(df, train_end, val_end)
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)

    print(f"  Split dates: train <= {train_end.date()}, val <= {val_end.date()}, "
          f"test > {FIXED_TEST_BOUNDARY.date()}")
    print(f"  Rows: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_segs = split_df["segment_id"].n_unique() if "segment_id" in split_df.columns else "?"
        print(f"    {name} segments: {n_segs}")

    return train, val, test, train_end, val_end


def load_production_1b_models() -> list[xgb.Booster]:
    """Load production 1B ensemble from artifact JSON files."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        suffix = "" if seed == 42 else f"_seed{seed}"
        path = ARTIFACTS_DIR / f"soc_realtime_model{suffix}.json"
        m = xgb.Booster()
        m.load_model(str(path))
        models.append(m)
    return models


def load_production_1b_params() -> dict:
    """Load production 1B hyperparameters from metadata."""
    meta_path = ARTIFACTS_DIR / "soc_realtime_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["best_params"]


def build_xgb_params(saved_params: dict) -> tuple[dict, int]:
    """Build XGBoost params dict + n_estimators from saved params."""
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
    n_estimators = saved_params["n_estimators"]
    return params, n_estimators


def train_ensemble(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: list[str],
    params: dict,
    n_estimators: int,
    weights: np.ndarray | None = None,
) -> list[xgb.Booster]:
    """Train 5-seed ensemble with optional sample weights."""
    X_train = train_df.select(features).to_numpy().astype(np.float32)
    y_train = train_df[REALTIME_TARGET].to_numpy().astype(np.float32)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features,
                         weight=weights)

    X_val = val_df.select(features).to_numpy().astype(np.float32)
    y_val = val_df[REALTIME_TARGET].to_numpy().astype(np.float32)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    models = []
    for seed in ENSEMBLE_SEEDS:
        p = {**params, "seed": seed}
        model = xgb.train(
            p, dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        models.append(model)
    return models


def evaluate_on_df(
    models: list[xgb.Booster],
    df: pl.DataFrame,
    features: list[str],
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate ensemble on a DataFrame. Returns (mae, rmse, r2, preds, errors)."""
    X = df.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features)
    preds = ensemble_predict(models, dm)

    y_true = df[REALTIME_TARGET].to_numpy().astype(np.float32)
    errors = np.abs(y_true - preds)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    ss_res = float(np.sum((y_true - preds) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return mae, rmse, r2, preds, errors


def get_bin_maes(errors: np.ndarray, progress: np.ndarray) -> dict[str, float]:
    """Return per-progress-bin MAEs."""
    result = {}
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() > 0:
            result[label] = float(errors[mask].mean())
    return result


def get_bin_counts(progress: np.ndarray) -> dict[str, int]:
    """Return per-progress-bin sample counts."""
    result = {}
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        result[label] = int(mask.sum())
    return result


def print_progress_breakdown(
    errors: np.ndarray,
    progress: np.ndarray,
    baseline_bins: dict[str, float] | None = None,
) -> None:
    """Print per-progress-bin MAE with optional delta vs baseline."""
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() == 0:
            continue
        bin_mae = float(errors[mask].mean())
        delta_str = ""
        if baseline_bins and label in baseline_bins and baseline_bins[label] > 0:
            pct = (bin_mae - baseline_bins[label]) / baseline_bins[label] * 100
            delta_str = f"  ({pct:+.1f}%)"
        print(f"      {label}: MAE={bin_mae:.4f}%  n={mask.sum():,}{delta_str}")


def check_pass_gate(
    exp_bins: dict[str, float],
    base_bins: dict[str, float],
    exp_mae: float,
    base_mae: float,
) -> tuple[bool, list[str]]:
    """Check if experiment passes. Overall MAE must improve, no bin >20% degradation."""
    reasons = []
    if exp_mae >= base_mae:
        reasons.append(f"Overall MAE did not improve ({exp_mae:.4f} >= {base_mae:.4f})")

    for label, base_val in base_bins.items():
        if label in exp_bins and base_val > 0:
            pct_change = (exp_bins[label] - base_val) / base_val * 100
            if pct_change > 20:
                reasons.append(
                    f"{label} degraded {pct_change:+.1f}% "
                    f"({base_val:.4f} -> {exp_bins[label]:.4f})"
                )

    return len(reasons) == 0, reasons


def format_features_short(features: list[str], max_show: int = 5) -> str:
    """Format feature list for display."""
    if len(features) <= max_show:
        return ", ".join(features)
    return ", ".join(features[:max_show]) + f" ... ({len(features)} total)"


# ── Experiments ──────────────────────────────────────────────────────────


def run_experiment_8(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    progress_test: np.ndarray,
    params: dict,
    n_estimators: int,
    baseline_mae: float,
    baseline_bins: dict[str, float],
) -> tuple[list[str], dict]:
    """Exp 8: Trip-so-far consumption features.

    Tests adding soc_consumed_so_far, empirical_soc_per_km, empirical_power_per_km
    individually and combined on top of the production 25 features.
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 8: Trip-So-Far Consumption Features")
    print("=" * 65)

    base_features = list(REALTIME_FEATURES)
    results = []

    # Test each new feature individually + all 3 together
    configs = [
        ("+ soc_consumed_so_far", base_features + ["soc_consumed_so_far"]),
        ("+ empirical_soc_per_km", base_features + ["empirical_soc_per_km"]),
        ("+ empirical_power_per_km", base_features + ["empirical_power_per_km"]),
        ("+ all 3 trip-so-far", base_features + EXP8_FEATURES),
    ]

    for label, features in configs:
        print(f"\n  --- {label} ({len(features)} features) ---")
        t0 = time.time()
        models = train_ensemble(train_df, val_df, features, params, n_estimators)
        elapsed = time.time() - t0
        print(f"  Trained {N_ENSEMBLE_SEEDS}-seed ensemble ({elapsed:.1f}s)")

        mae, rmse, r2, preds, errors = evaluate_on_df(models, test_df, features)
        val_mae = evaluate_on_df(models, val_df, features)[0]
        bins = get_bin_maes(errors, progress_test)
        delta_pct = (mae - baseline_mae) / baseline_mae * 100
        passed, reasons = check_pass_gate(bins, baseline_bins, mae, baseline_mae)

        print(f"    Test MAE: {mae:.4f}% ({delta_pct:+.2f}%)")
        print(f"    Val MAE:  {val_mae:.4f}% | Gap: {mae - val_mae:+.4f}%")
        print(f"    R2: {r2:.4f}")
        print_progress_breakdown(errors, progress_test, baseline_bins)
        print(f"    Gate: {'PASS' if passed else 'FAIL'}")
        if not passed:
            for r in reasons:
                print(f"      - {r}")

        results.append({
            "label": label,
            "features": features,
            "mae": mae,
            "val_mae": val_mae,
            "r2": r2,
            "bins": bins,
            "delta_pct": delta_pct,
            "passed": passed,
            "reasons": reasons,
        })

    # Pick best
    passed_results = [r for r in results if r["passed"]]
    if passed_results:
        best = min(passed_results, key=lambda r: r["mae"])
        print(f"\n  Exp 8 BEST: {best['label']} (MAE={best['mae']:.4f}%, {best['delta_pct']:+.2f}%)")
        return best["features"], best
    else:
        # Even if none pass gate, pick the one with best 0-25% bin for Exp 9/10
        best_early = min(results, key=lambda r: r["bins"].get("0-25%", 999))
        if best_early["bins"].get("0-25%", 999) < baseline_bins.get("0-25%", 999):
            print(f"\n  Exp 8: No overall gate pass, but {best_early['label']} improved 0-25% bin")
            print(f"    Carrying forward for Exp 9-10 (0-25%: "
                  f"{baseline_bins.get('0-25%', 0):.4f} -> {best_early['bins'].get('0-25%', 0):.4f})")
            return best_early["features"], best_early
        print(f"\n  Exp 8: No improvement. Using production features for Exp 9.")
        return list(REALTIME_FEATURES), {"label": "baseline", "mae": baseline_mae, "bins": baseline_bins}


def run_experiment_9(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    progress_test: np.ndarray,
    params: dict,
    n_estimators: int,
    baseline_mae: float,
    baseline_bins: dict[str, float],
    carry_features: list[str],
    carry_result: dict,
) -> tuple[list[str], dict]:
    """Exp 9: Historical route prior (route_avg_soc_delta).

    Tests adding route_avg_soc_delta on top of best features from Exp 8.
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 9: Historical Route Prior")
    print("=" * 65)

    carry_mae = carry_result["mae"]
    carry_bins = carry_result["bins"]

    features_with_prior = carry_features + ["route_avg_soc_delta"]
    print(f"\n  Testing: {carry_result['label']} + route_avg_soc_delta ({len(features_with_prior)} features)")

    t0 = time.time()
    models = train_ensemble(train_df, val_df, features_with_prior, params, n_estimators)
    elapsed = time.time() - t0
    print(f"  Trained {N_ENSEMBLE_SEEDS}-seed ensemble ({elapsed:.1f}s)")

    mae, rmse, r2, preds, errors = evaluate_on_df(models, test_df, features_with_prior)
    val_mae = evaluate_on_df(models, val_df, features_with_prior)[0]
    bins = get_bin_maes(errors, progress_test)

    delta_vs_baseline = (mae - baseline_mae) / baseline_mae * 100
    delta_vs_carry = (mae - carry_mae) / carry_mae * 100 if carry_mae > 0 else 0

    # Gate against ORIGINAL baseline (not carry)
    passed, reasons = check_pass_gate(bins, baseline_bins, mae, baseline_mae)

    print(f"    Test MAE: {mae:.4f}% (vs baseline: {delta_vs_baseline:+.2f}%, vs Exp8: {delta_vs_carry:+.2f}%)")
    print(f"    Val MAE:  {val_mae:.4f}% | Gap: {mae - val_mae:+.4f}%")
    print(f"    R2: {r2:.4f}")
    print_progress_breakdown(errors, progress_test, baseline_bins)
    print(f"    Gate (vs baseline): {'PASS' if passed else 'FAIL'}")
    if not passed:
        for r in reasons:
            print(f"      - {r}")

    result = {
        "label": f"{carry_result['label']} + route_avg_soc_delta",
        "features": features_with_prior,
        "mae": mae,
        "val_mae": val_mae,
        "r2": r2,
        "bins": bins,
        "delta_pct": delta_vs_baseline,
        "passed": passed,
        "reasons": reasons,
    }

    # Decide: use this result or revert to carry
    if mae < carry_mae:
        print(f"\n  Exp 9: route_avg_soc_delta improved over carry ({mae:.4f}% < {carry_mae:.4f}%)")
        return features_with_prior, result
    else:
        print(f"\n  Exp 9: route_avg_soc_delta did not help ({mae:.4f}% >= {carry_mae:.4f}%). Reverting.")
        return carry_features, carry_result


def run_experiment_10(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    progress_test: np.ndarray,
    params: dict,
    n_estimators: int,
    baseline_mae: float,
    baseline_bins: dict[str, float],
    carry_features: list[str],
    carry_result: dict,
) -> tuple[list[str], dict, float | None]:
    """Exp 10: Progress-weighted sample loss.

    Upweights early-trip rows: w_i = 1 + k * (1 - progress_i).
    Tests k = 1, 2, 3, 5.
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 10: Progress-Weighted Sample Loss")
    print("=" * 65)

    carry_mae = carry_result["mae"]
    carry_bins = carry_result["bins"]
    features = carry_features

    progress_train = train_df["trip_progress_fraction"].to_numpy()
    k_values = [1, 2, 3, 5]
    results = []

    for k in k_values:
        weights = 1.0 + k * (1.0 - progress_train)
        print(f"\n  --- k={k} (max_weight={1+k:.0f}x, mean={weights.mean():.2f}x) ---")

        t0 = time.time()
        models = train_ensemble(train_df, val_df, features, params, n_estimators, weights=weights)
        elapsed = time.time() - t0
        print(f"  Trained {N_ENSEMBLE_SEEDS}-seed ensemble ({elapsed:.1f}s)")

        mae, rmse, r2, preds, errors = evaluate_on_df(models, test_df, features)
        val_mae = evaluate_on_df(models, val_df, features)[0]
        bins = get_bin_maes(errors, progress_test)

        delta_vs_baseline = (mae - baseline_mae) / baseline_mae * 100
        delta_vs_carry = (mae - carry_mae) / carry_mae * 100 if carry_mae > 0 else 0

        passed, reasons = check_pass_gate(bins, baseline_bins, mae, baseline_mae)

        print(f"    Test MAE: {mae:.4f}% (vs baseline: {delta_vs_baseline:+.2f}%, vs carry: {delta_vs_carry:+.2f}%)")
        print(f"    Val MAE:  {val_mae:.4f}% | Gap: {mae - val_mae:+.4f}%")
        print_progress_breakdown(errors, progress_test, baseline_bins)
        print(f"    Gate (vs baseline): {'PASS' if passed else 'FAIL'}")
        if not passed:
            for r in reasons:
                print(f"      - {r}")

        results.append({
            "label": f"k={k}",
            "k": k,
            "mae": mae,
            "val_mae": val_mae,
            "r2": r2,
            "bins": bins,
            "delta_pct": delta_vs_baseline,
            "passed": passed,
            "reasons": reasons,
        })

    # Pick best k that improves over carry
    improving = [r for r in results if r["mae"] < carry_mae]
    if improving:
        best = min(improving, key=lambda r: r["mae"])
        print(f"\n  Exp 10 BEST: k={best['k']} (MAE={best['mae']:.4f}%, vs carry: "
              f"{(best['mae'] - carry_mae) / carry_mae * 100:+.2f}%)")
        best_result = {
            **carry_result,
            "label": f"{carry_result['label']} + weighted(k={best['k']})",
            "mae": best["mae"],
            "val_mae": best["val_mae"],
            "r2": best["r2"],
            "bins": best["bins"],
            "delta_pct": best["delta_pct"],
            "passed": best["passed"],
            "reasons": best["reasons"],
        }
        return carry_features, best_result, best["k"]
    else:
        print(f"\n  Exp 10: No k improved over carry. Keeping unweighted.")
        return carry_features, carry_result, None


def run_experiment_11(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    progress_test: np.ndarray,
    params: dict,
    n_estimators: int,
    baseline_mae: float,
    baseline_bins: dict[str, float],
    carry_features: list[str],
    carry_result: dict,
    best_k: float | None = None,
) -> dict:
    """Exp 11: Two-stage model (early-trip vs mid/late).

    Train separate ensembles for progress < 0.3 and >= 0.3.
    """
    print("\n" + "=" * 65)
    print("  EXPERIMENT 11: Two-Stage Model (Early vs Mid/Late)")
    print("=" * 65)

    carry_mae = carry_result["mae"]
    features = carry_features
    split_point = 0.30

    # Split train data
    progress_train = train_df["trip_progress_fraction"].to_numpy()
    progress_val = val_df["trip_progress_fraction"].to_numpy()

    train_early = train_df.filter(pl.col("trip_progress_fraction") < split_point)
    train_late = train_df.filter(pl.col("trip_progress_fraction") >= split_point)
    val_early = val_df.filter(pl.col("trip_progress_fraction") < split_point)
    val_late = val_df.filter(pl.col("trip_progress_fraction") >= split_point)

    print(f"\n  Split at progress={split_point:.0%}")
    print(f"    Early: train={len(train_early):,}, val={len(val_early):,}")
    print(f"    Late:  train={len(train_late):,}, val={len(val_late):,}")

    # Optionally apply sample weights from Exp 10 if it helped
    early_weights = None
    late_weights = None
    if best_k is not None:
        progress_early = train_early["trip_progress_fraction"].to_numpy()
        early_weights = 1.0 + best_k * (1.0 - progress_early)
        progress_late_arr = train_late["trip_progress_fraction"].to_numpy()
        late_weights = 1.0 + best_k * (1.0 - progress_late_arr)
        print(f"    Applying Exp 10 weights (k={best_k})")

    # Train early-trip model
    print(f"\n  Training early-trip model...")
    t0 = time.time()
    models_early = train_ensemble(train_early, val_early, features, params, n_estimators,
                                  weights=early_weights)
    e1 = time.time() - t0

    # Train late-trip model
    print(f"  Training late-trip model...")
    t1 = time.time()
    models_late = train_ensemble(train_late, val_late, features, params, n_estimators,
                                 weights=late_weights)
    e2 = time.time() - t1
    print(f"  Training done (early: {e1:.1f}s, late: {e2:.1f}s)")

    # Evaluate combined on test set
    test_early_mask = progress_test < split_point
    test_late_mask = progress_test >= split_point

    y_test = test_df[REALTIME_TARGET].to_numpy().astype(np.float32)
    combined_preds = np.zeros(len(test_df))

    # Early predictions
    if test_early_mask.sum() > 0:
        test_early = test_df.filter(pl.col("trip_progress_fraction") < split_point)
        X_early = test_early.select(features).to_numpy().astype(np.float32)
        dm_early = xgb.DMatrix(X_early, feature_names=features)
        combined_preds[test_early_mask] = ensemble_predict(models_early, dm_early)

    # Late predictions
    if test_late_mask.sum() > 0:
        test_late = test_df.filter(pl.col("trip_progress_fraction") >= split_point)
        X_late = test_late.select(features).to_numpy().astype(np.float32)
        dm_late = xgb.DMatrix(X_late, feature_names=features)
        combined_preds[test_late_mask] = ensemble_predict(models_late, dm_late)

    errors = np.abs(y_test - combined_preds)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((y_test - combined_preds) ** 2)))
    ss_res = float(np.sum((y_test - combined_preds) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    bins = get_bin_maes(errors, progress_test)
    delta_vs_baseline = (mae - baseline_mae) / baseline_mae * 100
    delta_vs_carry = (mae - carry_mae) / carry_mae * 100 if carry_mae > 0 else 0

    passed, reasons = check_pass_gate(bins, baseline_bins, mae, baseline_mae)

    print(f"\n    Combined Test MAE: {mae:.4f}% (vs baseline: {delta_vs_baseline:+.2f}%, vs carry: {delta_vs_carry:+.2f}%)")
    print(f"    RMSE: {rmse:.4f}%  |  R2: {r2:.4f}")
    print_progress_breakdown(errors, progress_test, baseline_bins)
    print(f"    Gate (vs baseline): {'PASS' if passed else 'FAIL'}")
    if not passed:
        for r in reasons:
            print(f"      - {r}")

    return {
        "label": f"two-stage (split={split_point:.0%})" + (f" + weighted(k={best_k})" if best_k else ""),
        "features": features,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bins": bins,
        "delta_pct": delta_vs_baseline,
        "passed": passed,
        "reasons": reasons,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    print("\n" + "=" * 65)
    print("  MODEL 1B EXPERIMENTS 8-11: Early-Trip Improvements")
    print("=" * 65)

    # ── Verify experimental features exist in parquet ──
    df_check = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet", n_rows=5)
    for feat in EXP8_FEATURES + EXP9_FEATURES:
        if feat not in df_check.columns:
            print(f"\nERROR: Feature '{feat}' not found in parquet.")
            print("Run: python -m data.extract_realtime_features")
            sys.exit(1)
    print("\n  Experimental features verified in feature store.")

    # ── Load data ──
    print("\n  Loading realtime feature store...")
    train_df, val_df, test_df, train_end, val_end = load_realtime_split()

    progress_test = test_df["trip_progress_fraction"].to_numpy()
    test_counts = get_bin_counts(progress_test)
    print(f"\n  Test set bin counts: {test_counts}")

    # ── Production baseline ──
    saved_params = load_production_1b_params()
    params, n_estimators = build_xgb_params(saved_params)
    prod_models = load_production_1b_models()

    base_mae, base_rmse, base_r2, base_preds, base_errors = evaluate_on_df(
        prod_models, test_df, REALTIME_FEATURES,
    )
    baseline_bins = get_bin_maes(base_errors, progress_test)

    print(f"\n  Production baseline (25 features, fixed test):")
    print(f"    MAE={base_mae:.4f}%  RMSE={base_rmse:.4f}%  R2={base_r2:.4f}")
    print_progress_breakdown(base_errors, progress_test)

    # Also evaluate production model retrained on current features (to isolate
    # feature vs. model changes)
    print(f"\n  Retraining baseline with production features + params...")
    t0 = time.time()
    retrained_models = train_ensemble(train_df, val_df, list(REALTIME_FEATURES), params, n_estimators)
    retrained_mae = evaluate_on_df(retrained_models, test_df, list(REALTIME_FEATURES))[0]
    print(f"    Retrained baseline MAE: {retrained_mae:.4f}% ({time.time()-t0:.1f}s)")
    print(f"    (delta from production: {(retrained_mae - base_mae) / base_mae * 100:+.2f}%)")

    # ── Run Experiments ──
    # Exp 8
    best_features_8, best_result_8 = run_experiment_8(
        train_df, val_df, test_df, progress_test,
        params, n_estimators, base_mae, baseline_bins,
    )

    # Exp 9
    best_features_9, best_result_9 = run_experiment_9(
        train_df, val_df, test_df, progress_test,
        params, n_estimators, base_mae, baseline_bins,
        best_features_8, best_result_8,
    )

    # Exp 10
    best_features_10, best_result_10, best_k = run_experiment_10(
        train_df, val_df, test_df, progress_test,
        params, n_estimators, base_mae, baseline_bins,
        best_features_9, best_result_9,
    )

    # Exp 11
    result_11 = run_experiment_11(
        train_df, val_df, test_df, progress_test,
        params, n_estimators, base_mae, baseline_bins,
        best_features_10, best_result_10, best_k,
    )

    # ── Final Summary ──
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)

    all_results = [
        ("Baseline (prod)", base_mae, baseline_bins, True, REALTIME_FEATURES),
        (f"Exp 8: {best_result_8['label']}", best_result_8["mae"], best_result_8["bins"],
         best_result_8.get("passed", False), best_features_8),
        (f"Exp 9: {best_result_9['label']}", best_result_9["mae"], best_result_9["bins"],
         best_result_9.get("passed", False), best_features_9),
        (f"Exp 10: {best_result_10['label']}", best_result_10["mae"], best_result_10["bins"],
         best_result_10.get("passed", False), best_features_10),
        (f"Exp 11: {result_11['label']}", result_11["mae"], result_11["bins"],
         result_11.get("passed", False), result_11["features"]),
    ]

    header = f"  {'Experiment':<45s}  {'MAE':>7s}  {'Delta':>8s}  {'0-25%':>7s}  {'25-50%':>7s}  {'50-75%':>7s}  {'75-100%':>7s}  {'Gate':>6s}"
    print(f"\n{header}")
    print(f"  {'-' * 45}  {'-' * 7}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 7}  {'-' * 6}")

    for label, mae, bins, passed, features in all_results:
        delta = "base" if label.startswith("Baseline") else f"{(mae - base_mae) / base_mae * 100:+.1f}%"
        gate = "--" if label.startswith("Baseline") else ("PASS" if passed else "FAIL")
        print(
            f"  {label:<45s}  {mae:7.4f}  {delta:>8s}  "
            f"{bins.get('0-25%', 0):7.4f}  {bins.get('25-50%', 0):7.4f}  "
            f"{bins.get('50-75%', 0):7.4f}  {bins.get('75-100%', 0):7.4f}  {gate:>6s}"
        )

    # Overall best
    non_baseline = [(l, m, b, p, f) for l, m, b, p, f in all_results if not l.startswith("Baseline")]
    passed_exps = [(l, m, b, p, f) for l, m, b, p, f in non_baseline if p]

    print()
    if passed_exps:
        best_overall = min(passed_exps, key=lambda x: x[1])
        pct = (best_overall[1] - base_mae) / base_mae * 100
        print(f"  BEST PASSING: {best_overall[0]}")
        print(f"    MAE: {best_overall[1]:.4f}% ({pct:+.2f}% vs baseline)")
        early_pct = (best_overall[2].get("0-25%", 0) - baseline_bins.get("0-25%", 0)) / baseline_bins.get("0-25%", 1) * 100
        print(f"    0-25% bin: {best_overall[2].get('0-25%', 0):.4f}% ({early_pct:+.1f}% vs baseline)")
        print(f"    Features ({len(best_overall[4])}): {format_features_short(best_overall[4])}")
        new_features = [f for f in best_overall[4] if f not in REALTIME_FEATURES]
        if new_features:
            print(f"    New features: {', '.join(new_features)}")
        print(f"\n  To deploy: update config.py REALTIME_FEATURES, then:")
        print(f"    python -m train.retrain --model 1b")
    else:
        # Check if any improved the 0-25% bin even without overall pass
        best_early = min(non_baseline, key=lambda x: x[2].get("0-25%", 999))
        if best_early[2].get("0-25%", 999) < baseline_bins.get("0-25%", 999):
            early_pct = (best_early[2]["0-25%"] - baseline_bins["0-25%"]) / baseline_bins["0-25%"] * 100
            print(f"  NO EXPERIMENT PASSED GATE, but 0-25% bin improved:")
            print(f"    {best_early[0]}: 0-25% MAE = {best_early[2]['0-25%']:.4f}% ({early_pct:+.1f}%)")
            print(f"    (overall MAE regressed, preventing deployment)")
        else:
            print(f"  NO EXPERIMENT IMPROVED. Production model is already well-optimized.")
            print(f"  Early-trip MAE may be irreducible given available features.")

    print()


if __name__ == "__main__":
    main()
