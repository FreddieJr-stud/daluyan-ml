"""SHAP feature pruning experiment for Model 1B (Realtime SOC).

Tests whether reducing Model 1B's 33 features via SHAP importance ranking
improves generalization, following the same approach that cut Model 1A's
features from 31→15 and improved test MAE by 3.8%.

Model 1B has ~40K training rows (vs 1A's 470), so overfitting from excess
features is less likely. However, correlated/redundant features (e.g., three
SOC rate windows, two power averages) may still add noise.

Usage:
    python -m train.experiment_1b
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


def evaluate_model(
    models: list[xgb.Booster],
    test_df: pl.DataFrame,
    features: list[str],
    target: str,
) -> dict:
    """Evaluate ensemble on test set. Returns metrics dict with preds and errors."""
    X = test_df.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features)
    preds = ensemble_predict(models, dm)

    y_true = test_df[target].to_numpy().astype(np.float32)
    abs_errors = np.abs(y_true - preds)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    ss_res = float(np.sum((y_true - preds) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds, "errors": abs_errors}


def get_progress_bin_maes(
    errors: np.ndarray, test_df: pl.DataFrame,
) -> dict[str, float]:
    """Return per-progress-bin MAEs as a dict."""
    progress = test_df["trip_progress_fraction"].to_numpy()
    bins = [
        (0.0, 0.25, "0-25%"),
        (0.25, 0.50, "25-50%"),
        (0.50, 0.75, "50-75%"),
        (0.75, 1.01, "75-100%"),
    ]
    result = {}
    for lo, hi, label in bins:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() > 0:
            result[label] = float(errors[mask].mean())
    return result


def print_progress_breakdown(errors: np.ndarray, test_df: pl.DataFrame) -> None:
    """Print per-progress-bin MAE with sample counts."""
    progress = test_df["trip_progress_fraction"].to_numpy()
    bins = [
        (0.0, 0.25, "0-25%"),
        (0.25, 0.50, "25-50%"),
        (0.50, 0.75, "50-75%"),
        (0.75, 1.01, "75-100%"),
    ]
    for lo, hi, label in bins:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() == 0:
            continue
        print(f"      {label}: MAE={errors[mask].mean():.4f}%  n={mask.sum():,}")


def check_pass_gate(
    exp_errors: np.ndarray,
    baseline_errors: np.ndarray,
    test_df: pl.DataFrame,
    exp_mae: float,
    baseline_mae: float,
) -> tuple[bool, list[str]]:
    """Check if experiment passes.

    Criteria:
    - Overall MAE must improve
    - No progress bin may degrade >20%

    Returns (passed, reasons).
    """
    reasons = []
    if exp_mae >= baseline_mae:
        reasons.append(f"Overall MAE did not improve ({exp_mae:.4f} >= {baseline_mae:.4f})")

    exp_bins = get_progress_bin_maes(exp_errors, test_df)
    base_bins = get_progress_bin_maes(baseline_errors, test_df)

    for label, base_val in base_bins.items():
        if label in exp_bins and base_val > 0:
            pct_change = (exp_bins[label] - base_val) / base_val * 100
            if pct_change > 20:
                reasons.append(
                    f"{label} degraded {pct_change:+.1f}% "
                    f"({base_val:.4f} -> {exp_bins[label]:.4f})"
                )

    return len(reasons) == 0, reasons


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("ERROR: 'shap' not installed. Run: pip install shap")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  MODEL 1B EXPERIMENT: SHAP Feature Pruning")
    print("=" * 65)

    # ── Load data ──
    print("\n  Loading realtime feature store...")
    train_df, val_df, test_df, train_end, val_end = load_realtime_split()

    # ── Production baseline ──
    saved_params = load_production_1b_params()
    prod_models = load_production_1b_models()

    base_result = evaluate_model(prod_models, test_df, REALTIME_FEATURES, REALTIME_TARGET)
    baseline_mae = base_result["MAE"]
    print(f"\n  Production baseline (fixed test): MAE={baseline_mae:.4f}%")
    print("  Per-progress breakdown:")
    print_progress_breakdown(base_result["errors"], test_df)

    # ── SHAP Analysis ──
    print("\n  Computing SHAP values on validation subsample...")
    import shap

    # Subsample validation set for tractability (~2K rows)
    n_shap = min(2000, len(val_df))
    rng = np.random.RandomState(42)
    shap_indices = rng.choice(len(val_df), n_shap, replace=False)
    X_val_shap = val_df.select(REALTIME_FEATURES).to_numpy().astype(np.float32)[shap_indices]
    dm_val_shap = xgb.DMatrix(X_val_shap, feature_names=REALTIME_FEATURES)

    # Average |SHAP| across ensemble members
    all_shap = np.zeros((n_shap, len(REALTIME_FEATURES)))
    for model in prod_models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(dm_val_shap)
        all_shap += np.abs(sv)
    all_shap /= len(prod_models)

    # Rank features by mean |SHAP|
    mean_shap = all_shap.mean(axis=0)
    ranked_indices = np.argsort(mean_shap)[::-1]
    ranked_features = [REALTIME_FEATURES[i] for i in ranked_indices]

    print("\n  Feature importance ranking (mean |SHAP|):")
    for i, idx in enumerate(ranked_indices):
        print(f"    {i+1:2d}. {REALTIME_FEATURES[idx]:<35s} {mean_shap[idx]:.4f}")

    # ── Test pruned models at N = 15, 20, 25, 30 ──
    n_values = [15, 20, 25, 30]
    results = []

    for n_keep in n_values:
        print(f"\n  {'-' * 55}")
        print(f"  Pruning to top {n_keep} features")
        print(f"  {'-' * 55}")

        pruned_features = ranked_features[:n_keep]
        print(f"  Kept: {', '.join(pruned_features[:5])} ... ({n_keep} total)")

        # Build XGBoost params (no monotonic constraints for 1B)
        best_params = {
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

        # Build DMatrices
        dtrain, _ = df_to_dmatrix(train_df, pruned_features, REALTIME_TARGET)
        dval, _ = df_to_dmatrix(val_df, pruned_features, REALTIME_TARGET)

        # Train 5-seed ensemble
        print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble...")
        t0 = time.time()
        models = []
        for seed in ENSEMBLE_SEEDS:
            p = {**best_params, "seed": seed}
            model = xgb.train(
                p, dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )
            models.append(model)
        elapsed = time.time() - t0
        print(f"  Training done ({elapsed:.1f}s)")

        # Evaluate on fixed test
        result = evaluate_model(models, test_df, pruned_features, REALTIME_TARGET)
        delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100

        print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
        print(f"    RMSE: {result['RMSE']:.4f}%")
        print(f"    R2:   {result['R2']:.4f}")
        print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

        print("  Per-progress breakdown:")
        print_progress_breakdown(result["errors"], test_df)

        # Val-test gap
        val_result = evaluate_model(models, val_df, pruned_features, REALTIME_TARGET)
        gap = result["MAE"] - val_result["MAE"]
        print(f"    Val MAE: {val_result['MAE']:.4f}% | Gap: {gap:+.4f}%")

        # Pass gate
        passed, reasons = check_pass_gate(
            result["errors"], base_result["errors"], test_df,
            result["MAE"], baseline_mae,
        )
        if passed:
            print("    PASS GATE: PASSED")
        else:
            print("    PASS GATE: FAILED")
            for r in reasons:
                print(f"      - {r}")

        results.append({
            "n_features": n_keep,
            "features": pruned_features,
            "mae": result["MAE"],
            "rmse": result["RMSE"],
            "r2": result["R2"],
            "delta_pct": delta_pct,
            "val_mae": val_result["MAE"],
            "gap": gap,
            "passed": passed,
            "reasons": reasons,
        })

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"\n  {'N':>3s}  {'Test MAE':>10s}  {'Delta':>8s}  {'Val MAE':>10s}  "
          f"{'Gap':>8s}  {'Pass':>6s}")
    print(f"  {'-' * 3}  {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 6}")
    print(f"  {33:3d}  {baseline_mae:10.4f}  {'base':>8s}  {'--':>10s}  {'--':>8s}  {'--':>6s}")
    for r in results:
        print(
            f"  {r['n_features']:3d}  {r['mae']:10.4f}  {r['delta_pct']:+7.2f}%  "
            f"{r['val_mae']:10.4f}  {r['gap']:+7.4f}  "
            f"{'PASS' if r['passed'] else 'FAIL':>6s}"
        )

    # Recommend
    passed_results = [r for r in results if r["passed"]]
    if passed_results:
        best = min(passed_results, key=lambda r: r["mae"])
        print(f"\n  RECOMMENDATION: Use top-{best['n_features']} features")
        print(f"  MAE improvement: {best['delta_pct']:+.2f}%")
        print(f"  Features to keep:")
        for i, f in enumerate(best["features"]):
            print(f"    {i+1:2d}. {f}")
    else:
        print(f"\n  RECOMMENDATION: Keep all 33 features (no pruning passed gate)")


if __name__ == "__main__":
    main()
