"""SHAP feature analysis + pruning experiment for Model 2 (Anomaly Detection).

Model 2 uses 4 reconstruction-error XGBoost models (one per motor target),
each predicting one target from the other 14 features. The anomaly score is
the sum of squared normalized reconstruction errors.

This script:
  1. Computes per-model SHAP rankings on validation data
  2. Identifies consistently low-importance features across all 4 models
  3. Tests pruned feature sets by retraining all 4 models and running the
     full anomaly pipeline (threshold calibration + injection test)

Usage:
    python -m train.experiment_2
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
    ANOMALY_FEATURES,
    ANOMALY_TARGETS,
    ARTIFACTS_DIR,
    FEATURE_STORE_DIR,
)
from evaluate.evaluate_anomaly import synthetic_injection_test
from train.retrain import compute_rolling_split_dates, rolling_temporal_split

# Fixed test boundary (original thesis benchmark)
FIXED_TEST_BOUNDARY = datetime(2026, 2, 12, 23, 59, 59)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_anomaly_split() -> tuple[
    pl.DataFrame, pl.DataFrame, pl.DataFrame, datetime, datetime
]:
    """Load anomaly feature store and return train/val/fixed_test DataFrames."""
    df = pl.read_parquet(FEATURE_STORE_DIR / "anomaly_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train, val, _ = rolling_temporal_split(df, train_end, val_end)
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)

    print(f"  Split dates: train <= {train_end.date()}, val <= {val_end.date()}, "
          f"test > {FIXED_TEST_BOUNDARY.date()}")
    print(f"  Rows: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    return train, val, test, train_end, val_end


def load_production_models() -> dict[str, xgb.Booster]:
    """Load production anomaly models."""
    models = {}
    for target in ANOMALY_TARGETS:
        path = ARTIFACTS_DIR / f"anomaly_{target}.json"
        m = xgb.Booster()
        m.load_model(str(path))
        models[target] = m
    return models


def load_production_params() -> dict[str, dict]:
    """Load per-target hyperparameters from metadata."""
    meta_path = ARTIFACTS_DIR / "anomaly_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["per_target_best_params"]


def load_feature_stats() -> tuple[dict[str, float], dict[str, float]]:
    """Load feature means and stds for normalisation."""
    with open(ARTIFACTS_DIR / "feature_statistics.json") as f:
        stats = json.load(f)
    return stats["feature_means"], stats["feature_stds"]


def compute_anomaly_scores(
    models: dict[str, xgb.Booster],
    data: np.ndarray,
    feature_names: list[str],
    feature_stds: dict[str, float],
) -> np.ndarray:
    """Compute combined anomaly scores from numpy array."""
    scores = np.zeros(len(data))
    for target_col, model in models.items():
        predictor_cols = [c for c in feature_names if c != target_col]
        target_idx = feature_names.index(target_col)
        predictor_idxs = [feature_names.index(c) for c in predictor_cols]
        X = data[:, predictor_idxs]
        y = data[:, target_idx]
        dm = xgb.DMatrix(X, feature_names=predictor_cols)
        pred = model.predict(dm)
        error = ((pred - y) / feature_stds[target_col]) ** 2
        scores += error
    return scores


def train_anomaly_models(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    feature_names: list[str],
    targets: list[str],
    per_target_params: dict[str, dict],
) -> tuple[dict[str, xgb.Booster], dict[str, float]]:
    """Train reconstruction models and compute feature stats.

    Returns (models, feature_stds).
    """
    feature_stds = {}
    for feat in feature_names:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_stds[feat] = float(max(np.std(col), 0.01))

    models = {}
    for target_col in targets:
        predictor_cols = [c for c in feature_names if c != target_col]

        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        saved = per_target_params[target_col]
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": saved["max_depth"],
            "learning_rate": saved["learning_rate"],
            "subsample": saved["subsample"],
            "colsample_bytree": saved["colsample_bytree"],
            "min_child_weight": saved["min_child_weight"],
            "reg_alpha": saved["reg_alpha"],
            "reg_lambda": saved["reg_lambda"],
            "gamma": saved["gamma"],
            "seed": 42,
        }

        model = xgb.train(
            params, dtrain,
            num_boost_round=saved["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )
        models[target_col] = model

        val_pred = model.predict(dval)
        val_mae = np.mean(np.abs(y_val - val_pred))
        print(f"      {target_col}: val MAE={val_mae:.4f}, "
              f"trees={model.num_boosted_rounds()}")

    return models, feature_stds


def run_injection_test(
    models: dict[str, xgb.Booster],
    test_data: np.ndarray,
    feature_names: list[str],
    feature_stds: dict[str, float],
    threshold: float,
) -> dict:
    """Run the synthetic injection test."""
    def score_fn(data):
        return compute_anomaly_scores(models, data, feature_names, feature_stds)

    return synthetic_injection_test(
        score_fn, test_data, feature_names,
        n_injections=500, corruption_factor=1.5, threshold=threshold,
    )


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("ERROR: 'shap' not installed. Run: pip install shap")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("  MODEL 2 EXPERIMENT: SHAP Feature Analysis + Pruning")
    print("=" * 65)

    # ── Load data ──
    print("\n  Loading anomaly feature store...")
    train_df, val_df, test_df, train_end, val_end = load_anomaly_split()

    # ── Production baseline ──
    print("\n  Loading production models...")
    prod_models = load_production_models()
    per_target_params = load_production_params()
    _, prod_feature_stds = load_feature_stats()

    test_data = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

    # Compute baseline scores and threshold from validation
    val_scores = compute_anomaly_scores(
        prod_models, val_data, ANOMALY_FEATURES, prod_feature_stds,
    )
    baseline_threshold = float(np.percentile(val_scores, 99))

    baseline_injection = run_injection_test(
        prod_models, test_data, ANOMALY_FEATURES, prod_feature_stds,
        baseline_threshold,
    )
    baseline_det = baseline_injection["detection_rate"]
    baseline_fpr = baseline_injection["false_positive_rate"]

    print(f"\n  Production baseline:")
    print(f"    Detection rate: {baseline_det:.1f}%")
    print(f"    FPR: {baseline_fpr:.1f}%")
    print(f"    Threshold (p99): {baseline_threshold:.6f}")
    print(f"    Features: {len(ANOMALY_FEATURES)} ({len(ANOMALY_FEATURES) - len(ANOMALY_TARGETS)} "
          f"unique predictors per model)")

    # ═════════════════════════════════════════════════════════════════════
    # SHAP Analysis — per reconstruction model
    # ═════════════════════════════════════════════════════════════════════
    print("\n  Computing SHAP values per reconstruction model...")
    import shap

    n_shap = min(2000, len(val_df))
    rng = np.random.RandomState(42)
    shap_indices = rng.choice(len(val_df), n_shap, replace=False)

    # Collect per-model SHAP rankings
    per_model_shap = {}  # {target: {feature: mean_abs_shap}}

    for target_col, model in prod_models.items():
        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]
        target_idx = ANOMALY_FEATURES.index(target_col)
        predictor_idxs = [ANOMALY_FEATURES.index(c) for c in predictor_cols]

        X_shap = val_data[shap_indices][:, predictor_idxs]
        dm_shap = xgb.DMatrix(X_shap, feature_names=predictor_cols)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(dm_shap)
        mean_abs = np.abs(sv).mean(axis=0)

        feature_shap = {f: float(v) for f, v in zip(predictor_cols, mean_abs)}
        per_model_shap[target_col] = feature_shap

        print(f"\n    --- {target_col} reconstruction ---")
        ranked = sorted(feature_shap.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, val) in enumerate(ranked):
            print(f"      {i+1:2d}. {feat:<35s} {val:.4f}")

    # ── Aggregate: mean SHAP across models where each feature appears ──
    print("\n  " + "=" * 55)
    print("  Aggregate SHAP (mean across reconstruction models)")
    print("  " + "=" * 55)

    agg_shap = {}
    for feat in ANOMALY_FEATURES:
        vals = []
        for target_col, feat_shap in per_model_shap.items():
            if feat in feat_shap:  # feature isn't this model's target
                vals.append(feat_shap[feat])
        if vals:
            agg_shap[feat] = np.mean(vals)

    ranked_agg = sorted(agg_shap.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Rank':>4s}  {'Feature':<35s}  {'Mean |SHAP|':>12s}  {'Models':>6s}")
    print(f"  {'-' * 4}  {'-' * 35}  {'-' * 12}  {'-' * 6}")
    for i, (feat, val) in enumerate(ranked_agg):
        # Count how many models use this feature as predictor
        n_models = sum(1 for t in ANOMALY_TARGETS if feat != t)
        is_target = " (target)" if feat in ANOMALY_TARGETS else ""
        print(f"  {i+1:4d}  {feat:<35s}  {val:12.4f}  {n_models:>6d}{is_target}")

    # ── Identify pruning candidates ──
    # Features below a clear gap in the ranking
    # Only prune derived features — raw features could be targets
    raw_features = set([
        "port_motor_power", "stbd_motor_power", "motor_power_combined",
        "port_rpm", "stbd_rpm", "speed_over_ground", "battery_power",
    ])
    derived_features = [f for f in ANOMALY_FEATURES if f not in raw_features]

    print(f"\n  Raw features ({len(raw_features)}): {sorted(raw_features)}")
    print(f"  Derived features ({len(derived_features)}): {derived_features}")

    # Rank derived features by aggregate SHAP
    derived_ranked = sorted(
        [(f, agg_shap.get(f, 0)) for f in derived_features],
        key=lambda x: x[1],
        reverse=True,
    )
    print(f"\n  Derived features ranked by aggregate SHAP:")
    for i, (feat, val) in enumerate(derived_ranked):
        print(f"    {i+1}. {feat:<35s} {val:.4f}")

    # ═════════════════════════════════════════════════════════════════════
    # Pruning Experiments
    # ═════════════════════════════════════════════════════════════════════
    # Test removing the bottom N derived features
    # N = 1, 2, 3, 4 (out of 8 derived features)
    n_remove_values = [1, 2, 3, 4]
    results = []

    for n_remove in n_remove_values:
        features_to_remove = [f for f, _ in derived_ranked[-n_remove:]]
        pruned_features = [f for f in ANOMALY_FEATURES if f not in features_to_remove]

        print(f"\n  {'-' * 55}")
        print(f"  Removing {n_remove} derived feature(s): {features_to_remove}")
        print(f"  Keeping {len(pruned_features)} features")
        print(f"  {'-' * 55}")

        # Retrain all 4 models with pruned features
        print(f"  Training 4 reconstruction models...")
        t0 = time.time()
        models, feature_stds = train_anomaly_models(
            train_df, val_df, pruned_features, ANOMALY_TARGETS, per_target_params,
        )
        elapsed = time.time() - t0
        print(f"  Training done ({elapsed:.1f}s)")

        # Recompute thresholds on validation
        pruned_val_data = val_df.select(pruned_features).to_numpy().astype(np.float32)
        pruned_test_data = test_df.select(pruned_features).to_numpy().astype(np.float32)

        val_scores = compute_anomaly_scores(
            models, pruned_val_data, pruned_features, feature_stds,
        )
        threshold_p99 = float(np.percentile(val_scores, 99))

        # Injection test
        injection = run_injection_test(
            models, pruned_test_data, pruned_features, feature_stds, threshold_p99,
        )
        det_rate = injection["detection_rate"]
        fpr = injection["false_positive_rate"]

        det_delta = det_rate - baseline_det
        fpr_delta = fpr - baseline_fpr

        print(f"    Detection rate: {det_rate:.1f}% (baseline: {baseline_det:.1f}%, "
              f"delta: {det_delta:+.1f}pp)")
        print(f"    FPR: {fpr:.1f}% (baseline: {baseline_fpr:.1f}%, "
              f"delta: {fpr_delta:+.1f}pp)")
        print(f"    Threshold (p99): {threshold_p99:.6f}")

        # Per-target reconstruction MAE on test set
        print(f"    Per-target test reconstruction:")
        for target_col in ANOMALY_TARGETS:
            predictor_cols = [c for c in pruned_features if c != target_col]
            target_idx = pruned_features.index(target_col)
            predictor_idxs = [pruned_features.index(c) for c in predictor_cols]
            X = pruned_test_data[:, predictor_idxs]
            y = pruned_test_data[:, target_idx]
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            pred = models[target_col].predict(dm)
            mae = float(np.mean(np.abs(y - pred)))
            print(f"      {target_col}: MAE={mae:.4f}")

        # Pass gate: detection must not drop, FPR must not increase >2pp
        passed = True
        reasons = []
        if det_rate < baseline_det - 0.5:  # allow 0.5pp tolerance
            passed = False
            reasons.append(f"Detection dropped {det_delta:+.1f}pp")
        if fpr > baseline_fpr + 2.0:
            passed = False
            reasons.append(f"FPR increased {fpr_delta:+.1f}pp (>{baseline_fpr + 2.0:.1f}%)")

        if passed:
            print(f"    PASS GATE: PASSED")
        else:
            print(f"    PASS GATE: FAILED")
            for r in reasons:
                print(f"      - {r}")

        results.append({
            "n_removed": n_remove,
            "removed": features_to_remove,
            "n_features": len(pruned_features),
            "det_rate": det_rate,
            "det_delta": det_delta,
            "fpr": fpr,
            "fpr_delta": fpr_delta,
            "threshold": threshold_p99,
            "passed": passed,
            "reasons": reasons,
        })

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"\n  {'Removed':>7s}  {'N feat':>6s}  {'Det Rate':>10s}  {'Det D':>7s}  "
          f"{'FPR':>7s}  {'FPR D':>7s}  {'Pass':>6s}")
    print(f"  {'-' * 7}  {'-' * 6}  {'-' * 10}  {'-' * 7}  "
          f"{'-' * 7}  {'-' * 7}  {'-' * 6}")
    print(f"  {'0':>7s}  {15:6d}  {baseline_det:9.1f}%  {'base':>7s}  "
          f"{baseline_fpr:6.1f}%  {'base':>7s}  {'--':>6s}")
    for r in results:
        print(
            f"  {r['n_removed']:7d}  {r['n_features']:6d}  {r['det_rate']:9.1f}%  "
            f"{r['det_delta']:+6.1f}pp  {r['fpr']:6.1f}%  {r['fpr_delta']:+6.1f}pp  "
            f"{'PASS' if r['passed'] else 'FAIL':>6s}"
        )

    # Recommendation
    passed_results = [r for r in results if r["passed"]]
    if passed_results:
        # Prefer most features removed while still passing
        best = max(passed_results, key=lambda r: r["n_removed"])
        print(f"\n  RECOMMENDATION: Remove {best['n_removed']} feature(s): {best['removed']}")
        print(f"  Detection: {best['det_rate']:.1f}% (delta: {best['det_delta']:+.1f}pp)")
        print(f"  FPR: {best['fpr']:.1f}% (delta: {best['fpr_delta']:+.1f}pp)")
    else:
        print(f"\n  RECOMMENDATION: Keep all 15 features (no pruning passed gate)")

    # Also show if detection actually improved in any case
    improved = [r for r in results if r["det_rate"] > baseline_det]
    if improved:
        best_det = max(improved, key=lambda r: r["det_rate"])
        print(f"\n  NOTE: Detection improved to {best_det['det_rate']:.1f}% "
              f"with {best_det['n_removed']} feature(s) removed: {best_det['removed']}")


if __name__ == "__main__":
    main()
