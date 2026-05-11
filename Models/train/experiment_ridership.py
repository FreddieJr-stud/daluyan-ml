"""Experiment: Does real ridership data make passengers_on_board a useful feature?

After populating segment_ridership with real data (493/836 segments),
this experiment tests whether adding passengers_on_board to Models 1A and 1B
improves accuracy.

Approach:
  1. Train each model with current features (baseline)
  2. Train with current features + passengers_on_board
  3. Compute SHAP importance to see where passengers ranks
  4. Compare test MAE — gate: must not degrade

Usage:
    python -m train.experiment_ridership
"""

from __future__ import annotations

import json
import sys
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
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRIP_FEATURES,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    get_monotonic_constraint_tuple,
)
from train.retrain import compute_rolling_split_dates, rolling_temporal_split
from train.train_v2 import df_to_dmatrix, ensemble_predict

FIXED_TEST_BOUNDARY = datetime(2026, 2, 12, 23, 59, 59)

PROGRESS_BINS = [
    (0.0, 0.25, "0-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 0.75, "50-75%"),
    (0.75, 1.01, "75-100%"),
]


# ── Helpers ───────────────────────────────────────────────────────────────

def load_trip_data():
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train, val, _ = rolling_temporal_split(df, train_end, val_end)
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)
    return train, val, test


def load_realtime_data():
    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train, val, _ = rolling_temporal_split(df, train_end, val_end)
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)
    return train, val, test


def build_params(metadata_path: str) -> tuple[dict, int]:
    with open(metadata_path) as f:
        saved = json.load(f)["best_params"]
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": saved["max_depth"],
        "learning_rate": saved["learning_rate"],
        "subsample": saved["subsample"],
        "colsample_bytree": saved["colsample_bytree"],
        "colsample_bylevel": saved.get("colsample_bylevel", 1.0),
        "min_child_weight": saved["min_child_weight"],
        "reg_alpha": saved["reg_alpha"],
        "reg_lambda": saved["reg_lambda"],
        "gamma": saved.get("gamma", 0.0),
    }
    return params, saved["n_estimators"]


def train_ensemble(train_df, val_df, features, target, params, n_est,
                   monotonic=None):
    dtrain, _ = df_to_dmatrix(train_df, features, target)
    dval, _ = df_to_dmatrix(val_df, features, target)
    if monotonic is not None:
        params = {**params, "monotone_constraints": monotonic}
    models = []
    for seed in ENSEMBLE_SEEDS:
        p = {**params, "seed": seed}
        m = xgb.train(
            p, dtrain,
            num_boost_round=n_est,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        models.append(m)
    return models


def evaluate(models, df, features, target):
    X = df.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features)
    preds = ensemble_predict(models, dm)
    y = df[target].to_numpy().astype(np.float32)
    errors = np.abs(y - preds)
    mae = float(errors.mean())
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2, preds, errors


def compute_shap(models, df, features, n_samples=3000):
    import shap
    n = min(n_samples, len(df))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(df), n, replace=False)
    X = df.select(features).to_numpy().astype(np.float32)[idx]
    dm = xgb.DMatrix(X, feature_names=features)

    all_shap = np.zeros((n, len(features)))
    for m in models:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(dm)
        all_shap += np.abs(sv)
    all_shap /= len(models)

    mean_shap = all_shap.mean(axis=0)
    ranked_idx = np.argsort(mean_shap)[::-1]
    return [(features[i], float(mean_shap[i])) for i in ranked_idx]


def bin_maes(errors, progress):
    result = {}
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() > 0:
            result[label] = float(errors[mask].mean())
    return result


# ── Model 1A Experiment ───────────────────────────────────────────────────

def experiment_1a():
    print("\n" + "=" * 70)
    print("  MODEL 1A: Ridership Feature Experiment")
    print("=" * 70)

    train, val, test = load_trip_data()
    print(f"  Data: train={len(train)}, val={len(val)}, test={len(test)}")

    params, n_est = build_params(str(ARTIFACTS_DIR / "soc_trip_metadata.json"))

    # Baseline: current 15 features
    print("\n  [Baseline] 15 production features...")
    baseline_features = list(TRIP_FEATURES)
    mono_base = get_monotonic_constraint_tuple(baseline_features, TRIP_MONOTONIC_CONSTRAINTS)
    baseline_models = train_ensemble(train, val, baseline_features, TRIP_TARGET,
                                     params, n_est, monotonic=mono_base)
    b_mae, b_rmse, b_r2, _, _ = evaluate(baseline_models, test, baseline_features, TRIP_TARGET)
    b_val_mae = evaluate(baseline_models, val, baseline_features, TRIP_TARGET)[0]
    print(f"    Test MAE={b_mae:.4f}%  RMSE={b_rmse:.4f}%  R2={b_r2:.4f}")
    print(f"    Val  MAE={b_val_mae:.4f}%  Gap={b_mae - b_val_mae:+.4f}%")

    # Candidate: 15 + passengers_on_board = 16 features
    print("\n  [Candidate] 16 features (+passengers_on_board)...")
    candidate_features = baseline_features + ["passengers_on_board"]
    mono_cand = get_monotonic_constraint_tuple(candidate_features, TRIP_MONOTONIC_CONSTRAINTS)
    candidate_models = train_ensemble(train, val, candidate_features, TRIP_TARGET,
                                      params, n_est, monotonic=mono_cand)
    c_mae, c_rmse, c_r2, _, _ = evaluate(candidate_models, test, candidate_features, TRIP_TARGET)
    c_val_mae = evaluate(candidate_models, val, candidate_features, TRIP_TARGET)[0]
    print(f"    Test MAE={c_mae:.4f}%  RMSE={c_rmse:.4f}%  R2={c_r2:.4f}")
    print(f"    Val  MAE={c_val_mae:.4f}%  Gap={c_mae - c_val_mae:+.4f}%")

    delta_pct = (c_mae - b_mae) / b_mae * 100
    print(f"\n  Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    # SHAP ranking on candidate model
    print("\n  SHAP importance ranking (16 candidate features):")
    shap_ranking = compute_shap(candidate_models, val, candidate_features,
                                n_samples=min(3000, len(val)))
    for i, (feat, imp) in enumerate(shap_ranking):
        marker = " <-- PASSENGERS" if feat == "passengers_on_board" else ""
        print(f"    {i+1:>2}. {feat:<35} {imp:.4f}{marker}")

    pax_rank = next(i + 1 for i, (f, _) in enumerate(shap_ranking) if f == "passengers_on_board")
    pax_shap = next(imp for f, imp in shap_ranking if f == "passengers_on_board")

    # Decision
    passed = delta_pct <= 0  # must not degrade
    gap_ok = (c_mae - c_val_mae) <= (b_mae - b_val_mae) * 1.5  # gap shouldn't blow up

    print(f"\n  Decision:")
    print(f"    MAE gate: {'PASS' if passed else 'FAIL'} ({delta_pct:+.2f}%)")
    print(f"    Gap gate: {'PASS' if gap_ok else 'FAIL'} "
          f"(base gap={b_mae - b_val_mae:+.4f}, new gap={c_mae - c_val_mae:+.4f})")
    print(f"    SHAP rank: #{pax_rank}/16 (importance={pax_shap:.4f})")

    keep = passed and gap_ok
    print(f"    >> {'KEEP passengers_on_board' if keep else 'DROP passengers_on_board'}")

    return {
        "model": "1A",
        "baseline_mae": b_mae,
        "candidate_mae": c_mae,
        "delta_pct": delta_pct,
        "pax_shap_rank": pax_rank,
        "pax_shap_importance": pax_shap,
        "keep": keep,
        "baseline_gap": b_mae - b_val_mae,
        "candidate_gap": c_mae - c_val_mae,
    }


# ── Model 1B Experiment ───────────────────────────────────────────────────

def experiment_1b():
    print("\n" + "=" * 70)
    print("  MODEL 1B: Ridership Feature Experiment")
    print("=" * 70)

    train, val, test = load_realtime_data()
    prog_test = test["trip_progress_fraction"].to_numpy()
    print(f"  Data: train={len(train):,}, val={len(val):,}, test={len(test):,}")

    params, n_est = build_params(str(ARTIFACTS_DIR / "soc_realtime_metadata.json"))

    # Baseline: current 25 features
    print("\n  [Baseline] 25 production features...")
    baseline_features = list(REALTIME_FEATURES)
    baseline_models = train_ensemble(train, val, baseline_features, REALTIME_TARGET,
                                     params, n_est)
    b_mae, b_rmse, b_r2, _, b_errors = evaluate(baseline_models, test,
                                                  baseline_features, REALTIME_TARGET)
    b_val_mae = evaluate(baseline_models, val, baseline_features, REALTIME_TARGET)[0]
    b_bins = bin_maes(b_errors, prog_test)
    print(f"    Test MAE={b_mae:.4f}%  RMSE={b_rmse:.4f}%  R2={b_r2:.4f}")
    print(f"    Val  MAE={b_val_mae:.4f}%  Gap={b_mae - b_val_mae:+.4f}%")
    print(f"    Per-progress: " + ", ".join(f"{k}={v:.4f}" for k, v in b_bins.items()))

    # Candidate: 25 + passengers_on_board = 26 features
    print("\n  [Candidate] 26 features (+passengers_on_board)...")
    candidate_features = baseline_features + ["passengers_on_board"]
    candidate_models = train_ensemble(train, val, candidate_features, REALTIME_TARGET,
                                      params, n_est)
    c_mae, c_rmse, c_r2, _, c_errors = evaluate(candidate_models, test,
                                                  candidate_features, REALTIME_TARGET)
    c_val_mae = evaluate(candidate_models, val, candidate_features, REALTIME_TARGET)[0]
    c_bins = bin_maes(c_errors, prog_test)
    print(f"    Test MAE={c_mae:.4f}%  RMSE={c_rmse:.4f}%  R2={c_r2:.4f}")
    print(f"    Val  MAE={c_val_mae:.4f}%  Gap={c_mae - c_val_mae:+.4f}%")
    print(f"    Per-progress: " + ", ".join(f"{k}={v:.4f}" for k, v in c_bins.items()))

    delta_pct = (c_mae - b_mae) / b_mae * 100
    print(f"\n  Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    # Check per-bin degradation
    bin_deltas = {}
    any_bin_fail = False
    for label in b_bins:
        if label in c_bins and b_bins[label] > 0:
            pct = (c_bins[label] - b_bins[label]) / b_bins[label] * 100
            bin_deltas[label] = pct
            if pct > 20:
                any_bin_fail = True
            print(f"    {label}: {pct:+.1f}%")

    # SHAP ranking
    print("\n  SHAP importance ranking (26 candidate features):")
    shap_ranking = compute_shap(candidate_models, val, candidate_features,
                                n_samples=min(3000, len(val)))
    for i, (feat, imp) in enumerate(shap_ranking):
        marker = " <-- PASSENGERS" if feat == "passengers_on_board" else ""
        print(f"    {i+1:>2}. {feat:<35} {imp:.4f}{marker}")

    pax_rank = next(i + 1 for i, (f, _) in enumerate(shap_ranking) if f == "passengers_on_board")
    pax_shap = next(imp for f, imp in shap_ranking if f == "passengers_on_board")

    # Decision
    passed = delta_pct <= 0
    gap_ok = (c_mae - c_val_mae) <= (b_mae - b_val_mae) * 1.5

    print(f"\n  Decision:")
    print(f"    MAE gate: {'PASS' if passed else 'FAIL'} ({delta_pct:+.2f}%)")
    print(f"    Bin gate: {'PASS' if not any_bin_fail else 'FAIL'}")
    print(f"    Gap gate: {'PASS' if gap_ok else 'FAIL'} "
          f"(base gap={b_mae - b_val_mae:+.4f}, new gap={c_mae - c_val_mae:+.4f})")
    print(f"    SHAP rank: #{pax_rank}/26 (importance={pax_shap:.4f})")

    keep = passed and gap_ok and not any_bin_fail
    print(f"    >> {'KEEP passengers_on_board' if keep else 'DROP passengers_on_board'}")

    return {
        "model": "1B",
        "baseline_mae": b_mae,
        "candidate_mae": c_mae,
        "delta_pct": delta_pct,
        "pax_shap_rank": pax_rank,
        "pax_shap_importance": pax_shap,
        "keep": keep,
        "baseline_gap": b_mae - b_val_mae,
        "candidate_gap": c_mae - c_val_mae,
        "bin_deltas": bin_deltas,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RIDERSHIP FEATURE EXPERIMENT")
    print("  Testing if real passenger data improves models")
    print("=" * 70)

    results = {}
    results["1A"] = experiment_1a()
    results["1B"] = experiment_1b()

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for model_key in ["1A", "1B"]:
        r = results[model_key]
        status = "KEEP" if r["keep"] else "DROP"
        print(f"  Model {model_key}: {status}")
        print(f"    MAE: {r['baseline_mae']:.4f}% -> {r['candidate_mae']:.4f}% ({r['delta_pct']:+.2f}%)")
        print(f"    SHAP rank: #{r['pax_shap_rank']} (importance={r['pax_shap_importance']:.4f})")

    # Save results
    out = ARTIFACTS_DIR / "ridership_experiment_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    import json
    serializable = json.loads(json.dumps(results, default=convert))
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\n  Results saved to: {out}")


if __name__ == "__main__":
    main()
