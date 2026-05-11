"""SHAP re-pruning for Model 1B with expanded feature candidates.

After Exp 8-11, we have 29 candidate features (25 production + 4 new
trip-so-far/route-prior features). This script:

1. Trains an ensemble on all 29 features
2. Computes SHAP importance ranking
3. Tests pruning at N = 20, 22, 24, 25, 26, 27, 28, 29
4. Reports per-progress-bin breakdowns for thesis

Usage:
    python -m train.experiment_1b_shap_reprune
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

FIXED_TEST_BOUNDARY = datetime(2026, 2, 12, 23, 59, 59)

# All candidate features: production 25 + 4 experimental
CANDIDATE_FEATURES = list(REALTIME_FEATURES) + [
    "soc_consumed_so_far",
    "empirical_soc_per_km",
    "empirical_power_per_km",
    "route_avg_soc_delta",
]

PROGRESS_BINS = [
    (0.0, 0.25, "0-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 0.75, "50-75%"),
    (0.75, 1.01, "75-100%"),
]


# -- Helpers --------------------------------------------------------------


def load_data():
    """Load and split data."""
    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_end, val_end = compute_rolling_split_dates(df)
    train, val, _ = rolling_temporal_split(df, train_end, val_end)
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)
    return train, val, test, train_end, val_end


def load_prod_params() -> dict:
    with open(ARTIFACTS_DIR / "soc_realtime_metadata.json") as f:
        return json.load(f)["best_params"]


def load_prod_models() -> list[xgb.Booster]:
    models = []
    for seed in ENSEMBLE_SEEDS:
        suffix = "" if seed == 42 else f"_seed{seed}"
        m = xgb.Booster()
        m.load_model(str(ARTIFACTS_DIR / f"soc_realtime_model{suffix}.json"))
        models.append(m)
    return models


def build_params(saved: dict) -> tuple[dict, int]:
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": saved["max_depth"],
        "learning_rate": saved["learning_rate"],
        "subsample": saved["subsample"],
        "colsample_bytree": saved["colsample_bytree"],
        "colsample_bylevel": saved["colsample_bylevel"],
        "min_child_weight": saved["min_child_weight"],
        "reg_alpha": saved["reg_alpha"],
        "reg_lambda": saved["reg_lambda"],
        "gamma": saved["gamma"],
    }
    return params, saved["n_estimators"]


def train_ensemble(train_df, val_df, features, params, n_est):
    dtrain, _ = df_to_dmatrix(train_df, features, REALTIME_TARGET)
    dval, _ = df_to_dmatrix(val_df, features, REALTIME_TARGET)
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


def evaluate(models, df, features):
    X = df.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features)
    preds = ensemble_predict(models, dm)
    y = df[REALTIME_TARGET].to_numpy().astype(np.float32)
    errors = np.abs(y - preds)
    mae = float(errors.mean())
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2, preds, errors


def bin_maes(errors, progress):
    result = {}
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() > 0:
            result[label] = float(errors[mask].mean())
    return result


def print_bins(errors, progress, base_bins=None):
    for lo, hi, label in PROGRESS_BINS:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() == 0:
            continue
        m = float(errors[mask].mean())
        d = ""
        if base_bins and label in base_bins and base_bins[label] > 0:
            d = f"  ({(m - base_bins[label]) / base_bins[label] * 100:+.1f}%)"
        print(f"      {label}: MAE={m:.4f}%  n={mask.sum():,}{d}")


# -- Main -----------------------------------------------------------------


def main():
    try:
        import shap  # noqa: F401
    except ImportError:
        print("ERROR: 'shap' not installed. Run: pip install shap")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  MODEL 1B: SHAP Re-Pruning with Expanded Feature Candidates (29)")
    print("=" * 70)

    # -- Load data --
    print("\n  Loading data...")
    train, val, test, train_end, val_end = load_data()
    progress_test = test["trip_progress_fraction"].to_numpy()
    progress_val = val["trip_progress_fraction"].to_numpy()

    print(f"  Split: train<={train_end.date()}, val<={val_end.date()}, test>{FIXED_TEST_BOUNDARY.date()}")
    print(f"  Rows: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    for name, s in [("train", train), ("val", val), ("test", test)]:
        print(f"    {name}: {s['segment_id'].n_unique()} segments")

    # Verify all candidate features exist
    for f in CANDIDATE_FEATURES:
        if f not in train.columns:
            print(f"\n  ERROR: '{f}' missing from parquet. Re-run extraction.")
            sys.exit(1)
    print(f"\n  All {len(CANDIDATE_FEATURES)} candidate features verified.")

    # -- Production baseline --
    saved = load_prod_params()
    params, n_est = build_params(saved)
    prod_models = load_prod_models()

    base_mae, base_rmse, base_r2, _, base_errors = evaluate(prod_models, test, REALTIME_FEATURES)
    base_bins = bin_maes(base_errors, progress_test)

    print(f"\n  Production baseline (25 features):")
    print(f"    MAE={base_mae:.4f}%  RMSE={base_rmse:.4f}%  R2={base_r2:.4f}")
    print_bins(base_errors, progress_test)

    # -- Train on all 29 candidates --
    print(f"\n  Training on all {len(CANDIDATE_FEATURES)} candidates...")
    t0 = time.time()
    models_29 = train_ensemble(train, val, CANDIDATE_FEATURES, params, n_est)
    print(f"  Done ({time.time()-t0:.1f}s)")

    mae_29, _, r2_29, _, err_29 = evaluate(models_29, test, CANDIDATE_FEATURES)
    bins_29 = bin_maes(err_29, progress_test)
    print(f"  All-29 test: MAE={mae_29:.4f}% ({(mae_29-base_mae)/base_mae*100:+.2f}%)  R2={r2_29:.4f}")
    print_bins(err_29, progress_test, base_bins)

    # -- SHAP analysis --
    print(f"\n  Computing SHAP values on validation subsample...")
    import shap

    n_shap = min(3000, len(val))  # larger sample for more stable ranking
    rng = np.random.RandomState(42)
    idx = rng.choice(len(val), n_shap, replace=False)
    X_shap = val.select(CANDIDATE_FEATURES).to_numpy().astype(np.float32)[idx]
    dm_shap = xgb.DMatrix(X_shap, feature_names=CANDIDATE_FEATURES)

    all_shap = np.zeros((n_shap, len(CANDIDATE_FEATURES)))
    for m in models_29:
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(dm_shap)
        all_shap += np.abs(sv)
    all_shap /= len(models_29)

    mean_shap = all_shap.mean(axis=0)
    ranked_idx = np.argsort(mean_shap)[::-1]
    ranked_features = [CANDIDATE_FEATURES[i] for i in ranked_idx]

    print(f"\n  Feature importance ranking (mean |SHAP|, {n_shap} val samples):")
    print(f"  {'Rank':>4s}  {'Feature':<35s}  {'SHAP':>8s}  {'Source':>10s}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*8}  {'-'*10}")
    for i, fi in enumerate(ranked_idx):
        fname = CANDIDATE_FEATURES[fi]
        src = "NEW" if fname not in REALTIME_FEATURES else "prod"
        print(f"  {i+1:4d}  {fname:<35s}  {mean_shap[fi]:8.4f}  {src:>10s}")

    # -- Pruning sweep --
    n_values = [20, 22, 24, 25, 26, 27, 28, 29]
    results = []

    for n_keep in n_values:
        pruned = ranked_features[:n_keep]
        print(f"\n  {'-'*60}")
        print(f"  Top-{n_keep} features")
        print(f"  {'-'*60}")

        new_in = [f for f in pruned if f not in REALTIME_FEATURES]
        old_out = [f for f in REALTIME_FEATURES if f not in pruned]
        if new_in:
            print(f"    New features included: {', '.join(new_in)}")
        if old_out:
            print(f"    Old features dropped:  {', '.join(old_out)}")

        t0 = time.time()
        models = train_ensemble(train, val, pruned, params, n_est)
        elapsed = time.time() - t0

        mae, rmse, r2, preds, errors = evaluate(models, test, pruned)
        val_mae = evaluate(models, val, pruned)[0]
        bins = bin_maes(errors, progress_test)
        delta = (mae - base_mae) / base_mae * 100

        # Pass gate: overall MAE improves, no bin >20% degradation
        passed = True
        reasons = []
        if mae >= base_mae:
            passed = False
            reasons.append(f"Overall MAE did not improve ({mae:.4f} >= {base_mae:.4f})")
        for label, bv in base_bins.items():
            if label in bins and bv > 0:
                pct = (bins[label] - bv) / bv * 100
                if pct > 20:
                    passed = False
                    reasons.append(f"{label} degraded {pct:+.1f}%")

        print(f"    MAE:  {mae:.4f}% ({delta:+.2f}%)  RMSE: {rmse:.4f}%  R2: {r2:.4f}")
        print(f"    Val:  {val_mae:.4f}%  Gap: {mae-val_mae:+.4f}%")
        print_bins(errors, progress_test, base_bins)
        print(f"    Gate: {'PASS' if passed else 'FAIL'}  ({elapsed:.1f}s)")
        if not passed:
            for r in reasons:
                print(f"      - {r}")

        results.append({
            "n": n_keep,
            "features": pruned,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "val_mae": val_mae,
            "gap": mae - val_mae,
            "bins": bins,
            "delta": delta,
            "passed": passed,
            "reasons": reasons,
            "new_in": new_in,
            "old_out": old_out,
        })

    # -- Summary table --
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    hdr = (f"  {'N':>3s}  {'MAE':>7s}  {'Delta':>8s}  {'Val':>7s}  {'Gap':>8s}  "
           f"{'0-25%':>7s}  {'25-50%':>7s}  {'50-75%':>7s}  {'75-100%':>7s}  {'Gate':>6s}")
    print(f"\n{hdr}")
    sep = f"  {'-'*3}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}"
    print(sep)

    # Baseline row
    print(f"  {25:3d}  {base_mae:7.4f}  {'base':>8s}  {'--':>7s}  {'--':>8s}  "
          f"{base_bins.get('0-25%',0):7.4f}  {base_bins.get('25-50%',0):7.4f}  "
          f"{base_bins.get('50-75%',0):7.4f}  {base_bins.get('75-100%',0):7.4f}  {'--':>6s}")

    for r in results:
        b = r["bins"]
        print(
            f"  {r['n']:3d}  {r['mae']:7.4f}  {r['delta']:+7.2f}%  {r['val_mae']:7.4f}  "
            f"{r['gap']:+7.4f}  "
            f"{b.get('0-25%',0):7.4f}  {b.get('25-50%',0):7.4f}  "
            f"{b.get('50-75%',0):7.4f}  {b.get('75-100%',0):7.4f}  "
            f"{'PASS' if r['passed'] else 'FAIL':>6s}"
        )

    # Recommendation
    passed_results = [r for r in results if r["passed"]]
    if passed_results:
        best = min(passed_results, key=lambda r: r["mae"])
        print(f"\n  RECOMMENDATION: Top-{best['n']} features")
        print(f"    MAE: {best['mae']:.4f}% ({best['delta']:+.2f}% vs production)")
        print(f"    Val-test gap: {best['gap']:+.4f}%")
        print(f"    0-25% bin: {best['bins'].get('0-25%',0):.4f}% "
              f"({(best['bins'].get('0-25%',0)-base_bins.get('0-25%',0))/base_bins.get('0-25%',1)*100:+.1f}% vs prod)")
        if best["new_in"]:
            print(f"    New features added: {', '.join(best['new_in'])}")
        if best["old_out"]:
            print(f"    Old features dropped: {', '.join(best['old_out'])}")
        print(f"\n    Feature list ({best['n']}):")
        for i, f in enumerate(best["features"]):
            marker = " (NEW)" if f not in REALTIME_FEATURES else ""
            print(f"      {i+1:2d}. {f}{marker}")
    else:
        print(f"\n  NO PRUNING LEVEL PASSED GATE.")
        print(f"  Production 25-feature model remains optimal.")

    print()


if __name__ == "__main__":
    main()
