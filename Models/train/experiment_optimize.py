"""A/B experiments for Model 1A optimization.

Each experiment trains on the same rolling split, evaluates on the fixed test set,
and compares to the current production baseline.

Usage:
    python -m train.experiment_optimize --exp 1    # Distance sample weighting
    python -m train.experiment_optimize --exp 2    # Temperature interaction features
    python -m train.experiment_optimize --exp 3    # soc_per_km target
    python -m train.experiment_optimize --exp all   # Run 1-3 sequentially
    python -m train.experiment_optimize --exp 4    # Physics residual modeling
    python -m train.experiment_optimize --exp 5    # Target jittering
    python -m train.experiment_optimize --exp 6    # SHAP feature pruning
    python -m train.experiment_optimize --exp 7    # CatBoost comparison
    python -m train.experiment_optimize --exp all2  # Run 4-7 sequentially
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    ENSEMBLE_SEEDS,
    FEATURE_STORE_DIR,
    MAX_PASSENGERS,
    N_ENSEMBLE_SEEDS,
    TRIP_FEATURES,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    get_monotonic_constraint_tuple,
)
from data.augment import augment_trip_data
from train.train_v2 import df_to_dmatrix, ensemble_predict

# ── Constants ────────────────────────────────────────────────────────────
# Match the rolling split dates from the most recent retrain run.
# Update these if the feature store changes.
FIXED_TEST_BOUNDARY = datetime(2026, 2, 12, 23, 59, 59)
TRAIN_END = datetime(2026, 2, 4, 23, 59, 59)
VAL_END = datetime(2026, 2, 14, 23, 59, 59)


# ── Helpers ──────────────────────────────────────────────────────────────

def load_split() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load feature store and return train/val/fixed_test DataFrames."""
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train = df.filter(pl.col("departure_time") <= TRAIN_END)
    val = df.filter(
        (pl.col("departure_time") > TRAIN_END)
        & (pl.col("departure_time") <= VAL_END)
    )
    test = df.filter(pl.col("departure_time") > FIXED_TEST_BOUNDARY)
    return train, val, test


def load_production_baseline() -> float:
    """Load current production 1A test MAE."""
    meta_path = ARTIFACTS_DIR / "soc_trip_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return meta["metrics"]["Test"]["MAE"]


def evaluate_model(
    models: list[xgb.Booster],
    test_df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    use_log: bool = False,
    transform_back_fn=None,
) -> dict:
    """Evaluate ensemble on test set. Returns metrics dict."""
    X = test_df.select(features).to_numpy().astype(np.float32)
    dm = xgb.DMatrix(X, feature_names=features)
    preds = ensemble_predict(models, dm)

    if use_log:
        preds = np.expm1(preds)
    if transform_back_fn is not None:
        preds = transform_back_fn(preds, test_df)

    y_true = test_df[target].to_numpy().astype(np.float32)
    abs_errors = np.abs(y_true - preds)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    ss_res = np.sum((y_true - preds) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "preds": preds, "errors": abs_errors}


def evaluate_by_distance(errors: np.ndarray, test_df: pl.DataFrame) -> None:
    """Print per-distance-bucket MAE."""
    dist = test_df["route_distance_km"].to_numpy()
    buckets = [(0, 3, "0-3km"), (3, 6, "3-6km"), (6, 10, "6-10km"), (10, 20, "10-20km")]
    for lo, hi, label in buckets:
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() == 0:
            continue
        print(f"      {label}: MAE={errors[mask].mean():.4f}%  n={mask.sum()}")

    # Feb 24 specifically
    dates = test_df["departure_time"].dt.date().to_list()
    feb24_mask = np.array([d == datetime(2026, 2, 24).date() for d in dates])
    if feb24_mask.sum() > 0:
        print(f"      Feb-24: MAE={errors[feb24_mask].mean():.4f}%  n={feb24_mask.sum()}")


def load_production_config() -> dict:
    """Load production metadata and return training config dict."""
    meta_path = ARTIFACTS_DIR / "soc_trip_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    return {
        "saved_params": meta["best_params"],
        "use_log": meta.get("use_log_transform", False),
        "use_huber": meta.get("use_huber_loss", False),
        "huber_slope": meta.get("huber_slope"),
    }


def load_production_models() -> list[xgb.Booster]:
    """Load production ensemble from artifact JSON files."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        suffix = "" if seed == 42 else f"_seed{seed}"
        path = ARTIFACTS_DIR / f"soc_trip_model{suffix}.json"
        m = xgb.Booster()
        m.load_model(str(path))
        models.append(m)
    return models


def evaluate_val_test_gap(
    models: list[xgb.Booster],
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: list[str],
    target: str,
    *,
    use_log: bool = False,
    transform_back_fn=None,
) -> tuple[float, float, float]:
    """Evaluate on both val and test, return (val_mae, test_mae, gap)."""
    val_r = evaluate_model(
        models, val_df, features, target,
        use_log=use_log, transform_back_fn=transform_back_fn,
    )
    test_r = evaluate_model(
        models, test_df, features, target,
        use_log=use_log, transform_back_fn=transform_back_fn,
    )
    gap = test_r["MAE"] - val_r["MAE"]
    return val_r["MAE"], test_r["MAE"], gap


def get_distance_bucket_maes(
    errors: np.ndarray, test_df: pl.DataFrame,
) -> dict[str, float]:
    """Return per-distance-bucket MAEs as a dict."""
    dist = test_df["route_distance_km"].to_numpy()
    buckets = [(0, 3, "0-3km"), (3, 6, "3-6km"), (6, 10, "6-10km"), (10, 20, "10-20km")]
    result = {}
    for lo, hi, label in buckets:
        mask = (dist >= lo) & (dist < hi)
        if mask.sum() > 0:
            result[label] = float(errors[mask].mean())
    return result


def check_pass_gate(
    exp_errors: np.ndarray,
    baseline_errors: np.ndarray,
    test_df: pl.DataFrame,
    exp_mae: float,
    baseline_mae: float,
) -> tuple[bool, list[str]]:
    """Check if experiment passes: overall MAE improves AND no bucket degrades >20%.

    Returns (passed, reasons).
    """
    reasons = []
    if exp_mae >= baseline_mae:
        reasons.append(f"Overall MAE did not improve ({exp_mae:.4f} >= {baseline_mae:.4f})")

    exp_buckets = get_distance_bucket_maes(exp_errors, test_df)
    base_buckets = get_distance_bucket_maes(baseline_errors, test_df)

    for label, base_val in base_buckets.items():
        if label in exp_buckets and base_val > 0:
            pct_change = (exp_buckets[label] - base_val) / base_val * 100
            if pct_change > 20:
                reasons.append(
                    f"{label} degraded {pct_change:+.1f}% "
                    f"({base_val:.4f} -> {exp_buckets[label]:.4f})"
                )

    return len(reasons) == 0, reasons


def build_xgb_params(
    saved_params: dict,
    mono_tuple: tuple,
    use_huber: bool = False,
    huber_slope: float | None = None,
) -> dict:
    """Build XGBoost parameter dict from saved Optuna params."""
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
    return params


def train_ensemble(
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    params: dict,
    n_estimators: int,
    *,
    use_log: bool = False,
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    features: list[str] | None = None,
    train_df: pl.DataFrame | None = None,
    val_df: pl.DataFrame | None = None,
) -> list[xgb.Booster]:
    """Train N_ENSEMBLE_SEEDS models."""
    models = []
    for seed in ENSEMBLE_SEEDS:
        p = {**params, "seed": seed}

        if use_log:
            assert y_train is not None and y_val is not None
            assert features is not None and train_df is not None and val_df is not None
            y_aug_log = np.log1p(y_train)
            dtrain_log = xgb.DMatrix(
                train_df.select(features).to_numpy().astype(np.float32),
                label=y_aug_log,
                feature_names=features,
                weight=dtrain.get_weight() if len(dtrain.get_weight()) > 0 else None,
            )
            y_val_log = np.log1p(y_val)
            dval_log = xgb.DMatrix(
                val_df.select(features).to_numpy().astype(np.float32),
                label=y_val_log,
                feature_names=features,
            )
            model = xgb.train(
                p, dtrain_log,
                num_boost_round=n_estimators,
                evals=[(dval_log, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )
        else:
            model = xgb.train(
                p, dtrain,
                num_boost_round=n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )
        models.append(model)
    return models


# ── Experiment 1: Distance-Stratified Sample Weighting ───────────────────

def compute_distance_weights(df: pl.DataFrame) -> np.ndarray:
    """Compute sample weights proportional to route distance.

    Short trips (0-6km) dominate training data (85%).
    Weight longer trips more to compensate.
    """
    dist = df["route_distance_km"].to_numpy()
    weights = np.ones(len(dist), dtype=np.float32)
    weights[(dist >= 6) & (dist < 10)] = 3.0
    weights[(dist >= 10) & (dist < 15)] = 5.0
    weights[dist >= 15] = 8.0
    return weights


def run_experiment_1(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
) -> dict:
    """Experiment 1: Distance-stratified sample weighting."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Distance-Stratified Sample Weighting")
    print("=" * 65)

    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    # Augment
    print("  Augmenting training data...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)

    # Compute distance weights
    weights = compute_distance_weights(train_aug)
    print(f"  Weight distribution: 1x={np.sum(weights==1)}, 3x={np.sum(weights==3)}, "
          f"5x={np.sum(weights==5)}, 8x={np.sum(weights==8)}")

    # Build DMatrices with weights
    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dtrain_aug.set_weight(weights)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)

    best_params = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
    n_estimators = saved_params["n_estimators"]

    print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble with distance weights...")
    t0 = time.time()
    models = train_ensemble(
        dtrain_aug, dval, best_params, n_estimators,
        use_log=use_log, y_train=y_train_aug, y_val=y_val,
        features=TRIP_FEATURES, train_df=train_aug, val_df=val_df,
    )
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Evaluate
    result = evaluate_model(
        models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    print(f"\n  Results (fixed test, n={len(test_df)}):")
    print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {result['RMSE']:.4f}%")
    print(f"    R2:   {result['R2']:.4f}")
    delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(result["errors"], test_df)

    return {
        "name": "distance_weighting",
        "mae": result["MAE"],
        "rmse": result["RMSE"],
        "r2": result["R2"],
        "delta_pct": delta_pct,
        "models": models,
        "use_log": use_log,
        "use_huber": use_huber,
        "huber_slope": huber_slope,
        "params": saved_params,
        "kept": delta_pct < 0,
    }


# ── Experiment 2: Temperature Interaction Features ───────────────────────

TEMP_INTERACTION_FEATURES = [
    "temp_x_distance",
    "temp_x_speed_squared",
]


def add_temp_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add temperature interaction features to a DataFrame."""
    return df.with_columns([
        (pl.col("temperature_c") * pl.col("route_distance_km")).alias("temp_x_distance"),
        (pl.col("temperature_c") * pl.col("target_speed_squared")).alias("temp_x_speed_squared"),
    ])


def run_experiment_2(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
    *,
    use_weights: bool = False,
) -> dict:
    """Experiment 2: Temperature interaction features."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: Temperature Interaction Features")
    if use_weights:
        print("  (stacked on top of Experiment 1 distance weighting)")
    print("=" * 65)

    features = TRIP_FEATURES + TEMP_INTERACTION_FEATURES
    # No monotonic constraints on the new features
    mono_constraints = dict(TRIP_MONOTONIC_CONSTRAINTS)
    for f in TEMP_INTERACTION_FEATURES:
        mono_constraints[f] = 0
    mono_tuple = get_monotonic_constraint_tuple(features, mono_constraints)

    # Add features
    train_ext = add_temp_features(train_df)
    val_ext = add_temp_features(val_df)
    test_ext = add_temp_features(test_df)

    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    # Augment
    print("  Augmenting training data...")
    train_aug = augment_trip_data(train_ext, features, TRIP_TARGET, verbose=False)

    # Build DMatrices
    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, features, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_ext, features, TRIP_TARGET)

    if use_weights:
        weights = compute_distance_weights(train_aug)
        dtrain_aug.set_weight(weights)
        print(f"  Distance weights applied: 1x={np.sum(weights==1)}, 3x={np.sum(weights==3)}, "
              f"5x={np.sum(weights==5)}, 8x={np.sum(weights==8)}")

    best_params = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
    n_estimators = saved_params["n_estimators"]

    print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble ({len(features)} features)...")
    t0 = time.time()
    models = train_ensemble(
        dtrain_aug, dval, best_params, n_estimators,
        use_log=use_log, y_train=y_train_aug, y_val=y_val,
        features=features, train_df=train_aug, val_df=val_ext,
    )
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Evaluate
    result = evaluate_model(
        models, test_ext, features, TRIP_TARGET, use_log=use_log,
    )
    print(f"\n  Results (fixed test, n={len(test_df)}):")
    print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {result['RMSE']:.4f}%")
    print(f"    R2:   {result['R2']:.4f}")
    delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(result["errors"], test_df)

    return {
        "name": "temp_interactions" + ("_weighted" if use_weights else ""),
        "mae": result["MAE"],
        "rmse": result["RMSE"],
        "r2": result["R2"],
        "delta_pct": delta_pct,
        "models": models,
        "features": features,
        "mono_constraints": mono_constraints,
        "use_log": use_log,
        "use_huber": use_huber,
        "huber_slope": huber_slope,
        "params": saved_params,
        "kept": delta_pct < 0,
    }


# ── Experiment 3: Predict soc_per_km ─────────────────────────────────────

SOC_PER_KM_TARGET = "soc_per_km"


def add_soc_per_km(df: pl.DataFrame) -> pl.DataFrame:
    """Add soc_per_km = soc_delta / route_distance_km."""
    return df.with_columns(
        (pl.col(TRIP_TARGET) / pl.col("route_distance_km").clip(lower_bound=0.1))
        .alias(SOC_PER_KM_TARGET)
    )


def run_experiment_3(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
    *,
    features: list[str] | None = None,
    mono_constraints: dict | None = None,
    use_weights: bool = False,
) -> dict:
    """Experiment 3: Predict soc_per_km, convert back to soc_delta."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: Predict soc_per_km Instead of soc_delta")
    extras = []
    if use_weights:
        extras.append("distance weighting")
    if features and len(features) > len(TRIP_FEATURES):
        extras.append("temp interactions")
    if extras:
        print(f"  (stacked on: {', '.join(extras)})")
    print("=" * 65)

    if features is None:
        features = TRIP_FEATURES
    if mono_constraints is None:
        mono_constraints = dict(TRIP_MONOTONIC_CONSTRAINTS)

    mono_tuple = get_monotonic_constraint_tuple(features, mono_constraints)

    # Add soc_per_km target
    train_ext = add_soc_per_km(train_df)
    val_ext = add_soc_per_km(val_df)
    test_ext = add_soc_per_km(test_df)

    # If temp interaction features are in the feature list, add them
    if "temp_x_distance" in features:
        train_ext = add_temp_features(train_ext)
        val_ext = add_temp_features(val_ext)
        test_ext = add_temp_features(test_ext)

    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    # Augment
    print("  Augmenting training data...")
    train_aug = augment_trip_data(train_ext, features, SOC_PER_KM_TARGET, verbose=False)

    # Build DMatrices with soc_per_km target
    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, features, SOC_PER_KM_TARGET)
    dval, y_val = df_to_dmatrix(val_ext, features, SOC_PER_KM_TARGET)

    if use_weights:
        weights = compute_distance_weights(train_aug)
        dtrain_aug.set_weight(weights)

    best_params = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
    n_estimators = saved_params["n_estimators"]

    print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble (target: soc_per_km)...")
    t0 = time.time()
    models = train_ensemble(
        dtrain_aug, dval, best_params, n_estimators,
        use_log=use_log, y_train=y_train_aug, y_val=y_val,
        features=features, train_df=train_aug, val_df=val_ext,
    )
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Evaluate: predict soc_per_km, convert back to soc_delta
    def transform_back(preds_per_km, df):
        dist = df["route_distance_km"].to_numpy().astype(np.float32)
        return preds_per_km * dist

    result = evaluate_model(
        models, test_ext, features, TRIP_TARGET,
        use_log=use_log, transform_back_fn=transform_back,
    )
    print(f"\n  Results (fixed test, n={len(test_df)}, converted back to soc_delta %):")
    print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {result['RMSE']:.4f}%")
    print(f"    R2:   {result['R2']:.4f}")
    delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(result["errors"], test_df)

    return {
        "name": "soc_per_km" + ("_weighted" if use_weights else "")
              + ("_tempint" if "temp_x_distance" in features else ""),
        "mae": result["MAE"],
        "rmse": result["RMSE"],
        "r2": result["R2"],
        "delta_pct": delta_pct,
        "models": models,
        "features": features,
        "mono_constraints": mono_constraints,
        "use_log": use_log,
        "use_huber": use_huber,
        "huber_slope": huber_slope,
        "params": saved_params,
        "kept": delta_pct < 0,
    }


# ── Experiment 4: Physics Residual Modeling ───────────────────────────────


def fit_physics_model(
    train_df: pl.DataFrame,
) -> dict[str, float]:
    """Fit physics base model: E = k * dist * v^2 * (1 + α*pax/max_pax) * dir_factor.

    Returns dict with fitted parameters {k, alpha, beta}.
    dir_factor = beta if direction_encoded == 1, else 1.0.
    """
    dist = train_df["route_distance_km"].to_numpy().astype(np.float64)
    v2 = train_df["target_speed_squared"].to_numpy().astype(np.float64)
    pax = train_df["passengers_on_board"].to_numpy().astype(np.float64)
    direction = train_df["direction_encoded"].to_numpy().astype(np.float64)
    y = train_df[TRIP_TARGET].to_numpy().astype(np.float64)

    def physics_loss(params):
        k, alpha, beta = params
        dir_factor = np.where(direction == 1, beta, 1.0)
        pred = k * dist * v2 * (1.0 + alpha * pax / MAX_PASSENGERS) * dir_factor
        return np.mean((y - pred) ** 2)

    # Optimize with bounds to keep parameters physical
    result = minimize(
        physics_loss,
        x0=[0.01, 0.5, 2.0],  # initial guesses
        method="L-BFGS-B",
        bounds=[(1e-6, 1.0), (0.0, 5.0), (0.5, 5.0)],
    )
    k, alpha, beta = result.x
    return {"k": float(k), "alpha": float(alpha), "beta": float(beta)}


def physics_predict(
    df: pl.DataFrame, params: dict[str, float],
) -> np.ndarray:
    """Compute E_physics for a DataFrame using fitted parameters."""
    dist = df["route_distance_km"].to_numpy().astype(np.float64)
    v2 = df["target_speed_squared"].to_numpy().astype(np.float64)
    pax = df["passengers_on_board"].to_numpy().astype(np.float64)
    direction = df["direction_encoded"].to_numpy().astype(np.float64)

    dir_factor = np.where(direction == 1, params["beta"], 1.0)
    return params["k"] * dist * v2 * (1.0 + params["alpha"] * pax / MAX_PASSENGERS) * dir_factor


def run_experiment_4(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
) -> dict:
    """Experiment 4: Physics residual modeling."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 4: Physics Residual Modeling")
    print("=" * 65)

    # Step 1: Fit physics model on raw training data
    print("  Fitting physics base model (k, alpha, beta)...")
    phys_params = fit_physics_model(train_df)
    print(f"    k={phys_params['k']:.6f}, alpha={phys_params['alpha']:.3f}, "
          f"beta={phys_params['beta']:.3f}")

    # Evaluate physics-only model
    phys_test_pred = physics_predict(test_df, phys_params)
    y_test = test_df[TRIP_TARGET].to_numpy().astype(np.float32)
    phys_mae = float(np.mean(np.abs(y_test - phys_test_pred)))
    print(f"    Physics-only test MAE: {phys_mae:.4f}%")

    # Step 2: Compute residuals for all splits
    train_residual = train_df.with_columns(
        (pl.col(TRIP_TARGET) - pl.Series("_phys", physics_predict(train_df, phys_params)))
        .alias("_residual")
    )
    val_residual = val_df.with_columns(
        (pl.col(TRIP_TARGET) - pl.Series("_phys", physics_predict(val_df, phys_params)))
        .alias("_residual")
    )

    # Step 3: Augment training data (augment original features, then recompute residuals)
    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    print("  Augmenting training data...")
    # Augment with _residual as target so augmentation perturbs it correctly
    train_aug = augment_trip_data(
        train_residual, TRIP_FEATURES, "_residual", verbose=False,
    )

    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)
    best_params = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
    n_estimators = saved_params["n_estimators"]

    # Step 4: Train XGBoost on residuals
    dtrain_aug, y_train_res = df_to_dmatrix(train_aug, TRIP_FEATURES, "_residual")
    dval, y_val_res = df_to_dmatrix(val_residual, TRIP_FEATURES, "_residual")

    print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble on residuals...")
    t0 = time.time()
    models = train_ensemble(dtrain_aug, dval, best_params, n_estimators)
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Step 5: Final prediction = physics + XGBoost_residual
    X_test = test_df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    dm_test = xgb.DMatrix(X_test, feature_names=TRIP_FEATURES)
    xgb_residual_pred = ensemble_predict(models, dm_test)
    final_pred = phys_test_pred.astype(np.float32) + xgb_residual_pred

    abs_errors = np.abs(y_test - final_pred)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_test - final_pred) ** 2)))
    ss_res = np.sum((y_test - final_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    delta_pct = (mae - baseline_mae) / baseline_mae * 100

    print(f"\n  Results (fixed test, n={len(test_df)}):")
    print(f"    MAE:  {mae:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {rmse:.4f}%")
    print(f"    R2:   {r2:.4f}")
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(abs_errors, test_df)

    # Val-test gap
    X_val = val_df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    dm_val = xgb.DMatrix(X_val, feature_names=TRIP_FEATURES)
    val_xgb_res = ensemble_predict(models, dm_val)
    val_final = physics_predict(val_df, phys_params).astype(np.float32) + val_xgb_res
    y_val = val_df[TRIP_TARGET].to_numpy().astype(np.float32)
    val_mae = float(np.mean(np.abs(y_val - val_final)))
    gap = mae - val_mae
    print(f"\n  Val MAE: {val_mae:.4f}% | Test MAE: {mae:.4f}% | Gap: {gap:+.4f}%")

    # Pass gate
    baseline_models = load_production_models()
    base_result = evaluate_model(
        baseline_models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    passed, reasons = check_pass_gate(
        abs_errors, base_result["errors"], test_df, mae, baseline_mae,
    )
    if passed:
        print("  PASS GATE: PASSED")
    else:
        print("  PASS GATE: FAILED")
        for r in reasons:
            print(f"    - {r}")

    return {
        "name": "physics_residual",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "delta_pct": delta_pct,
        "physics_params": phys_params,
        "physics_only_mae": phys_mae,
        "val_mae": val_mae,
        "gap": gap,
        "passed": passed,
        "kept": passed,
    }


# ── Experiment 5: Target Jittering ──────────────────────────────────────


def run_experiment_5(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
) -> dict:
    """Experiment 5: Add uniform noise to training targets to break quantization."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 5: Target Jittering")
    print("=" * 65)

    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    # Add uniform noise to training targets BEFORE augmentation
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, size=len(train_df))
    train_jittered = train_df.with_columns(
        (pl.col(TRIP_TARGET) + pl.Series("_jitter", jitter)).clip(lower_bound=0).alias(TRIP_TARGET)
    )
    print(f"  Added Uniform(-0.25, +0.25) jitter to {len(train_df)} training targets")

    # Augment jittered data
    print("  Augmenting training data...")
    train_aug = augment_trip_data(train_jittered, TRIP_FEATURES, TRIP_TARGET, verbose=False)

    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)
    best_params = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
    n_estimators = saved_params["n_estimators"]

    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)

    print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble with jittered targets...")
    t0 = time.time()
    models = train_ensemble(
        dtrain_aug, dval, best_params, n_estimators,
        use_log=use_log, y_train=y_train_aug, y_val=y_val,
        features=TRIP_FEATURES, train_df=train_aug, val_df=val_df,
    )
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Evaluate on original (non-jittered) test targets
    result = evaluate_model(
        models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100

    print(f"\n  Results (fixed test, n={len(test_df)}):")
    print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {result['RMSE']:.4f}%")
    print(f"    R2:   {result['R2']:.4f}")
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(result["errors"], test_df)

    # Val-test gap
    val_r = evaluate_model(
        models, val_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    gap = result["MAE"] - val_r["MAE"]
    print(f"\n  Val MAE: {val_r['MAE']:.4f}% | Test MAE: {result['MAE']:.4f}% | Gap: {gap:+.4f}%")

    # Pass gate
    baseline_models = load_production_models()
    base_result = evaluate_model(
        baseline_models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    passed, reasons = check_pass_gate(
        result["errors"], base_result["errors"], test_df, result["MAE"], baseline_mae,
    )
    if passed:
        print("  PASS GATE: PASSED")
    else:
        print("  PASS GATE: FAILED")
        for r in reasons:
            print(f"    - {r}")

    return {
        "name": "target_jittering",
        "mae": result["MAE"],
        "rmse": result["RMSE"],
        "r2": result["R2"],
        "delta_pct": delta_pct,
        "val_mae": val_r["MAE"],
        "gap": gap,
        "passed": passed,
        "kept": passed,
    }


# ── Experiment 6: SHAP Feature Pruning ──────────────────────────────────


def run_experiment_6(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
) -> dict:
    """Experiment 6: SHAP-based feature pruning at N=15, 20, 25."""
    try:
        import shap
    except ImportError:
        print("\n  ERROR: 'shap' not installed. Run: pip install shap")
        return {"name": "shap_pruning", "mae": float("inf"), "delta_pct": float("inf"), "kept": False}

    print("\n" + "=" * 65)
    print("  EXPERIMENT 6: SHAP Feature Pruning")
    print("=" * 65)

    cfg = load_production_config()
    saved_params = cfg["saved_params"]
    use_log, use_huber, huber_slope = cfg["use_log"], cfg["use_huber"], cfg["huber_slope"]

    # Load production models and compute SHAP on validation set
    print("  Loading production models for SHAP analysis...")
    prod_models = load_production_models()

    X_val = val_df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    dm_val_shap = xgb.DMatrix(X_val, feature_names=TRIP_FEATURES)

    # Average SHAP values across ensemble members
    print("  Computing SHAP values on validation set...")
    all_shap = np.zeros((len(X_val), len(TRIP_FEATURES)))
    for model in prod_models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(dm_val_shap)
        all_shap += np.abs(sv)
    all_shap /= len(prod_models)

    # Rank features by mean |SHAP|
    mean_shap = all_shap.mean(axis=0)
    ranked_indices = np.argsort(mean_shap)[::-1]
    ranked_features = [TRIP_FEATURES[i] for i in ranked_indices]

    print("  Feature importance ranking (mean |SHAP|):")
    for i, idx in enumerate(ranked_indices):
        print(f"    {i+1:2d}. {TRIP_FEATURES[idx]:<35s} {mean_shap[idx]:.4f}")

    # Test pruned models at N=15, 20, 25
    n_values = [15, 20, 25]
    best_result = None

    # Get baseline per-distance errors for pass gate
    base_result = evaluate_model(
        prod_models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )

    for n_keep in n_values:
        print(f"\n  --- Pruning to top {n_keep} features ---")
        pruned_features = ranked_features[:n_keep]

        # Rebuild monotonic constraints for retained features
        pruned_mono = {f: TRIP_MONOTONIC_CONSTRAINTS.get(f, 0) for f in pruned_features}
        mono_tuple = get_monotonic_constraint_tuple(pruned_features, pruned_mono)

        # Augment
        train_aug = augment_trip_data(train_df, pruned_features, TRIP_TARGET, verbose=False)

        dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, pruned_features, TRIP_TARGET)
        dval, y_val = df_to_dmatrix(val_df, pruned_features, TRIP_TARGET)

        best_xgb = build_xgb_params(saved_params, mono_tuple, use_huber, huber_slope)
        n_estimators = saved_params["n_estimators"]

        print(f"  Training {N_ENSEMBLE_SEEDS}-seed ensemble ({n_keep} features)...")
        t0 = time.time()
        models = train_ensemble(
            dtrain_aug, dval, best_xgb, n_estimators,
            use_log=use_log, y_train=y_train_aug, y_val=y_val,
            features=pruned_features, train_df=train_aug, val_df=val_df,
        )
        elapsed = time.time() - t0

        result = evaluate_model(
            models, test_df, pruned_features, TRIP_TARGET, use_log=use_log,
        )
        delta_pct = (result["MAE"] - baseline_mae) / baseline_mae * 100

        print(f"  Training done ({elapsed:.1f}s)")
        print(f"    MAE:  {result['MAE']:.4f}% (baseline: {baseline_mae:.4f}%)")
        print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")
        evaluate_by_distance(result["errors"], test_df)

        # Val-test gap
        val_r = evaluate_model(
            models, val_df, pruned_features, TRIP_TARGET, use_log=use_log,
        )
        gap = result["MAE"] - val_r["MAE"]
        print(f"    Val MAE: {val_r['MAE']:.4f}% | Gap: {gap:+.4f}%")

        passed, reasons = check_pass_gate(
            result["errors"], base_result["errors"], test_df, result["MAE"], baseline_mae,
        )
        if passed:
            print(f"    PASS GATE: PASSED")
        else:
            print(f"    PASS GATE: FAILED")
            for r in reasons:
                print(f"      - {r}")

        if best_result is None or result["MAE"] < best_result["mae"]:
            best_result = {
                "n_features": n_keep,
                "features": pruned_features,
                "mae": result["MAE"],
                "rmse": result["RMSE"],
                "r2": result["R2"],
                "delta_pct": delta_pct,
                "val_mae": val_r["MAE"],
                "gap": gap,
                "passed": passed,
                "errors": result["errors"],
            }

    # Summary
    print(f"\n  Best pruned model: top-{best_result['n_features']} features")
    print(f"    MAE:  {best_result['mae']:.4f}%")
    print(f"    Delta: {best_result['delta_pct']:+.2f}%")

    return {
        "name": f"shap_pruning_top{best_result['n_features']}",
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "r2": best_result["r2"],
        "delta_pct": best_result["delta_pct"],
        "val_mae": best_result["val_mae"],
        "gap": best_result["gap"],
        "n_features": best_result["n_features"],
        "pruned_features": best_result["features"],
        "passed": best_result["passed"],
        "kept": best_result["passed"],
    }


# ── Experiment 7: CatBoost Comparison ───────────────────────────────────


def run_experiment_7(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    baseline_mae: float,
) -> dict:
    """Experiment 7: CatBoost with Optuna + 5-seed ensemble."""
    try:
        import catboost as cb
        import optuna
    except ImportError:
        print("\n  ERROR: 'catboost' not installed. Run: pip install catboost")
        return {"name": "catboost", "mae": float("inf"), "delta_pct": float("inf"), "kept": False}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("\n" + "=" * 65)
    print("  EXPERIMENT 7: CatBoost Comparison")
    print("=" * 65)

    cfg = load_production_config()
    use_log = cfg["use_log"]

    # Augment
    print("  Augmenting training data...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)

    X_train = train_aug.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    y_train = train_aug[TRIP_TARGET].to_numpy().astype(np.float32)
    X_val = val_df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    y_val = val_df[TRIP_TARGET].to_numpy().astype(np.float32)
    X_test = test_df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    y_test = test_df[TRIP_TARGET].to_numpy().astype(np.float32)

    # Build monotonic constraints for CatBoost: feature_name:value format
    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)
    # CatBoost accepts "FeatureName1:1,FeatureName2:-1" (only non-zero)
    cb_mono_parts = []
    for feat, val in zip(TRIP_FEATURES, mono_tuple):
        if val != 0:
            cb_mono_parts.append(f"{feat}:{val}")
    cb_mono = ",".join(cb_mono_parts) if cb_mono_parts else None

    # Optuna search
    N_TRIALS = 150

    def _cb_params(seed: int = 42, **overrides) -> dict:
        """Build CatBoost params dict with optional monotonic constraints."""
        p = {
            "loss_function": "RMSE",
            "eval_metric": "MAE",
            "verbose": 0,
            "random_seed": seed,
            "early_stopping_rounds": 50,
        }
        if cb_mono:
            p["monotone_constraints"] = cb_mono
        p.update(overrides)
        return p

    def objective(trial):
        params = _cb_params(
            iterations=trial.suggest_int("iterations", 200, 5000),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 30),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        )

        train_pool = cb.Pool(X_train, y_train, feature_names=TRIP_FEATURES)
        val_pool = cb.Pool(X_val, y_val, feature_names=TRIP_FEATURES)

        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)

        preds = model.predict(X_val)
        return float(np.mean(np.abs(y_val - preds)))

    print(f"  Running Optuna ({N_TRIALS} trials)...")
    t0 = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    optuna_elapsed = time.time() - t0
    print(f"  Optuna done ({optuna_elapsed:.1f}s). Best val MAE: {study.best_value:.4f}%")

    best_p = study.best_params
    print(f"  Best params: depth={best_p['depth']}, lr={best_p['learning_rate']:.4f}, "
          f"iter={best_p['iterations']}, l2={best_p['l2_leaf_reg']:.4f}")

    # Train 5-seed ensemble with best params
    print(f"  Training {N_ENSEMBLE_SEEDS}-seed CatBoost ensemble...")
    t0 = time.time()
    cb_models = []
    for seed in ENSEMBLE_SEEDS:
        params = _cb_params(seed=seed, **best_p)
        train_pool = cb.Pool(X_train, y_train, feature_names=TRIP_FEATURES)
        val_pool = cb.Pool(X_val, y_val, feature_names=TRIP_FEATURES)
        model = cb.CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        cb_models.append(model)
    elapsed = time.time() - t0
    print(f"  Training done ({elapsed:.1f}s)")

    # Ensemble predict
    test_preds = np.column_stack([m.predict(X_test) for m in cb_models]).mean(axis=1)
    abs_errors = np.abs(y_test - test_preds)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_test - test_preds) ** 2)))
    ss_res = np.sum((y_test - test_preds) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    delta_pct = (mae - baseline_mae) / baseline_mae * 100

    print(f"\n  Results (fixed test, n={len(test_df)}):")
    print(f"    MAE:  {mae:.4f}% (baseline: {baseline_mae:.4f}%)")
    print(f"    RMSE: {rmse:.4f}%")
    print(f"    R2:   {r2:.4f}")
    print(f"    Delta: {delta_pct:+.2f}% {'BETTER' if delta_pct < 0 else 'WORSE'}")

    print("\n  Per-distance breakdown:")
    evaluate_by_distance(abs_errors, test_df)

    # Val-test gap
    val_preds = np.column_stack([m.predict(X_val) for m in cb_models]).mean(axis=1)
    val_mae = float(np.mean(np.abs(y_val - val_preds)))
    gap = mae - val_mae
    print(f"\n  Val MAE: {val_mae:.4f}% | Test MAE: {mae:.4f}% | Gap: {gap:+.4f}%")

    # Pass gate
    prod_models = load_production_models()
    base_result = evaluate_model(
        prod_models, test_df, TRIP_FEATURES, TRIP_TARGET, use_log=use_log,
    )
    passed, reasons = check_pass_gate(
        abs_errors, base_result["errors"], test_df, mae, baseline_mae,
    )
    if passed:
        print("  PASS GATE: PASSED")
    else:
        print("  PASS GATE: FAILED")
        for r in reasons:
            print(f"    - {r}")

    return {
        "name": "catboost",
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "delta_pct": delta_pct,
        "val_mae": val_mae,
        "gap": gap,
        "best_params": best_p,
        "passed": passed,
        "kept": passed,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="A/B optimization experiments for Model 1A",
    )
    parser.add_argument(
        "--exp",
        choices=["1", "2", "3", "4", "5", "6", "7", "all", "all2"],
        default="all",
        help="Which experiment to run (all=1-3, all2=4-7)",
    )
    args = parser.parse_args()

    run_all = args.exp == "all"
    run_all2 = args.exp == "all2"
    if run_all:
        experiments = ["1", "2", "3"]
    elif run_all2:
        experiments = ["4", "5", "6", "7"]
    else:
        experiments = [args.exp]

    # Load data
    print("\nLoading data...")
    train_df, val_df, test_df = load_split()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Fixed Test: {len(test_df)}")

    baseline_mae = load_production_baseline()
    print(f"  Production baseline MAE: {baseline_mae:.4f}%")

    results = []
    current_baseline = baseline_mae
    use_weights = False
    features = None
    mono_constraints = None

    # Experiment 1
    if "1" in experiments:
        r1 = run_experiment_1(train_df, val_df, test_df, current_baseline)
        results.append(r1)
        if r1["kept"] and run_all:
            use_weights = True
            current_baseline = r1["mae"]
            print(f"\n  >>> KEEPING Experiment 1. New baseline: {current_baseline:.4f}%")
        elif run_all:
            print(f"\n  >>> DISCARDING Experiment 1 (no improvement)")

    # Experiment 2
    if "2" in experiments:
        r2 = run_experiment_2(
            train_df, val_df, test_df, current_baseline,
            use_weights=use_weights,
        )
        results.append(r2)
        if r2["kept"] and run_all:
            features = r2["features"]
            mono_constraints = r2["mono_constraints"]
            current_baseline = r2["mae"]
            print(f"\n  >>> KEEPING Experiment 2. New baseline: {current_baseline:.4f}%")
        elif run_all:
            print(f"\n  >>> DISCARDING Experiment 2 (no improvement)")

    # Experiment 3
    if "3" in experiments:
        r3 = run_experiment_3(
            train_df, val_df, test_df, current_baseline,
            features=features,
            mono_constraints=mono_constraints,
            use_weights=use_weights,
        )
        results.append(r3)
        if r3["kept"] and run_all:
            current_baseline = r3["mae"]
            print(f"\n  >>> KEEPING Experiment 3. New baseline: {current_baseline:.4f}%")
        elif run_all:
            print(f"\n  >>> DISCARDING Experiment 3 (no improvement)")

    # ── Round 2 experiments (independent, not stacked) ──

    # Experiment 4: Physics Residual
    if "4" in experiments:
        r4 = run_experiment_4(train_df, val_df, test_df, baseline_mae)
        results.append(r4)

    # Experiment 5: Target Jittering
    if "5" in experiments:
        r5 = run_experiment_5(train_df, val_df, test_df, baseline_mae)
        results.append(r5)

    # Experiment 6: SHAP Pruning
    if "6" in experiments:
        r6 = run_experiment_6(train_df, val_df, test_df, baseline_mae)
        results.append(r6)

    # Experiment 7: CatBoost
    if "7" in experiments:
        r7 = run_experiment_7(train_df, val_df, test_df, baseline_mae)
        results.append(r7)

    # Summary
    print("\n" + "=" * 65)
    print("  EXPERIMENT SUMMARY")
    print("=" * 65)
    print(f"  {'Experiment':<40s} {'MAE':>8s} {'Delta':>8s} {'Pass?':>6s}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*6}")
    print(f"  {'Production baseline':<40s} {baseline_mae:>7.4f}%")
    for r in results:
        status = "YES" if r["kept"] else "NO"
        print(f"  {r['name']:<40s} {r['mae']:>7.4f}% {r['delta_pct']:>+7.2f}% {status:>6s}")

    # For round 2 (independent), report best passing experiment
    if run_all2:
        passing = [r for r in results if r.get("passed", False)]
        if passing:
            best = min(passing, key=lambda r: r["mae"])
            print(f"\n  Best passing experiment: {best['name']} ({best['mae']:.4f}%)")
            print(f"  Improvement: {best['delta_pct']:+.2f}%")
        else:
            print(f"\n  No experiments passed the gate.")
            print(f"  Model 1A is at optimization ceiling for current dataset size.")
    else:
        # Round 1 logic: track cumulative baseline
        final = current_baseline
        print(f"  {'Final':<40s} {final:>7.4f}%")
        total_improvement = (final - baseline_mae) / baseline_mae * 100
        print(f"\n  Total improvement: {total_improvement:+.2f}%")

    print("=" * 65)


if __name__ == "__main__":
    main()
