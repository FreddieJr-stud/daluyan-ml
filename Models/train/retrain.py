"""Retraining script for M/B Dalaray XGBoost models.

Re-extracts features from DuckDB/parquets, retrains with rolling temporal
splits, benchmarks against the original fixed test set, gates on regression,
and updates the RPi5 bundle.

Usage:
    python -m train.retrain                   # Quick retrain, all 3 models
    python -m train.retrain --model 1a        # Quick retrain Model 1A only
    python -m train.retrain --model 1b        # Quick retrain Model 1B only
    python -m train.retrain --model 2         # Quick retrain Model 2 only
    python -m train.retrain --full-tune       # Full Optuna (300 trials)
    python -m train.retrain --dry-run         # Extract + evaluate only, no deploy
    python -m train.retrain --no-extract      # Skip re-extraction
    python -m train.retrain --force-deploy    # Deploy even if regression gate HALTs
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANOMALY_FEATURES,
    ANOMALY_TARGETS,
    ARTIFACTS_DIR,
    ENSEMBLE_SEEDS,
    FEATURE_STORE_DIR,
    MODELS_ROOT,
    N_ENSEMBLE_SEEDS,
    OPTUNA_CONFIG,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRIP_FEATURES,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    get_monotonic_constraint_tuple,
)
from data.augment import augment_trip_data
from evaluate.evaluate_anomaly import synthetic_injection_test
from evaluate.evaluate_soc import compute_metrics, print_metrics
from train.train_v2 import (
    _anomaly_optuna_objective,
    _calibrate_conformal,
    _compute_scores_from_array,
    _realtime_optuna_objective,
    _train_quantile_models,
    _try_huber_loss_trip,
    _try_log_transform_trip,
    _trip_optuna_objective,
    df_to_dmatrix,
    ensemble_predict,
)

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────
BACKUPS_DIR = ARTIFACTS_DIR / "backups"
RPI5_BUNDLE = MODELS_ROOT / "rpi5_bundle"
RPI5_MODELS = RPI5_BUNDLE / "models"
RPI5_CONFIG = RPI5_BUNDLE / "config"
RPI5_TESTDATA = RPI5_BUNDLE / "testdata"
RETRAIN_LOG = MODELS_ROOT / "RETRAIN_LOG.md"

# Fixed test boundary (original thesis benchmark)
FIXED_TEST_BOUNDARY = "2026-02-12"


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════

def re_extract_features(models: list[str]) -> None:
    """Run the 3 extraction scripts via subprocess."""
    scripts = []
    if "1a" in models:
        scripts.append(("data.extract_trip_features", "trip"))
    if "1b" in models:
        scripts.append(("data.extract_realtime_features", "realtime"))
    if "2" in models:
        scripts.append(("data.extract_anomaly_features", "anomaly"))

    for module, label in scripts:
        print(f"\n  Extracting {label} features...")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(MODELS_ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  ERROR extracting {label}:")
            print(result.stderr[-500:] if result.stderr else "(no stderr)")
            sys.exit(1)
        print(f"  {label} extraction done ({elapsed:.0f}s)")


# ═══════════════════════════════════════════════════════════════════════════
# Splitting
# ═══════════════════════════════════════════════════════════════════════════

def compute_rolling_split_dates(
    df: pl.DataFrame,
) -> tuple[datetime, datetime]:
    """Compute 80/10/10 split boundaries from unique dates in the data.

    Splits by unique *dates* (not rows) to avoid splitting a day's trips
    across train/val.

    Returns (train_end, val_end) as datetimes.
    """
    dates = sorted(df["departure_time"].dt.date().unique().to_list())
    n_dates = len(dates)

    if n_dates < 3:
        raise ValueError(f"Only {n_dates} unique dates — cannot split")

    train_end_idx = int(n_dates * 0.8) - 1
    val_end_idx = int(n_dates * 0.9) - 1

    # Ensure at least 1 date in val and test
    train_end_idx = min(train_end_idx, n_dates - 3)
    val_end_idx = min(val_end_idx, n_dates - 2)
    val_end_idx = max(val_end_idx, train_end_idx + 1)

    train_end = datetime(
        dates[train_end_idx].year,
        dates[train_end_idx].month,
        dates[train_end_idx].day,
        23, 59, 59,
    )
    val_end = datetime(
        dates[val_end_idx].year,
        dates[val_end_idx].month,
        dates[val_end_idx].day,
        23, 59, 59,
    )

    return train_end, val_end


def rolling_temporal_split(
    df: pl.DataFrame,
    train_end: datetime,
    val_end: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data using custom date boundaries."""
    train = df.filter(pl.col("departure_time") <= train_end)
    val = df.filter(
        (pl.col("departure_time") > train_end)
        & (pl.col("departure_time") <= val_end)
    )
    test = df.filter(pl.col("departure_time") > val_end)
    return train, val, test


def get_fixed_test_set(df: pl.DataFrame) -> pl.DataFrame:
    """Return the fixed test set (everything after the original thesis boundary)."""
    boundary = datetime.fromisoformat(FIXED_TEST_BOUNDARY)
    return df.filter(pl.col("departure_time") > boundary)


# ═══════════════════════════════════════════════════════════════════════════
# Metadata Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_production_metadata(model_name: str) -> dict | None:
    """Load existing production metadata JSON."""
    name_map = {
        "1a": "soc_trip_metadata.json",
        "1b": "soc_realtime_metadata.json",
        "2": "anomaly_metadata.json",
    }
    path = ARTIFACTS_DIR / name_map[model_name]
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_old_segment_count(model_name: str) -> int | None:
    """Get segment count from existing metadata."""
    meta = load_production_metadata(model_name)
    if meta is None:
        return None
    splits = meta.get("splits", {})
    if model_name == "1b":
        return sum(splits.get(k, 0) for k in ["train_segments", "val_segments", "test_segments"])
    else:
        return sum(splits.get(k, 0) for k in ["train", "val", "test"])


def get_production_test_mae(model_name: str) -> float | None:
    """Get the fixed test MAE from production metadata."""
    meta = load_production_metadata(model_name)
    if meta is None:
        return None
    if model_name in ("1a", "1b"):
        return meta.get("metrics", {}).get("Test", {}).get("MAE")
    else:
        return meta.get("injection_test", {}).get("detection_rate")


# ═══════════════════════════════════════════════════════════════════════════
# Quick Retrain (reuse saved hyperparameters)
# ═══════════════════════════════════════════════════════════════════════════

def quick_retrain_trip(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[list[xgb.Booster], dict, bool, bool, float | None]:
    """Quick retrain Model 1A: use saved params, train ensemble only.

    Returns (models, best_params_dict, use_log, use_huber, huber_slope).
    """
    meta = load_production_metadata("1a")
    if meta is None:
        raise RuntimeError("No existing trip metadata — run --full-tune first")

    saved_params = meta["best_params"]
    use_log = meta.get("use_log_transform", False)
    use_huber = meta.get("use_huber_loss", False)
    huber_slope = meta.get("huber_slope")

    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    # Augment training data
    print("    Augmenting training data...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)

    best_params = {
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
        best_params["huber_slope"] = huber_slope

    best_n_estimators = saved_params["n_estimators"]

    # Build DMatrices
    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)

    # Train ensemble
    print(f"    Training {N_ENSEMBLE_SEEDS}-seed ensemble...")
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        params = {**best_params, "seed": seed}

        if use_log:
            y_aug_log = np.log1p(y_train_aug)
            dtrain_log = xgb.DMatrix(
                train_aug.select(TRIP_FEATURES).to_numpy().astype(np.float32),
                label=y_aug_log, feature_names=TRIP_FEATURES,
            )
            y_val_log = np.log1p(y_val)
            dval_log = xgb.DMatrix(
                val_df.select(TRIP_FEATURES).to_numpy().astype(np.float32),
                label=y_val_log, feature_names=TRIP_FEATURES,
            )
            model = xgb.train(
                params, dtrain_log,
                num_boost_round=best_n_estimators,
                evals=[(dval_log, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )
        else:
            model = xgb.train(
                params, dtrain_aug,
                num_boost_round=best_n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )

        ensemble_models.append(model)
        print(f"      Seed {seed}: {model.num_boosted_rounds()} trees")

    return ensemble_models, saved_params, use_log, use_huber, huber_slope


def quick_retrain_realtime(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[list[xgb.Booster], dict]:
    """Quick retrain Model 1B: use saved params, train ensemble only.

    Returns (models, best_params_dict).
    """
    meta = load_production_metadata("1b")
    if meta is None:
        raise RuntimeError("No existing realtime metadata — run --full-tune first")

    saved_params = meta["best_params"]

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
    best_n_estimators = saved_params["n_estimators"]

    dtrain, y_train = df_to_dmatrix(train_df, REALTIME_FEATURES, REALTIME_TARGET)
    dval, y_val = df_to_dmatrix(val_df, REALTIME_FEATURES, REALTIME_TARGET)

    print(f"    Training {N_ENSEMBLE_SEEDS}-seed ensemble...")
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        params = {**best_params, "seed": seed}
        model = xgb.train(
            params, dtrain,
            num_boost_round=best_n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        ensemble_models.append(model)
        print(f"      Seed {seed}: {model.num_boosted_rounds()} trees")

    return ensemble_models, saved_params


def quick_retrain_anomaly(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[dict[str, xgb.Booster], dict[str, float], dict[str, float], dict, dict]:
    """Quick retrain Model 2: use saved per-target params if available,
    otherwise 50-trial Optuna per target.

    Returns (models, feature_means, feature_stds, thresholds, per_target_best_params).
    """
    meta = load_production_metadata("2")
    per_target_params = None
    if meta is not None:
        per_target_params = meta.get("per_target_best_params")

    # Compute feature statistics from training data
    feature_means = {}
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_means[feat] = float(np.mean(col))
        feature_stds[feat] = float(max(np.std(col), 0.01))

    models: dict[str, xgb.Booster] = {}
    new_per_target_params = {}

    for target_col in ANOMALY_TARGETS:
        print(f"    Reconstruction: {target_col}")
        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]

        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        if per_target_params and target_col in per_target_params:
            # Use saved params — zero Optuna
            saved = per_target_params[target_col]
            print(f"      Using saved params (0 Optuna trials)")
        else:
            # Run reduced 50-trial Optuna
            n_trials = 50
            print(f"      Running {n_trials}-trial Optuna...")
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
            )
            study.optimize(
                lambda trial: _anomaly_optuna_objective(trial, dtrain, dval),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            saved = study.best_trial.params
            print(f"      Best val MAE: {study.best_value:.4f}")

        new_per_target_params[target_col] = saved

        final_params = {
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
            final_params, dtrain,
            num_boost_round=saved["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )
        models[target_col] = model
        print(f"      Trees: {model.num_boosted_rounds()}")

    # Calibrate thresholds on validation data
    def score_fn(data):
        return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)

    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    val_scores = score_fn(val_data)

    thresholds = {
        "p95": float(np.percentile(val_scores, 95)),
        "p97": float(np.percentile(val_scores, 97)),
        "p99": float(np.percentile(val_scores, 99)),
    }
    print(f"    Thresholds: p95={thresholds['p95']:.6f} p97={thresholds['p97']:.6f} "
          f"p99={thresholds['p99']:.6f}")

    return models, feature_means, feature_stds, thresholds, new_per_target_params


# ═══════════════════════════════════════════════════════════════════════════
# Full-Tune (full Optuna + experiments)
# ═══════════════════════════════════════════════════════════════════════════

def full_tune_trip(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[list[xgb.Booster], dict, bool, bool, float | None]:
    """Full Optuna + log/Huber experiments for Model 1A.

    Returns (models, best_params_dict, use_log, use_huber, huber_slope).
    """
    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    # Augment
    print("    Augmenting training data...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)

    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)

    # Optuna
    n_trials = OPTUNA_CONFIG["n_trials"]
    print(f"    Optuna ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=OPTUNA_CONFIG["n_startup_trials"], seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )
    study.optimize(
        lambda trial: _trip_optuna_objective(trial, dtrain_aug, dval, mono_tuple),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"    Optuna done ({elapsed/60:.1f} min) — best MAE: {best.value:.4f}%")

    # Log transform experiment
    print("    Log transform experiment...")
    log_mae = _try_log_transform_trip(train_aug, val_df, TRIP_FEATURES, best.params, mono_tuple)
    use_log = log_mae < best.value
    print(f"    log1p MAE: {log_mae:.4f}% vs raw: {best.value:.4f}% -> "
          f"{'USE LOG' if use_log else 'KEEP RAW'}")

    # Huber loss experiment
    print("    Huber loss experiment...")
    huber_mae, huber_slope = _try_huber_loss_trip(
        train_aug, val_df, TRIP_FEATURES, best.params, mono_tuple,
    )
    best_mae_so_far = min(best.value, log_mae) if use_log else best.value
    use_huber = huber_mae < best_mae_so_far
    print(f"    Huber MAE: {huber_mae:.4f}% (slope={huber_slope:.2f}) "
          f"vs best: {best_mae_so_far:.4f}% -> {'USE HUBER' if use_huber else 'KEEP CURRENT'}")

    if use_huber:
        use_log = False

    # Ensemble training
    best_params = {
        "objective": "reg:pseudohubererror" if use_huber else "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": best.params["max_depth"],
        "learning_rate": best.params["learning_rate"],
        "subsample": best.params["subsample"],
        "colsample_bytree": best.params["colsample_bytree"],
        "colsample_bylevel": best.params["colsample_bylevel"],
        "min_child_weight": best.params["min_child_weight"],
        "reg_alpha": best.params["reg_alpha"],
        "reg_lambda": best.params["reg_lambda"],
        "gamma": best.params["gamma"],
        "monotone_constraints": mono_tuple,
    }
    if use_huber:
        best_params["huber_slope"] = huber_slope

    best_n_estimators = best.params["n_estimators"]

    print(f"    Training {N_ENSEMBLE_SEEDS}-seed ensemble...")
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        params = {**best_params, "seed": seed}

        if use_log:
            y_aug_log = np.log1p(y_train_aug)
            dtrain_log = xgb.DMatrix(
                train_aug.select(TRIP_FEATURES).to_numpy().astype(np.float32),
                label=y_aug_log, feature_names=TRIP_FEATURES,
            )
            y_val_log = np.log1p(y_val)
            dval_log = xgb.DMatrix(
                val_df.select(TRIP_FEATURES).to_numpy().astype(np.float32),
                label=y_val_log, feature_names=TRIP_FEATURES,
            )
            model = xgb.train(
                params, dtrain_log,
                num_boost_round=best_n_estimators,
                evals=[(dval_log, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )
        else:
            model = xgb.train(
                params, dtrain_aug,
                num_boost_round=best_n_estimators,
                evals=[(dval, "val")],
                early_stopping_rounds=50,
                verbose_eval=0,
            )

        ensemble_models.append(model)
        print(f"      Seed {seed}: {model.num_boosted_rounds()} trees")

    # Return Optuna best params (for metadata saving)
    return ensemble_models, best.params, use_log, use_huber, huber_slope


def full_tune_realtime(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[list[xgb.Booster], dict]:
    """Full Optuna for Model 1B.

    Returns (models, best_params_dict).
    """
    dtrain, y_train = df_to_dmatrix(train_df, REALTIME_FEATURES, REALTIME_TARGET)
    dval, y_val = df_to_dmatrix(val_df, REALTIME_FEATURES, REALTIME_TARGET)

    n_trials = OPTUNA_CONFIG["n_trials"]
    print(f"    Optuna ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=OPTUNA_CONFIG["n_startup_trials"], seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )
    study.optimize(
        lambda trial: _realtime_optuna_objective(trial, dtrain, dval),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"    Optuna done ({elapsed/60:.1f} min) — best MAE: {best.value:.4f}%")

    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": best.params["max_depth"],
        "learning_rate": best.params["learning_rate"],
        "subsample": best.params["subsample"],
        "colsample_bytree": best.params["colsample_bytree"],
        "colsample_bylevel": best.params["colsample_bylevel"],
        "min_child_weight": best.params["min_child_weight"],
        "reg_alpha": best.params["reg_alpha"],
        "reg_lambda": best.params["reg_lambda"],
        "gamma": best.params["gamma"],
    }
    best_n_estimators = best.params["n_estimators"]

    print(f"    Training {N_ENSEMBLE_SEEDS}-seed ensemble...")
    ensemble_models = []
    for seed in ENSEMBLE_SEEDS:
        params = {**best_params, "seed": seed}
        model = xgb.train(
            params, dtrain,
            num_boost_round=best_n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        ensemble_models.append(model)
        print(f"      Seed {seed}: {model.num_boosted_rounds()} trees")

    return ensemble_models, best.params


def full_tune_anomaly(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
) -> tuple[dict[str, xgb.Booster], dict[str, float], dict[str, float], dict, dict]:
    """Full Optuna for Model 2 (100 trials per target).

    Returns (models, feature_means, feature_stds, thresholds, per_target_best_params).
    """
    feature_means = {}
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_means[feat] = float(np.mean(col))
        feature_stds[feat] = float(max(np.std(col), 0.01))

    models: dict[str, xgb.Booster] = {}
    per_target_params = {}

    for target_col in ANOMALY_TARGETS:
        print(f"    Reconstruction: {target_col}")
        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]

        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        print(f"      Optuna (100 trials)...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
        )
        study.optimize(
            lambda trial: _anomaly_optuna_objective(trial, dtrain, dval),
            n_trials=100,
            show_progress_bar=True,
        )

        best = study.best_trial
        per_target_params[target_col] = best.params
        print(f"      Best val MAE: {best.value:.4f}")

        final_params = {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": best.params["max_depth"],
            "learning_rate": best.params["learning_rate"],
            "subsample": best.params["subsample"],
            "colsample_bytree": best.params["colsample_bytree"],
            "min_child_weight": best.params["min_child_weight"],
            "reg_alpha": best.params["reg_alpha"],
            "reg_lambda": best.params["reg_lambda"],
            "gamma": best.params["gamma"],
            "seed": 42,
        }

        model = xgb.train(
            final_params, dtrain,
            num_boost_round=best.params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )
        models[target_col] = model
        print(f"      Trees: {model.num_boosted_rounds()}")

    # Calibrate thresholds
    def score_fn(data):
        return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)

    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    val_scores = score_fn(val_data)

    thresholds = {
        "p95": float(np.percentile(val_scores, 95)),
        "p97": float(np.percentile(val_scores, 97)),
        "p99": float(np.percentile(val_scores, 99)),
    }
    print(f"    Thresholds: p95={thresholds['p95']:.6f} p97={thresholds['p97']:.6f} "
          f"p99={thresholds['p99']:.6f}")

    return models, feature_means, feature_stds, thresholds, per_target_params


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def predict_trip_ensemble(
    models: list[xgb.Booster],
    dm: xgb.DMatrix,
    use_log: bool,
) -> np.ndarray:
    """Predict with trip ensemble, handling log transform."""
    if use_log:
        preds = np.column_stack([np.expm1(m.predict(dm)) for m in models])
    else:
        preds = np.column_stack([m.predict(dm) for m in models])
    return np.clip(preds.mean(axis=1), 0, None)


def evaluate_trip(
    models: list[xgb.Booster],
    df: pl.DataFrame,
    label: str,
    use_log: bool,
) -> dict:
    """Evaluate trip models on a dataset, return metrics dict."""
    dm, y_true = df_to_dmatrix(df, TRIP_FEATURES, TRIP_TARGET)
    y_pred = predict_trip_ensemble(models, dm, use_log)
    metrics = compute_metrics(y_true, y_pred)
    print(f"    {label}: MAE={metrics['MAE']:.4f}%  R2={metrics['R2']:.4f}  (n={len(df)})")
    return metrics


def evaluate_realtime(
    models: list[xgb.Booster],
    df: pl.DataFrame,
    label: str,
) -> dict:
    """Evaluate realtime models on a dataset, return metrics dict."""
    dm, y_true = df_to_dmatrix(df, REALTIME_FEATURES, REALTIME_TARGET)
    y_pred = ensemble_predict(models, dm)
    metrics = compute_metrics(y_true, y_pred)
    n_seg = df["segment_id"].n_unique() if "segment_id" in df.columns else "?"
    print(f"    {label}: MAE={metrics['MAE']:.4f}%  R2={metrics['R2']:.4f}  "
          f"(n={len(df)}, {n_seg} segs)")
    return metrics


def evaluate_anomaly(
    models: dict[str, xgb.Booster],
    df: pl.DataFrame,
    feature_stds: dict[str, float],
    thresholds: dict,
    label: str,
) -> dict:
    """Evaluate anomaly models. Returns injection test results."""
    def score_fn(data):
        return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)

    test_data = df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    test_scores = score_fn(test_data)
    p99 = thresholds["p99"]
    n_flagged = int((test_scores > p99).sum())
    fpr = n_flagged / len(test_scores) * 100

    # Injection test
    injection_results = synthetic_injection_test(
        score_fn, test_data, ANOMALY_FEATURES,
        n_injections=500, corruption_factor=1.5, threshold=p99,
    )
    det_rate = injection_results["detection_rate"]
    print(f"    {label}: det_rate={det_rate:.1f}%  FPR={fpr:.2f}%  (n={len(df)})")

    return {
        "detection_rate": det_rate,
        "false_positive_rate": fpr,
        **injection_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Regression Gate
# ═══════════════════════════════════════════════════════════════════════════

def regression_gate(
    model_name: str,
    old_value: float | None,
    new_value: float,
) -> str:
    """Compare new metric vs old and return status: OK / WARN / HALT.

    For SOC models (1a, 1b): compares MAE (lower is better).
    For anomaly (2): compares detection rate (higher is better).
    """
    if old_value is None:
        return "OK"

    if model_name in ("1a", "1b"):
        # MAE — lower is better
        if old_value == 0:
            return "OK"
        pct_change = (new_value - old_value) / old_value * 100
        if pct_change > 5.0:
            return "HALT"
        elif pct_change > 1.0:
            return "WARN"
        else:
            return "OK"
    else:
        # Detection rate — higher is better
        drop = old_value - new_value  # positive = regression
        if drop > 5.0:
            return "HALT"
        elif drop > 1.0:
            return "WARN"
        else:
            return "OK"


def format_delta(model_name: str, old_value: float | None, new_value: float) -> str:
    """Format the delta string for the log table."""
    if old_value is None:
        return "N/A"
    if model_name in ("1a", "1b"):
        if old_value == 0:
            return "N/A"
        pct = (new_value - old_value) / old_value * 100
        return f"{pct:+.1f}%"
    else:
        diff = new_value - old_value
        return f"{diff:+.1f}pp"


# ═══════════════════════════════════════════════════════════════════════════
# Backup & Deploy
# ═══════════════════════════════════════════════════════════════════════════

def backup_models(timestamp: str) -> Path:
    """Copy current artifacts to backup directory."""
    backup_dir = BACKUPS_DIR / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy model files and metadata
    patterns = [
        "soc_trip_model*.json",
        "soc_realtime_model*.json",
        "anomaly_*.json",
        "soc_trip_metadata.json",
        "soc_realtime_metadata.json",
        "anomaly_metadata.json",
        "feature_statistics.json",
        "thresholds.json",
        "conformal_*.json",
    ]
    for pattern in patterns:
        for src in ARTIFACTS_DIR.glob(pattern):
            shutil.copy2(str(src), str(backup_dir / src.name))

    n_files = len(list(backup_dir.iterdir()))
    print(f"  Backed up {n_files} files to {backup_dir.relative_to(MODELS_ROOT)}")
    return backup_dir


def deploy_trip_models(
    models: list[xgb.Booster],
    best_params: dict,
    use_log: bool,
    use_huber: bool,
    huber_slope: float | None,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_aug_size: int,
    metrics_results: dict,
) -> None:
    """Save trip models and metadata to artifacts + rpi5_bundle."""
    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    # Save ensemble models to artifacts
    for i, model in enumerate(models):
        if i == 0:
            path = ARTIFACTS_DIR / "soc_trip_model.json"
        else:
            path = ARTIFACTS_DIR / f"soc_trip_model_seed{ENSEMBLE_SEEDS[i]}.json"
        model.save_model(str(path))

    # Quantile regression
    print("    Training quantile models (q10/q90)...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)
    dtrain_aug, _ = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)
    dtest, y_test = df_to_dmatrix(test_df, TRIP_FEATURES, TRIP_TARGET)

    # Build params for quantile models
    qr_params = {
        "objective": "reg:squarederror",  # will be overridden
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": best_params["max_depth"],
        "learning_rate": best_params["learning_rate"],
        "subsample": best_params["subsample"],
        "colsample_bytree": best_params["colsample_bytree"],
        "colsample_bylevel": best_params["colsample_bylevel"],
        "min_child_weight": best_params["min_child_weight"],
        "reg_alpha": best_params["reg_alpha"],
        "reg_lambda": best_params["reg_lambda"],
        "gamma": best_params["gamma"],
        "monotone_constraints": mono_tuple,
    }
    best_n_estimators = best_params["n_estimators"]

    point_preds_test = predict_trip_ensemble(models, dtest, use_log)
    quantile_meta = _train_quantile_models(
        qr_params, best_n_estimators, dtrain_aug, dval, dtest,
        y_val, y_test, point_preds_test,
    )

    # Conformal prediction
    print("    Calibrating conformal prediction...")
    y_pred_val = predict_trip_ensemble(models, dval, use_log)
    y_pred_test = predict_trip_ensemble(models, dtest, use_log)
    conformal_meta = _calibrate_conformal(
        y_pred_val, y_val, y_pred_test, y_test, model_name="trip",
    )

    # Save metadata
    trip_meta = {
        "model": "soc_trip_v2",
        "trained_at": datetime.now().isoformat(),
        "features": TRIP_FEATURES,
        "target": TRIP_TARGET,
        "best_params": best_params,
        "monotonic_constraints": dict(zip(TRIP_FEATURES, mono_tuple)),
        "use_log_transform": use_log,
        "use_huber_loss": use_huber,
        "huber_slope": huber_slope if use_huber else None,
        "n_ensemble_seeds": N_ENSEMBLE_SEEDS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "augmentation_applied": True,
        "data_augmentation_factor": train_aug_size / len(train_df) if len(train_df) > 0 else 5.0,
        "splits": {
            "train": len(train_df),
            "train_augmented": train_aug_size,
            "val": len(val_df),
            "test": len(test_df),
        },
        "metrics": metrics_results,
        "quantile": quantile_meta,
        "conformal": conformal_meta,
    }
    meta_path = ARTIFACTS_DIR / "soc_trip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(trip_meta, f, indent=2, default=str)

    # Copy to rpi5_bundle
    _sync_to_rpi5("soc_trip_model", models)
    for name in ["soc_trip_model_q10.json", "soc_trip_model_q90.json"]:
        src = ARTIFACTS_DIR / name
        if src.exists():
            shutil.copy2(str(src), str(RPI5_MODELS / name))
    shutil.copy2(str(meta_path), str(RPI5_CONFIG / meta_path.name))
    conf_path = ARTIFACTS_DIR / "conformal_trip.json"
    if conf_path.exists():
        shutil.copy2(str(conf_path), str(RPI5_CONFIG / conf_path.name))

    print("    Trip models deployed")


def deploy_realtime_models(
    models: list[xgb.Booster],
    best_params: dict,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    metrics_results: dict,
) -> None:
    """Save realtime models and metadata to artifacts + rpi5_bundle."""
    # Save ensemble models
    for i, model in enumerate(models):
        if i == 0:
            path = ARTIFACTS_DIR / "soc_realtime_model.json"
        else:
            path = ARTIFACTS_DIR / f"soc_realtime_model_seed{ENSEMBLE_SEEDS[i]}.json"
        model.save_model(str(path))

    # Conformal prediction
    print("    Calibrating conformal prediction...")
    dval, y_val = df_to_dmatrix(val_df, REALTIME_FEATURES, REALTIME_TARGET)
    dtest, y_test = df_to_dmatrix(test_df, REALTIME_FEATURES, REALTIME_TARGET)

    y_pred_val = ensemble_predict(models, dval)
    y_pred_test = ensemble_predict(models, dtest)
    conformal_meta = _calibrate_conformal(
        y_pred_val, y_val, y_pred_test, y_test, model_name="realtime",
    )

    n_seg = {
        "train": train_df["segment_id"].n_unique() if "segment_id" in train_df.columns else len(train_df),
        "val": val_df["segment_id"].n_unique() if "segment_id" in val_df.columns else len(val_df),
        "test": test_df["segment_id"].n_unique() if "segment_id" in test_df.columns else len(test_df),
    }

    rt_meta = {
        "model": "soc_realtime_v2",
        "trained_at": datetime.now().isoformat(),
        "features": REALTIME_FEATURES,
        "target": REALTIME_TARGET,
        "best_params": best_params,
        "n_ensemble_seeds": N_ENSEMBLE_SEEDS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_segments": n_seg["train"],
            "val_segments": n_seg["val"],
            "test_segments": n_seg["test"],
        },
        "metrics": metrics_results,
        "conformal": conformal_meta,
    }
    meta_path = ARTIFACTS_DIR / "soc_realtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(rt_meta, f, indent=2, default=str)

    # Copy to rpi5_bundle
    _sync_to_rpi5("soc_realtime_model", models)
    shutil.copy2(str(meta_path), str(RPI5_CONFIG / meta_path.name))
    conf_path = ARTIFACTS_DIR / "conformal_realtime.json"
    if conf_path.exists():
        shutil.copy2(str(conf_path), str(RPI5_CONFIG / conf_path.name))

    print("    Realtime models deployed")


def deploy_anomaly_models(
    models: dict[str, xgb.Booster],
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    thresholds: dict,
    per_target_params: dict,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    injection_results: dict,
) -> None:
    """Save anomaly models and metadata to artifacts + rpi5_bundle."""
    # Save models
    for target_col, model in models.items():
        path = ARTIFACTS_DIR / f"anomaly_{target_col}.json"
        model.save_model(str(path))
        shutil.copy2(str(path), str(RPI5_MODELS / path.name))

    # Save feature statistics
    stats = {"feature_means": feature_means, "feature_stds": feature_stds}
    stats_path = ARTIFACTS_DIR / "feature_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    shutil.copy2(str(stats_path), str(RPI5_CONFIG / stats_path.name))

    # Save thresholds
    thresh_path = ARTIFACTS_DIR / "thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    shutil.copy2(str(thresh_path), str(RPI5_CONFIG / thresh_path.name))

    # Save metadata (with per-target params for future quick retrains)
    anom_meta = {
        "model": "anomaly_reconstruction_v2",
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "targets": ANOMALY_TARGETS,
        "thresholds": thresholds,
        "per_target_best_params": per_target_params,
        "injection_test": injection_results,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    meta_path = ARTIFACTS_DIR / "anomaly_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(anom_meta, f, indent=2, default=str)
    shutil.copy2(str(meta_path), str(RPI5_CONFIG / meta_path.name))

    print("    Anomaly models deployed")


def _sync_to_rpi5(prefix: str, models: list[xgb.Booster]) -> None:
    """Copy ensemble model files to rpi5_bundle/models/."""
    for i, _ in enumerate(models):
        if i == 0:
            name = f"{prefix}.json"
        else:
            name = f"{prefix}_seed{ENSEMBLE_SEEDS[i]}.json"
        src = ARTIFACTS_DIR / name
        if src.exists():
            shutil.copy2(str(src), str(RPI5_MODELS / name))


def regenerate_rpi5_testdata(
    trip_test_df: pl.DataFrame | None,
    realtime_test_df: pl.DataFrame | None,
    anomaly_test_df: pl.DataFrame | None,
) -> None:
    """Write new test CSVs to rpi5_bundle/testdata/."""
    if trip_test_df is not None and len(trip_test_df) > 0:
        cols = list(dict.fromkeys(TRIP_FEATURES + [TRIP_TARGET, "direction_encoded"]))
        available = [c for c in cols if c in trip_test_df.columns]
        trip_test_df.select(available).write_csv(str(RPI5_TESTDATA / "trip_test.csv"))
        print(f"  Updated trip_test.csv ({len(trip_test_df)} rows)")

    if realtime_test_df is not None and len(realtime_test_df) > 0:
        cols = list(dict.fromkeys(REALTIME_FEATURES + [REALTIME_TARGET, "trip_progress_fraction"]))
        available = [c for c in cols if c in realtime_test_df.columns]
        realtime_test_df.select(available).write_csv(str(RPI5_TESTDATA / "realtime_test.csv"))
        print(f"  Updated realtime_test.csv ({len(realtime_test_df)} rows)")

    if anomaly_test_df is not None and len(anomaly_test_df) > 0:
        available = [c for c in ANOMALY_FEATURES if c in anomaly_test_df.columns]
        anomaly_test_df.select(available).write_csv(str(RPI5_TESTDATA / "anomaly_test.csv"))
        print(f"  Updated anomaly_test.csv ({len(anomaly_test_df)} rows)")


# ═══════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════

def append_retrain_log(info: dict) -> None:
    """Append a retrain entry to RETRAIN_LOG.md."""
    timestamp = info["timestamp"]
    mode = info["mode"]
    models_trained = info["models_trained"]

    lines = []

    # Create file header on first run
    if not RETRAIN_LOG.exists():
        lines.append("# M/B Dalaray Model Retrain Log\n\n")
        lines.append("Auto-generated by `python -m train.retrain`.\n\n")
        lines.append("---\n\n")

    lines.append(f"## Retrain: {timestamp} — {mode}, {', '.join(models_trained)}\n\n")

    # Data summary
    old_segs = info.get("old_segments", "?")
    new_segs = info.get("new_segments", "?")
    if isinstance(old_segs, int) and isinstance(new_segs, int):
        delta = new_segs - old_segs
        new_days = info.get("new_days", "?")
        lines.append(f"**Data**: {old_segs} -> {new_segs} segments "
                      f"(+{delta} new, {new_days} new days)\n")
    else:
        lines.append(f"**Data**: {new_segs} segments\n")

    # Rolling split info
    if "train_end" in info:
        lines.append(f"**Rolling split**: train <= {info['train_end']} | "
                      f"val <= {info['val_end']} | test after\n\n")

    # Results table
    lines.append("| Model | Fixed MAE (old) | Fixed MAE (new) | Delta | Rolling MAE | Status |\n")
    lines.append("|-------|----------------|----------------|-------|------------|--------|\n")

    for entry in info.get("results", []):
        model = entry["model"]
        old_val = entry.get("old_fixed")
        new_fixed = entry.get("new_fixed")
        rolling = entry.get("rolling")
        status = entry.get("status", "?")

        if model == "2":
            old_str = f"{old_val:.1f}% det" if old_val is not None else "?"
            new_str = f"{new_fixed:.1f}% det" if new_fixed is not None else "?"
            roll_str = f"{rolling:.1f}% det" if rolling is not None else "?"
        else:
            old_str = f"{old_val:.3f}%" if old_val is not None else "?"
            new_str = f"{new_fixed:.3f}%" if new_fixed is not None else "?"
            roll_str = f"{rolling:.3f}%" if rolling is not None else "?"

        delta_str = entry.get("delta_str", "N/A")
        lines.append(f"| {model} | {old_str} | {new_str} | {delta_str} | {roll_str} | {status} |\n")

    lines.append("\n---\n\n")

    with open(RETRAIN_LOG, "a") as f:
        f.writelines(lines)

    print(f"  Logged to {RETRAIN_LOG.relative_to(MODELS_ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Retrain M/B Dalaray XGBoost models with rolling splits",
    )
    parser.add_argument(
        "--model", choices=["1a", "1b", "2", "all"], default="all",
        help="Which model to retrain (default: all)",
    )
    parser.add_argument(
        "--full-tune", action="store_true",
        help="Run full Optuna search (300 trials) instead of quick retrain",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Extract + train + evaluate only; do not deploy models",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip feature re-extraction (use existing feature stores)",
    )
    parser.add_argument(
        "--force-deploy", action="store_true",
        help="Deploy models even if regression gate HALTs (use when fixed test set composition changed)",
    )
    args = parser.parse_args()

    selected = [args.model] if args.model != "all" else ["1a", "1b", "2"]
    mode = "full-tune" if args.full_tune else "quick"
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Banner
    print("\n" + "=" * 65)
    print("  M/B Dalaray Model Retrain")
    print(f"  Mode: {mode}  |  Models: {', '.join(selected)}")
    print(f"  Dry run: {args.dry_run}  |  Re-extract: {not args.no_extract}  |  Force: {args.force_deploy}")
    print("=" * 65)

    start_time = time.time()

    # ── Step 1: Count old segments ──
    print("\n[1] Checking existing model metadata...")
    old_seg_counts = {}
    for m in selected:
        count = get_old_segment_count(m)
        old_seg_counts[m] = count
        if count is not None:
            print(f"  {m}: {count} segments in production")
        else:
            print(f"  {m}: no production metadata found")

    # ── Step 2: Re-extract features ──
    if not args.no_extract:
        print("\n[2] Re-extracting features...")
        re_extract_features(selected)
    else:
        print("\n[2] Skipping feature extraction (--no-extract)")

    # ── Step 3: Load feature stores and count ──
    print("\n[3] Loading feature stores...")
    trip_df = None
    realtime_df = None
    anomaly_df = None

    if "1a" in selected:
        trip_df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
        new_segs = len(trip_df)
        old = old_seg_counts.get("1a")
        delta_str = f" (+{new_segs - old} new)" if old else ""
        print(f"  trip_features: {new_segs} segments{delta_str}")

    if "1b" in selected:
        realtime_df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
        n_rows = len(realtime_df)
        n_segs = realtime_df["segment_id"].n_unique()
        old = old_seg_counts.get("1b")
        delta_str = f" (+{n_segs - old} new)" if old else ""
        print(f"  realtime_features: {n_rows} rows, {n_segs} segments{delta_str}")

    if "2" in selected:
        anomaly_df = pl.read_parquet(FEATURE_STORE_DIR / "anomaly_features.parquet")
        n_rows = len(anomaly_df)
        print(f"  anomaly_features: {n_rows} rows")

    # ── Step 4: Compute rolling split dates ──
    print("\n[4] Computing rolling split dates...")

    # Use trip_df for date computation if available, else realtime or anomaly
    ref_df = trip_df if trip_df is not None else (realtime_df if realtime_df is not None else anomaly_df)
    if ref_df is None:
        print("  ERROR: No feature data loaded")
        sys.exit(1)

    n_dates = len(ref_df["departure_time"].dt.date().unique())

    if n_dates < 20:
        print(f"  WARNING: Only {n_dates} unique dates — using fixed split dates as fallback")
        from config import TRAIN_END, VAL_END
        train_end = datetime.fromisoformat(TRAIN_END + "T23:59:59")
        val_end = datetime.fromisoformat(VAL_END + "T23:59:59")
    else:
        train_end, val_end = compute_rolling_split_dates(ref_df)

    print(f"  Rolling split: train <= {train_end.date()} | val <= {val_end.date()} | test after")

    # Also determine new unique days beyond original data
    all_dates = sorted(ref_df["departure_time"].dt.date().unique().to_list())
    original_boundary = datetime.fromisoformat(FIXED_TEST_BOUNDARY).date()
    # Count dates after old test boundary that weren't in the original dataset
    # (rough heuristic — exact count depends on old last date)
    old_meta = load_production_metadata("1a") or load_production_metadata("1b")
    if old_meta:
        old_trained_at = old_meta.get("trained_at", "")
        new_days = sum(1 for d in all_dates if d > original_boundary)
    else:
        new_days = "?"

    # ── Step 5: Backup existing models ──
    if not args.dry_run:
        print("\n[5] Backing up current models...")
        backup_models(timestamp_str)
    else:
        print("\n[5] Skipping backup (--dry-run)")

    # ── Step 6: Train and evaluate each model ──
    print("\n[6] Training and evaluating...")
    log_results = []
    any_halt = False

    # ── Model 1A ──
    if "1a" in selected and trip_df is not None:
        print("\n  --- Model 1A: Trip-Level SOC ---")
        train_df, val_df, rolling_test_df = rolling_temporal_split(trip_df, train_end, val_end)
        fixed_test_df = get_fixed_test_set(trip_df)

        print(f"  Rolling: Train={len(train_df)} | Val={len(val_df)} | Test={len(rolling_test_df)}")
        if len(fixed_test_df) > 0:
            print(f"  Fixed test (>{FIXED_TEST_BOUNDARY}): {len(fixed_test_df)} segments")
        else:
            print(f"  WARNING: Fixed test set is empty (no data after {FIXED_TEST_BOUNDARY})")

        if args.full_tune:
            trip_models, trip_params, use_log, use_huber, huber_slope = full_tune_trip(train_df, val_df)
        else:
            trip_models, trip_params, use_log, use_huber, huber_slope = quick_retrain_trip(train_df, val_df)

        # Evaluate on both test sets
        metrics_results = {}

        # Train/val metrics for metadata
        train_metrics = evaluate_trip(trip_models, train_df, "Train", use_log)
        val_metrics = evaluate_trip(trip_models, val_df, "Validation", use_log)
        metrics_results["Train"] = train_metrics
        metrics_results["Validation"] = val_metrics

        rolling_metrics = evaluate_trip(trip_models, rolling_test_df, "Rolling Test", use_log)
        metrics_results["Test"] = rolling_metrics

        fixed_mae = None
        if len(fixed_test_df) > 0:
            fixed_metrics = evaluate_trip(trip_models, fixed_test_df, "Fixed Test", use_log)
            fixed_mae = fixed_metrics["MAE"]

        old_mae = get_production_test_mae("1a")
        gate_value = fixed_mae if fixed_mae is not None else rolling_metrics["MAE"]
        status = regression_gate("1a", old_mae, gate_value)
        delta_str = format_delta("1a", old_mae, gate_value) if fixed_mae is not None else "N/A"

        print(f"  Regression gate: {status}")
        if status == "HALT" and args.force_deploy:
            print(f"  HALT overridden by --force-deploy (test set composition changed)")
            status = "FORCE"
        elif status == "HALT":
            print(f"  HALT: Fixed test MAE regressed by >{5}% — NOT deploying")
            any_halt = True
        elif status == "WARN":
            print(f"  WARN: Fixed test MAE regressed slightly (1-5%)")

        # Deploy if gate passes and not dry-run
        if status != "HALT" and not args.dry_run:
            train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=False)
            deploy_trip_models(
                trip_models, trip_params, use_log, use_huber, huber_slope,
                train_df, val_df, rolling_test_df if len(fixed_test_df) == 0 else fixed_test_df,
                len(train_aug), metrics_results,
            )

        log_results.append({
            "model": "1A",
            "old_fixed": old_mae,
            "new_fixed": fixed_mae if fixed_mae is not None else rolling_metrics["MAE"],
            "rolling": rolling_metrics["MAE"],
            "delta_str": delta_str,
            "status": status,
        })

    # ── Model 1B ──
    if "1b" in selected and realtime_df is not None:
        print("\n  --- Model 1B: Real-Time SOC ---")
        train_df, val_df, rolling_test_df = rolling_temporal_split(realtime_df, train_end, val_end)
        fixed_test_df = get_fixed_test_set(realtime_df)

        n_segs_train = train_df["segment_id"].n_unique() if "segment_id" in train_df.columns else "?"
        n_segs_val = val_df["segment_id"].n_unique() if "segment_id" in val_df.columns else "?"
        n_segs_test = rolling_test_df["segment_id"].n_unique() if "segment_id" in rolling_test_df.columns else "?"
        print(f"  Rolling: Train={len(train_df)} ({n_segs_train} segs) | "
              f"Val={len(val_df)} ({n_segs_val} segs) | Test={len(rolling_test_df)} ({n_segs_test} segs)")
        if len(fixed_test_df) > 0:
            n_fixed = fixed_test_df["segment_id"].n_unique() if "segment_id" in fixed_test_df.columns else len(fixed_test_df)
            print(f"  Fixed test (>{FIXED_TEST_BOUNDARY}): {len(fixed_test_df)} rows ({n_fixed} segs)")

        if args.full_tune:
            rt_models, rt_params = full_tune_realtime(train_df, val_df)
        else:
            rt_models, rt_params = quick_retrain_realtime(train_df, val_df)

        # Evaluate
        metrics_results = {}
        train_metrics = evaluate_realtime(rt_models, train_df, "Train")
        val_metrics = evaluate_realtime(rt_models, val_df, "Validation")
        metrics_results["Train"] = train_metrics
        metrics_results["Validation"] = val_metrics

        rolling_metrics = evaluate_realtime(rt_models, rolling_test_df, "Rolling Test")
        metrics_results["Test"] = rolling_metrics

        fixed_mae = None
        if len(fixed_test_df) > 0:
            fixed_metrics = evaluate_realtime(rt_models, fixed_test_df, "Fixed Test")
            fixed_mae = fixed_metrics["MAE"]

        old_mae = get_production_test_mae("1b")
        gate_value = fixed_mae if fixed_mae is not None else rolling_metrics["MAE"]
        status = regression_gate("1b", old_mae, gate_value)
        delta_str = format_delta("1b", old_mae, gate_value) if fixed_mae is not None else "N/A"

        print(f"  Regression gate: {status}")
        if status == "HALT" and args.force_deploy:
            print(f"  HALT overridden by --force-deploy (test set composition changed)")
            status = "FORCE"
        elif status == "HALT":
            print(f"  HALT: Fixed test MAE regressed by >{5}% — NOT deploying")
            any_halt = True

        if status != "HALT" and not args.dry_run:
            deploy_realtime_models(
                rt_models, rt_params,
                train_df, val_df,
                rolling_test_df if len(fixed_test_df) == 0 else fixed_test_df,
                metrics_results,
            )

        log_results.append({
            "model": "1B",
            "old_fixed": old_mae,
            "new_fixed": fixed_mae if fixed_mae is not None else rolling_metrics["MAE"],
            "rolling": rolling_metrics["MAE"],
            "delta_str": delta_str,
            "status": status,
        })

    # ── Model 2 ──
    if "2" in selected and anomaly_df is not None:
        print("\n  --- Model 2: Anomaly Detection ---")
        train_df, val_df, rolling_test_df = rolling_temporal_split(anomaly_df, train_end, val_end)
        fixed_test_df = get_fixed_test_set(anomaly_df)

        print(f"  Rolling: Train={len(train_df)} | Val={len(val_df)} | Test={len(rolling_test_df)}")
        if len(fixed_test_df) > 0:
            print(f"  Fixed test (>{FIXED_TEST_BOUNDARY}): {len(fixed_test_df)} rows")

        if args.full_tune:
            anom_models, feat_means, feat_stds, thresholds, per_target_params = \
                full_tune_anomaly(train_df, val_df)
        else:
            anom_models, feat_means, feat_stds, thresholds, per_target_params = \
                quick_retrain_anomaly(train_df, val_df)

        # Evaluate on both test sets
        rolling_results = evaluate_anomaly(
            anom_models, rolling_test_df, feat_stds, thresholds, "Rolling Test",
        )

        fixed_det_rate = None
        fixed_results = {}
        if len(fixed_test_df) > 0:
            fixed_results = evaluate_anomaly(
                anom_models, fixed_test_df, feat_stds, thresholds, "Fixed Test",
            )
            fixed_det_rate = fixed_results.get("detection_rate")

        old_det_rate = get_production_test_mae("2")  # returns detection_rate for anomaly
        gate_value = fixed_det_rate if fixed_det_rate is not None else rolling_results["detection_rate"]
        status = regression_gate("2", old_det_rate, gate_value)
        delta_str = format_delta("2", old_det_rate, gate_value) if fixed_det_rate is not None else "N/A"

        print(f"  Regression gate: {status}")
        if status == "HALT" and args.force_deploy:
            print(f"  HALT overridden by --force-deploy (test set composition changed)")
            status = "FORCE"
        elif status == "HALT":
            print(f"  HALT: Detection rate dropped by >5pp — NOT deploying")
            any_halt = True

        if status != "HALT" and not args.dry_run:
            deploy_anomaly_models(
                anom_models, feat_means, feat_stds, thresholds, per_target_params,
                train_df, val_df, rolling_test_df,
                fixed_results if fixed_results else rolling_results,
            )

        log_results.append({
            "model": "2",
            "old_fixed": old_det_rate,
            "new_fixed": fixed_det_rate if fixed_det_rate is not None else rolling_results["detection_rate"],
            "rolling": rolling_results["detection_rate"],
            "delta_str": delta_str,
            "status": status,
        })

    # ── Step 7: Update rpi5_bundle test data ──
    if not args.dry_run and not any_halt:
        print("\n[7] Updating RPi5 bundle test data...")
        trip_test = None
        rt_test = None
        anom_test = None

        if "1a" in selected and trip_df is not None:
            _, _, rolling_test = rolling_temporal_split(trip_df, train_end, val_end)
            fixed_test = get_fixed_test_set(trip_df)
            trip_test = fixed_test if len(fixed_test) > 0 else rolling_test

        if "1b" in selected and realtime_df is not None:
            _, _, rolling_test = rolling_temporal_split(realtime_df, train_end, val_end)
            fixed_test = get_fixed_test_set(realtime_df)
            rt_test = fixed_test if len(fixed_test) > 0 else rolling_test

        if "2" in selected and anomaly_df is not None:
            _, _, rolling_test = rolling_temporal_split(anomaly_df, train_end, val_end)
            fixed_test = get_fixed_test_set(anomaly_df)
            anom_test = fixed_test if len(fixed_test) > 0 else rolling_test

        regenerate_rpi5_testdata(trip_test, rt_test, anom_test)
    else:
        print("\n[7] Skipping RPi5 test data update")

    # ── Step 8: Log results ──
    print("\n[8] Writing retrain log...")
    # Compute representative segment count
    ref_df_for_count = trip_df if trip_df is not None else realtime_df
    new_seg_count = None
    if ref_df_for_count is not None:
        if "segment_id" in ref_df_for_count.columns:
            new_seg_count = ref_df_for_count["segment_id"].n_unique()
        else:
            new_seg_count = len(ref_df_for_count)

    # Find old segment count
    old_seg = old_seg_counts.get("1a") or old_seg_counts.get("1b") or old_seg_counts.get("2")

    append_retrain_log({
        "timestamp": timestamp_display,
        "mode": mode,
        "models_trained": [m.upper() if m != "2" else m for m in selected],
        "old_segments": old_seg,
        "new_segments": new_seg_count,
        "new_days": new_days,
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "results": log_results,
    })

    # ── Summary ──
    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print("  RETRAIN SUMMARY")
    print("=" * 65)
    print(f"  Mode: {mode}  |  Time: {total_time/60:.1f} min")
    if args.dry_run:
        print("  (DRY RUN — no models deployed)")
    print()
    print(f"  {'Model':<8s} {'Fixed (old)':<14s} {'Fixed (new)':<14s} {'Delta':<10s} {'Rolling':<12s} {'Status':<8s}")
    print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10} {'-'*12} {'-'*8}")

    for r in log_results:
        model = r["model"]
        old_v = r.get("old_fixed")
        new_v = r.get("new_fixed")
        rolling = r.get("rolling")
        delta = r.get("delta_str", "N/A")
        status = r.get("status", "?")

        if model == "2":
            old_str = f"{old_v:.1f}% det" if old_v is not None else "?"
            new_str = f"{new_v:.1f}% det" if new_v is not None else "?"
            roll_str = f"{rolling:.1f}% det" if rolling is not None else "?"
        else:
            old_str = f"{old_v:.3f}%" if old_v is not None else "?"
            new_str = f"{new_v:.3f}%" if new_v is not None else "?"
            roll_str = f"{rolling:.3f}%" if rolling is not None else "?"

        print(f"  {model:<8s} {old_str:<14s} {new_str:<14s} {delta:<10s} {roll_str:<12s} {status:<8s}")

    print("=" * 65)

    if any_halt:
        print("\n  Some models HALTED due to regression — check log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
