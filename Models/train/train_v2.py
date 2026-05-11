"""V2 Optimized Training Pipeline for all M/B Dalaray models.

Implements:
  - Optuna Bayesian optimization (TPE, 300 trials)
  - Multi-seed ensemble (5 seeds, averaged predictions)
  - Monotonic constraints (physics-enforced for Model 1A)
  - Data augmentation (Gaussian noise + C-Mixup for Model 1A)
  - Target transformation experiment (log1p)
  - Bigger models (up to 5000 trees, max_depth 10)

Usage:
    python -m train.train_v2 --model trip        # Model 1A only
    python -m train.train_v2 --model realtime     # Model 1B only
    python -m train.train_v2 --model anomaly      # Model 2 only
    python -m train.train_v2 --model all          # All models
"""

from __future__ import annotations

import argparse
import json
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
    N_ENSEMBLE_SEEDS,
    OPTUNA_CONFIG,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_FEATURES_DIRECTIONAL,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    VAL_END,
    XGBOOST_ANOMALY_PARAMS,
    get_monotonic_constraint_tuple,
)
from data.augment import augment_trip_data
from evaluate.evaluate_anomaly import (
    plot_anomaly_distribution,
    synthetic_injection_test,
    threshold_sensitivity,
)
from evaluate.evaluate_soc import (
    compute_metrics,
    plot_feature_importance,
    plot_predictions,
    print_direction_breakdown,
    print_metrics,
)

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Shared Utilities ────────────────────────────────────────────────────

def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split using config dates."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def df_to_dmatrix(
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[xgb.DMatrix, np.ndarray]:
    """Convert DataFrame to XGBoost DMatrix."""
    X = df.select(features).to_numpy().astype(np.float32)
    y = df[target].to_numpy().astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=features), y


def ensemble_predict(models: list[xgb.Booster], dm: xgb.DMatrix) -> np.ndarray:
    """Average predictions from multiple ensemble members."""
    preds = np.column_stack([m.predict(dm) for m in models])
    return preds.mean(axis=1)


# ── Model 1A: Trip-Level SOC ──────────────────────────────────────────

def _trip_optuna_objective(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    monotonic_tuple: tuple,
) -> float:
    """Optuna objective for Model 1A."""
    ss = OPTUNA_CONFIG["search_space"]

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *ss["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *ss["colsample_bylevel"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *ss["gamma"]),
        "monotone_constraints": monotonic_tuple,
        "seed": 42,
    }

    n_estimators = trial.suggest_int("n_estimators", *ss["n_estimators"])

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "val-mae")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
        callbacks=[pruning_callback],
    )

    y_val = dval.get_label()
    y_pred = model.predict(dval)
    mae = float(np.mean(np.abs(y_val - y_pred)))

    # Also report RMSE for multi-objective insight
    trial.set_user_attr("rmse", float(np.sqrt(np.mean((y_val - y_pred) ** 2))))
    trial.set_user_attr("n_trees", model.num_boosted_rounds())

    return mae


def train_trip_v2(verbose: bool = True) -> dict:
    """Train Model 1A with v2 optimizations."""
    print("\n" + "=" * 60)
    print("  MODEL 1A: Trip-Level SOC Prediction (v2)")
    print("=" * 60)

    # Load & split
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Data augmentation (training set only)
    print("\n  [1/4] Data Augmentation...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=verbose)

    # Build DMatrices
    dtrain_aug, y_train_aug = df_to_dmatrix(train_aug, TRIP_FEATURES, TRIP_TARGET)
    dtrain_orig, y_train_orig = df_to_dmatrix(train_df, TRIP_FEATURES, TRIP_TARGET)
    dval, y_val = df_to_dmatrix(val_df, TRIP_FEATURES, TRIP_TARGET)
    dtest, y_test = df_to_dmatrix(test_df, TRIP_FEATURES, TRIP_TARGET)

    # Monotonic constraints
    mono_tuple = get_monotonic_constraint_tuple(TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS)

    # ── Optuna search ──
    print(f"\n  [2/4] Optuna Bayesian Optimization ({OPTUNA_CONFIG['n_trials']} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=OPTUNA_CONFIG["n_startup_trials"],
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )

    study.optimize(
        lambda trial: _trip_optuna_objective(trial, dtrain_aug, dval, mono_tuple),
        n_trials=OPTUNA_CONFIG["n_trials"],
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")
    print(f"  Best trees: {best.user_attrs.get('n_trees', 'N/A')}")

    # ── Target transformation experiment ──
    print("\n  [2b] Target Transformation Experiment (log1p)...")
    log_mae = _try_log_transform_trip(train_aug, val_df, TRIP_FEATURES, best.params, mono_tuple)
    use_log = log_mae < best.value
    print(f"  log1p val MAE: {log_mae:.4f}% vs raw: {best.value:.4f}% -> {'USE LOG' if use_log else 'KEEP RAW'}")

    # ── Huber loss experiment (v4) ──
    print("\n  [2c] Huber Loss Experiment...")
    huber_mae, huber_slope = _try_huber_loss_trip(
        train_aug, val_df, TRIP_FEATURES, best.params, mono_tuple,
    )
    # Compare Huber against best so far (which may be log or raw)
    best_mae_so_far = min(best.value, log_mae) if use_log else best.value
    use_huber = huber_mae < best_mae_so_far
    print(f"  Huber val MAE: {huber_mae:.4f}% (slope={huber_slope:.2f}) "
          f"vs best raw/log: {best_mae_so_far:.4f}% -> {'USE HUBER' if use_huber else 'KEEP CURRENT'}")

    # If Huber wins, it overrides log transform choice
    if use_huber:
        use_log = False

    # ── Ensemble training ──
    print(f"\n  [3/4] Ensemble Training ({N_ENSEMBLE_SEEDS} seeds)...")

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

    ensemble_models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
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
        print(f"    Seed {seed}: {model.num_boosted_rounds()} trees")

    # ── Evaluate ──
    print("\n  [4/4] Evaluation...")

    def predict_ensemble(dm):
        if use_log:
            preds = np.column_stack([np.expm1(m.predict(dm)) for m in ensemble_models])
        else:
            preds = np.column_stack([m.predict(dm) for m in ensemble_models])
        return np.clip(preds.mean(axis=1), 0, None)

    results = {}
    for name, dm, y_true, split_df in [
        ("Train", dtrain_orig, y_train_orig, train_df),
        ("Validation", dval, y_val, val_df),
        ("Test", dtest, y_test, test_df),
    ]:
        y_pred = predict_ensemble(dm)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics
        print_metrics(name, metrics)
        if "direction_encoded" in split_df.columns:
            dirs = split_df["direction_encoded"].to_numpy()
            print_direction_breakdown(y_true, y_pred, dirs)

    # Plots
    y_pred_test = predict_ensemble(dtest)
    plot_predictions(
        y_test, y_pred_test,
        "Model 1A v2: Trip-Level SOC (Ensemble)",
        save_path=ARTIFACTS_DIR / "soc_trip_predictions_v2.png",
    )
    plot_feature_importance(
        ensemble_models[0], TRIP_FEATURES,
        "Model 1A v2: Trip-Level SOC",
        save_path=ARTIFACTS_DIR / "soc_trip_importance_v2.png",
    )

    # Save ensemble models
    total_size = 0
    for i, model in enumerate(ensemble_models):
        if i == 0:
            # Primary model (used by inference scripts)
            path = ARTIFACTS_DIR / "soc_trip_model.json"
        else:
            path = ARTIFACTS_DIR / f"soc_trip_model_seed{ENSEMBLE_SEEDS[i]}.json"
        model.save_model(str(path))
        total_size += path.stat().st_size
        print(f"    Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    print(f"    Total Model 1A size: {total_size / 1024:.0f} KB")

    # ── Quantile Regression (v4) ──
    print("\n  [5/6] Quantile Regression (q10, q90)...")
    quantile_meta = _train_quantile_models(
        best_params, best_n_estimators, dtrain_aug, dval, dtest,
        y_val, y_test, predict_ensemble(dtest),
    )

    # ── Conformal Prediction (v4) ──
    print("\n  [6/6] Conformal Prediction Calibration...")
    conformal_meta = _calibrate_conformal(
        predict_ensemble(dval), y_val,
        predict_ensemble(dtest), y_test,
        model_name="trip",
    )

    # Save metadata
    meta = {
        "model": "soc_trip_v2",
        "trained_at": datetime.now().isoformat(),
        "features": TRIP_FEATURES,
        "target": TRIP_TARGET,
        "best_params": best.params,
        "monotonic_constraints": dict(zip(TRIP_FEATURES, mono_tuple)),
        "use_log_transform": use_log,
        "use_huber_loss": use_huber,
        "huber_slope": huber_slope if use_huber else None,
        "huber_val_mae": huber_mae,
        "n_ensemble_seeds": N_ENSEMBLE_SEEDS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "optuna_trials": OPTUNA_CONFIG["n_trials"],
        "optuna_best_val_mae": best.value,
        "augmentation_applied": True,
        "data_augmentation_factor": len(train_aug) / len(train_df),
        "splits": {
            "train": len(train_df),
            "train_augmented": len(train_aug),
            "val": len(val_df),
            "test": len(test_df),
        },
        "metrics": results,
        "quantile": quantile_meta,
        "conformal": conformal_meta,
    }
    meta_path = ARTIFACTS_DIR / "soc_trip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


def _try_log_transform_trip(
    train_aug: pl.DataFrame,
    val_df: pl.DataFrame,
    features: list[str],
    params: dict,
    mono_tuple: tuple,
) -> float:
    """Try log1p transform on target and return val MAE (in original scale)."""
    X_train = train_aug.select(features).to_numpy().astype(np.float32)
    y_train = np.log1p(train_aug[TRIP_TARGET].to_numpy().astype(np.float32))

    X_val = val_df.select(features).to_numpy().astype(np.float32)
    y_val_raw = val_df[TRIP_TARGET].to_numpy().astype(np.float32)
    y_val_log = np.log1p(y_val_raw)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val_log, feature_names=features)

    full_params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "colsample_bylevel": params["colsample_bylevel"],
        "min_child_weight": params["min_child_weight"],
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "gamma": params["gamma"],
        "monotone_constraints": mono_tuple,
        "seed": 42,
    }

    model = xgb.train(
        full_params, dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
    )

    y_pred_log = model.predict(dval)
    y_pred = np.expm1(y_pred_log)
    return float(np.mean(np.abs(y_val_raw - y_pred)))


def _try_huber_loss_trip(
    train_aug: pl.DataFrame,
    val_df: pl.DataFrame,
    features: list[str],
    params: dict,
    mono_tuple: tuple,
    n_trials: int = 30,
) -> tuple[float, float]:
    """Try Huber loss with Optuna-tuned slope, return (best_val_mae, best_slope)."""
    X_train = train_aug.select(features).to_numpy().astype(np.float32)
    y_train = train_aug[TRIP_TARGET].to_numpy().astype(np.float32)
    X_val = val_df.select(features).to_numpy().astype(np.float32)
    y_val = val_df[TRIP_TARGET].to_numpy().astype(np.float32)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

    def objective(trial):
        slope = trial.suggest_float("huber_slope", 0.5, 5.0)
        huber_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": slope,
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_depth": params["max_depth"],
            "learning_rate": params["learning_rate"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "colsample_bylevel": params["colsample_bylevel"],
            "min_child_weight": params["min_child_weight"],
            "reg_alpha": params["reg_alpha"],
            "reg_lambda": params["reg_lambda"],
            "gamma": params["gamma"],
            "monotone_constraints": mono_tuple,
            "seed": 42,
        }
        model = xgb.train(
            huber_params, dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        y_pred = model.predict(dval)
        return float(np.mean(np.abs(y_val - y_pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_mae = study.best_value
    best_slope = study.best_params["huber_slope"]
    return best_mae, best_slope


# ── Uncertainty Quantification (v4) ───────────────────────────────────

def _train_quantile_models(
    best_params: dict,
    best_n_estimators: int,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
    dtest: xgb.DMatrix,
    y_val: np.ndarray,
    y_test: np.ndarray,
    point_preds_test: np.ndarray,
) -> dict:
    """Train quantile regression models (q10, q90) for prediction intervals."""
    quantile_meta = {}

    for alpha, label in [(0.1, "q10"), (0.9, "q90")]:
        print(f"    Training {label} (alpha={alpha})...")
        params = {k: v for k, v in best_params.items()}
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = alpha

        model = xgb.train(
            params, dtrain,
            num_boost_round=best_n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )

        # Evaluate on test set
        q_pred = model.predict(dtest)
        path = ARTIFACTS_DIR / f"soc_trip_model_{label}.json"
        model.save_model(str(path))
        print(f"    Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")

        quantile_meta[label] = {
            "alpha": alpha,
            "test_mean": float(np.mean(q_pred)),
            "n_trees": model.num_boosted_rounds(),
        }

    # Verify ordering: q10 < point < q90 on test set
    q10_model = xgb.Booster()
    q10_model.load_model(str(ARTIFACTS_DIR / "soc_trip_model_q10.json"))
    q90_model = xgb.Booster()
    q90_model.load_model(str(ARTIFACTS_DIR / "soc_trip_model_q90.json"))

    q10_pred = q10_model.predict(dtest)
    q90_pred = q90_model.predict(dtest)

    ordering_ok = np.all(q10_pred <= q90_pred)
    coverage = np.mean((y_test >= q10_pred) & (y_test <= q90_pred))
    avg_width = np.mean(q90_pred - q10_pred)

    print(f"    Quantile ordering valid: {ordering_ok}")
    print(f"    Test coverage (80% target): {coverage*100:.1f}%")
    print(f"    Average interval width: {avg_width:.2f}% SOC")

    quantile_meta["test_coverage_80"] = float(coverage)
    quantile_meta["avg_interval_width"] = float(avg_width)
    quantile_meta["ordering_valid"] = bool(ordering_ok)

    return quantile_meta


def _calibrate_conformal(
    y_pred_val: np.ndarray,
    y_val: np.ndarray,
    y_pred_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    alpha: float = 0.20,
) -> dict:
    """Calibrate conformal prediction intervals using validation residuals."""
    residuals = np.abs(y_val - y_pred_val)
    n = len(residuals)

    # Split conformal quantile
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(residuals, q_level))

    # Test set coverage
    test_residuals = np.abs(y_test - y_pred_test)
    test_coverage = float(np.mean(test_residuals <= q_hat))

    print(f"    Conformal q_hat (80%): {q_hat:.3f}% SOC")
    print(f"    Val residuals: mean={residuals.mean():.3f}, p80={np.percentile(residuals, 80):.3f}")
    print(f"    Test coverage: {test_coverage*100:.1f}% (target: {(1-alpha)*100:.0f}%)")

    # Save conformal calibration
    conformal_data = {
        "conformal_q_hat_80": q_hat,
        "alpha": alpha,
        "val_n": n,
        "test_coverage": test_coverage,
    }
    conf_path = ARTIFACTS_DIR / f"conformal_{model_name}.json"
    with open(conf_path, "w") as f:
        json.dump(conformal_data, f, indent=2)
    print(f"    Saved: {conf_path.name}")

    return conformal_data


# ── Model 1A Per-Direction Experiment (v4) ────────────────────────────

def train_trip_per_direction(verbose: bool = True) -> dict:
    """Train separate upstream/downstream models for Model 1A and compare."""
    print("\n" + "=" * 60)
    print("  MODEL 1A: Per-Direction Experiment (v4)")
    print("=" * 60)

    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    features = TRIP_FEATURES_DIRECTIONAL
    mono_tuple = get_monotonic_constraint_tuple(features, TRIP_MONOTONIC_CONSTRAINTS)

    direction_labels = {0: "downstream", 1: "upstream"}
    dir_results = {}
    dir_models = {}
    all_test_preds = np.zeros(len(test_df))
    all_test_true = test_df[TRIP_TARGET].to_numpy().astype(np.float32)
    test_dirs = test_df["direction_encoded"].to_numpy()

    for dir_code, dir_name in direction_labels.items():
        print(f"\n  --- {dir_name.upper()} (direction={dir_code}) ---")

        # Filter by direction
        tr = train_df.filter(pl.col("direction_encoded") == dir_code)
        va = val_df.filter(pl.col("direction_encoded") == dir_code)
        te = test_df.filter(pl.col("direction_encoded") == dir_code)

        print(f"  Split: Train={len(tr)} | Val={len(va)} | Test={len(te)}")

        if len(tr) < 10 or len(va) < 3:
            print(f"  SKIP: insufficient data for {dir_name}")
            continue

        # Augment training data
        tr_aug = augment_trip_data(tr, features, TRIP_TARGET, verbose=verbose)

        dtrain, y_train = df_to_dmatrix(tr_aug, features, TRIP_TARGET)
        dval, y_val = df_to_dmatrix(va, features, TRIP_TARGET)
        dtest, y_test = df_to_dmatrix(te, features, TRIP_TARGET)

        # Optuna search
        print(f"\n  Optuna ({OPTUNA_CONFIG['n_trials']} trials)...")
        t0 = time.time()

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=OPTUNA_CONFIG["n_startup_trials"],
                seed=42,
            ),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
        )

        def objective(trial, dt=dtrain, dv=dval, mt=mono_tuple):
            ss = OPTUNA_CONFIG["search_space"]
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "mae",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
                "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
                "subsample": trial.suggest_float("subsample", *ss["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", *ss["colsample_bylevel"]),
                "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
                "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
                "gamma": trial.suggest_float("gamma", *ss["gamma"]),
                "monotone_constraints": mt,
                "seed": 42,
            }
            n_est = trial.suggest_int("n_estimators", *ss["n_estimators"])
            pruning_cb = optuna.integration.XGBoostPruningCallback(trial, "val-mae")
            model = xgb.train(
                params, dt, num_boost_round=n_est,
                evals=[(dv, "val")], early_stopping_rounds=50,
                verbose_eval=0, callbacks=[pruning_cb],
            )
            y_pred = model.predict(dv)
            return float(np.mean(np.abs(dv.get_label() - y_pred)))

        study.optimize(objective, n_trials=OPTUNA_CONFIG["n_trials"], show_progress_bar=True)
        best = study.best_trial
        elapsed = time.time() - t0
        print(f"  Optuna complete in {elapsed/60:.1f} min | Best val MAE: {best.value:.4f}%")

        # Ensemble training
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
            "monotone_constraints": mono_tuple,
        }
        best_n = best.params["n_estimators"]

        ensemble = []
        for seed in ENSEMBLE_SEEDS:
            params = {**best_params, "seed": seed}
            model = xgb.train(
                params, dtrain, num_boost_round=best_n,
                evals=[(dval, "val")], early_stopping_rounds=50,
                verbose_eval=0,
            )
            ensemble.append(model)
            print(f"    Seed {seed}: {model.num_boosted_rounds()} trees")

        # Evaluate
        y_pred_test = ensemble_predict(ensemble, dtest)
        y_pred_test = np.clip(y_pred_test, 0, None)
        metrics = compute_metrics(y_test, y_pred_test)
        print_metrics(f"{dir_name.capitalize()} Test", metrics)

        dir_results[dir_name] = metrics
        dir_models[dir_name] = ensemble

        # Fill predictions into combined array
        test_mask = test_dirs == dir_code
        all_test_preds[test_mask] = y_pred_test

        # Save models
        total_size = 0
        for i, m in enumerate(ensemble):
            if i == 0:
                path = ARTIFACTS_DIR / f"soc_trip_{dir_name}_model.json"
            else:
                path = ARTIFACTS_DIR / f"soc_trip_{dir_name}_model_seed{ENSEMBLE_SEEDS[i]}.json"
            m.save_model(str(path))
            total_size += path.stat().st_size
            print(f"    Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        print(f"    Total {dir_name} size: {total_size / 1024:.0f} KB")

    # Combined per-direction test metrics
    combined = compute_metrics(all_test_true, all_test_preds)
    print("\n" + "=" * 60)
    print("  PER-DIRECTION COMBINED TEST RESULTS")
    print("=" * 60)
    print_metrics("Combined Per-Direction", combined)

    # Save metadata
    meta = {
        "model": "soc_trip_per_direction_v4",
        "trained_at": datetime.now().isoformat(),
        "features": features,
        "target": TRIP_TARGET,
        "n_ensemble_seeds": N_ENSEMBLE_SEEDS,
        "per_direction_metrics": {k: v for k, v in dir_results.items()},
        "combined_test_metrics": combined,
    }
    meta_path = ARTIFACTS_DIR / "soc_trip_directional_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"per_direction": dir_results, "combined": combined}


# ── Model 1B: Real-Time SOC ──────────────────────────────────────────

def _realtime_optuna_objective(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
) -> float:
    """Optuna objective for Model 1B."""
    ss = OPTUNA_CONFIG["search_space"]

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *ss["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *ss["colsample_bylevel"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *ss["gamma"]),
        "seed": 42,
    }

    n_estimators = trial.suggest_int("n_estimators", *ss["n_estimators"])

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "val-mae")

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=0,
        callbacks=[pruning_callback],
    )

    y_val = dval.get_label()
    y_pred = model.predict(dval)
    mae = float(np.mean(np.abs(y_val - y_pred)))

    trial.set_user_attr("n_trees", model.num_boosted_rounds())
    return mae


def train_realtime_v2(verbose: bool = True) -> dict:
    """Train Model 1B with v2 optimizations."""
    print("\n" + "=" * 60)
    print("  MODEL 1B: Real-Time SOC Range Estimation (v2)")
    print("=" * 60)

    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    n_seg = {
        "train": train_df["segment_id"].n_unique(),
        "val": val_df["segment_id"].n_unique(),
        "test": test_df["segment_id"].n_unique(),
    }
    print(f"  Split: Train={len(train_df)} ({n_seg['train']} segs) | "
          f"Val={len(val_df)} ({n_seg['val']} segs) | "
          f"Test={len(test_df)} ({n_seg['test']} segs)")

    dtrain, y_train = df_to_dmatrix(train_df, REALTIME_FEATURES, REALTIME_TARGET)
    dval, y_val = df_to_dmatrix(val_df, REALTIME_FEATURES, REALTIME_TARGET)
    dtest, y_test = df_to_dmatrix(test_df, REALTIME_FEATURES, REALTIME_TARGET)

    # ── Optuna search ──
    print(f"\n  [1/3] Optuna ({OPTUNA_CONFIG['n_trials']} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=OPTUNA_CONFIG["n_startup_trials"],
            seed=42,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )

    study.optimize(
        lambda trial: _realtime_optuna_objective(trial, dtrain, dval),
        n_trials=OPTUNA_CONFIG["n_trials"],
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Ensemble training ──
    print(f"\n  [2/3] Ensemble Training ({N_ENSEMBLE_SEEDS} seeds)...")

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

    ensemble_models = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        params = {**best_params, "seed": seed}
        model = xgb.train(
            params, dtrain,
            num_boost_round=best_n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        ensemble_models.append(model)
        print(f"    Seed {seed}: {model.num_boosted_rounds()} trees")

    # ── Evaluate ──
    print("\n  [3/3] Evaluation...")

    results = {}
    for name, dm, y_true, split_df in [
        ("Train", dtrain, y_train, train_df),
        ("Validation", dval, y_val, val_df),
        ("Test", dtest, y_test, test_df),
    ]:
        y_pred = ensemble_predict(ensemble_models, dm)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics
        print_metrics(name, metrics)

        if name == "Test":
            _evaluate_by_progress_ensemble(ensemble_models, split_df)

    # Plots
    y_pred_test = ensemble_predict(ensemble_models, dtest)
    plot_predictions(
        y_test, y_pred_test,
        "Model 1B v2: Real-Time SOC (Ensemble)",
        save_path=ARTIFACTS_DIR / "soc_realtime_predictions_v2.png",
    )
    plot_feature_importance(
        ensemble_models[0], REALTIME_FEATURES,
        "Model 1B v2: Real-Time SOC",
        save_path=ARTIFACTS_DIR / "soc_realtime_importance_v2.png",
    )

    # Save ensemble models
    total_size = 0
    for i, model in enumerate(ensemble_models):
        if i == 0:
            path = ARTIFACTS_DIR / "soc_realtime_model.json"
        else:
            path = ARTIFACTS_DIR / f"soc_realtime_model_seed{ENSEMBLE_SEEDS[i]}.json"
        model.save_model(str(path))
        total_size += path.stat().st_size
        print(f"    Saved: {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    print(f"    Total Model 1B size: {total_size / 1024:.0f} KB")

    # ── Conformal Prediction (v4) ──
    print("\n  [4/4] Conformal Prediction Calibration...")
    y_pred_val = ensemble_predict(ensemble_models, dval)
    y_pred_test_rt = ensemble_predict(ensemble_models, dtest)
    conformal_meta = _calibrate_conformal(
        y_pred_val, y_val,
        y_pred_test_rt, y_test,
        model_name="realtime",
    )

    # Save metadata
    meta = {
        "model": "soc_realtime_v2",
        "trained_at": datetime.now().isoformat(),
        "features": REALTIME_FEATURES,
        "target": REALTIME_TARGET,
        "best_params": best.params,
        "n_ensemble_seeds": N_ENSEMBLE_SEEDS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "optuna_trials": OPTUNA_CONFIG["n_trials"],
        "optuna_best_val_mae": best.value,
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_segments": n_seg["train"],
            "val_segments": n_seg["val"],
            "test_segments": n_seg["test"],
        },
        "metrics": results,
        "conformal": conformal_meta,
    }
    meta_path = ARTIFACTS_DIR / "soc_realtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


def _evaluate_by_progress_ensemble(
    models: list[xgb.Booster],
    df: pl.DataFrame,
) -> None:
    """Evaluate ensemble MAE at different trip progress bins."""
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    print("\n  MAE by trip progress:")
    for lo, hi in bins:
        subset = df.filter(
            (pl.col("trip_progress_fraction") >= lo)
            & (pl.col("trip_progress_fraction") < hi)
        )
        if len(subset) == 0:
            continue
        dm, y_true = df_to_dmatrix(subset, REALTIME_FEATURES, REALTIME_TARGET)
        y_pred = ensemble_predict(models, dm)
        mae = np.mean(np.abs(y_true - y_pred))
        label = f"{int(lo*100)}-{int(hi*100)}%"
        print(f"    {label}: MAE={mae:.3f}% SOC  (n={len(subset)})")


# ── Model 2: Motor Anomaly Detection ─────────────────────────────────

def _anomaly_optuna_objective(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
) -> float:
    """Optuna objective for anomaly reconstruction models."""
    ss = OPTUNA_CONFIG["search_space"]

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", *ss["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *ss["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *ss["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *ss["colsample_bytree"]),
        "min_child_weight": trial.suggest_int("min_child_weight", *ss["min_child_weight"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *ss["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *ss["reg_lambda"], log=True),
        "gamma": trial.suggest_float("gamma", *ss["gamma"]),
        "seed": 42,
    }

    n_estimators = trial.suggest_int("n_estimators", 100, 2000)

    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "val-mae")

    model = xgb.train(
        params, dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=0,
        callbacks=[pruning_callback],
    )

    y_pred = model.predict(dval)
    y_val = dval.get_label()
    return float(np.mean(np.abs(y_val - y_pred)))


def train_anomaly_v2(verbose: bool = True) -> dict:
    """Train Model 2 with v2 optimizations."""
    print("\n" + "=" * 60)
    print("  MODEL 2: Motor Anomaly Detection (v2)")
    print("=" * 60)

    df = pl.read_parquet(FEATURE_STORE_DIR / "anomaly_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Compute feature statistics for normalisation
    feature_means = {}
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_means[feat] = float(np.mean(col))
        feature_stds[feat] = float(max(np.std(col), 0.01))

    models: dict[str, xgb.Booster] = {}
    train_errors: dict[str, np.ndarray] = {}

    for target_col in ANOMALY_TARGETS:
        print(f"\n  --- Reconstruction: {target_col} ---")

        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]
        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        # Optuna for this target (100 trials per reconstruction model)
        print(f"    Optuna (100 trials)...")
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
        print(f"    Best val MAE: {best.value:.4f}")

        # Train final model with best params
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

        # Compute training reconstruction error
        pred_train = model.predict(dtrain)
        error = ((pred_train - y_train) / feature_stds[target_col]) ** 2
        train_errors[target_col] = error

        val_pred = model.predict(dval)
        val_mae = np.mean(np.abs(y_val - val_pred))
        print(f"    Final val MAE: {val_mae:.4f}, trees: {model.num_boosted_rounds()}")

        # Save
        model_path = ARTIFACTS_DIR / f"anomaly_{target_col}.json"
        model.save_model(str(model_path))
        models[target_col] = model
        print(f"    Saved: {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")

    # ── Combined anomaly scores ──
    # Calibrate thresholds on VALIDATION data (not training) because
    # Optuna-optimized models reconstruct training data near-perfectly,
    # making training-based thresholds unrealistically small.
    def score_fn(data):
        return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)

    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    val_scores = score_fn(val_data)

    p95 = float(np.percentile(val_scores, 95))
    p97 = float(np.percentile(val_scores, 97))
    p99 = float(np.percentile(val_scores, 99))

    print(f"\n  Anomaly Thresholds (val-calibrated): p95={p95:.4f}  p97={p97:.4f}  p99={p99:.4f}")
    print(f"  Val scores: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}, "
          f"max={val_scores.max():.4f}")

    # Also compute training scores for the distribution plot
    n_train = len(train_df)
    combined_train_scores = np.zeros(n_train)
    for errors in train_errors.values():
        combined_train_scores += errors

    # ── Score test set ──
    test_data = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    test_scores = score_fn(test_data)
    n_flagged = (test_scores > p99).sum()
    print(f"  Test: mean={test_scores.mean():.4f}, max={test_scores.max():.4f}, "
          f"flagged@p99={n_flagged}/{len(test_scores)} "
          f"({n_flagged/len(test_scores)*100:.1f}%)")

    # ── Injection test ──
    print("\n  Synthetic Injection Test...")
    injection_results = synthetic_injection_test(
        score_fn, test_data, ANOMALY_FEATURES,
        n_injections=500, corruption_factor=1.5, threshold=p99,
    )
    print(f"  Detection rate (port power x1.5): {injection_results['detection_rate']:.1f}%")
    print(f"  False positive rate: {injection_results['false_positive_rate']:.1f}%")

    # Threshold sensitivity
    corrupted = test_data[np.random.RandomState(42).choice(len(test_data), 500, replace=True)].copy()
    port_idx = ANOMALY_FEATURES.index("port_motor_power")
    corrupted[:, port_idx] *= 1.5
    corrupted_scores = score_fn(corrupted)
    threshold_sensitivity(test_scores, corrupted_scores)

    # Plot — use validation scores for distribution since thresholds are val-calibrated
    plot_anomaly_distribution(
        val_scores, p99,
        title="Motor Anomaly Score Distribution (v2, Val-Calibrated)",
        save_path=ARTIFACTS_DIR / "anomaly_score_distribution_v2.png",
    )

    # ── Save metadata ──
    thresholds = {"p95": p95, "p97": p97, "p99": p99}
    stats_out = {"feature_means": feature_means, "feature_stds": feature_stds}

    with open(ARTIFACTS_DIR / "feature_statistics.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    with open(ARTIFACTS_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    total_anomaly_size = sum(
        (ARTIFACTS_DIR / f"anomaly_{t}.json").stat().st_size
        for t in ANOMALY_TARGETS
    )
    print(f"\n  Total anomaly model size: {total_anomaly_size / 1024:.0f} KB")

    meta = {
        "model": "anomaly_reconstruction_v2",
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "targets": ANOMALY_TARGETS,
        "thresholds": thresholds,
        "optuna_trials_per_target": 100,
        "injection_test": injection_results,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    with open(ARTIFACTS_DIR / "anomaly_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"thresholds": thresholds, "injection": injection_results}


def _compute_scores_from_array(
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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V2 Optimized Training Pipeline")
    parser.add_argument("--model", choices=["trip", "realtime", "anomaly", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--direction", action="store_true",
                        help="Run per-direction experiment for Model 1A")
    parser.add_argument("--shap", action="store_true",
                        help="Run SHAP analysis after training (requires shap package)")
    args = parser.parse_args()

    start_time = time.time()
    all_results = {}

    if args.model in ("trip", "all"):
        all_results["trip"] = train_trip_v2()

    if args.direction:
        all_results["direction"] = train_trip_per_direction()

    if args.model in ("realtime", "all"):
        all_results["realtime"] = train_realtime_v2()

    if args.model in ("anomaly", "all"):
        all_results["anomaly"] = train_anomaly_v2()

    if args.shap:
        print("\n  Running SHAP analysis...")
        try:
            from evaluate.shap_analysis import run_shap_analysis
            run_shap_analysis()
        except ImportError:
            print("  ERROR: shap package not installed. Run: pip install shap")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Total training time: {total_time/60:.1f} min")
    print(f"{'=' * 60}")

    # Report bundle size
    model_files = list(ARTIFACTS_DIR.glob("*.json"))
    total_disk = sum(f.stat().st_size for f in model_files if "model" in f.name)
    print(f"  Total model files on disk: {total_disk / 1024:.0f} KB ({total_disk / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
