"""Training pipeline for Model 3: Battery Anomaly Detection.

Uses reconstruction-error paradigm (same as Model 2 motor anomaly):
  - Model 3a: Charging battery anomaly (abnormal charging curves, thermal events)
  - Model 3b: Operational battery anomaly (cell degradation, thermal issues during trips)

Usage:
    python -m train.train_battery --model charging      # Model 3a only
    python -m train.train_battery --model operational   # Model 3b only
    python -m train.train_battery --model all           # Both models
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
    ARTIFACTS_DIR,
    BATTERY_ANOMALY_CONFIG,
    BATTERY_FEATURES,
    BATTERY_TARGETS_CHARGING,
    BATTERY_TARGETS_OPERATIONAL,
    FEATURE_STORE_DIR,
    OPTUNA_CONFIG,
    TRAIN_END,
    VAL_END,
)
from evaluate.evaluate_anomaly import (
    plot_anomaly_distribution,
    synthetic_injection_test,
    threshold_sensitivity,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split using config dates."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def _battery_optuna_objective(
    trial: optuna.Trial,
    dtrain: xgb.DMatrix,
    dval: xgb.DMatrix,
) -> float:
    """Optuna objective for battery reconstruction models."""
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


def _battery_injection_test(
    compute_score_fn,
    normal_data: np.ndarray,
    feature_names: list[str],
    n_injections: int = 500,
    threshold: float | None = None,
) -> dict:
    """Test battery anomaly detection by injecting realistic degradation.

    Simulates cell imbalance: sets cell voltage spread to 0.2V (degraded cell)
    and increases cell temperature spread to 10C (thermal hotspot).
    Real battery anomalies affect multiple correlated features simultaneously.
    """
    normal_scores = compute_score_fn(normal_data)

    if threshold is None:
        threshold = float(np.percentile(normal_scores, 95))

    rng = np.random.RandomState(42)
    indices = rng.choice(len(normal_data), size=n_injections, replace=True)
    corrupted = normal_data[indices].copy()

    # Inject realistic cell degradation: set cell voltage spread to 0.15-0.25V
    # (normal is ~0.008-0.016V, degraded cell would be 0.1-0.3V)
    degrade_targets = {
        "port_cell_v_spread": 0.20,     # 0.20V (vs normal ~0.008V = 25x)
        "stbd_cell_v_spread": 0.15,     # asymmetric — port battery degrades first
        "cell_v_spread_combined": 0.20,
    }
    # Also inject thermal hotspot
    thermal_targets = {
        "port_cell_t_spread": 10.0,     # 10C spread (vs normal ~2-5C)
        "cell_t_spread_combined": 10.0,
    }

    all_injections = {**degrade_targets, **thermal_targets}
    injected_features = []
    for feat, value in all_injections.items():
        if feat in feature_names:
            idx = feature_names.index(feat)
            # Add noise around the injection value
            corrupted[:, idx] = value + rng.normal(0, value * 0.1, n_injections)
            injected_features.append(feat)

    corrupted_scores = compute_score_fn(corrupted)

    detection_rate = float((corrupted_scores > threshold).sum() / n_injections * 100)
    false_positive_rate = float((normal_scores > threshold).sum() / len(normal_scores) * 100)

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "threshold": float(threshold),
        "n_injections": n_injections,
        "injection_type": "multi-feature cell degradation + thermal hotspot",
        "injected_features": injected_features,
    }


def train_battery_model(
    mode: str,
    verbose: bool = True,
) -> dict:
    """Train a battery anomaly model (charging or operational).

    Args:
        mode: "charging" (Model 3a) or "operational" (Model 3b)
        verbose: print progress

    Returns:
        dict with thresholds and injection test results
    """
    prefix = f"battery_{mode}"
    targets = (
        BATTERY_TARGETS_CHARGING if mode == "charging"
        else BATTERY_TARGETS_OPERATIONAL
    )
    n_trials = BATTERY_ANOMALY_CONFIG["n_trials"]
    n_startup = BATTERY_ANOMALY_CONFIG["n_startup_trials"]

    print(f"\n{'=' * 60}")
    print(f"  MODEL 3{'a' if mode == 'charging' else 'b'}: "
          f"Battery Anomaly Detection ({mode.title()})")
    print(f"{'=' * 60}")

    # Load feature store
    parquet_path = FEATURE_STORE_DIR / f"bms_{mode}_features.parquet"
    if not parquet_path.exists():
        print(f"  ERROR: {parquet_path} not found. Run: python -m data.extract_bms_features")
        return {}

    df = pl.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} rows from {parquet_path.name}")

    # Check which features are available
    available_features = [f for f in BATTERY_FEATURES if f in df.columns]
    missing = [f for f in BATTERY_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: Missing features (will skip): {missing}")

    # Check which targets are available
    available_targets = [t for t in targets if t in df.columns]
    if not available_targets:
        print(f"  ERROR: No target features available! Have: {df.columns}")
        return {}
    if len(available_targets) < len(targets):
        print(f"  WARNING: Only {len(available_targets)}/{len(targets)} targets available")

    features = available_features
    targets = available_targets

    # Temporal split
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    if len(train_df) < 50:
        print(f"  ERROR: Insufficient training data ({len(train_df)} rows)")
        return {}
    if len(val_df) < 20:
        print(f"  ERROR: Insufficient validation data ({len(val_df)} rows)")
        return {}

    # Compute feature statistics on training set
    feature_means = {}
    feature_stds = {}
    for feat in features:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_means[feat] = float(np.mean(col))
        feature_stds[feat] = float(max(np.std(col), 0.001))

    # Train reconstruction models
    models: dict[str, xgb.Booster] = {}
    train_errors: dict[str, np.ndarray] = {}

    for target_col in targets:
        print(f"\n  --- Reconstruction: {target_col} ---")

        predictor_cols = [c for c in features if c != target_col]
        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        # Optuna
        print(f"    Optuna ({n_trials} trials)...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
        )
        study.optimize(
            lambda trial: _battery_optuna_objective(trial, dtrain, dval),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_trial
        print(f"    Best val MAE: {best.value:.6f}")

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
        print(f"    Final val MAE: {val_mae:.6f}, trees: {model.num_boosted_rounds()}")

        # Save
        model_path = ARTIFACTS_DIR / f"{prefix}_{target_col}.json"
        model.save_model(str(model_path))
        models[target_col] = model
        print(f"    Saved: {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")

    # ── Combined anomaly scores ──
    def score_fn(data):
        return _compute_scores_from_array(models, data, features, feature_stds)

    val_data = val_df.select(features).to_numpy().astype(np.float32)
    val_scores = score_fn(val_data)

    p95 = float(np.percentile(val_scores, 95))
    p97 = float(np.percentile(val_scores, 97))
    p99 = float(np.percentile(val_scores, 99))

    print(f"\n  Anomaly Thresholds (val-calibrated): p95={p95:.4f}  p97={p97:.4f}  p99={p99:.4f}")
    print(f"  Val scores: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}, "
          f"max={val_scores.max():.4f}")

    # ── Score test set ──
    test_data = test_df.select(features).to_numpy().astype(np.float32)
    test_scores = score_fn(test_data)
    n_flagged = int((test_scores > p99).sum())
    print(f"  Test: mean={test_scores.mean():.4f}, max={test_scores.max():.4f}, "
          f"flagged@p99={n_flagged}/{len(test_scores)} "
          f"({n_flagged/len(test_scores)*100:.1f}%)")

    # ── Injection test ──
    # Multi-feature degradation: simulate cell imbalance + thermal hotspot
    print(f"\n  Synthetic Injection Test (cell degradation + thermal hotspot)...")
    injection_results = _battery_injection_test(
        score_fn, test_data, features,
        n_injections=500, threshold=p99,
    )
    print(f"  Injected features: {injection_results['injected_features']}")
    print(f"  Detection rate: {injection_results['detection_rate']:.1f}%")
    print(f"  False positive rate: {injection_results['false_positive_rate']:.1f}%")

    # Threshold sensitivity with the same injected data
    rng = np.random.RandomState(42)
    indices = rng.choice(len(test_data), 500, replace=True)
    corrupted = test_data[indices].copy()
    inject_map = {"port_cell_v_spread": 0.20, "stbd_cell_v_spread": 0.15,
                  "cell_v_spread_combined": 0.20, "port_cell_t_spread": 10.0,
                  "cell_t_spread_combined": 10.0}
    for feat, val in inject_map.items():
        if feat in features:
            idx = features.index(feat)
            corrupted[:, idx] = val + rng.normal(0, val * 0.1, 500)
    corrupted_scores = score_fn(corrupted)
    threshold_sensitivity(test_scores, corrupted_scores)

    # Plot
    plot_anomaly_distribution(
        val_scores, p99,
        title=f"Battery Anomaly Score ({mode.title()}, Val-Calibrated)",
        save_path=ARTIFACTS_DIR / f"{prefix}_score_distribution.png",
    )

    # ── Save metadata ──
    thresholds = {"p95": p95, "p97": p97, "p99": p99}

    with open(ARTIFACTS_DIR / f"{prefix}_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    stats_out = {"feature_means": feature_means, "feature_stds": feature_stds}
    with open(ARTIFACTS_DIR / f"{prefix}_statistics.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    total_model_size = sum(
        (ARTIFACTS_DIR / f"{prefix}_{t}.json").stat().st_size
        for t in targets
    )
    print(f"\n  Total model size: {total_model_size / 1024:.0f} KB")

    meta = {
        "model": f"battery_anomaly_{mode}",
        "trained_at": datetime.now().isoformat(),
        "features": features,
        "targets": targets,
        "thresholds": thresholds,
        "optuna_trials_per_target": n_trials,
        "injection_test": injection_results,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "test_scores": {
            "mean": float(test_scores.mean()),
            "std": float(test_scores.std()),
            "max": float(test_scores.max()),
            "flagged_p99": n_flagged,
        },
    }
    with open(ARTIFACTS_DIR / f"{prefix}_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return {"thresholds": thresholds, "injection": injection_results}


def main():
    parser = argparse.ArgumentParser(description="Battery Anomaly Detection Training")
    parser.add_argument(
        "--model",
        choices=["charging", "operational", "all"],
        default="all",
        help="Which battery anomaly model to train",
    )
    args = parser.parse_args()

    start_time = time.time()
    results = {}

    if args.model in ("charging", "all"):
        results["charging"] = train_battery_model("charging")

    if args.model in ("operational", "all"):
        results["operational"] = train_battery_model("operational")

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Total battery training time: {total_time/60:.1f} min")
    print(f"{'=' * 60}")

    # Report model sizes
    for mode in ("charging", "operational"):
        model_files = list(ARTIFACTS_DIR.glob(f"battery_{mode}_*.json"))
        if model_files:
            total_disk = sum(f.stat().st_size for f in model_files)
            print(f"  {mode.title()} artifacts: {total_disk / 1024:.0f} KB")


if __name__ == "__main__":
    main()
