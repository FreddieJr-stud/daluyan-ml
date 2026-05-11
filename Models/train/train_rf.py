"""Random Forest Training Pipeline for M/B Dalaray SOC models.

Fair comparison against XGBoost v4:
  - Same features, targets, temporal splits, and augmentation
  - Optuna Bayesian optimization (150 trials)
  - No multi-seed ensemble (RF is already a bagged ensemble)
  - Evaluated with identical compute_metrics()

Usage:
    python -m train.train_rf --model trip        # Model 1A only
    python -m train.train_rf --model realtime    # Model 1B only
    python -m train.train_rf --model all         # Both models
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
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    FEATURE_STORE_DIR,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    RF_OPTUNA_CONFIG,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_TARGET,
    VAL_END,
)
from data.augment import augment_trip_data
from evaluate.evaluate_soc import (
    compute_metrics,
    print_direction_breakdown,
    print_metrics,
)

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Shared Utilities ────────────────────────────────────────────────────

def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split using config dates (identical to train_v2.py)."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def df_to_numpy(
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to numpy arrays."""
    X = df.select(features).to_numpy().astype(np.float32)
    y = df[target].to_numpy().astype(np.float32)
    return X, y


def plot_rf_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    title: str,
    save_path: Path | None = None,
    top_n: int = 20,
) -> None:
    """Plot RF feature importance (impurity-based)."""
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y_pos = range(len(names))
    ax.barh(y_pos, values, align="center", color="forestgreen")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Impurity-based Importance")
    ax.set_title(f"{title} - Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Importance plot saved to: {save_path}")
    plt.close(fig)


# ── Model 1A: Trip-Level SOC (Random Forest) ────────────────────────────

def _trip_rf_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for RF Model 1A."""
    max_features_choice = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"]
    )
    if max_features_choice in ("sqrt", "log2"):
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=max_features,
        max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = float(np.mean(np.abs(y_val - y_pred)))

    trial.set_user_attr("n_estimators_actual", model.n_estimators)
    return mae


def train_trip_rf(verbose: bool = True) -> dict:
    """Train Model 1A (trip-level SOC) with Random Forest."""
    print("\n" + "=" * 60)
    print("  MODEL 1A: Trip-Level SOC Prediction (Random Forest)")
    print("=" * 60)

    # Load & split
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Data augmentation (same as XGB)
    print("\n  [1/3] Data Augmentation...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=verbose)

    X_train, y_train = df_to_numpy(train_aug, TRIP_FEATURES, TRIP_TARGET)
    X_train_orig, y_train_orig = df_to_numpy(train_df, TRIP_FEATURES, TRIP_TARGET)
    X_val, y_val = df_to_numpy(val_df, TRIP_FEATURES, TRIP_TARGET)
    X_test, y_test = df_to_numpy(test_df, TRIP_FEATURES, TRIP_TARGET)

    # ── Optuna search ──
    n_trials = RF_OPTUNA_CONFIG["n_trials"]
    print(f"\n  [2/3] Optuna Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=RF_OPTUNA_CONFIG["n_startup_trials"],
            seed=42,
        ),
    )

    study.optimize(
        lambda trial: _trip_rf_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model with best params ──
    print("\n  [3/3] Training final model + Evaluation...")

    max_features_choice = best.params["max_features"]
    if max_features_choice in ("sqrt", "log2"):
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    final_model = RandomForestRegressor(
        n_estimators=best.params["n_estimators"],
        max_depth=best.params["max_depth"],
        min_samples_split=best.params["min_samples_split"],
        min_samples_leaf=best.params["min_samples_leaf"],
        max_features=max_features,
        max_samples=best.params["max_samples"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # ── Evaluate ──
    results = {}
    for name, X, y_true, split_df in [
        ("Train", X_train_orig, y_train_orig, train_df),
        ("Validation", X_val, y_val, val_df),
        ("Test", X_test, y_test, test_df),
    ]:
        y_pred = np.clip(final_model.predict(X), 0, None)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics
        print_metrics(name, metrics)
        if "direction_encoded" in split_df.columns:
            dirs = split_df["direction_encoded"].to_numpy()
            print_direction_breakdown(y_true, y_pred, dirs)

    # Feature importance plot
    plot_rf_feature_importance(
        final_model, TRIP_FEATURES,
        "Model 1A RF: Trip-Level SOC",
        save_path=ARTIFACTS_DIR / "rf_soc_trip_importance.png",
    )

    # Save model
    import joblib
    model_path = ARTIFACTS_DIR / "rf_soc_trip_model.joblib"
    joblib.dump(final_model, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "model": "rf_soc_trip",
        "architecture": "RandomForest",
        "trained_at": datetime.now().isoformat(),
        "features": TRIP_FEATURES,
        "target": TRIP_TARGET,
        "best_params": best.params,
        "optuna_trials": n_trials,
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
        "model_size_bytes": model_size,
    }
    meta_path = ARTIFACTS_DIR / "rf_soc_trip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


# ── Model 1B: Real-Time SOC (Random Forest) ─────────────────────────────

def _realtime_rf_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for RF Model 1B."""
    max_features_choice = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"]
    )
    if max_features_choice in ("sqrt", "log2"):
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=max_features,
        max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = float(np.mean(np.abs(y_val - y_pred)))
    return mae


def _evaluate_by_progress_rf(
    model: RandomForestRegressor,
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> dict[str, float]:
    """Evaluate RF MAE at different trip progress bins."""
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    progress_mae = {}
    print("\n  MAE by trip progress:")
    for lo, hi in bins:
        subset = df.filter(
            (pl.col("trip_progress_fraction") >= lo)
            & (pl.col("trip_progress_fraction") < hi)
        )
        if len(subset) == 0:
            continue
        X, y_true = df_to_numpy(subset, features, target)
        y_pred = model.predict(X)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        label = f"{int(lo*100)}-{int(hi*100)}%"
        progress_mae[label] = mae
        print(f"    {label}: MAE={mae:.3f}% SOC  (n={len(subset)})")
    return progress_mae


def train_realtime_rf(verbose: bool = True) -> dict:
    """Train Model 1B (real-time SOC) with Random Forest."""
    print("\n" + "=" * 60)
    print("  MODEL 1B: Real-Time SOC Range Estimation (Random Forest)")
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

    X_train, y_train = df_to_numpy(train_df, REALTIME_FEATURES, REALTIME_TARGET)
    X_val, y_val = df_to_numpy(val_df, REALTIME_FEATURES, REALTIME_TARGET)
    X_test, y_test = df_to_numpy(test_df, REALTIME_FEATURES, REALTIME_TARGET)

    # ── Optuna search ──
    n_trials = RF_OPTUNA_CONFIG["n_trials"]
    print(f"\n  [1/3] Optuna Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=RF_OPTUNA_CONFIG["n_startup_trials"],
            seed=42,
        ),
    )

    study.optimize(
        lambda trial: _realtime_rf_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model ──
    print(f"\n  [2/3] Training final model...")

    max_features_choice = best.params["max_features"]
    if max_features_choice in ("sqrt", "log2"):
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    final_model = RandomForestRegressor(
        n_estimators=best.params["n_estimators"],
        max_depth=best.params["max_depth"],
        min_samples_split=best.params["min_samples_split"],
        min_samples_leaf=best.params["min_samples_leaf"],
        max_features=max_features,
        max_samples=best.params["max_samples"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # ── Evaluate ──
    print("\n  [3/3] Evaluation...")

    results = {}
    for name, X, y_true, split_df in [
        ("Train", X_train, y_train, train_df),
        ("Validation", X_val, y_val, val_df),
        ("Test", X_test, y_test, test_df),
    ]:
        y_pred = final_model.predict(X)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics
        print_metrics(name, metrics)

        if name == "Test":
            progress_mae = _evaluate_by_progress_rf(
                final_model, split_df, REALTIME_FEATURES, REALTIME_TARGET,
            )
            results["progress_breakdown"] = progress_mae

    # Feature importance plot
    plot_rf_feature_importance(
        final_model, REALTIME_FEATURES,
        "Model 1B RF: Real-Time SOC",
        save_path=ARTIFACTS_DIR / "rf_soc_realtime_importance.png",
    )

    # Save model
    import joblib
    model_path = ARTIFACTS_DIR / "rf_soc_realtime_model.joblib"
    joblib.dump(final_model, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / (1024*1024):.1f} MB)")

    # Save metadata
    meta = {
        "model": "rf_soc_realtime",
        "architecture": "RandomForest",
        "trained_at": datetime.now().isoformat(),
        "features": REALTIME_FEATURES,
        "target": REALTIME_TARGET,
        "best_params": best.params,
        "optuna_trials": n_trials,
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
        "model_size_bytes": model_size,
    }
    meta_path = ARTIFACTS_DIR / "rf_soc_realtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random Forest Training Pipeline")
    parser.add_argument(
        "--model", choices=["trip", "realtime", "all"],
        default="all", help="Which model to train",
    )
    args = parser.parse_args()

    start_time = time.time()
    all_results = {}

    if args.model in ("trip", "all"):
        all_results["trip"] = train_trip_rf()

    if args.model in ("realtime", "all"):
        all_results["realtime"] = train_realtime_rf()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Total RF training time: {total_time/60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
