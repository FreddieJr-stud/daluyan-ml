"""Train Model 2: Motor anomaly detection via XGBoost reconstruction error.

Trains 4 XGBoost models, each predicting one motor feature from the
others.  The reconstruction error (sum of squared normalised errors)
serves as the anomaly score.

Usage:
    python -m train.train_anomaly
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
    ANOMALY_FEATURES,
    ANOMALY_TARGETS,
    ARTIFACTS_DIR,
    FEATURE_STORE_DIR,
    TRAIN_END,
    VAL_END,
    XGBOOST_ANOMALY_PARAMS,
)
from evaluate.evaluate_anomaly import (
    plot_anomaly_distribution,
    synthetic_injection_test,
    threshold_sensitivity,
)


def load_anomaly_data() -> pl.DataFrame:
    path = FEATURE_STORE_DIR / "anomaly_features.parquet"
    if not path.exists():
        print("Feature store not found. Run `python -m data.extract_anomaly_features` first.")
        sys.exit(1)
    return pl.read_parquet(path)


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def train(verbose: bool = True) -> tuple[dict[str, xgb.Booster], dict]:
    """Train reconstruction models and compute anomaly statistics."""
    df = load_anomaly_data()
    train_df, val_df, test_df = temporal_split(df)

    if verbose:
        print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # Compute feature statistics for normalisation of reconstruction error
    feature_means = {}
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_means[feat] = float(np.mean(col))
        feature_stds[feat] = float(max(np.std(col), 0.01))  # floor at 0.01

    models: dict[str, xgb.Booster] = {}
    train_errors: dict[str, np.ndarray] = {}

    for target_col in ANOMALY_TARGETS:
        if verbose:
            print(f"\n--- Training reconstruction model for: {target_col} ---")

        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]

        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=predictor_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=predictor_cols)

        params = XGBOOST_ANOMALY_PARAMS.copy()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=50 if verbose else 0,
        )

        # Compute train reconstruction error for this target
        pred_train = model.predict(dtrain)
        error = ((pred_train - y_train) / feature_stds[target_col]) ** 2
        train_errors[target_col] = error

        if verbose:
            val_pred = model.predict(dval)
            val_mae = np.mean(np.abs(y_val - val_pred))
            print(f"  Val MAE for {target_col}: {val_mae:.3f}")

        # Save model
        model_path = ARTIFACTS_DIR / f"anomaly_{target_col}.json"
        model.save_model(str(model_path))
        models[target_col] = model

        if verbose:
            print(f"  Saved: {model_path}")

    # ── Compute combined anomaly scores ──
    # Sum of squared normalised errors across all reconstruction targets
    n_train = len(train_df)
    combined_train_scores = np.zeros(n_train)
    for target_col, errors in train_errors.items():
        combined_train_scores += errors

    # Thresholds from training data
    p95 = float(np.percentile(combined_train_scores, 95))
    p97 = float(np.percentile(combined_train_scores, 97))
    p99 = float(np.percentile(combined_train_scores, 99))

    if verbose:
        print(f"\n{'=' * 50}")
        print("  Anomaly Score Statistics (Training Data)")
        print(f"{'=' * 50}")
        print(f"  Mean:   {combined_train_scores.mean():.3f}")
        print(f"  Std:    {combined_train_scores.std():.3f}")
        print(f"  p95:    {p95:.3f}")
        print(f"  p97:    {p97:.3f}")
        print(f"  p99:    {p99:.3f}")
        print(f"  Max:    {combined_train_scores.max():.3f}")

    # ── Score test set ──
    test_scores = _compute_scores(models, test_df, feature_stds)

    if verbose:
        print(f"\n  Test set scores: mean={test_scores.mean():.3f}, "
              f"max={test_scores.max():.3f}")
        n_flagged_99 = (test_scores > p99).sum()
        print(f"  Flagged at p99 threshold: {n_flagged_99}/{len(test_scores)} "
              f"({n_flagged_99/len(test_scores)*100:.1f}%)")

    # ── Synthetic injection test ──
    if verbose:
        print("\n--- Synthetic Anomaly Injection Test ---")

        test_features = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

        def score_fn(data: np.ndarray) -> np.ndarray:
            return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)

        injection_results = synthetic_injection_test(
            score_fn,
            test_features,
            ANOMALY_FEATURES,
            n_injections=500,
            corruption_factor=1.5,
            threshold=p99,
        )
        print(f"  Detection rate (port power x1.5): {injection_results['detection_rate']:.1f}%")
        print(f"  False positive rate: {injection_results['false_positive_rate']:.1f}%")

        # Threshold sensitivity
        corrupted = test_features[np.random.RandomState(42).choice(len(test_features), 500, replace=True)].copy()
        port_idx = ANOMALY_FEATURES.index("port_motor_power")
        corrupted[:, port_idx] *= 1.5
        corrupted_scores = score_fn(corrupted)
        threshold_sensitivity(test_scores, corrupted_scores)

        # Plot
        plot_anomaly_distribution(
            combined_train_scores,
            p99,
            title="Motor Anomaly Score Distribution (Training Data)",
            save_path=ARTIFACTS_DIR / "anomaly_score_distribution.png",
        )

    # ── Save metadata ──
    thresholds = {"p95": p95, "p97": p97, "p99": p99}
    stats_out = {
        "feature_means": feature_means,
        "feature_stds": feature_stds,
    }

    with open(ARTIFACTS_DIR / "feature_statistics.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    with open(ARTIFACTS_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    meta = {
        "model": "anomaly_reconstruction_v1",
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "targets": ANOMALY_TARGETS,
        "thresholds": thresholds,
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    if verbose:
        meta["injection_test"] = injection_results
    with open(ARTIFACTS_DIR / "anomaly_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        print(f"\n  Artifacts saved to: {ARTIFACTS_DIR}")

    return models, meta


def _compute_scores(
    models: dict[str, xgb.Booster],
    df: pl.DataFrame,
    feature_stds: dict[str, float],
) -> np.ndarray:
    """Compute combined anomaly scores for a DataFrame."""
    data = df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    return _compute_scores_from_array(models, data, ANOMALY_FEATURES, feature_stds)


def _compute_scores_from_array(
    models: dict[str, xgb.Booster],
    data: np.ndarray,
    feature_names: list[str],
    feature_stds: dict[str, float],
) -> np.ndarray:
    """Compute combined anomaly scores from a numpy array."""
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


if __name__ == "__main__":
    train()
