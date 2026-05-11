"""Train Model 1A: Trip-level SOC consumption prediction.

Usage:
    python -m train.train_soc_trip
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
    FEATURE_STORE_DIR,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_TARGET,
    VAL_END,
    XGBOOST_TRIP_PARAMS,
)
from evaluate.evaluate_soc import (
    compute_metrics,
    plot_feature_importance,
    plot_predictions,
    print_direction_breakdown,
    print_metrics,
)


def load_trip_data() -> pl.DataFrame:
    """Load the trip feature parquet from the feature store."""
    path = FEATURE_STORE_DIR / "trip_features.parquet"
    if not path.exists():
        print("Feature store not found. Run `python -m data.extract_trip_features` first.")
        sys.exit(1)
    return pl.read_parquet(path)


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split data temporally into train / val / test."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def df_to_dmatrix(df: pl.DataFrame) -> tuple[xgb.DMatrix, np.ndarray]:
    """Convert a DataFrame to XGBoost DMatrix + target array."""
    X = df.select(TRIP_FEATURES).to_numpy().astype(np.float32)
    y = df[TRIP_TARGET].to_numpy().astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=TRIP_FEATURES), y


def train(
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> tuple[xgb.Booster, dict]:
    """Train Model 1A and return (model, results_dict)."""
    if params is None:
        params = XGBOOST_TRIP_PARAMS.copy()

    # Load & split
    df = load_trip_data()
    train_df, val_df, test_df = temporal_split(df)

    if verbose:
        print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    dtrain, y_train = df_to_dmatrix(train_df)
    dval, y_val = df_to_dmatrix(val_df)
    dtest, y_test = df_to_dmatrix(test_df)

    # Train
    evals = [(dtrain, "train"), (dval, "val")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=25 if verbose else 0,
    )

    # Evaluate
    results = {}
    for name, dm, y_true, split_df in [
        ("Train", dtrain, y_train, train_df),
        ("Validation", dval, y_val, val_df),
        ("Test", dtest, y_test, test_df),
    ]:
        y_pred = model.predict(dm)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics

        if verbose:
            print_metrics(name, metrics)

            dirs = split_df["direction_encoded"].to_numpy()
            print_direction_breakdown(y_true, y_pred, dirs)

    # Plots (test set)
    if verbose:
        y_pred_test = model.predict(dtest)
        plot_predictions(
            y_test, y_pred_test,
            "Model 1A: Trip-Level SOC",
            save_path=ARTIFACTS_DIR / "soc_trip_predictions.png",
        )
        plot_feature_importance(
            model, TRIP_FEATURES,
            "Model 1A: Trip-Level SOC",
            save_path=ARTIFACTS_DIR / "soc_trip_importance.png",
        )

    # Save model
    model_path = ARTIFACTS_DIR / "soc_trip_model.json"
    model.save_model(str(model_path))
    if verbose:
        print(f"\nModel saved to: {model_path}")
        print(f"Best iteration: {model.best_iteration}")
        print(f"Number of trees: {model.num_boosted_rounds()}")

    # Save training metadata
    meta = {
        "model": "soc_trip_v1",
        "trained_at": datetime.now().isoformat(),
        "features": TRIP_FEATURES,
        "target": TRIP_TARGET,
        "params": params,
        "best_iteration": model.best_iteration,
        "num_trees": model.num_boosted_rounds(),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "metrics": results,
    }
    meta_path = ARTIFACTS_DIR / "soc_trip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return model, results


if __name__ == "__main__":
    train()
