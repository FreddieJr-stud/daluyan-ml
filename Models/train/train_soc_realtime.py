"""Train Model 1B: Real-time SOC range estimation.

Usage:
    python -m train.train_soc_realtime
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
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
    VAL_END,
    XGBOOST_REALTIME_PARAMS,
)
from evaluate.evaluate_soc import (
    compute_metrics,
    plot_feature_importance,
    plot_predictions,
    print_metrics,
)


def load_realtime_data() -> pl.DataFrame:
    path = FEATURE_STORE_DIR / "realtime_features.parquet"
    if not path.exists():
        print("Feature store not found. Run `python -m data.extract_realtime_features` first.")
        sys.exit(1)
    return pl.read_parquet(path)


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split — all rows from a segment stay in the same split."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def df_to_dmatrix(df: pl.DataFrame) -> tuple[xgb.DMatrix, np.ndarray]:
    X = df.select(REALTIME_FEATURES).to_numpy().astype(np.float32)
    y = df[REALTIME_TARGET].to_numpy().astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=REALTIME_FEATURES), y


def evaluate_by_progress(
    model: xgb.Booster,
    df: pl.DataFrame,
    progress_bins: list[tuple[float, float]] = None,
) -> None:
    """Evaluate MAE at different trip progress points."""
    if progress_bins is None:
        progress_bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]

    print("\n  MAE by trip progress:")
    for lo, hi in progress_bins:
        subset = df.filter(
            (pl.col("trip_progress_fraction") >= lo)
            & (pl.col("trip_progress_fraction") < hi)
        )
        if len(subset) == 0:
            continue
        dm, y_true = df_to_dmatrix(subset)
        y_pred = model.predict(dm)
        mae = np.mean(np.abs(y_true - y_pred))
        label = f"{int(lo*100)}-{int(hi*100)}%"
        print(f"    {label}: MAE={mae:.3f}% SOC  (n={len(subset)})")


def train(
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> tuple[xgb.Booster, dict]:
    if params is None:
        params = XGBOOST_REALTIME_PARAMS.copy()

    df = load_realtime_data()
    train_df, val_df, test_df = temporal_split(df)

    if verbose:
        n_seg_train = train_df["segment_id"].n_unique()
        n_seg_val = val_df["segment_id"].n_unique()
        n_seg_test = test_df["segment_id"].n_unique()
        print(f"Train: {len(train_df)} rows ({n_seg_train} segments)")
        print(f"Val:   {len(val_df)} rows ({n_seg_val} segments)")
        print(f"Test:  {len(test_df)} rows ({n_seg_test} segments)")

    dtrain, y_train = df_to_dmatrix(train_df)
    dval, y_val = df_to_dmatrix(val_df)
    dtest, y_test = df_to_dmatrix(test_df)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=25 if verbose else 0,
    )

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
            if name == "Test":
                evaluate_by_progress(model, split_df)

    if verbose:
        y_pred_test = model.predict(dtest)
        plot_predictions(
            y_test, y_pred_test,
            "Model 1B: Real-Time SOC",
            save_path=ARTIFACTS_DIR / "soc_realtime_predictions.png",
        )
        plot_feature_importance(
            model, REALTIME_FEATURES,
            "Model 1B: Real-Time SOC",
            save_path=ARTIFACTS_DIR / "soc_realtime_importance.png",
        )

    model_path = ARTIFACTS_DIR / "soc_realtime_model.json"
    model.save_model(str(model_path))
    if verbose:
        print(f"\nModel saved to: {model_path}")
        print(f"Best iteration: {model.best_iteration}")

    meta = {
        "model": "soc_realtime_v1",
        "trained_at": datetime.now().isoformat(),
        "features": REALTIME_FEATURES,
        "target": REALTIME_TARGET,
        "params": params,
        "best_iteration": model.best_iteration,
        "num_trees": model.num_boosted_rounds(),
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
        "metrics": results,
    }
    meta_path = ARTIFACTS_DIR / "soc_realtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return model, results


if __name__ == "__main__":
    train()
