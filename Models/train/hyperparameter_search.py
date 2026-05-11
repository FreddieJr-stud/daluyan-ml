"""Random hyperparameter search for XGBoost models.

Usage:
    python -m train.hyperparameter_search [--n_iter 50] [--model trip|realtime]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    TRIP_FEATURES,
    TRIP_TARGET,
    XGBOOST_TRIP_PARAMS,
)
from train.train_soc_trip import df_to_dmatrix, load_trip_data, temporal_split

SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1, 0.15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10, 15],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0],
}


def sample_params() -> dict:
    """Sample one random config from the search space."""
    base = {
        "objective": "reg:squarederror",
        "eval_metric": "mae",
        "tree_method": "hist",
        "seed": 42,
    }
    for key, values in SEARCH_SPACE.items():
        base[key] = random.choice(values)
    return base


def run_search(n_iter: int = 50, verbose: bool = True) -> dict:
    """Run random hyperparameter search, return best config."""
    random.seed(42)

    df = load_trip_data()
    train_df, val_df, _ = temporal_split(df)
    dtrain, _ = df_to_dmatrix(train_df)
    dval, y_val = df_to_dmatrix(val_df)

    best_mae = float("inf")
    best_params = None
    best_iter = 0
    results = []

    for i in range(n_iter):
        params = sample_params()

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=0,
        )

        y_pred = model.predict(dval)
        mae = np.mean(np.abs(y_val - y_pred))

        trial = {
            "trial": i + 1,
            "mae": float(mae),
            "best_iteration": model.best_iteration,
            **{k: v for k, v in params.items() if k in SEARCH_SPACE},
        }
        results.append(trial)

        if mae < best_mae:
            best_mae = mae
            best_params = params.copy()
            best_iter = model.best_iteration

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_iter}] Best MAE so far: {best_mae:.4f}%")

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  Best MAE: {best_mae:.4f}%  (iteration {best_iter})")
        print(f"  Best params:")
        for k in SEARCH_SPACE:
            print(f"    {k}: {best_params[k]}")
        print(f"{'=' * 50}")

    # Save search results
    search_log = {
        "best_mae": float(best_mae),
        "best_params": {k: v for k, v in best_params.items() if k in SEARCH_SPACE},
        "best_iteration": best_iter,
        "n_trials": n_iter,
        "all_trials": results,
    }
    log_path = ARTIFACTS_DIR / "hyperparam_search_results.json"
    with open(log_path, "w") as f:
        json.dump(search_log, f, indent=2)
    if verbose:
        print(f"  Search log saved to: {log_path}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iter", type=int, default=50)
    args = parser.parse_args()

    best = run_search(n_iter=args.n_iter)

    # Retrain with best params on full pipeline
    print("\nRetraining with best params...")
    from train.train_soc_trip import train
    train(params=best)
