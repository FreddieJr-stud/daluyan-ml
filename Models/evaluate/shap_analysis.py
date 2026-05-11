"""SHAP analysis for XGBoost SOC prediction models.

Generates thesis-quality interpretability plots:
  1. Summary (beeswarm) plot — top 20 features with direction of effect
  2. Dependence plots — top 5 features showing interaction effects
  3. Waterfall plots — 3 individual predictions (highest, lowest, median)

Applies to both Model 1A (trip) and Model 1B (realtime).

Usage:
    python -m evaluate.shap_analysis
    # Or via training pipeline:
    python -m train.train_v2 --model all --shap

Requires: pip install shap
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    FEATURE_STORE_DIR,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRIP_FEATURES,
    TRIP_TARGET,
    TRAIN_END,
    VAL_END,
)
from datetime import datetime


PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _temporal_split(df: pl.DataFrame):
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def _load_primary_model(name: str) -> xgb.Booster:
    path = ARTIFACTS_DIR / f"{name}.json"
    m = xgb.Booster()
    m.load_model(str(path))
    return m


def analyze_model(
    model_name: str,
    display_name: str,
    features: list[str],
    target: str,
    feature_store_file: str,
    max_display: int = 20,
) -> None:
    """Run full SHAP analysis for a single model."""
    print(f"\n{'=' * 60}")
    print(f"  SHAP Analysis: {display_name}")
    print(f"{'=' * 60}")

    # Load model and data
    model = _load_primary_model(model_name)
    df = pl.read_parquet(FEATURE_STORE_DIR / feature_store_file)
    _, _, test_df = _temporal_split(df)

    X_test = test_df.select(features).to_numpy().astype(np.float32)
    y_test = test_df[target].to_numpy().astype(np.float32)

    print(f"  Test samples: {len(X_test)}")

    # For realtime model, subsample to keep SHAP tractable
    if len(X_test) > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test), 2000, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]
        print(f"  Subsampled to {len(X_test)} for SHAP")

    # TreeExplainer — exact, fast for XGBoost
    dm = xgb.DMatrix(X_test, feature_names=features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(dm)
    print(f"  SHAP values computed: shape={shap_values.shape}")

    prefix = model_name.replace("_model", "")

    # ── 1. Summary (beeswarm) plot ──
    print("  Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=features,
        max_display=max_display,
        show=False,
    )
    plt.title(f"{display_name} — SHAP Feature Impact", fontsize=13)
    plt.tight_layout()
    path = PLOTS_DIR / f"{prefix}_shap_beeswarm.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved: {path}")

    # ── 2. Dependence plots — top 5 features ──
    print("  Generating dependence plots (top 5)...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(-mean_abs)[:5]

    for rank, feat_idx in enumerate(top5_idx):
        feat_name = features[feat_idx]
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(
            feat_idx, shap_values, X_test,
            feature_names=features,
            show=False,
            ax=ax,
        )
        ax.set_title(f"{display_name} — {feat_name} Dependence", fontsize=12)
        plt.tight_layout()
        path = PLOTS_DIR / f"{prefix}_shap_dep_{rank+1}_{feat_name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"    Saved: {path}")

    # ── 3. Waterfall plots — highest, median, lowest consumption ──
    print("  Generating waterfall plots (3 cases)...")
    preds = model.predict(dm)
    sorted_idx = np.argsort(preds)

    cases = {
        "lowest": sorted_idx[0],
        "median": sorted_idx[len(sorted_idx) // 2],
        "highest": sorted_idx[-1],
    }

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(shap_values), explainer.expected_value),
        data=X_test,
        feature_names=features,
    )

    for case_name, idx in cases.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(explanation[idx], max_display=15, show=False)
        plt.title(f"{display_name} — {case_name.capitalize()} Consumption "
                  f"(actual={y_test[idx]:.1f}%, pred={preds[idx]:.1f}%)", fontsize=11)
        plt.tight_layout()
        path = PLOTS_DIR / f"{prefix}_shap_waterfall_{case_name}.png"
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"    Saved: {path}")

    # Print top features summary
    print(f"\n  Top 10 features by mean |SHAP|:")
    for i in np.argsort(-mean_abs)[:10]:
        print(f"    {features[i]:40s} {mean_abs[i]:.4f}")


def run_shap_analysis() -> None:
    """Run SHAP analysis for all SOC models."""
    # Model 1A: Trip-level
    trip_store = FEATURE_STORE_DIR / "trip_features.parquet"
    if trip_store.exists() and (ARTIFACTS_DIR / "soc_trip_model.json").exists():
        analyze_model(
            "soc_trip_model", "Model 1A: Trip-Level SOC",
            TRIP_FEATURES, TRIP_TARGET, "trip_features.parquet",
        )
    else:
        print("  SKIP Model 1A SHAP: model or features not found")

    # Model 1B: Real-time
    rt_store = FEATURE_STORE_DIR / "realtime_features.parquet"
    if rt_store.exists() and (ARTIFACTS_DIR / "soc_realtime_model.json").exists():
        analyze_model(
            "soc_realtime_model", "Model 1B: Real-Time SOC",
            REALTIME_FEATURES, REALTIME_TARGET, "realtime_features.parquet",
        )
    else:
        print("  SKIP Model 1B SHAP: model or features not found")

    print("\n  SHAP analysis complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    run_shap_analysis()
