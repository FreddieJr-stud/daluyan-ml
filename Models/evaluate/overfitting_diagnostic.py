"""Comprehensive overfitting diagnostic for Models 1A and 1B.

Four analyses:
  1. Train/Val/Test gap report
  2. Temporal cross-validation (expanding window, 5 folds)
  3. Per-date residual analysis
  4. Learning curve (20%, 40%, 60%, 80%, 100% of training data)

Usage:
    python -m evaluate.overfitting_diagnostic
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    ENSEMBLE_SEEDS,
    FEATURE_STORE_DIR,
    N_ENSEMBLE_SEEDS,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_MONOTONIC_CONSTRAINTS,
    TRIP_TARGET,
    VAL_END,
    get_monotonic_constraint_tuple,
)
from data.augment import augment_trip_data
from evaluate.evaluate_soc import compute_metrics

# ── Plotting style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

COLORS = {
    "train": "#2196F3",
    "val": "#FF9800",
    "test": "#E53935",
    "accent": "#4CAF50",
}

OUTPUT_DIR = ARTIFACTS_DIR / "overfitting_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Shared utilities ────────────────────────────────────────────────────

def load_production_metadata(model: str) -> dict:
    names = {"1a": "soc_trip_metadata.json", "1b": "soc_realtime_metadata.json"}
    path = ARTIFACTS_DIR / names[model]
    with open(path) as f:
        return json.load(f)


def df_to_dmatrix(
    df: pl.DataFrame, features: list[str], target: str
) -> tuple[xgb.DMatrix, np.ndarray]:
    X = df.select(features).to_numpy().astype(np.float32)
    y = df[target].to_numpy().astype(np.float32)
    return xgb.DMatrix(X, label=y, feature_names=features), y


def train_ensemble(
    train_dm: xgb.DMatrix,
    val_dm: xgb.DMatrix,
    params: dict,
    n_estimators: int,
    seeds: list[int] | None = None,
) -> list[xgb.Booster]:
    if seeds is None:
        seeds = ENSEMBLE_SEEDS
    models = []
    for seed in seeds:
        p = {**params, "seed": seed}
        model = xgb.train(
            p, train_dm,
            num_boost_round=n_estimators,
            evals=[(val_dm, "val")],
            early_stopping_rounds=50,
            verbose_eval=0,
        )
        models.append(model)
    return models


def ensemble_predict(models: list[xgb.Booster], dm: xgb.DMatrix) -> np.ndarray:
    preds = np.column_stack([m.predict(dm) for m in models])
    return preds.mean(axis=1)


def get_date_from_segment_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("segment_id").str.slice(0, 10).alias("date"))


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


# ── Analysis 1: Train/Val/Test Gap Report ───────────────────────────────

def analysis_gap_report(
    trip_df: pl.DataFrame, realtime_df: pl.DataFrame
) -> dict:
    """Compute and visualize train/val/test gaps for both models."""
    print("\n" + "=" * 60)
    print("  ANALYSIS 1: Train / Val / Test Gap Report")
    print("=" * 60)

    results = {}

    for model_name, df, features, target, augment in [
        ("1A", trip_df, TRIP_FEATURES, TRIP_TARGET, True),
        ("1B", realtime_df, REALTIME_FEATURES, REALTIME_TARGET, False),
    ]:
        train, val, test = temporal_split(df)
        meta = load_production_metadata(model_name.lower())
        saved_params = meta["best_params"]

        mono_tuple = None
        params = {
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

        if model_name == "1A":
            mono_tuple = get_monotonic_constraint_tuple(
                TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS
            )
            params["monotone_constraints"] = mono_tuple
            train_data = augment_trip_data(train, features, target, verbose=False)
        else:
            train_data = train

        dtrain, _ = df_to_dmatrix(train_data, features, target)
        dtrain_orig, y_train = df_to_dmatrix(train, features, target)
        dval, y_val = df_to_dmatrix(val, features, target)
        dtest, y_test = df_to_dmatrix(test, features, target)

        models = train_ensemble(
            dtrain, dval, params, saved_params["n_estimators"]
        )

        train_metrics = compute_metrics(y_train, ensemble_predict(models, dtrain_orig))
        val_metrics = compute_metrics(y_val, ensemble_predict(models, dval))
        test_metrics = compute_metrics(y_test, ensemble_predict(models, dtest))

        train_val_gap = val_metrics["MAE"] - train_metrics["MAE"]
        val_test_gap = test_metrics["MAE"] - val_metrics["MAE"]
        overfit_ratio = test_metrics["MAE"] / train_metrics["MAE"]

        entry = {
            "train_mae": round(train_metrics["MAE"], 4),
            "val_mae": round(val_metrics["MAE"], 4),
            "test_mae": round(test_metrics["MAE"], 4),
            "train_r2": round(train_metrics["R2"], 4),
            "val_r2": round(val_metrics["R2"], 4),
            "test_r2": round(test_metrics["R2"], 4),
            "train_val_gap": round(train_val_gap, 4),
            "val_test_gap": round(val_test_gap, 4),
            "overfit_ratio": round(overfit_ratio, 2),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        }
        results[model_name] = entry

        print(f"\n  Model {model_name}:")
        print(f"    Train MAE:  {train_metrics['MAE']:.4f}%  (R2={train_metrics['R2']:.4f})")
        print(f"    Val MAE:    {val_metrics['MAE']:.4f}%  (R2={val_metrics['R2']:.4f})")
        print(f"    Test MAE:   {test_metrics['MAE']:.4f}%  (R2={test_metrics['R2']:.4f})")
        print(f"    Train->Val gap: +{train_val_gap:.4f}%")
        print(f"    Val->Test gap:  {val_test_gap:+.4f}%")
        print(f"    Overfit ratio (test/train): {overfit_ratio:.2f}x")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_name in zip(axes, ["1A", "1B"]):
        r = results[model_name]
        splits = ["Train", "Val", "Test"]
        maes = [r["train_mae"], r["val_mae"], r["test_mae"]]
        colors_list = [COLORS["train"], COLORS["val"], COLORS["test"]]
        bars = ax.bar(splits, maes, color=colors_list, edgecolor="white", width=0.6)

        for bar, mae in zip(bars, maes):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mae:.3f}%", ha="center", va="bottom", fontweight="bold", fontsize=11,
            )

        ax.set_ylabel("MAE (% SOC)")
        ax.set_title(
            f"Model {model_name} — Train/Val/Test Gap\n"
            f"Overfit ratio: {r['overfit_ratio']:.1f}x  |  "
            f"Val-Test gap: {r['val_test_gap']:+.3f}%",
            fontsize=11,
        )
        ax.set_ylim(0, max(maes) * 1.4)

    plt.tight_layout()
    path = OUTPUT_DIR / "1_gap_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    return results


# ── Analysis 2: Temporal Cross-Validation ───────────────────────────────

def analysis_temporal_cv(
    trip_df: pl.DataFrame, realtime_df: pl.DataFrame, n_folds: int = 5
) -> dict:
    """Expanding-window temporal CV."""
    print("\n" + "=" * 60)
    print("  ANALYSIS 2: Temporal Cross-Validation (Expanding Window)")
    print("=" * 60)

    results = {}

    for model_name, df, features, target, augment in [
        ("1A", trip_df, TRIP_FEATURES, TRIP_TARGET, True),
        ("1B", realtime_df, REALTIME_FEATURES, REALTIME_TARGET, False),
    ]:
        print(f"\n  Model {model_name}:")

        # Get unique dates sorted
        df_dates = get_date_from_segment_id(df)
        all_dates = sorted(df_dates["date"].unique().to_list())
        n_dates = len(all_dates)

        # Minimum train size: 40% of dates, then expand
        min_train_pct = 0.4
        min_train_dates = int(n_dates * min_train_pct)
        remaining = n_dates - min_train_dates
        # Val = 10% of total, Test = 10% of total (same proportions as production)
        val_size = max(3, int(n_dates * 0.10))
        test_size = max(3, int(n_dates * 0.10))
        fold_step = max(1, (remaining - val_size - test_size) // (n_folds - 1))

        meta = load_production_metadata(model_name.lower())
        saved_params = meta["best_params"]
        params = {
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
        if model_name == "1A":
            mono_tuple = get_monotonic_constraint_tuple(
                TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS
            )
            params["monotone_constraints"] = mono_tuple

        fold_results = []
        for fold_i in range(n_folds):
            train_end_idx = min_train_dates + fold_i * fold_step
            val_end_idx = train_end_idx + val_size
            test_end_idx = min(val_end_idx + test_size, n_dates)

            if test_end_idx > n_dates or val_end_idx >= n_dates:
                break

            train_dates = set(all_dates[:train_end_idx])
            val_dates = set(all_dates[train_end_idx:val_end_idx])
            test_dates = set(all_dates[val_end_idx:test_end_idx])

            train_fold = df_dates.filter(pl.col("date").is_in(train_dates))
            val_fold = df_dates.filter(pl.col("date").is_in(val_dates))
            test_fold = df_dates.filter(pl.col("date").is_in(test_dates))

            if len(train_fold) < 10 or len(val_fold) < 3 or len(test_fold) < 3:
                continue

            if augment:
                train_data = augment_trip_data(
                    train_fold, features, target, verbose=False
                )
            else:
                train_data = train_fold

            dtrain, _ = df_to_dmatrix(train_data, features, target)
            dtrain_orig, y_train = df_to_dmatrix(train_fold, features, target)
            dval, y_val = df_to_dmatrix(val_fold, features, target)
            dtest, y_test = df_to_dmatrix(test_fold, features, target)

            models = train_ensemble(
                dtrain, dval, params, saved_params["n_estimators"]
            )

            train_m = compute_metrics(y_train, ensemble_predict(models, dtrain_orig))
            val_m = compute_metrics(y_val, ensemble_predict(models, dval))
            test_m = compute_metrics(y_test, ensemble_predict(models, dtest))

            fold_entry = {
                "fold": fold_i + 1,
                "train_dates": f"{min(train_dates)}..{max(train_dates)}",
                "val_dates": f"{min(val_dates)}..{max(val_dates)}",
                "test_dates": f"{min(test_dates)}..{max(test_dates)}",
                "n_train": len(train_fold),
                "n_val": len(val_fold),
                "n_test": len(test_fold),
                "train_mae": round(train_m["MAE"], 4),
                "val_mae": round(val_m["MAE"], 4),
                "test_mae": round(test_m["MAE"], 4),
                "train_r2": round(train_m["R2"], 4),
                "val_r2": round(val_m["R2"], 4),
                "test_r2": round(test_m["R2"], 4),
                "val_test_gap": round(test_m["MAE"] - val_m["MAE"], 4),
            }
            fold_results.append(fold_entry)

            print(
                f"    Fold {fold_i + 1}: "
                f"Train={len(train_fold):>4}  Val={len(val_fold):>3}  Test={len(test_fold):>3}  |  "
                f"MAE: {train_m['MAE']:.3f} / {val_m['MAE']:.3f} / {test_m['MAE']:.3f}  |  "
                f"Gap: {test_m['MAE'] - val_m['MAE']:+.3f}"
            )

        # Summary stats
        test_maes = [f["test_mae"] for f in fold_results]
        val_test_gaps = [f["val_test_gap"] for f in fold_results]

        summary = {
            "folds": fold_results,
            "test_mae_mean": round(np.mean(test_maes), 4),
            "test_mae_std": round(np.std(test_maes), 4),
            "val_test_gap_mean": round(np.mean(val_test_gaps), 4),
            "val_test_gap_std": round(np.std(val_test_gaps), 4),
        }
        results[model_name] = summary

        print(f"\n    CV Test MAE: {summary['test_mae_mean']:.4f} +/- {summary['test_mae_std']:.4f}")
        print(f"    CV Val-Test Gap: {summary['val_test_gap_mean']:+.4f} +/- {summary['val_test_gap_std']:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model_name in zip(axes, ["1A", "1B"]):
        folds = results[model_name]["folds"]
        fold_nums = [f["fold"] for f in folds]
        train_maes = [f["train_mae"] for f in folds]
        val_maes = [f["val_mae"] for f in folds]
        test_maes = [f["test_mae"] for f in folds]

        ax.plot(fold_nums, train_maes, "o-", color=COLORS["train"], label="Train", linewidth=2, markersize=7)
        ax.plot(fold_nums, val_maes, "s-", color=COLORS["val"], label="Val", linewidth=2, markersize=7)
        ax.plot(fold_nums, test_maes, "^-", color=COLORS["test"], label="Test", linewidth=2, markersize=7)

        # Shade train-test gap
        ax.fill_between(
            fold_nums, train_maes, test_maes,
            alpha=0.1, color=COLORS["test"], label="Train-Test gap",
        )

        ax.set_xlabel("Fold")
        ax.set_ylabel("MAE (% SOC)")
        ax.set_title(
            f"Model {model_name} — Temporal CV ({len(folds)} folds)\n"
            f"Test MAE: {results[model_name]['test_mae_mean']:.3f} "
            f"+/- {results[model_name]['test_mae_std']:.3f}",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    path = OUTPUT_DIR / "2_temporal_cv.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    return results


# ── Analysis 3: Per-Date Residual Analysis ──────────────────────────────

def analysis_per_date_residuals(
    trip_df: pl.DataFrame, realtime_df: pl.DataFrame
) -> dict:
    """Per-date residual analysis to detect systematic bias."""
    print("\n" + "=" * 60)
    print("  ANALYSIS 3: Per-Date Residual Analysis")
    print("=" * 60)

    results = {}

    for model_name, df, features, target, augment in [
        ("1A", trip_df, TRIP_FEATURES, TRIP_TARGET, True),
        ("1B", realtime_df, REALTIME_FEATURES, REALTIME_TARGET, False),
    ]:
        print(f"\n  Model {model_name}:")

        df_dates = get_date_from_segment_id(df)
        train, val, test = temporal_split(df_dates)

        meta = load_production_metadata(model_name.lower())
        saved_params = meta["best_params"]
        params = {
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
        if model_name == "1A":
            mono_tuple = get_monotonic_constraint_tuple(
                TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS
            )
            params["monotone_constraints"] = mono_tuple
            train_data = augment_trip_data(train, features, target, verbose=False)
        else:
            train_data = train

        dtrain, _ = df_to_dmatrix(train_data, features, target)
        dval, y_val = df_to_dmatrix(val, features, target)

        models = train_ensemble(
            dtrain, dval, params, saved_params["n_estimators"]
        )

        # Evaluate ALL data (train + val + test) per date
        dall, y_all = df_to_dmatrix(df_dates, features, target)
        y_pred_all = ensemble_predict(models, dall)
        residuals_all = np.abs(y_pred_all - y_all)

        overall_mae = float(np.mean(residuals_all))
        flagging_threshold = overall_mae * 2.0

        # Split annotations
        train_dates_set = set(get_date_from_segment_id(train)["date"].unique().to_list())
        val_dates_set = set(get_date_from_segment_id(val)["date"].unique().to_list())
        test_dates_set = set(get_date_from_segment_id(test)["date"].unique().to_list())

        dates_col = df_dates["date"].to_list()
        unique_dates = sorted(set(dates_col))

        per_date = []
        flagged = []
        for d in unique_dates:
            mask = [i for i, dd in enumerate(dates_col) if dd == d]
            mae_d = float(np.mean(residuals_all[mask]))
            mean_residual = float(np.mean(y_pred_all[mask] - y_all[mask]))
            n_samples = len(mask)

            if d in train_dates_set:
                split = "train"
            elif d in val_dates_set:
                split = "val"
            else:
                split = "test"

            entry = {
                "date": d,
                "split": split,
                "n_samples": n_samples,
                "mae": round(mae_d, 4),
                "mean_residual": round(mean_residual, 4),
                "flagged": mae_d > flagging_threshold,
            }
            per_date.append(entry)
            if entry["flagged"]:
                flagged.append(entry)

        # Print flagged dates
        if flagged:
            print(f"\n    Flagged dates (MAE > {flagging_threshold:.3f}%, 2x overall):")
            for f in flagged:
                print(
                    f"      {f['date']} ({f['split']}): MAE={f['mae']:.3f}%, "
                    f"bias={f['mean_residual']:+.3f}%, n={f['n_samples']}"
                )
        else:
            print(f"    No dates flagged (threshold: {flagging_threshold:.3f}%)")

        # Check for train-date leakage: are train dates systematically lower?
        train_date_maes = [p["mae"] for p in per_date if p["split"] == "train"]
        test_date_maes = [p["mae"] for p in per_date if p["split"] == "test"]
        val_date_maes = [p["mae"] for p in per_date if p["split"] == "val"]

        print(f"\n    Per-split mean date MAE:")
        print(f"      Train dates: {np.mean(train_date_maes):.4f}% (n={len(train_date_maes)} dates)")
        if val_date_maes:
            print(f"      Val dates:   {np.mean(val_date_maes):.4f}% (n={len(val_date_maes)} dates)")
        print(f"      Test dates:  {np.mean(test_date_maes):.4f}% (n={len(test_date_maes)} dates)")
        print(f"      Ratio (test/train date MAE): {np.mean(test_date_maes) / np.mean(train_date_maes):.2f}x")

        results[model_name] = {
            "overall_mae": round(overall_mae, 4),
            "flagging_threshold": round(flagging_threshold, 4),
            "n_flagged": len(flagged),
            "flagged_dates": flagged,
            "train_date_mae_mean": round(float(np.mean(train_date_maes)), 4),
            "val_date_mae_mean": round(float(np.mean(val_date_maes)), 4) if val_date_maes else None,
            "test_date_mae_mean": round(float(np.mean(test_date_maes)), 4),
            "per_date": per_date,
        }

    # Plot — per-date MAE bar chart for both models
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    for ax, model_name in zip(axes, ["1A", "1B"]):
        per_date = results[model_name]["per_date"]
        dates = [p["date"] for p in per_date]
        maes = [p["mae"] for p in per_date]
        splits = [p["split"] for p in per_date]
        threshold = results[model_name]["flagging_threshold"]

        color_map = {"train": COLORS["train"], "val": COLORS["val"], "test": COLORS["test"]}
        bar_colors = [color_map[s] for s in splits]

        bars = ax.bar(range(len(dates)), maes, color=bar_colors, edgecolor="white", width=0.8)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                    label=f"2x flag ({threshold:.2f}%)")

        # Mark flagged
        for i, (mae, entry) in enumerate(zip(maes, per_date)):
            if entry["flagged"]:
                ax.annotate(
                    entry["date"][5:],  # Show MM-DD
                    xy=(i, mae), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="red", fontweight="bold",
                )

        # X-axis: show every Nth date
        n_show = max(1, len(dates) // 15)
        ax.set_xticks(range(0, len(dates), n_show))
        ax.set_xticklabels([dates[i][5:] for i in range(0, len(dates), n_show)],
                           rotation=45, ha="right", fontsize=8)

        # Legend for splits
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["train"], label=f"Train ({sum(1 for s in splits if s == 'train')} dates)"),
            Patch(facecolor=COLORS["val"], label=f"Val ({sum(1 for s in splits if s == 'val')} dates)"),
            Patch(facecolor=COLORS["test"], label=f"Test ({sum(1 for s in splits if s == 'test')} dates)"),
        ]
        legend_elements.append(plt.Line2D([0], [0], color="red", linestyle="--", label=f"2x flag"))
        ax.legend(handles=legend_elements, fontsize=9, loc="upper left")

        ax.set_ylabel("MAE (% SOC)")
        ax.set_title(
            f"Model {model_name} — Per-Date Residuals  |  "
            f"{results[model_name]['n_flagged']} flagged / {len(dates)} dates",
            fontsize=11,
        )

    plt.tight_layout()
    path = OUTPUT_DIR / "3_per_date_residuals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    return results


# ── Analysis 4: Learning Curve ──────────────────────────────────────────

def analysis_learning_curve(
    trip_df: pl.DataFrame, realtime_df: pl.DataFrame
) -> dict:
    """Train on increasing fractions of data to check data saturation."""
    print("\n" + "=" * 60)
    print("  ANALYSIS 4: Learning Curve")
    print("=" * 60)

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    for model_name, df, features, target, augment in [
        ("1A", trip_df, TRIP_FEATURES, TRIP_TARGET, True),
        ("1B", realtime_df, REALTIME_FEATURES, REALTIME_TARGET, False),
    ]:
        print(f"\n  Model {model_name}:")

        df_dates = get_date_from_segment_id(df)
        train_full, val, test = temporal_split(df_dates)

        # Get sorted unique train dates for consistent subsetting
        train_dates = sorted(
            get_date_from_segment_id(train_full)["date"].unique().to_list()
        )

        meta = load_production_metadata(model_name.lower())
        saved_params = meta["best_params"]
        params = {
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
        if model_name == "1A":
            mono_tuple = get_monotonic_constraint_tuple(
                TRIP_FEATURES, TRIP_MONOTONIC_CONSTRAINTS
            )
            params["monotone_constraints"] = mono_tuple

        dval, y_val = df_to_dmatrix(val, features, target)
        dtest, y_test = df_to_dmatrix(test, features, target)

        curve = []
        for frac in fractions:
            n_dates = max(3, int(len(train_dates) * frac))
            subset_dates = set(train_dates[:n_dates])
            train_subset = df_dates.filter(pl.col("date").is_in(subset_dates))

            if augment:
                train_data = augment_trip_data(
                    train_subset, features, target, verbose=False
                )
            else:
                train_data = train_subset

            dtrain, _ = df_to_dmatrix(train_data, features, target)
            dtrain_orig, y_train = df_to_dmatrix(train_subset, features, target)

            models = train_ensemble(
                dtrain, dval, params, saved_params["n_estimators"]
            )

            train_m = compute_metrics(y_train, ensemble_predict(models, dtrain_orig))
            val_m = compute_metrics(y_val, ensemble_predict(models, dval))
            test_m = compute_metrics(y_test, ensemble_predict(models, dtest))

            entry = {
                "fraction": frac,
                "n_train_dates": n_dates,
                "n_train_samples": len(train_subset),
                "train_mae": round(train_m["MAE"], 4),
                "val_mae": round(val_m["MAE"], 4),
                "test_mae": round(test_m["MAE"], 4),
            }
            curve.append(entry)

            print(
                f"    {frac:>4.0%} ({n_dates:>3} dates, {len(train_subset):>5} samples): "
                f"Train={train_m['MAE']:.3f}%  Val={val_m['MAE']:.3f}%  Test={test_m['MAE']:.3f}%"
            )

        # Check if still improving
        test_maes = [c["test_mae"] for c in curve]
        last_improvement = (test_maes[-2] - test_maes[-1]) / test_maes[-2] * 100 if len(test_maes) >= 2 else 0
        still_improving = last_improvement > 1.0  # >1% improvement from 80%→100%

        results[model_name] = {
            "curve": curve,
            "last_improvement_pct": round(last_improvement, 2),
            "still_improving": still_improving,
        }

        print(
            f"\n    80%->100% improvement: {last_improvement:+.1f}%"
            f"  {'(still improving — more data would help)' if still_improving else '(plateauing — near data saturation)'}"
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model_name in zip(axes, ["1A", "1B"]):
        curve = results[model_name]["curve"]
        fracs = [c["n_train_samples"] for c in curve]
        train_maes = [c["train_mae"] for c in curve]
        val_maes = [c["val_mae"] for c in curve]
        test_maes = [c["test_mae"] for c in curve]

        ax.plot(fracs, train_maes, "o-", color=COLORS["train"], label="Train", linewidth=2, markersize=8)
        ax.plot(fracs, val_maes, "s-", color=COLORS["val"], label="Val", linewidth=2, markersize=8)
        ax.plot(fracs, test_maes, "^-", color=COLORS["test"], label="Test", linewidth=2, markersize=8)

        # Shade the gap
        ax.fill_between(
            fracs, train_maes, test_maes,
            alpha=0.1, color=COLORS["test"],
        )

        # Annotate the last point
        ax.annotate(
            f"{test_maes[-1]:.3f}%",
            xy=(fracs[-1], test_maes[-1]),
            xytext=(10, 10), textcoords="offset points",
            fontweight="bold", color=COLORS["test"],
            arrowprops=dict(arrowstyle="->", color=COLORS["test"]),
        )

        ax.set_xlabel("Training Samples")
        ax.set_ylabel("MAE (% SOC)")
        status = "improving" if results[model_name]["still_improving"] else "plateauing"
        ax.set_title(
            f"Model {model_name} — Learning Curve ({status})\n"
            f"80%->100%: {results[model_name]['last_improvement_pct']:+.1f}% change",
            fontsize=11,
        )
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = OUTPUT_DIR / "4_learning_curve.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {path}")

    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 60)
    print("  OVERFITTING DIAGNOSTIC — Models 1A & 1B")
    print("#" * 60)

    t0 = time.time()

    # Load data
    print("\n  Loading feature stores...")
    trip_df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    realtime_df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    print(f"    Trip: {trip_df.shape[0]} segments")
    print(f"    Realtime: {realtime_df.shape[0]} rows ({realtime_df['segment_id'].n_unique()} segments)")

    # Run all analyses
    gap_results = analysis_gap_report(trip_df, realtime_df)
    cv_results = analysis_temporal_cv(trip_df, realtime_df, n_folds=5)
    residual_results = analysis_per_date_residuals(trip_df, realtime_df)
    learning_results = analysis_learning_curve(trip_df, realtime_df)

    elapsed = time.time() - t0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for m in ["1A", "1B"]:
        g = gap_results[m]
        cv = cv_results[m]
        r = residual_results[m]
        lc = learning_results[m]

        print(f"\n  Model {m}:")
        print(f"    Overfit ratio (test/train MAE):  {g['overfit_ratio']:.1f}x")
        print(f"    Val-Test gap:                    {g['val_test_gap']:+.4f}%")
        print(f"    CV Test MAE (mean +/- std):      {cv['test_mae_mean']:.4f} +/- {cv['test_mae_std']:.4f}")
        print(f"    CV Val-Test gap (mean +/- std):   {cv['val_test_gap_mean']:+.4f} +/- {cv['val_test_gap_std']:.4f}")
        print(f"    Per-date: {r['n_flagged']} flagged / {len(r['per_date'])} dates")
        print(f"    Date MAE ratio (test/train):     {r['test_date_mae_mean'] / r['train_date_mae_mean']:.2f}x")
        print(f"    Learning curve:                  {'still improving' if lc['still_improving'] else 'plateauing'} ({lc['last_improvement_pct']:+.1f}%)")

    # Overfitting verdict
    print("\n  VERDICT:")
    for m in ["1A", "1B"]:
        g = gap_results[m]
        cv = cv_results[m]
        r = residual_results[m]

        issues = []
        if g["overfit_ratio"] > 3.0:
            issues.append(f"high overfit ratio ({g['overfit_ratio']:.1f}x)")
        if cv["val_test_gap_std"] > 0.5:
            issues.append(f"unstable CV gap (std={cv['val_test_gap_std']:.3f})")
        if r["n_flagged"] > len(r["per_date"]) * 0.1:
            issues.append(f"{r['n_flagged']} flagged dates ({r['n_flagged']/len(r['per_date'])*100:.0f}%)")

        if not issues:
            print(f"    Model {m}: HEALTHY — no significant overfitting detected")
        else:
            print(f"    Model {m}: ATTENTION — {'; '.join(issues)}")

    # Save results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed),
        "gap_report": gap_results,
        "temporal_cv": cv_results,
        "per_date_residuals": {
            m: {k: v for k, v in residual_results[m].items() if k != "per_date"}
            for m in ["1A", "1B"]
        },
        "learning_curve": learning_results,
    }

    results_path = OUTPUT_DIR / "overfitting_diagnostic_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    print(f"  Plots saved to: {OUTPUT_DIR}/")
    print(f"  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
