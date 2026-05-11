"""Evaluation utilities for SOC prediction models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics for SOC prediction."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — avoid div-by-zero for near-zero targets
    mask = np.abs(y_true) > 0.5
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  MAE:  {metrics['MAE']:.3f}% SOC")
    print(f"  RMSE: {metrics['RMSE']:.3f}% SOC")
    print(f"  R2:   {metrics['R2']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.1f}%")


def print_direction_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    directions: np.ndarray,
) -> None:
    """Print MAE broken down by direction."""
    print("\n  Per-direction MAE:")
    for d in sorted(set(directions)):
        mask = directions == d
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        label = "downstream" if d == 0 else "upstream"
        print(f"    {label}: {mae:.3f}% SOC  (n={mask.sum()})")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Predicted vs actual scatter + residual histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual SOC Delta (%)")
    ax.set_ylabel("Predicted SOC Delta (%)")
    ax.set_title(f"{title} - Predicted vs Actual")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Residual histogram
    ax = axes[1]
    residuals = y_pred - y_true
    ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual (Predicted - Actual) %")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals  |  Mean={residuals.mean():.2f}, Std={residuals.std():.2f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list[str],
    title: str,
    save_path: Path | None = None,
    top_n: int = 20,
) -> None:
    """Plot XGBoost feature importance (gain)."""
    importance = model.get_score(importance_type="gain")

    # Sort and take top N
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y_pos = range(len(names))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title(f"{title} - Feature Importance (Gain)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Importance plot saved to: {save_path}")
    plt.close(fig)
