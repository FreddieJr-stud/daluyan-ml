"""Evaluation utilities for motor anomaly detection model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_distribution(
    scores: np.ndarray,
    threshold: float,
    title: str = "Anomaly Score Distribution",
    save_path: Path | None = None,
) -> None:
    """Plot anomaly score distribution with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(scores, bins=100, edgecolor="black", alpha=0.7, density=True)
    ax.axvline(threshold, color="r", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.2f}")

    n_anomalous = (scores > threshold).sum()
    pct = n_anomalous / len(scores) * 100
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\n{n_anomalous}/{len(scores)} flagged ({pct:.1f}%)")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Distribution plot saved to: {save_path}")
    plt.close(fig)


def synthetic_injection_test(
    compute_score_fn,
    normal_data: np.ndarray,
    feature_names: list[str],
    n_injections: int = 500,
    corruption_factor: float = 1.5,
    threshold: float = None,
) -> dict:
    """Test anomaly detection by injecting synthetic anomalies.

    Corrupts port_motor_power by +-corruption_factor for random samples
    and checks if the model detects them.

    Args:
        compute_score_fn: callable(features_array) -> scores_array
        normal_data: (n, d) array of normal feature rows
        feature_names: list of feature names
        n_injections: number of corrupted samples to inject
        corruption_factor: multiply port_motor_power by this
        threshold: anomaly threshold; if None, uses 95th percentile of normal scores

    Returns:
        dict with detection_rate, false_positive_rate, threshold
    """
    # Score normal data
    normal_scores = compute_score_fn(normal_data)

    if threshold is None:
        threshold = np.percentile(normal_scores, 95)

    # Inject anomalies: pick random rows and corrupt port_motor_power
    rng = np.random.RandomState(42)
    indices = rng.choice(len(normal_data), size=n_injections, replace=True)
    corrupted = normal_data[indices].copy()

    port_idx = feature_names.index("port_motor_power")
    # Randomly increase or decrease by corruption_factor
    signs = rng.choice([-1, 1], size=n_injections)
    corrupted[:, port_idx] *= corruption_factor * np.where(signs == 1, 1, -0.5)

    corrupted_scores = compute_score_fn(corrupted)

    detection_rate = (corrupted_scores > threshold).sum() / n_injections * 100
    false_positive_rate = (normal_scores > threshold).sum() / len(normal_scores) * 100

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "threshold": threshold,
        "n_injections": n_injections,
        "corruption_factor": corruption_factor,
    }


def threshold_sensitivity(
    scores_normal: np.ndarray,
    scores_corrupted: np.ndarray,
    percentiles: list[float] = None,
) -> None:
    """Print detection rate and FPR at various threshold percentiles."""
    if percentiles is None:
        percentiles = [90, 95, 97, 99]

    print("\n  Threshold sensitivity:")
    print(f"  {'Percentile':>12}  {'Threshold':>10}  {'FPR':>8}  {'Detection':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*10}")

    for p in percentiles:
        t = np.percentile(scores_normal, p)
        fpr = (scores_normal > t).sum() / len(scores_normal) * 100
        det = (scores_corrupted > t).sum() / len(scores_corrupted) * 100
        print(f"  {p:>11.0f}%  {t:>10.3f}  {fpr:>7.1f}%  {det:>9.1f}%")
