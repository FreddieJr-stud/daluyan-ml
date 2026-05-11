"""Data augmentation for trip-level SOC model (Model 1A).

Implements:
  1. Gaussian noise injection — perturb features/target with small noise
  2. C-Mixup — create synthetic samples by mixing similar trips

Only applied to the TRAINING split.  Not used for real-time or anomaly models
(those already have >>10K rows and augmentation would weaken the signal).

Usage:
    from data.augment import augment_trip_data
    train_aug = augment_trip_data(train_df, feature_cols, target_col)
"""

from __future__ import annotations

import numpy as np
import polars as pl

from config import AUGMENTATION


def gaussian_noise(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_copies: int | None = None,
    noise_std: float | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Create augmented copies with Gaussian noise added to features and target.

    Noise magnitude is relative to each feature's standard deviation.
    """
    cfg = AUGMENTATION["gaussian_noise"]
    n_copies = n_copies or cfg["n_copies"]
    noise_std = noise_std or cfg["noise_std"]

    rng = np.random.default_rng(seed)
    augmented_parts: list[pl.DataFrame] = []

    # Compute feature std from original data
    feature_stds = {}
    for col in feature_cols:
        std = df[col].std()
        feature_stds[col] = std if std and std > 0 else 1e-6

    target_std = df[target_col].std() or 1e-6

    for copy_idx in range(n_copies):
        # Convert to numpy for efficient noise addition
        aug_data = {}
        n = len(df)

        for col in feature_cols:
            vals = df[col].to_numpy().astype(np.float64)
            noise = rng.normal(0, noise_std * feature_stds[col], size=n)
            aug_data[col] = vals + noise

        # Perturb target proportionally
        target_vals = df[target_col].to_numpy().astype(np.float64)
        target_noise = rng.normal(0, noise_std * target_std, size=n)
        aug_data[target_col] = np.clip(target_vals + target_noise, 0, None)

        # Copy metadata columns
        for col in df.columns:
            if col not in feature_cols and col != target_col:
                aug_data[col] = df[col].to_list()

        aug_df = pl.DataFrame(aug_data)
        augmented_parts.append(aug_df)

    return pl.concat(augmented_parts, how="diagonal_relaxed")


def cmixup(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float | None = None,
    sigma: float | None = None,
    n_synthetic: float | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Create synthetic samples via C-Mixup (context-aware mixup).

    Pairs samples with similar targets (SOC delta), then interpolates
    between them with a Beta-distributed mixing coefficient.

    Args:
        alpha: Beta distribution parameter (small = close to original)
        sigma: Similarity bandwidth — pairs with target diff within sigma are preferred
        n_synthetic: multiplier of original dataset size for synthetic samples
    """
    cfg = AUGMENTATION["cmixup"]
    alpha = alpha or cfg["alpha"]
    sigma = sigma or cfg["sigma"]
    n_synthetic = n_synthetic or cfg["n_synthetic"]

    rng = np.random.default_rng(seed)
    n = len(df)
    n_pairs = int(n * n_synthetic)

    # Get target values for similarity-based pairing
    targets = df[target_col].to_numpy().astype(np.float64)

    # Build feature matrix
    X = np.column_stack([df[col].to_numpy().astype(np.float64) for col in feature_cols])

    # Similarity-weighted pairing: prefer pairs with close SOC deltas
    synthetic_features = np.zeros((n_pairs, len(feature_cols)))
    synthetic_targets = np.zeros(n_pairs)

    for i in range(n_pairs):
        # Pick a random anchor
        idx_a = rng.integers(0, n)

        # Compute similarity weights based on target distance
        diffs = np.abs(targets - targets[idx_a])
        weights = np.exp(-diffs**2 / (2 * sigma**2))
        weights[idx_a] = 0  # don't pair with self
        weights /= weights.sum() + 1e-10

        # Pick partner weighted by similarity
        idx_b = rng.choice(n, p=weights)

        # Mix with Beta-distributed lambda
        lam = rng.beta(alpha, alpha)

        synthetic_features[i] = lam * X[idx_a] + (1 - lam) * X[idx_b]
        synthetic_targets[i] = lam * targets[idx_a] + (1 - lam) * targets[idx_b]

    # Build DataFrame
    syn_data = {col: synthetic_features[:, j] for j, col in enumerate(feature_cols)}
    syn_data[target_col] = np.clip(synthetic_targets, 0, None)

    return pl.DataFrame(syn_data)


def augment_trip_data(
    train_df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
    verbose: bool = True,
) -> pl.DataFrame:
    """Apply all augmentation strategies to trip training data.

    Returns concatenation of original + augmented data.
    """
    original_n = len(train_df)
    parts = [train_df]

    # 1. Gaussian noise injection
    noise_df = gaussian_noise(train_df, feature_cols, target_col)
    parts.append(noise_df)
    if verbose:
        print(f"  Gaussian noise: +{len(noise_df)} rows "
              f"({AUGMENTATION['gaussian_noise']['n_copies']} copies)")

    # 2. C-Mixup
    cmixup_df = cmixup(train_df, feature_cols, target_col)
    parts.append(cmixup_df)
    if verbose:
        print(f"  C-Mixup: +{len(cmixup_df)} synthetic rows")

    combined = pl.concat(parts, how="diagonal_relaxed")

    # Fill any nulls in metadata columns from augmented data
    combined = combined.fill_null(strategy="forward")

    if verbose:
        total = len(combined)
        print(f"  Total: {original_n} -> {total} rows ({total/original_n:.1f}x)")

    return combined
