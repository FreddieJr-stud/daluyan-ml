"""Train alternative anomaly detection architectures for Model 2 comparison.

Three alternatives to the production XGBoost reconstruction-error approach:
  1. Random Forest Reconstruction — same 4-target paradigm, drop-in replacement
  2. MLP Autoencoder — encode→bottleneck→decode all 14 features
  3. Isolation Forest — direct anomaly scoring (no reconstruction)

All architectures use the same data, temporal splits, synthetic injection test,
and evaluation pipeline as the XGBoost baseline for a fair comparison.

Usage:
    python -m train.train_anomaly_alternatives --model rf        # RF only
    python -m train.train_anomaly_alternatives --model ae        # Autoencoder only
    python -m train.train_anomaly_alternatives --model if        # Isolation Forest only
    python -m train.train_anomaly_alternatives --model all       # All three
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
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ANOMALY_AE_CONFIG,
    ANOMALY_FEATURES,
    ANOMALY_IF_CONFIG,
    ANOMALY_RF_CONFIG,
    ANOMALY_TARGETS,
    ARTIFACTS_DIR,
)
from evaluate.evaluate_anomaly import (
    plot_anomaly_distribution,
    synthetic_injection_test,
    threshold_sensitivity,
)
from train.train_anomaly import load_anomaly_data, temporal_split

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Shared Evaluation ─────────────────────────────────────────────────


def run_unified_evaluation(
    arch_name: str,
    score_fn,
    val_data: np.ndarray,
    test_data: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Run standard anomaly evaluation: threshold calibration, injection test, sensitivity.

    Calibrates thresholds on **validation** data (matching XGBoost v2 approach).

    Args:
        arch_name: Architecture name for printing/metadata
        score_fn: callable(features_array) -> scores_array
        val_data: (n, d) validation feature array
        test_data: (n, d) test feature array
        feature_names: list of feature column names

    Returns:
        dict with thresholds, injection_test results, split counts
    """
    # Score validation and test sets
    val_scores = score_fn(val_data)
    test_scores = score_fn(test_data)

    # Calibrate thresholds on validation data
    p95 = float(np.percentile(val_scores, 95))
    p97 = float(np.percentile(val_scores, 97))
    p99 = float(np.percentile(val_scores, 99))

    print(f"\n  {arch_name} — Anomaly Score Statistics (Validation Data)")
    print(f"  {'=' * 50}")
    print(f"  Mean:   {val_scores.mean():.4f}")
    print(f"  Std:    {val_scores.std():.4f}")
    print(f"  p95:    {p95:.4f}")
    print(f"  p97:    {p97:.4f}")
    print(f"  p99:    {p99:.4f}")
    print(f"  Max:    {val_scores.max():.4f}")

    print(f"\n  Test set scores: mean={test_scores.mean():.4f}, max={test_scores.max():.4f}")
    n_flagged_99 = int((test_scores > p99).sum())
    print(f"  Flagged at p99 threshold: {n_flagged_99}/{len(test_scores)} "
          f"({n_flagged_99/len(test_scores)*100:.1f}%)")

    # Synthetic injection test (same protocol as XGBoost)
    print(f"\n  --- {arch_name}: Synthetic Anomaly Injection Test ---")
    injection_results = synthetic_injection_test(
        score_fn,
        test_data,
        feature_names,
        n_injections=500,
        corruption_factor=1.5,
        threshold=p99,
    )
    print(f"  Detection rate (port power x1.5): {injection_results['detection_rate']:.1f}%")
    print(f"  False positive rate: {injection_results['false_positive_rate']:.1f}%")

    # Threshold sensitivity
    rng = np.random.RandomState(42)
    corrupted = test_data[rng.choice(len(test_data), 500, replace=True)].copy()
    port_idx = feature_names.index("port_motor_power")
    corrupted[:, port_idx] *= 1.5
    corrupted_scores = score_fn(corrupted)
    threshold_sensitivity(test_scores, corrupted_scores)

    return {
        "thresholds": {"p95": p95, "p97": p97, "p99": p99},
        "injection_test": injection_results,
    }


# ── RF Reconstruction ─────────────────────────────────────────────────


def _rf_anomaly_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """Optuna objective for one RF reconstruction model."""
    max_features_choice = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"]
    )
    if max_features_choice in ("sqrt", "log2"):
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        max_depth=trial.suggest_int("max_depth", 3, 25),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
        max_features=max_features,
        max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return float(np.mean(np.abs(y_val - y_pred)))


def _compute_rf_scores(
    models: dict[str, RandomForestRegressor],
    data: np.ndarray,
    feature_names: list[str],
    feature_stds: dict[str, float],
) -> np.ndarray:
    """Compute sum-of-squared-normalized reconstruction errors for RF models."""
    scores = np.zeros(len(data))
    for target_col, model in models.items():
        predictor_idxs = [feature_names.index(c) for c in feature_names if c != target_col]
        target_idx = feature_names.index(target_col)

        X = data[:, predictor_idxs]
        y = data[:, target_idx]
        pred = model.predict(X)
        error = ((pred - y) / feature_stds[target_col]) ** 2
        scores += error

    return scores


def train_rf_anomaly(verbose: bool = True) -> dict:
    """Train RF reconstruction models for anomaly detection."""
    print("\n" + "=" * 70)
    print("  MODEL 2 ALT: Motor Anomaly Detection (Random Forest Reconstruction)")
    print("=" * 70)

    df = load_anomaly_data()
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Feature statistics for normalization (from training data)
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_stds[feat] = float(max(np.std(col), 0.01))

    models: dict[str, RandomForestRegressor] = {}
    n_trials = ANOMALY_RF_CONFIG["n_trials"]

    for target_col in ANOMALY_TARGETS:
        print(f"\n  --- RF reconstruction for: {target_col} ({n_trials} Optuna trials) ---")

        predictor_cols = [c for c in ANOMALY_FEATURES if c != target_col]
        X_train = train_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_train = train_df[target_col].to_numpy().astype(np.float32)
        X_val = val_df.select(predictor_cols).to_numpy().astype(np.float32)
        y_val = val_df[target_col].to_numpy().astype(np.float32)

        t0 = time.time()
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=ANOMALY_RF_CONFIG["n_startup_trials"],
                seed=42,
            ),
        )
        study.optimize(
            lambda trial: _rf_anomaly_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_trial
        elapsed = time.time() - t0
        print(f"  Optuna done in {elapsed/60:.1f} min | Best val MAE: {best.value:.4f}")

        # Train final model with best params
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
        models[target_col] = final_model

        # Save individual model
        import joblib
        model_path = ARTIFACTS_DIR / f"rf_anomaly_{target_col}.joblib"
        joblib.dump(final_model, model_path)
        print(f"  Saved: {model_path.name} ({model_path.stat().st_size / 1024:.0f} KB)")

    # Score function for unified evaluation
    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    test_data = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

    def score_fn(data: np.ndarray) -> np.ndarray:
        return _compute_rf_scores(models, data, ANOMALY_FEATURES, feature_stds)

    eval_results = run_unified_evaluation(
        "RF Reconstruction", score_fn, val_data, test_data, ANOMALY_FEATURES,
    )

    # Plot distribution
    val_scores = score_fn(val_data)
    plot_anomaly_distribution(
        val_scores,
        eval_results["thresholds"]["p99"],
        title="RF Reconstruction — Anomaly Score Distribution (Validation)",
        save_path=ARTIFACTS_DIR / "rf_anomaly_score_distribution.png",
    )

    # Save metadata
    meta = {
        "model": "rf_anomaly_reconstruction",
        "architecture": "RandomForest",
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "targets": ANOMALY_TARGETS,
        "optuna_trials_per_target": n_trials,
        "thresholds": eval_results["thresholds"],
        "injection_test": eval_results["injection_test"],
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    meta_path = ARTIFACTS_DIR / "rf_anomaly_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  Metadata saved: {meta_path.name}")
    return meta


# ── MLP Autoencoder ───────────────────────────────────────────────────


def train_ae_anomaly(verbose: bool = True) -> dict:
    """Train MLP Autoencoder for anomaly detection."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  MODEL 2 ALT: Motor Anomaly Detection (MLP Autoencoder)")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    df = load_anomaly_data()
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Prepare data
    X_train_raw = train_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    X_val_raw = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    X_test_raw = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

    # StandardScaler fit on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    input_dim = X_train.shape[1]  # 14 (all ANOMALY_FEATURES)

    # Indices of the 4 ANOMALY_TARGETS within ANOMALY_FEATURES (for scoring)
    target_idxs = [ANOMALY_FEATURES.index(t) for t in ANOMALY_TARGETS]

    # Feature stds on raw training data (for scoring normalization)
    feature_stds = {}
    for feat in ANOMALY_FEATURES:
        col = train_df[feat].to_numpy().astype(np.float32)
        feature_stds[feat] = float(max(np.std(col), 0.01))

    # ── Autoencoder Architecture ──

    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, hidden_sizes: list[int],
                     bottleneck: int, dropout: float):
            super().__init__()
            # Encoder
            encoder_layers = []
            prev = input_dim
            for h in hidden_sizes:
                encoder_layers.append(nn.Linear(prev, h))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout))
                prev = h
            encoder_layers.append(nn.Linear(prev, bottleneck))
            encoder_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder (mirror)
            decoder_layers = []
            prev = bottleneck
            for h in reversed(hidden_sizes):
                decoder_layers.append(nn.Linear(prev, h))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
                prev = h
            decoder_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    # ── Score function (only 4 ANOMALY_TARGETS for fair comparison) ──

    def ae_score_fn_factory(model, scaler_obj, target_idxs_list, feat_stds, feat_names):
        """Create a score function for the autoencoder."""
        def score_fn(data: np.ndarray) -> np.ndarray:
            scaled = scaler_obj.transform(data).astype(np.float32)
            model.eval()
            with torch.no_grad():
                X_t = torch.tensor(scaled, dtype=torch.float32).to(DEVICE)
                recon = model(X_t).cpu().numpy()

            # Inverse-transform to get reconstruction in original scale
            recon_orig = scaler_obj.inverse_transform(recon)

            # Sum of squared normalized errors on the 4 target features only
            scores = np.zeros(len(data))
            for idx in target_idxs_list:
                feat_name = feat_names[idx]
                error = ((recon_orig[:, idx] - data[:, idx]) / feat_stds[feat_name]) ** 2
                scores += error
            return scores
        return score_fn

    # ── Optuna search ──

    cfg = ANOMALY_AE_CONFIG
    n_trials = cfg["optuna_trials"]
    print(f"\n  Optuna Bayesian Optimization ({n_trials} trials)...")

    def ae_objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_sizes = []
        for i in range(n_layers):
            h = trial.suggest_int(f"hidden_{i}", 16, 128)
            hidden_sizes.append(h)
        bottleneck = trial.suggest_int("bottleneck", 3, 10)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

        model = Autoencoder(input_dim, hidden_sizes, bottleneck, dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10,
        )
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(cfg["max_epochs"]):
            model.train()
            for (batch,) in loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
                val_recon = model(X_val_t)
                val_loss = criterion(val_recon, X_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if patience_counter >= cfg["patience"]:
                break

        return best_val_loss

    t0 = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    study.optimize(ae_objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna done in {elapsed/60:.1f} min | Best val MSE: {best.value:.6f}")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model ──
    print("\n  Training final autoencoder...")

    n_layers = best.params["n_layers"]
    hidden_sizes = [best.params[f"hidden_{i}"] for i in range(n_layers)]
    bottleneck = best.params["bottleneck"]
    dropout = best.params["dropout"]
    lr = best.params["learning_rate"]
    batch_size = best.params["batch_size"]

    final_model = Autoencoder(input_dim, hidden_sizes, bottleneck, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        final_model.train()
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = final_model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        final_model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            val_recon = final_model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in final_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: val_mse={val_loss:.6f}")

        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch + 1} (best val MSE: {best_val_loss:.6f})")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # Score function
    score_fn = ae_score_fn_factory(
        final_model, scaler, target_idxs, feature_stds, ANOMALY_FEATURES,
    )

    # Unified evaluation
    val_data = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    test_data = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

    eval_results = run_unified_evaluation(
        "MLP Autoencoder", score_fn, val_data, test_data, ANOMALY_FEATURES,
    )

    # Plot
    val_scores = score_fn(val_data)
    plot_anomaly_distribution(
        val_scores,
        eval_results["thresholds"]["p99"],
        title="MLP Autoencoder — Anomaly Score Distribution (Validation)",
        save_path=ARTIFACTS_DIR / "ae_anomaly_score_distribution.png",
    )

    # Save model
    architecture_info = {
        "type": "autoencoder",
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "bottleneck": bottleneck,
        "dropout": dropout,
    }
    model_path = ARTIFACTS_DIR / "ae_anomaly_model.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "architecture": architecture_info,
    }, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "model": "ae_anomaly_autoencoder",
        "architecture": "MLP Autoencoder",
        "architecture_detail": architecture_info,
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "targets": ANOMALY_TARGETS,
        "scoring_targets": ANOMALY_TARGETS,
        "optuna_trials": n_trials,
        "best_val_mse": best_val_loss,
        "best_params": best.params,
        "standardized": True,
        "device": str(DEVICE),
        "thresholds": eval_results["thresholds"],
        "injection_test": eval_results["injection_test"],
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    meta_path = ARTIFACTS_DIR / "ae_anomaly_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Metadata saved: {meta_path.name}")
    return meta


# ── Isolation Forest ──────────────────────────────────────────────────


def train_if_anomaly(verbose: bool = True) -> dict:
    """Train Isolation Forest for anomaly detection."""
    print("\n" + "=" * 70)
    print("  MODEL 2 ALT: Motor Anomaly Detection (Isolation Forest)")
    print("=" * 70)

    df = load_anomaly_data()
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    X_train = train_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    X_val = val_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)
    X_test = test_df.select(ANOMALY_FEATURES).to_numpy().astype(np.float32)

    # Prepare synthetic corruptions on validation data for Optuna objective
    rng = np.random.RandomState(42)
    val_corrupt_idxs = rng.choice(len(X_val), size=min(500, len(X_val)), replace=True)
    val_corrupted = X_val[val_corrupt_idxs].copy()
    port_idx = ANOMALY_FEATURES.index("port_motor_power")
    signs = rng.choice([-1, 1], size=len(val_corrupted))
    val_corrupted[:, port_idx] *= 1.5 * np.where(signs == 1, 1, -0.5)

    n_trials = ANOMALY_IF_CONFIG["n_trials"]
    print(f"\n  Optuna Bayesian Optimization ({n_trials} trials)...")

    def if_objective(trial: optuna.Trial) -> float:
        model = IsolationForest(
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            max_samples=trial.suggest_float("max_samples", 0.1, 1.0),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            contamination=trial.suggest_float("contamination", 0.005, 0.05, log=True),
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

        # Score: negate decision_function (higher = more anomalous)
        val_scores = -model.decision_function(X_val)
        corrupt_scores = -model.decision_function(val_corrupted)

        # Use p99 threshold on validation data
        threshold = float(np.percentile(val_scores, 99))

        # Maximize detection rate on synthetic corruptions
        detection_rate = float((corrupt_scores > threshold).sum()) / len(corrupt_scores)
        return -detection_rate  # Minimize negative detection rate

    t0 = time.time()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=ANOMALY_IF_CONFIG["n_startup_trials"],
            seed=42,
        ),
    )
    study.optimize(if_objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna done in {elapsed/60:.1f} min | Best detection rate: {-best.value:.1%}")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model ──
    print("\n  Training final Isolation Forest...")

    final_model = IsolationForest(
        n_estimators=best.params["n_estimators"],
        max_samples=best.params["max_samples"],
        max_features=best.params["max_features"],
        contamination=best.params["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train)

    # Score function
    def score_fn(data: np.ndarray) -> np.ndarray:
        return -final_model.decision_function(data)

    # Unified evaluation
    eval_results = run_unified_evaluation(
        "Isolation Forest", score_fn, X_val, X_test, ANOMALY_FEATURES,
    )

    # Plot
    val_scores = score_fn(X_val)
    plot_anomaly_distribution(
        val_scores,
        eval_results["thresholds"]["p99"],
        title="Isolation Forest — Anomaly Score Distribution (Validation)",
        save_path=ARTIFACTS_DIR / "if_anomaly_score_distribution.png",
    )

    # Save model
    import joblib
    model_path = ARTIFACTS_DIR / "if_anomaly_model.joblib"
    joblib.dump(final_model, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / 1024:.0f} KB)")

    # Save metadata
    meta = {
        "model": "if_anomaly_isolation_forest",
        "architecture": "Isolation Forest",
        "trained_at": datetime.now().isoformat(),
        "features": ANOMALY_FEATURES,
        "paradigm": "direct_scoring",
        "optuna_trials": n_trials,
        "best_params": best.params,
        "thresholds": eval_results["thresholds"],
        "injection_test": eval_results["injection_test"],
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "model_size_bytes": model_size,
    }
    meta_path = ARTIFACTS_DIR / "if_anomaly_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  Metadata saved: {meta_path.name}")
    return meta


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train alternative anomaly detection architectures"
    )
    parser.add_argument(
        "--model", choices=["rf", "ae", "if", "all"],
        default="all", help="Which architecture to train",
    )
    args = parser.parse_args()

    start_time = time.time()
    results = {}

    if args.model in ("rf", "all"):
        results["rf"] = train_rf_anomaly()

    if args.model in ("ae", "all"):
        results["ae"] = train_ae_anomaly()

    if args.model in ("if", "all"):
        results["if"] = train_if_anomaly()

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  Total anomaly alternatives training time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")

    # Summary table
    print("\n  Architecture Comparison Summary:")
    print(f"  {'Architecture':<25s} | {'Detection':>10s} | {'FPR':>8s} | {'Threshold':>10s}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for name, meta in results.items():
        inj = meta["injection_test"]
        print(f"  {meta['architecture']:<25s} | {inj['detection_rate']:>9.1f}% | "
              f"{inj['false_positive_rate']:>7.1f}% | {inj['threshold']:>10.4f}")


if __name__ == "__main__":
    main()
