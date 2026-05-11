"""MLP + LSTM Training Pipeline for M/B Dalaray SOC models.

Fair comparison against XGBoost v4 and Random Forest:
  - Same features, targets, temporal splits, and augmentation
  - Model 1A: MLP (feedforward NN) — trip-level, 1 row per trip, no sequence
  - Model 1B: LSTM (sequence-to-sequence) — 1Hz temporal data per trip segment
  - Optuna Bayesian optimization with pruning
  - StandardScaler on features (critical for neural nets)
  - Evaluated with identical compute_metrics()

Usage:
    python -m train.train_lstm --model trip        # Model 1A (MLP) only
    python -m train.train_lstm --model realtime    # Model 1B (LSTM) only
    python -m train.train_lstm --model all         # Both models
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
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    FEATURE_STORE_DIR,
    LSTM_CONFIG,
    REALTIME_FEATURES,
    REALTIME_TARGET,
    TRAIN_END,
    TRIP_FEATURES,
    TRIP_TARGET,
    VAL_END,
)
from data.augment import augment_trip_data
from evaluate.evaluate_soc import (
    compute_metrics,
    print_direction_breakdown,
    print_metrics,
)

# Suppress Optuna INFO spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared Utilities ────────────────────────────────────────────────────

def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Temporal split using config dates (identical to train_v2.py)."""
    train = df.filter(pl.col("departure_time") <= datetime.fromisoformat(TRAIN_END))
    val = df.filter(
        (pl.col("departure_time") > datetime.fromisoformat(TRAIN_END))
        & (pl.col("departure_time") <= datetime.fromisoformat(VAL_END))
    )
    test = df.filter(pl.col("departure_time") > datetime.fromisoformat(VAL_END))
    return train, val, test


def df_to_numpy(
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to numpy arrays."""
    X = df.select(features).to_numpy().astype(np.float32)
    y = df[target].to_numpy().astype(np.float32)
    return X, y


# ── MLP Architecture (Model 1A) ─────────────────────────────────────────

class SOCPredictor_MLP(nn.Module):
    """Feedforward NN for trip-level SOC prediction."""

    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ── LSTM Architecture (Model 1B) ────────────────────────────────────────

class SOCPredictor_LSTM(nn.Module):
    """LSTM for real-time SOC sequence-to-sequence prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with packed sequences for variable-length trips.

        Args:
            x: padded input (batch, max_seq_len, input_dim)
            lengths: actual lengths per sequence (batch,)

        Returns:
            predictions at each timestep (batch, max_seq_len)
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Project each timestep to scalar
        predictions = self.fc(out).squeeze(-1)  # (batch, max_seq_len)
        return predictions


# ── Datasets ─────────────────────────────────────────────────────────────

class TripDataset(Dataset):
    """Simple dataset for MLP (Model 1A)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """Dataset for LSTM (Model 1B) — one item per trip segment."""

    def __init__(self, sequences: list[np.ndarray], targets: list[np.ndarray]):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = [torch.tensor(t, dtype=torch.float32) for t in targets]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def sequence_collate_fn(batch):
    """Collate variable-length sequences with padding."""
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    return padded_seqs, padded_targets, lengths


# ── MLP Training (Model 1A) ─────────────────────────────────────────────

def _train_mlp_one_epoch(
    model: SOCPredictor_MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Train MLP for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_mlp(
    model: SOCPredictor_MLP,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Evaluate MLP, return MAE."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).cpu().numpy()
    return float(np.mean(np.abs(y - preds)))


def _mlp_optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> float:
    """Optuna objective for MLP Model 1A."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_sizes = []
    for i in range(n_layers):
        h = trial.suggest_int(f"hidden_{i}", 32, 256)
        hidden_sizes.append(h)

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    input_dim = X_train.shape[1]
    model = SOCPredictor_MLP(input_dim, hidden_sizes, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    dataset = TripDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        _train_mlp_one_epoch(model, loader, optimizer, criterion)
        val_mae = _eval_mlp(model, X_val, y_val)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= cfg["patience"]:
            break

    return best_val_mae


def train_trip_mlp(verbose: bool = True) -> dict:
    """Train Model 1A (trip-level SOC) with MLP."""
    cfg = LSTM_CONFIG["model_1a"]
    print("\n" + "=" * 60)
    print("  MODEL 1A: Trip-Level SOC Prediction (MLP)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load & split
    df = pl.read_parquet(FEATURE_STORE_DIR / "trip_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    print(f"  Split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Data augmentation (same as XGB/RF)
    print("\n  [1/4] Data Augmentation...")
    train_aug = augment_trip_data(train_df, TRIP_FEATURES, TRIP_TARGET, verbose=verbose)

    X_train_raw, y_train = df_to_numpy(train_aug, TRIP_FEATURES, TRIP_TARGET)
    X_train_orig_raw, y_train_orig = df_to_numpy(train_df, TRIP_FEATURES, TRIP_TARGET)
    X_val_raw, y_val = df_to_numpy(val_df, TRIP_FEATURES, TRIP_TARGET)
    X_test_raw, y_test = df_to_numpy(test_df, TRIP_FEATURES, TRIP_TARGET)

    # StandardScaler (fit on augmented training data)
    print("  [2/4] StandardScaler fit on training data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_train_orig = scaler.transform(X_train_orig_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    # ── Optuna search ──
    n_trials = cfg["optuna_trials"]
    print(f"\n  [3/4] Optuna Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    study.optimize(
        lambda trial: _mlp_optuna_objective(trial, X_train, y_train, X_val, y_val, cfg),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model with best params ──
    print("\n  [4/4] Training final model + Evaluation...")

    n_layers = best.params["n_layers"]
    hidden_sizes = [best.params[f"hidden_{i}"] for i in range(n_layers)]
    dropout = best.params["dropout"]
    lr = best.params["learning_rate"]
    weight_decay = best.params["weight_decay"]
    batch_size = best.params["batch_size"]

    input_dim = X_train.shape[1]
    final_model = SOCPredictor_MLP(input_dim, hidden_sizes, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    dataset = TripDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        _train_mlp_one_epoch(final_model, loader, optimizer, criterion)
        val_mae = _eval_mlp(final_model, X_val, y_val)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in final_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch + 1} (best val MAE: {best_val_mae:.4f}%)")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ── Evaluate ──
    final_model.eval()
    results = {}
    for name, X, y_true, split_df in [
        ("Train", X_train_orig, y_train_orig, train_df),
        ("Validation", X_val, y_val, val_df),
        ("Test", X_test, y_test, test_df),
    ]:
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            y_pred = np.clip(final_model(X_t).cpu().numpy(), 0, None)
        metrics = compute_metrics(y_true, y_pred)
        results[name] = metrics
        print_metrics(name, metrics)
        if "direction_encoded" in split_df.columns:
            dirs = split_df["direction_encoded"].to_numpy()
            print_direction_breakdown(y_true, y_pred, dirs)

    # Save model
    architecture_info = {
        "type": "mlp",
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
    }
    model_path = ARTIFACTS_DIR / "mlp_soc_trip_model.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "architecture": architecture_info,
    }, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / 1024:.0f} KB)")

    # Save test predictions for RPi5 benchmark (pre-computed)
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        test_preds = np.clip(final_model(X_t).cpu().numpy(), 0, None)
    pred_path = ARTIFACTS_DIR / "mlp_trip_test_predictions.csv"
    np.savetxt(pred_path, np.column_stack([y_test, test_preds]),
               delimiter=",", header="y_true,y_pred", comments="")
    print(f"  Test predictions saved to: {pred_path.name}")

    # Save metadata
    meta = {
        "model": "mlp_soc_trip",
        "architecture": "MLP",
        "architecture_detail": architecture_info,
        "trained_at": datetime.now().isoformat(),
        "features": TRIP_FEATURES,
        "target": TRIP_TARGET,
        "best_params": best.params,
        "optuna_trials": n_trials,
        "optuna_best_val_mae": best.value,
        "augmentation_applied": True,
        "data_augmentation_factor": len(train_aug) / len(train_df),
        "standardized": True,
        "device": str(DEVICE),
        "splits": {
            "train": len(train_df),
            "train_augmented": len(train_aug),
            "val": len(val_df),
            "test": len(test_df),
        },
        "metrics": results,
        "model_size_bytes": model_size,
    }
    meta_path = ARTIFACTS_DIR / "mlp_soc_trip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


# ── LSTM Training (Model 1B) ─────────────────────────────────────────────

def _build_sequences(
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Group realtime data by segment_id into sequences."""
    sequences = []
    targets = []

    for seg_id in df["segment_id"].unique().sort().to_list():
        seg_df = df.filter(pl.col("segment_id") == seg_id).sort("elapsed_time_s")
        if len(seg_df) < 5:  # skip very short segments
            continue
        X = seg_df.select(features).to_numpy().astype(np.float32)
        y = seg_df[target].to_numpy().astype(np.float32)
        sequences.append(X)
        targets.append(y)

    return sequences, targets


def _scale_sequences(
    sequences: list[np.ndarray],
    scaler: StandardScaler,
    fit: bool = False,
) -> list[np.ndarray]:
    """Apply StandardScaler to sequences (flatten, transform, re-split)."""
    lengths = [len(s) for s in sequences]
    flat = np.vstack(sequences)

    if fit:
        flat_scaled = scaler.fit_transform(flat).astype(np.float32)
    else:
        flat_scaled = scaler.transform(flat).astype(np.float32)

    result = []
    idx = 0
    for length in lengths:
        result.append(flat_scaled[idx:idx + length])
        idx += length
    return result


def _train_lstm_one_epoch(
    model: SOCPredictor_LSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train LSTM for one epoch with masked MSE loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for padded_seqs, padded_targets, lengths in loader:
        padded_seqs = padded_seqs.to(DEVICE)
        padded_targets = padded_targets.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        preds = model(padded_seqs, lengths)

        # Masked MSE: ignore padded positions
        mask = torch.zeros_like(padded_targets, dtype=torch.bool)
        for i, l in enumerate(lengths):
            mask[i, :l] = True

        masked_preds = preds[mask]
        masked_targets = padded_targets[mask]
        loss = nn.functional.mse_loss(masked_preds, masked_targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * masked_targets.numel()
        n_samples += masked_targets.numel()

    return total_loss / max(n_samples, 1)


def _eval_lstm(
    model: SOCPredictor_LSTM,
    loader: DataLoader,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate LSTM, return MAE + flattened (y_true, y_pred)."""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for padded_seqs, padded_targets, lengths in loader:
            padded_seqs = padded_seqs.to(DEVICE)
            lengths = lengths.to(DEVICE)
            preds = model(padded_seqs, lengths)

            for i, l in enumerate(lengths):
                l = l.item()
                all_true.append(padded_targets[i, :l].numpy())
                all_pred.append(preds[i, :l].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae, y_true, y_pred


def _lstm_optuna_objective(
    trial: optuna.Trial,
    train_seqs: list[np.ndarray],
    train_targets: list[np.ndarray],
    val_loader: DataLoader,
    input_dim: int,
    cfg: dict,
) -> float:
    """Optuna objective for LSTM Model 1B."""
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    model = SOCPredictor_LSTM(input_dim, hidden_size, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_dataset = SequenceDataset(train_seqs, train_targets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=sequence_collate_fn,
    )

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        _train_lstm_one_epoch(model, train_loader, optimizer)
        val_mae, _, _ = _eval_lstm(model, val_loader)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
        else:
            patience_counter += 1

        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if patience_counter >= cfg["patience"]:
            break

    return best_val_mae


def _evaluate_lstm_by_progress(
    model: SOCPredictor_LSTM,
    test_df: pl.DataFrame,
    features: list[str],
    target: str,
    scaler: StandardScaler,
) -> dict[str, float]:
    """Evaluate LSTM MAE at different trip progress bins."""
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    progress_mae = {}
    print("\n  MAE by trip progress:")

    model.eval()
    for lo, hi in bins:
        subset = test_df.filter(
            (pl.col("trip_progress_fraction") >= lo)
            & (pl.col("trip_progress_fraction") < hi)
        )
        if len(subset) == 0:
            continue
        X_raw = subset.select(features).to_numpy().astype(np.float32)
        y_true = subset[target].to_numpy().astype(np.float32)
        X_scaled = scaler.transform(X_raw).astype(np.float32)

        # Treat each row individually (no packing needed for single-timestep eval)
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            lengths = torch.ones(len(X_t), dtype=torch.long)
            preds = model(X_t, lengths).cpu().numpy()

        mae = float(np.mean(np.abs(y_true - preds)))
        label = f"{int(lo*100)}-{int(hi*100)}%"
        progress_mae[label] = mae
        print(f"    {label}: MAE={mae:.3f}% SOC  (n={len(subset)})")

    return progress_mae


def train_realtime_lstm(verbose: bool = True) -> dict:
    """Train Model 1B (real-time SOC) with LSTM."""
    cfg = LSTM_CONFIG["model_1b"]
    print("\n" + "=" * 60)
    print("  MODEL 1B: Real-Time SOC Range Estimation (LSTM)")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Load & split
    df = pl.read_parquet(FEATURE_STORE_DIR / "realtime_features.parquet")
    train_df, val_df, test_df = temporal_split(df)

    n_seg = {
        "train": train_df["segment_id"].n_unique(),
        "val": val_df["segment_id"].n_unique(),
        "test": test_df["segment_id"].n_unique(),
    }
    print(f"  Split: Train={len(train_df)} ({n_seg['train']} segs) | "
          f"Val={len(val_df)} ({n_seg['val']} segs) | "
          f"Test={len(test_df)} ({n_seg['test']} segs)")

    # Build sequences grouped by segment_id
    print("\n  [1/5] Building sequences per segment...")
    train_seqs_raw, train_targets = _build_sequences(train_df, REALTIME_FEATURES, REALTIME_TARGET)
    val_seqs_raw, val_targets = _build_sequences(val_df, REALTIME_FEATURES, REALTIME_TARGET)
    test_seqs_raw, test_targets = _build_sequences(test_df, REALTIME_FEATURES, REALTIME_TARGET)

    print(f"  Train: {len(train_seqs_raw)} segments, "
          f"Val: {len(val_seqs_raw)} segments, "
          f"Test: {len(test_seqs_raw)} segments")

    # StandardScaler (fit on training data)
    print("  [2/5] StandardScaler fit on training data...")
    scaler = StandardScaler()
    train_seqs = _scale_sequences(train_seqs_raw, scaler, fit=True)
    val_seqs = _scale_sequences(val_seqs_raw, scaler, fit=False)
    test_seqs = _scale_sequences(test_seqs_raw, scaler, fit=False)

    input_dim = len(REALTIME_FEATURES)

    # Build val/test loaders (fixed for evaluation)
    val_dataset = SequenceDataset(val_seqs, val_targets)
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        collate_fn=sequence_collate_fn,
    )
    test_dataset = SequenceDataset(test_seqs, test_targets)
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False,
        collate_fn=sequence_collate_fn,
    )

    # ── Optuna search ──
    n_trials = cfg["optuna_trials"]
    print(f"\n  [3/5] Optuna Bayesian Optimization ({n_trials} trials)...")
    t0 = time.time()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: _lstm_optuna_objective(
            trial, train_seqs, train_targets, val_loader, input_dim, cfg,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    elapsed = time.time() - t0
    print(f"  Optuna complete in {elapsed/60:.1f} min")
    print(f"  Best val MAE: {best.value:.4f}%")
    print(f"  Best params: {json.dumps(best.params, indent=4)}")

    # ── Train final model ──
    print("\n  [4/5] Training final model...")

    hidden_size = best.params["hidden_size"]
    num_layers = best.params["num_layers"]
    dropout = best.params["dropout"]
    lr = best.params["learning_rate"]
    weight_decay = best.params["weight_decay"]
    batch_size = best.params["batch_size"]

    final_model = SOCPredictor_LSTM(input_dim, hidden_size, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )

    train_dataset = SequenceDataset(train_seqs, train_targets)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=sequence_collate_fn,
    )

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        train_loss = _train_lstm_one_epoch(final_model, train_loader, optimizer)
        val_mae, _, _ = _eval_lstm(final_model, val_loader)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in final_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f}, val_mae={val_mae:.4f}%")

        if patience_counter >= cfg["patience"]:
            print(f"  Early stopping at epoch {epoch + 1} (best val MAE: {best_val_mae:.4f}%)")
            break

    if best_state is not None:
        final_model.load_state_dict(best_state)

    # ── Evaluate ──
    print("\n  [5/5] Evaluation...")

    results = {}

    # Train eval (flatten all sequences)
    train_mae, train_true, train_pred = _eval_lstm(final_model, train_loader)
    results["Train"] = compute_metrics(train_true, np.clip(train_pred, 0, None))
    print_metrics("Train", results["Train"])

    # Val eval
    val_mae, val_true, val_pred = _eval_lstm(final_model, val_loader)
    results["Validation"] = compute_metrics(val_true, np.clip(val_pred, 0, None))
    print_metrics("Validation", results["Validation"])

    # Test eval
    test_mae, test_true, test_pred = _eval_lstm(final_model, test_loader)
    test_pred_clipped = np.clip(test_pred, 0, None)
    results["Test"] = compute_metrics(test_true, test_pred_clipped)
    print_metrics("Test", results["Test"])

    # Progress breakdown
    progress_mae = _evaluate_lstm_by_progress(
        final_model, test_df, REALTIME_FEATURES, REALTIME_TARGET, scaler,
    )
    results["progress_breakdown"] = progress_mae

    # Save model
    architecture_info = {
        "type": "lstm",
        "input_dim": input_dim,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    model_path = ARTIFACTS_DIR / "lstm_soc_realtime_model.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "architecture": architecture_info,
    }, model_path)
    model_size = model_path.stat().st_size
    print(f"\n  Saved: {model_path.name} ({model_size / 1024:.0f} KB)")

    # Save test predictions for RPi5 benchmark (pre-computed)
    pred_path = ARTIFACTS_DIR / "lstm_realtime_test_predictions.csv"
    np.savetxt(pred_path, np.column_stack([test_true, test_pred_clipped]),
               delimiter=",", header="y_true,y_pred", comments="")
    print(f"  Test predictions saved to: {pred_path.name}")

    # Save metadata
    meta = {
        "model": "lstm_soc_realtime",
        "architecture": "LSTM",
        "architecture_detail": architecture_info,
        "trained_at": datetime.now().isoformat(),
        "features": REALTIME_FEATURES,
        "target": REALTIME_TARGET,
        "best_params": best.params,
        "optuna_trials": n_trials,
        "optuna_best_val_mae": best.value,
        "standardized": True,
        "device": str(DEVICE),
        "splits": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_segments": n_seg["train"],
            "val_segments": n_seg["val"],
            "test_segments": n_seg["test"],
        },
        "metrics": results,
        "model_size_bytes": model_size,
    }
    meta_path = ARTIFACTS_DIR / "lstm_soc_realtime_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLP + LSTM Training Pipeline")
    parser.add_argument(
        "--model", choices=["trip", "realtime", "all"],
        default="all", help="Which model to train (trip=MLP, realtime=LSTM)",
    )
    args = parser.parse_args()

    start_time = time.time()
    all_results = {}

    if args.model in ("trip", "all"):
        all_results["trip_mlp"] = train_trip_mlp()

    if args.model in ("realtime", "all"):
        all_results["realtime_lstm"] = train_realtime_lstm()

    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Total MLP/LSTM training time: {total_time/60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
