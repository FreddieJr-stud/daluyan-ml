"""Package model artifacts and inference scripts for RPi5 deployment.

Creates a self-contained directory that can be copied to the RPi5.
Supports v2 ensemble models.

Usage:
    python -m deploy.package_for_rpi [--output_dir PATH]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ARTIFACTS_DIR, MODELS_ROOT

# Core model files (always included)
MODEL_FILES = [
    "soc_trip_model.json",
    "soc_realtime_model.json",
    "anomaly_port_motor_power.json",
    "anomaly_stbd_motor_power.json",
    "anomaly_port_rpm.json",
    "anomaly_stbd_rpm.json",
]

# Ensemble model files (included if they exist)
ENSEMBLE_PATTERNS = [
    "soc_trip_model_seed*.json",
    "soc_realtime_model_seed*.json",
    # v4: per-direction models
    "soc_trip_downstream_model*.json",
    "soc_trip_upstream_model*.json",
    # v4: quantile models
    "soc_trip_model_q10.json",
    "soc_trip_model_q90.json",
]

CONFIG_FILES = [
    "soc_trip_metadata.json",
    "soc_realtime_metadata.json",
    "anomaly_metadata.json",
    "feature_statistics.json",
    "thresholds.json",
    # v4: uncertainty quantification
    "conformal_trip.json",
    "conformal_realtime.json",
    "soc_trip_directional_metadata.json",
]

INFERENCE_SCRIPTS = [
    "inference_soc_trip.py",
    "inference_soc_realtime.py",
    "inference_anomaly.py",
]

# v3: Feature modules needed for inference (tide prediction)
FEATURE_MODULES = [
    "features/__init__.py",
    "features/tide.py",
]


def package(output_dir: Path | None = None) -> Path:
    """Create deployment bundle."""
    if output_dir is None:
        output_dir = MODELS_ROOT / "rpi5_bundle"

    # Clean and create
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    models_dir = output_dir / "models"
    config_dir = output_dir / "config"
    inference_dir = output_dir / "inference"
    models_dir.mkdir()
    config_dir.mkdir()
    inference_dir.mkdir()

    # Copy core model files
    for f in MODEL_FILES:
        src = ARTIFACTS_DIR / f
        if src.exists():
            shutil.copy2(src, models_dir / f)
            print(f"  Model: {f} ({src.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  WARNING: {f} not found, skipping")

    # Copy ensemble model files
    ensemble_count = 0
    for pattern in ENSEMBLE_PATTERNS:
        for src in sorted(ARTIFACTS_DIR.glob(pattern)):
            shutil.copy2(src, models_dir / src.name)
            ensemble_count += 1
            print(f"  Ensemble: {src.name} ({src.stat().st_size / 1024:.0f} KB)")

    if ensemble_count > 0:
        print(f"  ({ensemble_count} ensemble members total)")

    # Copy config files
    for f in CONFIG_FILES:
        src = ARTIFACTS_DIR / f
        if src.exists():
            shutil.copy2(src, config_dir / f)
            print(f"  Config: {f}")

    # Copy inference scripts
    deploy_dir = MODELS_ROOT / "deploy"
    for f in INFERENCE_SCRIPTS:
        src = deploy_dir / f
        if src.exists():
            shutil.copy2(src, inference_dir / f)
            print(f"  Script: {f}")

    # Copy feature modules (v3: tide prediction needed at inference time)
    for f in FEATURE_MODULES:
        src = MODELS_ROOT / f
        dst = inference_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Feature module: {f}")

    # Write requirements.txt
    (output_dir / "requirements.txt").write_text("xgboost>=2.0.0\nnumpy>=1.26.0\n")

    # Write a minimal README
    (output_dir / "README.txt").write_text(
        "M/B Dalaray Digital Shadow - RPi5 Inference Bundle (v4)\n"
        "=======================================================\n\n"
        "Install: pip install -r requirements.txt\n\n"
        "Models:\n"
        "  - soc_trip_model*.json: Trip-level SOC consumption (5-model ensemble)\n"
        "  - soc_trip_{downstream,upstream}_model*.json: Per-direction models (v4)\n"
        "  - soc_trip_model_q{10,90}.json: Quantile regression bounds (v4)\n"
        "  - soc_realtime_model*.json: Real-time SOC range estimator (5-model ensemble)\n"
        "  - anomaly_*.json: Motor anomaly reconstruction models (4 targets)\n\n"
        "Config:\n"
        "  - conformal_*.json: Conformal prediction calibration (v4)\n\n"
        "Inference scripts in inference/ are standalone (xgboost + numpy only).\n"
        "See config/ for feature names, metadata, and anomaly thresholds.\n"
    )

    # Report bundle size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    model_size = sum(f.stat().st_size for f in models_dir.rglob("*") if f.is_file())
    print(f"\nBundle created: {output_dir}")
    print(f"  Models: {model_size / 1024:.1f} KB ({model_size / (1024*1024):.1f} MB)")
    print(f"  Total:  {total_size / 1024:.1f} KB ({total_size / (1024*1024):.1f} MB)")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    package(args.output_dir)
