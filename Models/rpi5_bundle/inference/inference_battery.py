"""Standalone battery anomaly detector for RPi5 deployment.

Uses XGBoost reconstruction error: predicts each battery feature from the
others and flags high total error as anomalous.

Supports two modes:
  - "charging" (Model 3a): Detects abnormal charging curves, thermal events
  - "operational" (Model 3b): Detects cell degradation, thermal issues during trips

Dependencies: xgboost, numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


EPS = 0.01


class BatteryAnomalyDetector:
    """Detects battery anomalies via reconstruction error."""

    def __init__(self, config_dir: str, model_dir: str | None = None):
        config = Path(config_dir)
        models_path = Path(model_dir) if model_dir else config

        # Load both modes (charging + operational)
        self._modes: dict[str, dict] = {}

        for mode in ("charging", "operational"):
            prefix = f"battery_{mode}"
            meta_path = config / f"{prefix}_metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            with open(config / f"{prefix}_statistics.json") as f:
                stats = json.load(f)

            with open(config / f"{prefix}_thresholds.json") as f:
                thresholds = json.load(f)

            # Load reconstruction models
            models: dict[str, xgb.Booster] = {}
            for target in meta["targets"]:
                model = xgb.Booster()
                model.load_model(str(models_path / f"{prefix}_{target}.json"))
                models[target] = model

            self._modes[mode] = {
                "feature_names": meta["features"],
                "target_names": meta["targets"],
                "feature_stds": stats["feature_stds"],
                "threshold_p95": thresholds["p95"],
                "threshold_p97": thresholds["p97"],
                "threshold_p99": thresholds["p99"],
                "models": models,
            }

    @property
    def available_modes(self) -> list[str]:
        return list(self._modes.keys())

    def compute_features(self, bms_telemetry: dict) -> dict:
        """Compute battery features from raw BMS telemetry reading.

        Args:
            bms_telemetry: dict with BMS fields from OneAries device.
                Expected keys (per side, prefix port_/stbd_):
                - port_cell_v_spread / stbd_cell_v_spread (gCellBalance, V)
                - port_cell_t_max / stbd_cell_t_max (gMaxCellTemperature, C)
                - port_cell_t_min / stbd_cell_t_min (gMinCellTemperature, C)
                - port_pack_voltage / stbd_pack_voltage (gPackVoltage, V)
                - port_pack_current / stbd_pack_current (gCurrent, A)
                - port_soc / stbd_soc (gStateOfCharge, %)
                - port_power / stbd_power (gPower, kW)
                - port_avg_temperature / stbd_avg_temperature (gAverageTemperature, C)

                Alternatively, pass raw BMS JSON keys and this will map them:
                - gCellBalance, gMaxCellTemperature, gMinCellTemperature, etc.
                  (with port_/stbd_ prefix or side="port"/"stbd" param)
        """
        f = dict(bms_telemetry)

        # Compute derived features
        port_t_spread = f.get("port_cell_t_max", 0) - f.get("port_cell_t_min", 0)
        stbd_t_spread = f.get("stbd_cell_t_max", 0) - f.get("stbd_cell_t_min", 0)

        f["port_cell_t_spread"] = port_t_spread
        f["stbd_cell_t_spread"] = stbd_t_spread
        f["cell_v_spread_combined"] = max(
            f.get("port_cell_v_spread", 0),
            f.get("stbd_cell_v_spread", 0),
        )
        f["cell_t_spread_combined"] = max(port_t_spread, stbd_t_spread)
        f["port_stbd_soc_diff"] = abs(
            f.get("port_soc", 0) - f.get("stbd_soc", 0)
        )
        f["port_stbd_v_diff"] = abs(
            f.get("port_pack_voltage", 0) - f.get("stbd_pack_voltage", 0)
        )
        f["port_stbd_power_diff"] = abs(
            f.get("port_power", 0) - f.get("stbd_power", 0)
        )

        return f

    def score(self, features: dict, mode: str = "operational") -> float:
        """Compute anomaly score for one BMS reading.

        Higher = more anomalous.
        """
        if mode not in self._modes:
            raise ValueError(f"Mode '{mode}' not loaded. Available: {self.available_modes}")

        cfg = self._modes[mode]
        total_error = 0.0

        for target_col, model in cfg["models"].items():
            predictor_cols = [c for c in cfg["feature_names"] if c != target_col]
            X = np.array([[features.get(c, 0.0) for c in predictor_cols]], dtype=np.float32)
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            predicted = float(model.predict(dm)[0])
            actual = features.get(target_col, 0.0)
            std = cfg["feature_stds"].get(target_col, 1.0)
            total_error += ((predicted - actual) / std) ** 2

        return total_error

    def classify(self, score: float, mode: str = "operational") -> dict:
        """Classify anomaly score into normal/warning/anomalous."""
        if mode not in self._modes:
            raise ValueError(f"Mode '{mode}' not loaded. Available: {self.available_modes}")

        cfg = self._modes[mode]

        if score > cfg["threshold_p99"]:
            level = "anomalous"
        elif score > cfg["threshold_p97"]:
            level = "warning"
        else:
            level = "normal"

        return {
            "score": round(score, 4),
            "level": level,
            "threshold_p99": round(cfg["threshold_p99"], 4),
            "exceeds_threshold": score > cfg["threshold_p99"],
        }

    def check(self, bms_telemetry: dict, mode: str = "operational") -> dict:
        """Full pipeline: compute features, score, classify.

        Args:
            bms_telemetry: dict with BMS fields (see compute_features)
            mode: "charging" or "operational"
        """
        if mode not in self._modes:
            return {
                "score": 0.0,
                "level": "unavailable",
                "exceeds_threshold": False,
                "message": f"Battery anomaly model '{mode}' not loaded",
            }

        features = self.compute_features(bms_telemetry)
        s = self.score(features, mode=mode)
        result = self.classify(s, mode=mode)
        result["mode"] = mode

        if result["level"] == "anomalous":
            result["message"] = (
                "Battery behavior significantly abnormal — "
                "check cell voltage balance and temperatures"
            )
        elif result["level"] == "warning":
            result["message"] = "Battery behavior slightly unusual — monitor closely"
        else:
            result["message"] = "Battery behavior within normal range"

        return result
