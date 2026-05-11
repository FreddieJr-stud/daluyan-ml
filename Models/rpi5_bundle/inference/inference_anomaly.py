"""Standalone motor anomaly detector for RPi5 deployment.

Uses XGBoost reconstruction error: predicts each motor feature from the
others and flags high total error as anomalous.

Dependencies: xgboost, numpy only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


EPS = 0.01


class MotorAnomalyDetector:
    """Detects motor anomalies via reconstruction error."""

    def __init__(self, config_dir: str, model_dir: str | None = None):
        config = Path(config_dir)
        models_path = Path(model_dir) if model_dir else config

        with open(config / "anomaly_metadata.json") as f:
            meta = json.load(f)
        self.feature_names = meta["features"]
        self.target_names = meta["targets"]

        with open(config / "feature_statistics.json") as f:
            stats = json.load(f)
        self.feature_stds = stats["feature_stds"]

        with open(config / "thresholds.json") as f:
            thresholds = json.load(f)
        self.threshold_p95 = thresholds["p95"]
        self.threshold_p97 = thresholds["p97"]
        self.threshold_p99 = thresholds["p99"]

        # Load reconstruction models
        self.models: dict[str, xgb.Booster] = {}
        for target in self.target_names:
            model = xgb.Booster()
            model.load_model(str(models_path / f"anomaly_{target}.json"))
            self.models[target] = model

    def compute_features(self, telemetry: dict) -> dict:
        """Compute anomaly features from raw telemetry.

        Args:
            telemetry: dict with port_motor_power, stbd_motor_power,
                       motor_power_combined, port_rpm, stbd_rpm,
                       speed_over_ground, battery_power
        """
        pp = telemetry["port_motor_power"]
        sp = telemetry["stbd_motor_power"]
        mc = telemetry["motor_power_combined"]
        pr = telemetry["port_rpm"]
        sr = telemetry["stbd_rpm"]
        sog = telemetry["speed_over_ground"]
        bp = telemetry["battery_power"]

        return {
            "port_motor_power": pp,
            "stbd_motor_power": sp,
            "motor_power_combined": mc,
            "port_rpm": pr,
            "stbd_rpm": sr,
            "speed_over_ground": sog,
            "battery_power": bp,
            "port_stbd_power_ratio": pp / (pp + sp + EPS),
            "port_stbd_power_diff": abs(pp - sp),
            "port_stbd_rpm_ratio": pr / (pr + sr + EPS),
            "port_stbd_rpm_diff": abs(pr - sr),
            "port_power_per_rpm": pp / (pr + EPS),
            "stbd_power_per_rpm": sp / (sr + EPS),
            "combined_power_per_speed": mc / (sog + EPS),
            "power_speed_squared_ratio": mc / (sog ** 2 + EPS),
        }

    def score(self, features: dict) -> float:
        """Compute anomaly score for one telemetry reading.

        Higher = more anomalous.
        """
        total_error = 0.0
        for target_col, model in self.models.items():
            predictor_cols = [c for c in self.feature_names if c != target_col]
            X = np.array([[features[c] for c in predictor_cols]], dtype=np.float32)
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            predicted = float(model.predict(dm)[0])
            actual = features[target_col]
            std = self.feature_stds[target_col]
            total_error += ((predicted - actual) / std) ** 2

        return total_error

    def classify(self, score: float, threshold: str = "p99") -> dict:
        """Classify anomaly score into normal/warning/anomalous.

        Args:
            score: anomaly score from self.score()
            threshold: which threshold to use ("p95", "p97", "p99")
        """
        t = getattr(self, f"threshold_{threshold}")

        if score > self.threshold_p99:
            level = "anomalous"
        elif score > self.threshold_p97:
            level = "warning"
        else:
            level = "normal"

        return {
            "score": round(score, 4),
            "level": level,
            "threshold_used": threshold,
            "threshold_value": round(t, 4),
            "exceeds_threshold": score > t,
        }

    def check(self, telemetry: dict) -> dict:
        """Full pipeline: compute features, score, classify.

        Only meaningful when speed >= 0.5 kn (active transit).
        """
        if telemetry.get("speed_over_ground", 0) < 0.5:
            return {
                "score": 0.0,
                "level": "idle",
                "exceeds_threshold": False,
                "message": "Ferry not in active transit",
            }

        features = self.compute_features(telemetry)
        s = self.score(features)
        result = self.classify(s)

        if result["level"] == "anomalous":
            result["message"] = "Motor behavior significantly deviates from normal — suggest inspection"
        elif result["level"] == "warning":
            result["message"] = "Motor behavior slightly unusual — monitor closely"
        else:
            result["message"] = "Motor behavior within normal range"

        return result
