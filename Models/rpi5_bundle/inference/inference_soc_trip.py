"""Standalone trip-level SOC predictor for RPi5 deployment.

Supports v3 ensemble inference with tide/current features.

Dependencies: xgboost, numpy only (tide prediction is built-in).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

# Allow importing tide module from deploy/ or from Models/
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from features.tide import compute_tide_features, predict_tide_height
except ImportError:
    # Fallback: tide module not available (return zeros)
    def compute_tide_features(dt):
        return {"tide_height_m": 0.0, "hours_since_high_tide": 6.0, "tide_phase": 0}
    def predict_tide_height(dt):
        return 0.0


class SOCTripPredictor:
    """Predicts SOC consumption for a planned trip."""

    def __init__(self, model_dir: str, config_dir: str):
        config = Path(config_dir)
        with open(config / "soc_trip_metadata.json") as f:
            meta = json.load(f)
        self.feature_names = meta["features"]

        # Load ensemble models (primary + seed variants)
        model_path = Path(model_dir)
        self.models: list[xgb.Booster] = []

        # Primary model always present
        primary = model_path / "soc_trip_model.json"
        m = xgb.Booster()
        m.load_model(str(primary))
        self.models.append(m)

        # Load additional ensemble members if present
        for f in sorted(model_path.glob("soc_trip_model_seed*.json")):
            m = xgb.Booster()
            m.load_model(str(f))
            self.models.append(m)

        # Load quantile models if present (v4)
        self.q10_model = None
        self.q90_model = None
        q10_path = model_path / "soc_trip_model_q10.json"
        q90_path = model_path / "soc_trip_model_q90.json"
        if q10_path.exists() and q90_path.exists():
            self.q10_model = xgb.Booster()
            self.q10_model.load_model(str(q10_path))
            self.q90_model = xgb.Booster()
            self.q90_model.load_model(str(q90_path))

        # Load conformal calibration if present (v4)
        self.conformal_q_hat = None
        conformal_path = config / "conformal_trip.json"
        if conformal_path.exists():
            with open(conformal_path) as f:
                conf = json.load(f)
            self.conformal_q_hat = conf.get("conformal_q_hat_80")

        # Station registry
        self.stations = {
            "Napindan": {"order": 0, "lat": 14.55713, "lon": 121.06836},
            "Guadalupe": {"order": 1, "lat": 14.56811, "lon": 121.04792},
            "Hulo": {"order": 2, "lat": 14.56794, "lon": 121.03364},
            "Valenzuela": {"order": 3, "lat": 14.57400, "lon": 121.02578},
            "Lambingan": {"order": 4, "lat": 14.58731, "lon": 121.01847},
            "Sta Ana": {"order": 5, "lat": 14.58236, "lon": 121.01167},
            "PUP": {"order": 6, "lat": 14.59603, "lon": 121.01072},
            "Quinta": {"order": 7, "lat": 14.59572, "lon": 120.98142},
            "Escolta": {"order": 8, "lat": 14.59653, "lon": 120.97761},
        }

    def predict(
        self,
        departure_station: str,
        arrival_station: str,
        current_soc: float,
        hv_capacity: float,
        target_speed_kn: float,
        passengers: int,
        temperature_c: float = 28.0,
        humidity: float = 75.0,
        wind_speed_kn: float = 5.0,
        wind_direction_deg: float = 180.0,
        precipitation_mm: float = 0.0,
        wave_height_m: float = 0.1,
        hour: int = 10,
        departure_time: "datetime | None" = None,
        # v4: ocean current & wave params
        ocean_current_velocity_ms: float = 0.0,
        ocean_current_direction_deg: float = 0.0,
        wave_direction_deg: float = 0.0,
        wave_period_s: float = 0.0,
    ) -> float:
        """Predict SOC consumption (%) for a trip.

        Returns the predicted SOC delta (positive = consumption).
        """
        dep = self.stations[departure_station]
        arr = self.stations[arrival_station]

        dep_order = dep["order"]
        arr_order = arr["order"]
        hop_count = abs(arr_order - dep_order)
        direction = 0 if arr_order > dep_order else 1  # 0=downstream, 1=upstream

        route_heading = self._bearing(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        route_distance = self._haversine_km(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        wind_component = wind_speed_kn * math.cos(math.radians(wind_direction_deg - route_heading))
        wind_cross = wind_speed_kn * math.sin(math.radians(wind_direction_deg - route_heading))
        is_peak = int(7 <= hour <= 9 or 16 <= hour <= 19)

        # v4: Ocean current & wave projections
        current_comp = ocean_current_velocity_ms * math.cos(
            math.radians(ocean_current_direction_deg - route_heading)
        )
        wave_comp = wave_height_m * math.cos(
            math.radians(wave_direction_deg - route_heading)
        )

        # v3: Tide features
        if departure_time is not None:
            tide = compute_tide_features(departure_time)
        else:
            # Approximate: construct a datetime from hour (today)
            from datetime import datetime
            approx_dt = datetime.now().replace(hour=hour, minute=0, second=0)
            tide = compute_tide_features(approx_dt)

        features = {
            "departure_station_encoded": dep_order,
            "arrival_station_encoded": arr_order,
            "hop_count": hop_count,
            "direction_encoded": direction,
            "route_distance_km": route_distance,
            "start_soc": current_soc,
            "start_hv_capacity": hv_capacity,
            "target_speed": target_speed_kn,
            "target_speed_squared": target_speed_kn ** 2,
            "passengers_on_board": passengers,
            "passenger_load_ratio": passengers / 40.0,
            "temperature_c": temperature_c,
            "relative_humidity": humidity,
            "wind_speed_kn": wind_speed_kn,
            "wind_direction_deg": wind_direction_deg,
            "wind_component_along_route": wind_component,
            "precipitation_mm": precipitation_mm,
            "wave_height_m": wave_height_m,
            "hour_of_day": hour,
            "is_peak_hour": is_peak,
            # v2 interaction features
            "speed_distance_interaction": target_speed_kn * route_distance,
            "energy_rate_proxy": (target_speed_kn ** 2) * hop_count,
            "wind_cross_component": wind_cross,
            "soc_per_km_capacity": current_soc / (route_distance + 0.01),
            # v3 tide features
            "tide_height_m": tide["tide_height_m"],
            "hours_since_high_tide": tide["hours_since_high_tide"],
            "tide_phase": tide["tide_phase"],
            # v4 ocean current & wave features
            "current_component_along_route": current_comp,
            "current_velocity_ms": ocean_current_velocity_ms,
            "wave_component_along_route": wave_comp,
            "wave_period_s": wave_period_s,
        }

        values = [features[f] for f in self.feature_names]
        dm = xgb.DMatrix(
            np.array([values], dtype=np.float32),
            feature_names=self.feature_names,
        )

        # Ensemble: average predictions from all models
        preds = [float(m.predict(dm)[0]) for m in self.models]
        return max(sum(preds) / len(preds), 0.0)

    def predict_interval(self, method: str = "conformal", **kwargs) -> dict:
        """Predict SOC with uncertainty interval.

        Args:
            method: "conformal" (default) or "quantile"
            **kwargs: same args as predict()

        Returns:
            dict with point, lower, upper, method, coverage, ensemble_std
        """
        point = self.predict(**kwargs)

        # Build DMatrix for quantile models
        dm = self._build_dmatrix(**kwargs)
        preds = [float(m.predict(dm)[0]) for m in self.models]
        ensemble_std = float(np.std(preds))

        if method == "quantile" and self.q10_model and self.q90_model:
            lower = max(float(self.q10_model.predict(dm)[0]), 0.0)
            upper = max(float(self.q90_model.predict(dm)[0]), 0.0)
            # Ensure ordering
            if lower > upper:
                lower, upper = upper, lower
            return {
                "point": round(point, 3),
                "lower": round(lower, 3),
                "upper": round(upper, 3),
                "method": "quantile",
                "coverage": 0.80,
                "ensemble_std": round(ensemble_std, 3),
            }
        elif self.conformal_q_hat is not None:
            q = self.conformal_q_hat
            return {
                "point": round(point, 3),
                "lower": round(max(point - q, 0.0), 3),
                "upper": round(point + q, 3),
                "method": "conformal",
                "coverage": 0.80,
                "ensemble_std": round(ensemble_std, 3),
            }
        else:
            # Fallback: use ensemble std for rough interval
            return {
                "point": round(point, 3),
                "lower": round(max(point - 2 * ensemble_std, 0.0), 3),
                "upper": round(point + 2 * ensemble_std, 3),
                "method": "ensemble_std",
                "coverage": None,
                "ensemble_std": round(ensemble_std, 3),
            }

    def _build_dmatrix(self, **kwargs) -> xgb.DMatrix:
        """Build a DMatrix from predict() arguments (for reuse)."""
        dep = self.stations[kwargs["departure_station"]]
        arr = self.stations[kwargs["arrival_station"]]
        dep_order, arr_order = dep["order"], arr["order"]
        hop_count = abs(arr_order - dep_order)
        direction = 0 if arr_order > dep_order else 1

        route_heading = self._bearing(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        route_distance = self._haversine_km(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        wind_speed = kwargs.get("wind_speed_kn", 5.0)
        wind_dir = kwargs.get("wind_direction_deg", 180.0)
        wind_comp = wind_speed * math.cos(math.radians(wind_dir - route_heading))
        wind_cross = wind_speed * math.sin(math.radians(wind_dir - route_heading))
        hour = kwargs.get("hour", 10)
        target_speed = kwargs["target_speed_kn"]
        current_soc = kwargs["current_soc"]
        passengers = kwargs["passengers"]

        # v4: ocean current & wave
        oc_vel = kwargs.get("ocean_current_velocity_ms", 0.0)
        oc_dir = kwargs.get("ocean_current_direction_deg", 0.0)
        current_comp = oc_vel * math.cos(math.radians(oc_dir - route_heading))
        wave_h = kwargs.get("wave_height_m", 0.1)
        wave_d = kwargs.get("wave_direction_deg", 0.0)
        wave_comp = wave_h * math.cos(math.radians(wave_d - route_heading))
        wave_period = kwargs.get("wave_period_s", 0.0)

        departure_time = kwargs.get("departure_time")
        if departure_time is not None:
            tide = compute_tide_features(departure_time)
        else:
            from datetime import datetime as dt
            tide = compute_tide_features(dt.now().replace(hour=hour, minute=0, second=0))

        features = {
            "departure_station_encoded": dep_order,
            "arrival_station_encoded": arr_order,
            "hop_count": hop_count,
            "direction_encoded": direction,
            "route_distance_km": route_distance,
            "start_soc": current_soc,
            "start_hv_capacity": kwargs["hv_capacity"],
            "target_speed": target_speed,
            "target_speed_squared": target_speed ** 2,
            "passengers_on_board": passengers,
            "passenger_load_ratio": passengers / 40.0,
            "temperature_c": kwargs.get("temperature_c", 28.0),
            "relative_humidity": kwargs.get("humidity", 75.0),
            "wind_speed_kn": wind_speed,
            "wind_direction_deg": wind_dir,
            "wind_component_along_route": wind_comp,
            "precipitation_mm": kwargs.get("precipitation_mm", 0.0),
            "wave_height_m": wave_h,
            "hour_of_day": hour,
            "is_peak_hour": int(7 <= hour <= 9 or 16 <= hour <= 19),
            "speed_distance_interaction": target_speed * route_distance,
            "energy_rate_proxy": (target_speed ** 2) * hop_count,
            "wind_cross_component": wind_cross,
            "soc_per_km_capacity": current_soc / (route_distance + 0.01),
            "tide_height_m": tide["tide_height_m"],
            "hours_since_high_tide": tide["hours_since_high_tide"],
            "tide_phase": tide["tide_phase"],
            # v4 ocean current & wave features
            "current_component_along_route": current_comp,
            "current_velocity_ms": oc_vel,
            "wave_component_along_route": wave_comp,
            "wave_period_s": wave_period,
        }
        values = [features[f] for f in self.feature_names]
        return xgb.DMatrix(
            np.array([values], dtype=np.float32),
            feature_names=self.feature_names,
        )

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlon = lon2_r - lon1_r
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        return math.degrees(math.atan2(x, y)) % 360

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))


class DirectionalSOCTripPredictor:
    """Routes predictions to direction-specific models (v4).

    Falls back to global model if directional models are not available.
    """

    def __init__(self, model_dir: str, config_dir: str):
        model_path = Path(model_dir)

        # Try to load directional models
        self._downstream = self._load_ensemble(model_path, "downstream")
        self._upstream = self._load_ensemble(model_path, "upstream")
        self._has_directional = self._downstream is not None and self._upstream is not None

        # Always load global as fallback
        self._global = SOCTripPredictor(model_dir, config_dir)

        # Load directional feature list (no direction_encoded)
        if self._has_directional:
            config = Path(config_dir)
            meta_path = config / "soc_trip_directional_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self._dir_features = meta["features"]
            else:
                # Derive from global features
                self._dir_features = [f for f in self._global.feature_names
                                      if f != "direction_encoded"]

        if self._has_directional:
            print(f"  DirectionalSOCTripPredictor: loaded "
                  f"{len(self._downstream)} downstream + {len(self._upstream)} upstream models")
        else:
            print("  DirectionalSOCTripPredictor: directional models not found, using global")

    @staticmethod
    def _load_ensemble(model_path: Path, direction: str) -> list[xgb.Booster] | None:
        primary = model_path / f"soc_trip_{direction}_model.json"
        if not primary.exists():
            return None
        models = []
        m = xgb.Booster()
        m.load_model(str(primary))
        models.append(m)
        for f in sorted(model_path.glob(f"soc_trip_{direction}_model_seed*.json")):
            m = xgb.Booster()
            m.load_model(str(f))
            models.append(m)
        return models

    def predict(self, **kwargs) -> float:
        """Predict SOC using direction-specific model if available."""
        if not self._has_directional:
            return self._global.predict(**kwargs)

        # Determine direction from station ordering
        dep = self._global.stations[kwargs["departure_station"]]
        arr = self._global.stations[kwargs["arrival_station"]]
        ensemble = self._downstream if arr["order"] > dep["order"] else self._upstream

        features = self._build_features(**kwargs)
        values = [features[f] for f in self._dir_features]
        dm = xgb.DMatrix(
            np.array([values], dtype=np.float32),
            feature_names=self._dir_features,
        )
        preds = [float(m.predict(dm)[0]) for m in ensemble]
        return max(sum(preds) / len(preds), 0.0)

    def _build_features(self, **kwargs) -> dict:
        """Build the full feature dict (reuses SOCTripPredictor logic)."""
        g = self._global
        dep = g.stations[kwargs["departure_station"]]
        arr = g.stations[kwargs["arrival_station"]]
        dep_order, arr_order = dep["order"], arr["order"]
        hop_count = abs(arr_order - dep_order)
        direction = 0 if arr_order > dep_order else 1

        route_heading = g._bearing(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        route_distance = g._haversine_km(dep["lat"], dep["lon"], arr["lat"], arr["lon"])

        wind_speed = kwargs.get("wind_speed_kn", 5.0)
        wind_dir = kwargs.get("wind_direction_deg", 180.0)
        wind_comp = wind_speed * math.cos(math.radians(wind_dir - route_heading))
        wind_cross = wind_speed * math.sin(math.radians(wind_dir - route_heading))

        hour = kwargs.get("hour", 10)
        current_soc = kwargs["current_soc"]
        target_speed = kwargs["target_speed_kn"]
        passengers = kwargs["passengers"]

        # v4: ocean current & wave
        oc_vel = kwargs.get("ocean_current_velocity_ms", 0.0)
        oc_dir = kwargs.get("ocean_current_direction_deg", 0.0)
        current_comp = oc_vel * math.cos(math.radians(oc_dir - route_heading))
        wave_h = kwargs.get("wave_height_m", 0.1)
        wave_d = kwargs.get("wave_direction_deg", 0.0)
        wave_comp = wave_h * math.cos(math.radians(wave_d - route_heading))
        wave_period = kwargs.get("wave_period_s", 0.0)

        departure_time = kwargs.get("departure_time")
        if departure_time is not None:
            tide = compute_tide_features(departure_time)
        else:
            from datetime import datetime as dt
            tide = compute_tide_features(dt.now().replace(hour=hour, minute=0, second=0))

        return {
            "departure_station_encoded": dep_order,
            "arrival_station_encoded": arr_order,
            "hop_count": hop_count,
            "direction_encoded": direction,
            "route_distance_km": route_distance,
            "start_soc": current_soc,
            "start_hv_capacity": kwargs["hv_capacity"],
            "target_speed": target_speed,
            "target_speed_squared": target_speed ** 2,
            "passengers_on_board": passengers,
            "passenger_load_ratio": passengers / 40.0,
            "temperature_c": kwargs.get("temperature_c", 28.0),
            "relative_humidity": kwargs.get("humidity", 75.0),
            "wind_speed_kn": wind_speed,
            "wind_direction_deg": wind_dir,
            "wind_component_along_route": wind_comp,
            "precipitation_mm": kwargs.get("precipitation_mm", 0.0),
            "wave_height_m": wave_h,
            "hour_of_day": hour,
            "is_peak_hour": int(7 <= hour <= 9 or 16 <= hour <= 19),
            "speed_distance_interaction": target_speed * route_distance,
            "energy_rate_proxy": (target_speed ** 2) * hop_count,
            "wind_cross_component": wind_cross,
            "soc_per_km_capacity": current_soc / (route_distance + 0.01),
            "tide_height_m": tide["tide_height_m"],
            "hours_since_high_tide": tide["hours_since_high_tide"],
            "tide_phase": tide["tide_phase"],
            # v4 ocean current & wave features
            "current_component_along_route": current_comp,
            "current_velocity_ms": oc_vel,
            "wave_component_along_route": wave_comp,
            "wave_period_s": wave_period,
        }
