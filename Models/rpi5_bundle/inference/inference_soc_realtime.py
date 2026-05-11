"""Standalone real-time SOC range estimator for RPi5 deployment.

Maintains a rolling buffer and predicts remaining SOC delta + reachable
stations at each 1Hz telemetry tick.  Supports v3 ensemble inference
with tide/current features and v4 conformal prediction intervals.

Dependencies: xgboost, numpy only (tide prediction is built-in).
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import xgboost as xgb

# Allow importing tide module
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from features.tide import compute_tide_features, predict_tide_height
except ImportError:
    def compute_tide_features(dt):
        return {"tide_height_m": 0.0, "hours_since_high_tide": 6.0, "tide_phase": 0}
    def predict_tide_height(dt):
        return 0.0


EPS = 0.01
SAFETY_SOC = 15.0  # minimum SOC before flagging range limit


class SOCRealtimePredictor:
    """Real-time SOC predictor with rolling state buffer."""

    def __init__(
        self,
        realtime_model_dir: str,
        trip_model_dir: str,
        config_dir: str,
    ):
        config = Path(config_dir)

        with open(config / "soc_realtime_metadata.json") as f:
            rt_meta = json.load(f)
        self.rt_features = rt_meta["features"]

        with open(config / "soc_trip_metadata.json") as f:
            trip_meta = json.load(f)
        self.trip_features = trip_meta["features"]

        # Load ensemble realtime models
        rt_dir = Path(realtime_model_dir)
        self.rt_models: list[xgb.Booster] = []
        primary = rt_dir / "soc_realtime_model.json"
        m = xgb.Booster()
        m.load_model(str(primary))
        self.rt_models.append(m)
        for f in sorted(rt_dir.glob("soc_realtime_model_seed*.json")):
            m = xgb.Booster()
            m.load_model(str(f))
            self.rt_models.append(m)

        # Load ensemble trip models
        trip_dir = Path(trip_model_dir)
        self.trip_models: list[xgb.Booster] = []
        primary = trip_dir / "soc_trip_model.json"
        m = xgb.Booster()
        m.load_model(str(primary))
        self.trip_models.append(m)
        for f in sorted(trip_dir.glob("soc_trip_model_seed*.json")):
            m = xgb.Booster()
            m.load_model(str(f))
            self.trip_models.append(m)

        # Load conformal calibration (v4 UQ)
        self.conformal_q_hat: float | None = None
        conformal_path = config / "conformal_realtime.json"
        if conformal_path.exists():
            with open(conformal_path) as f:
                conf = json.load(f)
            self.conformal_q_hat = conf.get("conformal_q_hat_80")

        # Station data
        self.stations = [
            {"name": "Napindan",   "order": 0, "lat": 14.55713, "lon": 121.06836},
            {"name": "Guadalupe",  "order": 1, "lat": 14.56811, "lon": 121.04792},
            {"name": "Hulo",       "order": 2, "lat": 14.56794, "lon": 121.03364},
            {"name": "Valenzuela", "order": 3, "lat": 14.57400, "lon": 121.02578},
            {"name": "Lambingan",  "order": 4, "lat": 14.58731, "lon": 121.01847},
            {"name": "Sta Ana",    "order": 5, "lat": 14.58236, "lon": 121.01167},
            {"name": "PUP",        "order": 6, "lat": 14.59603, "lon": 121.01072},
            {"name": "Quinta",     "order": 7, "lat": 14.59572, "lon": 120.98142},
            {"name": "Escolta",    "order": 8, "lat": 14.59653, "lon": 120.97761},
        ]
        self.station_by_name = {s["name"]: s for s in self.stations}

        # Rolling buffers (120 seconds max)
        self._buffer_size = 120
        self._soc_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._power_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._speed_buf: deque[float] = deque(maxlen=self._buffer_size)
        self._rpm_buf: deque[float] = deque(maxlen=self._buffer_size)  # v3
        self._tick_count = 0

        # v3: Cached tide features (refreshed every 60 ticks)
        self._tide_cache: dict | None = None
        self._tide_cache_tick = -999

        # Trip context (set once per trip via start_trip)
        self._trip_ctx: dict | None = None

    def start_trip(
        self,
        departure_station: str,
        arrival_station: str,
        passengers: int = 15,
        temperature_c: float = 28.0,
        wind_speed_kn: float = 5.0,
        wind_direction_deg: float = 180.0,
        wave_height_m: float = 0.1,
        total_distance_km: float | None = None,
        # v4: ocean current & wave params (used by trip model only)
        ocean_current_velocity_ms: float = 0.0,
        ocean_current_direction_deg: float = 0.0,
        wave_direction_deg: float = 0.0,
        wave_period_s: float = 0.0,
    ) -> None:
        """Call once at trip departure to set route context."""
        dep = self.station_by_name[departure_station]
        arr = self.station_by_name[arrival_station]

        route_heading = self._bearing(dep["lat"], dep["lon"], arr["lat"], arr["lon"])
        if total_distance_km is None:
            total_distance_km = self._haversine_km(dep["lat"], dep["lon"], arr["lat"], arr["lon"])

        self._trip_ctx = {
            "departure_station": departure_station,
            "arrival_station": arrival_station,
            "departure_station_encoded": dep["order"],
            "arrival_station_encoded": arr["order"],
            "direction_encoded": 0 if arr["order"] > dep["order"] else 1,
            "hop_count": abs(arr["order"] - dep["order"]),
            "route_heading": route_heading,
            "total_distance_km": total_distance_km,
            "passengers_on_board": float(passengers),
            "temperature_c": temperature_c,
            "wind_speed_kn": wind_speed_kn,
            "wind_direction_deg": wind_direction_deg,
            "wind_component_along_route": wind_speed_kn * math.cos(
                math.radians(wind_direction_deg - route_heading)
            ),
            "wave_height_m": wave_height_m,
            # v4: ocean current & wave features (used by trip model 1A only,
            # NOT by realtime model 1B which uses telemetry rpm_per_speed instead)
            "current_component_along_route": ocean_current_velocity_ms * math.cos(
                math.radians(ocean_current_direction_deg - route_heading)
            ),
            "current_velocity_ms": ocean_current_velocity_ms,
            "ocean_current_direction_deg": ocean_current_direction_deg,
            "wave_component_along_route": wave_height_m * math.cos(
                math.radians(wave_direction_deg - route_heading)
            ),
            "wave_direction_deg": wave_direction_deg,
            "wave_period_s": wave_period_s,
        }

        # Reset buffers
        self._soc_buf.clear()
        self._power_buf.clear()
        self._speed_buf.clear()
        self._rpm_buf.clear()
        self._tick_count = 0
        self._tide_cache = None
        self._tide_cache_tick = -999

        # v6: Trip-so-far tracking (set on first update tick)
        self._start_soc: float | None = None
        self._start_nm: float | None = None
        self._cumulative_power: float = 0.0

    def update(self, telemetry: dict, current_time: "datetime | None" = None) -> dict | None:
        """Process one 1Hz telemetry reading.

        Args:
            telemetry: dict with keys soc_percent, speed_over_ground,
                       motor_power_combined, battery_power, heading,
                       port_rpm, stbd_rpm, trip_nm (nautical miles odometer)
            current_time: optional datetime for tide feature computation

        Returns:
            dict with predicted_arrival_soc, soc_remaining_delta,
            arrival_soc_lower, arrival_soc_upper, ensemble_std,
            uq_method, reachable_stations, or None if not enough warmup data.
        """
        if self._trip_ctx is None:
            return None

        # Push to buffers
        soc = telemetry["soc_percent"]
        power = telemetry["motor_power_combined"]
        speed = telemetry["speed_over_ground"]
        avg_rpm = (telemetry.get("port_rpm", 0) + telemetry.get("stbd_rpm", 0)) / 2.0

        self._soc_buf.append(soc)
        self._power_buf.append(power)
        self._speed_buf.append(speed)
        self._rpm_buf.append(avg_rpm)
        self._tick_count += 1

        # v6: Track trip-so-far state
        if self._start_soc is None:
            self._start_soc = soc
            self._start_nm = telemetry.get("trip_nm", 0)
        self._cumulative_power += power

        # Need at least 30 ticks for core rolling features (soc_rate_30s).
        # Features needing longer windows (60s/120s) degrade gracefully.
        if self._tick_count < 30:
            return None

        ctx = self._trip_ctx
        soc_arr = np.array(self._soc_buf)
        power_arr = np.array(self._power_buf)
        speed_arr = np.array(self._speed_buf)
        n = len(soc_arr)

        # Trip progress (distance relative to trip start)
        trip_nm_raw = telemetry.get("trip_nm", 0)
        trip_km = (trip_nm_raw - (self._start_nm or 0)) * 1.852
        total_km = max(ctx["total_distance_km"], EPS)
        progress = min(trip_km / total_km, 1.0)
        remaining = max(total_km - trip_km, 0.0)

        # Rolling SOC rates with graceful degradation
        _rate_30 = float((soc_arr[-1] - soc_arr[-30]) / 0.5) if n >= 30 else 0.0
        _rate_60 = float((soc_arr[-1] - soc_arr[-60]) / 1.0) if n >= 60 else _rate_30
        _rate_120 = float((soc_arr[-1] - soc_arr[-120]) / 2.0) if n >= 120 else _rate_60

        # v2 stability features (use available window, min 10 ticks)
        soc_diffs = np.diff(soc_arr[-min(61, n):]) if n >= 2 else np.array([0.0])
        soc_rate_std = float(np.std(soc_diffs)) if len(soc_diffs) > 1 else 0.0
        power_std = float(np.std(power_arr[-min(60, n):])) if n >= 10 else 0.0
        speed_accel = float(speed_arr[-1] - speed_arr[-min(30, n)]) if n >= 2 else 0.0

        features = {
            "current_soc": soc,
            "current_speed": speed,
            "current_motor_power": power,
            "current_battery_power": telemetry.get("battery_power", 0),
            "current_heading": telemetry.get("heading", 0),
            "speed_squared": speed ** 2,
            "distance_remaining_km": remaining,
            "trip_progress_fraction": progress,
            "elapsed_time_s": float(self._tick_count),
            # Rolling windows (graceful degradation for 60s/120s)
            "soc_rate_30s": _rate_30,
            "soc_rate_60s": _rate_60,
            "soc_rate_120s": _rate_120,
            "avg_power_30s": float(power_arr[-30:].mean()),
            "avg_power_60s": float(power_arr[-min(60, n):].mean()),
            "avg_speed_30s": float(speed_arr[-30:].mean()),
            "avg_speed_60s": float(speed_arr[-min(60, n):].mean()),
            # Route context
            "departure_station_encoded": ctx["departure_station_encoded"],
            "arrival_station_encoded": ctx["arrival_station_encoded"],
            "direction_encoded": ctx["direction_encoded"],
            "hop_count": ctx["hop_count"],
            # External
            "passengers_on_board": ctx["passengers_on_board"],
            "temperature_c": ctx["temperature_c"],
            "wind_speed_kn": ctx["wind_speed_kn"],
            "wind_component_along_route": ctx["wind_component_along_route"],
            "wave_height_m": ctx["wave_height_m"],
            # Physics
            "power_speed_ratio": power / (speed + EPS),
            # v2 stability features
            "soc_rate_std_60s": soc_rate_std,
            "power_std_60s": power_std,
            "speed_acceleration_30s": speed_accel,
            # v3 current proxy
            "rpm_per_speed": avg_rpm / (speed + EPS),
            # v3 tide features (cached, refreshed every 60 ticks)
            **self._get_tide_features(current_time),
            # Note: v4 ocean current/wave features excluded from realtime model —
            # rpm_per_speed telemetry proxy captures current effects better than
            # hourly weather data
            # v6: Trip-so-far consumption features
            "soc_consumed_so_far": max(self._start_soc - soc, 0.0),
            "empirical_soc_per_km": max(self._start_soc - soc, 0.0) / max(trip_km, EPS),
            "empirical_power_per_km": self._cumulative_power / max(trip_km, EPS),
        }

        values = [features[f] for f in self.rt_features]
        dm = xgb.DMatrix(
            np.array([values], dtype=np.float32),
            feature_names=self.rt_features,
        )

        # Ensemble: average predictions from all realtime models
        preds = [float(m.predict(dm)[0]) for m in self.rt_models]
        soc_remaining = max(sum(preds) / len(preds), 0.0)
        ensemble_std = float(np.std(preds))
        predicted_arrival = soc - soc_remaining

        # Uncertainty bounds (conformal prediction)
        if self.conformal_q_hat is not None:
            q = self.conformal_q_hat
            lower_arrival = round(max(predicted_arrival - q, 0.0), 2)
            upper_arrival = round(predicted_arrival + q, 2)
        else:
            # Fallback: ±2 ensemble std
            lower_arrival = round(max(predicted_arrival - 2 * ensemble_std, 0.0), 2)
            upper_arrival = round(predicted_arrival + 2 * ensemble_std, 2)

        # Estimate range beyond current trip (use conservative lower bound)
        reachable = self._estimate_range(
            predicted_arrival,
            ctx["arrival_station"],
            ctx["direction_encoded"],
            ctx,
        )

        return {
            "predicted_arrival_soc": round(predicted_arrival, 2),
            "soc_remaining_delta": round(soc_remaining, 2),
            "arrival_soc_lower": lower_arrival,
            "arrival_soc_upper": upper_arrival,
            "ensemble_std": round(ensemble_std, 4),
            "uq_method": "conformal" if self.conformal_q_hat is not None else "ensemble_std",
            "reachable_stations": reachable,
        }

    def _get_tide_features(self, current_time) -> dict:
        """Get tide features, cached for 60 ticks (1 minute) to save compute."""
        if (self._tide_cache is None
                or self._tick_count - self._tide_cache_tick >= 60):
            if current_time is not None:
                self._tide_cache = compute_tide_features(current_time)
            else:
                # Fallback: use current system time
                from datetime import datetime
                self._tide_cache = compute_tide_features(datetime.now())
            self._tide_cache_tick = self._tick_count
        return {
            "tide_height_m": self._tide_cache["tide_height_m"],
            "hours_since_high_tide": self._tide_cache["hours_since_high_tide"],
            "tide_phase": float(self._tide_cache["tide_phase"]),
        }

    def _estimate_range(
        self,
        starting_soc: float,
        from_station: str,
        direction: int,
        ctx: dict,
    ) -> list[dict]:
        """Use trip model ensemble to estimate reachable stations beyond current trip."""
        reachable = []
        running_soc = starting_soc
        current = self.station_by_name[from_station]

        # Get stations in travel direction order
        if direction == 0:  # downstream (increasing order)
            candidates = [s for s in self.stations if s["order"] > current["order"]]
            candidates.sort(key=lambda s: s["order"])
        else:  # upstream (decreasing order)
            candidates = [s for s in self.stations if s["order"] < current["order"]]
            candidates.sort(key=lambda s: -s["order"])

        for next_stn in candidates:
            # Build trip features for this leg
            dep_order = current["order"]
            arr_order = next_stn["order"]
            hop = abs(arr_order - dep_order)
            dist = self._haversine_km(current["lat"], current["lon"],
                                      next_stn["lat"], next_stn["lon"])
            heading = self._bearing(current["lat"], current["lon"],
                                    next_stn["lat"], next_stn["lon"])
            wind_speed = ctx["wind_speed_kn"]
            wind_dir = ctx.get("wind_direction_deg", 180)
            wind_comp = wind_speed * math.cos(math.radians(wind_dir - heading))
            wind_cross = wind_speed * math.sin(math.radians(wind_dir - heading))
            speed = 5.0  # default cruising speed

            trip_feat = {
                "departure_station_encoded": dep_order,
                "arrival_station_encoded": arr_order,
                "hop_count": hop,
                "direction_encoded": direction,
                "route_distance_km": dist,
                "start_soc": running_soc,
                "start_hv_capacity": running_soc * 1.6,  # approx
                "target_speed": speed,
                "target_speed_squared": speed ** 2,
                "passengers_on_board": ctx["passengers_on_board"],
                "passenger_load_ratio": ctx["passengers_on_board"] / 40.0,
                "temperature_c": ctx["temperature_c"],
                "relative_humidity": 75.0,
                "wind_speed_kn": wind_speed,
                "wind_direction_deg": float(wind_dir),
                "wind_component_along_route": wind_comp,
                "precipitation_mm": 0.0,
                "wave_height_m": ctx["wave_height_m"],
                "hour_of_day": 10,
                "is_peak_hour": 0,
                # v2 interaction features
                "speed_distance_interaction": speed * dist,
                "energy_rate_proxy": (speed ** 2) * hop,
                "wind_cross_component": wind_cross,
                "soc_per_km_capacity": running_soc / (dist + 0.01),
                # v3 tide features (use current cached values)
                "tide_height_m": self._tide_cache["tide_height_m"] if self._tide_cache else 0.0,
                "hours_since_high_tide": self._tide_cache["hours_since_high_tide"] if self._tide_cache else 6.0,
                "tide_phase": float(self._tide_cache["tide_phase"]) if self._tide_cache else 0.0,
                # v4 ocean current & wave features (re-project along this leg's heading)
                "current_component_along_route": ctx.get("current_velocity_ms", 0.0) * math.cos(
                    math.radians(ctx.get("ocean_current_direction_deg", 0.0) - heading)
                ),
                "current_velocity_ms": ctx.get("current_velocity_ms", 0.0),
                "wave_component_along_route": ctx["wave_height_m"] * math.cos(
                    math.radians(ctx.get("wave_direction_deg", 0.0) - heading)
                ),
                "wave_period_s": ctx.get("wave_period_s", 0.0),
            }

            values = [trip_feat[f] for f in self.trip_features]
            dm = xgb.DMatrix(
                np.array([values], dtype=np.float32),
                feature_names=self.trip_features,
            )

            # Ensemble: average predictions from all trip models
            preds = [float(m.predict(dm)[0]) for m in self.trip_models]
            delta = max(sum(preds) / len(preds), 0.0)
            running_soc -= delta

            reachable.append({
                "station": next_stn["name"],
                "predicted_soc": round(running_soc, 2),
                "reachable": running_soc >= SAFETY_SOC,
            })

            if running_soc < SAFETY_SOC:
                break
            current = next_stn

        return reachable

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2):
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlon = lon2_r - lon1_r
        x = math.sin(dlon) * math.cos(lat2_r)
        y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
        return math.degrees(math.atan2(x, y)) % 360

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
        dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))
