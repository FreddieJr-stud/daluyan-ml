"""RPi5 Inference Smoke Test — verifies all three models load and predict correctly.

Run from inside the rpi5_bundle directory:
    cd rpi5_bundle
    python test_inference.py

Tests:
  1. Model loading (all three models)
  2. Trip prediction (Model 1A) — multiple routes, directions, uncertainty intervals
  3. Realtime prediction (Model 1B) — simulated 1Hz telemetry stream
  4. Anomaly detection (Model 2) — normal + injected fault
  5. Inference timing benchmarks
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# Resolve paths relative to this script
BUNDLE_DIR = Path(__file__).resolve().parent
MODELS_DIR = str(BUNDLE_DIR / "models")
CONFIG_DIR = str(BUNDLE_DIR / "config")

sys.path.insert(0, str(BUNDLE_DIR / "inference"))

from inference_soc_trip import SOCTripPredictor
from inference_soc_realtime import SOCRealtimePredictor
from inference_anomaly import MotorAnomalyDetector

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    mark = "+" if passed else "X"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))


# ── 1. Model Loading ──────────────────────────────────────────────────

def test_model_loading():
    print("\n1. Model Loading")
    print("-" * 50)

    try:
        trip = SOCTripPredictor(MODELS_DIR, CONFIG_DIR)
        n_trip = len(trip.models)
        has_quantile = trip.q10_model is not None
        has_conformal = trip.conformal_q_hat is not None
        record("Trip model (1A)", True,
               f"{n_trip} ensemble, quantile={has_quantile}, conformal={has_conformal}")
    except Exception as e:
        record("Trip model (1A)", False, str(e))
        trip = None

    try:
        rt = SOCRealtimePredictor(MODELS_DIR, MODELS_DIR, CONFIG_DIR)
        n_rt = len(rt.rt_models)
        record("Realtime model (1B)", True, f"{n_rt} ensemble")
    except Exception as e:
        record("Realtime model (1B)", False, str(e))
        rt = None

    try:
        anom = MotorAnomalyDetector(CONFIG_DIR, MODELS_DIR)
        n_targets = len(anom.models)
        record("Anomaly model (2)", True,
               f"{n_targets} reconstruction targets, p99={anom.threshold_p99:.4f}")
    except Exception as e:
        record("Anomaly model (2)", False, str(e))
        anom = None

    return trip, rt, anom


# ── 2. Trip Prediction ────────────────────────────────────────────────

def test_trip_prediction(trip: SOCTripPredictor):
    print("\n2. Trip Prediction (Model 1A)")
    print("-" * 50)

    if trip is None:
        record("Trip prediction", False, "model not loaded")
        return

    test_cases = [
        # (dep, arr, soc, hv_cap, speed, pax, label, expected_range)
        ("Guadalupe", "Escolta", 85.0, 136.0, 5.0, 20, "downstream 7-hop", (2.0, 25.0)),
        ("Escolta", "Guadalupe", 85.0, 136.0, 5.0, 20, "upstream 7-hop", (5.0, 40.0)),
        ("Guadalupe", "Hulo", 90.0, 144.0, 4.5, 10, "short 1-hop", (0.1, 10.0)),
        ("Napindan", "Escolta", 95.0, 152.0, 5.5, 30, "full route downstream", (5.0, 35.0)),
    ]

    now = datetime.now()

    for dep, arr, soc, hv, spd, pax, label, (lo, hi) in test_cases:
        try:
            pred = trip.predict(
                departure_station=dep,
                arrival_station=arr,
                current_soc=soc,
                hv_capacity=hv,
                target_speed_kn=spd,
                passengers=pax,
                departure_time=now,
            )
            in_range = lo <= pred <= hi
            record(f"Trip: {label}", in_range,
                   f"predicted={pred:.2f}% SOC (expected {lo}-{hi}%)")
        except Exception as e:
            record(f"Trip: {label}", False, str(e))

    # Test uncertainty interval
    try:
        interval = trip.predict_interval(
            method="conformal",
            departure_station="Guadalupe",
            arrival_station="Escolta",
            current_soc=85.0,
            hv_capacity=136.0,
            target_speed_kn=5.0,
            passengers=20,
            departure_time=now,
        )
        has_bounds = "lower" in interval and "upper" in interval
        pt = interval.get("point", "?")
        lo = interval.get("lower", "?")
        hi = interval.get("upper", "?")
        record("Uncertainty interval", has_bounds,
               f"point={pt}, [{lo}, {hi}], method={interval.get('method', '?')}")
    except Exception as e:
        record("Uncertainty interval", False, str(e))


# ── 3. Realtime Prediction ────────────────────────────────────────────

def test_realtime_prediction(rt: SOCRealtimePredictor):
    print("\n3. Realtime Prediction (Model 1B)")
    print("-" * 50)

    if rt is None:
        record("Realtime prediction", False, "model not loaded")
        return

    try:
        rt.start_trip("Guadalupe", "Escolta", passengers=20)
        record("start_trip()", True, "Guadalupe -> Escolta")
    except Exception as e:
        record("start_trip()", False, str(e))
        return

    # Simulate 119 warmup ticks (need 120s of data for rolling features;
    # tick 120 is the first that returns a prediction)
    warmup_result = None
    for i in range(119):
        warmup_result = rt.update({
            "soc_percent": 85.0 - i * 0.04,
            "speed_over_ground": 5.0,
            "motor_power_combined": 30.0,
            "battery_power": 35.0,
            "heading": 270.0,
            "port_rpm": 400,
            "stbd_rpm": 400,
            "trip_nm": i * 0.0025,
        })

    record("Warmup (119 ticks)", warmup_result is None,
           "correctly returns None during warmup" if warmup_result is None else "unexpected output")

    # First real prediction at tick 121
    result = rt.update({
        "soc_percent": 80.2,
        "speed_over_ground": 5.2,
        "motor_power_combined": 32.0,
        "battery_power": 36.0,
        "heading": 270.0,
        "port_rpm": 410,
        "stbd_rpm": 405,
        "trip_nm": 0.30,
    })

    if result is not None:
        arrival_soc = result.get("predicted_arrival_soc", -1)
        soc_delta = result.get("soc_remaining_delta", -1)
        reachable = result.get("reachable_stations", [])
        reasonable = 50 < arrival_soc < 85 and soc_delta > 0
        record("First prediction", reasonable,
               f"arrival_soc={arrival_soc:.1f}%, delta={soc_delta:.2f}%, "
               f"reachable={len(reachable)} stations")
    else:
        record("First prediction", False, "returned None after warmup")

    # Simulate further into the trip (75% progress)
    for i in range(200):
        rt.update({
            "soc_percent": 80.0 - i * 0.02,
            "speed_over_ground": 5.0,
            "motor_power_combined": 30.0,
            "battery_power": 35.0,
            "heading": 270.0,
            "port_rpm": 400,
            "stbd_rpm": 400,
            "trip_nm": 0.30 + i * 0.003,
        })

    late_result = rt.update({
        "soc_percent": 76.0,
        "speed_over_ground": 5.0,
        "motor_power_combined": 30.0,
        "battery_power": 35.0,
        "heading": 270.0,
        "port_rpm": 400,
        "stbd_rpm": 400,
        "trip_nm": 0.90,
    })

    if late_result is not None:
        late_delta = late_result.get("soc_remaining_delta", -1)
        # At 75%+ progress, remaining delta should be small
        record("Late-trip prediction", 0 < late_delta < 15,
               f"remaining_delta={late_delta:.2f}% SOC at ~75% progress")
    else:
        record("Late-trip prediction", False, "returned None")


# ── 4. Anomaly Detection ──────────────────────────────────────────────

def test_anomaly_detection(anom: MotorAnomalyDetector):
    print("\n4. Anomaly Detection (Model 2)")
    print("-" * 50)

    if anom is None:
        record("Anomaly detection", False, "model not loaded")
        return

    # Normal telemetry — values from training distribution means
    normal = {
        "port_motor_power": 18.5,
        "stbd_motor_power": 20.1,
        "motor_power_combined": 38.9,
        "port_rpm": 1180,
        "stbd_rpm": 1138,
        "speed_over_ground": 5.76,
        "battery_power": 159.4,
    }

    try:
        result = anom.check(normal)
        is_normal = result["level"] == "normal"
        record("Normal telemetry", is_normal,
               f"score={result['score']:.4f}, level={result['level']}")
    except Exception as e:
        record("Normal telemetry", False, str(e))

    # Injected fault: port motor at 1.5x power (asymmetric)
    fault = dict(normal)
    fault["port_motor_power"] = 27.75  # 1.5x normal (18.5 * 1.5)
    fault["motor_power_combined"] = 47.85  # 27.75 + 20.1

    try:
        result = anom.check(fault)
        detected = result["level"] in ("warning", "anomalous")
        record("Injected fault (port x1.5)", detected,
               f"score={result['score']:.4f}, level={result['level']}")
    except Exception as e:
        record("Injected fault", False, str(e))

    # Idle ferry — should return 'idle', not flag
    idle = dict(normal)
    idle["speed_over_ground"] = 0.1

    try:
        result = anom.check(idle)
        is_idle = result["level"] == "idle"
        record("Idle ferry", is_idle,
               f"level={result['level']}")
    except Exception as e:
        record("Idle ferry", False, str(e))


# ── 5. Timing Benchmarks ──────────────────────────────────────────────

def test_timing(trip, rt, anom):
    print("\n5. Inference Timing (100 iterations each)")
    print("-" * 50)

    now = datetime.now()
    n_iters = 100

    # Trip timing
    if trip is not None:
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            trip.predict(
                departure_station="Guadalupe", arrival_station="Escolta",
                current_soc=85.0, hv_capacity=136.0,
                target_speed_kn=5.0, passengers=20, departure_time=now,
            )
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        record(f"Trip (1A)", avg < 50,
               f"avg={avg:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Realtime timing (must have warmed up buffer already)
    if rt is not None:
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            rt.update({
                "soc_percent": 78.0,
                "speed_over_ground": 5.0,
                "motor_power_combined": 30.0,
                "battery_power": 35.0,
                "heading": 270.0,
                "port_rpm": 400, "stbd_rpm": 400,
                "trip_nm": 0.5,
            })
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        record(f"Realtime (1B)", avg < 20,
               f"avg={avg:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Anomaly timing
    if anom is not None:
        telemetry = {
            "port_motor_power": 18.5, "stbd_motor_power": 20.1,
            "motor_power_combined": 38.9, "port_rpm": 1180, "stbd_rpm": 1138,
            "speed_over_ground": 5.76, "battery_power": 159.4,
        }
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            anom.check(telemetry)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        record(f"Anomaly (2)", avg < 20,
               f"avg={avg:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  M/B Dalaray Digital Shadow — RPi5 Inference Smoke Test")
    print(f"  Bundle: {BUNDLE_DIR}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    trip, rt, anom = test_model_loading()
    test_trip_prediction(trip)
    test_realtime_prediction(rt)
    test_anomaly_detection(anom)
    test_timing(trip, rt, anom)

    # Summary
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"  Results: {n_pass}/{total} passed", end="")
    if n_fail > 0:
        print(f", {n_fail} FAILED:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    X {name}: {detail}")
    else:
        print(" — all good!")
    print("=" * 60)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
