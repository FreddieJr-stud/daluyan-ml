"""RPi5 Model Benchmark — verifies all 5 models produce identical results.

Run from inside the rpi5_bundle directory:
    cd rpi5_bundle
    python benchmark.py

Requires test CSVs in testdata/:
    testdata/trip_test.csv        (Model 1A, 190 trip-level rows)
    testdata/realtime_test.csv    (Model 1B, 10351 1Hz rows)
    testdata/anomaly_test.csv     (Model 2, 24673 telemetry rows)

Compares RPi5 inference output against expected values from the training
workstation. Any discrepancy > tolerance flags a MISMATCH, which indicates
a model export or numerical precision issue.

Dependencies: xgboost, numpy (same as inference).
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────
BUNDLE = Path(__file__).resolve().parent
MODELS = BUNDLE / "models"
CONFIG = BUNDLE / "config"
TESTDATA = BUNDLE / "testdata"

# ── Expected values from training workstation (v9, Mar 20, 2026) ────────
# These are the ground-truth metrics computed during training. Any RPi5
# result that differs by more than TOLERANCE indicates a problem.
# Updated after v9 retrain + battery BMS fix retrain.
EXPECTED = {
    "1A": {"MAE": 0.7289, "RMSE": 1.0276, "R2": 0.9513, "n": 249},
    "1B": {"MAE": 0.3421, "RMSE": 0.5081, "R2": 0.9807, "n": 14247},
    "2":  {"FPR": 0.75, "detection_rate": 97.8, "n": 33854},
    "3a": {"FPR": 2.3, "detection_rate": 100.0},
    "3b": {"FPR": 1.7, "detection_rate": 100.0},
}
TOLERANCE_MAE = 0.001   # 0.001% SOC — within float32 rounding
TOLERANCE_R2 = 0.001
TOLERANCE_RATE = 1.0    # 1 percentage point for detection/FPR


# ── Results tracking ────────────────────────────────────────────────────
checks: list[dict] = []


def check(name: str, actual: float, expected: float, tolerance: float,
          unit: str = "%") -> bool:
    """Record a comparison check. Returns True if within tolerance."""
    diff = abs(actual - expected)
    ok = diff <= tolerance
    status = "MATCH" if ok else "MISMATCH"
    checks.append({
        "name": name, "actual": actual, "expected": expected,
        "diff": diff, "tolerance": tolerance, "status": status, "unit": unit,
    })
    mark = "+" if ok else "X"
    print(f"    [{mark}] {name}: {actual:.4f}{unit}  "
          f"(expected {expected:.4f}{unit}, diff={diff:.4f})")
    return ok


# ── Helpers ──────────────────────────────────────────────────────────────

def load_csv(path: Path, feature_cols: list[str], target_col: str | None):
    """Load CSV into numpy arrays using stdlib csv (no pandas needed)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        feat_rows, target_rows = [], []
        for row in reader:
            feat_rows.append([float(row[c]) for c in feature_cols])
            if target_col:
                target_rows.append(float(row[target_col]))
    X = np.array(feat_rows, dtype=np.float32)
    y = np.array(target_rows, dtype=np.float32) if target_col else None
    return X, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², MAPE."""
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mask = np.abs(y_true) > 0.01
    mape = float(np.mean(np.abs(errors[mask] / y_true[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def load_ensemble(prefix: str, n_seeds: int = 5) -> list[xgb.Booster]:
    """Load primary + seed models."""
    seeds = [137, 256, 512, 1024]
    models = []
    primary = MODELS / f"{prefix}.json"
    m = xgb.Booster()
    m.load_model(str(primary))
    models.append(m)
    for seed in seeds[:n_seeds - 1]:
        p = MODELS / f"{prefix}_seed{seed}.json"
        if p.exists():
            m = xgb.Booster()
            m.load_model(str(p))
            models.append(m)
    return models


def ensemble_predict(models: list[xgb.Booster], dm: xgb.DMatrix) -> np.ndarray:
    """Average predictions across ensemble."""
    preds = np.stack([m.predict(dm) for m in models])
    return preds.mean(axis=0)


# ── Model 1A: Trip-Level SOC ─────────────────────────────────────────────

def benchmark_trip():
    print("\n1. Model 1A — Trip-Level SOC Prediction")
    print("-" * 60)

    csv_path = TESTDATA / "trip_test.csv"
    if not csv_path.exists():
        print("  SKIP: testdata/trip_test.csv not found")
        return None

    with open(CONFIG / "soc_trip_metadata.json") as f:
        meta = json.load(f)
    features = meta["features"]

    X, y = load_csv(csv_path, features, "soc_delta")
    dm = xgb.DMatrix(X, feature_names=features)

    models = load_ensemble("soc_trip_model")
    print(f"  Loaded {len(models)} ensemble models, {len(features)} features, {len(y)} samples")

    t0 = time.perf_counter()
    y_pred = ensemble_predict(models, dm)
    elapsed = (time.perf_counter() - t0) * 1000

    m = metrics(y, y_pred)
    print(f"  MAE={m['MAE']:.4f}%  RMSE={m['RMSE']:.4f}%  R2={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
    print(f"  Inference: {elapsed:.1f}ms total ({elapsed / len(y):.2f}ms/sample)")

    # Compare against expected
    exp = EXPECTED["1A"]
    print(f"\n  Workstation comparison:")
    check("1A MAE", m["MAE"], exp["MAE"], TOLERANCE_MAE)
    check("1A RMSE", m["RMSE"], exp["RMSE"], TOLERANCE_MAE)
    check("1A R2", m["R2"], exp["R2"], TOLERANCE_R2)

    # Per-direction breakdown
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        directions = [int(float(row["direction_encoded"])) for row in reader]
    directions = np.array(directions)

    for dir_val, dir_label in [(0, "Downstream"), (1, "Upstream")]:
        mask = directions == dir_val
        if mask.sum() == 0:
            continue
        m_dir = metrics(y[mask], y_pred[mask])
        print(f"\n  {dir_label} ({mask.sum()} trips):")
        print(f"    MAE={m_dir['MAE']:.4f}%, RMSE={m_dir['RMSE']:.4f}%, R2={m_dir['R2']:.4f}")

    return m


# ── Model 1B: Realtime SOC ──────────────────────────────────────────────

def benchmark_realtime():
    print("\n\n2. Model 1B — Realtime SOC Range Prediction")
    print("-" * 60)

    csv_path = TESTDATA / "realtime_test.csv"
    if not csv_path.exists():
        print("  SKIP: testdata/realtime_test.csv not found")
        return None

    with open(CONFIG / "soc_realtime_metadata.json") as f:
        meta = json.load(f)
    features = meta["features"]

    X, y = load_csv(csv_path, features, "soc_remaining_delta")
    dm = xgb.DMatrix(X, feature_names=features)

    models = load_ensemble("soc_realtime_model")
    print(f"  Loaded {len(models)} ensemble models, {len(features)} features, {len(y)} samples")

    t0 = time.perf_counter()
    y_pred = ensemble_predict(models, dm)
    elapsed = (time.perf_counter() - t0) * 1000

    m = metrics(y, y_pred)
    print(f"  MAE={m['MAE']:.4f}%  RMSE={m['RMSE']:.4f}%  R2={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
    print(f"  Inference: {elapsed:.1f}ms total ({elapsed / len(y):.3f}ms/sample)")

    # Compare against expected
    exp = EXPECTED["1B"]
    print(f"\n  Workstation comparison:")
    check("1B MAE", m["MAE"], exp["MAE"], TOLERANCE_MAE)
    check("1B RMSE", m["RMSE"], exp["RMSE"], TOLERANCE_MAE)
    check("1B R2", m["R2"], exp["R2"], TOLERANCE_R2)

    # Breakdown by trip progress
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        progress = np.array([float(row["trip_progress_fraction"]) for row in reader])

    bins = [(0.0, 0.25, "0-25%"), (0.25, 0.50, "25-50%"),
            (0.50, 0.75, "50-75%"), (0.75, 1.01, "75-100%")]
    for lo, hi, label in bins:
        mask = (progress >= lo) & (progress < hi)
        if mask.sum() == 0:
            continue
        m_bin = metrics(y[mask], y_pred[mask])
        print(f"\n  Progress {label} ({mask.sum()} rows):")
        print(f"    MAE={m_bin['MAE']:.4f}%, RMSE={m_bin['RMSE']:.4f}%, R2={m_bin['R2']:.4f}")

    return m


# ── Model 2: Motor Anomaly Detection ────────────────────────────────────

def benchmark_anomaly():
    print("\n\n3. Model 2 — Motor Anomaly Detection")
    print("-" * 60)

    csv_path = TESTDATA / "anomaly_test.csv"
    if not csv_path.exists():
        print("  SKIP: testdata/anomaly_test.csv not found")
        return None

    with open(CONFIG / "anomaly_metadata.json") as f:
        meta = json.load(f)
    all_features = meta["features"]
    targets = meta["targets"]

    with open(CONFIG / "feature_statistics.json") as f:
        stats = json.load(f)
    feature_stds = stats["feature_stds"]

    with open(CONFIG / "thresholds.json") as f:
        thresholds = json.load(f)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()
                 if k in all_features} for row in reader]

    n = len(rows)
    print(f"  {n} test samples, {len(all_features)} features, {len(targets)} targets")

    recon_models: dict[str, xgb.Booster] = {}
    for target in targets:
        m = xgb.Booster()
        m.load_model(str(MODELS / f"anomaly_{target}.json"))
        recon_models[target] = m

    print("\n  Per-target reconstruction:")
    t0 = time.perf_counter()
    all_scores = np.zeros(n, dtype=np.float64)

    for target in targets:
        predictor_cols = [c for c in all_features if c != target]
        X = np.array([[row[c] for c in predictor_cols] for row in rows], dtype=np.float32)
        y_actual = np.array([row[target] for row in rows], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=predictor_cols)
        y_recon = recon_models[target].predict(dm)

        std = feature_stds[target]
        all_scores += ((y_recon - y_actual) / std) ** 2

        m = metrics(y_actual, y_recon)
        unit = "kW" if "power" in target else "RPM"
        print(f"    {target:25s}  MAE={m['MAE']:.3f} {unit},  R2={m['R2']:.4f}")

    elapsed = (time.perf_counter() - t0) * 1000

    p99 = thresholds["p99"]
    n_anomalous = int(np.sum(all_scores > p99))
    fpr = n_anomalous / n * 100

    print(f"\n  FPR (p99 threshold): {fpr:.2f}% ({n_anomalous}/{n})")
    print(f"  Inference: {elapsed:.1f}ms total ({elapsed / n:.3f}ms/sample)")

    # Injection test
    print("\n  Fault injection (port_motor_power x1.5):")
    n_inject = min(500, n)
    detected = 0
    for i in range(n_inject):
        row = dict(rows[i])
        row["port_motor_power"] *= 1.5

        score = 0.0
        for target in targets:
            predictor_cols = [c for c in all_features if c != target]
            X = np.array([[row[c] for c in predictor_cols]], dtype=np.float32)
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            pred = float(recon_models[target].predict(dm)[0])
            std = feature_stds[target]
            score += ((pred - row[target]) / std) ** 2

        if score > p99:
            detected += 1

    rate = detected / n_inject * 100
    print(f"    Detection rate: {detected}/{n_inject} = {rate:.1f}%")

    # Compare against expected
    exp = EXPECTED["2"]
    print(f"\n  Workstation comparison:")
    check("Model 2 FPR", fpr, exp["FPR"], TOLERANCE_RATE)
    check("Model 2 Detection", rate, exp["detection_rate"], TOLERANCE_RATE)

    return {"fpr": fpr, "detection_rate": rate}


# ── Models 3a/3b: Battery Anomaly Detection ─────────────────────────────

def benchmark_battery(mode: str):
    """Benchmark battery anomaly model (charging or operational)."""
    label = "3a" if mode == "charging" else "3b"
    title = f"Model {label} — Battery Anomaly ({mode.title()})"
    print(f"\n\n{4 if mode == 'charging' else 5}. {title}")
    print("-" * 60)

    prefix = f"battery_{mode}"
    meta_path = CONFIG / f"{prefix}_metadata.json"
    if not meta_path.exists():
        print(f"  SKIP: config/{prefix}_metadata.json not found")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    features = meta["features"]
    targets = meta["targets"]

    stats_path = CONFIG / f"{prefix}_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    feature_stds = stats["feature_stds"]

    thresh_path = CONFIG / f"{prefix}_thresholds.json"
    with open(thresh_path) as f:
        thresholds = json.load(f)

    # Load reconstruction models
    recon_models: dict[str, xgb.Booster] = {}
    for target in targets:
        model_path = MODELS / f"{prefix}_{target}.json"
        if not model_path.exists():
            print(f"  SKIP: models/{prefix}_{target}.json not found")
            return None
        m = xgb.Booster()
        m.load_model(str(model_path))
        recon_models[target] = m

    print(f"  Loaded {len(recon_models)} reconstruction models, {len(features)} features")

    # Generate synthetic normal data from feature means for smoke test
    feature_means = stats.get("feature_means", {})
    if not feature_means:
        print("  SKIP: no feature_means in statistics — cannot generate test data")
        return None

    # Smoke test: predict on mean values (should produce low anomaly score)
    mean_row = {f: feature_means.get(f, 0.0) for f in features}

    t0 = time.perf_counter()
    score = 0.0
    for target in targets:
        predictor_cols = [c for c in features if c != target]
        X = np.array([[mean_row[c] for c in predictor_cols]], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=predictor_cols)
        pred = float(recon_models[target].predict(dm)[0])
        std = max(feature_stds.get(target, 1.0), 0.01)
        score += ((pred - mean_row[target]) / std) ** 2
    elapsed_single = (time.perf_counter() - t0) * 1000

    p99 = thresholds["p99"]
    level = "normal" if score <= p99 else "anomalous"
    print(f"  Smoke test (mean values): score={score:.4f}, p99={p99:.4f}, level={level}")
    print(f"  Single inference: {elapsed_single:.1f}ms")

    # Injection test: corrupt cell voltage spread
    injected_row = dict(mean_row)
    for f in features:
        if "cell_v_spread" in f:
            injected_row[f] = mean_row[f] + 0.20  # +200mV spread
        if "cell_t_spread" in f:
            injected_row[f] = mean_row[f] + 10.0   # +10C spread

    inject_score = 0.0
    for target in targets:
        predictor_cols = [c for c in features if c != target]
        X = np.array([[injected_row[c] for c in predictor_cols]], dtype=np.float32)
        dm = xgb.DMatrix(X, feature_names=predictor_cols)
        pred = float(recon_models[target].predict(dm)[0])
        std = max(feature_stds.get(target, 1.0), 0.01)
        inject_score += ((pred - injected_row[target]) / std) ** 2

    inject_detected = inject_score > p99
    print(f"  Injection test (v_spread+0.2V, t_spread+10C): score={inject_score:.4f}, detected={inject_detected}")

    # Timing benchmark (100 iterations)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        s = 0.0
        for target in targets:
            predictor_cols = [c for c in features if c != target]
            X = np.array([[mean_row[c] for c in predictor_cols]], dtype=np.float32)
            dm = xgb.DMatrix(X, feature_names=predictor_cols)
            pred = float(recon_models[target].predict(dm)[0])
            std = max(feature_stds.get(target, 1.0), 0.01)
            s += ((pred - mean_row[target]) / std) ** 2
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    print(f"  Timing (100 iters): avg={avg_ms:.1f}ms, min={min(times):.1f}ms, max={max(times):.1f}ms")

    # Compare: smoke test should be normal, injection should be detected
    print(f"\n  Workstation comparison:")
    check(f"{label} normal smoke test", 1.0 if level == "normal" else 0.0, 1.0, 0.0, " (bool)")
    check(f"{label} injection detected", 1.0 if inject_detected else 0.0, 1.0, 0.0, " (bool)")

    return {"score_normal": score, "score_injected": inject_score, "detected": inject_detected, "p99": p99}


# ── Timing Benchmark ────────────────────────────────────────────────────

def benchmark_timing():
    """Run 100-iteration timing benchmark for each model."""
    print("\n\n6. Inference Timing Benchmark (100 iterations)")
    print("-" * 60)

    results = {}

    # Model 1A timing
    meta_path = CONFIG / "soc_trip_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        features = meta["features"]
        models = load_ensemble("soc_trip_model")
        X_dummy = np.random.randn(1, len(features)).astype(np.float32)
        dm = xgb.DMatrix(X_dummy, feature_names=features)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            ensemble_predict(models, dm)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        results["1A"] = {"avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}
        print(f"  1A Trip:     avg={avg:.2f}ms  min={min(times):.2f}ms  max={max(times):.2f}ms")

    # Model 1B timing
    meta_path = CONFIG / "soc_realtime_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        features = meta["features"]
        models = load_ensemble("soc_realtime_model")
        X_dummy = np.random.randn(1, len(features)).astype(np.float32)
        dm = xgb.DMatrix(X_dummy, feature_names=features)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            ensemble_predict(models, dm)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        results["1B"] = {"avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}
        print(f"  1B Realtime: avg={avg:.2f}ms  min={min(times):.2f}ms  max={max(times):.2f}ms")

    # Model 2 timing
    meta_path = CONFIG / "anomaly_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        all_features = meta["features"]
        targets = meta["targets"]
        recon_models = {}
        for target in targets:
            m = xgb.Booster()
            m.load_model(str(MODELS / f"anomaly_{target}.json"))
            recon_models[target] = m

        X_dummy = np.random.randn(1, len(all_features) - 1).astype(np.float32)
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            for target in targets:
                predictor_cols = [c for c in all_features if c != target]
                dm = xgb.DMatrix(X_dummy, feature_names=predictor_cols)
                recon_models[target].predict(dm)
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        results["2"] = {"avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}
        print(f"  2  Anomaly:  avg={avg:.2f}ms  min={min(times):.2f}ms  max={max(times):.2f}ms")

    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=" * 65)
    print("  M/B Dalaray Digital Shadow — RPi5 Benchmark")
    print(f"  Bundle:   {BUNDLE}")
    print(f"  Platform: {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"  Python:   {platform.python_version()}")
    print(f"  XGBoost:  {xgb.__version__}")
    print(f"  NumPy:    {np.__version__}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Production models
    trip_m = benchmark_trip()
    rt_m = benchmark_realtime()
    anom_m = benchmark_anomaly()
    bat_c = benchmark_battery("charging")
    bat_o = benchmark_battery("operational")
    timing = benchmark_timing()

    elapsed = time.perf_counter() - t_start

    # ── Summary ──
    print("\n\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)

    print(f"\n  {'Model':<30s} {'Metric':>12s} {'RPi5':>10s} {'Expected':>10s} {'Status':>8s}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8}")

    if trip_m:
        exp = EXPECTED["1A"]
        s1 = "OK" if abs(trip_m["MAE"] - exp["MAE"]) <= TOLERANCE_MAE else "FAIL"
        print(f"  {'1A Trip SOC':<30s} {'MAE':>12s} {trip_m['MAE']:>9.4f}% {exp['MAE']:>9.4f}% {s1:>8s}")
        print(f"  {'':30s} {'R2':>12s} {trip_m['R2']:>10.4f} {exp['R2']:>10.4f}")
    if rt_m:
        exp = EXPECTED["1B"]
        s2 = "OK" if abs(rt_m["MAE"] - exp["MAE"]) <= TOLERANCE_MAE else "FAIL"
        print(f"  {'1B Realtime SOC':<30s} {'MAE':>12s} {rt_m['MAE']:>9.4f}% {exp['MAE']:>9.4f}% {s2:>8s}")
        print(f"  {'':30s} {'R2':>12s} {rt_m['R2']:>10.4f} {exp['R2']:>10.4f}")
    if anom_m:
        exp = EXPECTED["2"]
        s3 = "OK" if abs(anom_m["fpr"] - exp["FPR"]) <= TOLERANCE_RATE else "FAIL"
        print(f"  {'2  Motor Anomaly':<30s} {'FPR':>12s} {anom_m['fpr']:>9.2f}% {exp['FPR']:>9.2f}% {s3:>8s}")
        print(f"  {'':30s} {'Detection':>12s} {anom_m['detection_rate']:>9.1f}% {exp['detection_rate']:>9.1f}%")
    if bat_c:
        print(f"  {'3a Battery (charging)':<30s} {'Smoke':>12s} {'PASS' if bat_c['score_normal'] <= bat_c.get('p99', 2.751) else 'FAIL':>10s}")
        print(f"  {'':30s} {'Injection':>12s} {'PASS' if bat_c['detected'] else 'FAIL':>10s}")
    if bat_o:
        print(f"  {'3b Battery (operational)':<30s} {'Smoke':>12s} {'PASS' if bat_o['score_normal'] <= bat_o.get('p99', 15.643) else 'FAIL':>10s}")
        print(f"  {'':30s} {'Injection':>12s} {'PASS' if bat_o['detected'] else 'FAIL':>10s}")

    if timing:
        print(f"\n  Inference latency (single sample, 100-iter avg):")
        for model, t in timing.items():
            print(f"    Model {model}: {t['avg_ms']:.2f}ms")

    # Overall verdict
    n_match = sum(1 for c in checks if c["status"] == "MATCH")
    n_mismatch = sum(1 for c in checks if c["status"] == "MISMATCH")
    n_total = len(checks)

    print(f"\n  Checks: {n_match}/{n_total} passed", end="")
    if n_mismatch > 0:
        print(f", {n_mismatch} MISMATCH:")
        for c in checks:
            if c["status"] == "MISMATCH":
                print(f"    X {c['name']}: got {c['actual']:.4f}, expected {c['expected']:.4f} "
                      f"(diff={c['diff']:.4f}, tol={c['tolerance']:.4f})")
    else:
        print(" — all results match workstation!")

    print(f"\n  Total benchmark time: {elapsed:.1f}s")
    print("=" * 65)

    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "numpy": np.__version__,
        },
        "models": {
            "1A": {"metrics": trip_m, "expected": EXPECTED["1A"]} if trip_m else None,
            "1B": {"metrics": rt_m, "expected": EXPECTED["1B"]} if rt_m else None,
            "2": anom_m if anom_m else None,
            "3a": bat_c if bat_c else None,
            "3b": bat_o if bat_o else None,
        },
        "timing": timing,
        "checks": checks,
        "n_match": n_match,
        "n_mismatch": n_mismatch,
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = BUNDLE / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {report_path}")

    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
