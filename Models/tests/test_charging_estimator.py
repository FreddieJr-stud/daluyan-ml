"""Tests for ChargingEstimator."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rpi5_bundle"))

from inference.charging_estimator import ChargingEstimator


@pytest.fixture
def sample_curve(tmp_path):
    """Create a simple test charging curve."""
    curve = {
        "battery_capacity_kwh": 160.0,
        "bins": [20, 21, 22, 23, 24, 25, 50, 75, 95, 96, 97],
        "median_kw": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 300.0, 300.0, 300.0, 300.0],
        "p25_kw": [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 180.0, 280.0, 280.0, 280.0, 280.0],
        "p75_kw": [110.0, 110.0, 110.0, 110.0, 110.0, 110.0, 220.0, 320.0, 320.0, 320.0, 320.0],
        "sample_counts": [50, 50, 50, 50, 50, 50, 100, 150, 200, 200, 200],
    }
    path = tmp_path / "charging_curve.json"
    with open(path, "w") as f:
        json.dump(curve, f)
    return path


@pytest.fixture
def estimator(sample_curve):
    return ChargingEstimator(str(sample_curve))


def test_loads_curve(estimator):
    """Estimator loads the curve JSON and builds lookup."""
    assert estimator.capacity_kwh == 160.0
    assert len(estimator._power_at_soc) > 0


def test_estimate_minutes_basic(estimator):
    """Estimate from 20% to 25% at 100 kW = 5% * 1.6 kWh / 100 kW * 60 = 4.8 min."""
    mins = estimator.estimate_minutes(current_soc=20.0, target_soc=25.0)
    assert mins is not None
    assert abs(mins - 4.8) < 0.5


def test_estimate_minutes_partial_first_bin(estimator):
    """Starting at 20.5% should give less time than starting at 20.0%."""
    full = estimator.estimate_minutes(current_soc=20.0, target_soc=25.0)
    partial = estimator.estimate_minutes(current_soc=20.5, target_soc=25.0)
    assert partial < full


def test_estimate_to_full(estimator):
    """estimate_to_full always targets 100%."""
    result = estimator.estimate_to_full(current_soc=95.0)
    assert result is not None
    assert result > 0


def test_current_soc_equals_target_returns_zero(estimator):
    """If already at target, return 0."""
    mins = estimator.estimate_minutes(current_soc=50.0, target_soc=50.0)
    assert mins == 0.0


def test_current_soc_above_target_returns_zero(estimator):
    """If SOC is already above target, return 0."""
    mins = estimator.estimate_minutes(current_soc=80.0, target_soc=50.0)
    assert mins == 0.0


def test_charging_power_at_soc(estimator):
    """get_power_at_soc returns known power for a known bin."""
    power = estimator.get_power_at_soc(20.0)
    assert power == 100.0


def test_get_power_interpolates(estimator):
    """Power between known bins is linearly interpolated."""
    # Between bin 25 (100 kW) and bin 50 (200 kW)
    power = estimator.get_power_at_soc(37.5)
    assert 100 < power < 200
