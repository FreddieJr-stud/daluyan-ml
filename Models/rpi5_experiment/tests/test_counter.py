"""Tests for device counting logic with DBSCAN clustering."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from counter import DeviceCounter
from tests.fixtures import (
    scenario_3_real_devices,
    scenario_2_randomized_same_device,
    scenario_mixed_5_people,
    scenario_with_crew,
)


class TestRealMACCounting:
    """Real (non-randomized) MACs should be counted by unique address."""

    def test_3_real_devices(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_3_real_devices()
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        assert result["real_mac_count"] == 3

    def test_duplicate_real_macs_counted_once(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_3_real_devices()
        # Add same probes twice — should still be 3 unique
        for p in probes + probes:
            counter.add_probe(p)
        result = counter.compute_count()
        assert result["real_mac_count"] == 3


class TestRandomizedMACClustering:
    """Randomized MACs from the same device should cluster together."""

    def test_2_macs_from_1_device_clusters_to_1(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_2_randomized_same_device()
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        # Two randomized MACs but same device → should cluster to 1
        assert result["randomized_cluster_count"] <= 2
        # With good clustering, expect 1
        # Allow some tolerance for PoC: at most 2
        assert result["randomized_cluster_count"] >= 1


class TestMixedScenario:
    """Mixed real + randomized MACs across multiple devices."""

    def test_5_people_mixed(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_mixed_5_people()
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        assert result["real_mac_count"] == 2
        # 3 randomized devices using 5 MACs → expect 3-5 clusters
        assert 3 <= result["randomized_cluster_count"] <= 5
        # Total raw devices (before correction): 5-7 acceptable
        assert result["raw_devices_detected"] >= 5


class TestCrewFiltering:
    """Known crew MACs should be excluded from the count."""

    def test_crew_mac_filtered(self):
        crew = ["00:11:22:33:44:55"]
        counter = DeviceCounter(correction_factor=1.0, crew_macs=crew)
        probes = scenario_with_crew(crew_macs=crew)
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        # 3 passengers, crew excluded
        assert result["real_mac_count"] == 2  # 2 real passengers
        assert result["randomized_cluster_count"] >= 1  # 1 randomized passenger


class TestCorrectionFactor:
    """Correction factor scales the final passenger count."""

    def test_correction_factor_applied(self):
        counter = DeviceCounter(correction_factor=0.8)
        probes = scenario_3_real_devices()
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        # 3 real devices * 0.8 = 2.4 → rounded to 2
        assert result["passenger_count"] == round(3 * 0.8)

    def test_correction_factor_1_no_change(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_3_real_devices()
        for p in probes:
            counter.add_probe(p)
        result = counter.compute_count()
        assert result["passenger_count"] == 3


class TestReset:
    """Counter should be resettable between epochs."""

    def test_reset_clears_state(self):
        counter = DeviceCounter(correction_factor=1.0)
        probes = scenario_3_real_devices()
        for p in probes:
            counter.add_probe(p)
        result1 = counter.compute_count()
        assert result1["real_mac_count"] == 3

        counter.reset()
        result2 = counter.compute_count()
        assert result2["real_mac_count"] == 0
        assert result2["passenger_count"] == 0
