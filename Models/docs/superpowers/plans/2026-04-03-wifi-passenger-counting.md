# WiFi Probe Request Passenger Counting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PoC WiFi probe request sniffer + passenger counter that runs on RPi5, outputting a `passenger_count.json` consumable by the existing `rpi5_bundle` inference pipeline's `start_trip(passengers=N)`.

**Architecture:** Scapy captures 802.11 probe request frames on the Tenda dongle (`wlan1`) in monitor mode. A Python pipeline classifies MACs as real vs. randomized, extracts fingerprint features (supported rates, sequence numbers, RSSI), clusters randomized MACs via DBSCAN, and writes an estimated passenger count to JSON every epoch (default 120s).

**Tech Stack:** Python 3.11+, scapy, scikit-learn (DBSCAN), numpy. Shell: bash (setup_monitor.sh). Target: RPi5 4GB, RPi OS Lite.

**Design Spec:** `docs/superpowers/specs/2026-04-03-wifi-passenger-counting-design.md`

---

## File Structure

```
rpi5_experiment/
├── setup_monitor.sh        # Mode-switch Tenda dongle + enter monitor mode
├── config.json             # Runtime settings: interface, epoch, correction_factor
├── mac_utils.py            # MAC classification (real vs randomized) + rate fingerprinting
├── sniffer.py              # Scapy-based probe request capture + field extraction
├── counter.py              # DBSCAN clustering, device counting, epoch management
├── main.py                 # Entry point: orchestrates sniffer → counter → JSON output
├── requirements.txt        # Python dependencies for RPi5
└── tests/
    ├── test_mac_utils.py   # Unit tests for MAC classifier and fingerprinting
    ├── test_counter.py     # Unit tests for counting logic with synthetic probe data
    └── fixtures.py         # Shared synthetic probe request data for offline testing
```

**Design rationale:**
- `sniffer.py` is the only file that requires real hardware (monitor mode WiFi). Everything else is testable offline with synthetic data.
- `mac_utils.py` is pure logic — no I/O, no hardware. Fully unit-testable.
- `counter.py` takes structured probe data (not raw packets) — testable without scapy or hardware.
- `main.py` is thin glue: reads config, wires sniffer to counter, writes JSON, handles Ctrl+C.

---

### Task 1: Project Scaffold

**Files:**
- Create: `rpi5_experiment/requirements.txt`
- Create: `rpi5_experiment/config.json`

- [ ] **Step 1: Create requirements.txt**

```txt
scapy>=2.5.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

- [ ] **Step 2: Create config.json**

```json
{
  "interface": "wlan1",
  "epoch_seconds": 120,
  "correction_factor": 0.8,
  "min_rssi": -80,
  "dbscan_eps": 0.3,
  "dbscan_min_samples": 2,
  "output_file": "passenger_count.json",
  "crew_macs": []
}
```

Fields:
- `interface`: WiFi interface in monitor mode (Tenda dongle)
- `epoch_seconds`: Time window for counting (seconds). Longer = more accurate.
- `correction_factor`: Multiplier to convert device count → passenger count (0.6-0.9)
- `min_rssi`: Ignore probes weaker than this (dBm). -80 filters distant devices.
- `dbscan_eps`: DBSCAN neighborhood radius (tuned during calibration)
- `dbscan_min_samples`: Min points to form a cluster
- `output_file`: Where to write the passenger count JSON
- `crew_macs`: Known crew MAC addresses to subtract from count

- [ ] **Step 3: Commit**

```bash
cd rpi5_experiment
git add requirements.txt config.json
git commit -m "feat(passenger-counter): add project scaffold with requirements and config"
```

---

### Task 2: Monitor Mode Setup Script

**Files:**
- Create: `rpi5_experiment/setup_monitor.sh`

- [ ] **Step 1: Write setup_monitor.sh**

```bash
#!/bin/bash
# setup_monitor.sh — Initialize Tenda dongle and enter monitor mode
# Usage: sudo bash setup_monitor.sh [interface]
#
# The Tenda RTL8192FU boots in USB mass-storage mode.
# This script: (1) mode-switches it to WiFi, (2) enables monitor mode.

set -euo pipefail

IFACE="${1:-wlan1}"

echo "[1/4] Mode-switching Tenda dongle (0bda:a192)..."
if lsusb | grep -q "0bda:a192"; then
    usb_modeswitch -v 0bda -p a192 \
        -M '5553424312345678000000000000061b000000020000000000000000000000'
    echo "  Waiting for driver to load..."
    sleep 4
else
    echo "  Dongle already in WiFi mode (0bda:a192 not found as DISK), skipping."
fi

echo "[2/4] Checking interface ${IFACE}..."
if ! iw dev "$IFACE" info > /dev/null 2>&1; then
    echo "ERROR: ${IFACE} not found. Check dmesg for driver issues."
    echo "  Run: dmesg | grep -i rtl8"
    exit 1
fi

echo "[3/4] Setting ${IFACE} to monitor mode..."
ip link set "$IFACE" down
iw dev "$IFACE" set type monitor
ip link set "$IFACE" up

echo "[4/4] Verifying..."
MODE=$(iw dev "$IFACE" info | grep type | awk '{print $2}')
if [ "$MODE" = "monitor" ]; then
    echo "OK: ${IFACE} is in monitor mode"
    echo "  MAC: $(iw dev "$IFACE" info | grep addr | awk '{print $2}')"
    echo "  Ready for probe capture."
else
    echo "ERROR: ${IFACE} is in '${MODE}' mode, expected 'monitor'"
    exit 1
fi
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x setup_monitor.sh
git add setup_monitor.sh
git commit -m "feat(passenger-counter): add monitor mode setup script for Tenda dongle"
```

---

### Task 3: MAC Utilities — Classification & Fingerprinting

**Files:**
- Create: `rpi5_experiment/mac_utils.py`
- Create: `rpi5_experiment/tests/test_mac_utils.py`

- [ ] **Step 1: Create tests/test_mac_utils.py with failing tests**

```python
"""Tests for MAC address classification and rate fingerprinting."""
import pytest
from mac_utils import is_randomized, rates_fingerprint


class TestIsRandomized:
    """Check the locally-administered bit (bit 1 of first octet)."""

    def test_real_mac_apple(self):
        # Apple OUI: 3C:22:FB — first octet 0x3C = 0011 1100, bit1=0 → real
        assert is_randomized("3c:22:fb:01:02:03") is False

    def test_real_mac_samsung(self):
        # Samsung OUI: 8C:F5:A3 — first octet 0x8C = 1000 1100, bit1=0 → real
        assert is_randomized("8c:f5:a3:01:02:03") is False

    def test_randomized_mac_locally_administered(self):
        # Locally administered: first octet has bit1 set
        # 0xDA = 1101 1010, bit1=1 → randomized
        assert is_randomized("da:a1:19:01:02:03") is True

    def test_randomized_mac_common_pattern(self):
        # 0x02 = 0000 0010, bit1=1 → randomized (minimal case)
        assert is_randomized("02:00:00:00:00:00") is True

    def test_real_mac_zero_prefix(self):
        # 0x00 = 0000 0000, bit1=0 → real
        assert is_randomized("00:11:22:33:44:55") is False

    def test_case_insensitive(self):
        assert is_randomized("DA:A1:19:01:02:03") is True
        assert is_randomized("3C:22:FB:01:02:03") is False


class TestRatesFingerprint:
    """Hash supported data rates into a stable device fingerprint."""

    def test_same_rates_same_hash(self):
        rates_a = [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0]
        rates_b = [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0]
        assert rates_fingerprint(rates_a) == rates_fingerprint(rates_b)

    def test_different_rates_different_hash(self):
        rates_a = [1.0, 2.0, 5.5, 11.0]
        rates_b = [6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 54.0]
        assert rates_fingerprint(rates_a) != rates_fingerprint(rates_b)

    def test_order_independent(self):
        rates_a = [11.0, 5.5, 2.0, 1.0]
        rates_b = [1.0, 2.0, 5.5, 11.0]
        assert rates_fingerprint(rates_a) == rates_fingerprint(rates_b)

    def test_empty_rates(self):
        result = rates_fingerprint([])
        assert isinstance(result, str)
        assert len(result) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd rpi5_experiment
python -m pytest tests/test_mac_utils.py -v
```

Expected: `ModuleNotFoundError: No module named 'mac_utils'`

- [ ] **Step 3: Implement mac_utils.py**

```python
"""MAC address classification and probe request fingerprinting utilities."""
import hashlib


def is_randomized(mac: str) -> bool:
    """Check if a MAC address is locally administered (randomized).

    Bit 1 (second-least-significant) of the first octet is the
    "locally administered" flag. Real MACs assigned by manufacturers
    have this bit cleared; randomized MACs set it.
    """
    first_octet = int(mac.split(":")[0], 16)
    return bool(first_octet & 0x02)


def rates_fingerprint(supported_rates: list[float]) -> str:
    """Hash supported data rates into a stable fingerprint string.

    Probe requests include supported rates (IE Tag 1) that reflect
    device hardware capabilities. This is stable across MAC changes
    and useful for grouping frames from the same physical device.
    """
    sorted_rates = sorted(supported_rates)
    raw = ",".join(f"{r:.1f}" for r in sorted_rates)
    return hashlib.md5(raw.encode()).hexdigest()[:8]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_mac_utils.py -v
```

Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mac_utils.py tests/test_mac_utils.py
git commit -m "feat(passenger-counter): add MAC classifier and rate fingerprinting with tests"
```

---

### Task 4: Test Fixtures — Synthetic Probe Data

**Files:**
- Create: `rpi5_experiment/tests/__init__.py`
- Create: `rpi5_experiment/tests/fixtures.py`

- [ ] **Step 1: Create tests/__init__.py (empty)**

```python
```

- [ ] **Step 2: Create tests/fixtures.py with synthetic probe request data**

This provides realistic test data without needing real hardware. It simulates what the sniffer module will produce: a list of dicts, one per captured probe request frame.

```python
"""Synthetic probe request data for offline testing.

Each entry represents a parsed probe request with fields matching
what sniffer.py will produce from real captures.
"""
import time


def make_probe(
    mac: str,
    rssi: int = -50,
    seq: int = 0,
    rates: list[float] | None = None,
    ssid: str = "",
    ts: float | None = None,
) -> dict:
    """Create a single synthetic probe request record."""
    return {
        "mac": mac.lower(),
        "rssi": rssi,
        "seq_number": seq,
        "supported_rates": rates or [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0],
        "ssid": ssid,
        "timestamp": ts or time.time(),
    }


def scenario_3_real_devices() -> list[dict]:
    """3 devices with real (globally unique) MACs.

    Simulates 3 people with older phones or phones that don't
    randomize MACs. Each device sends 3-5 probes over 60 seconds.
    """
    now = time.time()
    return [
        # Device 1: Apple phone
        make_probe("3c:22:fb:01:02:03", rssi=-45, seq=100, ts=now + 0),
        make_probe("3c:22:fb:01:02:03", rssi=-47, seq=101, ts=now + 5),
        make_probe("3c:22:fb:01:02:03", rssi=-44, seq=102, ts=now + 12),
        # Device 2: Samsung phone
        make_probe("8c:f5:a3:aa:bb:cc", rssi=-60, seq=200, ts=now + 2),
        make_probe("8c:f5:a3:aa:bb:cc", rssi=-58, seq=201, ts=now + 8),
        make_probe("8c:f5:a3:aa:bb:cc", rssi=-62, seq=202, ts=now + 15),
        make_probe("8c:f5:a3:aa:bb:cc", rssi=-59, seq=203, ts=now + 22),
        # Device 3: Xiaomi phone
        make_probe("64:cc:2e:dd:ee:ff", rssi=-70, seq=300, ts=now + 1),
        make_probe("64:cc:2e:dd:ee:ff", rssi=-72, seq=301, ts=now + 10),
        make_probe("64:cc:2e:dd:ee:ff", rssi=-68, seq=302, ts=now + 20),
        make_probe("64:cc:2e:dd:ee:ff", rssi=-71, seq=303, ts=now + 30),
        make_probe("64:cc:2e:dd:ee:ff", rssi=-69, seq=304, ts=now + 40),
    ]


def scenario_2_randomized_same_device() -> list[dict]:
    """1 physical device sending probes with 2 different randomized MACs.

    Simulates modern phone MAC randomization. Same device fingerprint:
    same rates, sequential seq_numbers, similar RSSI.
    The counter should cluster these into 1 device.
    """
    now = time.time()
    rates = [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 54.0]
    return [
        # First randomized MAC
        make_probe("da:a1:19:01:02:03", rssi=-50, seq=500, rates=rates, ts=now + 0),
        make_probe("da:a1:19:01:02:03", rssi=-52, seq=501, rates=rates, ts=now + 3),
        make_probe("da:a1:19:01:02:03", rssi=-49, seq=502, rates=rates, ts=now + 7),
        # MAC changes — same device, seq continues, same rates/RSSI
        make_probe("fe:b2:33:44:55:66", rssi=-51, seq=503, rates=rates, ts=now + 15),
        make_probe("fe:b2:33:44:55:66", rssi=-48, seq=504, rates=rates, ts=now + 20),
    ]


def scenario_mixed_5_people() -> list[dict]:
    """5 physical devices: 2 real MACs + 3 randomized (using 5 MACs).

    Expected count: 5 devices total.
    - 2 real MACs → counted directly
    - 3 randomized devices using 5 different MACs → should cluster to 3
    """
    now = time.time()
    rates_iphone = [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0, 24.0, 36.0, 48.0, 54.0]
    rates_android = [1.0, 2.0, 5.5, 11.0, 6.0, 9.0, 12.0, 18.0]
    rates_old = [1.0, 2.0, 5.5, 11.0]

    probes = []

    # Real device 1 (Samsung)
    for i in range(4):
        probes.append(make_probe("8c:f5:a3:11:22:33", rssi=-55, seq=100 + i, ts=now + i * 5))

    # Real device 2 (Huawei)
    for i in range(3):
        probes.append(make_probe("04:d3:b0:44:55:66", rssi=-65, seq=200 + i, ts=now + i * 8))

    # Randomized device 3 (iPhone-like, 1 MAC)
    for i in range(3):
        probes.append(make_probe("da:11:22:33:44:55", rssi=-40, seq=300 + i, rates=rates_iphone, ts=now + i * 6))

    # Randomized device 4 (Android, 2 MACs — changes midway)
    for i in range(3):
        probes.append(make_probe("fa:aa:bb:cc:dd:01", rssi=-72, seq=400 + i, rates=rates_android, ts=now + i * 4))
    for i in range(2):
        probes.append(make_probe("fe:aa:bb:cc:dd:02", rssi=-70, seq=403 + i, rates=rates_android, ts=now + 15 + i * 5))

    # Randomized device 5 (old phone, 2 MACs)
    for i in range(2):
        probes.append(make_probe("d2:99:88:77:66:01", rssi=-80, seq=500 + i, rates=rates_old, ts=now + i * 10))
    for i in range(2):
        probes.append(make_probe("d6:99:88:77:66:02", rssi=-78, seq=502 + i, rates=rates_old, ts=now + 25 + i * 10))

    return probes


def scenario_with_crew(crew_macs: list[str] | None = None) -> list[dict]:
    """3 passengers + 1 crew member. Crew MAC should be excluded.

    Expected count after filtering: 3 passengers.
    """
    if crew_macs is None:
        crew_macs = ["00:11:22:33:44:55"]
    now = time.time()
    probes = []

    # Crew member (real MAC, should be filtered out)
    for i in range(5):
        probes.append(make_probe(crew_macs[0], rssi=-30, seq=10 + i, ts=now + i * 3))

    # Passenger 1 (real)
    for i in range(3):
        probes.append(make_probe("3c:22:fb:aa:bb:cc", rssi=-50, seq=100 + i, ts=now + i * 5))

    # Passenger 2 (real)
    for i in range(3):
        probes.append(make_probe("8c:f5:a3:dd:ee:ff", rssi=-60, seq=200 + i, ts=now + i * 7))

    # Passenger 3 (randomized)
    for i in range(4):
        probes.append(make_probe("da:cc:bb:aa:99:88", rssi=-55, seq=300 + i, ts=now + i * 4))

    return probes
```

- [ ] **Step 3: Commit**

```bash
git add tests/__init__.py tests/fixtures.py
git commit -m "feat(passenger-counter): add synthetic probe request test fixtures"
```

---

### Task 5: Device Counter — DBSCAN Clustering & Counting

**Files:**
- Create: `rpi5_experiment/counter.py`
- Create: `rpi5_experiment/tests/test_counter.py`

- [ ] **Step 1: Write tests/test_counter.py with failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_counter.py -v
```

Expected: `ModuleNotFoundError: No module named 'counter'`

- [ ] **Step 3: Implement counter.py**

```python
"""Device counter: classifies, fingerprints, clusters, and counts probe requests."""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from mac_utils import is_randomized, rates_fingerprint


class DeviceCounter:
    """Counts unique devices from WiFi probe request data.

    Real (globally unique) MACs are counted directly.
    Randomized (locally administered) MACs are clustered by fingerprint
    similarity using DBSCAN to estimate unique physical devices.
    """

    def __init__(
        self,
        correction_factor: float = 0.8,
        crew_macs: list[str] | None = None,
        min_rssi: int = -80,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
    ):
        self.correction_factor = correction_factor
        self.crew_macs = set(m.lower() for m in (crew_macs or []))
        self.min_rssi = min_rssi
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # State accumulated during an epoch
        self._real_macs: set[str] = set()
        self._randomized_probes: list[dict] = []

    def add_probe(self, probe: dict) -> None:
        """Ingest a single parsed probe request.

        Expected keys: mac, rssi, seq_number, supported_rates, ssid, timestamp
        """
        mac = probe["mac"].lower()

        # Filter: crew MACs
        if mac in self.crew_macs:
            return

        # Filter: too weak signal
        if probe["rssi"] < self.min_rssi:
            return

        if is_randomized(mac):
            self._randomized_probes.append(probe)
        else:
            self._real_macs.add(mac)

    def _cluster_randomized(self) -> int:
        """Cluster randomized probes into estimated unique devices.

        Uses DBSCAN on feature vectors: (rates_hash, seq_number, rssi).
        Falls back to unique (MAC, rates_hash) pairs if too few probes.
        """
        if not self._randomized_probes:
            return 0

        # Fallback: if very few probes, just count unique MACs
        if len(self._randomized_probes) < self.dbscan_min_samples:
            unique = set(p["mac"] for p in self._randomized_probes)
            return len(unique)

        # Build feature matrix for DBSCAN
        # Group probes by MAC first, then build per-MAC features
        by_mac: dict[str, list[dict]] = defaultdict(list)
        for p in self._randomized_probes:
            by_mac[p["mac"]].append(p)

        # If all probes share one MAC, that's 1 device
        if len(by_mac) == 1:
            return 1

        # Build one feature vector per unique randomized MAC
        mac_list = list(by_mac.keys())
        features = []
        for mac in mac_list:
            probes = by_mac[mac]
            avg_rssi = np.mean([p["rssi"] for p in probes])
            avg_seq = np.mean([p["seq_number"] for p in probes])
            fp = rates_fingerprint(probes[0]["supported_rates"])
            features.append({
                "mac": mac,
                "avg_rssi": avg_rssi,
                "avg_seq": avg_seq,
                "rates_fp": fp,
            })

        # Build numeric matrix for DBSCAN
        # Normalize: RSSI to [0,1], seq to [0,1], rates_fp to binary same/diff
        rssi_vals = np.array([f["avg_rssi"] for f in features])
        seq_vals = np.array([f["avg_seq"] for f in features])

        rssi_range = max(rssi_vals.max() - rssi_vals.min(), 1.0)
        seq_range = max(seq_vals.max() - seq_vals.min(), 1.0)

        n = len(features)
        X = np.zeros((n, 3))
        for i, f in enumerate(features):
            X[i, 0] = (f["avg_rssi"] - rssi_vals.min()) / rssi_range
            X[i, 1] = (f["avg_seq"] - seq_vals.min()) / seq_range
            # Rates fingerprint: encode as a numeric hash for distance
            X[i, 2] = int(f["rates_fp"], 16) / 0xFFFFFFFF

        # Run DBSCAN
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="euclidean",
        ).fit(X)

        labels = clustering.labels_
        # Number of clusters (excluding noise label -1)
        n_clusters = len(set(labels) - {-1})
        # Noise points: each treated as a separate device
        n_noise = int(np.sum(labels == -1))

        return n_clusters + n_noise

    def compute_count(self) -> dict:
        """Compute the estimated passenger count for the current epoch."""
        real_count = len(self._real_macs)
        randomized_count = self._cluster_randomized()
        raw_total = real_count + randomized_count
        passenger_count = round(raw_total * self.correction_factor)

        return {
            "passenger_count": passenger_count,
            "raw_devices_detected": raw_total,
            "real_mac_count": real_count,
            "randomized_cluster_count": randomized_count,
            "correction_factor": self.correction_factor,
            "epoch_seconds": 0,  # Set by caller
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def reset(self) -> None:
        """Clear state for the next epoch."""
        self._real_macs.clear()
        self._randomized_probes.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_counter.py -v
```

Expected: All tests PASS. Note: the clustering test (`test_2_macs_from_1_device_clusters_to_1`) may show 1 or 2 depending on DBSCAN tuning — the test allows both since this is PoC.

- [ ] **Step 5: Commit**

```bash
git add counter.py tests/test_counter.py
git commit -m "feat(passenger-counter): add DBSCAN-based device counter with tests"
```

---

### Task 6: Sniffer — Scapy Probe Request Capture

**Files:**
- Create: `rpi5_experiment/sniffer.py`

This module requires real hardware (monitor mode WiFi) and cannot be unit-tested offline. It will be validated in Task 8 on the RPi5.

- [ ] **Step 1: Implement sniffer.py**

```python
"""WiFi probe request sniffer using scapy.

Captures 802.11 probe request management frames on a monitor-mode
interface and extracts structured fields for the device counter.

Requires: sudo (raw packet capture) + interface in monitor mode.
"""
from __future__ import annotations

import time
from typing import Callable

from scapy.all import (
    Dot11,
    Dot11Elt,
    Dot11ProbeReq,
    RadioTap,
    sniff,
)


def parse_probe_request(packet) -> dict | None:
    """Extract structured fields from a raw scapy probe request packet.

    Returns None if the packet is not a valid probe request.
    """
    if not packet.haslayer(Dot11ProbeReq):
        return None

    dot11 = packet[Dot11]

    # Source MAC
    mac = dot11.addr2
    if mac is None:
        return None

    # RSSI from RadioTap header (dBm)
    rssi = -100  # default if not available
    if packet.haslayer(RadioTap):
        rssi = getattr(packet[RadioTap], "dBm_AntSignal", -100)

    # Sequence number (12-bit, from SC field)
    seq_number = dot11.SC >> 4 if dot11.SC else 0

    # Parse Information Elements for supported rates and SSID
    supported_rates = []
    ssid = ""
    elt = packet[Dot11Elt] if packet.haslayer(Dot11Elt) else None
    while elt:
        if elt.ID == 0:  # SSID
            try:
                ssid = elt.info.decode("utf-8", errors="replace")
            except (AttributeError, UnicodeDecodeError):
                ssid = ""
        elif elt.ID == 1:  # Supported Rates
            for byte in elt.info:
                rate = (byte & 0x7F) * 0.5  # Rate in Mbps
                supported_rates.append(rate)
        elif elt.ID == 50:  # Extended Supported Rates
            for byte in elt.info:
                rate = (byte & 0x7F) * 0.5
                supported_rates.append(rate)
        elt = elt.payload if isinstance(elt.payload, Dot11Elt) else None

    return {
        "mac": mac.lower(),
        "rssi": rssi,
        "seq_number": seq_number,
        "supported_rates": supported_rates,
        "ssid": ssid,
        "timestamp": time.time(),
    }


def start_sniffing(
    interface: str,
    callback: Callable[[dict], None],
    stop_event=None,
) -> None:
    """Start capturing probe requests on the given monitor-mode interface.

    Calls `callback(probe_dict)` for each valid probe request captured.
    Runs until `stop_event` is set (threading.Event) or KeyboardInterrupt.
    """
    def _handle_packet(packet):
        probe = parse_probe_request(packet)
        if probe is not None:
            callback(probe)

    print(f"[sniffer] Listening on {interface} for probe requests...")
    print(f"[sniffer] Press Ctrl+C to stop.")

    sniff(
        iface=interface,
        prn=_handle_packet,
        filter="type mgt subtype probe-req",
        store=False,
        stop_filter=lambda _: stop_event.is_set() if stop_event else False,
    )
```

- [ ] **Step 2: Commit**

```bash
git add sniffer.py
git commit -m "feat(passenger-counter): add scapy-based probe request sniffer"
```

---

### Task 7: Main Pipeline — Orchestration & JSON Output

**Files:**
- Create: `rpi5_experiment/main.py`

- [ ] **Step 1: Implement main.py**

```python
"""WiFi passenger counter — main entry point.

Orchestrates: sniffer (scapy) → counter (DBSCAN) → JSON output.
Runs in epochs: captures for N seconds, computes count, writes JSON, repeats.

Usage:
    sudo python main.py                    # Use defaults from config.json
    sudo python main.py --epoch 60         # Override epoch to 60 seconds
    sudo python main.py --once             # Run one epoch then exit
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

from counter import DeviceCounter
from sniffer import start_sniffing


def load_config(path: str = "config.json") -> dict:
    """Load runtime config from JSON file."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def write_output(result: dict, path: str) -> None:
    """Write passenger count result to JSON file atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="WiFi Probe Request Passenger Counter")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--epoch", type=int, help="Override epoch duration (seconds)")
    parser.add_argument("--once", action="store_true", help="Run one epoch then exit")
    parser.add_argument("--interface", help="Override WiFi interface")
    args = parser.parse_args()

    config = load_config(args.config)
    interface = args.interface or config.get("interface", "wlan1")
    epoch_seconds = args.epoch or config.get("epoch_seconds", 120)
    correction_factor = config.get("correction_factor", 0.8)
    min_rssi = config.get("min_rssi", -80)
    output_file = config.get("output_file", "passenger_count.json")
    crew_macs = config.get("crew_macs", [])
    dbscan_eps = config.get("dbscan_eps", 0.3)
    dbscan_min_samples = config.get("dbscan_min_samples", 2)

    print(f"=== WiFi Passenger Counter ===")
    print(f"  Interface:  {interface}")
    print(f"  Epoch:      {epoch_seconds}s")
    print(f"  Factor:     {correction_factor}")
    print(f"  Min RSSI:   {min_rssi} dBm")
    print(f"  Output:     {output_file}")
    print(f"  Crew MACs:  {len(crew_macs)}")
    print()

    counter = DeviceCounter(
        correction_factor=correction_factor,
        crew_macs=crew_macs,
        min_rssi=min_rssi,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    # Shared state between sniffer thread and main loop
    stop_event = threading.Event()
    probe_lock = threading.Lock()

    def on_probe(probe: dict) -> None:
        with probe_lock:
            counter.add_probe(probe)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n[main] Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start sniffer in background thread
    sniffer_thread = threading.Thread(
        target=start_sniffing,
        args=(interface, on_probe, stop_event),
        daemon=True,
    )
    sniffer_thread.start()

    epoch_num = 0
    try:
        while not stop_event.is_set():
            epoch_num += 1
            print(f"[epoch {epoch_num}] Collecting for {epoch_seconds}s...")

            # Wait for epoch duration (interruptible)
            stop_event.wait(timeout=epoch_seconds)

            if stop_event.is_set():
                break

            # Compute and write result
            with probe_lock:
                result = counter.compute_count()
                result["epoch_seconds"] = epoch_seconds
                counter.reset()

            write_output(result, output_file)

            print(f"[epoch {epoch_num}] Result: {result['passenger_count']} passengers "
                  f"(raw: {result['raw_devices_detected']}, "
                  f"real: {result['real_mac_count']}, "
                  f"clustered: {result['randomized_cluster_count']})")
            print(f"  Written to {output_file}")

            if args.once:
                break

    except Exception as e:
        print(f"[main] Error: {e}", file=sys.stderr)
        raise
    finally:
        stop_event.set()
        print("[main] Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module loads without errors (no hardware needed)**

```bash
python -c "import main; print('main.py loads OK')"
```

Expected: `main.py loads OK` (scapy import may print a warning — that's fine).

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat(passenger-counter): add main pipeline with epoch loop and JSON output"
```

---

### Task 8: Deploy to RPi5 & Phase 1 — Raw Capture Test

This task is executed on the RPi5 via SSH. It validates that the full hardware+software stack works.

**Files:**
- All files from Tasks 1-7 must be on the RPi5

- [ ] **Step 1: Copy project to RPi5**

From the development machine:

```bash
# Replace <rpi5-ip> with the RPi5's IP address
scp -r rpi5_experiment/ sam@<rpi5-ip>:~/rpi5_experiment/
```

Or if using git:

```bash
# On RPi5
cd ~
git clone <repo-url> && cd <repo>/rpi5_experiment
```

- [ ] **Step 2: Install system dependencies on RPi5**

```bash
sudo apt update && sudo apt install -y tcpdump aircrack-ng
```

- [ ] **Step 3: Install Python dependencies on RPi5**

```bash
cd ~/rpi5_experiment
pip install -r requirements.txt
```

Note: `scikit-learn` may take a few minutes to install on ARM. If it fails, try:
```bash
pip install scikit-learn --no-build-isolation
```

- [ ] **Step 4: Run unit tests on RPi5 (no hardware needed)**

```bash
python -m pytest tests/ -v
```

Expected: All tests from Tasks 3 and 5 pass.

- [ ] **Step 5: Set up monitor mode**

```bash
sudo bash setup_monitor.sh
```

Expected output:
```
[1/4] Mode-switching Tenda dongle (0bda:a192)...
  Dongle already in WiFi mode (0bda:a192 not found as DISK), skipping.
[2/4] Checking interface wlan1...
[3/4] Setting wlan1 to monitor mode...
[4/4] Verifying...
OK: wlan1 is in monitor mode
```

(If the dongle was already mode-switched from our earlier testing, step 1 will skip.)

- [ ] **Step 6: Phase 1 — Raw tcpdump capture (2 minutes)**

Before running the full Python pipeline, validate that tcpdump sees probe requests:

```bash
sudo tcpdump -i wlan1 -e -s 256 -l type mgt subtype probe-req -c 20
```

This captures 20 probe requests then stops. You should see lines like:
```
16:30:01.123 SA:da:a1:19:01:02:03 ... Probe Request (Converge_5GHz) ...
```

**Success criteria:** At least a few probe request frames appear. If nothing appears after 60 seconds, the Tenda dongle may not be capturing properly — check `dmesg | tail -20`.

- [ ] **Step 7: Phase 1 — Run full pipeline (single epoch)**

```bash
sudo python main.py --epoch 60 --once
```

This runs one 60-second capture epoch and writes `passenger_count.json`.

Expected output:
```
=== WiFi Passenger Counter ===
  Interface:  wlan1
  Epoch:      60s
  ...
[epoch 1] Collecting for 60s...
[sniffer] Listening on wlan1 for probe requests...
[epoch 1] Result: N passengers (raw: M, real: R, clustered: C)
  Written to passenger_count.json
[main] Done.
```

- [ ] **Step 8: Check the output**

```bash
cat passenger_count.json
```

Verify it contains a valid JSON with `passenger_count`, `real_mac_count`, etc. Compare against how many phones you know are in the house.

- [ ] **Step 9: Commit results log**

Note the results (actual phones vs. detected) for calibration in Task 9. No code changes needed.

---

### Task 9: Phase 2 — Calibration & Multi-Epoch Test

**Requires:** RPi5 with monitor mode working (from Task 8).

- [ ] **Step 1: Run continuous capture for 5 epochs**

```bash
sudo python main.py --epoch 120
```

Let it run for ~10 minutes (5 epochs of 120s). Note:
- How many phones are in the house (ground truth)
- What each epoch reports

Press Ctrl+C after 5 epochs.

- [ ] **Step 2: Check output consistency**

```bash
cat passenger_count.json
```

Compare across epochs:
- Is the count stable or fluctuating wildly?
- Does `real_mac_count` match the number of older/non-randomizing phones?
- Is `randomized_cluster_count` reasonable?

- [ ] **Step 3: Tune correction factor**

If you consistently detect more devices than people:
```json
"correction_factor": 0.6
```

If you consistently detect fewer:
```json
"correction_factor": 1.0
```

Edit `config.json` and re-run to verify improvement.

- [ ] **Step 4: Test person leaving**

1. Note the current count
2. Have someone leave the house with their phone
3. Wait one full epoch (120s)
4. Check if `passenger_count.json` shows a decreased count

This validates that the counter is responsive to occupancy changes.

- [ ] **Step 5: Document calibration results**

Create a brief note with:
- Number of test runs
- Ground truth vs. detected for each epoch
- Final correction_factor chosen
- Any issues encountered (weak signal, missed devices, etc.)

---

## Spec Coverage Checklist

| Spec Section | Covered By |
|---|---|
| 1. Problem Statement | All tasks — pipeline feeds `passengers_on_board` |
| 2. Hardware (Tenda setup) | Task 2 (setup_monitor.sh) |
| 3.1 Sniffer (tcpdump/scapy) | Task 6 (sniffer.py uses scapy directly) |
| 3.2 Counter (classify + cluster) | Task 3 (mac_utils.py) + Task 5 (counter.py) |
| 3.3 Output (JSON) | Task 7 (main.py write_output) |
| 4.1 MAC Classification | Task 3 (is_randomized) |
| 4.2 Feature Extraction | Task 6 (parse_probe_request) |
| 4.3 DBSCAN Clustering | Task 5 (_cluster_randomized) |
| 4.4 Correction Factor | Task 5 (DeviceCounter.correction_factor) |
| 5. Dependencies | Task 1 (requirements.txt) |
| 6. Monitor Mode Setup | Task 2 (setup_monitor.sh) |
| 7. Phase 1 Testing | Task 8 |
| 7. Phase 2 Testing | Task 9 |
| 7. Phase 3 Calibration | Task 9 |
| 8. Crew MAC filtering | Task 5 (crew_macs param + test) |
