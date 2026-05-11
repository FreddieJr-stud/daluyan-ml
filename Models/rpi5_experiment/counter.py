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
