"""Charging time estimator for RPi5 deployment.

Uses an empirical SOC-vs-charging-power lookup curve (built from historical
telemetry) to estimate time-to-target and time-to-full.

Dependencies: json, math only (no numpy/xgboost).
"""
from __future__ import annotations

import json
import math
from pathlib import Path


class ChargingEstimator:
    """Estimates charging time from an empirical power curve."""

    def __init__(self, curve_path: str):
        with open(curve_path) as f:
            curve = json.load(f)

        self.capacity_kwh: float = curve["battery_capacity_kwh"]
        self._bins: list[int] = curve["bins"]
        self._median_kw: list[float] = curve["median_kw"]

        # Build SOC -> power lookup (integer keys)
        self._power_at_soc: dict[int, float] = dict(
            zip(self._bins, self._median_kw)
        )

        self._max_soc_bin: int = max(self._bins)
        self._min_soc_bin: int = min(self._bins)

    def get_power_at_soc(self, soc: float) -> float:
        """Get estimated charging power (kW) at a given SOC%.

        Uses linear interpolation between known bins.
        """
        soc_int = int(math.floor(soc))

        # Direct lookup
        if soc_int in self._power_at_soc:
            return self._power_at_soc[soc_int]

        # Interpolate between nearest known bins
        lower = None
        upper = None
        for b in self._bins:
            if b <= soc_int:
                lower = b
            if b > soc_int and upper is None:
                upper = b

        if lower is not None and upper is not None:
            frac = (soc - lower) / (upper - lower)
            p_low = self._power_at_soc[lower]
            p_high = self._power_at_soc[upper]
            return p_low + frac * (p_high - p_low)

        if lower is not None:
            return self._power_at_soc[lower]
        if upper is not None:
            return self._power_at_soc[upper]

        # Fallback: use the lowest known power
        return min(self._median_kw)

    def estimate_minutes(self, current_soc: float, target_soc: float) -> float:
        """Estimate minutes to charge from current_soc to target_soc.

        Walks 1% bins, using the empirical power at each bin.
        Handles partial first bin proportionally.
        """
        if current_soc >= target_soc:
            return 0.0

        energy_per_percent = self.capacity_kwh / 100.0  # 1.6 kWh per 1%
        total_minutes = 0.0

        # Partial first bin
        first_bin_ceil = math.ceil(current_soc)
        if first_bin_ceil > current_soc and first_bin_ceil <= target_soc:
            frac = first_bin_ceil - current_soc
            power = self.get_power_at_soc(current_soc)
            if power > 0:
                total_minutes += (frac * energy_per_percent / power) * 60.0

        # Full bins
        start_bin = int(math.ceil(current_soc))
        end_bin = int(math.floor(target_soc))

        for soc_bin in range(start_bin, end_bin):
            power = self.get_power_at_soc(float(soc_bin))
            if power > 0:
                total_minutes += (energy_per_percent / power) * 60.0

        # Partial last bin
        last_frac = target_soc - end_bin
        if last_frac > 0 and end_bin < target_soc:
            power = self.get_power_at_soc(float(end_bin))
            if power > 0:
                total_minutes += (last_frac * energy_per_percent / power) * 60.0

        return round(total_minutes, 1)

    def estimate_to_full(self, current_soc: float) -> float:
        """Estimate minutes to 100% SOC."""
        return self.estimate_minutes(current_soc, 100.0)

    def estimate_to_target(self, current_soc: float, target_soc: float = 80.0) -> float:
        """Estimate minutes to a configurable target SOC."""
        return self.estimate_minutes(current_soc, target_soc)
