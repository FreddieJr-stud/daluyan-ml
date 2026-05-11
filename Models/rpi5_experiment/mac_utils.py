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
