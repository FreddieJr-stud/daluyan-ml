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
