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
    - 2 real MACs counted directly
    - 3 randomized devices using 5 different MACs should cluster to 3
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
