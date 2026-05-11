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
