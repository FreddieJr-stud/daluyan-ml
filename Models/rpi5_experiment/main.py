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
