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
