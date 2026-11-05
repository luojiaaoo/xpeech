#!/usr/bin/env bash
set -euo pipefail
name="${1:-kernel-evidence-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$name"/{logs,config,firmware,sysfs}
uname -a > "$name/logs/uname.txt"
cat /proc/version > "$name/logs/proc-version.txt"
cat /proc/cmdline > "$name/logs/cmdline.txt"
dmesg -T > "$name/logs/dmesg.txt"
lsmod > "$name/logs/lsmod.txt"
zcat /proc/config.gz > "$name/config/running.config" 2>/dev/null || true
cat /sys/kernel/debug/devices_deferred > "$name/logs/devices_deferred.txt" 2>/dev/null || true
dtc -I fs -O dts /sys/firmware/devicetree/base > "$name/firmware/running.dts" 2>/dev/null || true
sudo acpidump -o "$name/firmware/acpi.dat" 2>/dev/null || true
lspci -nnvv > "$name/logs/lspci.txt" 2>&1 || true
lsusb -t > "$name/logs/lsusb-tree.txt" 2>&1 || true
i2cdetect -l > "$name/logs/i2c-buses.txt" 2>&1 || true
