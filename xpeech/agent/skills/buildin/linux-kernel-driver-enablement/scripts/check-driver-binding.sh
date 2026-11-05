#!/usr/bin/env bash
set -euo pipefail
driver="${1:?driver name or module}"
pattern="${2:-$driver}"
echo "== modinfo =="
modinfo "$driver" 2>/dev/null || true
echo "== loaded modules =="
lsmod | grep -i "$driver" || true
echo "== dmesg matches =="
dmesg -T | grep -Ei "$driver|$pattern|probe|defer|firmware|regulator|clock|reset" || true
echo "== deferred =="
cat /sys/kernel/debug/devices_deferred 2>/dev/null || true
echo "== sysfs matches =="
find /sys/bus -maxdepth 5 -iname "*$driver*" -o -iname "*$pattern*" 2>/dev/null || true
