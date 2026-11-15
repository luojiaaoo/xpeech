---
name: linux-kernel-driver-enablement
description: Use when enabling, porting, debugging, or upstreaming Linux kernel support for hardware devices. Covers kernel baselines, driver binding, device tree/ACPI resources, probe failures, regulators/clocks/resets/pinctrl, dynamic debug, tracing, userspace handoff, evidence capture, pitfalls, and verification.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [linux, kernel, drivers, device-tree, acpi, embedded, bringup, debugging, upstream]
    related_skills: [hardware-firmware-linux-stack, board-bringup-lab, acpi-platform-enablement, pcie-debug-and-design]
---

# Linux Kernel Driver Enablement

## Overview

Use this skill to enable hardware in Linux once the board is sufficiently alive. It focuses on firmware description, driver match tables, resources, probe order, subsystem APIs, runtime behavior, and user-visible device nodes.

Core rule: prove hardware, power, reset, clock, bus visibility, and firmware description before changing driver logic.

## When to Use

Use when adding/enabling a Linux driver, debugging probe failure/deferral/missing device nodes, writing DT bindings/DTS/ACPI resources, porting vendor kernel support upstream, or investigating regulators/clocks/resets/pinctrl/interrupts/DMA/firmware blobs/udev handoff.

Do not use as primary when hardware has not passed basic bring-up, power/reset/clock are unmeasured, issue is pure userspace after driver known-good, or pure UEFI table architecture with no Linux driver work.

## Required Inputs

```text
Board:
SoC/chipset:
Kernel version/commit:
Vendor or upstream tree:
Target device/bus:
Driver path:
Firmware interface: DT / ACPI / platform data
Known-good kernel/board:
Symptom and exact failing command:
```

Minimum artifacts: full `dmesg`, kernel `.config`, `/proc/cmdline`, running DT or ACPI dump, driver path, binding/schema, relevant schematic or measurements, firmware version.

## Evidence Capture

```bash
export KEN_DIR="kernel-enablement/$(date +%Y%m%d-%H%M%S)-<device>"
mkdir -p "$KEN_DIR"/{logs,config,firmware,sysfs,patches,notes}
uname -a | tee "$KEN_DIR/logs/uname.txt"
cat /proc/version | tee "$KEN_DIR/logs/proc-version.txt"
cat /proc/cmdline | tee "$KEN_DIR/logs/cmdline.txt"
dmesg -T | tee "$KEN_DIR/logs/dmesg.txt"
zcat /proc/config.gz > "$KEN_DIR/config/running.config" 2>/dev/null || true
lsmod | tee "$KEN_DIR/logs/lsmod.txt"
```

In a kernel repo, record `git status --short`, `git rev-parse HEAD`, `git describe --tags --always --dirty`, and `git diff --stat`.

## Workflow

### 1. Can Linux see the device?

```bash
dmesg -T | grep -Ei 'probe|defer|fail|error|timeout|firmware|regulator|clock|reset|pinctrl'
cat /sys/kernel/debug/devices_deferred 2>/dev/null || true
find /sys/bus -maxdepth 4 -iname '*<device-or-driver>*' 2>/dev/null
```

Bus checks: `lspci -nnvv`, `lsusb -t`, `i2cdetect -l`, `ls /sys/bus/spi/devices`, platform sysfs. If the bus cannot see the device, return to board bring-up/cross-layer skill.

### 2. Confirm driver is built and matchable

```bash
modinfo <driver> 2>/dev/null || true
lsmod | grep -i <driver> || true
grep -R "CONFIG_<DRIVER>" .config arch/*/configs 2>/dev/null
git grep -n "MODULE_DEVICE_TABLE\|of_match_table\|acpi_match_table\|i2c_device_id\|spi_device_id\|pci_device_id\|usb_device_id" -- drivers/
```

Check Kconfig, dependencies, built-in vs module, match IDs, module autoload, and firmware blobs.

### 3. Validate firmware description

Device tree:
```bash
dtc -I fs -O dts /sys/firmware/devicetree/base > "$KEN_DIR/firmware/running.dts"
make dt_binding_check DT_SCHEMA_FILES=<binding>.yaml
make dtbs_check DT_SCHEMA_FILES=<binding>.yaml
```

ACPI:
```bash
sudo acpidump -o "$KEN_DIR/firmware/acpi.dat"
cd "$KEN_DIR/firmware" && iasl -d acpi.dat
grep -R "HID\|CID\|CRS\|DSD\|STA\|PR0\|PR3" -n *.dsl
```

Validate address/resources, IRQ polarity, GPIO mapping, clocks/resets/regulators, power domains, DMA/coherency, pinctrl, DT `status`, ACPI `_STA`/`_CRS`/`_DSD`.

### 4. Triage probe failures

Common meanings: `-EPROBE_DEFER` supplier missing; `-ENODEV` ID mismatch/absent; `-EINVAL` bad property/resource; timeout bus/clock/reset/power/interrupt; firmware load error packaging/path.

```bash
dmesg -T | grep -Ei 'defer|supplier|probe|regulator|clock|clk|reset|gpio|pinctrl|firmware|timeout'
cat /sys/kernel/debug/devices_deferred 2>/dev/null || true
sudo cat /sys/kernel/debug/regulator/regulator_summary 2>/dev/null
sudo cat /sys/kernel/debug/clk/clk_summary 2>/dev/null
sudo cat /sys/kernel/debug/pinctrl/*/pinmux-pins 2>/dev/null
```

### 5. Instrument

```bash
sudo mount -t debugfs none /sys/kernel/debug 2>/dev/null || true
sudo sh -c 'echo "file drivers/path/to/driver.c +p" > /sys/kernel/debug/dynamic_debug/control'
dmesg -w
sudo trace-cmd record -e module -e irq -e gpio -e regulator -e clk -e spi -e i2c sleep 10
sudo trace-cmd report | less
```

### 6. Make smallest correct change

Fix the boundary that is actually wrong: DT/ACPI supplier, compatible/HID, clock/reset, IRQ polarity, dependency modeling, firmware blob packaging, or minimal upstreamable driver change.

## Upstreaming Notes

Split patches by subsystem: binding, DTS, driver, defconfig. Binding precedes DTS usage. Avoid board hacks in generic drivers. Run `checkpatch.pl`, `get_maintainer.pl`, relevant build, `dt_binding_check`, and `dtbs_check`.

## Common Pitfalls

1. Editing driver before proving firmware resources.
2. Assuming source DTS is what booted.
3. Ignoring `-EPROBE_DEFER` and suppliers.
4. Adding sleeps instead of modeling dependencies.
5. Missing `MODULE_DEVICE_TABLE()`.
6. Wrong GPIO/IRQ polarity.
7. Blind I2C probing.
8. Confusing built-in and module behavior.
9. Debugging with stale firmware tables.
10. Overfitting to one board revision.

## Verification Checklist

- [ ] Hardware presence and basic electrical state proven or referenced
- [ ] Kernel commit, `.config`, full `dmesg`, runtime DT/ACPI captured
- [ ] Driver build mode and match table confirmed
- [ ] Regulators/clocks/resets/pinctrl/IRQs validated
- [ ] No unresolved probe deferrals
- [ ] Probe success visible in logs/sysfs
- [ ] Device node or subsystem interface appears and runtime operation tested
- [ ] Suspend/resume/runtime PM tested if relevant
- [ ] Userspace handoff checked if applicable
- [ ] Binding/schema/build/checkpatch/get_maintainer run for upstreamable patches

## References

See `references/driver-enable-flow.md`, `references/probe-failure-triage.md`, `references/dt-acpi-validation.md`, `references/subsystem-command-recipes.md`, templates, and scripts.
