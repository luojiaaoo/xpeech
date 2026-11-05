# Probe Failure Triage

| Error | Meaning | First Checks |
| --- | --- | --- |
| `-EPROBE_DEFER` | Supplier missing/not ready | `devices_deferred`, regulators, clocks, resets |
| `-ENODEV` | Device absent or ID mismatch | compatible/HID/ID, bus visibility |
| `-EINVAL` | Bad property/resource | DT/ACPI resource parsing |
| `-ETIMEDOUT` | Device did not respond | power, reset, clock, bus, interrupt |
| `-ENOENT` | Missing property/file | firmware property or blob path |

Template:
```text
Driver:
Probe function:
Error:
First failing API:
Required resource:
Runtime firmware node:
Evidence:
Next check:
```
