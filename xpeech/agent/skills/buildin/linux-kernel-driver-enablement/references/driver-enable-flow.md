# Driver Enablement Flow

1. Prove hardware visibility.
2. Capture kernel baseline.
3. Confirm driver is built.
4. Confirm firmware description exists.
5. Confirm match table matches.
6. Check suppliers: regulators, clocks, resets, power domains.
7. Triage probe result.
8. Instrument with dynamic debug/tracing.
9. Make minimal fix.
10. Verify sysfs/device node/runtime behavior.
11. Package/upstream with checks.

Decision gates: no bus visibility -> board bring-up; no firmware node -> DT/ACPI; no driver match -> ID table; probe defer -> supplier; runtime failure after probe -> subsystem instrumentation.
