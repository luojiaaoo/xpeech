---
name: can-bus-analysis
short_description: CAN 与 DBC 报文分析
description: Analyze CAN bus traces - decode frames with DBC knowledge, interpret error frames, diagnose bus-off and communication problems, and assess bus load
---

# CAN Bus Analysis

## When to Use This Skill

Use this skill when the user wants to:
- Decode raw CAN frames (ID, DLC, payload) into signals and meaning
- Diagnose communication problems: missing messages, error frames, bus-off, CRC errors
- Understand DBC concepts (signals, byte order, scaling) or write signal extraction logic
- Assess bus load, cycle time violations, or timeout behavior
- Mentions keywords like: "CAN", "CAN bus", "DBC", "candump", "CANoe trace", "bus-off", "error frame", "cycle time", "CAN FD"

## Core Capabilities

- Parse and structure raw traces (candump, CANoe/CANalyzer logs, BLF/ASC exports)
- Decode signals: start bit, length, byte order (Intel/Motorola), scaling `physical = raw * factor + offset`
- Diagnose error patterns and map them to physical/protocol/application causes
- Evaluate timing: cycle times, jitter, timeouts, bus load estimation

## Decoding Method

### Step 1: Structure the trace
Normalize into a table: timestamp, channel, ID (hex), direction, DLC, payload bytes. Note trace format quirks (CANoe ASC uses hex IDs with `x` suffix for extended).

### Step 2: Decode signals (with or without DBC)
With a DBC: apply start bit/length/byte order/scaling per signal.
Without a DBC, reverse-engineer:
- Constant bytes → mux/counters/status fields
- Incrementing nibbles → alive/rolling counters (often 0–14 or 0–15)
- Last byte changing with content → checksum/CRC (XOR or CRC8 typical)
- Values scaling smoothly with vehicle behavior → physical signals (speed, temperature, torque)

**Byte order pitfall:** Motorola (big-endian) start bits count differently than Intel. If a decoded value jumps wildly with small physical changes, byte order is usually wrong.

### Step 3: Check communication health

| Symptom | Likely causes |
|---------|---------------|
| Error frames burst then silence | Bus-off: TX error counter ≥ 256 — short circuit, wrong bitrate, termination |
| Single node missing | Node not powered/initialized, wrong variant config, filtered by gateway |
| Cyclic message stops after event | Application/state machine issue, not bus — check preceding traffic |
| ACK errors on every frame | Only one active node on the bus (or listen-only tester) |
| Sporadic CRC/stuff errors | EMC, wiring, marginal termination (measure ~60 Ω across CAN-H/L) |
| Cycle time jitter under load | Priority inversion: low-ID (high-priority) traffic starving the message |

### Step 4: Timing and load
- Verify cycle times against spec (e.g., 10 ms ±10%); list violations with timestamps
- Estimate bus load: `sum(frame_bits) / bitrate` per second window; >80% classic CAN is a red flag
- Check timeout monitoring: a "missing" message may just exceed the receiver's timeout (typically 2.5–3× cycle time)

## Output Format

```markdown
# CAN TRACE ANALYSIS

## Overview
- Duration: Xs · Frames: N · Unique IDs: M · Error frames: E
- Estimated peak bus load: XX%

## Decoded traffic (excerpt)
| Time | ID | Name (if known) | Decoded signals |
|------|----|-----------------|-----------------|
| 1.203 | 0x1A0 | BrakeStatus | pressure=42.5 bar, plausibility=OK, alive=7 |

## Findings
1. [CRITICAL] 0x1A0 cycle time violation: spec 20 ms, observed gaps up to 180 ms at t=4.51–4.69s
2. [WARNING] Alive counter of 0x2C4 frozen at 0xB between t=2.0–2.4s → sender stall, receivers likely raised timeout DTCs
3. ...

## Root-cause hypothesis & next steps
```

## Common Pitfalls

1. **Blaming the bus for application faults** — a frozen signal with healthy frames is an application bug, not a CAN problem.
2. **Ignoring alive counters/CRCs (E2E)** — receivers discard frames failing E2E checks; the message is "on the bus" yet functionally missing.
3. **Termination measured with bus powered** — measure resistance with nodes unpowered; active transceivers skew readings.
4. **Extended vs standard ID confusion** — 0x18DAF110 (29-bit, ISO-TP diag) is not comparable with 11-bit IDs.
5. **CAN FD mixed assumptions** — bit-rate switching frames corrupt classic-CAN-only analyzers; confirm the trace tool supported FD.

## Key Principle

Separate the layers before concluding: **physical (wiring/termination) → protocol (errors/arbitration) → application (signal content/timing)**. Most confusing CAN problems are one layer masquerading as another.
