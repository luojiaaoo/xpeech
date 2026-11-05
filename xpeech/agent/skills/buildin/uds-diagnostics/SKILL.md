---
name: uds-diagnostics
short_description: UDS 诊断日志分析
description: Analyze UDS (ISO 14229) diagnostic communication - decode services and NRCs, explain request/response sequences, and troubleshoot failed diagnostic flows
---

# UDS Diagnostics Analyzer

## When to Use This Skill

Use this skill when the user wants to:
- Decode a UDS request/response byte sequence or a diagnostic trace (CANoe log, candump, tester log)
- Understand why an ECU answered with a negative response (NRC)
- Design or debug a diagnostic sequence (session change, security access, flashing, DTC reading)
- Mentions keywords like: "UDS", "ISO 14229", "NRC", "0x7F", "diagnostic session", "security access", "DTC", "0x22", "ReadDataByIdentifier", "OBD tester"

## Core Capabilities

- Decode raw UDS bytes into service, sub-function, DID/DTC and meaning
- Interpret negative responses (0x7F + NRC) with likely root causes
- Reconstruct and validate multi-step diagnostic flows (session → security → operation)
- Explain ISO-TP framing effects visible in CAN traces (SF/FF/CF/FC)

## Decoding Basics

**Frame anatomy:**
```
Request:            [SID] [sub-function / parameters...]
Positive response:  [SID + 0x40] [parameters...]
Negative response:  [0x7F] [original SID] [NRC]
```

**Core services:**

| SID | Service | Common use |
|-----|---------|-----------|
| 0x10 | DiagnosticSessionControl | 01 default, 02 programming, 03 extended |
| 0x11 | ECUReset | 01 hard, 03 soft |
| 0x14 | ClearDiagnosticInformation | Clear DTCs (often FF FF FF = all) |
| 0x19 | ReadDTCInformation | 02 by status mask, 04 snapshot, 06 extended data |
| 0x22 | ReadDataByIdentifier | Read DID (2-byte identifier) |
| 0x27 | SecurityAccess | odd = requestSeed, even = sendKey |
| 0x28 | CommunicationControl | Disable/enable normal communication |
| 0x2E | WriteDataByIdentifier | Write DID |
| 0x31 | RoutineControl | 01 start, 02 stop, 03 requestResults |
| 0x34/0x36/0x37 | RequestDownload / TransferData / TransferExit | Flashing sequence |
| 0x3E | TesterPresent | 80 = suppress positive response |
| 0x85 | ControlDTCSetting | 01 on, 02 off (during flashing) |

**Most common NRCs:**

| NRC | Name | Typical root cause |
|-----|------|--------------------|
| 0x10 | generalReject | Internal error — check ECU logs |
| 0x11 | serviceNotSupported | Service not implemented in this ECU/variant |
| 0x12 | subFunctionNotSupported | Sub-function missing (e.g., session 03 not configured) |
| 0x13 | incorrectMessageLengthOrInvalidFormat | Byte count wrong — check DID length, ISO-TP framing |
| 0x22 | conditionsNotCorrect | Preconditions failed: wrong session, voltage, speed, engine state |
| 0x24 | requestSequenceError | Steps out of order (e.g., sendKey before requestSeed) |
| 0x31 | requestOutOfRange | DID/RID not supported or parameter out of bounds |
| 0x33 | securityAccessDenied | Operation needs unlocked security level |
| 0x35 | invalidKey | Key algorithm mismatch or wrong seed used |
| 0x36 | exceedNumberOfAttempts | Too many wrong keys — ECU lockout, wait for timeout |
| 0x72 | generalProgrammingFailure | Flash write failed — memory, driver, or data issue |
| 0x78 | requestCorrectlyReceived-ResponsePending | Not an error — ECU needs time; extend P2* timeout and wait |
| 0x7E | subFunctionNotSupportedInActiveSession | Right service, wrong session |
| 0x7F | serviceNotSupportedInActiveSession | Switch session first (0x10 03) |

## Analysis Method

### Step 1: Parse the bytes
For each frame, identify direction (request/response), SID, sub-function/DID, and payload. Present a decoded table before interpreting.

### Step 2: Reconstruct the flow
Diagnostic operations are stateful. Map the trace against the expected sequence:

```
Typical protected operation:
0x10 03 (extended session) → 0x27 01/02 (unlock) → operation (0x2E / 0x31 / ...)
+ 0x3E 80 (TesterPresent) keeping the session alive

Typical flashing sequence:
0x10 03 → 0x85 02 (DTC off) → 0x28 03 01 (comm off) → 0x10 02 (programming)
→ 0x27 unlock → 0x34 → 0x36 ×N → 0x37 → 0x31 01 (checksum routine) → 0x11 01 (reset)
```

### Step 3: Diagnose failures
For every NRC, check in order:
1. **Session** — is the active session correct for this service? (0x7E/0x7F/0x22 hints)
2. **Security** — is the required level unlocked? (0x33) Was the seed/key sequence correct? (0x24/0x35)
3. **Preconditions** — vehicle state (speed=0, engine off, voltage in range)? (0x22)
4. **Format** — request length vs DID definition, ISO-TP first-frame length field (0x13)
5. **Timing** — S3 session timeout (no TesterPresent → silent fallback to default session, subsequent 0x7F responses), P2/P2* handling of 0x78

### Step 4: Report

## Output Format

```markdown
# UDS TRACE ANALYSIS

## Decoded frames
| # | Dir | Raw | Service | Meaning |
|---|-----|-----|---------|---------|
| 1 | →ECU | 10 03 | DiagnosticSessionControl(extended) | Enter extended session |
| 2 | ←ECU | 50 03 00 32 01 F4 | Positive | P2=50ms, P2*=5000ms |
| 3 | →ECU | 2E F1 90 ... | WriteDataByIdentifier(F190) | Write VIN |
| 4 | ←ECU | 7F 2E 33 | **NRC 0x33 securityAccessDenied** | Security unlock missing |

## Root cause
Write to F190 requires security level 0x03; no 0x27 exchange precedes the write.

## Fix
Insert after frame 2:
  27 03 (requestSeed) → 27 04 [key] (sendKey), then retry 2E F1 90.
Keep session alive with 3E 80 every ~2s.
```

## Common Pitfalls

1. **Forgotten TesterPresent** — S3 timeout silently drops the ECU to default session; the *next* request fails mysteriously with session NRCs.
2. **0x78 treated as failure** — it's a "please wait"; abort only after P2* expires.
3. **Security lockout confusion** — after 0x36 (attempts exceeded), correct keys still fail until the delay timer expires; this is expected behavior.
4. **Functional vs physical addressing** — broadcast requests with sub-function suppression bits behave differently; check the addressing mode before blaming the ECU.
5. **ISO-TP length mismatches** — a "wrong" NRC 0x13 is often a transport-layer framing bug, not an application bug.

## Key Principle

UDS failures are almost never random: an NRC is the ECU telling you **which precondition you missed**. Decode first, reconstruct the state machine second, and only then judge — most "ECU bugs" are tester sequence bugs.
