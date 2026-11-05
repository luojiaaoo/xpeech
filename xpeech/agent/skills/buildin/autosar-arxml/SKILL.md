---
name: autosar-arxml
short_description: ARXML 结构与引用分析
description: Read, explain, and sanity-check AUTOSAR ARXML - software components, ports, interfaces, signals, and BSW configuration concepts
---

# AUTOSAR ARXML Assistant

## When to Use This Skill

Use this skill when the user wants to:
- Understand what a piece of ARXML actually defines (SWCs, ports, interfaces, signals, runnables)
- Trace a signal path through an ARXML set (COM signal → PDU → frame, or port → interface → data element)
- Sanity-check ARXML for common integration errors before feeding it to Tresos / DaVinci / EB tools
- Understand AUTOSAR Classic concepts: RTE, BSW layers, MCAL, communication stack
- Mentions keywords like: "ARXML", "AUTOSAR", "SWC", "RTE", "port interface", "PDU", "Tresos", "DaVinci", "MCAL", "BSW"

## Core Capabilities

- Explain ARXML elements in plain engineering language
- Map the containment tree: which package defines what, and what references what
- Detect common inconsistencies (dangling references, mismatched interface versions, missing mappings)
- Explain the Classic Platform layered architecture and where a given config element lives

## Concept Map (Classic Platform)

```
Application SWCs  ← described by SWC ARXML (ports, runnables, IB data types)
      │  RTE      ← generated from SWC descriptions + ECU extract
──────┼─────────────────────────────
   BSW: COM stack (Com, PduR, CanIf, CanTp) · Services (Dem, Dcm, NvM, EcuM, BswM)
   MCAL (Can, Dio, Adc, Mcu, Port…) ← generated via Tresos from NXP/vendor modules
──────┴─────────────────────────────
   Hardware
```

**Key ARXML element families:**

| Element | What it defines |
|---------|-----------------|
| `AR-PACKAGE` | Namespace/folder for everything |
| `APPLICATION-SW-COMPONENT-TYPE` | An application SWC with its ports |
| `P-PORT` / `R-PORT` | Provided / required port on an SWC |
| `SENDER-RECEIVER-INTERFACE` | Data-oriented interface (data elements) |
| `CLIENT-SERVER-INTERFACE` | Operation-oriented interface |
| `RUNNABLE-ENTITY` + `RTE-EVENT` | Executable function + its trigger (timing, data-received) |
| `I-SIGNAL` / `I-SIGNAL-I-PDU` / `CAN-FRAME` | COM signal → PDU → frame chain |
| `SYSTEM-SIGNAL-MAPPING` | Port data element ↔ system signal link |
| `ECUC-MODULE-CONFIGURATION-VALUES` | BSW/MCAL module parameters (what Tresos edits) |

## Review Method

### Step 1: Identify scope
Is this an SWC description, a system/ECU extract, a communication matrix export, or ECUC (BSW config)? Each has different failure modes.

### Step 2: Walk the reference chains
ARXML errors are mostly broken chains. Verify end to end:
- **Signal path:** `I-SIGNAL` → `I-SIGNAL-TO-I-PDU-MAPPING` → `PDU-TO-FRAME-MAPPING` → frame → controller
- **Port path:** R-PORT → interface → data element → system signal mapping → I-SIGNAL
- **Runnable triggers:** every runnable referenced by at least one RTE event; timing events have periods

### Step 3: Check the classic integration errors
- **Dangling `DEST` references** — path points to a package/element that doesn't exist in the delivered set
- **Interface mismatch** — P-port and R-port referencing different interface versions/definitions with same short-name
- **Unmapped data elements** — port communication defined but no system signal mapping → RTE generates stubs, data never moves
- **Length/type drift** — I-SIGNAL length ≠ data type bit size (classic cause of shifted signals on the bus)
- **Init value gaps** — missing init values on sender-receiver comm (leads to tool-dependent defaults)
- **Short-name collisions** — same short-name at same package level (tool import silently overwrites)

### Step 4: Report

## Output Format

```markdown
# ARXML REVIEW

**Scope:** [SWC description / ECU extract / ECUC]
**Elements:** X SWCs · Y interfaces · Z signals

## Structure summary
[Containment tree of the relevant packages]

## Signal/port trace (requested item)
BrakePressure (data element)
 └─ mapped to SystemSignal /Signals/BrakePressure_Sig
     └─ I-SIGNAL BrakePressure_ISig (16 bit)
         └─ PDU Brake_IPdu (byte 2–3, little-endian)
             └─ CAN frame 0x1A0, 20 ms cyclic

## Findings
1. [ERROR] R-PORT /SWC/Brake/RP_VehSpeed references interface /IF/VehSpeed_IF_v2 — not present in delivery (dangling DEST)
2. [WARN] I-SIGNAL TorqueReq length 12 bit vs base type uint16 without compu-method range limit
```

## Common Pitfalls

1. **Editing generated ARXML by hand** — extracts regenerate; fix the source model or the generator config, not the output.
2. **Assuming short-name uniqueness globally** — uniqueness is per package level; always compare full paths.
3. **Ignoring `UUID` mismatches on re-import** — tools may duplicate rather than update elements.
4. **Reviewing one file of a multi-file delivery** — references routinely cross files; request the full set before judging "missing" elements.
5. **Confusing system constants with calibration** — `SwSystemconst` resolves at generation time; calibration parameters change at runtime/flash time.

## Key Principle

ARXML is a **reference graph, not a document**. Never evaluate an element in isolation — an ARXML review is graph traversal: does every reference resolve, and does the resolved chain make engineering sense end to end?
