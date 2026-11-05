---
name: iso26262-fmea
short_description: FMEA 失效分析辅助
description: Guide FMEA for automotive systems - structure/function/failure analysis, severity-occurrence-detection rating, action prioritization, and safety-analysis alignment with ISO 26262
---

# Automotive FMEA Assistant

## When to Use This Skill

Use this skill when the user wants to:
- Create or review an FMEA (design/system/software FMEA) for an automotive component
- Derive failure modes systematically from functions
- Rate Severity / Occurrence / Detection and prioritize actions (AP or RPN)
- Align FMEA results with ISO 26262 safety analysis (safety goals, safety mechanisms)
- Mentions keywords like: "FMEA", "failure mode", "AIAG-VDA", "RPN", "action priority", "safety analysis", "DFMEA"

## Core Capabilities

- Structure the analysis in AIAG-VDA 7 steps (scoping → structure → function → failure → risk → optimization → documentation)
- Derive failure modes from function descriptions (not brainstorming randomly)
- Apply S/O/D ratings with consistent anchors and derive Action Priority
- Connect FMEA rows to safety mechanisms and diagnostic coverage claims

> ⚠️ FMEA for safety-relevant items supports (not replaces) ISO 26262 safety analyses. Confirmed ratings and actions belong to the responsible engineering team.

## Method (AIAG-VDA style)

### Step 1–2: Scope and structure
Define the analysis boundary and build the structure tree: system → subsystem → component. Every element gets its functions listed **as verb + noun + measurable condition** ("regulate cell voltage within X–Y V"), because failure modes are derived from functions.

### Step 3: Function → failure chain
For each function, derive failure modes mechanically:

| Failure category | Pattern | Example (function: "send brake pressure signal every 20 ms") |
|------------------|---------|--------------------------------------------------------------|
| Loss | No function | Signal not sent |
| Degradation | Partial/out-of-spec | Sent with wrong value / excessive jitter |
| Intermittent | Sporadic | Occasionally missing frames |
| Unintended | Function without demand | Signal sent during shutdown |
| Wrong timing | Too early/late | Sent after timeout window |

Then build the chain per row: **Failure Effect (FE) ← Failure Mode (FM) ← Failure Cause (FC)** across the structure levels (cause at lower level = mode of the element below).

### Step 4: Risk analysis — S, O, D

- **Severity (1–10):** rate the *effect* at the top level. Align with HARA where one exists: an effect violating an ASIL-rated safety goal ⇒ S = 9–10.
- **Occurrence (1–10):** rate the *cause* considering current prevention controls (design rules, derating, proven-in-use).
- **Detection (1–10):** rate the *detection control's* ability to catch the cause/mode **before the item reaches the customer** (for DFMEA) — runtime monitoring counts for safety mechanisms discussion, but classic D is about development-time detection. State which convention the team uses; mixing them is the most common inconsistency.

**Action Priority (AIAG-VDA):** derive H/M/L from the S-O-D combination table rather than multiplying RPN. If the team uses RPN, still flag any S ≥ 9 row regardless of RPN — high severity with "good" RPN is the classic blind spot.

### Step 5: Optimization
For every H (and safety-relevant M) row define actions with owner + due date:
- **Prevention action** → lowers O (design change, derating, redundancy)
- **Detection action** → lowers D (new test, analysis, monitoring)
Re-rate after actions close — an FMEA without re-rating is a to-do list, not a risk analysis.

## Output Format

```markdown
# FMEA WORKSHEET — [item], v[x]

| ID | Element | Function | Failure Mode | Effect (top) | S | Cause | Prevention | O | Detection | D | AP | Action |
|----|---------|----------|--------------|--------------|---|-------|-----------|---|-----------|---|----|--------|
| F-12 | CAN driver | Send brake msg 20ms | Message loss | Loss of brake indication to ESP | 9 | TX buffer overflow under burst | Buffer sizing rule R-33 | 3 | HiL bus-load stress test TL-88 | 3 | H | Add E2E timeout monitoring (owner, date) |

## Alignment with safety analysis
- Rows F-12, F-15 relate to SG-01 (ASIL C) — safety mechanism SM-04 (E2E + timeout) provides runtime detection; diagnostic coverage claim to be confirmed in FMEDA.

## Open actions
```

## Common Pitfalls

1. **Effects rated at the wrong level** — severity belongs to the end effect (vehicle/user), not the local symptom.
2. **Causes that are modes in disguise** — "software bug" is not a cause; drill to the mechanism (unprotected shared variable, missing range check).
3. **Detection credit for hopes** — "code review" as detection control only counts if it's planned, scoped, and evidenced.
4. **RPN arithmetic hiding S=10 rows** — always sort by S first, AP second, RPN last (if at all).
5. **FMEA/HARA vocabulary mixing** — S in FMEA (1–10) is not S in HARA (S0–S3); map explicitly when cross-referencing.
6. **One-shot FMEA** — not updated after field issues or design changes; a living FMEA cites problem reports as new causes.

## Key Principle

An FMEA earns its keep through **causal chains and closed actions**, not filled cells. Every high-severity row must answer: what prevents the cause, what detects it, and what evidence proves both — and whoever reads the row six months later must reach the same rating from the written rationale.
