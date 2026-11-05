---
name: iso26262-hara
short_description: HARA 危害分析辅助
description: Guide Hazard Analysis and Risk Assessment (HARA) per ISO 26262 - hazard identification, S/E/C classification, ASIL determination, and safety goal formulation
---

# ISO 26262 HARA Assistant

## When to Use This Skill

Use this skill when the user wants to:
- Perform or review a Hazard Analysis and Risk Assessment (HARA) for a vehicle function
- Classify Severity (S), Exposure (E), and Controllability (C) for a hazardous event
- Determine an ASIL and formulate safety goals
- Understand why a function is rated e.g. ASIL C vs ASIL B
- Mentions keywords like: "HARA", "ISO 26262", "ASIL", "hazard analysis", "safety goal", "severity exposure controllability", "functional safety"

## Core Capabilities

- Structure a HARA from item definition to safety goals
- Systematically derive hazards using guidewords and malfunction analysis
- Apply S/E/C classification with justifications (the part assessors challenge)
- Determine ASIL via the standard matrix
- Formulate well-formed safety goals with safe states and FTTI considerations

> ⚠️ HARA is a safety-critical, organization-owned activity. This skill structures the analysis and challenges reasoning — final classifications must be made and approved by qualified functional safety engineers.

## HARA Method

### Step 1: Item definition (prerequisite)
Confirm the analysis has a defined item: function description, boundaries, interfaces, operating modes. If missing, help the user write it first — a HARA without an item definition is unreviewable.

### Step 2: Hazard identification
Derive malfunctions systematically. For each function, apply guidewords:

| Guideword | Example (Steer-by-Wire torque request) |
|-----------|----------------------------------------|
| Loss of function | No steering assist available |
| Unintended activation | Self-steering without driver input |
| Too much / too high | Excessive assist torque |
| Too little / too low | Insufficient assist at high load |
| Wrong direction | Assist opposes driver input |
| Delayed | Assist arrives late (latency) |
| Stuck / frozen | Torque frozen at last value |

Each malfunction + driving situation = a **hazardous event**.

### Step 3: Situation analysis
For each hazardous event, define the operational situation: road type, vehicle speed, traffic, weather, vehicle state. Choose the **credibly worst reasonable** situation — not an absurd worst case, and not a convenient mild one.

### Step 4: Classify S, E, C

**Severity (harm to persons):**
| Class | Meaning | Reference points |
|-------|---------|------------------|
| S0 | No injuries | Damage only |
| S1 | Light/moderate injuries | Low-speed collisions |
| S2 | Severe, survival probable | Higher-speed collision with protection systems effective |
| S3 | Life-threatening/fatal | High-speed collisions, pedestrian impact, loss of vehicle control at speed |

**Exposure (probability of the situation):**
| Class | Meaning | Rule of thumb |
|-------|---------|---------------|
| E1 | Very low | Rare situations (< once a year for most drivers) |
| E2 | Low | A few times a year |
| E3 | Medium | Once a month or more for average driver |
| E4 | High | Almost every drive (highway driving, braking, steering) |

**Controllability (by driver or others):**
| Class | Meaning | Rule of thumb |
|-------|---------|---------------|
| C0 | Controllable in general | Distraction-level effect |
| C1 | Simply controllable | >99% of drivers can avoid harm |
| C2 | Normally controllable | 90–99% can avoid harm |
| C3 | Difficult / uncontrollable | <90% can avoid harm |

**Classification discipline (what assessors challenge):**
- Every S/E/C value needs a **written justification** — statistics, studies, or explicit engineering argument
- E is about the *situation*, not the malfunction (highway driving is E4 regardless of how rare the fault is)
- C must consider the *unwarned* driver; do not credit reactions that require training
- Do not "shop" situations to lower E — assessors compare against similar published HARAs

### Step 5: Determine ASIL

| | C1 | C2 | C3 |
|---|---|---|---|
| **S1, E1** | QM | QM | QM |
| **S1, E2** | QM | QM | QM |
| **S1, E3** | QM | QM | A |
| **S1, E4** | QM | A | B |
| **S2, E1** | QM | QM | QM |
| **S2, E2** | QM | QM | A |
| **S2, E3** | QM | A | B |
| **S2, E4** | A | B | C |
| **S3, E1** | QM | QM | A |
| **S3, E2** | QM | A | B |
| **S3, E3** | A | B | C |
| **S3, E4** | B | C | D |

### Step 6: Formulate safety goals

A well-formed safety goal:
- Is a **top-level safety requirement**, not a design solution ("Unintended self-steering shall be prevented" — not "add a torque limiter")
- Inherits the **highest ASIL** of the hazardous events it covers
- States or references a **safe state** (e.g., assist off + driver warning) where applicable
- Considers **FTTI** (fault tolerant time interval) when timing matters

## Output Format

```markdown
# HARA WORKSHEET

**Item:** [function]  |  **Version:** [x]  |  **Analysts:** [names]

| ID | Malfunction | Situation | Hazardous event | S | E | C | ASIL | Justification summary |
|----|-------------|-----------|-----------------|---|---|---|------|----------------------|
| HE-01 | Unintended assist torque | Highway, 130 km/h, dense traffic | Vehicle departs lane at speed | S3 | E4 | C3 | D | S3: high-speed loss of control; E4: highway driving is everyday; C3: unwarned driver cannot counter sudden torque |

## Safety Goals
| ID | Safety goal | ASIL | Safe state | Derived from |
|----|-------------|------|-----------|--------------|
| SG-01 | Unintended steering torque shall be prevented | D | Assist disabled + warning | HE-01 |
```

## Common HARA Mistakes

1. **Classifying the fault, not the situation** — E rates how often the *driving situation* occurs, not how often the fault happens.
2. **Optimistic controllability** — assuming an alert, trained driver. Use the average unwarned driver.
3. **Design solutions as safety goals** — "monitor the sensor" is a requirement, not a goal.
4. **One situation per malfunction** — the same malfunction in parking vs highway yields different ASILs; analyze both, keep the worst credible.
5. **Missing justifications** — a bare "S3 E4 C3" table row is not assessable evidence.
6. **QM treated as "no work"** — QM still means quality-managed development, not "ignore".

## Key Principle

The ASIL is only as defensible as its **justifications**. When in doubt between two classes, document the argument for both and escalate — a HARA's value is the reasoning trail, not the letter at the end.
