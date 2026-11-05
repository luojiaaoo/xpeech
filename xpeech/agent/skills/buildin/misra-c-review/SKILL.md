---
name: misra-c-review
short_description: MISRA C 代码与偏离审查
description: Review embedded C code against MISRA C guidelines - identify violations, explain the safety rationale, propose compliant fixes, and write proper deviations
---

# MISRA C Review

## When to Use This Skill

Use this skill when the user wants to:
- Review C code for MISRA C (2012/2023) style violations and undefined-behavior risks
- Understand *why* a MISRA rule exists and whether a deviation is justifiable
- Triage a static analysis report (prioritize, group, disposition findings)
- Write a formal deviation record that survives an audit
- Mentions keywords like: "MISRA", "MISRA C", "static analysis", "rule 10.x", "deviation", "PC-lint", "Polyspace", "coding guideline"

## Core Capabilities

- Spot violations in the highest-risk rule groups and explain the underlying hazard
- Propose minimal compliant rewrites (not rewrites that change behavior)
- Classify findings: fix / deviate / false positive — with justification
- Draft deviation records with scope, rationale, and risk mitigation

## High-Risk Rule Groups (review priority)

| Theme | Typical rules | Underlying hazard |
|-------|--------------|-------------------|
| Essential type model | 10.1–10.8 | Implicit conversions: silent truncation, sign surprises |
| Pointers & arrays | 18.x, 11.x | Out-of-bounds, misaligned/incompatible pointer casts |
| Side effects & evaluation order | 13.x | `x++` in expressions, volatile reads in logical operators |
| Control flow | 14.x, 15.x, 16.x | Unreachable code, fallthrough `switch`, single-exit debates |
| Preprocessor | 20.x | Unparenthesized macro args, `#undef` games |
| Standard library | 21.x | `malloc` family, `<stdio.h>` in production ECU code |
| Concurrency-adjacent | 1.3 + volatile use | Data races via ISR-shared data, "UB by omission" |

## Review Method

### Step 1: Establish context
Ask/determine: MISRA edition, mandatory/required/advisory policy, target compiler + word size, existing deviation permits. A finding's disposition depends on this context.

### Step 2: Review in order of hazard, not line number
1. **Undefined behavior candidates first** (rule 1.3 territory): uninitialized reads, out-of-bounds, sequence-point abuse, integer overflow paths
2. **Type model** (10.x): every implicit narrowing or signed/unsigned mix on arithmetic/comparison
3. **ISR/shared data**: anything accessed from interrupt and mainline without `volatile` + atomicity strategy
4. Everything else

### Step 3: Classify every finding
- **FIX** — real risk, compliant rewrite exists with identical behavior
- **DEVIATE** — rule violated intentionally for a documented reason (hardware access, generated code, performance-critical section)
- **FALSE POSITIVE** — tool limitation; document why, don't just suppress silently

### Step 4: Propose fixes that don't change semantics

```c
/* Violation (10.4): mixing signed/unsigned in comparison */
if (len > buffer_size - offset)        /* wraps if offset > buffer_size! */

/* Compliant AND fixes the underlying wrap hazard: */
if ((offset >= buffer_size) || (len > (buffer_size - offset)))
```

The rewrite must be reviewed for the *hazard*, not just tool silence — silencing the tool while keeping the bug is the worst outcome.

## Deviation Record Template

```markdown
## Deviation DEV-014
- **Rule:** MISRA C:2012 Rule 11.4 (required) — cast between pointer and integer
- **Scope:** mcal_reg.h, macros REG_READ/REG_WRITE only
- **Rationale:** Memory-mapped register access requires fixed-address casts; addresses
  are vendor-defined constants from the reference manual.
- **Risk & mitigation:** Misuse limited by confining casts to one header; addresses
  static-asserted against the vendor header; reviewed per hardware revision.
- **Approval:** [role, date]
```

A deviation without **scope confinement** and **mitigation** is a finding, not a deviation.

## Common Pitfalls

1. **Zero-violation worship** — 100% "clean" code full of casts inserted only to silence the tool is *less* safe. Casts hide exactly the conversions rule 10.x wants visible.
2. **Blanket suppressions** — file-level or rule-level suppression comments without records; auditors treat these as non-compliance.
3. **Fixing generated code by hand** — deviate at the generator boundary instead; hand edits evaporate on regeneration.
4. **Treating advisory rules as noise** — advisory ≠ ignorable; document a policy decision per rule, once, project-wide.
5. **Reviewing the report, not the code** — tools miss inter-procedural and ISR-concurrency hazards; a MISRA review is a *code* review guided by the report.

## Key Principle

MISRA is a proxy for one question: **"can this code surprise us on this compiler and this hardware?"** Every finding disposition — fix, deviate, false positive — must answer that question explicitly, in writing. Tool silence is not the goal; explained behavior is.
