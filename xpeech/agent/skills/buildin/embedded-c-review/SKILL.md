---
name: embedded-c-review
short_description: 嵌入式 C 代码审查
description: Review safety-critical embedded C code - ISR/shared-data races, volatile correctness, stack and timing hazards, defensive patterns for automotive ECUs
---

# Embedded C Safety Review

## When to Use This Skill

Use this skill when the user wants to:
- Review embedded C code for an ECU/microcontroller beyond style — concurrency, timing, memory, hardware interaction
- Check ISR safety, `volatile` usage, atomicity, or shared-data protection
- Harden code defensively for safety-critical contexts (plausibility checks, safe states)
- Mentions keywords like: "embedded C review", "ISR", "interrupt safety", "volatile", "race condition", "stack overflow", "watchdog", "bare metal", "RTOS task"

## Core Capabilities

- Find concurrency hazards between ISRs, tasks, and mainline code
- Judge `volatile`, atomic access, and critical-section correctness for the target word size
- Review hardware interaction (registers, timing, peripherals) and startup/shutdown paths
- Propose defensive-programming upgrades appropriate for ASIL-rated software

## Review Method

### Step 1: Establish the execution model
Before reading logic, determine: bare-metal loop or RTOS? Which ISRs exist and their priorities? Word size and whether 32-bit access is atomic on this core? Compiler and optimization level? A review without this context produces false confidence.

### Step 2: Hunt concurrency hazards (highest yield)

```c
/* HAZARD: 16/32-bit shared variable, no protection */
volatile uint32 g_speed;          /* written by CAN RX ISR */

void mainloop(void) {
  if (g_speed > LIMIT) { ... }    /* atomic read? only if bus width guarantees it */
}
```

Checklist:
- Every variable shared between ISR and mainline: `volatile` **and** atomic access (or critical section) — `volatile` alone does not provide atomicity
- Read-modify-write on shared data (`flags |= BIT`) inside/outside ISRs — needs protection even for 8-bit targets
- Multi-word data (structs, 64-bit, buffers) — consistent snapshot strategy (double buffer, sequence counter, disable IRQ)
- Critical sections: are they short, and are nesting/restore semantics correct (`save-disable / restore`, not blind enable)?
- ISR duration: any loops, floating-point, or blocking calls inside ISRs?

### Step 3: Memory and stack
- Stack usage: recursion, large locals (buffers on stack), worst-case call depth including ISR nesting
- No dynamic allocation after init (heap fragmentation is not testable); check hidden `malloc` in library calls
- Buffer arithmetic: every index derived from external data (CAN payload, DID length) bounds-checked **before** use
- `memcpy` lengths from wire data — the classic ECU vulnerability

### Step 4: Hardware interaction
- Register access through `volatile` pointers with correct width; read-back where the manual requires it
- Peripheral timing: delays implemented via timers, not instruction-count loops (break under optimization/clock change)
- Startup order: clocks → watchdog config → peripherals → interrupts last; nothing uses a peripheral before its init
- Watchdog: kicked from one place, in the main flow, **conditionally on system health** — an unconditional kick in a timer ISR keeps a dead system "alive"

### Step 5: Defensive posture (safety-critical additions)
- External inputs plausibility-checked (range, rate-of-change, freshness/alive counters)
- Every `switch` on external data has a `default` that handles the impossible value (log + safe value, not empty)
- Error paths lead somewhere defined: degraded mode / safe state, not silent `return`
- Timeouts on every wait-for-hardware loop, with a defined action on expiry

## Output Format

```markdown
# EMBEDDED C REVIEW

**Target:** [MCU, bare-metal/RTOS] · **Files:** [list] · **Context:** [ASIL, if any]

## Findings

### [CRITICAL] C-01: Non-atomic shared access
- **Where:** speed_monitor.c:47, `g_speed` (uint32) written in CAN_RX_ISR, read in 10ms task
- **Hazard:** Torn read on this 16-bit bus target → transient garbage value passes limit check
- **Fix:** Read under IRQ-disabled section or use double-read-compare pattern; document atomicity assumption next to the declaration

### [MAJOR] C-02: Watchdog kicked unconditionally in timer ISR ...
### [MINOR] C-03: ...

## What was NOT reviewed
[e.g., generated MCAL code, third-party stack] — state the boundary explicitly.
```

## Common Pitfalls (reviewer's own)

1. **Reviewing logic before execution model** — half of embedded bugs are invisible without knowing what preempts what.
2. **Trusting `volatile` as a lock** — it prevents caching/reordering by the compiler only; atomicity and hardware reordering are separate problems.
3. **Ignoring the failure path** — reviewers verify the happy path works; safety reviews verify the *unhappy* path is defined.
4. **Assuming the compiler is friendly** — check behavior at the shipped optimization level; `-O2` deletes naive delay loops and "unused" reads.
5. **Skipping generated/vendor code boundaries silently** — fine to exclude, but the report must say so.

## Key Principle

Embedded review asks a different question than desktop review: not "is the logic right?" but **"what happens when the interrupt fires between these two lines, the sensor lies, and the optimizer rewrites it?"** Review the code the hardware runs, not the code the author imagined.
