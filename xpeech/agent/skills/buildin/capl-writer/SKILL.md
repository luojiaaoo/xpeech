---
name: capl-writer
short_description: CAPL 编写与审查
description: Write and review CAPL scripts for Vector CANoe/CANalyzer - ECU simulation, message handling, diagnostics automation, and test node patterns
---

# CAPL Script Writer

## When to Use This Skill

Use this skill when the user wants to:
- Write a CAPL script (ECU simulation node, gateway, test node, diagnostic tester)
- Handle messages, timers, environment/system variables, or keyboard events in CANoe
- Automate a UDS diagnostic sequence in CAPL
- Debug typical CAPL problems (handlers not firing, timing drift, output going to wrong channel)
- Mentions keywords like: "CAPL", "CANoe", "CANalyzer", "on message", "on timer", "simulation node", "test node", "IG block alternative"

## Core Capabilities

- Generate idiomatic CAPL with correct event-driven structure
- Implement standard patterns: cyclic senders, gateways, counters/CRCs, fault injection, diagnostic sequences
- Review scripts for classic CAPL mistakes
- Explain CANoe integration concepts (measurement setup, node vs test node, sysvars)

## CAPL Mental Model

CAPL is **event-driven**: no main loop. Code runs only in event handlers:

```c
variables {
  msTimer tCyclic;              // millisecond timer
  message BrakeStatus msBrake;  // message object from DBC
  byte aliveCounter = 0;
}

on start {                       // measurement start
  setTimer(tCyclic, 20);
}

on timer tCyclic {               // fires once per setTimer!
  aliveCounter = (aliveCounter + 1) % 15;
  msBrake.AliveCounter = aliveCounter;          // DBC signal access
  msBrake.BrakePressure.phys = 42.5;            // physical value (uses DBC scaling)
  output(msBrake);
  setTimer(tCyclic, 20);         // re-arm — timers are one-shot
}

on message VehicleSpeed {        // reception handler (DBC name or ID)
  if (this.VehSpd.phys > 120.0) write("Overspeed: %.1f", this.VehSpd.phys);
}
```

**Core rules:**
- Timers are **one-shot** — re-arm inside the handler for cyclic behavior
- `this` refers to the triggering object inside a handler
- `.phys` applies DBC scaling; raw access does not — mixing them is the top signal bug
- Handlers must return fast: no busy-wait loops (they freeze the simulation)

## Standard Patterns

**Fault injection (corrupt CRC for E2E testing):**
```c
on message BrakeStatus {
  if (@sysvar::Test::CorruptCrc == 1) {
    message BrakeStatus m;
    m = this;
    m.Crc8 = (byte)(m.Crc8 ^ 0xFFU);   // deliberately wrong
    output(m);                          // note: original also passes unless node replaces it
  }
}
```

**UDS request/response sequence (raw on ISO-TP via diag layer preferred; simplified raw pattern):**
```c
// Prefer CANoe's diagnostic layer (diagRequest/diagResponse) with a CDD/ODX:
diagRequest EnterExtendedSession req;

on key 's' {
  diagSendRequest(req);
}

on diagResponse EnterExtendedSession {
  if (this.IsPositiveResponse()) write("Extended session active");
  else write("NRC: 0x%02X", this.GetRespCode());
}
```

**Gateway (forward + manipulate between channels):**
```c
on message CAN1.* {
  message CAN2.* fwd;
  fwd = this;
  fwd.can = 2;
  output(fwd);
}
```

## Review Checklist

- ✅ Every cyclic timer re-armed; no orphan `setTimer` typos (silent death of cyclic TX)
- ✅ `.phys` vs raw signal access consistent with intent
- ✅ Channel explicitly set when multiple channels exist (`msg.can = n`)
- ✅ No blocking constructs in handlers (`testWaitFor*` belongs in **test nodes**, not simulation nodes)
- ✅ Counters/CRCs updated *before* `output()`
- ✅ `on message *` used sparingly — it fires for everything and hides intent
- ✅ Sysvars/envvars used for test control instead of magic globals edited mid-measurement

## Common Pitfalls

1. **Busy-wait in handler** — freezes the node; use timers/state machines instead of loops-with-delay.
2. **Assuming timer periodicity** — `setTimer` is one-shot; forgetting re-arm makes "cyclic" messages send once.
3. **Simulation vs test node confusion** — `testWaitForMessage` etc. only exist in test nodes (Test Feature Set); simulation nodes need event-driven equivalents.
4. **Editing the received object** — `this` is read-only context; copy to a local `message` before modifying and re-outputting.
5. **DBC drift** — script compiled against an old DBC silently misplaces signals; recompile after every DBC update.

## Key Principle

Write CAPL as **small state machines wired to events**, never as sequential scripts. If the logic reads like "do A, wait, do B", it belongs in a test node with the test API — or restructured into timer-driven states.
