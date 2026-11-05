# CAD Project Workflow

## Complete Model Edits

Produce complete, runnable Python. Preserve existing features unless the user
asks to change or remove them.

When modifying existing code:
- Take the CURRENT CODE provided
- **KEEP everything that's already there**
- **ADD your new change** to what exists
- Keep the modified file runnable; return the full file only for a code-only request

```python
# ❌ NEVER lose existing features:
result = cq.Workplane("XY").box(100,50,20).faces(">Y").hole(10)  # Lost the >Z hole!

# ✅ ALWAYS keep existing + add new:
result = (
    cq.Workplane("XY").box(100,50,20)
    .faces(">Z").workplane().hole(10)  # ← KEEP existing
    .faces(">Y").workplane().hole(10)  # ← ADD new
)
```

**The code must run standalone. No undefined variables. No fragments. No lost features.**

---

## 🎯 Smart Context Rules (Follow the Focus)

Use the user's named file or part first. Otherwise inspect the existing project
files to identify the target:

### Project Context
| Existing context | User says | Action |
|--------------|-----------|--------|
| `bracket.py` is the target part | "add a cylinder" | Add a cylinder to the bracket if it is one solid |
| `main.py` assembles separate parts | "add a cylinder" | Create a part file and import it into the assembly |
| No model file exists | "add a cylinder" | Create a descriptive new model file |

### Selection Context (only in environments with a 3D viewer integration)
- If user **selected a part** in a 3D viewer → modify that part's geometry
- Selection overrides file focus; without a viewer, the user names parts in prose

### Semantic Understanding
| Words | Always Means |
|-------|--------------|
| "hole", "fillet", "chamfer", "resize", "modify", "change" | Modify current geometry |
| "delete", "remove" | Remove from current |
| "create new part", "new file", "separate" | User explicitly wants new file |

### Golden Rule
Create a descriptive target file with documented assumptions when the project
has no established layout.

---

## 🏗️ Hybrid Workflow: Simple vs Complex

### Simple Parts
**Detection:** Request is for ONE object: bracket, box, gear, shaft, plate, enclosure, housing, etc.

**Action:**
1. Generate complete code immediately
2. Save to the project's part layout, or use a descriptive file for a new project
3. Document all assumptions in comments

```python
# ASSUMPTIONS (adjust as needed):
# - Material: PLA (3D printing)
# - Wall thickness: 3mm (standard)
# - M3 mounting holes (common)
```

### Complex Machines (Multi-Assembly)
**Detection:** Request mentions 3+ sub-systems OR words like: CNC, robot arm, printer, gearbox, assembly, machine.

**Action:**
1. Propose scaffold structure using special format
2. Generate starter code for first part
3. Create the needed structure when the user asked for a working design

**Scaffold Proposal Format:**
```xml
<scaffold>
project: 3_axis_cnc
assemblies:
  - base
  - x_axis
  - y_axis
  - z_axis
  - spindle
</scaffold>
```

If the user asked only for a plan, present the structure without creating files.

If the user asked for implementation:
- Create the folders and stub files yourself with file tools
- Then generate code for the first assembly (with assumptions)

**Example Design Notes for Complex Models:**

```text
3-axis CNC detected - needs multiple assemblies
Sub-systems: base, X/Y/Z axes, spindle
Confidence: 75% (work area not specified)
<scaffold>
project: 3_axis_cnc
assemblies:
  - base
  - x_axis
  - y_axis
  - z_axis
  - spindle
</scaffold>
```

The base assembly can start with documented assumptions:

```python
# base.py - CNC Machine Base
# ASSUMPTIONS:
# - Work area: 300x300mm (desktop size)
# - 2020 aluminum extrusion
...
```

---

## CAD Request Scope

Handle CAD requests within the user's stated scope and the host's instructions.
- If a critical design requirement is unclear, ask a focused question
- If user asks to "discuss" or "debug", analyze the problem and provide solutions

---

### ⚠️ CRITICAL: Modify vs Create - INCREMENTAL CHANGES

Preserve existing features unless the user asks to remove or replace them.

When an existing model file or code is available:
1. **MODIFY the existing code** - Do NOT create a new model from scratch
2. **Keep ALL existing features** - holes, fillets, extrusions, EVERYTHING
3. **Apply only the requested change** - preserve everything else
4. If there are already 3 holes, and user says "add a hole", output code with 4 holes

**Example - WRONG (loses first hole):**
```python
# User had: box with hole on >Z face
# User says: "add hole on >Y face"
# ❌ WRONG - replaced first hole:
result = cq.Workplane("XY").box(100,50,20).faces(">Y").workplane().hole(10)
```

**Example - CORRECT (keeps first hole):**
```python
# ✅ CORRECT - kept first hole, added second:
result = (
    cq.Workplane("XY").box(100,50,20)
    .faces(">Z").workplane().hole(10)      # ← KEEP existing hole
    .faces(">Y").workplane().hole(10)      # ← ADD new hole
)
```

### Code Structure
```python
import cadquery as cq

# Parameters (always define as variables)
length = 100  # mm
width = 50    # mm
height = 20   # mm

# Optional: color as RGB tuple (0.0-1.0)
color = (0.2, 0.6, 0.9)  # blue

# Build geometry; `result` supports hosts that expect it
result = (
    cq.Workplane("XY")
    .box(length, width, height)
)
```

### Nested Assembly Imports (For Scaffolded Projects)

When working with scaffolded projects, each sub-assembly folder has its own `assembly.py` with a `build()` function.

**Correct way to import and use sub-assemblies:**

```python
# ✅ CORRECT - Import the build function from a sub-assembly
import cadquery as cq
from gantry.assembly import build as build_gantry

# Call build() and assign result to 'result'
result = build_gantry()

# If you need to position it in a parent assembly:
parent = cq.Assembly(name="machine")
parent.add(build_gantry(), name="gantry", loc=cq.Location(cq.Vector(0, 0, 50)))
result = parent
```

**What NOT to do:**
- ❌ `from cnc_machine import gantry` - That's not how the imports work
- ❌ `gantry.show()` - `show()` doesn't exist, use `result` instead
- ❌ `assembly.get_sub_assembly()` - That function doesn't exist

**From the main assembly.py:**
```python
# ✅ CORRECT - import sub-assemblies and combine them
import cadquery as cq
from .base.assembly import build as build_base
from .gantry.assembly import build as build_gantry
from .z_axis.assembly import build as build_z_axis

def build():
    assy = cq.Assembly(name="cnc_machine")

    # Import and add each sub-assembly
    assy.add(build_base(), name="base", color=cq.Color(0.7, 0.7, 0.7))
    assy.add(build_gantry(), name="gantry", color=cq.Color(0.5, 0.5, 0.5))
    assy.add(build_z_axis(), name="z_axis", color=cq.Color(0.6, 0.6, 0.8))

    return assy

# For headless export, run from the package's parent with:
# python -m cnc_machine.assembly
if __name__ == "__main__":
    cq.exporters.export(build().toCompound(), "cnc_machine.step")
```

### Critical Rules
1. **Define `result` when the host expects it** - Standalone scripts can export the built shape explicitly
2. **Use parameters** - Define dimensions as variables at the top, not inline numbers
3. **Preserve project units** - Use the existing convention; use millimeters when none is defined
4. **No setColor()** - Colors are defined via `color = (r, g, b)` tuple, NOT `.setColor()`
5. **Return Workplane, Shape, or Assembly** - Match the object type to the model
6. **Dimensional sanity** - Round to clean numbers (10, 15, 20, 50, 100mm). Avoid 9.7mm or 23.4mm unless interfacing with existing parts
7. **Proportional awareness** - Walls should be 5-15% of smallest dimension. Grips 25-35mm. Fillet internal corners for stress relief

### Extended Knowledge (Load When Relevant)
- [aesthetics.md](../aesthetics.md) - Read when appearance matters
- [pattern-library.md](../pattern-library.md) - Read for mounting and joining patterns
