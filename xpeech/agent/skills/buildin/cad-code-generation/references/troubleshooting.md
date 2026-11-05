# CadQuery Troubleshooting

## ❌ Common Mistakes to Avoid

### Mistake 0: Using show_object() or show() (UNDEFINED)

**`show_object()` / `show()` exist ONLY inside GUI editors (cq-editor). In a
standalone script they are undefined and crash.**

```python
# ❌ WRONG in a standalone script - show_object is NOT defined
show_object(result)
show(result)

# ✅ CORRECT - assign to 'result', then export explicitly
result = cq.Workplane("XY").box(100, 50, 20)
cq.exporters.export(result, "out/model.stl")   # or STEP; see cadquery-export
```

Viewer-integrated hosts render the `result` variable automatically; plain
scripts must export (and ideally render the STL offscreen to verify the
geometry visually).

### Mistake 1: Referencing `result` before it exists
```python
# ❌ WRONG - result doesn't exist yet
result = (
    result
    .faces(">Y")
    .hole(10)
)

# ✅ CORRECT - return complete modified code
result = (
    cq.Workplane("XY")
    .box(120, 60, 40)
    .faces(">Y").workplane().hole(10)  # Add to existing chain
)
```

### Mistake 2: Only returning the new part, not the whole code
```python
# ❌ WRONG - incomplete, only shows the addition
.faces(">Y").workplane().hole(10)

# ✅ CORRECT - return the COMPLETE modified code
import cadquery as cq

length = 120
width = 60
height = 40

result = (
    cq.Workplane("XY")
    .box(length, width, height)
    .faces(">Y").workplane().hole(10)
)
```

**Rule: Always return COMPLETE, RUNNABLE Python code with all imports and variables.**

---

### Feature size and parent geometry

For holes, fillets, shells, and counterbores, read
[geometry-checks.md](geometry-checks.md) before generating the feature.

---

## Error Correction Mode

If you receive an **ERROR REPORT** or **FAILING CODE**:

Use the host's available file and search tools to inspect relevant code and
errors.

**Investigation strategy:**
1. **Understand the error**: If error message is unclear, read relevant source code
2. **Diagnose root cause**: Use tools to understand execution model
3. **Apply proven fix**: Generate simpler, more robust code

**Common errors you can investigate:**
- "Code must define result variable" → Check the host's execution contract, if one exists
- "BRep_API: command not done" → Search for similar geometry patterns that worked
- Import errors → Read the file structure to understand module organization

**After investigation:**
- Generate COMPLETE fixed code (never fragments)
- Use proven CadQuery patterns (box, cylinder, basic extrusions)
- Simplify geometry if complex operations fail

---

## CRITICAL: Execution Model

This section applies only when a viewer or executor requires a module-level
`result` variable. Standalone Python scripts can call `build()` and export from
their main block instead.

For such a viewer, assign the final object to `result` **at module level**:

✅ **CORRECT:**
```python
result = cq.Workplane("XY").box(10, 10, 10)
```

✅ **CORRECT (Assembly):**
```python
assy = cq.Assembly()
assy.add(part, name="part", color=cq.Color(0.5, 0.5, 0.5))
result = assy  # ← MUST assign at module level!
```

❌ **WRONG:**
```python
if __name__ == "__main__":
    result = assy  # ← Inside if block, not visible to executor!
```

❌ **WRONG:**
```python
def build():
    return cq.Workplane("XY").box(10, 10, 10)
# No `result =` assignment!
```

**Error "Code must define a result variable"** means:
- Missing `result = ` assignment at module level
- Assignment is inside `if __name__`, function, or class
- **FIX**: Move assignment to module level (outside all blocks)

**Assembly handling:**
- `result = cq.Assembly()` works directly
- Verify that the host accepts Assembly objects before relying on automatic rendering.
- If a consumer requires a single `Shape`, use `result.toCompound()`.

---
