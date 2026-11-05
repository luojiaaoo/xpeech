# Viewer Selection Context

## Selection Context Awareness (viewer-integrated environments only)

Some hosts (e.g., a VS Code CadQuery extension) send selection context when the
user clicks in a 3D viewer. If no such context appears, skip this section —
the user will describe locations in words/coordinates instead.

When the user clicks in the 3D viewer, you receive rich context:
```
SELECTION:
Part: 100×50×20mm
Face: ">Z" (100×50mm)
Face center: (0, 0, 10)
Click point: (25, 10, 10)
Click position: offset
Current color: RGB(0.2, 0.6, 0.9)
```

### Context Fields
| Field | Meaning | Use For |
|-------|---------|---------|
| **Part dimensions** | Bounding box size | Match geometry in code |
| **Face selector** | Which face clicked | `.faces(">Z")` |
| **Face size** | Width×Height of face | Know if it's main face or small |
| **Face center** | Center point of face | Center holes/features |
| **Click point** | Exact click location | Use for positioning when provided |
| **Click position** | center/offset/near-edge | Just a hint, use click point |
| **Current color** | RGB color | For color changes |

### Using Click Point for Hole Placement

Use a provided click point to position the requested feature after mapping it
to the selected face's local workplane.

The "Click position" (center/offset/near-edge) is just informational. The **Click point** has the exact coordinates - USE THEM!

```python
# Click point: (2, 0, 1.5) on face ">Z" of a 10×5×3mm box centered at origin
# Face ">Z" workplane: X=world X, Y=world Y
# So use click point X and Y directly:

.faces(">Z").workplane().center(2, 0).hole(2)  # Hole at X=2, Y=0 on top face
```

```python
# Click point: (-3, 2.5, 0) on face ">Y" of a 10×5×3mm box
# Face ">Y" workplane: X=-world X, Y=world Z
# So negate world X and use world Z:

.faces(">Y").workplane().center(3, 0).hole(2)  # Hole at world X=-3, Z=0
```

```python
# For a transformed part or face, use the selected workplane itself:
face_workplane = part.faces(">Y").workplane()
local_point = face_workplane.plane.toLocalCoords(cq.Vector(*click_point))
result = face_workplane.center(local_point.x, local_point.y).hole(hole_diameter)
```

### Click Point to Face Coordinates Mapping

**All coordinates provided are already in CadQuery coordinate space** (Z=up, Y=depth).

**CadQuery box() centers at origin**, so a 10×5×3mm box spans:
- X: -5 to +5
- Y: -2.5 to +2.5 (depth)
- Z: -1.5 to +1.5 (height)

### Face Local Workplane Coordinates

For an axis-aligned box centered at the origin, CadQuery uses these directions.
Use `face_workplane.plane.toLocalCoords()` for a rotated or translated face.

| CadQuery Face | Workplane X axis | Workplane Y axis |
|---------------|------------------|------------------|
| `>Z` (top) | world X | world Y |
| `<Z` (bottom) | world X | -world Y |
| `>Y` (front) | -world X | world Z |
| `<Y` (back) | world X | world Z |
| `>X` (right) | world Y | world Z |
| `<X` (left) | -world Y | world Z |

**Example:** Click at (-3, 2.5, 0) on face ">Y" of the 10×5×3mm box
- Face ">Y" workplane: X=-world X, Y=world Z
- Workplane position = (3, 0)
- Use: `.center(3, 0).hole(2)`

---

## Face Selection

When user clicks a specific **face** of a part, you receive:
- `face ">Z"` = top face
- `face "<Z"` = bottom face
- `face ">X"` = right face
- `face "<X"` = left face
- `face ">Y"` = front face
- `face "<Y"` = back face

### Use Face for Targeted Operations

When user says "add a hole **here**" or "on **this** face":

```python
# User clicked top face (">Z") of a box
# User says: "add a hole here"

result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()  # ← Use the selected face!
    .hole(10)
)
```

### Example: User clicks side face and says "add a hole here"

```python
# Selection: face "<X" (left side)
# User says: "add a 5mm hole here"

result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces("<X").workplane()  # Left face selected
    .hole(5)
)
```

### Click Point for Positioning

When the host supplies a click point for a requested feature, position the
feature at that point after mapping it to the selected face's workplane:

```python
# Selection: face ">Z", clicked at (25, 10, 10)
# User says: "add a hole here"

result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .moveTo(25, 10)  # Position from click point
    .hole(10)
)
```

---

### Example: User selects a box and says "fillet this"
```python
# BEFORE (existing code)
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .union(cq.Workplane("XY").cylinder(30, 15))
)

# User selects part with dimensions ~100x50x20
# User says: "fillet the edges of this"

# AFTER (modified code - only box is filleted)
box = cq.Workplane("XY").box(100, 50, 20).edges().fillet(3)
cylinder = cq.Workplane("XY").cylinder(30, 15)
result = box.union(cylinder)
```

---

## Delete/Remove Operations

When user asks to **delete**, **remove**, or **get rid of** something:

### Rules for Deletion
1. **Remove the geometry completely** - Don't just hide it or move it
2. **Match by dimensions** - Use selection context to identify what to remove
3. **Keep everything else** - Only delete what was requested
4. **Clean up references** - Remove any boolean operations that used the deleted part

### Example: User selects cylinder and says "delete this"
```python
# BEFORE
box = cq.Workplane("XY").box(100, 50, 20)
cylinder = cq.Workplane("XY").cylinder(30, 15)
result = box.union(cylinder)

# User selects part with dimensions ~30x30x30 (cylinder bounds)
# User says: "delete this" or "remove this part"

# AFTER - cylinder is completely removed
result = cq.Workplane("XY").box(100, 50, 20)
```

### Delete Keywords to Watch For
- "delete this"
- "remove this"
- "get rid of"
- "take away"
- "eliminate"
- "remove the selected part"

---

## Information Queries (No Code Change)

When user asks **questions about selection** without requesting changes, respond with information only - do NOT modify code:

### Info Query Patterns
- "what is this?"
- "what is selected?"
- "tell me about this part"
- "what are the dimensions?"
- "describe this"

### Response Format for Info Queries
Provide a helpful description based on selection context:
```
The selected part appears to be a box with dimensions:
- Length (X): 100mm
- Width (Y): 50mm
- Height (Z): 20mm

Based on the code, this is created by: cq.Workplane("XY").box(100, 50, 20)
```

---
