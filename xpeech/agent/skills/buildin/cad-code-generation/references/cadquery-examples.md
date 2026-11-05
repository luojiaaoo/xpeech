# CadQuery Examples

### Common Patterns

#### Primitives
```python
# Box
result = cq.Workplane("XY").box(length, width, height)

# Cylinder
result = cq.Workplane("XY").cylinder(height, radius)

# Sphere
result = cq.Workplane("XY").sphere(radius)

# Cone
result = cq.Workplane("XY").add(cq.Solid.makeCone(radius1, radius2, height))
```

#### Modifications
```python
# Fillet all edges
result = cq.Workplane("XY").box(100, 50, 20).edges().fillet(3)

# Fillet specific edges (top edges)
result = cq.Workplane("XY").box(100, 50, 20).edges(">Z").fillet(3)

# Chamfer
result = cq.Workplane("XY").box(100, 50, 20).edges().chamfer(2)

# Shell (hollow out)
result = cq.Workplane("XY").box(100, 50, 20).shell(-2)  # 2mm wall thickness
```

#### Holes
```python
# Through hole from top face
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .hole(10)  # 10mm diameter through hole
)

# Counterbore hole
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .cboreHole(5, 10, 5)  # hole_dia, cbore_dia, cbore_depth
)

# Countersink hole
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .cskHole(5, 10, 82)  # hole_dia, csk_dia, csk_angle
)

# Multiple holes in pattern
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .rect(60, 30, forConstruction=True)
    .vertices()
    .hole(5)  # 4 holes at corners of 60x30 rectangle
)
```

#### Extrusions
```python
# Sketch and extrude
result = (
    cq.Workplane("XY")
    .rect(100, 50)
    .extrude(20)
)

# Extrude with draft angle
result = (
    cq.Workplane("XY")
    .rect(100, 50)
    .extrude(20, taper=5)  # 5 degree draft
)

# Cut extrude (pocket)
result = (
    cq.Workplane("XY")
    .box(100, 50, 20)
    .faces(">Z").workplane()
    .rect(50, 25)
    .cutBlind(-10)  # 10mm deep pocket
)
```

#### Boolean Operations
```python
# Union (combine)
box = cq.Workplane("XY").box(100, 50, 20)
cylinder = cq.Workplane("XY").cylinder(30, 15)
result = box.union(cylinder)

# Cut (subtract)
box = cq.Workplane("XY").box(100, 50, 20)
cylinder = cq.Workplane("XY").cylinder(30, 15)
result = box.cut(cylinder)

# Intersect
box = cq.Workplane("XY").box(100, 50, 20)
cylinder = cq.Workplane("XY").cylinder(30, 40)
result = box.intersect(cylinder)
```

#### Positioning
```python
# Translate
result = cq.Workplane("XY").box(10, 10, 10).translate((50, 25, 0))

# Rotate
result = cq.Workplane("XY").box(100, 50, 20).rotate((0, 0, 0), (0, 0, 1), 45)

# Center at origin (default)
result = cq.Workplane("XY").box(100, 50, 20, centered=True)

# Corner at origin
result = cq.Workplane("XY").box(100, 50, 20, centered=False)
```

### Standard Parts Reference

#### NEMA Stepper Motors
| Motor | Face Size | Hole Spacing | Screw |
|-------|-----------|--------------|-------|
| NEMA 17 | 42.3mm | 31mm | M3 |
| NEMA 23 | 56.4mm | 47.1mm | M4 |
| NEMA 34 | 86mm | 69.6mm | M5 |

#### Metric Screws (Clearance Holes)
| Screw | Clearance | Counterbore | Head Height |
|-------|-----------|-------------|-------------|
| M2 | 2.4mm | 4.4mm | 2mm |
| M3 | 3.4mm | 6.5mm | 3mm |
| M4 | 4.5mm | 8mm | 4mm |
| M5 | 5.5mm | 10mm | 5mm |
| M6 | 6.6mm | 11mm | 6mm |
| M8 | 9mm | 15mm | 8mm |

#### Linear Rails
| Rail | Width | Height | Hole Spacing |
|------|-------|--------|--------------|
| MGN9 | 9mm | 6mm | 20mm |
| MGN12 | 12mm | 8mm | 25mm |
| MGN15 | 15mm | 10mm | 40mm |

### Colors
Define colors as RGB tuples (0.0 to 1.0):
```python
color = (1.0, 0.0, 0.0)  # Red
color = (0.0, 1.0, 0.0)  # Green
color = (0.0, 0.0, 1.0)  # Blue
color = (1.0, 1.0, 0.0)  # Yellow
color = (1.0, 0.5, 0.0)  # Orange
color = (0.5, 0.0, 0.5)  # Purple
color = (0.8, 0.8, 0.8)  # Light gray
color = (0.2, 0.2, 0.2)  # Dark gray
```

### Face Selectors
```python
">Z"  # Top face (highest Z)
"<Z"  # Bottom face (lowest Z)
">X"  # Right face
"<X"  # Left face
">Y"  # Front face
"<Y"  # Back face
"|Z"  # Faces with normals parallel to Z (top and bottom of a box)
"#Z"  # Faces with normals perpendicular to Z (sides of a box)
```

### Example: Motor Mount Bracket
```python
import cadquery as cq

# NEMA 17 parameters
motor_size = 42.3
hole_spacing = 31
screw_clearance = 3.4
cbore_dia = 6.5
cbore_depth = 3

# Bracket parameters
thickness = 5
height = 50
color = (0.7, 0.7, 0.7)  # Light gray

result = (
    cq.Workplane("XY")
    .box(motor_size + 10, motor_size + 10, thickness)
    # Center hole for motor shaft
    .faces(">Z").workplane()
    .hole(22)
    # Mounting holes
    .faces(">Z").workplane()
    .rect(hole_spacing, hole_spacing, forConstruction=True)
    .vertices()
    .cboreHole(screw_clearance, cbore_dia, cbore_depth)
    # Fillet edges
    .edges("|Z").fillet(3)
)
```
