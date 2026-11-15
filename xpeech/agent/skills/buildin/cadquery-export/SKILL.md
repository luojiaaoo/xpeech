---
name: cadquery-export
description: Export CadQuery models to STL/STEP/AMF/3MF and other formats for manufacturing, 3D printing, and visualization.
---

# CadQuery Export Skill

## Description
Export CadQuery models to various CAD formats for manufacturing, 3D printing, and visualization.

## Instructions

When the user asks to export a model, help them choose the right format and generate the export code.

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **STEP** | .step, .stp | Professional CAD interchange, CNC machining |
| **STL** | .stl | 3D printing, mesh-based workflows |
| **GLTF/GLB** | .gltf, .glb | Web visualization, AR/VR |
| **DXF** | .dxf | 2D drawings, laser cutting |
| **SVG** | .svg | 2D vector graphics |
| **BREP** | .brep | OpenCASCADE native format |
| **AMF** | .amf | Advanced 3D printing with colors |

### Export Commands

#### STEP Export (Recommended for CAD)
```python
import cadquery as cq
from cadquery import exporters

# ... your model code ...
result = cq.Workplane("XY").box(100, 50, 20)

# Export to STEP
exporters.export(result, "model.step")
```

#### STL Export (3D Printing)
```python
import cadquery as cq
from cadquery import exporters

result = cq.Workplane("XY").box(100, 50, 20)

# Export to STL (binary, smaller file)
exporters.export(result, "model.stl")

# Export to STL with tolerance control
exporters.export(result, "model.stl", tolerance=0.01, angularTolerance=0.1)
```

#### DXF Export (2D)
```python
import cadquery as cq
from cadquery import exporters

result = cq.Workplane("XY").box(100, 50, 20)

# Export top face as DXF
exporters.exportDXF(result, "top_view.dxf", direction=cq.Vector(0, 0, 1))

# Export front face as DXF
exporters.exportDXF(result, "front_view.dxf", direction=cq.Vector(0, 1, 0))
```

#### SVG Export
```python
import cadquery as cq
from cadquery import exporters

result = cq.Workplane("XY").box(100, 50, 20)

# Export as SVG (isometric view)
exporters.export(result, "model.svg", opt={
    "width": 800,
    "height": 600,
    "marginLeft": 10,
    "marginTop": 10,
    "showAxes": False,
    "projectionDir": (1, 1, 1),
    "strokeWidth": 0.5
})
```

### Format Selection Guide

**Choose STEP when:**
- Sharing with professional CAD software (SolidWorks, Fusion 360, FreeCAD)
- CNC machining
- You need exact geometry preserved

**Choose STL when:**
- 3D printing (FDM, SLA, SLS)
- Simple mesh visualization
- Game engines (with conversion)

**Choose GLTF/GLB when:**
- Web-based 3D viewers
- AR/VR applications
- Compact file size with colors

**Choose DXF when:**
- Laser cutting
- CNC routing
- 2D CAD drawings

### Quality Settings

#### STL Quality
```python
# Low quality (fast, large triangles)
exporters.export(result, "model.stl", tolerance=0.1, angularTolerance=0.5)

# Medium quality (default)
exporters.export(result, "model.stl", tolerance=0.01, angularTolerance=0.1)

# High quality (slow, smooth curves)
exporters.export(result, "model.stl", tolerance=0.001, angularTolerance=0.05)
```

### Batch Export
```python
import cadquery as cq
from cadquery import exporters

result = cq.Workplane("XY").box(100, 50, 20)

# Export all formats at once
exporters.export(result, "model.step")
exporters.export(result, "model.stl")
exporters.export(result, "model.svg")
```

## Output Format
When user asks to export, provide the appropriate export code and explain the format choice.
