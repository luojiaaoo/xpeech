# Geometry Checks

Read this before adding a hole, fillet, shell, or counterbore. These checks are
engineering heuristics; run CadQuery to confirm that the resulting solid is
valid.

- A hole or counterbore must fit on its selected face with the intended edge
  clearance. Compare its diameter and position with the face width and height.
  Part thickness controls through depth, not the maximum hole diameter on a
  top face.
- A fillet radius must fit the selected edges and neighboring faces. Start
  smaller when the available material is narrow, and verify the kernel result.
- A shell thickness must leave the intended walls and openings. Check local
  narrow features as well as the overall bounding box.
- Verify the result after booleans. An oversized cut may create an unintended
  open side; a fillet or shell can fail when the topology is too tight.

For example, a centered 50 mm hole on a 100 × 50 mm top face reaches both side
edges. A 10 mm hole leaves room for surrounding material:

```python
import cadquery as cq

length = 100
width = 50
thickness = 20
hole_diameter = 10

result = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z").workplane()
    .hole(hole_diameter)
)
```

Check the actual design's specified clearance rather than treating the example
as a universal minimum. See the [CadQuery API reference](https://cadquery.readthedocs.io/en/latest/apireference.html)
for the operations' parameters.
