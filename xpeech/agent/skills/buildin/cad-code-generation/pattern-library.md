# Common 3D Design Patterns Library

## Purpose
Load this knowledge when user mentions: mount, attach, snap, clip, hinge, slide, connect, fasten, grip, texture, hold, join, assemble.

This library represents patterns from thousands of real-world 3D designs—web 3D models, CAD libraries, manufactured products.

---

## Attachment Features

### Mounting Boss (raised pad with hole)

**Use:** Screw mounting on thin panels, PCB standoffs

```python
# Mounting boss: raised cylinder with centered hole
boss = (
    cq.Workplane("XY")
    .cylinder(6, 8)  # 6mm tall, 8mm diameter
    .faces(">Z").workplane().hole(3.4)  # M3 clearance
    .edges(">Z").fillet(1)
)
```

**Rules:**
- Boss diameter = 2× screw head diameter
- Boss height = screw head height + 2mm minimum
- Fillet base for stress relief

### Mounting Ear/Tab (side projection)

**Use:** Bracket attachment, side mounting

```python
# Mounting ear with screw hole
ear = (
    cq.Workplane("XY")
    .rect(20, 12).extrude(4)
    .faces(">Z").workplane().hole(4.5)  # M4 clearance
    .edges("|Z").fillet(2)  # Rounded corners
)
```

### Screw Boss (for self-tapping screws)

**Use:** Plastic housings, 3D printed enclosures

```python
# Self-tapping screw boss (no threads needed)
screw_boss = (
    cq.Workplane("XY")
    .cylinder(12, 5)  # tall, narrow
    .faces(">Z").workplane().hole(2.5)  # pilot hole for 3mm screw
)
```

**Rule:** Pilot hole = 80-85% of screw diameter for plastic

---

## Connection Interfaces

### Tongue and Groove

**Use:** Alignment + connection between panels

```python
# Tongue (male side)
tongue = cq.Workplane("XY").rect(20, 5).extrude(8)

# Groove (female side) - add 0.2mm clearance
groove_cut = cq.Workplane("XY").rect(20.4, 5.4).extrude(8.2)
panel_with_groove = panel.cut(groove_cut)
```

**Rule:** Clearance = 0.2-0.3mm for slide fit, 0.1mm for press fit

### Dovetail Slide

**Use:** Linear sliding connection, drawer slides

```python
# Dovetail profile (trapezoidal)
dovetail_male = (
    cq.Workplane("XY")
    .moveTo(-5, 0).lineTo(-3, 4).lineTo(3, 4).lineTo(5, 0).close()
    .extrude(50)
)
# Female: same profile + 0.3mm clearance on angles
```

### Slot and Tab

**Use:** Flat-pack assembly, laser-cut parts

```python
# Tab (protrudes from edge)
tab = cq.Workplane("XY").rect(10, material_thickness).extrude(5)

# Slot (cut into receiving part)
slot = cq.Workplane("XY").rect(10.2, material_thickness + 0.2).cutThruAll()
```

---

## Snap-Fit Features

### Cantilever Snap Hook

**Use:** Tool-less assembly, battery covers, access panels

```python
# Cantilever snap with retention lip
snap_hook = (
    cq.Workplane("XZ")
    .moveTo(0, 0)
    .lineTo(15, 0)  # arm length
    .lineTo(15, 2)  # hook base
    .lineTo(17, 2)  # retention lip
    .lineTo(17, 3)
    .lineTo(0, 3)
    .close()
    .extrude(5)
)
```

**Rules:**
- Arm length ≥ 5× thickness for flexibility
- Retention lip = 0.5-1mm overhang
- Draft angle on entry side for easy insertion

### Press-Fit Pin

**Use:** Permanent joining, bearing seats

```python
# Press-fit hole (interference fit)
hole_dia = 10  # nominal
press_fit_hole = hole_dia - 0.05  # 0.05mm interference

# Press-fit pin
press_fit_pin = hole_dia + 0.02  # slightly oversize
```

**Rule:** Interference = 0.02-0.05mm for plastic, 0.01-0.03mm for metal

---

## Surface Features

### Knurling (grip texture)

**Use:** Hand grips, adjustment knobs

```python
# Diamond knurl pattern (simplified as dimples)
knurled_grip = (
    cq.Workplane("XY")
    .cylinder(30, 12)
    .faces("|Z").workplane()
    .rarray(3, 3, 12, 8)  # circumferential pattern
    .hole(1.5, 1)  # shallow dimples
)
```

### Ribbed Texture

**Use:** Grip surfaces, ventilation appearance

```python
# Parallel ribs
ribbed_surface = (
    cq.Workplane("XY")
    .rect(50, 30).extrude(3)
    .faces(">Z").workplane()
    .rarray(4, 1, 12, 1)  # 12 ribs, 4mm spacing
    .rect(2, 30).extrude(2)  # rib cross-section
)
```

---

## Motion Features

### Simple Hinge (pin joint)

**Use:** Doors, lids, rotating connections

```python
# Hinge knuckle
knuckle = (
    cq.Workplane("XY")
    .cylinder(10, 5)  # barrel
    .faces("|Z").workplane().hole(3.2)  # pin clearance
)
# Alternate knuckles on each side, thread pin through
```

**Rule:** Pin clearance = pin diameter + 0.2mm for free rotation

### Living Hinge

**Use:** Flip-top lids, one-piece foldable parts (plastic only)

```python
# Living hinge (thin flexible section)
hinge_zone = cq.Workplane("XY").rect(50, 0.4).extrude(10)
# 0.3-0.5mm thickness for PP/PE, 0.6-0.8mm for PLA
```

**Rule:** Requires flexible material (PP, PE, TPU). PLA works but cracks after few cycles.

### Linear Slide Rail

**Use:** Drawers, CNC axes, adjustable mounts

```python
# T-slot rail profile
t_rail = (
    cq.Workplane("XY")
    .moveTo(-5, 0).lineTo(-5, 8).lineTo(-8, 8)
    .lineTo(-8, 10).lineTo(8, 10).lineTo(8, 8)
    .lineTo(5, 8).lineTo(5, 0).close()
    .extrude(100)
)
```

### Cam Follower

**Use:** Converting rotation to linear motion

```python
# Eccentric cam
cam_lobe = (
    cq.Workplane("XY")
    .circle(20).extrude(8)  # outer profile
    .faces(">Z").workplane().center(8, 0).hole(6)  # off-center axle
)
# Eccentricity = distance from center to axle = stroke / 2
```

---

## Practical Construction Features

### Cable/Wire Clip

**Use:** Wire management, cable routing

```python
# U-channel clip
cable_clip = (
    cq.Workplane("XY")
    .rect(8, 10).extrude(15)
    .faces(">Z").workplane(offset=-3)
    .rect(5, 12).cutBlind(-12)  # U-channel
)
```

### Retention Lip (undercut)

**Use:** Holding lids, preventing pull-out

```python
# Hole with retention lip
retained_hole = (
    cq.Workplane("XY")
    .circle(10).extrude(10)
    .faces(">Z").workplane(offset=-2)
    .circle(9).cutBlind(-6)  # undercut lip
)
```

### Chamfer Lead-In

**Use:** Assembly guide, easy part insertion

```python
# Hole with entry chamfer
guided_hole = (
    cq.Workplane("XY")
    .box(20, 20, 10)
    .faces(">Z").workplane().hole(8)
    .faces(">Z").edges("%Circle").chamfer(1.5)  # 45° lead-in
)
```

---

## Manufacturing-Aware Features

### Draft Angle (injection molding)

**Use:** Parts that will be molded, easier mold release

```python
# 3° draft on vertical walls
molded_part = cq.Workplane("XY").rect(50, 30).extrude(40, taper=3)
```

**Rule:** 1-3° draft per side, more for textured surfaces

### Support-Free Overhang (3D printing)

**Use:** Minimize supports, cleaner prints

```python
# Self-supporting overhang (45° max)
bridge_support = (
    cq.Workplane("XY")
    .rect(30, 20).extrude(10)
    .faces(">Z").workplane()
    .rect(20, 10).extrude(15, taper=-22.5)  # 45° angle
)
```

**Rule:** Max 45° overhang without supports (some printers handle 60°)

---

## Pattern Recognition Triggers

When user says → suggest this pattern:

| User Request | Pattern to Apply |
|--------------|------------------|
| "attach to wall/panel" | Mounting boss or ear |
| "needs to grip" | Knurling or ribbed texture |
| "connect two parts" | Tongue/groove, dovetail, or slot/tab |
| "no screws" | Snap-fit hooks |
| "needs to fold" | Living hinge |
| "rotates/swings" | Hinge knuckle |
| "slides" | T-slot rail or dovetail |
| "hold wires/cables" | Cable clip |
| "keep lid closed" | Retention lip or snap |
| "easy to insert" | Chamfer lead-in |
| "will be molded" | Add draft angles |
| "3D printed" | Check overhang angles, add chamfers |

---

## Fastener Selection Quick Guide

| Connection Need | Best Choice |
|-----------------|-------------|
| Permanent, high strength | Machine screw (M3-M8) |
| Into plastic boss | Self-tapping screw |
| Frequent access | Snap-fit |
| Permanent, no tools | Press-fit |
| Must be invisible | Adhesive + hidden pocket |
| Sliding connection | Dovetail or T-slot |
| Rotational joint | Hinge pin |
