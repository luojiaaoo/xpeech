# Aesthetic Design Guidelines

## Purpose
Load this knowledge when user mentions: style, modern, industrial, look, aesthetic, beautiful, clean, organic, minimalist, professional, sleek.

---

## Form and Function Harmony

Good design balances how something looks with how it works. A handle should feel right in the hand AND look intentional.

---

## Edge Treatment Guide

### When to Use Fillets (Rounded Edges)

**Best for:**
- Consumer products (friendly, safe, approachable)
- Anywhere hands touch (comfort + safety)
- Fluid/air flow paths (aerodynamic, hydrodynamic)
- Organic/natural aesthetic
- Stress relief at internal corners

**Radius guide:**
| Radius | Character |
|--------|-----------|
| 0.5-2mm | Sharp but safe (technical feel) |
| 3-5mm | Soft, friendly (consumer products) |
| 8-15mm | Pillow-soft (toys, handles) |
| Variable | Organic, flowing (sculpture-like) |

```python
# Consumer product feel - generous fillets
result = cq.Workplane("XY").box(80, 50, 20).edges().fillet(5)
```

### When to Use Chamfers (Angled Edges)

**Best for:**
- Industrial/mechanical aesthetic
- Assembly lead-ins (guide parts together)
- 3D printing (easier than fillets, no supports)
- Sharp, technical appearance
- Cost reduction vs. fillets in manufacturing

**Angle guide:**
| Chamfer | Character |
|---------|-----------|
| 45° × 1mm | Subtle, just breaks edge |
| 45° × 2-3mm | Visible, intentional |
| 30° | Aggressive, stealth look |
| 60° | Gentle transition |

```python
# Industrial tool feel - chamfers
result = cq.Workplane("XY").box(80, 50, 20).edges().chamfer(2)
```

### When to Leave Sharp Edges

- Mating/alignment surfaces (precision required)
- Internal hidden structures
- High-tech/precision aesthetic (deliberate sharpness)
- Where a seam or parting line is intentional

---

## Visual Balance

### Symmetry Rules

| Design Type | Symmetry Approach |
|-------------|-------------------|
| Functional parts | Full symmetry = easier to manufacture, predictable |
| Decorative pieces | Slight asymmetry = visual interest (within 20% variation) |
| Assemblies | Balance mass distribution (heavy elements low) |

### Proportion Guidelines

**Golden ratio (1.618:1)** — Naturally pleasing length-to-width
```python
# Golden ratio box
width = 100
length = width * 1.618  # ≈ 162mm
result = cq.Workplane("XY").box(length, width, 30)
```

**2:1 ratio** — Clean, functional simplicity

**Avoid exact 1:1** (squares/cubes) unless deliberate minimalism—they often feel static or boring.

### Transitions

- **Gradual radius changes:** R3 → R5 → R8, not R3 → R15 (jarring)
- **Taper angles:** < 15° for smooth visual flow
- **Avoid "lumpy" booleans:** Blend union joints with fillets
- **Continuous curves:** Use splines, not connected arcs

```python
# Smooth taper transition
result = cq.Workplane("XY").rect(50, 30).extrude(40, taper=5)
```

---

## Design Language Recognition

| Style | Characteristics | When to Apply |
|-------|-----------------|---------------|
| **Industrial** | Chamfers, exposed fasteners, dark colors, angular forms | Tools, machinery, shop equipment |
| **Consumer** | Fillets, hidden screws, white/bright colors, smooth | Household items, appliances, toys |
| **Organic** | Variable radii, flowing curves, natural colors | Artistic pieces, ergonomic handles |
| **Minimalist** | Clean lines, single material, monochrome, simple geometry | Modern furniture, tech accessories |
| **Mechanical** | Visible gears, bolts, knurling, metallic tones | Steampunk, maker aesthetic, display |
| **Professional** | Subtle details, matte finish, neutral colors | Office equipment, medical devices |

---

## Proportional Sense

### Scale Reasonableness Checks

Before finalizing, verify proportions make sense:

| Feature | Comfortable Range | Why |
|---------|------------------|-----|
| Wall thickness | 2-5mm (3D print), 1-3mm (metal) | Structural + visual weight |
| Grip diameter | 25-35mm | Adult hand comfort |
| Button/knob | 10-20mm | Finger-operable |
| Handle width | 100-120mm | Palm span |
| Fillet radius | 5-15% of smallest dimension | Visual proportion |

### Dimensional Sanity

Round to clean numbers for intentional, designed feel:
- ✅ 10, 15, 20, 25, 50, 100mm
- ❌ 9.7, 23.4, 47.8mm (looks accidental)

Exception: When interfacing with existing parts that have specific dimensions.

---

## Color Psychology for Models

**Functional color coding (for assemblies):**

| Color | RGB Tuple | Meaning |
|-------|-----------|---------|
| Gray | `(0.7, 0.7, 0.7)` | Structural, neutral, frame |
| Blue | `(0.2, 0.6, 0.9)` | Moving parts, mechanisms |
| Red | `(1.0, 0.2, 0.2)` | Caution, heat, power |
| Green | `(0.2, 0.8, 0.2)` | Safe, status OK, go |
| Yellow | `(1.0, 0.9, 0.0)` | Warning, attention needed |
| Orange | `(1.0, 0.5, 0.0)` | Energy, active element |
| Black | `(0.15, 0.15, 0.15)` | Premium, tech, accent |
| White | `(0.95, 0.95, 0.95)` | Clean, medical, consumer |

**Material suggestion colors:**

| Material Look | RGB Tuple |
|---------------|-----------|
| Aluminum | `(0.8, 0.8, 0.85)` |
| Steel | `(0.6, 0.6, 0.65)` |
| Brass | `(0.85, 0.75, 0.4)` |
| Copper | `(0.85, 0.55, 0.4)` |
| Black plastic | `(0.15, 0.15, 0.15)` |
| White plastic | `(0.95, 0.95, 0.92)` |

---

## Designer's Eye

Think like someone who has seen thousands of products:

1. **Every line should be intentional** — No random dimensions or accidental proportions
2. **Consistency within a design** — Same fillet radius throughout, same chamfer angle
3. **Visual hierarchy** — Important features stand out, secondary features recede
4. **Negative space matters** — Gaps and voids are part of the design
5. **Simplify until it breaks** — Remove everything non-essential, then add back only what's needed

**Test question:** "Would this look at home in an IKEA catalog? A machine shop? A medical device brochure?" — Match the aesthetic to the context.
