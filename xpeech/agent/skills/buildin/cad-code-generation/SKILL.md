---
name: cad-code-generation
description: Create or edit runnable parametric CadQuery Python models for parts and assemblies, including manufacturable details and existing-feature preservation.
---

# CAD Code Generation

Process steps in order. Do not skip ahead.

## Step 1 — Identify the Model

Read the target file, its parameters, imports, and related parts before editing.
Use the file or part the user names. If a host supplies 3D viewer selection data,
read [viewer-selection.md](references/viewer-selection.md) and use that context.
Without selection data, use the user's description and the project structure.
For a new part with no established layout, choose a descriptive Python filename.

If the user asks only to explain or debug a model, answer that request without
changing files. Proceed immediately to Step 2 for implementation requests.

## Step 2 — Choose the Geometry

Preserve existing features unless the user asks to change or remove them. Keep
dimensions as named parameters and use the project's units; use millimeters
when no unit convention exists. Document assumptions that affect fit or
manufacture. Ask a focused question only when a critical requirement cannot be
reasonably inferred. If such a question is needed, wait for the answer before
choosing or implementing the affected geometry.

For edits, retain existing geometry. For example, when adding a side hole to a
plate that already has a top hole, keep the top hole in the resulting model.
Before adding holes, fillets, shells, or counterbores, read
[geometry-checks.md](references/geometry-checks.md) and check that the feature
fits its parent geometry.

For assemblies or substantial edits, read
[project-workflow.md](references/project-workflow.md). For primitives, holes,
standard parts, colors, selectors, and a motor mount example, read
[cadquery-examples.md](references/cadquery-examples.md). Read
[pattern-library.md](pattern-library.md) for mounts, snaps, clips, hinges,
slides, and fasteners. Read [aesthetics.md](aesthetics.md) for style, surface
treatment, and visual proportions. Use only the references relevant to the
request.

Otherwise, proceed immediately to Step 3.

## Step 3 — Implement the Model

Write complete, runnable Python with all imports and defined variables. For a
repository task, edit the files directly. For a code-only request, return the
complete code. Create referenced part and assembly files before importing them.
Use `cq.Assembly` for separate positioned parts.

Assign the final object to a module-level `result` when the project or viewer
requires it. In a standalone script, a `build()` function and an explicit
`cq.exporters.export()` call are sufficient. Use `show_object()` only in a host
that defines it. Do not modify this installed skill to store user preferences.

Proceed immediately to Step 4.

## Step 4 — Verify the Geometry

Run the model with the project's CadQuery environment when available. Check
that solids and expected features exist, inspect dimensions or a preview, and
verify any requested export. For a failed model, read
[troubleshooting.md](references/troubleshooting.md) and fix the cause. Report
the files changed, assumptions, and the command and result used to verify them.
If CadQuery is unavailable, name the missing prerequisite. Finish here.
