---
name: algorithmic-poster-philosophy
description: 当用户提到海报、视觉设计、品牌视觉、排版系统、极简风格、设计哲学、美学方案，或说"帮我做张海报"时，必须触发此技能。该技能先构建一套设计哲学（Algorithmic Philosophy），再基于该哲学生成 SVG 海报文件并发送给用户。SVG 不得直接输出在聊天内容中。即使用户没有明确提到"哲学"或"系统"，只要涉及视觉设计或海报生成，都应触发。适用于品牌视觉、作品集封面、极简主义海报、AI 生成艺术、任何需要系统性美学表达的场景。
---

# Algorithmic Poster Philosophy Generator

## Overview

This skill transforms the task from:

- "generate a design"

into:

- "create a design philosophy and express it visually"

Philosophy always comes first.  
The visual output is only an expression of the system.

**Output tools**:

- Philosophy → written inline as structured markdown
- Poster → saved as `.svg` file and sent to the user
- Never paste raw SVG code directly in the chat

---

# Workflow

## STEP 1 — Create Algorithmic Philosophy

Before generating any visual output, construct a design philosophy inline as structured markdown.

Write 4–6 paragraphs covering the following:

### 1. Concept — Core Idea

Define the central aesthetic idea.

- Avoid vague artistic language
- Keep it abstract but actionable
- Must be translatable into design behavior

### 2. Visual Logic

Translate the concept into layout logic:

- Grid vs non-grid systems
- Information density: minimal / dense
- Whitespace strategy

### 3. System Behavior

Describe how the design behaves:

- Hierarchy: primary / secondary / tertiary
- Alignment vs deviation
- Rhythm, repetition, offset

### 4. Parametric Thinking

Convert design into variables:

- Font size ratios, e.g. title : subtitle : meta = 10 : 3 : 1
- Alignment rules, e.g. left-anchored with 1 intentional break
- Margin ratios, e.g. top margin = 15% of height
- Color count, recommended ≤ 2
- Number of active elements, recommended ≤ 5

Avoid result-based descriptions such as "cool style".  
Focus on controllable parameters.

### 5. Emergence

Explain what the system produces when executed:

- Visual feeling
- Perceived structure
- Emotional tone

---

## CRITICAL GUIDELINES — Philosophy Stage

- Avoid redundancy
- Every sentence must be convertible into design rules
- No purely decorative or empty language
- Think like both a designer and a system builder

---

# STEP 2 — Visual Expression — SVG Poster File

Based on the philosophy, generate a complete SVG poster.

**Important output rule:**

- Do not print raw SVG code in the chat
- Save the SVG content as a `.svg` file
- Send the `.svg` file to the user as the final poster artifact
- The chat response may include the philosophy and a download link / attachment reference only
- Never use inline SVG display as the final output

---

## SVG Poster Specifications

- Format: SVG file
- Filename: use a meaningful lowercase filename, e.g. `algorithmic_poster.svg`
- Aspect ratio: 3:4
- Recommended size: `viewBox="0 0 600 800"`
- Background: white or near-black only
- Font: load via `<style>@import url('https://fonts.googleapis.com/css2?family=...')</style>` inside SVG
- Output method: save to file, then send the file to the user

---

## Design Rules — Derived from Philosophy

### Information Strategy

- Reduce content automatically
- Keep only essential text
- Max 3 information blocks:
  - title
  - secondary
  - meta
- Remove anything that does not serve hierarchy

### Layout System

- Use strict alignment OR one intentional deviation
- Maintain strong vertical reading flow
- Use generous whitespace
- Let negative space carry visual weight

### Typography System

- Clear hierarchy:
  - title: large
  - secondary: medium
  - meta: small
- Limit to 1–2 typefaces
- Use refined spacing
- Use `letter-spacing` and line-height deliberately

### Color System

- Use black / white / grayscale OR low-saturation palette
- Maximum 2 colors excluding white / black
- Color must serve hierarchy, not decoration

### Element Control

- Minimal elements only
- Active elements on canvas ≤ 5
- No decorative icons
- No illustrations
- No ornaments
- Design relies on layout, typography, and space
- Thin rules or geometric lines are allowed only if they reinforce structure

---

# ADVANCED PRINCIPLES

## Concept Embedding

The user's concept must NOT be written explicitly on the poster.

Instead, embed it into:

- Structure
- Grid logic
- Spacing rhythm
- Typographic alignment
- Intentional deviation

Think:

> A hidden reference inside the system — felt, not read.

---

## Controlled Chaos

Variation is allowed, but only within constraints.

This is not randomness.  
This is designed, intentional variation operating inside a defined system.

---

## Craftsmanship Standard

The output must feel:

- Balanced and intentional
- Refined through many iterations
- Minimal
- Nothing added that could be removed

Like a poster that a senior designer spent a week on.

---

# Output Format

## 1 — Philosophy

Write inline as structured markdown.

Use 4–6 paragraphs following this structure:

1. Concept
2. Visual Logic
3. System Behavior
4. Parametric Thinking
5. Emergence

## 2 — Poster File

Generate a valid `.svg` file.

Rules:

- Do not paste the SVG source code into the chat
- Save the SVG as a file
- Send the file to the user
- The file must be directly usable in browsers, Figma, Illustrator, or vector editing tools
- The poster must be typography-driven
- The poster must not summarize the philosophy literally
- The poster must visually express the system

---

# Full Example

## Input

> 帮我做一张关于“沉默”主题的极简海报

## Philosophy Output

**Concept**: Silence is not the absence of sound — it is the space between signals. The design treats whitespace as the primary element, with text as intrusion.

**Visual Logic**: The layout uses a near-empty grid. One dominant typographic anchor is placed with deliberate restraint. Secondary information is pushed toward the margin. Density approaches zero.

**System Behavior**: The system allows only one primary element and one secondary element. The primary element is offset by exactly one grid unit. The secondary element is present but visually receding through scale and weight.

**Parametric Thinking**: Title size = 11% of canvas height. Margin = 18% of canvas width. Secondary text = title size × 0.18. Color count = 1. Active elements = 2.

**Emergence**: The viewer experiences stillness. The eye finds one point, then rests. Meaning accumulates in what is not shown.

## Poster Output

Save the generated SVG as:

`silence_poster.svg`

Then send the SVG file to the user.

Do not output the raw SVG code.

---

# Key Principle

Do NOT generate just a poster.

ALWAYS generate:

1. A philosophy system
2. A valid SVG poster file that expresses the system

The SVG must be saved as a file and sent to the user. It must never be directly pasted into the chat.