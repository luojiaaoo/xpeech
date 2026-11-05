---
name: aspice-review
short_description: ASPICE 工作产品审查
description: Review automotive software work products against Automotive SPICE (A-SPICE) base practices for SWE.1-SWE.6 and produce assessment-style findings
---

# A-SPICE Work Product Review

## When to Use This Skill

Use this skill when the user wants to:
- Review a work product (requirements spec, architecture/design document, unit test spec, integration test spec, qualification test spec, traceability matrix) against Automotive SPICE
- Prepare for an A-SPICE assessment and find gaps before the assessor does
- Check whether test cases properly cover requirements (SWE.5/SWE.6)
- Understand what a specific base practice (e.g., "SWE.4 BP.2") requires
- Mentions keywords like: "A-SPICE", "ASPICE", "SWE.4", "SWE.5", "SWE.6", "base practice", "assessment", "work product review", "process compliance"

## Core Capabilities

- Identify the work product type and map it to the right A-SPICE process area
- Review content against the base practices of that process area
- Produce structured, evidence-based findings with severity ratings
- Rate practices on the NPLF scale (Not / Partially / Largely / Fully achieved)
- Suggest concrete remediation for each finding

## Process Area Map

Identify which process the work product belongs to before reviewing:

| Process | Scope | Typical work products |
|---------|-------|----------------------|
| SWE.1 | Software Requirements Analysis | SW requirements spec, req attributes, traceability to system reqs |
| SWE.2 | Software Architectural Design | Architecture doc, interfaces, dynamic behavior, resource budgets |
| SWE.3 | Software Detailed Design & Unit Construction | Detailed design (e.g., UML in Enterprise Architect), unit interfaces |
| SWE.4 | Software Unit Verification | Unit test specs/results, static analysis (MISRA) reports, coverage |
| SWE.5 | Software Integration & Integration Test | Integration test specs/results, interface tests, SiL/HiL runs |
| SWE.6 | Software Qualification Test | Qualification test specs/results against SW requirements |

## Review Method

### Step 1: Classify the work product
Determine the process area and the intended verification level. If ambiguous, ask the user (e.g., "Is this spec used for integration testing (SWE.5) or qualification (SWE.6)?").

### Step 2: Check universal work product hygiene
These apply to every work product and are the most common assessment findings:

- ✅ **Identification & versioning** — unique ID, version, status (draft/reviewed/released), author
- ✅ **Traceability** — bidirectional links to the upstream artifact (requirement ↔ test case, requirement ↔ design element)
- ✅ **Review evidence** — has the work product itself been reviewed? By whom, when, findings closed?
- ✅ **Status tracking** — every item has a lifecycle state (e.g., test case: specified → implemented → passed/failed)
- ✅ **Criteria defined** — entry/exit criteria, pass/fail criteria, environment described

### Step 3: Check process-specific base practices

**SWE.4 (Unit Verification) checklist:**
- Unit test strategy exists (incl. regression strategy) and covers the criteria for unit verification
- Static verification performed (MISRA compliance, static analysis findings dispositioned — not just "ran the tool")
- Test cases traceable to detailed design / unit requirements
- Structural coverage measured and target justified (statement/branch/MC-DC per criticality)
- Results recorded with verdict, environment, SW version under test

**SWE.5 (Integration Test) checklist:**
- Integration strategy matches the architecture's integration order (bottom-up/top-down/feature-based)
- Interface test cases exist for each architectural interface (signals, data ranges, timing, error behavior)
- Test cases traceable to architectural design elements
- Test environment specified (SiL/HiL, bus simulation, stubs/mocks) and representative
- Regression scope defined for re-integration after change

**SWE.6 (Qualification Test) checklist:**
- Every software requirement has at least one qualification test case (or a justified exception)
- Test cases specify: preconditions, stimulus, expected result, pass/fail criteria — measurable, not "works correctly"
- Negative tests / boundary conditions covered for safety-relevant requirements
- Test environment consistency: results state SW/HW version, configuration, environment
- Failed tests linked to problem reports with tracked resolution

### Step 4: Produce findings

## Output Format

```markdown
# A-SPICE REVIEW REPORT

**Work product:** [name, version]
**Process area:** [e.g., SWE.6 Software Qualification Test]
**Review scope:** [what was reviewed]

## Summary
| Metric | Value |
|--------|-------|
| Findings (Major) | X |
| Findings (Minor) | Y |
| Observations | Z |
| Provisional practice rating | N / P / L / F |

## Findings

### [MAJOR] F-01: Missing bidirectional traceability
- **Base practice:** SWE.6 BP.1 (Develop qualification test strategy incl. traceability)
- **Evidence:** Test cases TC-041..TC-098 have no requirement reference; requirements REQ-112, REQ-115 have no test case.
- **Risk:** Coverage claim cannot be demonstrated in assessment; potential untested requirements.
- **Recommendation:** Add requirement IDs to each test case; generate a coverage matrix; justify or close the two uncovered requirements.

### [MINOR] F-02: ...
```

Severity guide:
- **MAJOR** — would directly downgrade a base practice rating in an assessment (missing traceability, no review evidence, unmeasurable pass criteria)
- **MINOR** — weakens the practice but core evidence exists (inconsistent statuses, missing environment details)
- **OBSERVATION** — improvement suggestion, no rating impact

Rating guide (per practice, honest and conservative):
- **F (Fully, >85%)** — systematic, no significant weaknesses
- **L (Largely, 50–85%)** — done, with some weaknesses
- **P (Partially, 15–50%)** — approach exists but unpredictable/incomplete
- **N (Not, <15%)** — little or no evidence

## Common Findings Patterns (from real programs)

1. **"Excel traceability"** — a matrix exists but is manually maintained and out of sync with the test tool. Check freshness against actual test case IDs.
2. **Unmeasurable expected results** — "system behaves correctly" instead of concrete signal values/timing. Always flag.
3. **Static analysis theater** — MISRA report generated but violations neither fixed nor formally deviated. The deviation record IS the evidence.
4. **Coverage without justification** — "we measured 78% branch coverage" with no target or rationale for gaps.
5. **Review checkbox** — "reviewed by peer" with no findings recorded ever. Zero findings across many reviews is itself a finding.
6. **Environment drift** — results recorded against a different SW version than the release under assessment.

## Key Principle

An A-SPICE review is about **demonstrable evidence, not engineering quality alone**. Excellent software with undocumented verification rates poorly; the review's job is to make good engineering *visible and defensible*. Always tie every finding to a base practice and to concrete evidence in the work product — never generic advice.
