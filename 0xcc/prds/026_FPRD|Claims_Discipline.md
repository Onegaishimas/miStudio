# Feature PRD: Claims Discipline & Evidence-Ladder Integration

**Document ID:** 026_FPRD|Claims_Discipline
**Version:** 1.0
**Status:** Planned
**Related:** PPRD §3.27 (row 27) · PADR IDL-44 · BRD-MIS-JSPACE-001 v0.3 BR-019, BR-020, BR-024 · consumed by every other J-space feature

---

## 1. Overview

### 1.1 Purpose

Give J-space evidence a place on the product's **existing** evidence ladder, and make it
structurally hard to describe that evidence in language it has not earned.

This feature ships **first among the remaining five** because the others consume it: dictionary
annotation (024), the intervention engine (025), the contracts (027) and the runtime handoff (028)
all emit claims, and they need one vocabulary rather than four.

### 1.2 Why this is a feature and not a style guide

A style guide is advice. The three failures this feature prevents are all cases where the code was
*correct* and the words around it were not:

- a readout rendered under a Jacobian label reads as a causal finding when it is a rung-0
  observation;
- "the model was not thinking about X" reads as coverage when the technique cannot see
  multi-token concepts or automatic computation at all;
- any copy implying a served model has subjective experience is a claim the source paper
  explicitly declines to make.

None is caught by a test of behaviour. Each is caught by a test of the words.

---

## 2. User Stories

- As a **reviewer**, I want every J-space claim to carry a rung from the same ladder circuits use,
  so I do not have to learn a second vocabulary.
- As an **alignment auditor**, I want the product to tell me what a negative result does NOT mean.
- As a **user reading any surface**, I want never to be invited to infer that the model has
  experiences.

---

## 3. Functional Requirements

### 3.1 J-space rungs on the existing ladder (BR-019)

J-space evidence SHALL use `EvidenceRung`, the ladder already implemented for circuits. At minimum:

| evidence | rung |
|---|---|
| a READOUT — a concept appearing at a position | lowest; explicitly NOT a causal claim |
| a PROBE THRESHOLD CROSSING | a readout with a stated threshold; still not causal |
| an INTERVENTION with a matched control | the first rung that may be described causally |

The ladder SHALL remain the **single** claims vocabulary. A second enum for J-space would be a
second ladder, and two ladders is no ladder.

### 3.2 A lower rung may never wear a higher rung's language (BR-019)

Surfaces SHALL NOT describe rung-0 or rung-1 evidence in intervention language. The existing
causal-language audit (`test_causal_language_audit.py`) SHALL be extended to cover every J-space
module, because that audit's own history is the argument: its `SURFACES` list was hand-maintained
at 5 files while 16 circuit modules went unaudited.

Coverage SHALL be **discovered**, not listed.

### 3.3 Absence is not evidence of absence (BR-020)

Any surface reporting that a concept was NOT found SHALL state that this is not evidence the
computation did not occur. Two mechanisms are named in the source and both SHALL be stated:
sufficiently automatic computation proceeds without engaging the workspace, and a concept with no
single-token name may not surface even when represented.

The product SHALL NOT present workspace evidence as **comprehensive coverage** of what a model is
doing.

### 3.4 No consciousness claims (BR-024)

Product copy, UI labels, documentation and export metadata SHALL NOT assert, imply, or invite the
inference that a served model has subjective experience or phenomenal consciousness.

This SHALL be enforced by an audit over shipped text, not by review alone. The source paper takes no
position and describes the implications as unclear; the product inherits that restraint.

### 3.5 The audit is a build gate

Violations SHALL fail the suite. A warning is advice, and this feature exists because advice does
not hold.

---

## 4. Non-Goals

- Changing the ladder's rung definitions, which circuits own.
- Policing language in code comments or 0xcc documents — the audit governs SHIPPED text: UI copy,
  API/MCP descriptions, exported metadata, and the manual.

---

## 5. Testing Requirements

- J-space evidence maps onto `EvidenceRung` with no new enum.
- The causal-language audit DISCOVERS J-space modules rather than listing them; adding a module
  without adding it to a list is still audited.
- A readout surface asserting causal language fails the suite.
- A "not found" surface without the absence caveat fails the suite.
- Consciousness-implying copy fails the suite, including in MCP tool descriptions and exported
  metadata.

---

## 6. Traceability

| Source | Covered by |
|---|---|
| BR-019 (rungs; no lower rung in higher language) | §3.1, §3.2 |
| BR-020 (absence is not evidence of absence; no coverage claim) | §3.3 |
| BR-024 (no consciousness claims) | §3.4 |
| PADR IDL-44 (J-space rungs on the existing ladder) | §3.1 |
