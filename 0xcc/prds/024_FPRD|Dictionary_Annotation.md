# Feature PRD: Dictionary Annotation & Weight-Space Readouts

**Document ID:** 024_FPRD|Dictionary_Annotation
**Version:** 1.0 · **Status:** Planned
**Related:** PPRD §3.25 (row 25) · BRD-MIS-JSPACE-001 v0.3 BR-012..015 · consumes 026 (claims) and 021 (artifacts)

---

## 1. Purpose

Project every SAE feature's decoder direction through the lens and say what it looks like in
J-space — so a dictionary miStudio already holds gains a second, independent description.

This is where the two substrates meet. Everything else in the arc reads the model; this reads the
**dictionary**.

## 2. Two fields, not one (BR-012)

Workspace classification SHALL use **at least two independent fields**:

- a **GEOMETRIC** field — excess kurtosis of the projected vocabulary distribution;
- a **BEHAVIOURAL** field — whether the feature reads as motor (committing to output) or workspace
  (reportable content).

They are separate because **motor features share high lens-kurtosis with workspace features**. A
single field collapses the two and mislabels every motor feature as workspace. This is a locked
decision in the BRD, not an implementation preference.

Inapplicable is **absent**, never zero — the rule this arc has applied throughout.

## 3. Label disagreement is a queue, not a warning (BR-013)

Where a feature's existing auto-generated label and its lens readout diverge semantically, the
system SHALL raise a **LABEL DISAGREEMENT** flag and make it a **filterable, sortable, reviewable
queue**.

The failure mode is documented and consequential: example-driven labels name what a feature fires
ON, and the lens readout names what it pushes TOWARD. Those differ often enough that a silent
divergence is a systematic labelling error nobody sees.

A flag that is not a queue is a warning, and warnings are not read.

## 4. Validate against the published distribution (BR-014)

The annotation implementation SHALL be validated against the source paper's reported distributional
findings on the reference dictionary: that only a modest fraction of features are J-aligned once
motor features are excluded, and that non-aligned features are dominated by low-level syntactic and
bookkeeping roles.

A pipeline that labels *most* features workspace has a bug, and the distribution is how that bug is
visible. This is a **shape check on the output**, not a correctness proof.

## 5. Weight-space readouts (BR-015)

The system SHALL project arbitrary weight-space directions through the lens and present ranked
tokens as a component interpretation, for at minimum: SAE decoder directions; transcoder encoder
**and** decoder directions **as separate readouts**, so the input-to-output transformation a feature
performs is visible; and attention Q/K/V/O matrices.

Encoder and decoder as one readout hides the transformation, which is the thing worth seeing.

## 6. Non-Goals

- Training a workspace-aligned SAE (BRD non-goal, gated on Phase 0).
- Re-labelling features automatically from lens readouts — disagreement is surfaced for review, not
  auto-resolved. The lens is rung 0 and does not overrule a human label.

## 7. Testing Requirements

- Two independent fields; a motor-like feature is not classified workspace on kurtosis alone.
- Inapplicable is absent, never zero.
- Disagreement is filterable and sortable, not merely flagged.
- The distributional check FAILS on a fixture where most features are called workspace.
- Encoder and decoder readouts are separate and distinguishable.
- Every annotation carries rung 0 and no causal language (feature 026's audit covers the module).

## 8. Traceability

| Source | Covered by |
|---|---|
| BR-012 (two independent fields) | §2 |
| BR-013 (disagreement queue) | §3 |
| BR-014 (validate against the published distribution) | §4 |
| BR-015 (weight-space readouts, encoder/decoder separate) | §5 |
| BR-019 (rung discipline) | §7, via feature 026 |
