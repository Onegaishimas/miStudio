# Feature PRD: Intervention Engine Extension

**Document ID:** 025_FPRD|Intervention_Engine
**Version:** 1.0 · **Status:** Planned
**Related:** PPRD §3.26 (row 26) · BRD-MIS-JSPACE-001 v0.3 BR-016..018 · consumes 026 (claims), 021 (artifacts)

---

## 1. Purpose

Extend the existing intervention engine with the primitives that operate on **lens directions**, so
J-space evidence can climb past rung 0.

This is the only feature in the arc that produces **causal** claims. Everything else observes.

## 2. A run without its control is INVALID (BR-018)

Every intervention run SHALL execute against a **size-matched random-direction control** at the same
layers and positions, and SHALL report control results alongside intervened results **by default**.

**A run reported without its control SHALL be treated as invalid** — not "unreviewed", not
"preliminary". Invalid.

The rationale is that every interpretation of an intervention rests on the comparison: an
intervention that moves the output tells you nothing until you know what moving a random direction of
the same size does. Making the control optional makes the finding optional.

This is the single most load-bearing requirement in the feature, and it is enforced structurally:
**a result object cannot be constructed without its control.**

## 3. Paired-run execution with clamping (BR-016)

The engine SHALL support a **clean reference pass** whose per-position results parameterise a second
**intervened** pass, and the ability to **hold specified lens coordinates at their clean-pass values**
at every position and layer.

Clamping is what makes a mediation analysis possible: without it, an intervention's effect and its
downstream consequences are indistinguishable.

## 4. Four primitives (BR-017)

| primitive | semantics |
|---|---|
| ADDITIVE | steer along a named token direction at given layers/positions |
| PROJECTIVE ABLATION | remove the activation's component along the direction |
| DYNAMIC TOP-K ABLATION | ablate the top-k J-space coordinates, excluding those in the clean pass |
| LENS-COORDINATE SWAP | replace one coordinate's value with another's |

Each SHALL record its semantics with the result. A run whose primitive is unrecorded cannot be
reproduced or compared.

**Scale-aware swap default (BR-017 v0.2):** coordinate swaps oversteer at small scale, so the
default number of layers selected SHALL adapt to model size rather than being a constant.

## 5. Rung (BR-019, via feature 026)

An intervention **with its control** is the first J-space evidence that may be described in causal
language. Without the control it is not evidence at all, so the rung question does not arise.

## 6. Non-Goals

- Replacing the circuits intervention engine — this EXTENDS it; the ladder and the machinery stay
  shared.
- Runtime intervention during serving — that is miLLM's plane.

## 7. Testing Requirements

- A result CANNOT be constructed without its control (structural, not a check).
- The control is size-matched and its seed recorded.
- Clamping holds the named coordinates at clean-pass values.
- Each primitive records its semantics.
- The swap's layer count varies with model size rather than being fixed.
- No causal language on a run reported without a control — it is invalid, not weakly-worded.

## 8. Traceability

| Source | Covered by |
|---|---|
| BR-016 (paired run with clamping) | §3 |
| BR-017 (four primitives; scale-aware swap) | §4 |
| BR-018 (control mandatory; no control ⇒ invalid) | §2 |
| BR-019 (rung) | §5 |
