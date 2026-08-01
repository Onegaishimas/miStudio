# Feature PRD: Contracts & Two-Track Neuronpedia Conformance

**Document ID:** 027_FPRD|Contracts_And_Conformance
**Version:** 1.0 · **Status:** Planned
**Related:** PPRD §3.28 (row 28) · PADR IDL-45, IDL-46 · BRD-MIS-JSPACE-001 v0.3 BR-021..023

---

## 1. Purpose

Make J-space artifacts portable — **additively**, so nothing already shipped changes shape.

## 2. Additive only (BR-021)

New interchange kinds for: the J-lens artifact and its recipe provenance; the per-feature workspace
annotation; the position × layer readout record; and the runtime watchlist.

**Existing kinds SHALL NOT change.** `mistudio.cluster-definition/v1`, `cluster-bundle/v1` and
`circuit-definition/v1` keep working byte-for-byte, because miLLM consumes them today and a schema
change is a silent import failure at the far end.

This project has already been bitten here twice, and both lessons apply:

- re-vendoring a contract without updating miLLM's hand-written mirror **silently drops the new
  field** on import/re-export;
- a pydantic `alias` renames on **output** as well as input, which once republished a schema without
  its wire field and invalidated every exported document.

## 3. Two independent tracks (BR-022)

| track | what it is | shares with the other |
|---|---|---|
| **A — artifact supply** | a conformant on-disk lens directory the consumer MOUNTS | nothing but a name |
| **B — SAE workspace annotation** | export through the EXISTING feature/explanation upload path | nothing but a name |

They may ship in either order. **There is no J-lens ingestion API** — building one would mean
building a Neuronpedia feature that does not exist (explicit BRD non-goal).

## 4. Template lens (BR-023)

A path for **multi-token concepts**: generate contexts where a phrase is the natural continuation,
average the residual at the final position, mean-centre against a baseline set, and whiten by the
regularised inverse covariance.

Scoped as forward-passes-only. It may land as a fast-follow, but its **contract fields are day-one**
— adding a field later to a shipped kind is the change §2 forbids.

## 5. Non-Goals

- A J-lens ingestion API against Neuronpedia (no such upstream path).
- Changing any existing interchange kind.

## 6. Testing Requirements

- Every existing kind round-trips **unchanged** after the new kinds are added.
- New kinds carry a version and a recipe provenance block.
- A new field added to an EXISTING kind fails the suite.
- Track A's layout matches the consumer's expectations; Track B uses the existing upload path.
- Template-lens contract fields exist even though the compute path may not.
- No alias renames a field on output.

## 7. Traceability

| Source | Covered by |
|---|---|
| BR-021 (additive kinds; existing unchanged) | §2 |
| BR-022 (two independent tracks; no ingestion API) | §3 |
| BR-023 (template lens; contract fields day-one) | §4 |
| PADR IDL-45 (upstream wire format) / IDL-46 (mount, not upload) | §3 |
