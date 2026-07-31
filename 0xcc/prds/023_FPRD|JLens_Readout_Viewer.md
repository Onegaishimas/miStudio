# Feature PRD: J-Lens Readout Viewer & Interpretability Framing

**Document ID:** 023_FPRD|JLens_Readout_Viewer
**Version:** 1.0
**Status:** Planned
**Related:** 000_PPRD §3.24 (row 24) · PADR IDL-40, IDL-44, IDL-45 · BRD-MIS-JSPACE-001 v0.3 BR-010, BR-011, BR-019, BR-020 · consumes Feature 023's substrate (doc chain 022) · reference implementation `0xcc/brds/JSpacePanel.jsx`

---

## 1. Overview

### 1.1 Purpose

Make *"what is this model poised to say here?"* a question a researcher answers by looking, not by
writing code.

### 1.2 The reference implementation is the specification

`0xcc/brds/JSpacePanel.jsx` is not a sketch. Per BR-010 it is **the interaction specification**, and
it is already built against the upstream lens wire format — its own header notes that swapping to a
live stream means replacing `buildFixture()` and nothing else. This feature ports it; it does not
redesign it.

Everything the mock does with fixture data, the panel does with a live logit-lens stream. Nothing
synthetic ships.

### 1.3 What makes this honest rather than decorative

Three of the mock's elements exist to stop the panel overclaiming, and they are requirements, not
chrome: the **evidence-rung card**, the **provenance strip**, and the **interpretability caveat**. A
readout is the weakest evidence this product produces and the easiest to describe in causal
language by accident.

---

## 2. User Stories

- As an **interpretability researcher**, I want a position × layer grid so I can see where a concept
  first appears and where it gives way to the output.
- As a **researcher tracking a concept**, I want to pin tokens and see their rank across layers, so
  "when does the model know this?" is a chart rather than a guess.
- As an **alignment auditor**, I want to find a decision position in a transcript and read what was
  present there.
- As **any user**, I want to be told when a readout is expected to be uninterpretable, so I do not
  mistake noise for a finding — or a finding for nothing.
- As a **reviewer**, I want to see at a glance which evidence rung a claim sits on.

---

## 3. Functional Requirements

### 3.1 Nav placement

The panel SHALL appear in the primary navigation as **"J-Lens"**, positioned **immediately before
Steering**. Ordering is array position; there is no sort key.

### 3.2 Position × layer grid (BR-010)

The panel SHALL present the prompt with per-token selection; a grid of the top-ranked token at each
(position, layer) cell; hover revealing the full top-k at that cell; and click-to-pin from the hover
list.

When tokens are pinned the grid SHALL become a **rank heatmap** over those tokens rather than a
top-1 display.

### 3.3 Layer axis is model-derived

The layer axis SHALL come from the stream's `layers_by_type`. The panel SHALL NOT assume a layer
count or spacing. The reference implementation hardcodes 21 layers at 0, 5, …, 100; real models have
16 (LFM2) or 26 (gemma-2-2b) — this must follow the stream.

### 3.4 Lens modes and honest disablement (BR-019)

The panel SHALL offer **Jacobian**, **Logit** and **Diff** modes. Until a validated Jacobian
artifact exists for the selected model, Jacobian and Diff SHALL be **visibly disabled with a stated
reason**.

Logit-lens data SHALL NEVER render under a Jacobian label. A disabled control with an explanation is
required; silently showing logit data in Jacobian mode is a defect.

### 3.5 Bands are earned, not defaulted (BR-002)

Sensory / workspace / motor band shading SHALL be drawn **only** from a band report computed for the
selected model. Absent a report, the panel SHALL draw no bands and SHALL say why.

The reference implementation's `L40 / L90` boundaries are the source paper's Sonnet-4.5 figures.
Porting them to another model is prohibited, and the panel SHALL make porting them impossible by
construction — there is no default band constant.

### 3.6 Rank-vs-layer trajectories

For pinned tokens, the panel SHALL chart rank against layer at the selected position, with lower
rank displayed as stronger, and SHALL not connect across layers where a token is absent from the
top-k.

### 3.7 Interpretability framing (BR-011)

The panel SHALL state, adjacent to the readout, that:

- readouts are limited to concepts with **single-token names**;
- a non-trivial fraction of workspace-layer readouts **resist interpretation**, which may be
  averaging noise, a multi-token concept, or genuine content we cannot yet name;
- **absence of a signal is not evidence of absence** of the underlying computation (BR-020).

Early-layer readouts SHALL be marked as *expected* to be uninterpretable rather than presented as
findings.

### 3.8 Evidence rung (BR-019)

Every readout SHALL carry its rung. A readout is **rung 0** and is explicitly **not a causal claim**.
The panel SHALL name what would raise the rung rather than merely labelling the current one.

### 3.9 Provenance strip (BR-007)

The panel SHALL display the artifact identity and recipe behind the current readout — target layer,
attention-gradient treatment, position scope, aggregation, corpus, prompt count, sequence length,
dtype — and for the logit lens SHALL state that no artifact is involved.

---

## 4. User Interface

Per the reference implementation: header with model chip and lens-mode toggle; prompt strip; the
grid with band legend; hover detail; trajectory chart; right rail carrying pinned tokens, the
by-layer readout list, and the rung card; provenance strip in the footer.

Dark/light dual-mode per house convention — the mock is dark-only and must be adapted, not pasted.

---

## 5. API / Integration

Consumes `POST /api/v1/jlens/readout` (doc chain 022) and renders its `meta` / `token` / `slice`
messages directly. No adaptation layer: the same component renders a Neuronpedia stream.

---

## 6. Data / Types

Wire-format types shared with the backend schema: `LensMetaMessage`, `LensTokenMessage`,
`LensTypeSlice`, `LayerApplicability`. Declared once in the frontend client and imported by the
panel.

---

## 7. Dependencies

- Feature 023 substrate (doc chain 022) for the stream.
- `recharts` (2.15.4) and `lucide-react` (0.468.0) — both already dependencies.
- `frontend/src/config/brand.ts` for styling tokens.
- Band report (doc chain 021) for §3.5 — optional; absent means no bands.

---

## 8. Success Criteria

- "J-Lens" appears immediately before Steering and renders a live readout.
- The grid follows the model's real layer count for at least two architectures.
- Jacobian and Diff are disabled with a reason until an artifact exists.
- No band shading appears without a band report.
- No fixture constant is reachable from the shipped bundle.
- The rung card, caveat text and provenance strip are present on every readout.

---

## 9. Non-Goals

- Fitting or validating artifacts (doc chain 021).
- Interventions, annotation, watchlists.
- Server-side Diff computation — Diff is a client-side comparison of two slices, matching upstream.
- Shipping the mock's fixture data.

---

## 10. Testing Requirements

- Nav entry renders **before** Steering (ordering is array-positional and load-bearing).
- Layer axis follows `layers_by_type` for differing layer counts.
- Jacobian/Diff disabled with a reason when no artifact; enabled when one exists.
- No band shading without a report.
- Rung card, caveat and provenance strip present.
- `grep -r "FIXTURES\|buildFixture" frontend/src` returns nothing.
- Panel does not unmount on background refresh (the house regression, per
  `ExtractionsPanel.nounmount.test.tsx`).

---

## 11. Traceability

| Source | Covered by |
|---|---|
| PPRD §3.24 row 24 | §1–§9 |
| PADR IDL-40 (logit-first; single substitution point) | §3.4 |
| PADR IDL-44 (rungs on the existing ladder) | §3.8 |
| PADR IDL-45 (upstream wire format, no adaptation layer) | §5, §6 |
| BR-010 (position × layer panel, lens modes, bands) | §3.2, §3.3, §3.4, §3.5, §3.6 |
| BR-011 (interpretability caveat) | §3.7 |
| BR-019 (rung discipline; no lower rung in higher language) | §3.4, §3.8 |
| BR-020 (absence is not evidence of absence) | §3.7 |
| BR-002 (bands never ported) | §3.5 |
| BR-007 (recipe provenance surfaced) | §3.9 |
