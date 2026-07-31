# Task List: J-Lens Readout Viewer & Interpretability Framing

**Document ID:** 023_FTASKS|JLens_Readout_Viewer
**Version:** 1.0
**Status:** Planned
**Related:** 023_FPRD · 023_FTDD · 023_FTID · PPRD §3.24 · PADR IDL-40, IDL-44, IDL-45

| Phase | Delivers | Gates |
|---|---|---|
| 1 | Wire types + API client + store | every component |
| 2 | Presentational components | the panel |
| 3 | Panel composition | nav wiring |
| 4 | Nav entry + chunking | user-visible delivery |
| 5 | Framing + honesty surfaces | shipping at all (BR-011/019/020) |
| 6 | Verification + acceptance | — |

---

## Phase 1: Types, client, store

- [ ] 1.1 `types/jlens.ts` mirroring the backend schema shapes.
- [ ] 1.2 `api/jlens.ts` — `postReadout(request)`, typed both ways.
- [ ] 1.3 `stores/jlensStore.ts` — prompt, meta, tokens, pinned, selection, lensMode, status.
- [ ] 1.4 No fixture seed anywhere in the store.

## Phase 2: Presentational components

- [ ] 2.1 `ReadoutGrid` — layer axis from `meta.layers_by_type`, **never a constant**.
- [ ] 2.2 Rank heatmap when tokens are pinned; colour ramp scaled by `meta.top_n`.
- [ ] 2.3 Hover reveals the full top-k; clicking a hover token pins it.
- [ ] 2.4 `TrajectoryChart` — recharts, rank inverted (lower = stronger), gaps not connected.
- [ ] 2.5 `ByLayerRail` — per-layer readout at the selected position.
- [ ] 2.6 `LensModeTabs` — enablement derived from `meta.types`; disabled controls carry a reason.
- [ ] 2.7 Dual-mode styling throughout; tokens from `config/brand.ts`.

## Phase 3: Panel composition

- [ ] 3.1 `JLensPanel.tsx` — named export, `px-6 py-8` root, no page shell.
- [ ] 3.2 Prompt entry + submit; loading, empty and error states.
- [ ] 3.3 Background refetch must not unmount the panel or drop pinned state.

## Phase 4: Nav + chunking

- [ ] 4.1 `Sidebar.tsx` — add `'jlens'` to the union.
- [ ] 4.2 `Sidebar.tsx` — nav entry `{ id: 'jlens', label: 'J-Lens', icon: Sparkles }` **between
      `circuits` and `steering`**.
- [ ] 4.3 `App.tsx` — union.
- [ ] 4.4 `App.tsx` — `validPanels` (the third hardcoded copy; missing it breaks localStorage restore).
- [ ] 4.5 `App.tsx` — render chain.
- [ ] 4.6 `vite.config.ts` — `feature-jlens` chunk for `/components/jlens/`.

## Phase 5: Framing + honesty surfaces

- [ ] 5.1 `EvidenceRungCard` — Rung 0 · Readout, naming what raises it.
- [ ] 5.2 Single-token-name limitation stated.
- [ ] 5.3 "A readout resisting interpretation is not a null result."
- [ ] 5.4 "Absence of a signal is not evidence of absence."
- [ ] 5.5 Early-layer cells de-emphasised and labelled *expected* to be uninterpretable.
- [ ] 5.6 `ProvenanceStrip` — recipe/artifact identity; states explicitly when no artifact is
      involved (logit lens).
- [ ] 5.7 **No band constant exists in the panel.** Bands render only from a band report.

## Phase 6: Verification + acceptance

- [ ] 6.1 Nav entry renders **before** Steering.
- [ ] 6.2 Layer axis follows `layers_by_type` for 16 and 26 layers.
- [ ] 6.3 Jacobian/Diff disabled with a reason when absent from `meta.types`; enabled when present.
- [ ] 6.4 No bands without a report; no `40`/`90` constant in the source.
- [ ] 6.5 `grep -r "FIXTURES\|buildFixture\|scoreAt" frontend/src` returns nothing.
- [ ] 6.6 Type-check and full frontend suite green against baseline.
- [ ] 6.7 **Mutation controls**, each must go red: hardcode the layer axis; reintroduce a band
      default; enable Jacobian regardless of `meta.types`; render logit under a Jacobian label;
      move the nav entry after Steering; drop `'jlens'` from `validPanels`; hardcode the colour-ramp
      top-n; remove the interpretability caveat.
- [ ] 6.8 Three rounds of security-review + review; all findings fixed and re-verified.

**Acceptance:** "J-Lens" sits immediately before Steering and renders a live logit-lens readout whose
layer axis matches the model; Jacobian and Diff are visibly disabled with a stated reason; no band
shading appears without a band report; the rung card, caveats and provenance strip are present; and
no fixture constant is reachable from the shipped bundle.

---

## Relevant Files

| file | purpose |
|---|---|
| `frontend/src/types/jlens.ts` | wire-format mirror |
| `frontend/src/api/jlens.ts` | readout client |
| `frontend/src/stores/jlensStore.ts` | panel state |
| `frontend/src/components/jlens/*.tsx` | grid, chart, rail, tabs, rung, provenance |
| `frontend/src/components/panels/JLensPanel.tsx` | composition |
| `frontend/src/components/layout/Sidebar.tsx` | nav entry + union |
| `frontend/src/App.tsx` | union, validPanels, render chain |
| `frontend/vite.config.ts` | chunking |

Spec input, not shipped: `0xcc/brds/JSpacePanel.jsx`.

---

## Coverage audit (instruct 007)

| FPRD requirement | Phase |
|---|---|
| §3.1 nav placement before Steering | 4 |
| §3.2 grid, hover, pin, heatmap | 2 |
| §3.3 model-derived layer axis | 2.1 |
| §3.4 lens modes + honest disablement | 2.6 |
| §3.5 bands only from a report | 5.7 |
| §3.6 rank-vs-layer trajectories | 2.4 |
| §3.7 interpretability framing | 5.2–5.5 |
| §3.8 evidence rung | 5.1 |
| §3.9 provenance strip | 5.6 |

---

## Recorded follow-up debt

- Jacobian and Diff modes cannot be exercised end-to-end until doc chain 021 produces a validated
  artifact; until then they are tested via a stream fixture carrying both types.
- The readout is request/response; the wire format is designed for streaming and an SSE transport
  should follow once prompt lengths grow.
- Band rendering is implemented but unreachable until a band report exists (doc chain 021).
