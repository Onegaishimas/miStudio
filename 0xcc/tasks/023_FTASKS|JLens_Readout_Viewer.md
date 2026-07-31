# Task List: J-Lens Readout Viewer & Interpretability Framing

**Document ID:** 023_FTASKS|JLens_Readout_Viewer
**Version:** 1.0
**Status:** ✅ Implemented (2026-07-31) — 3 review rounds, 15 findings, 15 mutation controls verified
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

- [x] 1.1 `types/jlens.ts` mirroring the backend schema shapes.
- [x] 1.2 `api/jlens.ts` — `postReadout(request)`, typed both ways.
- [x] 1.3 `stores/jlensStore.ts` — prompt, meta, tokens, pinned, selection, lensMode, status.
- [x] 1.4 No fixture seed anywhere in the store.

## Phase 2: Presentational components

- [x] 2.1 `ReadoutGrid` — layer axis from `meta.layers_by_type`, **never a constant**.
- [x] 2.2 Rank heatmap when tokens are pinned; colour ramp scaled by `meta.top_n`.
- [x] 2.3 Hover reveals the full top-k; clicking a hover token pins it.
- [x] 2.4 `TrajectoryChart` — recharts, rank inverted (lower = stronger), gaps not connected.
- [x] 2.5 `ByLayerRail` — per-layer readout at the selected position.
- [x] 2.6 `LensModeTabs` — enablement derived from `meta.types`; disabled controls carry a reason.
- [x] 2.7 Dual-mode styling throughout; tokens from `config/brand.ts`.

## Phase 3: Panel composition

- [x] 3.1 `JLensPanel.tsx` — named export, `px-6 py-8` root, no page shell.
- [x] 3.2 Prompt entry + submit; loading, empty and error states.
- [x] 3.3 Background refetch must not unmount the panel or drop pinned state.

## Phase 4: Nav + chunking

- [x] 4.1 `Sidebar.tsx` — add `'jlens'` to the union.
- [x] 4.2 `Sidebar.tsx` — nav entry `{ id: 'jlens', label: 'J-Lens', icon: Sparkles }` **between
      `circuits` and `steering`**.
- [x] 4.3 `App.tsx` — union.
- [x] 4.4 `App.tsx` — `validPanels` (the third hardcoded copy; missing it breaks localStorage restore).
- [x] 4.5 `App.tsx` — render chain.
- [x] 4.6 `vite.config.ts` — `feature-jlens` chunk for `/components/jlens/`.

## Phase 5: Framing + honesty surfaces

- [x] 5.1 `EvidenceRungCard` — Rung 0 · Readout, naming what raises it.
- [x] 5.2 Single-token-name limitation stated.
- [x] 5.3 "A readout resisting interpretation is not a null result."
- [x] 5.4 "Absence of a signal is not evidence of absence."
- [x] 5.5 Early-layer cells de-emphasised and labelled *expected* to be uninterpretable.
- [x] 5.6 `ProvenanceStrip` — recipe/artifact identity; states explicitly when no artifact is
      involved (logit lens).
- [x] 5.7 **No band constant exists in the panel.** Bands render only from a band report.

## Phase 6: Verification + acceptance

- [x] 6.1 Nav entry renders **before** Steering.
- [x] 6.2 Layer axis follows `layers_by_type` for 16 and 26 layers.
- [x] 6.3 Jacobian/Diff disabled with a reason when absent from `meta.types`; enabled when present.
- [x] 6.4 No bands without a report; no `40`/`90` constant in the source.
- [x] 6.5 `grep -r "FIXTURES\|buildFixture\|scoreAt" frontend/src` returns nothing.
- [x] 6.6 Type-check and full frontend suite green against baseline.
- [x] 6.7 **Mutation controls** — 15 run, all verified biting: hardcode the layer axis (MUT-J24);
      enable Jacobian regardless of `meta.types` (J25); default a band report (J26); hardcode the
      colour-ramp top-n (J27); drop the caveat (J28); clear state on refetch (J29); remove the
      stale-response guard (J30); leave an unavailable mode selected (J31); index by array position
      (J32); move the nav entry after Steering (J33); drop `jlens` from the registry (J34); keep
      pins across a model change (J35); send the wrong lens type (J36); revert the keyboard
      fallback (J37); clamp against a fixed lens axis (J38); key chart series by token text (J39b);
      category x-axis (J40); draw bands with no report (J41).

      **Two initially SURVIVED** — J38 and J40 — both because the fixtures agreed by construction
      (one axis shared between lens types; an evenly spaced axis where category and numeric
      positions coincide). Each was rebuilt to separate the cases and re-run as a negative control.
- [x] 6.8 Three rounds of security-review + review; all findings fixed and re-verified.

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
- Band rendering is implemented but **unreachable from the panel** until a band report exists (doc
  chain 021). It is covered at its real caller in `TrajectoryChart.test.tsx` rather than left
  untested — which is how the category-axis defect (J40) was found at all.
- **The backend `/jlens/readout` endpoint returns 501** until doc chain 021 binds model resolution
  and artifact loading. The panel is complete and its request is asserted by payload and call
  count; end-to-end readout against a live model lands with 021.
- **MCP parity** for readout/probe is owned by doc chain 028 (BR-025..028) per the increment plan,
  and must be covered by the reachability harness the same way the REST route is here.

---

## Review record

| round | findings | notes |
|---|---|---|
| 1 | 8 | prompt bound, position-vs-index, mode clamp, pins on model change, selector-scoped subscription, stale-response guard, broadened band guard, **reachability test** (nothing failed when the App render line was deleted) |
| 2 | 5 + 1 test finding | effective-axis clamp, reset sequence bump, **keyboard path to pinning** (hover-only), provenance noise, payload+call-count assertion. The clamp fix survived its own mutation — fixtures agreed by construction. |
| 3 | 2 + 5 pre-existing | category x-axis drops band shading; chart series collided with the x-axis key. Also fixed: 4 tests pinned to single-theme classes, and `DatasetsPanel.handleDownload` rejecting unhandled under a test named "handles download errors gracefully". |

One finding was **mis-diagnosed and corrected**: the series-key defect was first attributed to
recharts path-resolving dotted `dataKey`s. An empirical check refuted that; the real defect is the
namespace collision with the x-axis key, and the fix is the same.

Frontend suite after round 3: **1007/1007 passing, 0 unhandled errors.**
