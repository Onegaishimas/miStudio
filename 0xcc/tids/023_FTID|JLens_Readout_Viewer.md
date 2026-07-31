# Technical Implementation Document: J-Lens Readout Viewer & Interpretability Framing

**Document ID:** 023_FTID|JLens_Readout_Viewer
**Version:** 1.0
**Status:** Planned
**Related:** 023_FPRD · 023_FTDD · reference implementation `0xcc/brds/JSpacePanel.jsx`

---

## 1. Implementation Order

1. **Wire types** (`types/jlens.ts`) — mirror the backend schema. Everything else consumes these.
2. **API client** (`api/jlens.ts`) — one call, typed.
3. **Store** (`stores/jlensStore.ts`) — prompt, meta, tokens, pinned, selection, lens mode, status.
4. **Presentational sub-components** under `components/jlens/`, ported from the mock one at a time,
   each driven by props only.
5. **`JLensPanel.tsx`** — composition and fetch.
6. **Nav wiring** — all five edit sites in one change so no partial state exists.
7. **Chunking** — `manualChunks` clause.
8. **Tests**, including the ordering assertion and the no-fixtures guard.

---

## 2. File-by-file

| file | contents |
|---|---|
| `frontend/src/types/jlens.ts` | `LensMetaMessage`, `LensTokenMessage`, `LensTypeSlice`, `LayerApplicability`, `ReadoutRequest`, `ReadoutResponse` |
| `frontend/src/api/jlens.ts` | `postReadout(request)` |
| `frontend/src/stores/jlensStore.ts` | zustand store; no fixture seed |
| `frontend/src/components/jlens/ReadoutGrid.tsx` | grid, hover, pin, rank heatmap |
| `frontend/src/components/jlens/TrajectoryChart.tsx` | recharts rank-vs-layer |
| `frontend/src/components/jlens/ByLayerRail.tsx` | per-layer readout list |
| `frontend/src/components/jlens/LensModeTabs.tsx` | Jacobian / Logit / Diff + disablement reasons |
| `frontend/src/components/jlens/EvidenceRungCard.tsx` | rung + what raises it |
| `frontend/src/components/jlens/ProvenanceStrip.tsx` | recipe / artifact identity |
| `frontend/src/components/panels/JLensPanel.tsx` | composition, fetch, empty/error states |
| `frontend/src/components/layout/Sidebar.tsx` | union + nav entry **before `steering`** |
| `frontend/src/App.tsx` | union, `validPanels`, render chain |
| `frontend/vite.config.ts` | `feature-jlens` chunk |
| `frontend/src/components/panels/JLensPanel.test.tsx` | ordering, disablement, bands, framing, no fixtures |

---

## 3. Pitfalls

Each of these produces a panel that *looks* right.

1. **Do not port `LAYERS`.** The mock's `Array.from({length: 21}, (_, i) => i * 5)` is 0,5,…,100.
   Real models have 16 or 26 layers. Drive the axis from `meta.layers_by_type[type]`.

2. **Do not port `BAND`.** `{ workspaceStart: 40, motorStart: 90 }` are the source paper's
   **Sonnet-4.5** boundaries. BR-002 forbids porting them and requires the product make porting
   impossible by construction — so there is **no band constant in the panel**. Bands render only
   from a band report, and are absent otherwise.

3. **Do not port `TOP_N`.** `rankColor` uses it for the opacity ramp; hardcoding 8 mis-scales the
   heatmap whenever the server sent a different top-n. Use `meta.top_n`.

4. **Jacobian/Diff enablement comes from `meta.types`,** not a flag and not a guess. Rendering logit
   data under a Jacobian label is a rung-discipline defect (BR-019), and it is invisible to the user.

5. **`readLens = mode === 'DIFF' ? 'JACOBIAN_LENS' : mode`** — the mock's line is correct but only
   reachable once both lenses exist. Guard it, or Diff mode reads a slice that is not in the stream
   and throws on `.top_tokens` of `undefined`.

6. **Delete the fixture generator entirely.** `FIXTURES`, `buildFixture`, `scoreAt`, `NOISE`, `gauss`,
   `rnd`. Leaving them "for tests" means they ship — put fixtures in the test file instead.

7. **The mock is dark-only.** `bg-slate-900 text-slate-200` with no `dark:` prefixes. Every panel in
   this app is dual-classed; a straight paste renders unreadable in light mode.

8. **The mock owns a full page shell** (`min-h-screen w-full`, `max-w-[1400px] mx-auto`). Panels do
   not — `<main>` supplies the sidebar offset already.

9. **Five nav edit sites, not three.** The `ActivePanel` union is duplicated in `Sidebar.tsx` *and*
   `App.tsx`, and `validPanels` in `App.tsx` is a third hardcoded copy. Miss `validPanels` and
   localStorage restore silently drops the user back to Datasets.

10. **Nav order is array position.** There is no sort key; the entry must be physically between
    `circuits` and `steering`.

11. **`manualChunks` matches sub-component directories, never `/components/panels/`.** Heavy
    imports (recharts) must live under `components/jlens/` or they land in the main bundle, as
    `SteeringPanel` and `CircuitsPanel` currently do.

12. **Do not let a background refetch unmount the panel** and drop pinned state — the same defect
    fixed in `ExtractionsPanel` this cycle, where a loading flag blanked the list every 15s.

---

## 4. Testing

**Ordering** — assert the J-Lens nav entry renders before Steering. Array position is the only
mechanism, so a test that merely asserts presence would pass with it in the wrong place.

**Layer axis** — render with `layers_by_type` of 16 and of 26 and assert the row count follows.
A single-shape test would pass against a hardcoded 21.

**Disablement** — with `meta.types = ['LOGIT_LENS']`, Jacobian and Diff are disabled **and carry a
reason**; with both types present they are enabled.

**Bands** — no band report ⇒ no shading, and a stated reason. Assert no `40` / `90` constant exists
in the panel source.

**Framing** — rung card, single-token caveat, "not a null result", and absence-is-not-evidence text
all present.

**No fixtures** — `grep -r "FIXTURES\|buildFixture\|scoreAt" frontend/src` returns nothing.

**Mutation controls to run** (each must go red):
- hardcode the layer axis to 21
- reintroduce a default band constant
- enable Jacobian regardless of `meta.types`
- render logit data under the Jacobian label
- move the nav entry after Steering
- drop `'jlens'` from `validPanels`
- hardcode `TOP_N` in the colour ramp
- remove the interpretability caveat
