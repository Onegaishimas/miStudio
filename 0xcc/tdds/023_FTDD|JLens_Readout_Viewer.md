# Technical Design Document: J-Lens Readout Viewer & Interpretability Framing

**Document ID:** 023_FTDD|JLens_Readout_Viewer
**Version:** 1.0
**Status:** Planned
**Related:** 023_FPRD · PADR IDL-40, IDL-44, IDL-45 · reference implementation `0xcc/brds/JSpacePanel.jsx`

---

## 1. Porting the reference implementation

`JSpacePanel.jsx` is the interaction specification (BR-010), and it is already written against the
upstream wire format. Four adaptations are required, and they are the whole of the port:

| # | change | why |
|---|---|---|
| 1 | `.jsx` → `.tsx` with shared wire types | the app is TypeScript; types come from the same shapes the backend emits |
| 2 | dark-only → dual-mode | the mock roots at `min-h-screen bg-slate-900 text-slate-200`; every panel here is dual-classed and pulls tokens from `config/brand.ts` |
| 3 | drop the page shell | `<main>` already supplies the sidebar offset; panels own a `px-6 py-8` root |
| 4 | delete `FIXTURES` / `buildFixture` / `scoreAt` / `NOISE`, drive from the API | nothing synthetic ships |

Every interaction is preserved: lens-mode tabs, prompt strip, layer × position grid, rank heatmap,
hover top-k, click-to-pin, trajectory chart, by-layer rail, rung card, provenance strip.

## 2. Three constants in the mock that are model-specific and must not survive the port

The mock hardcodes what a real panel must derive. Each is a silent-wrongness risk rather than a
crash:

```js
const LAYERS = Array.from({length: 21}, (_, i) => i * 5);   // 0,5,...,100
const BAND   = { workspaceStart: 40, motorStart: 90 };
const TOP_N  = 8;
```

- **`LAYERS`** — real models have 16 (LFM2) or 26 (gemma-2-2b). Drive from `meta.layers_by_type`.
- **`BAND`** — these are the source paper's **Sonnet-4.5** boundaries. BR-002 forbids porting them
  to any other model and requires the product make porting impossible *by construction*. So there
  is **no band constant in the panel at all**: bands are absent unless a band report supplies them.
- **`TOP_N`** — comes from `meta.top_n`, since the server decides what it sent.

The mock's own `rankColor` uses `TOP_N` for its opacity ramp; that must follow the meta value or the
heatmap mis-scales on any other top-n.

## 3. Lens modes and the disablement rule

The mock offers Jacobian / Logit / Diff and treats `readLens = lensMode === "DIFF" ? "JACOBIAN_LENS"
: lensMode`. That is correct *once both lenses exist*.

Until a validated artifact exists for the selected model, the stream carries only `LOGIT_LENS`.
Jacobian and Diff are therefore **disabled with a stated reason**, driven by what `meta.types`
actually contains — not by a feature flag, and never by silently substituting logit data.

```
available = new Set(meta.types)
jacobianEnabled = available.has('JACOBIAN_LENS')
diffEnabled     = jacobianEnabled && available.has('LOGIT_LENS')
```

This is BR-019 rung discipline expressed in the UI: a lower rung must never appear in a higher
rung's clothing.

## 4. Bands are data, not defaults

```
bands = bandReport ? { sensoryEnd, workspaceStart, motorStart } : null
```

When `bands` is null the grid renders without shading and the legend states that no band report
exists for this model. There is deliberately no fallback object — a default would be the Sonnet
figures by another name.

## 5. Evidence rung and framing

The mock's rung card and footer caveat are requirements (BR-011, BR-019, BR-020), not decoration:

- **Rung 0 · Readout**, with the sentence naming what would raise it ("run a coordinate swap with a
  matched control").
- Readouts are limited to **single-token names**.
- A readout that resists interpretation **is not a null result**.
- **Absence of a signal is not evidence of absence** of the underlying computation.

Early-layer cells are visually de-emphasised and labelled *expected to be uninterpretable*, so a
user reading noise there knows it is anticipated.

## 6. Architecture / types

```
frontend/src/types/jlens.ts          LensMetaMessage, LensTokenMessage, LensTypeSlice,
                                     LayerApplicability, ReadoutRequest   (mirror of the backend)
frontend/src/api/jlens.ts            postReadout()
frontend/src/stores/jlensStore.ts    prompt, meta, tokens, pinned, selection, lensMode, status
frontend/src/components/jlens/
    ReadoutGrid.tsx                  position x layer grid + hover + pin
    TrajectoryChart.tsx              recharts rank-vs-layer  (heaviest import)
    ByLayerRail.tsx                  per-layer readout list
    EvidenceRungCard.tsx             rung + what raises it
    ProvenanceStrip.tsx              recipe / artifact identity
    LensModeTabs.tsx                 Jacobian | Logit | Diff with disablement reasons
frontend/src/components/panels/JLensPanel.tsx    composition + data fetch
```

Sub-components live under `components/jlens/` specifically so `vite.config.ts` `manualChunks` can
route them to a `feature-jlens` chunk — those rules match sub-component directories and never
`/components/panels/`, which is why `SteeringPanel` and `CircuitsPanel` currently land in the main
bundle.

## 7. Risks

| risk | mitigation |
|---|---|
| Mock's 21-layer axis survives the port; grid silently wrong on every real model | layer axis asserted against differing `layers_by_type` in tests |
| Sonnet band constants ported; bands look authoritative and are foreign | no band constant exists in the panel; bands absent without a report |
| Diff/Jacobian render logit data under the wrong label | enablement derived from `meta.types`; disabled state carries a reason |
| Fixture data ships | `grep` guard in tests; fixtures stay in `0xcc/` |
| `TOP_N` mismatch mis-scales the heatmap colour ramp | ramp derived from `meta.top_n` |
| Panel unmounts on background refresh, losing pinned state | house regression already known from `ExtractionsPanel.nounmount.test.tsx`; same guard applied |
| Trajectory chart pulls recharts into the main bundle | sub-components under `components/jlens/` + a `manualChunks` clause |
