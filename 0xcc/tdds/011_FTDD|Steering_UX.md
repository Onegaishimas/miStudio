# Technical Design Document: Steering UX Enhancements

**Document ID:** 011_FTDD|Steering_UX
**Version:** 1.0
**Last Updated:** 2026-07-15
**Status:** Planned
**Related PRD:** [011_FPRD|Steering_UX](../prds/011_FPRD|Steering_UX.md)

---

## 1. Architecture

Frontend-heavy. The backend already blends N features (`steering_service` registers all feature configs and
sums them in one hook pass — no 4-cap); the only backend changes are lifting two `max_length` limits, one
validator, and adding one response field. All new logic lives in the frontend.

```
Feature Groups member (has freq+max_act)  ─┐
SAE feature browser (add freq field)       ─┼─▶ addFeature(feature)  ──▶ computeBaselineStrength(freq)
Manual index add (no stats)                ─┘                              │
                                                                          ▼
                                              SelectedFeature{ strength, strengthSource, freq, max_act, color }
                                                                          │
                                          ┌───────────────────────────────┴───────────────┐
                                     Blended (/combined)                          Compare (/compare)
                                     all summed, 1 output                    per-feature, N outputs
```

## 2. Baseline strength formula

Derived from experiment `c4a273f1` (4 features, freq 0.037–0.484):

```
computeBaselineStrength(freq):
  if freq == null:  return { value: 10,  source: 'default' }
  s = 2.9 - 2.6 * freq
  s = clamp(s, 1.0, 3.0)
  return { value: round(s, 1), source: 'auto' }
```

| freq | S (auto) |
|------|----------|
| 0.037 | 2.8 |
| 0.214 | 2.3 |
| 0.368 | 1.9 |
| 0.484 | 1.6 |
| null | 10 (default) |

**Why frequency-only:** on this SAE the JumpReLU decoder columns are unit-norm, so the injected vector's
magnitude equals the raw strength — a feature's `max_activation` doesn't change how hard steering hits.
`max_activation` is stored/shown for context only (see PADR IDL-27). The 2.9/2.6 constants are SAE-local
but treated as global this phase.

## 3. Color palette (20 entries)

**Tailwind purge constraint:** dynamically-built class names (`bg-${color}-500`) are purged. The palette
must be a fixed map of **literal, statically-analyzable class strings**. Expand `FEATURE_COLORS` to 20 named
entries, each with `{ bg, border, text, light }` full literal classes, using Tailwind's built-in hues
(teal, blue, purple, amber first for continuity; then rose, cyan, lime, orange, fuchsia, sky, emerald,
violet, pink, indigo, yellow, red, green, slate-lighter, etc.). `FEATURE_COLOR_ORDER` becomes the 20-key
array. `addFeature` picks the next unused color, then wraps (duplicates acceptable — backend no longer
enforces uniqueness).

## 4. Type changes

`frontend/src/types/steering.ts`:
```ts
export type FeatureColor = /* 20 literal names */;
export interface SelectedFeature {
  … existing …
  max_activation?: number | null;
  activation_frequency?: number | null;
  strengthSource?: 'auto' | 'default' | 'manual';
}
```
`addFeature` signature stays `Omit<SelectedFeature, 'color' | 'instance_id'>`; callers may now include the
two stat fields. When they do and don't pass an explicit strength, the store computes the baseline.

Backend `SAEFeatureSummary` (`backend/src/schemas/sae.py`): add `activation_frequency: float | None = None`.

## 5. Blended vs Compare

Replace `combinedMode: boolean` with `steeringMode: 'blended' | 'compare'` in the store (default `'compare'`).
`SteeringPanel.handleGenerate` dispatch:
```
steeringMode === 'blended' && selected ≥ 2 → generateCombined()
isBatchMode                                → generateBatchComparison()   (unchanged)
else                                       → generateComparison()        (compare)
```
UI: a segmented two-button toggle (Blended disabled with a tooltip until ≥2 features). Combined-results and
comparison-results rendering paths already exist and are unchanged.

## 6. Compact tile design

`SelectedFeatureCard.tsx` — before ~170px, after ~85px:
- Root `p-3 → p-2`, `border-2 → border`, tighter `gap`/`mb`.
- **Row 1:** color dot · `#idx` · `L{layer}` · strength badge (`auto 1.6` / `default 10`) · additional-strengths
  expander chevron · remove ×.
- **Row 2:** compact `StrengthSlider` (already supports `compact`).
- **Row 3 (muted, tiny):** `freq 0.21 · max 6.6` (only when stats present).
- Additional-strengths section **collapsed by default** behind the Row-1 chevron; expands to today's chip UI.
- Label shown as a truncated single line (`line-clamp-1`) instead of `line-clamp-2`.

## 7. Backend changes (exact)

- `backend/src/schemas/steering.py`
  - `CombinedSteeringRequest.selected_features`: `max_length=4 → 20`.
  - `SteeringComparisonRequest.selected_features`: `max_length=4 → 20`.
  - Remove/relax `validate_selected_features` unique-color check (drop the validator; colors are cosmetic).
  - Widen `SelectedFeature.color` from the 4-value `Literal["teal","blue","purple","amber"]` to the full
    20-name palette Literal (original 4 first, for continuity). **Discovered during implementation:** without
    this, features 5–20 would 422 on the color field even after the max_length lift, since the UI assigns the
    new palette names. Kept as a Literal (not `str`) so the accepted set stays documented and in lock-step
    with the frontend `FeatureColor` union.
- `backend/src/schemas/sae.py`: `SAEFeatureSummary` += `activation_frequency: float | None = None`.
- `backend/src/api/v1/endpoints/saes.py` `browse_sae_features`: populate `activation_frequency=f.activation_frequency`.

## 8. Risks

| Risk | Mitigation |
|------|-----------|
| Tailwind purges 20th color | Literal class strings only; add a safelist comment; verify in `npm run build` |
| 20 blended features over-drive output | Budget guidance in docs; the sum-of-strengths rule from experiments; user controls each strength |
| Baseline formula wrong for non-LFM2.5 SAEs | Constants documented as SAE-local; fallback + manual override always available |
| combinedMode→steeringMode rename breaks saved state | Store is not persisted across reloads; no migration needed |

---

*Related: [PRD](../prds/011_FPRD|Steering_UX.md) | [TID](../tids/011_FTID|Steering_UX.md) | [Tasks](../tasks/011_FTASKS|Steering_UX.md)*
