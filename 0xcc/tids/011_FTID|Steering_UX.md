# Technical Implementation Document: Steering UX Enhancements

**Document ID:** 011_FTID|Steering_UX
**Version:** 1.0
**Last Updated:** 2026-07-15
**Status:** Planned
**Related TDD:** [011_FTDD|Steering_UX](../tdds/011_FTDD|Steering_UX.md)

---

## 1. Implementation Order

1. Backend limits + freq field (unblocks everything, independently shippable)
2. Frontend types (palette + SelectedFeature) + `computeBaselineStrength` util + its test
3. Store (`MAX 20`, addFeature auto-baseline, steeringMode) + Feature Groups selection-map widening + hand-offs
4. Compact `SelectedFeatureCard` + FeatureSelector label/limit fixes + Blended/Compare toggle
5. Tests + build + deploy + E2E

## 2. File-by-file

### 2.1 Backend
`backend/src/schemas/steering.py`
```python
# CombinedSteeringRequest.selected_features  (~:445)
selected_features: List[SelectedFeature] = Field(..., min_length=1, max_length=20,
    description="List of features to apply together (1-20)")
# SteeringComparisonRequest.selected_features  (~:114)
selected_features: List[SelectedFeature] = Field(..., min_length=1, max_length=20,
    description="List of features to steer with (1-20)")
```
Delete the `validate_selected_features` unique-color validator (`~:135–142`) — with 20 features and cosmetic
colors, uniqueness cannot hold. (Keep the feature-index validation if any is separate.)

`backend/src/schemas/sae.py` — `SAEFeatureSummary` (`~:270`): add `activation_frequency: float | None = None`.

`backend/src/api/v1/endpoints/saes.py` — `browse_sae_features` (`~:449`): add
`activation_frequency=f.activation_frequency` to the `SAEFeatureSummary(...)` construction. (`Feature.activation_frequency`
is a non-null column, so `f.activation_frequency` is always present.)

### 2.2 `frontend/src/utils/steeringStrength.ts` (NEW)
```ts
export interface BaselineStrength { value: number; source: 'auto' | 'default'; }
export const DEFAULT_STRENGTH = 10;
export function computeBaselineStrength(freq: number | null | undefined): BaselineStrength {
  if (freq == null) return { value: DEFAULT_STRENGTH, source: 'default' };
  const s = Math.min(3.0, Math.max(1.0, 2.9 - 2.6 * freq));
  return { value: Math.round(s * 10) / 10, source: 'auto' };
}
```

### 2.3 `frontend/src/types/steering.ts`
- Widen `FeatureColor` union to 20 literal names.
- Widen `FEATURE_COLORS` to 20 entries — **literal Tailwind classes only** (Pitfall 1). Keep teal/blue/purple/amber
  as the first 4.
- `FEATURE_COLOR_ORDER` → the 20-name array.
- `SelectedFeature` += `max_activation?`, `activation_frequency?`, `strengthSource?`.

### 2.4 `frontend/src/stores/steeringStore.ts`
- `MAX_SELECTED_FEATURES = 20` (`:40`) — **and `export` it** so `FeatureSelector` uses it (Pitfall 2).
- `addFeature` (`:405`): if caller didn't pass `strength`, compute from
  `computeBaselineStrength(feature.activation_frequency)`; set `strength` + `strengthSource`; carry
  `max_activation`/`activation_frequency` onto the stored feature. Color: next unused from the 20-order, wrap on overflow.
- Replace `combinedMode`/`setCombinedMode` with `steeringMode: 'blended' | 'compare'` (default `'compare'`) and
  `setSteeringMode`. Keep `isCombinedGenerating`/`combinedResults`/`generateCombined` as-is.

### 2.5 `frontend/src/stores/featureGroupsStore.ts`
- Widen `selection` Map value: `Map<string, number>` → `Map<string, { neuron_index: number; max_activation: number | null; activation_frequency: number | null }>` (`:47`).
- `toggleSelect` (`:214`) and `setSelected` (`:214–235`): store the stat object; callers in
  `GroupMembersTable.tsx` pass `member.max_activation` + `member.activation_frequency`.

### 2.6 `frontend/src/components/panels/FeatureGroupsPanel.tsx`
`handleSteerSelected` (`:72–82`): for each `[featureId, { neuron_index, max_activation, activation_frequency }]`,
`addFeature({ feature_idx: neuron_index, layer, label: null, feature_id: featureId, max_activation, activation_frequency })`
— **omit strength** so the store auto-computes it.

### 2.7 `frontend/src/components/steering/FeatureBrowser.tsx`
`handleSelectFeature` (`:215`): pass `activation_frequency: item.activation_frequency` + `max_activation`,
omit strength. `handleManualAdd` (`:229`): no stats → omit both → store falls back to default 10.

### 2.8 `frontend/src/components/steering/SelectedFeatureCard.tsx`
Compact rewrite per FTDD §6. Add a local `expanded` state for additional-strengths. Show strength-source
badge (`auto`/`default`) next to the strength value; show `freq`/`max` muted row when present.

### 2.9 `frontend/src/components/steering/FeatureSelector.tsx`
Replace `4` literals at `:194,208,215,250,332,371,383` with `MAX_SELECTED_FEATURES`. Add the **"Auto"**
apply-to-all button that maps each feature through `computeBaselineStrength(f.activation_frequency)`.

### 2.10 `frontend/src/components/steering/ComparisonPreview.tsx`
`maxFeatures` default `4 → 20` (`:22`).

### 2.11 `frontend/src/components/panels/SteeringPanel.tsx`
Replace the `combinedMode` checkbox (`:543–561`) with a **Blended | Compare** segmented toggle bound to
`steeringMode`/`setSteeringMode`. Update `handleGenerate` (`:185–192`) to branch on `steeringMode`.

## 3. Pitfalls

1. **Tailwind purge** — never build color classes dynamically; the 20 `FEATURE_COLORS` entries must be full
   literal strings. Verify the 20th color renders after `npm run build`.
2. **Scattered `4` literals** — the limit lives in ~7 places outside the constant (FeatureSelector, ComparisonPreview,
   FeatureBrowser). Grep `\b4\b` in `components/steering/` and change all.
3. **Compare unique-color** — the backend validator must go, or compare with >4 features 422s even after `max_length`.
4. **freq absent from SAE browser** until the backend field lands — sequence backend first, or accept default 10
   for browser-added features in the interim.
5. **Groups selection-map migration** — `GroupMembersTable` currently calls `toggleSelect(id, neuron_index)`;
   update every call site to the new 3-field shape or TS will error.

## 4. Testing

- `frontend/src/utils/steeringStrength.test.ts`: formula table (0.037→2.8, 0.214→2.3, 0.484→1.6), null→{10,default},
  clamp at both ends (freq 0 → 2.9 not >3; freq 1 → 1.0 floor).
- Store test: `addFeature` succeeds for 5..20; distinct colors up to 20 then wraps; auto-baseline applied when
  freq present, default when absent.
- Backend: extend `tests/**/test_steering*` (or add) asserting 20-feature combined/compare requests validate and
  duplicate colors on compare no longer 422.

---

*Related: [PRD](../prds/011_FPRD|Steering_UX.md) | [TDD](../tdds/011_FTDD|Steering_UX.md) | [Tasks](../tasks/011_FTASKS|Steering_UX.md)*
