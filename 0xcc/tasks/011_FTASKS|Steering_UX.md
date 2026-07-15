# Feature Tasks: Steering UX Enhancements

**Document ID:** 011_FTASKS|Steering_UX
**Version:** 1.0
**Last Updated:** 2026-07-15
**Status:** Planned
**Related PRD:** [011_FPRD|Steering_UX](../prds/011_FPRD|Steering_UX.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Backend limits + freq field | 3 tasks | ✅ Complete |
| Phase 2: Types + baseline util | 4 tasks | ✅ Complete |
| Phase 3: Store + hand-offs | 4 tasks | ✅ Complete |
| Phase 4: Compact tiles + toggle UI | 5 tasks | ✅ Complete |
| Phase 5: Tests + deploy + E2E | 4 tasks | 🔄 In progress (tests ✅, deploy/E2E pending) |

**Total: 20 tasks**

---

## Phase 1: Backend limits + freq field

### Task 1.1: Raise combined/compare feature caps
- [x] `CombinedSteeringRequest.selected_features` max_length 4 → 20
- [x] `SteeringComparisonRequest.selected_features` max_length 4 → 20

### Task 1.2: Drop compare unique-color validator + widen color Literal
- [x] Remove `validate_selected_features` unique-color check (colors cosmetic at 20)
- [x] Widen `SelectedFeature.color` Literal 4 → 20 names (else features 5–20 would 422 on color)

### Task 1.3: Expose activation_frequency on SAE browser
- [x] `SAEFeatureSummary` += `activation_frequency: float | None`
- [x] `browse_sae_features` maps `activation_frequency=f.activation_frequency`

**Files:** `backend/src/schemas/steering.py`, `backend/src/schemas/sae.py`, `backend/src/api/v1/endpoints/saes.py`

---

## Phase 2: Types + baseline util

### Task 2.1: computeBaselineStrength util + test
- [x] `frontend/src/utils/steeringStrength.ts` (formula + fallback)
- [x] `steeringStrength.test.ts` (formula table, clamp, null→default)

### Task 2.2: 20-color palette
- [x] Widen `FeatureColor`, `FEATURE_COLORS` (literal Tailwind classes), `FEATURE_COLOR_ORDER` to 20

### Task 2.3: SelectedFeature fields
- [x] Add `max_activation?`, `activation_frequency?`, `strengthSource?`

### Task 2.4: SAEFeatureSummary frontend type
- [x] Add `activation_frequency?` to `frontend/src/types/sae.ts`

**Files:** `frontend/src/utils/steeringStrength.ts` (+test), `frontend/src/types/steering.ts`, `frontend/src/types/sae.ts`

---

## Phase 3: Store + hand-offs

### Task 3.1: MAX 20 + export + addFeature auto-baseline
- [x] `MAX_SELECTED_FEATURES = 20`, exported
- [x] `addFeature` computes baseline from freq, sets `strengthSource`, carries stats, wraps colors
- [x] `applyAutoBaseline` action recomputes every tile from its stored frequency

### Task 3.2: Steering mode (Blended vs Compare)
- [x] Kept the existing `combinedMode` boolean (true = Blended /combined, false = Compare /compare) —
      per the plan's "keep boolean if simpler; TDD decides." Rendered as a two-way segmented toggle
      (no `steeringMode` enum needed; `handleGenerate` already branches on `combinedMode`).

### Task 3.3: Feature Groups selection-map widening
- [x] `selection` Map value → `{ neuron_index, max_activation, activation_frequency }`
- [x] `toggleSelect`/`setSelected` + all `GroupMembersTable` call sites updated

### Task 3.4: Hand-off paths pass frequency
- [x] `FeatureGroupsPanel.handleSteerSelected` omits strength, passes stats
- [x] `FeatureBrowser` handleSelectFeature passes freq; handleManualAdd falls back

**Files:** `steeringStore.ts`, `featureGroupsStore.ts`, `FeatureGroupsPanel.tsx`, `GroupMembersTable.tsx`, `FeatureBrowser.tsx`

---

## Phase 4: Compact tiles + toggle UI

### Task 4.1: Compact SelectedFeatureCard
- [x] p-3→p-2, condensed rows, additional-strengths behind expander, keep all controls

### Task 4.2: Strength-source + stats display
- [x] `auto`/`default` badge; muted `f · m` stats inline when present

### Task 4.3: FeatureSelector limit/labels
- [x] Replace all `4` literals with `MAX_SELECTED_FEATURES`; update "(N/20)" and "Select up to 20"

### Task 4.4: Auto apply-to-all preset
- [x] "Auto" button recomputes each tile's baseline via `applyAutoBaseline`

### Task 4.5: Blended | Compare toggle + dispatch
- [x] Segmented toggle in SteeringPanel; `handleGenerate` branches on `combinedMode`
- [x] `ComparisonPreview` maxFeatures 4→20 (default now `MAX_SELECTED_FEATURES`)

**Files:** `SelectedFeatureCard.tsx`, `FeatureSelector.tsx`, `SteeringPanel.tsx`, `ComparisonPreview.tsx`

---

## Phase 5: Tests + deploy + E2E

### Task 5.1: Backend tests
- [x] `tests/unit/test_steering_schema.py` — 20-feature combined/compare validate; 21 rejected;
      compare duplicate-color no longer 422; new palette colors accepted; unknown color rejected (9 tests)

### Task 5.2: Frontend tests + build
- [x] Store test (up to 20, auto-baseline, applyAutoBaseline); `steeringStrength.test.ts`; type-check + build green

### Task 5.3: Commit + push + k8s deploy
- [x] docs commit (53f2245), then backend+frontend (e959ce5); CI green (mirror build success); `k8s_deploy` ✅

### Task 5.4: E2E + closeout
- [x] Playwright E2E (LAN host, headless): selected SAE, added 3 features → **Selected Features (3/20)**,
      compact tiles with **default** badges (manual index-add has no freq), **Blended|Compare** toggle present,
      **Auto** apply-to-all preset present. Screenshot → `0xcc/caps/miStudio_Steering_Panel-CompactTiles_20260715.png`.
      (Note: the available SAE had no browsable features / no freq, so the auto-baseline number path is covered
      by unit tests rather than E2E; the default-fallback path is E2E-verified.)
- [x] Statuses Planned→Implemented; PPRD v3.5 row 12 → Complete; §3.12 updated; CLAUDE.md log

**Files:** `backend/tests/**`, `frontend/src/stores/steeringStore.test.ts`, docs

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `schemas/steering.py` | max_length 4→20, drop compare color validator |
| `schemas/sae.py` | `SAEFeatureSummary.activation_frequency` |
| `api/v1/endpoints/saes.py` | map activation_frequency in browser |

### Frontend
| File | Purpose |
|------|---------|
| `utils/steeringStrength.ts` | baseline formula (+test) |
| `types/steering.ts` | 20-color palette, SelectedFeature fields |
| `stores/steeringStore.ts` | MAX 20, auto-baseline, steeringMode |
| `stores/featureGroupsStore.ts` | selection-map widening |
| `components/steering/SelectedFeatureCard.tsx` | compact tile |
| `components/steering/FeatureSelector.tsx` | limit/labels/auto preset |
| `components/panels/SteeringPanel.tsx` | Blended/Compare toggle |

---

*Related: [PRD](../prds/011_FPRD|Steering_UX.md) | [TDD](../tdds/011_FTDD|Steering_UX.md) | [TID](../tids/011_FTID|Steering_UX.md)*
