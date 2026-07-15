# Feature PRD: Steering UX Enhancements

**Document ID:** 011_FPRD|Steering_UX
**Version:** 1.0
**Last Updated:** 2026-07-15
**Status:** Planned
**Priority:** P1 (Important Feature)
**Origin:** Steering experiments this session (MCP-driven, experiment `c4a273f1`) — see PADR IDL-27

---

## 1. Overview

### 1.1 Purpose
Bring three empirically-validated steering capabilities into the UI: **blended multi-feature steering at
scale (up to 20)**, **auto-computed baseline strength** from a feature's activation frequency, and a clear
**Blended vs Compare** mode toggle — with a **compact selected-feature tile** that makes 20 features usable.

### 1.2 User Problem
The Steering panel caps selection at **4 features**, hard-codes a starting strength of **10** for every
feature regardless of its behavior, and buries the blend-vs-separate distinction in a "Combined Mode"
checkbox that only appears at ≥2 features. Hands-on experimentation this session showed:
- The backend already blends N features in one pass (no 4-cap in the model/worker) — 4 is a UI/schema limit.
- A feature's **optimal steering strength is predictable from its activation frequency** (denser features
  need weaker steering); the fixed `10` is wrong for most features.
- The selected tile is ~170px tall — fine for 4, unusable for 20.

### 1.3 Solution
- Raise the selection limit to **20** (frontend constant + backend `max_length` + palette).
- A **Blended | Compare** segmented toggle replacing the checkbox: Blended sums all features in one pass
  (`/steering/async/combined`), Compare steers each feature separately, side-by-side (`/steering/async/compare`).
- **Auto-baseline strength**: whenever a feature is selected (Feature Groups hand-off or manual add),
  compute the starting strength from its activation frequency; fall back to the current default where
  frequency is unavailable, and show which was used.
- **Compact tiles** retaining every control (strength slider, additional strengths, remove, drag, color,
  label) plus the feature's `freq`/`max_activation` context.

---

## 2. User Stories

- **US-1** — As a researcher who found a concept cluster on the Feature Groups page, I want to hand 6–20
  members to Steering at once and blend them, so I can test a whole cluster's combined effect.
- **US-2** — As a researcher, I want each feature to start at a sensible strength derived from its
  activation frequency, so I'm not manually correcting a wrong default `10` on every feature.
- **US-3** — As a researcher, I want an obvious choice between "apply all features together" (blended) and
  "steer each feature separately" (compare), instead of an ambiguous checkbox.
- **US-4** — As a researcher steering many features, I want the selected-feature list compact so all my
  features are visible without endless scrolling, without losing any control.

---

## 3. Functional Requirements

### 3.1 Feature selection limit (blended scale)
| ID | Requirement | Status |
|----|-------------|--------|
| FR-1.1 | Selection limit raised from 4 to **20** features (all UI counters, guards, and labels) | Planned |
| FR-1.2 | Backend `CombinedSteeringRequest` and `SteeringComparisonRequest` accept up to 20 features | Planned |
| FR-1.3 | A **20-entry color palette** distinguishes tiles; colors are cosmetic (uniqueness no longer required) | Planned |
| FR-1.4 | The backend compare-path unique-color validator is removed/relaxed (auto-recolor at add time) | Planned |

### 3.2 Blended vs Compare toggle
| ID | Requirement | Status |
|----|-------------|--------|
| FR-2.1 | A **Blended \| Compare** segmented toggle replaces the "Combined Mode" checkbox | Planned |
| FR-2.2 | **Blended** → `/steering/async/combined` (all features summed, one output) | Planned |
| FR-2.3 | **Compare** → `/steering/async/compare` (each feature its own steered output) | Planned |
| FR-2.4 | The toggle is available whenever ≥1 feature is selected (Compare works with 1; Blended needs ≥2) | Planned |
| FR-2.5 | No new "true sequential" backend mode this phase (non-goal, §9) | Planned |

### 3.3 Auto-baseline strength
| ID | Requirement | Status |
|----|-------------|--------|
| FR-3.1 | On feature select, compute the starting strength from **activation frequency**: `S = clamp(2.9 − 2.6·freq, 1.0, 3.0)` (rounded to 0.1) | Planned |
| FR-3.2 | `max_activation` is stored and displayed for context but **not** used in the number (unit-norm decoder — see IDL-27) | Planned |
| FR-3.3 | When frequency is unavailable, fall back to the current default strength (**10**) | Planned |
| FR-3.4 | The tile indicates whether the strength is **auto** (from frequency) or **default** (fallback) | Planned |
| FR-3.5 | An **"Auto (from frequency)"** apply-to-all preset recomputes each tile's baseline from its stored frequency | Planned |
| FR-3.6 | The Feature Groups → Steering hand-off preserves each member's `activation_frequency` + `max_activation` | Planned |
| FR-3.7 | The in-steering SAE feature browser exposes `activation_frequency` so browser-added features auto-baseline too | Planned |

### 3.4 Compact tiles
| ID | Requirement | Status |
|----|-------------|--------|
| FR-4.1 | The selected-feature tile is roughly **half its current height** (~170px → ~80–90px collapsed) | Planned |
| FR-4.2 | All existing controls remain: strength slider, additional strengths, remove, drag-reorder, color, label | Planned |
| FR-4.3 | "Additional Strengths" collapses behind a small expander (hidden by default) to save space | Planned |
| FR-4.4 | The tile shows `freq` + `max_act` context and the auto/default strength badge | Planned |

---

## 4. User Interface

### 4.1 Steering sidebar (compact)
```
┌─────────────────────────────────────────┐
│ Selected Features (6/20)      🗑 Clear all │
│ Apply to all: [Auto] [Subtle][Mod][Strong]│
│ ┌───────────────────────────────────────┐ │
│ │ ● #24640 · L12   auto 1.6   ▸ +   ×   │ │  ← collapsed tile (~1 row + slider)
│ │ ▂▃▅▆ strength slider                  │ │
│ │ freq 0.21 · max 6.6                    │ │
│ └───────────────────────────────────────┘ │
│ … 5 more tiles …                           │
└─────────────────────────────────────────┘
```

### 4.2 Mode toggle (in the generation area)
```
Mode:  [ Blended ]  [ Compare ]      (segmented; Blended disabled until ≥2 features)
```

---

## 5. API / Integration

No new endpoints. Uses existing `/steering/async/combined` (blended) and `/steering/async/compare`.
Backend changes are limited to raising `max_length` (4→20) on both request schemas, relaxing the compare
unique-color validator, and adding `activation_frequency` to `SAEFeatureSummary` + the SAE browser query.

---

## 6. Data / Types

- Frontend `SelectedFeature` gains optional `max_activation`, `activation_frequency`, and
  `strengthSource: 'auto' | 'default' | 'manual'`.
- Frontend `FeatureColor` / `FEATURE_COLORS` / `FEATURE_COLOR_ORDER` widen to 20 entries.
- Backend `SAEFeatureSummary` gains `activation_frequency: float | None`.
- No database schema change (all stats already exist on `features`).

---

## 7. Dependencies

| Feature | Dependency |
|---------|-----------|
| Model Steering (006) | The async compare/combined endpoints being enhanced |
| Feature Discovery / Grouping (010) | Feature Groups hand-off; group members already carry both stats |

---

## 8. Success Criteria

- A user can select up to 20 features and blend them in one generation; the summed output is coherent when
  the strength budget is respected.
- Selecting a feature (via Groups hand-off or SAE browser) sets a frequency-derived starting strength that
  matches the measured formula (e.g. freq 0.037 → ~2.8, freq 0.48 → ~1.65); unknown-frequency → 10.
- The Blended vs Compare choice is unambiguous and dispatches to the correct endpoint.
- 20 tiles fit in the sidebar with far less scrolling than before, retaining every control.

---

## 9. Non-Goals

- **True sequential steering** (apply feature A, then B on A's output) — not supported by the backend; out of scope.
- **Per-SAE calibration** of the baseline constants — the 2.9/2.6 constants are treated as global this phase.
- **Changing the compare color semantics beyond de-duplication** — colors remain purely visual.

---

## 10. Testing Requirements

- Unit: `computeBaselineStrength` (formula, clamp bounds, null→default); store allows up to 20 and assigns
  colors; both hand-off paths pass frequency through.
- Backend: steering schema accepts 20 features; compare no longer rejects duplicate colors.
- E2E (Playwright, headless, LAN host): Groups → select 6+ → Steer selected → 6 compact tiles with auto
  baselines → toggle Blended/Compare → one blended generation on the fear trio.

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/011_FTDD|Steering_UX.md) | [TID](../tids/011_FTID|Steering_UX.md) | [Tasks](../tasks/011_FTASKS|Steering_UX.md)*
