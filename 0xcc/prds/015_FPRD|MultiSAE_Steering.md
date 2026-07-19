# Feature PRD: Multi-SAE Cross-Layer Steering

**Document ID:** 015_FPRD|MultiSAE_Steering
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CIRCUITS-001 (BR-001, BR-002, BR-003, BR-004) as amended by BRD-MIS-CIRCUITS-002 (BR-024) · 000_PPRD §3.16 · IDL-31, IDL-35 · builds on 013 (budget model) and 014 (profiles) · hazard-v2 consumes 017 validation data

---

## 1. Overview

### 1.1 Purpose
Make cross-layer combined steering **real**: features from multiple layers, each applied through the SAE
trained on its own layer, in one generation — with the same trust and never-start-useless guarantees the
cluster arc established, extended across layers.

### 1.2 User Problem
Hooks already register per-layer (`_register_steering_hooks` groups by `feature.layer`), but every hook
shares the single loaded SAE — a feature placed on another layer would be steered with a decoder direction
from the *wrong layer's basis*. The UI hides this by pinning features to the selected SAE's layer, so
cross-layer circuits (e.g. L13 Condolences + L14 Reassurance) simply cannot be steered today.

### 1.3 Solution
- **Per-(SAE, layer) application:** requests carry per-feature `sae_id`; every referenced SAE loads; each
  layer's hook steers through its own SAE. Mismatched layer/SAE configs are rejected, never mis-served.
- **Per-layer budgets + global λ:** IDL-29 runs independently per layer; one intensity dial scales the
  circuit; new formula id `freq-budget/sim-alloc/per-layer@1`.
- **Hazard detection:** compounding and cancellation across layers are surfaced — warned, never silently
  corrected.

## 2. User Stories
- **US-1:** I select features from the L13 SAE and the L14 SAE and run one Blended generation; each feature
  steers at its own layer through its own SAE, and the applied-features summary shows layer + SAE per member.
- **US-2:** Selecting a multi-layer circuit computes per-layer starting strengths automatically; I dial the
  whole thing with one λ slider and never hand-tune to get a usable first run.
- **US-3:** When my configuration steers an upstream feature that is known (or weight-suspected) to drive a
  downstream steered member, I see a compounding warning naming the pair before I generate.
- **US-4:** If I try to steer an L14-trained feature at L10, the request is rejected with a clear message —
  not silently served through the wrong decoder.
- **US-5 (agent):** Via MCP I submit a combined request whose features span two SAEs and poll the result;
  `features_applied` reports per-member layer/SAE so I can verify contribution.

## 3. Functional Requirements

### 3.1 Multi-SAE application (BR-001)
1. `SteerFeature` gains optional `sae_id`; omitted ⇒ the request-level `sae_id` (back-compatible).
2. The steering service loads every referenced SAE (existing cache), groups features by `(sae_id, layer)`,
   and passes each group its own `LoadedSAE` to the hook factory.
3. A feature whose `layer` ≠ its SAE's trained layer ⇒ 422 with the offending member listed. No silent
   single-decoder serving, ever.
4. Only SAEs referenced by the active configuration are loaded; exit-mode frees all.

### 3.2 Verification & titling (BR-002)
5. `features_applied` entries gain `sae_id` (layer already present); the Blended result's applied summary
   groups members by layer.
6. Result titles follow the IDL-28 chain (profile/circuit name → display token → "Blended (N features)"),
   unchanged semantics, now spanning layers.

### 3.3 Per-layer budgets + global λ (BR-003)
7. A multi-layer allocation endpoint (extension of 013's) partitions members by layer, runs IDL-29 per
   layer against that layer's SAE decoder, and returns per-layer `{B, B_dir, G, weights, strengths, flags}`
   + the global composition under formula id `freq-budget/sim-alloc/per-layer@1`.
8. λ∈[0,2] (default 1) scales all layers' strengths at generation time (013's dial, unchanged UX).
9. Budget-preserving rebalance operates within a layer (pin/redistribute per IDL-29); cross-layer manual
   moves are out of scope v1.
10. Single-layer configurations behave byte-identically to 013 (regression guarantee).

### 3.4 Hazard detection v2 (BR-004 as amended by BR-024)
11. **Compounding:** when a steered member at Lᵢ is joined to a steered member at Lⱼ>Lᵢ by a **validated
    positive edge** (rung ≥2, from 018's stored circuits), the warning **quantifies the expected
    double-counting from the measured effect size** ("validated edge, ES=X — combined influence
    ≈ Y× the naive sum"). Where no validated edge exists, the weight-prior heuristic (IDL-32) still
    warns but every heuristic-sourced warning is **labeled `heuristic`** per the evidence-ladder
    language rules (IDL-35).
12. **Cancellation:** per-layer G flags per IDL-29; validated NEGATIVE edges between co-steered members
    produce quantified cancellation warnings; opposite-signed heuristic pairs warn with the `heuristic`
    label.
13. Warnings never mutate the configuration; generation proceeds if the user continues.
14. **Sequencing:** 015 ships with heuristic-labeled hazards only; the validated-ES upgrade activates
    when 017/018 land (the hazard module consumes stored edges by rung — no rework).

## 4. User Interface
- **Steering panel:** the SAE selector becomes additive — features carry an SAE/layer chip on their tiles
  (the existing `L{n}` chip gains the SAE association); the Cluster-profiles list shows multi-layer
  profiles with a layer-span badge (e.g. "L13+L14").
- **Budget bar:** one bar per layer (stacked compactly) + the single global λ dial (unchanged position).
- **Hazard warnings:** amber banner above Generate, listing pairs + evidence source; dismissible per run.
- **Applied summary:** grouped by layer with per-member SAE id on hover.

## 5. API / Integration
- `POST /api/v1/steering/async/combined`: `features[].sae_id` optional (additive); response
  `features_applied[].sae_id`.
- `POST /api/v1/steering/cluster-allocation`: accepts mixed-layer members with per-member `sae_id`;
  returns per-layer allocations + composition block. Single-layer requests unchanged.
- MCP: `steer_combined` + `compute_cluster_allocation` schemas gain the optional per-member `sae_id`;
  guardrail messages mention the layer-count envelope.
- 014 profiles: `ProfileMember` already carries no layer — multi-layer profiles are Feature 018's circuit
  records; this feature only *consumes* them (loading sets per-member sae_id/layer).

## 6. Data / Types
- Backend: `SteerFeature.sae_id: Optional[str]`; `FeatureSteeringConfig.sae_id`; allocation
  request/response extensions; no migrations.
- Frontend: `SelectedFeature.sae_id?: string` (defaults to selected SAE); steeringStore multi-SAE
  awareness (per-layer budget map replaces the single `clusterBudget` when layers > 1).

## 7. Dependencies
- 013 allocation service (per-layer reuse), 014 profile loading (single-layer unchanged), 018 circuits
  (multi-layer member source + rung-typed edges for hazards — heuristic-labeled until 017/018 land),
  017 validation data (hazard-v2 effect sizes), IDL-32 weight prior, IDL-35 ladder language,
  steering worker lifecycle fixes (2026-07-18).

## 8. Success Criteria
1. A two-SAE (L13+L14) Blended run verifiably applies every member through its own SAE
   (`features_applied` reports both layers; E2E).
2. Layer/SAE mismatch rejected with actionable 422 (unit + API test).
3. Multi-layer allocation returns per-layer budgets; single-layer path regression-identical to 013
   (golden test).
4. λ scales all layers; rebalance stays within-layer (store tests).
5. Compounding warning quantifies ES on a validated-edge fixture; heuristic-only pairs carry the `heuristic` label (unit + integration).
6. VRAM: loading 2 extra 8k SAEs adds <200 MB (measured, logged).

## 9. Non-Goals
- Joint cross-layer budget calibration (future, empirical); cross-layer manual rebalance; >single-model
  configurations; miLLM multi-SAE serving (follow-on BRD); automatic hazard correction.

## 10. Testing Requirements
- Backend unit: (SAE,layer) grouping; mismatch rejection; per-layer allocation partition; composition
  formula id; single-layer golden regression.
- Backend integration: two-SAE combined generation on GPU (real hooks, both layers verified).
- Frontend: store multi-SAE selection, per-layer budget map, λ scaling, hazard banner rendering.
- E2E: load a two-layer circuit → Blended run → applied summary shows both layers → title = circuit name.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-001 (own-layer SAE application, no silent single-decoder) | §3.1 |
| BR-002 (verification + circuit titling across layers) | §3.2 |
| BR-003 (per-layer budgets + global λ + formula id) | §3.3 |
| BR-004 (hazards surfaced, never auto-corrected) | §3.4 |
| BR-024 (CIRCUITS-002: hazards grounded in validated effect sizes; heuristic labeled) | §3.4 items 11–12, 14 |
