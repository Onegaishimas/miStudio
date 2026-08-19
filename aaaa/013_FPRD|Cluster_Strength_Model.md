# Feature PRD: Cluster Strength Budget Model

**Document ID:** 013_FPRD|Cluster_Strength_Model
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CLUSTERS-001 (BR-004, BR-005, BR-006, BR-011) · 000_PPRD §3.14 · IDL-29 · builds on 011 (solo auto-baseline) and 012 (cluster context)

---

## 1. Overview

### 1.1 Purpose
When a user steers a cluster, give them a **principled, outcome-grounded starting allocation** — a total
influence budget derived from the cluster's activation frequencies, split across members by similarity —
so tuning starts from a sensible point, never a guess. Keep the total budget invariant under manual edits.

### 1.2 User Problem
Today every cluster experiment starts from arbitrary strengths (uniform 10s, or hand-copied values). The
product owner: *"I don't want to guess, and I don't want to be perpetually starting at a completely useless
point."* MCP-driven experiments empirically converged on a constant total budget (~2.1) split across
members — evidence a lawful model exists — but nothing in the product encodes it.

### 1.3 Solution
The IDL-29 formula set: similarity-normalized weights → sim-weighted effective frequency → direction budget
via the validated Feature-011 solo law → **exact resultant-norm gain** computed server-side from the SAE
decoder → total budget → per-member strengths; plus pin-and-rebalance, coherence warnings, a master
cluster-intensity dial, and an MCP-driven empirical validation protocol gating the constants.

## 2. User Stories
- **US-1:** As a researcher, selecting a 7-member cluster instantly yields per-member strengths that
  produce coherent steered output on the first run — no manual tuning required.
- **US-2:** As a tuner, when I set member #3 from 0.3 to 0.4, the other members adjust automatically so the
  cluster's total stays within budget; my edited value stays pinned.
- **US-3:** As a researcher, one **intensity dial** scales the whole cluster (off → subtle → strong) so I
  can compare identical prompts at λ=0 / 0.5 / 1 / 2 — the same experience Open WebUI will later expose.
- **US-4:** As a user steering an incoherent (low-cohesion) cluster, I'm warned and offered per-feature
  solo baselines instead of a misleading cluster budget.
- **US-5:** As an MCP agent, I can request the computed allocation, run sweeps around it, and record
  calibration results per SAE.

## 3. Functional Requirements

### 3.1 Allocation computation (BR-004, BR-005)
1. A **server-side allocation endpoint** computes, for a set of same-layer members with stats:
   weights `wᵢ = s̃ᵢ/Σs̃ⱼ`; `f_eff = Σwᵢfᵢ/Σwᵢ` (known-f members); `B_dir = clamp(a − b·f_eff, m, M)`;
   gain `G = ‖Σσᵢwᵢdᵢ‖₂` from the SAE decoder; `B = min(B_dir/max(G, 0.05), Σb*(fᵢ))`;
   `strengthᵢ = σᵢ·B·wᵢ` rounded to 0.1 — returning `{B, B_dir, G, f_eff, weights, strengths, flags}`.
2. Equal similarities ⇒ equal strengths (BR-004 letter); all constants from per-SAE-namespaced config.
3. Missing data handled per the FTDD edge-case table (never errors, always a defensible allocation).
4. Decoder unavailable ⇒ `G = 1` (conservative constant-budget fallback), response flagged `approximate`.
5. **v1 restriction:** members must share one layer; mixed-layer sets are refused → per-feature solo
   baselines + notice.

### 3.2 Application in the UI (BR-011)
6. When features arrive via a cluster hand-off (012 context), the steering panel requests the allocation
   and applies it; tiles show `strengthSource: 'cluster'` badge with B/G surfaced in a budget bar.
7. Features added individually keep the Feature-011 solo baseline (`auto`/`default` badges unchanged);
   N=1 clusters route through the solo path verbatim.
8. Coherence flags render as warnings: near-cancellation (`G < 1/√N`, all positive) names the offending
   pair; low group `cohesion` (below config threshold) downgrades to solo baselines with a notice.

### 3.3 Budget-preserving rebalance (BR-006)
9. Manually editing a member's strength **pins** it; remaining budget `B − Σ|pinned|` redistributes across
   unpinned members by renormalized weights. Pins are visually marked and un-pinnable.
10. Over-budget pins (R < 0) warn (amber budget bar), set unpinned to 0, never silently rescale pins.
11. B and G recompute only on membership/sign changes — never on strength edits.

### 3.4 Master intensity dial
12. A cluster-level dial λ ∈ [0, 2] (default 1, step 0.05) scales all applied strengths post-rebalance;
    λ=0 previews unsteered. The dial value rides the profile/export (Feature 014).

### 3.5 Empirical validation & calibration (BR-005)
13. Constants live in config: `{default: {a,b,m,M}, per_sae: {<sae_id>: {...}}}` — recalibration without
    code change; the allocation response echoes the constants used.
14. A documented **MCP validation protocol** must pass before the model ships as default: sweeps on ≥3
    real clusters (varying N, coherence) + 1 low-cohesion cluster (gate test) + 1 sim-proportional-vs-
    uniform comparison. Acceptance: predicted B within ±30% of empirically-optimal total AND
    non-degenerate output at predicted strengths (hard gate).

## 4. User Interface
- **Budget bar** above the tiles (cluster mode): `B used/total`, G readout, amber on over-budget.
- **Tiles:** `cluster` badge; pin icon on manually-edited members; existing sliders drive rebalance.
- **Intensity dial:** compact slider next to the Blended|Compare toggle, visible in cluster mode.
- **Warnings:** inline banner for cancellation/low-cohesion/mixed-layer/approximate.

## 5. API / Integration
- **New:** `POST /api/v1/steering/cluster-allocation` — body `{sae_id, members: [{feature_idx, layer,
  similarity?, activation_frequency?, sign?}]}` → the §3.1 response. Loads decoder weights only (no model,
  no GPU generation; CPU norm computation acceptable).
- **MCP:** expose the same as a `compute_cluster_allocation` tool (category `steering`), so agents run the
  validation protocol with the product's own math.
- Config: extend application settings with the constants namespace; surfaced read-only in the response.

## 6. Data / Types
- Frontend: `ClusterAllocation` type; `SelectedFeature` += `pinned?: boolean`, `sign` implicit in strength;
  `strengthSource` union += `'cluster'`; steeringStore: `clusterBudget: {B, B_dir, G, flags} | null`,
  `intensity: number`; rebalance action.
- Backend: allocation schema (request/response); settings keys for constants; no DB migration.

## 7. Dependencies
- 011 (solo law + steeringStrength.ts), 012 (cluster context — determines when cluster mode engages).
- SAE loadable server-side (already true wherever steering works).

## 8. Success Criteria
1. Selecting a coherent cluster yields first-run allocations with no manual edits (US-1) — verified live on
   ≥2 real clusters.
2. Identical-member sanity: B ≈ B_dir; N=1 equals the solo baseline exactly (unit-tested).
3. Rebalance preserves Σ|strength| = B within rounding (0.1 grain) across any edit sequence (unit-tested).
4. Validation protocol executed and recorded (results in `0xcc/docs/` + constants committed to config);
   both acceptance gates pass.
5. Low-cohesion cluster triggers the downgrade path with visible notice.
6. Allocation endpoint p50 < 500 ms warm (decoder cached).

## 9. Non-Goals
- Multi-layer cluster budgets (refused in v1); per-SAE auto-calibration pipeline (config namespace only);
  changing the solo (individual-add) baseline path; any MILLM surface.

## 10. Testing Requirements
- Backend unit: formula math (weights/f_eff/B_dir/G/B/strengths) against hand-computed fixtures; every
  edge-case-table row; config override resolution; mixed-layer refusal.
- Backend API: endpoint contract + approximate fallback (decoder unload path).
- Frontend unit: rebalance invariants (property-style: random pin/edit sequences hold Σ within grain);
  intensity scaling; source badges.
- E2E: cluster hand-off → budget bar + allocations render → edit → rebalance → Blended run completes.
- MCP validation protocol runbook executed once pre-ship (recorded, not CI).

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-004 (budget from agg frequency; sim-relative shares; equal⇒equal) | §3.1 items 1–2 |
| BR-005 (grounded, never useless; validated) | §3.1, §3.5 |
| BR-006 (edit one ⇒ rebalance, budget preserved) | §3.3 |
| BR-011 (stats through selection) | §3.2 (consumes 011/012 plumbing) |
