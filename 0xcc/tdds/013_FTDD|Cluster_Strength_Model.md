# Technical Design Document: Cluster Strength Budget Model

**Document ID:** 013_FTDD|Cluster_Strength_Model
**Version:** 1.0
**Status:** Planned
**Related:** 013_FPRD · IDL-29 · IDL-27 (solo law) · pressure-tested design (2026-07-16)

---

## 1. Derivation (why this formula, not a guess)

The steering hook adds `v = Σ strengthᵢ·dᵢ` to every token's residual stream
(`steering_service.py:903-1056`); decoder columns dᵢ are unit-norm on the SAEs in use; hooks register per
layer (`:1058-1098`). The only empirically-validated magnitude knowledge we own is the **solo law**
(IDL-27, experiment c4a273f1): injecting along a single decoder direction, the optimal coefficient is
`b(f) = clamp(a − b·f, m, M)` with (a,b,m,M) = (2.9, 2.6, 1.0, 3.0).

For a cluster, what the model *feels* is the injected vector v — its magnitude ‖v‖ and direction v̂. With
weights wᵢ (Σwᵢ=1) and total budget B: `v = B·Σσᵢwᵢdᵢ`, hence `‖v‖ = B·G` where **G = ‖Σσᵢwᵢdᵢ‖₂**.
Setting ‖v‖ equal to the solo-optimal magnitude for the cluster's direction, `B = B_dir/G`. Everything else
follows:

- **B_dir** comes from the solo law applied at cluster level with `f_eff = Σwᵢfᵢ/Σwᵢ` — the same weights
  that form the direction form its effective frequency (internally consistent centroid rationale).
  Rejected alternatives: union frequency `1−Π(1−fᵢ)` assumes independence that is maximally wrong for
  co-firing clusters and saturates → pins B_dir at the floor for large clusters; max(fᵢ) lets one dense
  straggler zero the headroom.
- **Sanity anchors:** identical members ⇒ G=1 ⇒ B=B_dir, each B_dir/N — exactly the constant-budget
  heuristic (~2.1 regardless of N) the MCP agent discovered empirically. Orthogonal members, equal weights
  ⇒ G=1/√N ⇒ B=B_dir·√N (independent directions add energy in quadrature). N=1 ⇒ b(f) exactly.
- **Failure mode, correctly framed:** small G does NOT inflate ‖v‖ (it's B_dir by construction). It means
  members nearly cancel — v̂ becomes cancellation *residue*, semantically garbage at any magnitude, and the
  displayed per-member strengths become absurd. Therefore the guards are a **coherence gate** (warnings +
  cohesion-based downgrade), plus a numeric floor `G ≥ 0.05` and the interpretable cap `B ≤ Σb*(fᵢ)` (mean
  member strength never exceeds mean solo optimum; when the cap binds, ‖v‖ < B_dir — conservative).
- **Why sim-proportional weights:** v̂ = Σwᵢdᵢ/G — weighting by sᵢ biases the direction toward the most
  cluster-representative members ("steer toward the cluster's meaning"). Inverse weighting favors outliers;
  uniform discards information. Grouping threshold 0.35 bounds sims to [~0.35,1] ⇒ ratios ≤ ~3:1, no
  winner-take-all. No softmax/temperature (free parameter, nothing to fit it to). No re-baselining to
  (s−0.35) (over-concentrates without evidence).
- **Cut by design review:** the offline `√N/√(1+(N−1)ρ)` approximation with ρ≈mean similarity — identifying
  *context* similarity with *decoder* cosine was the one unprincipled step. Fallback is `G=1` (constant
  budget: empirically decent, errs weak).
- **Honest caveat (recorded):** b(f) was fit along decoder directions (the model's own write-directions);
  a loose cluster's v̂ is an arbitrary residual direction — extrapolation. The cohesion gate confines the
  model to the regime resembling the fit.

## 2. Formula set (normative — implement exactly)

Members i=1..N (sim sᵢ, freq fᵢ, sign σᵢ∈{±1} default +1, layer L identical for all):

| # | Step | Definition |
|---|------|------------|
| 1 | Weights | s̃ᵢ = sᵢ if present, else mean(present sims); all missing ⇒ s̃ᵢ=1. wᵢ = s̃ᵢ/Σs̃ⱼ |
| 2 | Eff. frequency | f_eff = Σ_{f known} wᵢfᵢ / Σ_{f known} wᵢ; none known ⇒ skip, B_dir=2.0 (`default`) |
| 3 | Direction budget | B_dir = clamp(a − b·f_eff, m, M) |
| 4 | Gain | G = ‖Σσᵢwᵢdᵢ‖₂ (decoder, fp32 accumulate); unavailable ⇒ G=1, flag `approximate` |
| 5 | Total budget | B = min( B_dir / max(G, 0.05), Σb*(fᵢ) ), b*(missing f)=2.0 |
| 6 | Allocation | strengthᵢ = σᵢ·B·wᵢ, round 0.1; residual → largest unpinned member |
| 7 | Flags | `cancellation` if G < 1/√N ∧ all σᵢ=+1 (+ min-pairwise-cosine pair); `low_cohesion` if group cohesion < threshold (config, default 0.5) ⇒ recommend solo; `mixed_layer` ⇒ refuse; `approximate` per step 4 |
| 8 | Rebalance (client) | pins P; R = B − Σ_P\|strength\|; R<0 ⇒ warn, unpinned→0; else strengthᵢ = σᵢ·R·wᵢ/Σ_unpinned wⱼ. B,G fixed under strength edits |
| 9 | Intensity | appliedᵢ = λ·strengthᵢ, λ∈[0,2] default 1 (post-rebalance; pins stored pre-λ) |
| 10 | N=1 | Feature-011 solo path verbatim (incl. DEFAULT_STRENGTH=10 fallback — documented divergence from cluster default 2.0) |

Constants config (application settings): `steering_cluster_constants = {"default": {"a":2.9,"b":2.6,"m":1.0,"M":3.0,"cohesion_gate":0.5}, "per_sae": {}}`.

## 3. Edge-case table (normative)

| Case | Handling |
|---|---|
| fᵢ null/NaN/∉[0,1] for some | exclude from f_eff (renormalize step-2 weights over known-f); b*(f)=2.0 in cap |
| all fᵢ missing | B_dir=2.0, source `default`, UI notice — NOT 10·anything |
| sᵢ missing | impute mean of present sims |
| sᵢ = 0 | wᵢ=0 ⇒ strength 0; hook skips zero coefficients; UI "member inactive" |
| all sims missing/zero | uniform weights 1/N |
| N = 1 | exact solo path (incl. its 10 fallback) |
| σᵢ = −1 member | signed weight in G; budget/rebalance over \|strength\|; exempt from cancellation warning |
| near-cancelling same-sign (G < 1/√N) | warn + offending pair; suggest remove/flip |
| G < 0.05 | numeric floor; cap almost certainly binds ⇒ ‖v‖ < B_dir (conservative) |
| mixed-layer member set | refuse cluster model; per-feature solo baselines + notice |
| decoder unavailable | G=1, `approximate: true` |
| Σ\|pinned\| > B | warn, unpinned→0, amber budget bar; never silently rescale pins |
| all pinned | rebalance no-op; show drift B vs Σ\|pinned\| |
| rounding drift (0.1 grain) | fold residual into largest unpinned member |

## 4. Architecture

```
FeatureGroupsPanel (cluster hand-off, 012 context)
        │  members + stats
        ▼
steeringStore.requestClusterAllocation()
        │  POST /api/v1/steering/cluster-allocation
        ▼
backend ClusterAllocationService ── loads SAE decoder (reuses load_sae cache) ── computes steps 1–7
        │  {B, B_dir, G, f_eff, weights, strengths, flags, constants_used, approximate}
        ▼
steeringStore: apply strengths (strengthSource:'cluster'), hold clusterBudget; REBALANCE (step 8) and
INTENSITY (step 9) live client-side only — no decoder needed, budget-preserving by construction.
```

- **One source of truth:** the law is implemented once, server-side (`cluster_allocation_service.py`).
  `steeringStrength.ts` remains the solo/N=1 path only. The MCP tool calls the same service.
- The endpoint loads **decoder weights only** — no model load, no GPU generation, no steering-mode
  requirement; norm on CPU fp32 is fine (N≤20 × d_model ≤ 4k).
- Response echoes `constants_used` + `formula_id: "freq-budget/sim-alloc@1"` — consumed later by the 014
  export so definitions are self-describing.

## 5. Validation protocol (pre-ship gate, runbook in 0xcc/docs/)

1. Pick ≥3 real clusters on the LFM2.5 L12 SAE: small-coherent (N≈3–5), large-coherent (N≥10), mid-N; plus
   one low-cohesion cluster (gate test).
2. Via MCP (`compute_cluster_allocation` + `steer_combined` sweeps): sweep total budget ×{0.25,0.5,1,1.5,2}
   around predicted B at fixed weights; judge coherence/degeneracy per output (existing experiment flow,
   cf. c4a273f1).
3. One cluster: sim-proportional vs uniform weights at identical B (grounds the allocation choice).
4. Acceptance (hard): empirically-best total within ±30% of predicted B on coherent clusters AND predicted-B
   output non-degenerate on all coherent clusters AND low-cohesion cluster correctly flagged.
5. Record results in `0xcc/docs/cluster-strength-validation.md`; write fitted constants into
   `per_sae.<sae_id>` config. Failure ⇒ adjust constants (or law) BEFORE enabling cluster mode by default.

## 6. Type changes

- Backend: `ClusterAllocationRequest/Response` schemas; `cluster_allocation_service.py`; settings key;
  MCP tool `compute_cluster_allocation` (category `steering`, read-only — no approval needed).
- Frontend: `ClusterAllocation`, `clusterBudget`, `intensity`, `SelectedFeature.pinned?`,
  `StrengthSource += 'cluster'`; store actions `requestClusterAllocation`, `rebalance(instanceId, value)`,
  `setIntensity`, `togglePin`.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Law mispredicts on other SAE families | per-SAE constants + validation protocol is a shipping gate; solo baselines remain one click away |
| G computation needs decoder while steering mode off | endpoint loads decoder only (no GPU generation); worst case flag `approximate`, G=1 |
| Rebalance drift over many edits | property-based test: random edit sequences preserve Σ within 0.05·N |
| Users confused by pins/budget | budget bar + pin icons + notices; solo mode untouched for individual adds |
| Endpoint latency on cold SAE load | reuse `load_sae` cache; p50 target 500 ms warm; UI applies solo baselines immediately and upgrades when allocation arrives (progressive) |
