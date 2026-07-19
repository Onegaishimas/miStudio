# Technical Design Document: Multi-SAE Cross-Layer Steering

**Document ID:** 015_FTDD|MultiSAE_Steering
**Version:** 1.0
**Status:** Planned
**Related:** 015_FPRD · IDL-31 · consumes 013 (allocation) + 014 (profiles) · consumed by 018 (circuits); hazard-v2 data from 017

---

## 1. Steering path changes

```
steer_combined request { sae_id, features[{feature_idx, layer, strength, sae_id?}] }
  └ steering_tasks.steering_combined_task
      └ SteeringService.generate_combined
          ├ resolve_sae_map: {sae_id → LoadedSAE} for the DISTINCT sae_ids referenced
          │   (per-feature sae_id ?? request sae_id); each via existing load_sae cache
          ├ validate: for every feature, feature.layer == sae_map[feature.sae_id].layer
          │   else 422 listing offenders (BR-001 — never silently mis-serve)
          └ _register_steering_hooks(model, sae_map, configs)
              groups by (sae_id, layer) → one hook per layer, created with THAT group's LoadedSAE
```

Key mechanical fact: `_create_steering_hook(sae, layer_features)` already takes the SAE per hook —
the change is threading a `sae_map` through `_register_steering_hooks` instead of a single `sae`
(`steering_service.py:1090`), plus per-feature `sae_id` resolution. Baseline generation, cleanup, and the
solo/compare paths are untouched (they remain single-SAE).

`features_applied` entries gain `sae_id` (source of truth: the config used at hook time, not the request).

## 2. Per-layer allocation

`POST /steering/cluster-allocation` with mixed-layer members:

```jsonc
// request members gain sae_id (required when layers differ)
// response (multi-layer form; single-layer form unchanged for 013 compat)
{
  "formula_id": "freq-budget/sim-alloc/per-layer@1",
  "layers": {
    "13": { "sae_id": "sae_2d07…", "B": 1.05, "B_dir": 1.05, "G": 0.41, "f_eff": 0.61,
             "weights": [...], "strengths": [...], "flags": [] },
    "14": { … }
  },
  "hazards": [ { "type": "compounding", "up": {"layer":13,"feature_idx":712},
                 "down": {"layer":14,"feature_idx":231}, "evidence": "weight_prior:0.62" } ],
  "strengths": [...]   // flattened in request member order (client convenience)
}
```

- Partition members by layer → run the existing IDL-29 pipeline per partition against that layer's SAE
  decoder (`compute_allocation` reuse — no formula changes).
- **Single-layer requests return the 013 shape byte-identically** (golden-tested); the multi-layer shape
  only appears when >1 distinct layer is present.
- Hazards v2 (FPRD §3.4, BR-024): computed server-side — PRIMARY source is stored circuit edges at
  rung ≥2 (018), whose measured ES quantifies the warning ("expected double-counting ≈ ES×strength");
  fallback is the IDL-32 weight prior above a threshold (default 0.5, config) for steered pairs only
  (≤20×20 dot products), with `evidence: "heuristic:weight_prior=…"` labeling per IDL-35. Hazard model:
  `{type, up, down, evidence, rung, quantified_effect?}`.

## 3. Frontend state

- `SelectedFeature.sae_id?: string` — set on add (selected SAE) and on circuit load (member's SAE).
- steeringStore: `clusterBudget: ClusterBudget | null` generalizes to
  `layerBudgets: Record<number, ClusterBudget> | null` (single-layer keeps the existing field populated
  too — components migrate incrementally); λ (`intensity`) unchanged, applied to all layers.
- Rebalance: `rebalance(layer, instance_id, strength)` — within-layer only (IDL-29 math per layer).
- ClusterBudgetBar renders one row per layer with the layer chip; λ dial unchanged.
- Hazard banner component consumes `allocation.hazards` (amber, dismiss-per-run, lists pairs+evidence).

## 4. VRAM & guardrails

| Concern | Handling |
|---|---|
| N SAEs resident | load only referenced (map keys); 8k×2048 fp32 ≈ 130 MB each; envelope documented ≤4 extra |
| exit-mode | frees ALL cached SAEs (existing cache clear covers the map) |
| MCP guardrails | steer_combined docstring: layer-count envelope note; existing 2-in-flight guardrail unchanged |
| worker lifecycle | busy-marker/reconcile fixes (2026-07-18) apply unchanged — one task at a time regardless of SAE count |

## 5. Type changes

- Backend: `SteerFeature.sae_id?`, `FeatureSteeringConfig.sae_id`, `AppliedFeature.sae_id`,
  allocation request/response models (multi-layer variants), hazard model. No DB migrations.
- Frontend: `SelectedFeature.sae_id?`, `LayerBudgets` map, `Hazard` type, per-layer budget-bar props.
- MCP: `steer_combined`/`compute_cluster_allocation` param schemas gain per-member `sae_id?`.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Silent regression of the single-SAE path | golden tests: single-layer allocation response byte-equal; solo/compare untouched by construction |
| Wrong-SAE decoder used for a layer | the 422 validation + a unit pin asserting each hook received the SAE whose layer matches |
| VRAM creep with many layers | referenced-only loading + documented envelope + guardrail note |
| Hazard false positives annoy | weight-prior threshold configurable; warnings dismissible; evidence string shows WHY |
| 017/018 not yet landed (validated edges absent) | hazards run heuristic-labeled only — explicitly designed order; the module consumes edges by rung so the upgrade is data arrival, not rework |
