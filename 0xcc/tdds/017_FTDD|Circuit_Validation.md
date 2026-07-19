# Technical Design Document: Intervention, Validation & Faithfulness

**Document ID:** 017_FTDD|Circuit_Validation
**Version:** 1.0
**Status:** Planned
**Related:** 017_FPRD · IDL-34 · CIRCUITS-002 A.5/A.7 normative · consumes 016 (store, candidates, null machinery) + 015 (hook machinery) · rungs into 018's ladder

---

## 1. Intervention engine

```
_create_suppression_hook(sae, feature_idx, a_base, positions=None)   # steering-hook variant
  out[:, t, :] -= (a_u(t) − a_base) · W_dec[:, i]     for t in positions (v1: all captured-fire positions)
  # ε untouched BY CONSTRUCTION — we subtract from the residual, never re-decode.
  # Pin: ‖x'(t) − x(t) + (a_u(t)−a_base)·d_i‖ ≈ 0  and  ε recomputed pre/post identical.
```

- `a_u(t)` from a same-pass encode at the hook point (encode is cheap; reuse the layer capture hook).
- `a_base`: 0 (default) | corpus-mean of `u` from the 016 store (manifest-recorded).
- Measurement pass: greedy, fixed seeds, teacher-forced prompt windows (no sampling loop) — matched
  clean/intervened pairs.

## 2. Edge validation pipeline

```
validate(run_id, scope{ordering, K}, cfg{null_K, percentile, sign_frac}):
  for edge (u→d) in top-K of chosen ordering:
    prompts = windows around u's strongest firings (016 store + tokenization — SAME tokenization as capture)
    for each prompt: clean pass → a_d(t); intervened pass → a_d(t)
      Δ_p = mean_t[a_d clean − a_d intervened] over tokens where clean F_u(t)=1
    ES = mean_p(Δ_p) / σ_d          # σ_d from capture store — no extra passes
  null: shuffled NON-edge pairs (random u'∈same layer support-matched, d fixed) → null ES distribution
        (reuses 016's null machinery pattern; separate from the mining null)
  status: rung2 iff |ES| > null percentile AND sign-consistent ≥ sign_frac; else tested_and_failed
  batch: survival per rank tier per ordering → uplift = survival(attr order) − survival(coact order)
  persist: manifest per batch {intervention, baseline, prompts, seeds, cfg, null summary, per-edge values}
```

## 3. Faithfulness runner (A.7)

```
faithfulness(circuit, mode, k=256/layer, metric=compare_output_shift):
  M = circuit members (cluster_ref members expand to features)
  B_clean       = metric(prompts)                       # compare-workflow measures, reused
  B_ablate_M    = metric with simultaneous BR-017 suppression of all m∈M
  B_ablate_all  = metric with suppression of ALL features at circuit layers (per-layer top-N proxy where
                  literal-all is intractable; N recorded)
  necessity   = (B_clean − B_ablate_M) / (B_clean − B_ablate_all)
  sufficiency = (B_ablate_topk_nonmembers − B_ablate_all) / (B_clean − B_ablate_all)   # mode=both
  persist scores + {k, metric_id, prompts, manifest_ref} on the circuit (018 record + contract)
```

- Behavior metric v1 = the compare workflow's output-shift measures (continuity with the 2-validated-
  profiles bar); metric identity always recorded — the open question stays visible in the manifest.

## 4. Manifests

`validation_manifests` table: `vman_` id, kind (`edge_batch` | `faithfulness` | `reproduction`),
payload JSONB (everything §2/§3 lists), parent refs (discovery run / circuit), created_at.
`POST /{id}/reproduce` re-executes from the payload and stores a `reproduction` manifest with deltas —
the acceptance test asserts tolerance. Manifest ids travel into 018's contract (`validation_manifest_ref`).

## 5. Remediation map

| Surface | Change |
|---|---|
| `analysis_service.calculate_ablation` | docstring corrected; response `method: "statistical_estimate"` |
| Feature Detail modal ablation section | retitle "Impact estimate (statistical — no model inference)" |
| MCP `get_feature_ablation` | docstring relabel; points to real validation |
| shared copy-audit suite | "causal" forbidden below rung 2 across 015–018 surfaces |

## 6. Architecture / types

```
CircuitInterventionService — suppression hooks, matched passes, ES, null, batch orchestration (GPU task)
CircuitFaithfulnessService — member expansion, multi-feature suppression, metric reuse, scores (GPU task)
ManifestService — persist/retrieve/reproduce
endpoints: circuit_validation.py, circuit_faithfulness.py, validation_manifests.py; WS channels
MCP: validate_circuit_edges, get_validation_manifest, reproduce_validation, run_circuit_faithfulness
tables: validation_manifests (+ results written into discovery-run candidates / circuit records)
frontend: Validation tab in CircuitsPanel; manifest drawer; faithfulness display on circuit cards (018)
```

## 7. Risks

| Risk | Mitigation |
|---|---|
| ε mishandled (the canonical mistake) | directional subtraction never re-decodes; ε-invariance pin; reconstruction-swap variant (if built) has a re-add-ε pin |
| Noise floor / null miscalibration | shuffled-non-edge null support-matched; deterministic greedy passes; reproduction test |
| "Ablate all at layers" intractable | per-layer top-N proxy with N recorded and disclosed on the badge |
| Metric ambiguity (open question) | metric identity in every manifest; default = compare measures; swap = new manifests, old ones stay reproducible |
| GPU cost of faithfulness | sampled prompts; necessity-only mode; steering-mode discipline + 2026-07-18 worker fixes |
| Sequencing (rung enum lives in 018) | 018's shared ladder model lands first (one enum, IDL-35); 017 imports it |
