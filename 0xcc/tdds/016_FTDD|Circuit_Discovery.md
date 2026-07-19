# Technical Design Document: Capture, Statistical Mining & Attribution

**Document ID:** 016_FTDD|Circuit_Discovery
**Version:** 2.0 (reworked for BRD-MIS-CIRCUITS-002)
**Status:** Planned
**Related:** 016_FPRD · IDL-32, IDL-36 · normative math: CIRCUITS-002 A.3/A.4/A.6 · shares weight
resolvers with 015 · feeds 017 (validation) + 018 (typing/rung/review)

---

## 1. Capture store design

```
/data/circuit_captures/{cap_<hex12>}/
  manifest.json      # corpus {dataset_id, tokenization_id, sample_cap}, layers [{layer, sae_id,
                     #  threshold_mode, epsilon}], split {method: per_document, ratio: 0.8, seed,
                     #  heldout_docs[]}, counts, bytes, sae_fingerprints, attention_capture?:
                     #  {layers, heads, top_k}, created_at, stale
  layer_13.events    # sorted columnar: (doc_id u32, token_pos u16, feature_idx u16, act f16)
  layer_13.index     # feature_idx → [start, end) offsets
  layer_13.errnorm   # per-(doc, token) ‖ε‖ f16 (dense over captured tokens — small)
  attn_l12.topk      # OPTIONAL: (doc, t_q, head, t_k, mass f16) top-k keys per query
```

- Sparse events: `act > max(θ_floor, ε_thresh·max_activation_i)`; measured sparsity 2–10% ⇒ reference
  corpus ≈ 100–500 MB/2 layers at f16.
- **Positions are first-class** (u16 column + sort order) — the Tier-2.5 join key exists from day one.
- **Error norms** per (layer, token) stored always (dense but scalar — ~2 bytes/token/layer); full ε
  vectors optional per run (config flag; large).
- **Split at capture time:** per-document 80/20 (seeded RNG recorded); `heldout_docs` listed in the
  manifest so every downstream run uses the same partition.
- ONE forward pass per batch captures all selected layers (multi-layer HookManager) → per-layer encode →
  threshold → append. Cost estimate from a ~32-sample probe.

## 2. Statistics (A.3 — normative)

```
mine(store, granularity, mode):
  units = features | supernodes(profiles with cohesion ≥ floor; A_C(t) = max over members)
  binarize F_u(t) at capture θ over DISCOVERY docs only
  pairs: seeded → up ∈ seed units; open → up with support ≥ floor; down ∈ layers > up.layer
  for each pair: n_ud via merge-join on (doc, pos) → PMI = log(n_ud·N / n_u·n_d); Spearman secondary
  support filter: n_ud ≥ s_min (20)
  null: K shuffles (default 100) of within-document CIRCULAR SHIFTS per unit → null PMI distribution
        → threshold = percentile (default 99th)   # naive whole-corpus permutation FORBIDDEN (pinned)
  FDR: Benjamini–Hochberg q=0.05 on empirical p-values vs null; discipline recorded in report
  held-out: recompute PMI + null on HELDOUT docs; replicated = exceeds held-out threshold, same sign
  report: {null_summary, fdr, replication_rate, counts_by_stage, granularity, params}
```

Supernode drill-down = the same pipeline restricted to two clusters' members (on demand, review UX).

## 3. Tier-2 attribution (A.6 — normative; IDL-36)

```
attribution(run, candidates, prompts):
  model with SAE pass-through at captured layers: x_l := x̂_l + ε_l   (weights frozen; grads to f, ε)
  group candidates by DOWNSTREAM unit d:
    m = mean_t a_d(t)  →  ONE backward per (prompt, d)
    for each upstream u in group: attr(u→d) += Σ_t ∂m/∂a_u(t) · a_u(t)
  aggregate over prompts: mean attr + sign-consistency fraction
  rung-1 gate: sign agrees with mined association AND |attr| ≥ run percentile floor
  persist per-candidate {attr, sign_consistency, method: raw|ig_lite}; keep BOTH orderings
  (coactivation-only and attribution-re-ranked) for 017's uplift measurement
```

- Memory (RSK-009): frozen SAEs, activations-only grads, gradient checkpointing over blocks; per-layer
  subset attribution as the pressure valve. Envelope metric: ≤1.5× capture wall-clock.
- Supernode attribution: `m = mean_t A_C(t)` (differentiable via the max — subgradient to the argmax
  member; document the choice).

## 4. Run/report persistence

- `circuit_capture_runs`: manifest mirror + status.
- `circuit_discovery_runs`: params, **report JSONB** (null summary, FDR, replication rate, stage
  counts, attribution envelope), candidates JSONB
  `[{up, down, granularity, stats: {pmi, lift, support, spearman, null_pct}, replicated_heldout,
     attribution: {score, sign_consistency, method} | null, orderings: {coact_rank, attr_rank}}]`
  (cap default 2000, truncation noted). Edge typing/rung live in 018's layer — discovery emits raw
  candidates; 018's classifier annotates.

## 5. Architecture / types

```
CircuitCaptureService — probe/estimate, capture task, store IO, split, manifest, stale flagging
CircuitStatsService   — binarization, merge-join counts, PMI, null shuffles, FDR, held-out replication
CircuitDiscoveryService — unit selection (feature/supernode), pipeline orchestration, report assembly
CircuitAttributionService — pass-through wiring, backward batching by downstream, scores, envelope
endpoints: circuit_capture.py / circuit_discovery.py (+ /attribution sub-route); WS channels
MCP tools/circuits.py (category "circuits"): 5 tools per FPRD §5
frontend: CircuitsPanel (Capture + Discovery tabs), circuitStore, types
docs deliverable: 0xcc/docs/tier-2.5-attention-mediated-mining.md (from A.8)
```

## 6. Risks

| Risk | Mitigation |
|---|---|
| Store blowout (positions/errnorm/attention) | probe estimate + confirm gate; errnorm is scalar; attention sidecar opt-in top-k |
| Null miscalibration (RSK-011) | circular-shift construction pinned by unit test (marginal rates preserved); audit fixture sanity check |
| Base-rate pairs still dominate | PMI (not counts) + support floor + FDR — planted-pair tests assert high-base-rate pairs do NOT surface |
| Supernode incoherence (RSK-012) | cohesion floor eligibility; feature granularity always available; activation definition recorded per run |
| Backward-pass OOM (RSK-009) | checkpointing + layer subsets + envelope metric in CI-adjacent GPU test |
| Split leakage | per-document split, seeded, recorded; unit test: no doc in both partitions |
| Attribution on saturated features | IG-lite option documented + selectable; raw is the disclosed default |
