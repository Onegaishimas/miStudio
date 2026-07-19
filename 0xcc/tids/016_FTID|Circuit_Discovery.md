# Technical Implementation Document: Capture, Statistical Mining & Attribution

**Document ID:** 016_FTID|Circuit_Discovery
**Version:** 2.0 (reworked for BRD-MIS-CIRCUITS-002)
**Status:** Planned
**Related:** 016_FPRD · 016_FTDD · IDL-32, IDL-36 · CIRCUITS-002 Appendix A is normative

---

## 1. Implementation Order

1. Store IO (events + index + errnorm + optional attention sidecar) — pure, no GPU.
2. Tables + migration + capture service/task/endpoints (probe, estimate, split-at-capture).
3. Stats service (PMI, circular-shift null, FDR, held-out) — synthetic-store tested.
4. Discovery service (granularities, orchestration, report) + endpoints.
5. Attribution service (pass-through, batched backward, scores, envelope) + sub-route (GPU).
6. MCP `circuits` tools + category gating.
7. Frontend CircuitsPanel (Capture/Discovery tabs, report card).
8. Tier-2.5 design doc + manual + GPU integration + E2E.

## 2. File-by-file

### 2.1 `backend/src/services/circuit_capture_store.py` (NEW)
- `EventWriter/EventReader` per FTDD §1 (numpy structured memmap; sorted; per-feature index);
  `ErrNormWriter/Reader` (dense scalar per (doc, token)); `AttnTopKWriter/Reader` (optional sidecar).
- Assert `feature_idx < 65536` at write. Unit round-trips for all three artifact kinds.

### 2.2 `backend/src/models/circuit_runs.py` (NEW) + Alembic
- `circuit_capture_runs` (`cap_`, manifest JSONB incl. split + fingerprints, status, stale);
  `circuit_discovery_runs` (`dsc_`, params, report JSONB, candidates JSONB w/ both orderings).
  Single-head check; up+down tested.

### 2.3 `backend/src/services/circuit_capture_service.py` + `workers/circuit_capture_tasks.py` (NEW)
- Probe (~32 samples) → estimate {events, bytes, minutes}. `confirm=true` enqueues.
- Capture batch loop: ONE forward pass, multi-layer HookManager capture → per-layer `sae.encode`
  on-GPU → threshold `max(θ_floor, ε·max_act_i)` (missing max_act ⇒ floor-only, never skip) →
  writer.append + errnorm append (+ attention top-k when enabled). Progress/cancel between batches.
- **Split at capture:** seeded per-document 80/20; `heldout_docs` in manifest; unit test: disjoint.
- `sae_fingerprints` (decoder shape+sum hash); stale-flag hook on SAE update.

### 2.4 `backend/src/services/circuit_stats_service.py` (NEW)
- `binarize(reader, units, docs)`; `pair_counts` merge-join on (doc, pos); `pmi/lift/spearman`.
- `circular_shift_null(K=100)`: per-unit per-document shift — **unit-pin that marginal rates are
  EXACTLY preserved** and that whole-corpus permutation is not reachable from config.
- `bh_fdr(pvals, q=0.05)`; `heldout_replication(candidates)` on `heldout_docs` with its own null.
- Supernode units: `A_C(t) = max` (mean alt.) from profile members; cohesion floor from profile budget.
- All pure over readers — exhaustively testable on synthetic stores (planted edges, planted
  high-base-rate pairs that must NOT surface, planted echoes for 018's fixture).

### 2.5 `backend/src/services/circuit_discovery_service.py` (NEW) + task
- Orchestration per FTDD §2: unit selection (granularity/mode), stats pipeline, report assembly
  (null summary, FDR discipline, replication rate, stage counts), candidate persistence (cap 2000,
  truncation noted). Uncovered-seed-members reported. Stale-store refusal (`force` override).

### 2.6 `backend/src/services/circuit_attribution_service.py` (NEW) + task
- SAE pass-through module: wrap captured layers so `x := x̂ + stopgrad-free(ε)` with
  `requires_grad_(False)` on SAE params — **numerical-identity unit test** (pass-through output ==
  clean output within tolerance) is the guard against the classic ε-dropping mistake.
- Backward batching grouped by downstream unit; `attr = Σ g·a`; IG-lite (α ∈ {¼,½,¾,1}) behind a flag.
- Gradient checkpointing over blocks; per-layer-subset fallback; wall-clock + peak-VRAM recorded in the
  report (envelope ≤1.5× capture).
- Toy-model analytic test: 2-layer linear toy where attr is computable by hand.

### 2.7 Endpoints + MCP + WS
- `circuit_capture.py`, `circuit_discovery.py` (+ `/{id}/attribution`); WS `capture/{id}`,
  `discovery/{id}`, `attribution/{id}`. House error/pattern conventions.
- `mcp_server/tools/circuits.py`: 5 tools (FPRD §5); new `circuits` category via gating + compose/k8s
  env defaults + mcp-server.md (mirror the 014 `profiles` checklist). Tool docstrings carry the lag-0
  disclosure line (IDL-35 language discipline).

### 2.8 Frontend
- `CircuitsPanel.tsx` (nav "Circuits"), `circuitStore.ts`, `types/circuits.ts`, `api/circuits.ts`.
- Capture tab: config (+attention toggle) → estimate card → confirm; runs list (WS progress, size,
  split, stale, delete).
- Discovery tab: granularity toggle (**Cluster default when mode=Seeded**), floors, run; candidate
  table (labels, PMI, support, null-pct, attribution when present, replicated ✓); run-report card
  pinned above results; lag-0 notice; "Run attribution pass" action.

### 2.9 Docs deliverables
- `0xcc/docs/tier-2.5-attention-mediated-mining.md` — expand A.8's five points into the fast-follow
  design (capture join, candidate generation, shuffled-join null, frozen-attention attribution,
  position-restricted validation, contract fields already present).
- `manual/docs/core-workflow/circuits.md` (capture → mine → attribute loop; evidence honesty framing);
  mcp-server.md += circuits category.

## 3. Pitfalls

- **ε-dropping** in attribution pass-through — the numerical-identity test is non-negotiable.
- Circular-shift null must shift WITHIN documents; crossing boundaries inflates significance.
- Held-out must reuse the capture-time split — never re-split at mine time.
- Supernode max is piecewise-differentiable: attribution subgradient goes to the argmax member per
  token (document; don't silently mean).
- u16 positions cap at 65535 tokens/doc — assert at capture (corpus docs are ≤512 tokens anyway).
- Attribution batching: group by downstream FIRST (cost is O(prompts × downstreams)); a per-candidate
  backward loop is the naive trap that makes Tier-2 unaffordable.
- Candidates JSONB cap: keep BOTH orderings even when truncating (truncate by best-of-either-rank).
- The word "causal" must not appear in any 016 surface — this feature produces rungs 0–1 only
  (copy-audit shared with 017/018).

## 4. Testing

- Unit: store round-trips ×3 artifact kinds; PMI/planted-edge/planted-base-rate; null marginal-rate
  pin; FDR; split disjointness; supernode activation + cohesion gate; attribution toy-model analytic;
  pass-through numerical identity; estimator.
- Integration (GPU): mini-capture (positions + errnorms verified); attribution envelope on the 1.2B
  model (peak VRAM < 24 GB, wall-clock recorded).
- API: estimate/confirm, cancel, stale+force, hostile configs.
- Frontend: report card, granularity defaults, evidence columns, attribution action.
- E2E: capture → seeded cluster-granularity discovery → attribution → report (replication rate,
  envelope); screenshot `0xcc/caps/miStudio_Circuit_Discovery_<date>.png`.
