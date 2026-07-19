# Task List: Capture, Statistical Mining & Attribution

**Document ID:** 016_FTASKS|Circuit_Discovery
**Version:** 2.0 (reworked for BRD-MIS-CIRCUITS-002)
**Status:** Planned
**Source:** 016_FPRD · 016_FTDD · 016_FTID · IDL-32, IDL-36 · CIRCUITS-002 A.3/A.4/A.6 normative

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Store IO | 1 task | ⏳ |
| Phase 2: Capture | 3 tasks | ⏳ |
| Phase 3: Statistics + discovery | 3 tasks | ⏳ |
| Phase 4: Attribution | 2 tasks | ⏳ |
| Phase 5: MCP + frontend + docs | 3 tasks | ⏳ |
| Phase 6: Verification + acceptance | 2 tasks | ⏳ |

---

## Phase 1: Store IO

### Task 1.1: circuit_capture_store.py
- [ ] EventWriter/Reader (sorted memmap + per-feature index) · ErrNormWriter/Reader · optional AttnTopKWriter/Reader · u16 bounds asserts · unit round-trips ×3 artifact kinds

## Phase 2: Capture

### Task 2.1: Tables + migration
- [ ] `circuit_capture_runs` (manifest incl. per-document 80/20 split + sae_fingerprints) + `circuit_discovery_runs` (params, report, candidates w/ BOTH orderings) · Alembic single-head check, up+down

### Task 2.2: Capture service + task
- [ ] Probe→estimate · ONE-forward-pass-per-batch multi-layer capture → encode → `max(θ_floor, ε·max_act)` threshold (floor-only when max_act missing) → events + errnorms (+ attention top-k opt-in) · progress/cancel · split drawn at capture (seeded, disjointness unit-tested) · stale-flag on SAE update

### Task 2.3: Endpoints + WS
- [ ] Estimate/confirm gate · list/delete · WS `capture/{id}` · hostile-config tests

## Phase 3: Statistics + discovery

### Task 3.1: circuit_stats_service.py
- [ ] PMI/lift/Spearman via (doc,pos) merge-join · support floor · **circular-shift null (within-document; marginal-rate preservation PINNED; whole-corpus permutation unreachable)** · BH-FDR q=0.05 · held-out replication on the capture-time split · synthetic-store tests: planted edges rank, planted high-base-rate pairs DON'T surface

### Task 3.2: Granularities
- [ ] Supernode units from cluster profiles: `A_C(t)=max` (mean alt recorded) · cohesion-floor eligibility · cluster_ref candidates · drill-down (same stats restricted to two clusters' members) · Cluster default for seeded mode

### Task 3.3: Discovery service + endpoints
- [ ] Orchestration + report assembly (null summary, FDR discipline, replication rate, stage counts) · candidate cap w/ both-orderings-preserving truncation · uncovered-seed-members report · stale refusal + force · WS + tests

## Phase 4: Attribution

### Task 4.1: Pass-through + scores
- [ ] SAE pass-through module (`x := x̂ + ε`, frozen weights) — **numerical-identity test vs clean model (the ε guard)** · backward batched BY DOWNSTREAM unit · `attr = Σ g·a` (IG-lite flag) · supernode subgradient-to-argmax documented · toy-model analytic test

### Task 4.2: Attribution task + endpoint
- [ ] `POST /circuit-discovery/{id}/attribution` · rung-1 gate (sign agreement + percentile floor) recorded per candidate · both orderings persisted for 017's uplift · gradient checkpointing + layer-subset fallback · wall-clock + peak-VRAM in report (envelope ≤1.5× capture)

## Phase 5: MCP + frontend + docs

### Task 5.1: MCP circuits tools
- [ ] 5 tools (capture start/list, discovery run/get, attribution run) · new `circuits` category via gating + compose/k8s env defaults + mcp-server.md · lag-0 disclosure in docstrings

### Task 5.2: CircuitsPanel
- [ ] Capture tab (config + attention toggle → estimate → confirm; runs list w/ split + stale) · Discovery tab (granularity toggle w/ Cluster-seeded default, floors, candidate table w/ PMI/support/null-pct/attr/replicated, run-report card pinned, lag-0 notice, attribution action)

### Task 5.3: Docs deliverables
- [ ] `0xcc/docs/tier-2.5-attention-mediated-mining.md` (A.8 expanded — BR-023 deliverable) · manual `circuits.md` (capture→mine→attribute; honesty framing) · mcp-server.md category

## Phase 6: Verification + acceptance

### Task 6.1: GPU integration + E2E
- [ ] Mini-capture (positions + errnorms verified vs probe) · attribution envelope on 1.2B (VRAM < 24 GB, wall-clock ≤1.5× capture, recorded) · E2E capture→seeded cluster discovery→attribution→report (replication rate shown) · cap `0xcc/caps/miStudio_Circuit_Discovery_<date>.png`

### Task 6.2: Acceptance (per instruct 007)
- [ ] FPRD §8 criteria 1–7 verified · copy audit: no causal language in 016 surfaces (rungs 0–1 only) · suites green · CLAUDE.md + PPRD row 17 status update

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/services/circuit_capture_store.py` (NEW) | events/errnorm/attention store IO |
| `backend/src/models/circuit_runs.py` (NEW) + migration | run manifests + reports |
| `backend/src/services/circuit_{capture,stats,discovery,attribution}_service.py` (NEW) | pipeline |
| `backend/src/workers/circuit_*_tasks.py` (NEW) | managed tasks |
| `backend/src/api/v1/endpoints/circuit_*.py` (NEW) | REST + WS |
| `backend/src/mcp_server/tools/circuits.py` (NEW) + gating | MCP |
| `frontend/src/components/panels/CircuitsPanel.tsx` (NEW) + store/types/api | UI |
| `0xcc/docs/tier-2.5-attention-mediated-mining.md` (NEW) · `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- Data ✅ (Ph1-2, migration both directions, split determinism) · API ✅ (Ph2-4) · MCP ✅ (Ph5) ·
  UI/State ✅ (Ph5) · Tests ✅ (planted-edge/base-rate/null pins, ε-guard identity test, toy-model
  analytic, GPU envelope, E2E) · Docs ✅ (Ph5, incl. the Tier-2.5 design deliverable) ·
  Acceptance ✅ (Ph6). Security: hostile-config tests; run-id-derived paths only (no user paths).
