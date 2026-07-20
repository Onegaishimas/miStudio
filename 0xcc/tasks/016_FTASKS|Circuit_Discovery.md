# Task List: Capture, Statistical Mining & Attribution

**Document ID:** 016_FTASKS|Circuit_Discovery
**Version:** 2.0 (reworked for BRD-MIS-CIRCUITS-002)
**Status:** ✅ IMPLEMENTED + 3 review rounds (27+21+verify findings; R1:17 fixed, R2:21 addressed, R3: GO for 017, 0 must-fix). Records: `.claude/context/sessions/review_feature016_R{1,2,3}_2026-07-*.md`. Phase 6 (GPU E2E) at close-out.
**Source:** 016_FPRD · 016_FTDD · 016_FTID · IDL-32, IDL-36 · CIRCUITS-002 A.3/A.4/A.6 normative

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Store IO | 1 task | ⏳ |
| Phase 2: Capture | 3 tasks | ⏳ |
| Phase 3: Statistics + discovery | 3 tasks | ⏳ |
| Phase 4: Attribution | 2 tasks | ⏳ |
| Phase 5: MCP + frontend + docs | 3 tasks | ✅ |
| Phase 6: Verification + acceptance | 2 tasks | ⏳ |
| Phase 0: 018 hand-off preconditions | 5 tasks | ⏳ |

---

## Phase 0: 018 hand-off preconditions (deferred findings — do FIRST or alongside Phase 1)

Recorded by 018 reviews R1–R3 (records in `.claude/context/sessions/review_feature018_*_2026-07-19.md`);
016 executes from this file, so they live here, not just in 018's FTASKS.

### Task 0.1: Encoder-weight orientation pins (R2-A4 — GATES the IDL-32 weight prior, Phase 3/4)
- [x] DONE (e990573: test_weight_resolvers.py — all 3 SAE formats + tied self-cosine identity) Unit tests for `resolve_encoder_weight` covering BOTH SAE formats (community_standard + internal orientation), mirroring the `resolve_decoder_weight` conventions. The cosine prior `W_dec(Lᵢ) @ W_enc(Lⱼ)` is silently wrong if orientation flips — pin before any prior computation lands.

### Task 0.2: `from_candidates` seam on CircuitService (R1 deferral)
- [ ] `CircuitService.from_candidates(discovery_run, selection) -> Circuit` — the discovery→circuit promotion path 018 stubbed out; discovery provenance + run id threading pinned.

### Task 0.3: Circuits store extraction + frontend test bundle (R1/R2 deferral)
- [ ] Extract CircuitsPanel fetch/state into `circuitsStore.ts` (house Zustand pattern) as member-edit UI + list filter controls land; add the panel/store vitest bundle (018 shipped with tsc/ESLint gates only).

### Task 0.4: SQL-side list pagination + filtering (R1 deferral)
- [ ] Move `edge_type` filter + limit/offset into the query in `CircuitService.list` (in-memory slice today; fine at N<200, wrong once discovery mints circuits in bulk).

### Task 0.5: Import/body cap hardening + MCP smoke (R3-B4/B6 ride-alongs)
- [ ] Extend the 1 MB cap to POST /circuits + PATCH (import-only today); consider actual-body enforcement (chunked requests bypass the Content-Length guard); bound per-edge `type_signals` size; fix `docker/nginx.gcp.conf` `client_max_body_size 0` (global exposure). Add first MCP circuit-tool smoke tests (9 tools, zero invocations today); `list_circuits` MCP tool: expose `edge_type`/`limit`/`offset`.

## Phase 1: Store IO

### Task 1.1: circuit_capture_store.py
- [x] DONE (e990573) EventWriter/Reader (sorted memmap + per-feature index) · ErrNormWriter/Reader · optional AttnTopKWriter/Reader · u16 bounds asserts · unit round-trips ×3 artifact kinds

## Phase 2: Capture

### Task 2.1: Tables + migration
- [x] DONE (e990573: migration c3d4e5f6a7b8 up/down/up) `circuit_capture_runs` (manifest incl. per-document 80/20 split + sae_fingerprints) + `circuit_discovery_runs` (params, report, candidates w/ BOTH orderings) · Alembic single-head check, up+down

### Task 2.2: Capture service + task
- [x] DONE (b495620: probe/estimate/confirm, atomic manifest, stale-flag, containment delete) Probe→estimate · ONE-forward-pass-per-batch multi-layer capture → encode → `max(θ_floor, ε·max_act)` threshold (floor-only when max_act missing) → events + errnorms (+ attention top-k opt-in) · progress/cancel · split drawn at capture (seeded, disjointness unit-tested) · stale-flag on SAE update

### Task 2.3: Endpoints + WS
- [x] DONE (b495620: circuit_discovery.py — 202/409/cancel/delete; WS circuit-capture/{id}) Estimate/confirm gate · list/delete · WS `capture/{id}` · hostile-config tests

## Phase 3: Statistics + discovery

### Task 3.1: circuit_stats_service.py
- [x] DONE (e990573: PMI/null/FDR/replication; planted-world pins). NOTE: FDR uses POOLED-standardized p (per-pair floor 1/(K+1) unreachable by BH — spec deviation recorded in FTDD) PMI/lift/Spearman via (doc,pos) merge-join · support floor · **circular-shift null (within-document; marginal-rate preservation PINNED; whole-corpus permutation unreachable)** · BH-FDR q=0.05 · held-out replication on the capture-time split · synthetic-store tests: planted edges rank, planted high-base-rate pairs DON'T surface

### Task 3.2: Granularities
- [x] DONE (b495620: supernode A_C=max, cohesion floor from budget.G, seeds survive unit cap). Drill-down (member-pair restriction) → deferred to review polish Supernode units from cluster profiles: `A_C(t)=max` (mean alt recorded) · cohesion-floor eligibility · cluster_ref candidates · drill-down (same stats restricted to two clusters' members) · Cluster default for seeded mode

### Task 3.3: Discovery service + endpoints
- [x] DONE (b495620: report + no-silent-caps + uncovered-seeds + stale refusal/force; E2E pinned) Orchestration + report assembly (null summary, FDR discipline, replication rate, stage counts) · candidate cap w/ both-orderings-preserving truncation · uncovered-seed-members report · stale refusal + force · WS + tests

## Phase 4: Attribution

### Task 4.1: Pass-through + scores
- [x] DONE (b495620: ε-identity pinned on LOSSY SAE, analytic toy-model gradient check, one-backward-per-downstream). IG-lite flag → deferred (raw default disclosed) SAE pass-through module (`x := x̂ + ε`, frozen weights) — **numerical-identity test vs clean model (the ε guard)** · backward batched BY DOWNSTREAM unit · `attr = Σ g·a` (IG-lite flag) · supernode subgradient-to-argmax documented · toy-model analytic test

### Task 4.2: Attribution task + endpoint
- [x] DONE (b495620: rung-1 gate, both orderings, VRAM/wall-clock envelope). Gradient-checkpointing/layer-subset fallback → deferred to GPU-integration hardening `POST /circuit-discovery/{id}/attribution` · rung-1 gate (sign agreement + percentile floor) recorded per candidate · both orderings persisted for 017's uplift · gradient checkpointing + layer-subset fallback · wall-clock + peak-VRAM in report (envelope ≤1.5× capture)

## Phase 5: MCP + frontend + docs

### Task 5.1: MCP circuits tools
- [x] DONE (618f14f: 5 tools in circuits category, lag-0 discipline in docstrings) 5 tools (capture start/list, discovery run/get, attribution run) · new `circuits` category via gating + compose/k8s env defaults + mcp-server.md · lag-0 disclosure in docstrings

### Task 5.2: CircuitsPanel
- [ ] Capture tab (config + attention toggle → estimate → confirm; runs list w/ split + stale) · Discovery tab (granularity toggle w/ Cluster-seeded default, floors, candidate table w/ PMI/support/null-pct/attr/replicated, run-report card pinned, lag-0 notice, attribution action)

### Task 5.3: Docs deliverables
- [x] DONE (618f14f: tier-2.5 design doc + manual circuits.md + concepts/evidence-ladder + concepts/tier-2.5; build green) `0xcc/docs/tier-2.5-attention-mediated-mining.md` (A.8 expanded — BR-023 deliverable) · manual `circuits.md` (capture→mine→attribute; honesty framing) · mcp-server.md category

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
| `frontend/src/components/panels/CircuitsPanel.tsx` (EXISTS — shipped by 018; 016 adds discovery/review affordances + store extraction per Task 0.3) | UI |
| `0xcc/docs/tier-2.5-attention-mediated-mining.md` (NEW) · `manual/docs/**` | docs |

## Coverage audit (instruct 007)
- Data ✅ (Ph1-2, migration both directions, split determinism) · API ✅ (Ph2-4) · MCP ✅ (Ph5) ·
  UI/State ✅ (Ph5) · Tests ✅ (planted-edge/base-rate/null pins, ε-guard identity test, toy-model
  analytic, GPU envelope, E2E) · Docs ✅ (Ph5, incl. the Tier-2.5 design deliverable) ·
  Acceptance ✅ (Ph6). Security: hostile-config tests; run-id-derived paths only (no user paths).
