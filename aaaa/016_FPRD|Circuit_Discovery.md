# Feature PRD: Capture, Statistical Mining & Attribution

**Document ID:** 016_FPRD|Circuit_Discovery
**Version:** 2.0 (reworked for BRD-MIS-CIRCUITS-002; validation moved to 017)
**Status:** Planned
**Related:** BRD-MIS-CIRCUITS-001 (BR-005..008) as amended by BRD-MIS-CIRCUITS-002 (BR-015, BR-016, BR-022, BR-023) · 000_PPRD §3.17 · IDL-32, IDL-36 · normative math: CIRCUITS-002 Appendix A.3/A.4/A.6/A.8 · feeds 017 (validation) and 018 (review/ladder)

---

## 1. Overview

### 1.1 Purpose
Build the evidence base circuit discovery needs — a position-carrying, per-token, multi-layer
feature-activation store — then mine it with **statistically sound** methods at two granularities
(feature-level and cluster-level supernodes) and re-rank candidates with a **Tier-2 gradient-attribution
pass**, so the expensive causal tier (017) is spent on an enriched shortlist.

### 1.2 User Problem
Extraction discards the per-token code matrix, so cross-layer co-activation is unanswerable from stored
data. And naive mining would be worse than nothing: raw co-occurrence rank is dominated by high-base-rate
pairs; lag-0 + a residual weight prior preferentially finds the trivial echoes the filter then removes;
feature splitting fragments true circuits below feature granularity. Discovery must produce candidates
whose statistics survive a null model, a held-out corpus, and the interpretability literature's standards.

### 1.3 Solution
- **Capture (BR-005/006, BR-023):** managed GPU task; sparse above-threshold events with **token
  positions**, per-(layer,token) **SAE error norms**, optional attention artifacts; cost-estimated;
  Tier-2.5-ready by construction.
- **Statistics (BR-015):** PMI/lift, minimum support, within-document circular-shift null, BH-FDR,
  per-document 80/20 held-out split with replication rate reported first-class.
- **Granularities (BR-016):** feature-level + **cluster-level supernodes** (curated profiles as units;
  recommended default for seeded mode); drill-down refinement.
- **Attribution (BR-022):** one forward + one backward per prompt, stop-gradient through SAE error;
  re-ranks before ablation; **uplift vs co-activation-only ranking is a measured, reported number**.
- **Tier-2.5 design doc (BR-023):** the attention-mediated mining design (A.8) authored as a deliverable.

## 2. User Stories
- **US-1:** I pick layers L10–L14 + the shared extraction dataset, see "≈1.4 GB, ≈40 min" before
  launching, and watch progress; I can cancel. The store records positions and error norms without my
  asking.
- **US-2:** I run seeded discovery at cluster granularity on my curated profiles and get ranked
  cluster→cluster candidates — each showing PMI, support, null percentile, and (after the Tier-2 pass)
  an attribution score — with base-rate noise absent from the top of the list.
- **US-3:** The run report tells me the held-out replication rate and which multiple-comparison
  discipline was applied; I can defend the list to a colleague.
- **US-4:** I drill into a promising cluster edge and see which member-feature pairs carry it.
- **US-5:** I launch the attribution pass on the candidate list; the re-ranked order and per-candidate
  scores appear, and later (017) the run reports whether attribution re-ranking actually raised the
  ablation survival rate.
- **US-6 (agent):** Via MCP I run capture → discovery (either granularity) → attribution and fetch the
  per-run discovery report — the full loop without the UI.

## 3. Functional Requirements

### 3.1 Capture (BR-005, BR-006, BR-023 capture-side)
1. Configure: layer set, eval corpus (existing dataset/tokenization), sparsity threshold (default
   ε·max_activation, ε=0.1), sample cap, optional attention-artifact capture (top-k keys per query for
   selected heads/layers).
2. Pre-launch **cost estimate** (probe batch → projected events/bytes/wall-clock); shown in UI and API.
3. The store persists per-event `(doc_id, token_pos, layer, feature_idx, activation)` — **position is a
   first-class column** — plus per-(layer, token) SAE reconstruction-error norms (full residuals optional
   per run). Manifest records corpus refs, layers, thresholds, SAE fingerprints, and the **per-document
   80/20 discovery/held-out split** drawn at capture time.
4. Managed Celery GPU task (progress, cancel, guardrails); stores listed/reusable/deletable;
   stale-flagged on SAE change.

### 3.2 Statistical mining (BR-007, BR-015)
5. Association = **PMI/lift** (A.3), never raw counts; continuous co-activation (Spearman) reported as a
   secondary column.
6. **Minimum support** `n_ud ≥ s_min` (default 20) before ranking.
7. **Null model:** within-document circular shift per feature (marginal-rate/burstiness preserving);
   significance = configurable high percentile (default 99th) of the null distribution.
8. **Multiple comparisons:** Benjamini–Hochberg FDR (q=0.05) over post-support pairs; the discipline
   applied is disclosed in the run report.
9. **Held-out replication:** candidates are mined on the discovery partition; each surfaced candidate is
   re-tested on the held-out partition; the **replication rate is a first-class number** in the run
   report.
10. Modes: **seeded** (around chosen profiles/members) and **open-corpus**, both first-class (BR-007
    unchanged).

### 3.3 Granularities (BR-016)
11. **Feature-level** (as CIRCUITS-001) and **cluster-level supernodes**: `A_C(t) = max` over member
    activations (mean recorded as alternative), binarized at the same θ; only clusters with cohesion ≥
    floor participate; **cluster granularity is the recommended default for seeded mode** (UI default).
12. Cluster-level candidates carry `member_kind: cluster_ref` resolvable to feature membership; review
    drill-down runs the same statistics restricted to the two clusters' members.

### 3.4 Tier-2 attribution (BR-022) — IDL-36
13. Attribution pass over a candidate list: SAE pass-through forward (`x = x̂ + ε`, weights frozen), one
    backward per prompt; `attr(u→d) = Σ_t g_u(t)·a_u(t)` (raw default; IG-lite on demand).
14. Scores re-rank the candidate list before 017's ablation sampling; disclosed per candidate as the
    third evidence signal; sign agreement + magnitude percentile gate rung 1 (`attribution_supported`).
15. The run records both orderings so 017 can report **survival-rate uplift** (attribution-re-ranked vs
    co-activation-only) — a mandatory number, null results reportable.
16. Wall-clock envelope: attribution over the reference corpus ≤1.5× the capture pass (reported).

### 3.5 Tier-2.5 readiness (BR-023)
17. Lag-0 limitation disclosed wherever discovery results appear (UI + MCP descriptions + run reports).
18. The **Tier-2.5 design doc** (from Appendix A.8: attention-join candidate generation, shuffled-join
    null, frozen-attention attribution, position-restricted validation) is authored as a deliverable of
    this feature (`0xcc/docs/tier-2.5-attention-mediated-mining.md`).

### 3.6 Run reports
19. Every discovery run produces a **discovery report**: parameters, null-model summary (threshold
    percentile achieved), FDR discipline, held-out replication rate, granularity, candidate counts by
    stage, attribution uplift placeholder (completed by 017), echo-filter effect (018's classifier
    feeds back counts). Rendered in UI and returned via MCP.

## 4. User Interface
- **Circuits panel — Capture tab:** config form (+ attention-artifact toggle), cost-estimate card,
  runs list (progress, size, split shown, stale flag, delete).
- **Circuits panel — Discovery tab:** granularity toggle (Cluster default for seeded), mode toggle,
  floors, run; candidate table (up→down with labels, PMI, support, null-percentile, attribution score
  when run, rung chip via 018); "Run attribution pass" action; **run-report card** (replication rate,
  null summary, disciplines) pinned above results.
- Lag-0 limitation notice on the Discovery tab.

## 5. API / Integration
- `POST /api/v1/circuit-capture` (estimate; `confirm=true` launches) · `GET` list · `DELETE /{id}`.
- `POST /api/v1/circuit-discovery` (granularity, mode, store id, seed refs, floors) · `GET` runs/results
  incl. the report · `POST /api/v1/circuit-discovery/{id}/attribution` (candidate scope, prompt set).
- MCP (new `circuits` category): `start_circuit_capture`, `list_circuit_captures`,
  `run_circuit_discovery`, `get_discovery_results` (returns the report), `run_attribution_pass`.
- WS progress channels per house pattern.

## 6. Data / Types
- Store: per-layer columnar memmap events + per-feature index + error-norm sidecar + optional attention
  sidecar, under `/data/circuit_captures/{run_id}/`; manifest row `circuit_capture_runs`.
- `circuit_discovery_runs` (params, report, candidates JSONB with stats + attribution fields).
- Alembic migration (single-head check).

## 7. Dependencies
- Existing: activation_service memmap patterns, layer-agnostic `SparseAutoencoder.encode`,
  decoder/encoder weight resolvers (015), Celery/WS infra, dataset management.
- 30 curated profiles + cohesion scores (supernode eligibility). 017 consumes candidates + prompt
  machinery; 018 consumes candidates for typing/rung/review.
- Attribution: autograd through the host model with frozen SAEs (gradient checkpointing available).

## 8. Success Criteria
1. Capture (L13+L14 reference corpus) completes within estimate ±30%; positions + error norms present;
   split recorded.
2. Discovery at both granularities returns candidates whose run report shows: null threshold achieved,
   FDR discipline, held-out replication rate ≥ the calibration target (60%) on the seeded calibration
   run — or the miss is reported, not hidden.
3. Planted-edge synthetic tests: known edges rank above null; known high-base-rate pairs do NOT surface.
4. Attribution pass completes ≤1.5× capture wall-clock; per-candidate scores + rung-1 gating recorded;
   re-ranked ordering persisted for 017's uplift measurement.
5. Cluster drill-down produces member-pair statistics consistent with the supernode edge.
6. Tier-2.5 design doc exists and covers A.8's five design points.
7. MCP agent completes capture → discovery → attribution → report retrieval without the UI.

## 9. Non-Goals
- Interventional validation & manifests (017); edge typing, rung display, review/promotion, contract
  (018); Tier-2.5 mining implementation (named fast-follow); Tier-3 attribution graphs; transcoder
  substrates (research track); miLLM anything.

## 10. Testing Requirements
- Unit: store round-trips (positions, error norms, attention sidecar), PMI/lift on planted synthetic
  stores, circular-shift null construction (marginal rates preserved — pinned), BH-FDR application,
  split determinism, supernode activation (max/mean), cohesion gating, attribution math on a toy model
  (analytic gradient check), estimator extrapolation.
- Integration (GPU): mini-capture with positions/error norms verified; attribution pass memory envelope
  on the 1.2B model (fits 24 GB with checkpointing).
- API: estimate/confirm gate, cancel, stale refusal, hostile configs.
- Frontend: run-report card, granularity defaults, candidate table with evidence columns.
- E2E: capture → seeded cluster-granularity discovery → attribution → report with replication rate;
  screenshot `0xcc/caps/miStudio_Circuit_Discovery_<date>.png`.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-005 (bounded configurable capture) | §3.1 |
| BR-006 (managed tasks + guardrails) | §3.1 item 4, §3.2, §3.4 |
| BR-007 (auto-proposed ranked candidates; seeded + open) | §3.2 items 5–10 |
| BR-008 (independent evidence signals, disclosed) | §3.2, §3.4 item 14, §3.6 |
| BR-015 (002: PMI, support, null, FDR, held-out replication) | §3.2 items 5–9, §3.6 |
| BR-016 (002: cluster-level supernode granularity) | §3.3 |
| BR-022 (002: Tier-2 attribution re-ranking + uplift) | §3.4 |
| BR-023 (002: positions/attention capture-side + Tier-2.5 design doc + lag-0 disclosure) | §3.1 items 1, 3 · §3.5 |
