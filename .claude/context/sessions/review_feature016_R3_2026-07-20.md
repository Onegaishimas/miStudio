# Review Record — Feature 016: Circuit Discovery — ROUND 3 (FINAL / CLOSING GATE)

**Date:** 2026-07-20
**Scope:** post-R2 state at HEAD `e6b6cdc`. Closing round after R1 (27 findings/17 fixed) and R2 (21 findings/all addressed). Verified R2 fixes at runtime, final fresh sweep, explicit GO/NO-GO for Feature 017.
**Verification:** backend circuit suite green (~127 unit tests across circuit/alembic/causal/weight/mcp); frontend `tsc` clean + CircuitsPanel component test 5/5; single migration head; CI green on the R2 push (backend + frontend).

---

## Verdict: **GO for Feature 017** — zero must-fixes

All six R2 fixes verified **HOLD** with no regressions. The fresh sweep cleared every R2-introduced concern (advisory-lock deadlock/release, `_guard_and_mark` race, rank-Δ sort null-safety, cleanup false-reclaim). One new LOW finding (B-R3-1), fixed this round.

## PART A — R2 fix verification (all HOLD)

| R2 fix | Verdict |
|--------|---------|
| B1 value-check discrimination (3 sites) — cluster-seed flow works end-to-end | **HOLDS** |
| GPU concurrency guard covers capture+attribution; advisory lock; estimate+confirm+attribution | **HOLDS** (no bypass path) |
| Attribution cancel sets status + revokes attribution_task_id; worker polls attribution_status; normal-return not exception (so cancelled isn't clobbered to failed) | **HOLDS** |
| cleanup_stuck_circuit_runs: autodiscover+beat; reclaims all 3 lifecycles; `_task_is_active` fails safe; 60-min threshold >> commit cadence (no false-reclaim) | **HOLDS** |
| Migration idempotency + attribution_task_id + single-head + no-duplicate-revision test | **HOLDS** |
| Frontend attribution lifecycle + rank-Δ + null-rate→'n/a' | **HOLDS** |

## PART B — fresh sweep

All R2-introduced code cleared. **One new LOW finding, FIXED this round:**
- **B-R3-1 (disk hygiene, LOW):** `store_path` was persisted only on capture SUCCESS, but the store dir is created earlier — an OOM-killed capture left an orphaned partial store that cleanup's `if store_path` guard skipped. **Fixed:** persist `store_path` at store-dir creation, so `cleanup_stuck_circuit_runs` always rmtrees it. (Never a correctness/lockout issue — the row still marked failed, releasing the guard; size bounded by the 5× ceiling.)

## PART C — 017 gate

Seam readiness confirmed:
- Candidates + both orderings + `discovery_run_id` + rung-1 gate all returned by `GET /circuit-discovery/{id}` — 017 can read everything.
- Stable `candidate_id` (R2 A2) does NOT block 017's first task — 017 keys off the `(up, down)` endpoint tuple or adds the id in its Phase 3; standing deferral to land WITH 017's uplift consumer.
- 018 Task 3.0 preconditions (optimistic concurrency + `update()`-only edge-write) correctly recorded in `017_FTASKS` Task 3.0 + `017_FTID` §2.4; `CircuitService.update` already validates+recomputes-rung, `Circuit.updated_at` exists to key the precondition. 017 adds the version check as its Phase-3 gate, as designed.
- No 016 landmine for 017 (its services are new files; capture store exposes σ_d/corpus-mean/upstream-firing per the FTID dependency list; rung enum in 018, already landed).

### MUST-FIX-BEFORE-017: none
### RIDE-ALONG (during 017):
1. B-R3-1 (fixed this round — persist store_path early).
2. Stable candidate_id (R2 A2) — with 017's uplift consumer.
3. Task 3.0 (017's own edge-validation precondition — optimistic concurrency + update()-only edge writes; must land before 3.1/3.2 write any edge).

Standing 016 close-out deferrals (cluster drill-down US-4, seed-ref picker UI, WS-first migration, B7 granularity 422) remain out of 017's path.

---

## 016 final tally
- **3 review rounds:** 27 + 21 + (verify + 1 new) findings. R1: 17 fixed. R2: 21 addressed (incl. the P1 regression R1's own fix introduced + the demo-day GPU-safety chain). R3: all R2 fixes verified holding + B-R3-1 fixed.
- **Implementation:** Phases 0–5 complete (capture store, statistics, capture/discovery/attribution services, endpoints, MCP, frontend tabs, Tier-2.5 doc + manual). Phase 6 (GPU integration + E2E screenshot) at 016 close-out.
- **Suites:** backend green, frontend build + component test green, CI green on every push.

*Round 3 (final) 2026-07-20 · scope HEAD e6b6cdc · verdict GO for 017 (0 must-fix) · all R2 fixes verified.*
