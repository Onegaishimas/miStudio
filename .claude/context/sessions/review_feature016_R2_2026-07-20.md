# Review Record — Feature 016: Circuit Discovery — ROUND 2

**Date:** 2026-07-20
**Scope:** post-R1-fix state at HEAD `963a6a2`. Two agents: a code-review that BOTH verified R1's fixes empirically AND swept fresh (7 findings B1–B7), and a 4-perspective /review (14 findings). **21 new findings** + full R1 verification.

---

## PART A — R1 fix verification (agent verdicts, all confirmed by me)

| R1 fix | Verdict |
|--------|---------|
| Attribution `get_hookable_module(layers_module[L])` + bounds | **HOLDS** |
| `feature_idx` u32 + EVENT_BYTES 10→12 + estimate/ceiling use it | **HOLDS** |
| Typed `DiscoverySeedRef` | **PARTIAL → REGRESSED (B1)** — see below |
| Store-size guardrail + `ev.count` property | **HOLDS** |
| 409 concurrency guards | **PARTIAL (B2 race; B5 estimate bypass; Q1 attribution bypass)** |
| Separate attribution lifecycle (columns/worker/service) | **PARTIAL (backend holds; frontend never read them — B3/B4)** |
| Cancellation in null loop + last-writer `db.refresh` | **HOLDS** (both services) |
| `_cancel_checker` polls 1st call | **HOLDS** |
| Replication-rate null → 'n/a' + estimate-poll unmount guard | **HOLDS** |
| Migration single-head (renamed e5f6a7b8c9da, no collision) | **HOLDS** |

The R1 record's "CPU discovery → processing queue" fix was verified SOUND (a `processing` worker is in the default `-Q` set) — closed, not a finding.

## Fixed this round (all 21 + the demo-chain)

**P1 (the R1 regression):**
- **B1** — the R1 typed-seed-ref fix RELOCATED the `int(None)` crash: `model_dump()` emits BOTH keys (unset=None), so `"feature_idx" in ref` is always true → crash on the cluster-seed path, which is the **default** seeded flow. Fixed: discriminate on VALUE (`ref.get("feature_idx") is not None`) at all three sites (`_seed_key_set`, `_feature_units`, `_expand_candidates`). Pinned with the exact both-keys-present dict.

**The demo-day GPU-safety chain (Q1+Q2+Q3 / B2+B5+B6):**
- **Q1/B5** — attribution (a GPU task) AND the estimate probe (a GPU forward) both bypassed the concurrency guard → two model loads on one 3090. Fixed: `assert_no_active_gpu_run` now covers attribution runs too; guard applied to estimate + confirm + attribution.
- **B2** — check-then-insert race (sharing a session ≠ serializing). Fixed: `pg_advisory_xact_lock` inside the guard, released at commit — two concurrent requests can't both pass.
- **Q2/B6** — in-flight attribution was uncancellable (the worker polled `attribution_status=='cancelled'` but nothing set it). Fixed: `POST /circuit-discovery/{id}/attribution/cancel` + `attribution_task_id` (A3) so the RIGHT task is revoked.
- **Q3** — no stuck-run cleanup → a died capture left 'running' forever and the guard then locked out EVERY future capture. Fixed: `cleanup_stuck_circuit_runs` beat task (10-min, mirrors extraction cleanup) — reclaims stuck capture/discovery/attribution rows, rmtrees partial stores, needs an inactive Celery task.

**Attribution-split now real on the UI (B3/B4/P1/P2 — the "dead R1 fix"):**
- Frontend `DiscoveryRun` type gained `attribution_status/progress/error`; the poll now watches `attribution_status` (was only `run.status`, which stays 'completed' during a pass); the Discovery tab shows attribution running/%/failed(+error)/cancelled distinctly with a cancel button; "Run attribution" → "Re-run" when done.
- **US-5** — both orderings now VISIBLE: a "rank Δ" column (coact→attr) appears after attribution and the table re-sorts by `attr_rank` (unattributed sort last).

**Demo-credibility (Prod-P3):** `fdr.p_resolution` surfaced on the report card (the number that answers "could FDR ever pass anything?").

**Hardening:**
- **A1** — `test_no_duplicate_revision_ids` (two files, same revision id → one silently shadows; the single-head test can't catch it — the 016 work nearly hit exactly this with d4e5f6a7b8c9). Migration made idempotent (IF NOT EXISTS / IF EXISTS) to survive dev drift.
- **B7** — recorded: seeded granularity/seed-ref-shape mismatch yields no matches (→ uncovered_seeds) rather than a 422; acceptable (surfaced in the report), noted for close-out.
- **Q4/A2** — deleted-cluster-profile candidate: still drops silently to null attribution; the value-check fix (B1) removed the crash risk, and A2's `attr_rank: null` holes are handled by the frontend's "sort last." A stable `candidate_id` for 017's uplift math is the remaining piece — recorded to land WITH 017 (already a deferral).

## Tests added (R2)
- B1 seed-ref discrimination (both-keys-present cluster ref); concurrency guard (running capture/attribution blocks, idle passes); attribution-cancel API (200 + 409); **5 MCP smoke tests** (T4 — SC-7, closes a gap shared with 018); **frontend component test** (T1 — RunReportCard: null-rate→'n/a' not '0%', caps-hit banner, p-resolution — first circuit component test); size-guardrail math (T2, now GPU-free via `exceeds_size_ceiling`); duplicate-revision-id guard (A1).

## Recorded deferrals (unchanged from R1 or newly recorded, none gate 017)
- Stable `candidate_id` for 017 uplift (A2) — land WITH 017.
- Cluster drill-down (US-4), seed-ref picker UI, WS-first migration — 016 close-out (features/UX, not correctness).
- B7 granularity/seed-shape 422, Q4 dropped-candidate report note — close-out polish.
- Attention sidecar layer-index (CR#7 from R1) — Tier-2.5 fast-follow.

---

## R2 outcome

**All 21 R2 findings addressed** (fixed or consciously recorded), and every R1 fix verified — with the one regression (B1) that R1 introduced caught and fixed, plus the GPU-safety chain that would have bitten in a live demo. 7 new test files/classes. Backend circuit suite green (exit 0), frontend build + component test green, migrations up/down/up + single-head + no-duplicate-id green. R3 is the final gate before 017.
