# Review Record — Feature 016: Circuit Discovery (Capture / Mining / Attribution) — ROUND 1

**Date:** 2026-07-19
**Scope:** 016 implementation at HEAD `ae2e6bf` (commits e990573, b495620, 618f14f, ae2e6bf). Two parallel review agents: a high-recall code-review (bugs) + a 4-perspective /review (resilience/reliability/UX).
**Finding count:** **27** (12 code-review + 15 four-perspective) — over the ≥20 bar.

---

## Fixed this round (17)

| # | Sev | Finding | Fix |
|---|-----|---------|-----|
| CR#1 | P1 | Attribution passed the int layer index to `get_hookable_module` (expects the module) → EVERY attribution pass crashed | `structure.layers_module[L]` + bounds check |
| CR#2 | P2 | `feature_idx` u16 → wide SAEs (Gemma Scope 16k–131k latents) aborted capture at the first 65k+ index | widened EVENT_DTYPE feature_idx to u32 (keys pack only pos<<16, so independent); pin updated |
| CR#3 | P2 | Malformed seed_ref (`feature_idx` null/NaN) → `int(None)` TypeError deep in the worker | typed `DiscoverySeedRef` (exactly-one-of validator) → 422 at submit |
| CR#4 | P2 | `model_id` could resolve to None → tokenization filter on NULL matched wrong/no row | assert resolved with a clear 422 |
| CR#5 | P2 | Cancellation not polled in the null/replication loops (the dominant cost) | poll `cancel_check` every 25 candidates in the null loop |
| CR#6 | P2 | Worker wrote `completed` without re-checking status → last-writer race clobbered a cancel | `db.refresh` + skip terminal write if `cancelled` (capture + discovery) |
| CR#8 | P3 | `_cancel_checker % 5` skipped the FIRST 4 checks | poll on the 1st call then every 5th |
| CR#9 | P3 | Recursive `setTimeout` estimate poll had no unmount guard | `mountedRef` guard on every tick/setState |
| QA-P1 | P1 | No store-size guardrail → runaway capture fills `/data` (FPRD §3.1 "guardrails") | running byte ceiling (5× estimate) + free-disk check → abort+rmtree+failed |
| QA-P1 | P1 | No 409-on-concurrent despite the docstring promising it | `assert_no_active_gpu_run` (capture/confirm) + per-store discovery guard, both in the insert transaction |
| QA-P1 | P1 | CPU discovery routed to the GPU `extraction` queue → head-of-line blocks GPU work | discovery → `processing` queue; capture+attribution stay on `extraction` |
| QA-P2 | P2 | Attribution reused the discovery run's `status`/`progress` → a failed attribution made the completed DISCOVERY present as failed | separate `attribution_status/progress/error` columns (migration e5f6a7b8c9da); worker/service/endpoint write those; 409 if a pass is in flight |
| Prod-P2 | P2 | `replication.rate` null (0 held-out tested) rendered a confident false **"0%"** on the trust-surface hero number | type `number \| null`; render "n/a — no held-out candidates tested" |
| CR#12 | P3 | `estimating` status was unreachable (probe ran under `running`) | set `estimating` around the probe |
| Arch-P2 | P2 | No 016 causal-language audit test (018 established the pattern) | `test_causal_language_audit.py` — greps discovery/attribution reports + MCP docstrings for unqualified causal language |
| Test-P1 | P1 | Zero API-level tests for 016 endpoints | `test_circuit_discovery_api.py` — 202/409/422/cancel + seed-ref validation + attribution-lifecycle 409s |
| Test-P2/P3 | P2/P3 | Attribution gate + both-orderings, cancellation, pooled-FDR edges unpinned | added gate/orderings pin, cancel-stops-and-persists pin, zero-variance/single-pair pooled-null pins |

Plus the **CI-green fix** (not from the review sweep but found alongside): the discovery-service test read `DATABASE_URL_SYNC` whose CI user `mistudio_test` failed auth → PendingRollbackError cascaded across the suite. Now derives the sync DSN from `async_engine.url` (same _test DB + creds conftest connected with). This was the actual cause of the two red CI runs; `fix(circuits): CI green` is verified success.

## Verified NON-bugs (checked, no action)

- **JSONB in-place mutation** — every site reassigns the attribute (`run.manifest = …`), so change-tracking fires. Sound.
- **Circular-shift null collisions** — within-doc constant rotation over unique positions is a bijection; `np.unique` never drops. Marginals preserved.
- `pair_stats(assume_unique=True)`, `pooled_null_pvalues` sd==0, `ErrNormReader.doc_norms` searchsorted, `_max_acts_per_key` clip guard — all verified correct.

## Recorded deferrals (with rationale — carry to 016 close-out or a named follow-on)

- **Prod-P1 cluster drill-down (US-4 / SC-5):** named in FTDD §57 but unbuilt — a genuine FEATURE (endpoint + member-pair stats + UI), not a review patch. → 016 FTASKS Phase 3.2 follow-up / close-out; the stats primitives (`pair_stats`/`null_test`) are ready to restrict.
- **Prod-P1 seed-ref picker UI:** the seeded flow uses a raw textarea for `layer:cluster:id`. Backend now validates (CR#3), so it's UX polish, not correctness. → 016 close-out UI pass.
- **Arch-P1 WebSocket-first migration:** emitters exist and fire; the panel polls (with cleanup). A `useCircuitRunWebSocket` hook is the house pattern but a larger refactor. → 016 close-out.
- **Test-P1 MCP invocation tests:** the 5 tools are thin `client.post` wrappers; zero invocation tests (shared with 018's same gap). → 016 close-out MCP smoke.
- **Arch-P3 stable `candidate_id`:** both orderings use list-index ranks; 017 should key uplift off a stable id. → land WITH 017's uplift consumer.
- **CR#7 attention sidecar layer-index alignment:** `out.attentions[L]` assumes residual-hook idx == attention-tuple idx; convention-dependent. Attention capture is opt-in and Tier-2.5-only; correctness only matters when Tier-2.5 mining ships. → note for the Tier-2.5 fast-follow.
- **Arch-P2 O(units²) all-pairs + all-keys-in-RAM:** fine at the reference-corpus scale; the scaling wall for open-mode on large corpora. → recorded in FTDD as tech debt (inverted-index pre-filter is the fix).
- **CR#11 sync-bridge two-transactions:** latent consistency hazard on the delete path; works because create commits before the async UPDATE. → close-out hardening.

---

## R1 outcome

**17 of 27 fixed** (all P1s, all correctness P2s, the trust-surface UX bug, + 5 new test files/classes). Remaining 10 are recorded deferrals — genuine features (drill-down), UX polish (picker, WS), or scale/hardening that don't gate 017. Backend circuit suite green (exit 0), frontend tsc+build green, CI green. R2 will verify these fixes hold and sweep fresh.
