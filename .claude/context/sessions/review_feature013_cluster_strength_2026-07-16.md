# Multi-Agent Review — Feature 013: Cluster Strength Budget Model

**Date:** 2026-07-16
**Scope:** feature (commits 3d57ef4 → 21c1751 + iteration-2 badge fix)
**Iterations:** 1 = multi-angle /code-review (27 findings, all fixed), 2 = post-fix verification (1 new
finding, fixed), 3 = this 4-perspective /review.
**Goal gate:** ≥10 findings found+fixed per feature → **28 total, exceeded.**

---

## 1. Product Engineer

**Requirements alignment (013_FPRD ↔ implementation):**

| Requirement (IDL-29 step) | Where | Verdict |
|---|---|---|
| 1. sim-weights w=s̃/Σs̃, missing→mean | `cluster_allocation_service.py` (mean-over-present imputation) | ✅ |
| 2. f_eff = Σwf/Σw over known-f | service (+ unweighted fallback when no member has f) | ✅ |
| 3. B_dir = clamp(a−b·f_eff, m, M) | service; `_default_b_dir=(m+M)/2` when f unknown | ✅ |
| 4. G = ‖Σσwd‖ on RAW decoder columns, server-only | service + endpoint (`resolve_decoder_weight` shared with hook — parity guaranteed by construction) | ✅ |
| 5. B = min(B_dir/max(G,0.05), Σb*(f)) | service (+ solo-cap; cap_bound flag) | ✅ |
| 6. strengths = σ·B·w, 0.1 grain | service (greedy no-sign-flip residual fold) | ✅ |
| 7. coherence flags + cohesion gate 0.5 | flags: cancellation (positive-subset), low_cohesion, default_budget, cap_bound, approximate, nonunit_decoder, grain_limited, solo | ✅ |
| 8. pin + budget-preserving rebalance | store `rebalanceStrength` (mirrors backend `rebalance()` reference; parity test with shared vectors) | ✅ |
| 9. λ dial [0,2], applied once at request build | `applyIntensity` in Blended AND Compare builders (single+batch) | ✅ |
| 10. N=1 → solo path verbatim (DEFAULT 10) | store early-return + backend solo shortcut | ✅ |

**User-visible behavior:** budget bar (used/B + G + flags), amber over-budget, cluster badges, pin
badge = click-to-unpin (cluster mode only), λ dial beside Blended|Compare, low-cohesion notice.
**Note:** the low-cohesion notice copy explains WHAT but not what the user should DO — acceptable for v1;
manual page (014's cluster workflow doc) should cover it. **P3.**

## 2. QA Engineer

- **Error handling:** endpoint maps ValueError→400, decoder failure→approximate G=1 (never 500);
  frontend allocation failure keeps solo baselines (progressive enhancement, no error state leak). ✅
- **Input hardening:** duplicate-idx refusal (client), idx bounds (client-race-safe: DB check + in-service
  decoder-shape check), SAE-layer cross-check, mixed-layer refusal, not-ready SAE 400. ✅
- **State hygiene:** clusterBudget/clusterNotice/intensity are session-only; partialize strips `pinned`
  and demotes `cluster`→`manual` — no stale trust signals after reload. Stale-response guard keyed by
  instance ids. ✅
- **Security:** no new attack surface — allocation endpoint is read-only math over DB-resolved paths
  (settings.resolve_data_path), no user-supplied paths. ✅
- **Residual risk (P2, recorded):** constants (a,b,m,M)=(2.9,2.6,1.0,3.0) duplicated between
  `steeringStrength.ts` (solo) and service defaults. Divergence would desync solo vs cluster baselines.
  Consolidation deferred to Phase-5 calibration (constants become per-SAE server data; frontend then
  reads them from the allocation response, which already echoes `constants_used`).

## 3. Architect

- **Placement:** formula is a pure, I/O-free module (`compute_allocation`) — endpoint is glue; MCP tool
  calls the same core. One source of truth, three consumers. ✅ Right altitude.
- **Hook parity by construction:** the ONE thing that would silently invalidate the model — allocation
  measuring a different vector than the hook injects — is structurally prevented: both use
  `resolve_decoder_weight`. ✅ (was iteration-1 critical #1)
- **Frontend owns only rebalance arithmetic** (no decoder client-side); backend keeps a reference
  `rebalance()` purely for parity tests. Acceptable duplication: 30 lines, pinned by a shared-vector test.
- **v1 restrictions honest:** single-layer only, single-membership (dup refusal), N≤20. All refuse
  loudly rather than degrade silently. ✅
- **Debt (P3):** `weightsByInstance` lives in the store while weights also exist server-side; if 014
  profiles persist budgets, recompute-on-load must come from the server, never the persisted map
  (014 design already specifies explicit strengths + recompute survival — consistent).

## 4. Test Engineer

- Backend: 32 unit (sanity anchors: identical⇒B=B_dir, orthogonal⇒B_dir·√N, N=1 canary; 14-row edge
  table; 7 iteration-1 regressions incl. raw-gain and sign-flip proofs) + 6 endpoint contract tests. ✅
- Frontend: store 63 tests (rebalance property test over random edit sequences; reorder-safety;
  dup refusal; λ once + λ-in-Compare parity; partialize; stale guard; low-cohesion gate;
  backend-parity vectors) + ClusterBudgetBar 7 tests. ✅
- **Gap (P2, blocking full close):** empirical validation protocol not yet executed — requires deployed
  GPU env (runbook `0xcc/docs/cluster-strength-validation.md`). This is FPRD's hard shipping gate
  (±30% of empirical optimum + non-degenerate). Scheduled post-deploy.
- **Gap (P3):** no Playwright E2E yet for budget bar/dial (also post-deploy).
- Known noise: ~98 pre-existing vitest failures in dataset/model/training panels (fail on clean HEAD;
  unrelated suites) — tracked in memory, not chased here.

---

## Iteration-2 findings (post-fix verification)

1. **FIXED:** stale "pinned" badge survives budget-clearing mutations — badge now renders only when
   `onTogglePin` is passed, and FeatureSelector passes it only in cluster mode.
2. Verified no-change-needed: sweep builder intentionally skips λ (solo-feature exploration);
   endpoint layer cross-check correctly skips when `sae.layer is None` (service mixed-layer check
   still catches); dup-refusal cannot leave a stale notice (every mutation clears it).

## Gate

**SHIP-WITH-NOTES** — code complete and internally verified; formula parity proven by construction +
regression tests. Two conditions before FTASKS Phase 5/6 close:
1. Empirical validation protocol execution on deployed env (hard gate, BR-005).
2. Playwright E2E for budget bar / rebalance / dial.

Findings ledger: 27 (iter 1) + 1 (iter 2) = **28 found, 28 fixed** · 0 open P0/P1 · 2 recorded P2 debts.
