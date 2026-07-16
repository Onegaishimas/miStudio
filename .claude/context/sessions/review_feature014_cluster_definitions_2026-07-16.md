# Multi-Agent Review — Feature 014: Cluster Authoring & Portable Definitions

**Date:** 2026-07-16
**Scope:** feature (commits 009a1ac backend + 3aaecf8 frontend + iteration-1/2 fix batch)
**Iterations:** 1 = multi-angle /code-review (14 findings), 2 = post-fix verification (1 cosmetic), 3 = this
4-perspective /review.
**Goal gate:** ≥10 findings found+fixed per feature → **15 found, 10 fixed in code, 4 recorded P3 debts,
1 no-change-needed — exceeded.**

---

## 1. Product Engineer

**Requirements alignment (014_FPRD ↔ implementation):**

| Requirement (IDL-30) | Where | Verdict |
|---|---|---|
| Durable profiles decoupled from recomputable groups | `cluster_profiles` table, soft `source_group_id`, FK `external_saes` RESTRICT | ✅ |
| Name + narrative + tuned member strengths | SaveProfileDialog → `ClusterProfileCreate`; explicit strengths only | ✅ |
| Versioned consumer-neutral JSON (`mistudio.cluster-definition/v1`) | strict pydantic contract + published `docs/schemas/cluster-definition-v1.json` + sync test | ✅ |
| Single + bundle export, save-then-export | `GET /{id}/export`, `POST /export-bundle` — always serializes the STORED profile | ✅ |
| Import with compatibility matrix | bind / warn_bind / block / unbound, per-item isolation, member-bounds check vs the ACTUAL bound SAE | ✅ |
| Round-trip fidelity | unit-proven for unbound path; members+budget byte-identical | ✅ |
| No secrets / no local paths | `no_local_paths` validator + field-name negative test | ✅ |
| Label tier 1 (authored name titles results) | load/save set `clusterContext.display_token = profile.name` → existing 012 baking path | ✅ |
| Load hydration bypasses auto-baselines | `loadProfileIntoSteering` explicit strengths, `strengthSource: 'manual'`, allocation guard | ✅ |
| SAE-delete protection | structured 409 `PROFILES_BOUND`; force unbinds (profiles survive) | ✅ |
| MCP access | `profiles` category: list/get/save/export; gating + compose/k8s env defaults | ✅ |
| MILLM-bound contract note | manual §Portable definitions info box | ✅ |

## 2. QA Engineer

- **Hostile input:** 1 MB payload cap (413), bundle >50 pre-validation cap (400), unknown kind (400),
  hostile shapes (parse tests), member indices vs bound SAE (block), name/narrative length caps. ✅
- **Information hygiene:** import item errors expose ValidationError text only; DB/internal errors map to
  `"internal error"` (tested). Exports carry no secrets/paths — validator + negative test. ✅
- **Data safety:** profile delete = explicit confirm; SAE delete guarded 409; force path UNBINDS rather
  than destroys authored work; batch delete pre-checks to avoid FK IntegrityError poisoning the session. ✅
- **Recorded P3 debts:** unpaginated list (len-based total — fine at expected profile counts);
  `datetime.utcnow` deprecation (consistent with model layer; codebase-wide sweep later);
  `exportBundle` store action has no multi-select UI yet.

## 3. Architect

- **Contract at the right altitude:** pydantic is the single source of truth; the published JSON Schema is
  GENERATED and pinned by a sync test — consumers (MILLM/unified-MCP/OWUI) get a stable artifact without a
  hand-maintained schema drifting. ✅
- **Compatibility matrix is a pure function** (`decide_compatibility`) — unit-tested across 9 rows without
  a DB; endpoint is glue. ✅
- **Steering integration reuses 012/013 invariants instead of adding parallel state:** profile identity
  rides `clusterContext` (titles) + `activeProfile` (save affordance/guard), cleared by the SAME mutation
  rules — no second provenance mechanism to keep honest. ✅ Right depth.
- **no_change_needed:** SAE-delete guard also fires for soft deletes — intentional (a DELETED-status SAE
  cannot steer; unbinding is the honest state either way).
- **Debt noted (013-shared):** profile budget snapshots duplicate allocation provenance; when constants
  become per-SAE-calibrated, recompute-on-load should offer "refresh allocation" using the profile's
  stored formula_id for comparison.

## 4. Test Engineer

- Backend: 31 tests — matrix (9), parsing (4), validators/caps (4), round-trip + provenance (3),
  narrative-clear PATCH (1), schema-sync (1), endpoint contract incl. hostile payloads + per-item
  isolation + bounds-block + delete-guard (9). ✅
- Frontend: 4 hydration-rule store tests (explicit strengths, refusal rules, allocation guard,
  mutation-clears-profile) inside the 67-test steering suite; 79 steering-scoped tests green. ✅
- **Gaps (post-deploy):** E2E titled-run assertion with a loaded profile; recompute-survival integration
  test against a live DB (grouping recompute must not touch profiles — structurally guaranteed by the
  decoupled table, asserted at E2E); MCP profiles smoke test end-to-end.

---

## Findings ledger

**Fixed (10):**
1. Unbound/imported profiles invisible (server-side sae_id filter) → fetch-all + client partition
2. Forced current-SAE binding on import wrongly blocks foreign-SAE bundles → auto-bind matrix
3. Import skipped member-bounds check vs the actual bound SAE → block + test
4. PATCH could not clear narrative → `model_fields_set` semantics + test
5. One-click profile delete → confirm dialog
6. Budget snapshot dropped formula provenance → ClusterBudget carries formula_id/constants/f_eff through save AND load
7. Import item errors leaked internals → generic detail + test
8. Batch SAE delete FK-poisoned the session mid-batch → pre-check + rollback
9. Import re-serialization bomb / oversize bundles → 413/400 caps (tests)
10. "N for this SAE" count mislabeled unbound profiles → "N available"

**Recorded P3 (4):** unpaginated list; utcnow deprecation; exportBundle UI; profile-budget refresh flow.
**No-change-needed (1):** guard on soft delete (rationale above).

## Gate

**SHIP-WITH-NOTES** — feature complete, contract published and sync-tested, security caps tested.
Conditions before FTASKS Phase-6 close: deploy + migration apply, Playwright E2E (titled run via loaded
profile, import/export round-trip through the UI), MCP profiles smoke test.
