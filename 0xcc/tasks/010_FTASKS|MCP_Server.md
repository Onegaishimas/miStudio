# Feature Tasks: MCP Server & Cross-Feature Grouping

**Document ID:** 010_FTASKS|MCP_Server
**Version:** 1.0
**Last Updated:** 2026-07-12
**Status:** Planned
**Related PRD:** [010_FPRD|MCP_Server](../prds/010_FPRD|MCP_Server.md)

---

## Task Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Data Layer & Migrations | 5 tasks | ⬜ Not started |
| Phase 2: Grouping Service + Celery Job | 6 tasks | ⬜ Not started |
| Phase 3: Grouping REST Endpoints | 7 tasks | ⬜ Not started |
| Phase 4: Label Write-Back & Aqua Guard | 4 tasks | ⬜ Not started |
| Phase 5: MCP Server Foundation | 6 tasks | ⬜ Not started |
| Phase 6: MCP Tool Catalog + Approval Mode | 8 tasks | ⬜ Not started |
| Phase 7: Deployment | 5 tasks | ⬜ Not started |
| Phase 8: Frontend Feature Groups + Approvals | 9 tasks | ⬜ Not started |
| Phase 9: Documentation & E2E Acceptance | 5 tasks | ⬜ Not started |

**Total: 55 tasks**

---

## Phase 1: Data Layer & Migrations

### Task 1.1: Grouping tables migration
- [ ] Alembic revision: `feature_grouping_runs`, `feature_token_index`, `feature_groups`, `feature_group_members` (+ indexes per TDD §2.1)
- [ ] Include `agent_approval_requests` table
- [ ] Downgrade path drops in FK order

**Files:**
- `backend/alembic/versions/xxx_feature_grouping_tables.py`

### Task 1.2: label_source enum migration (separate revision)
- [ ] `ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'mcp_agent'` in its own revision (TID Pitfall 1)
- [ ] Add `mcp_agent` to conftest enum-creation list

**Files:**
- `backend/alembic/versions/xxx_add_mcp_agent_label_source.py`
- `backend/tests/conftest.py`

### Task 1.3: SQLAlchemy models
- [ ] `FeatureGroupingRun`, `FeatureTokenIndex`, `FeatureGroup`, `FeatureGroupMember` in `models/feature_grouping.py`
- [ ] `AgentApprovalRequest` in `models/agent_approval.py`
- [ ] Register all in `models/__init__.py` (TID Pitfall 2)

### Task 1.4: Token normalization utility
- [ ] `normalize_token()` per TDD §3.2 (BPE markers, NFKC, lowercase, punctuation strip)
- [ ] Unit tests incl. `▁Love`, `Ġthe`, `##ing`, `...`→None, `don't`

**Files:**
- `backend/src/utils/token_normalization.py`
- `backend/tests/unit/test_token_normalization.py`

### Task 1.5: Grouping schemas
- [ ] `GroupingParams` (defaults: window 5, threshold 0.35, top_examples 10, min_group_size 2) + response models per TID §2.3

**Files:**
- `backend/src/schemas/feature_group.py`

---

## Phase 2: Grouping Service + Celery Job

### Task 2.1: Index build in FeatureGroupingService
- [ ] Run lifecycle (create, params_hash idempotency, replace-on-success)
- [ ] Batched top-example loads from partitioned `feature_activations` (TID Pitfall 3)
- [ ] Activation-weighted token ranking (ranks 1–3, weight ≥ 0.2) + context bags

### Task 2.2: Bucketing, subgroups, cohesion
- [ ] Per-bucket TF-IDF (`analyzer=lambda bag: bag`, TID Pitfall 7) → cosine → connected components at threshold
- [ ] Cohesion + member similarity + context_snippet (≤160 chars)

**Files:**
- `backend/src/services/feature_grouping_service.py`

### Task 2.3: Celery task
- [ ] `compute_feature_groups` mirroring `nlp_analysis_tasks.py` (BaseTask, task_queue entry `task_type="feature_grouping"`)
- [ ] Autodiscover + route to `low_priority` in `core/celery_app.py`

**Files:**
- `backend/src/workers/feature_grouping_tasks.py`
- `backend/src/core/celery_app.py`

### Task 2.4: WebSocket emitters
- [ ] `emit_feature_groups_progress/completed/failed` on channel `extractions/{id}/feature-groups`

**Files:**
- `backend/src/workers/websocket_emitter.py`

### Task 2.5: Service unit tests
- [ ] Synthetic extraction → expected buckets/subgroups/cohesion; idempotency; replace-on-recompute

**Files:**
- `backend/tests/unit/test_feature_grouping_service.py`

### Task 2.6: Job integration test
- [ ] Seeded extraction end-to-end: compute → rows persisted → progress events emitted

**Files:**
- `backend/tests/integration/test_feature_grouping_job.py`

---

## Phase 3: Grouping REST Endpoints

### Task 3.1: Router scaffold + registration
- [ ] `endpoints/feature_groups.py`, registered in `router.py` with `tags=["feature-groups"]`

### Task 3.2: Compute + status endpoints
- [ ] `POST /extractions/{id}/feature-groups/compute` (202; 409 while computing; params_hash short-circuit; `force`)
- [ ] `GET /extractions/{id}/feature-groups/status`

### Task 3.3: Group list endpoint
- [ ] `GET /extractions/{id}/feature-groups` — `token`, `search` (ILIKE), `min_group_size`, `sort_by`, pagination `limit ≤ 100`

### Task 3.4: Group members endpoint
- [ ] `GET .../feature-groups/{group_id}` — live join to `features` (TID Pitfall 4); filters `category`, `has_label`, `star_color`, `is_favorite`

### Task 3.5: By-token endpoint
- [ ] `GET /extractions/{id}/features/by-token` — `match=exact|normalized|prefix`

### Task 3.6: Related-features endpoint
- [ ] `GET /features/{id}/related` — shared-token lookup + context Jaccard + correlations blend; `link_types` per result

### Task 3.7: Endpoint tests
- [ ] Pagination/filter/409/short-circuit coverage; Swagger renders all routes (BR-2.5a)

**Files:**
- `backend/src/api/v1/endpoints/feature_groups.py`
- `backend/src/api/v1/router.py`
- `backend/tests/integration/test_feature_groups_api.py`

---

## Phase 4: Label Write-Back & Aqua Guard

### Task 4.1: FeatureUpdate schema changes
- [ ] Optional `label_source` (whitelist `user|mcp_agent`) + `override_protected: bool = False`

### Task 4.2: Aqua guard in PATCH handler
- [ ] 409 `PROTECTED_LABEL` structured detail when editing name/category/description of aqua feature without override

### Task 4.3: Provenance persistence
- [ ] `label_source='mcp_agent'` stored and returned; `labeled_at` updated

### Task 4.4: Guard tests
- [ ] 409 path, override path, whitelist 422, notes-only edit passes without override

**Files:**
- `backend/src/schemas/feature.py`
- `backend/src/api/v1/endpoints/features.py`
- `backend/tests/unit/test_feature_label_guard.py`

---

## Phase 5: MCP Server Foundation

### Task 5.1: Package scaffold + dependency
- [ ] `backend/src/mcp_server/` (`__main__.py`, `config.py`, `server.py`, `client.py`, `tools/__init__.py`)
- [ ] Add `mcp>=1.9` to `backend/requirements.txt`

### Task 5.2: Configuration
- [ ] `MCPSettings` (env prefix `MCP_`, `MISTUDIO_API_URL` alias) per TDD §4.2

### Task 5.3: Auth + startup guard
- [ ] Bearer middleware (`hmac.compare_digest`, TID Pitfall 5); refuse startup on empty token unless `MCP_ALLOW_ANONYMOUS=true`

### Task 5.4: REST client wrapper
- [ ] `MiStudioClient` with structured-error passthrough (503/circuit-breaker text verbatim, BR-6.2)

### Task 5.5: Health endpoint + stdio flag
- [ ] `/health` on the HTTP app; `--stdio` transport switch in `__main__.py`

### Task 5.6: Foundation unit tests
- [ ] Auth 401/ok/anonymous-refusal; config parsing; error translation

**Files:**
- `backend/src/mcp_server/*`
- `backend/requirements.txt`
- `backend/tests/unit/test_mcp_server_foundation.py`

---

## Phase 6: MCP Tool Catalog + Approval Mode

### Task 6.1: read tools (11)
- [ ] `discovery.py` + `features.py` per FPRD §8; list caps ≤ 100 + snippet truncation (TID Pitfall 6)

### Task 6.2: groups tools (5)
- [ ] `groups.py` wrapping the Phase 3 endpoints

### Task 6.3: steering tools (8) + guardrails
- [ ] `steering.py`: range validation, `max_new_tokens` clamp, in-flight semaphore

### Task 6.4: experiments + labeling + jobs tools
- [ ] `experiments.py` (3), `labeling.py` (3, evidence-notes convention in docstrings), `jobs.py` (`get_task_status`)

### Task 6.5: admin tools (opt-in)
- [ ] `admin.py`: `delete_experiment`, `delete_extraction` marked destructive

### Task 6.6: Category gating + audit log
- [ ] Conditional registration from `MCP_TOOL_CATEGORIES`; structured per-call log line

### Task 6.7: Approvals backend
- [ ] `endpoints/mcp_approvals.py` (list/detail/approve/deny; approve submits stored payload, records `steering_task_id`); WS `mcp/approvals` events; 24 h expiry sweep in Beat schedule
- [ ] `get_approval_status` MCP tool; steering tools route to approvals when `MCP_STEERING_APPROVAL=true`

### Task 6.8: Tool integration tests
- [ ] Tools against live backend: start job → poll → result; approval create→approve→submitted; disabled category absent from `tools/list`

**Files:**
- `backend/src/mcp_server/tools/*.py`
- `backend/src/api/v1/endpoints/mcp_approvals.py`
- `backend/tests/integration/test_mcp_tools.py`

---

## Phase 7: Deployment

### Task 7.1: Docker Compose service
- [ ] `mcp-server` under profile `mcp` per TDD §8.1; port 8765; healthcheck; depends_on backend

### Task 7.2: Environment template
- [ ] `.env.example`: `MCP_AUTH_TOKEN`, `MCP_TOOL_CATEGORIES`, `MCP_STEERING_APPROVAL` + comments incl. LAN-exposure warning

### Task 7.3: K8s manifests
- [ ] `mistudio-mcp` Deployment + ClusterIP Service in `k8s/mistudio-deployment.yaml`; token in secrets template

### Task 7.4: Disable-path verification
- [ ] Compose without profile / k8s scale-0: REST API + Feature Groups UI unaffected (BRD acceptance #3)

### Task 7.5: Client config docs snippet
- [ ] `claude mcp add` example + generic MCP client config

**Files:**
- `docker-compose.yml`, `.env.example`, `k8s/mistudio-deployment.yaml`, `k8s/mistudio-secrets.yaml.example`

---

## Phase 8: Frontend Feature Groups + Approvals

### Task 8.1: Types + API client
- [ ] `types/featureGroups.ts`, `api/featureGroups.ts` (status/compute/list/members/byToken/related)

### Task 8.2: Zustand store
- [ ] `featureGroupsStore.ts` with WS-connected flag + polling fallback

### Task 8.3: WebSocket hook
- [ ] `useFeatureGroupsWebSocket.ts` on `extractions/{id}/feature-groups`

### Task 8.4: ComputeIndexBanner
- [ ] No-index / computing (progress) / completed-stale states

### Task 8.5: GroupList
- [ ] Expandable groups, token search, min-size/sort controls, filters (label/category/star/favorite)

### Task 8.6: GroupMembersTable
- [ ] Selection checkboxes; member click → existing Feature Detail modal; context snippets

### Task 8.7: RelatedFeaturesDrawer
- [ ] Seed-feature related lookup with `link_types` badges

### Task 8.8: Steering hand-off + approvals surface
- [ ] "Steer selected" pre-populates `steeringStore`; `ApprovalsBanner` in Steering panel with Approve/Deny + `mcp/approvals` subscription

### Task 8.9: Frontend tests
- [ ] Store logic, banner states, hand-off (Vitest + RTL)

**Files:**
- `frontend/src/components/panels/FeatureGroupsPanel.tsx`
- `frontend/src/components/featureGroups/*.tsx`
- `frontend/src/components/steering/ApprovalsBanner.tsx`
- `frontend/src/stores/featureGroupsStore.ts`
- `frontend/src/hooks/useFeatureGroupsWebSocket.ts`

---

## Phase 9: Documentation & E2E Acceptance

### Task 9.1: Manual — MCP Server section
- [ ] Install/enable, client configuration (Claude Code + generic), full tool catalog, worked analyze→group→steer→relabel example (BR-7.1)

### Task 9.2: Manual — API reference updates
- [ ] Feature-groups + approvals endpoints alongside Features & Labeling reference (BR-7.2 / BR-2.5a)

### Task 9.3: Manual — Feature Groups UI page
- [ ] Core Workflow addition with screenshots (BR-8 documentation)

### Task 9.4: E2E acceptance walkthrough
- [ ] BRD §14 criteria 1–5 executed and recorded (fresh Claude Code session, UI walkthrough, disable paths, provenance audit)

### Task 9.5: 0xcc closeout
- [ ] Update FPRD/FTDD/FTID statuses Planned → Implemented; PPRD §2.1 row → Complete; CLAUDE.md session log

**Files:**
- `manual/docs/**`, `0xcc/prds/010_FPRD|MCP_Server.md`, `CLAUDE.md`

---

## Relevant Files Summary

### Backend
| File | Purpose |
|------|---------|
| `backend/src/utils/token_normalization.py` | BPE-aware token normalization |
| `backend/src/services/feature_grouping_service.py` | Index build, bucketing, cohesion |
| `backend/src/workers/feature_grouping_tasks.py` | Celery precompute job (low_priority) |
| `backend/src/api/v1/endpoints/feature_groups.py` | Grouping REST endpoints |
| `backend/src/api/v1/endpoints/mcp_approvals.py` | Approval-mode endpoints |
| `backend/src/models/feature_grouping.py` | 4 grouping tables (ORM) |
| `backend/src/models/agent_approval.py` | Approval requests (ORM) |
| `backend/src/schemas/feature_group.py` | Request/response schemas |
| `backend/src/mcp_server/` | MCP server package (config/auth/client/tools) |
| `backend/alembic/versions/…` | 2 migrations (tables; enum value) |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/components/panels/FeatureGroupsPanel.tsx` | Feature Groups view |
| `frontend/src/components/featureGroups/*` | Banner, list, members, related drawer |
| `frontend/src/components/steering/ApprovalsBanner.tsx` | Agent-approval surface |
| `frontend/src/stores/featureGroupsStore.ts` | State + polling fallback |
| `frontend/src/hooks/useFeatureGroupsWebSocket.ts` | Job progress subscription |

---

*Related: [PRD](../prds/010_FPRD|MCP_Server.md) | [TDD](../tdds/010_FTDD|MCP_Server.md) | [TID](../tids/010_FTID|MCP_Server.md)*
