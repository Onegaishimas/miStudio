# Feature PRD: MCP Server & Cross-Feature Grouping

**Document ID:** 010_FPRD|MCP_Server
**Version:** 1.0
**Last Updated:** 2026-07-12
**Status:** Planned
**Priority:** P1 (Important Feature)
**Source BRD:** [BRD-MIS-MCP-001](../brds/miStudio-MCP-Server-BRD.md)

---

## 1. Overview

### 1.1 Purpose
Make miStudio agent-native: expose the post-extraction workflow (analyze → group → steer → relabel) to MCP-capable AI clients (Claude Code, Codex, any MCP client), and add a first-class **cross-feature grouping** capability — usable from both agents and the frontend — that answers "which features fire on the same top activating token with similar context?"

### 1.2 User Problem
- There is no first-class way for an agentic AI to operate miStudio; raw HTTP scripting forces every agent session to rediscover API conventions (202/polling, pagination, channels).
- The REST API is single-resource-oriented: finding groups of semantically linked features requires paging through thousands of features client-side — prohibitively slow and token-expensive for agents, and impossible in the UI.
- The value proposition is **iteration speed** (BRD BO-2): an agent can run hundreds of query → hypothesize → steer → validate → relabel loops per session; today a human runs a handful.

### 1.3 Solution
1. An **MCP server** shipped with the stack (separate container, official `mcp` Python SDK, streamable-HTTP transport, bearer-token auth) exposing ~24 tools across gated categories.
2. **New REST endpoints** for cross-feature grouping backed by a precompute job (token→feature inverted index + context-similarity subgroups), consumed identically by MCP tools and a new frontend **Feature Groups** view (one source of truth, BR-8.6).
3. **Guardrails**: tool-category configuration, steering concurrency/token ceilings, operator-approval mode, agent provenance on label writes, protected-star handling.

---

## 2. User Stories

From BRD §6 (IDs preserved):

- **US-1 Find linked feature groups** — agent queries labeled features from an extraction, returns groups sharing a top activating token with similar context.
- **US-2 Interpret a feature group** — agent pulls members' examples, token statistics, logit-lens data; synthesizes a proposed shared meaning.
- **US-3 Validate by steering** — agent runs comparisons/sweeps (individual and combined) and evaluates whether generations shift as the hypothesized label predicts.
- **US-4 Refine labels with evidence** — agent updates names/descriptions/notes with the validated interpretation and links saved experiments as evidence.
- **US-5 Iterate fast** — full loop runs over many groups in one session with an end-of-session summary.
- **US-6 Stay in control** — operator configures tool categories, guardrails, and (optionally) approval mode for GPU-consuming operations.
- **US-7 Explore groups in the UI** — researcher browses the same groups in the frontend, inspects members, jumps to feature detail or steering.

---

## 3. Functional Requirements

### 3.1 MCP Server Foundation (BRD §7.1)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-1.1 | MCP server exposing backend capabilities as MCP tools; streamable-HTTP transport (port 8765) + stdio dev mode | BR-1.1 | Planned |
| FR-1.2 | Deployable in Docker Compose (profile `mcp`) and K8s; independently disableable | BR-1.2 | Planned |
| FR-1.3 | Backend access exclusively via `/api/v1` REST (httpx client) — never DB/service bypass | BR-1.3 | Planned |
| FR-1.4 | Self-documenting tool schemas: typed params, documented ranges (strength [-300,+300], max_new_tokens [1,2048]) enforced and surfaced | BR-1.4 | Planned |
| FR-1.5 | Async 202 pattern translated: start-tool returns task/job ID; `get_task_status` supports polling; structured backend error detail passed through | BR-1.5 | Planned |

### 3.2 Feature Query & Cross-Feature Grouping (BRD §7.2)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-2.1 | Tools to list extractions/trainings, list/filter features (label presence, category, star/favorite, activation stats), fetch feature detail | BR-2.1 | Planned |
| FR-2.2 | Tools for per-feature analysis: examples with per-token activations, token statistics, logit-lens, correlations, ablation, NLP analysis | BR-2.2 | Planned |
| FR-2.3 | Cross-feature grouping: groups sharing the same normalized top activating token, refined by context similarity (TF-IDF/cosine over ±5-token windows) | BR-2.3 | Planned |
| FR-2.4 | Query by token (exact/normalized/prefix match) and by seed feature (related via shared tokens/context/correlations) | BR-2.4 | Planned |
| FR-2.5 | Grouping implemented as first-class REST endpoints under `/api/v1` (see §5); consumed by both MCP tools and frontend; documented in API reference + Swagger | BR-2.5, BR-2.5a | Planned |
| FR-2.6 | Server-side execution with compact results; precompute runs as a standard background job (Celery `low_priority`, task_queue record, WebSocket progress), results persisted for reuse | BR-2.5b | Planned |
| FR-2.7 | Group results include per member: feature ID, neuron index, label/category, top token(s), context snippet, activation stats; plus group-level cohesion | BR-2.6 | Planned |
| FR-2.8 | All list results pageable (`limit ≤ 100` + offset) and bounded for agent context windows | BR-2.7 | Planned |

### 3.3 Steering & Guardrails (BRD §7.3)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-3.1 | Tools wrapping async compare/sweep/combined, result polling, task cancellation | BR-3.1 | Planned |
| FR-3.2 | Steering-mode tools (status/enter/exit/health) with GPU-load implications documented in tool descriptions | BR-3.2 | Planned |
| FR-3.3 | Saved-experiment CRUD tools | BR-3.3 | Planned |
| FR-3.4 | Guardrails: `MCP_STEERING_MAX_CONCURRENT` (default 2), `MCP_STEERING_MAX_NEW_TOKENS` (default 512) | BR-3.4 | Planned |
| FR-3.5 | **Operator-approval mode** (`MCP_STEERING_APPROVAL=true`): agent steering calls create durable approval requests; operator approves/denies in UI; backend submits the stored payload on approval; agent polls request status | BR-3.4, US-6 | Planned |

### 3.4 Labeling & Write-Back (BRD §7.4)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-4.1 | Tool to update feature name, category, description, notes | BR-4.1 | Planned |
| FR-4.2 | Tools to trigger enhanced labeling + fetch latest result; (category-gated) bulk labeling start/monitor | BR-4.2 | Planned |
| FR-4.3 | Agent writes carry `label_source='mcp_agent'` (new enum value); aqua-starred features return 409 on label edits unless `override_protected=true` (tool default false) | BR-4.3 | Planned |
| FR-4.4 | Documented evidence convention: `[MCP <date>] evidence: experiment <id> — <summary>` appended to notes | BR-4.4 | Planned |

### 3.5 Security & Safety (BRD §7.5)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-5.1 | Bearer-token auth (`MCP_AUTH_TOKEN`, `hmac.compare_digest`); refuses startup with empty token unless `MCP_ALLOW_ANONYMOUS=true`; port LAN-reachable by default (documented risk + firewall guidance) | BR-5.1 | Planned |
| FR-5.2 | Category gating via `MCP_TOOL_CATEGORIES` (`read, groups, steering, labeling, experiments, jobs, admin`); default excludes `admin` | BR-5.2 | Planned |
| FR-5.3 | Destructive tools (`admin` category) opt-in, marked destructive in tool metadata | BR-5.3 | Planned |
| FR-5.4 | Structured audit log per tool call (tool, args digest, backend status, duration) | BR-5.4 | Planned |

### 3.6 Performance & Reliability (BRD §7.6)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-6.1 | Feature queries < 2 s; grouping queries < 10 s over ~50k features (warm ≤ 2 s, precomputed index) | BR-6.1 | Planned |
| FR-6.2 | Graceful degradation: labeling 503 / steering circuit-breaker states surfaced verbatim with the reset path | BR-6.2 | Planned |
| FR-6.3 | Long-running tasks trackable across agent reconnects (durable task_queue/DB records) | BR-6.3 | Planned |

### 3.7 Documentation (BRD §7.7)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-8.1 | Manual gains an "MCP Server" section: install/enable, client configs (Claude Code + generic), full tool catalog, worked analyze→group→steer→relabel example | BR-7.1 | Planned |
| FR-8.2 | API reference documents the grouping + approvals endpoints; Core Workflow documents the Feature Groups view | BR-7.2 | Planned |

### 3.8 Frontend Feature Groups View (BRD §7.8)
| Requirement | Description | BRD | Status |
|-------------|-------------|-----|--------|
| FR-7.1 | Feature Groups view consuming the grouping endpoints; groups by top activating token with context subgroups | BR-8.1 | Planned |
| FR-7.2 | Same query dimensions as the API: token search, seed-feature related lookup, filters (label presence, category, star/favorite) | BR-8.2 | Planned |
| FR-7.3 | Group rows show shared token(s) + cohesion; member rows show ID/index, label, context snippet; member click → feature detail | BR-8.3 | Planned |
| FR-7.4 | "Steer selected" hand-off pre-populates the Steering panel with selected members | BR-8.4 | Planned |
| FR-7.5 | Precompute job surfaced with standard start/progress/completion UI (WebSocket + polling fallback) | BR-8.5 | Planned |
| FR-7.6 | UI and MCP consume the same records/queries — no agent-only or UI-only results | BR-8.6 | Planned |

---

## 4. User Interface

### 4.1 Feature Groups View
```
┌────────────────────────────────────────────────────────────────────┐
│ Feature Groups — extraction: ext_abc123           [Compute Index]  │
├────────────────────────────────────────────────────────────────────┤
│ Search token: [ love___ ]  Min size: [2▾]  Sort: [Size▾]           │
│ Filters: [Has label ▾] [Category ▾] [Star ▾] [Favorite ☐]          │
├────────────────────────────────────────────────────────────────────┤
│ ▼ "love" (12 members, cohesion 0.81)                               │
│   ☐ #4821  expressions of romantic love     "I really *love* you"  │
│   ☐ #1290  affection toward family          "we *love* our mom"    │
│   ☐ #7734  (unlabeled)                      "would *love* to see"  │
│   [Steer selected →]                                               │
│ ▶ "heart" (7 members, cohesion 0.74)                               │
│ ▶ "amour" (3 members, cohesion 0.69)                               │
├────────────────────────────────────────────────────────────────────┤
│ Index: computed 2026-07-12 14:02 · 48,912 features · 2,317 groups  │
└────────────────────────────────────────────────────────────────────┘
```
- No index yet → banner with **Compute Index** button; job progress bar during computation.
- Member click → existing Feature Detail modal. "Related features" drawer from any feature (seed lookup).

### 4.2 Steering Approvals Surface (approval mode only)
Banner/drawer in the Steering panel: pending agent requests with payload summary (features, strengths, prompt, max_new_tokens) and **Approve / Deny** actions; badge count in the panel header.

---

## 5. API Endpoints (new)

### 5.1 Feature grouping (router `feature_groups.py`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/extractions/{id}/feature-groups/compute` | POST | Start precompute job (202, `{task_id, run_id}`); `?force=true` recomputes |
| `/api/v1/extractions/{id}/feature-groups/status` | GET | `{status, progress, run_id, params, group_count, computed_at}` |
| `/api/v1/extractions/{id}/feature-groups` | GET | Paginated groups; params `token`, `search`, `min_group_size`, `sort_by`, `limit`, `offset` |
| `/api/v1/extractions/{id}/feature-groups/{group_id}` | GET | Group members (labels/stars joined live); filters `category`, `has_label`, `star_color`, `is_favorite` |
| `/api/v1/extractions/{id}/features/by-token` | GET | Flat feature list from the inverted index; `token`, `match=exact\|normalized\|prefix` |
| `/api/v1/features/{id}/related` | GET | Seed-feature related set; `min_similarity`, `limit`; `link_types` per result |

### 5.2 Agent approvals (router `mcp_approvals.py`, approval mode)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/mcp/approvals` | GET | List approval requests (filter by status) |
| `/api/v1/mcp/approvals/{id}` | GET | Request detail incl. stored steering payload |
| `/api/v1/mcp/approvals/{id}/approve` | POST | Approve → backend submits the stored steering task; returns steering task_id |
| `/api/v1/mcp/approvals/{id}/deny` | POST | Deny with optional reason |

### 5.3 Modified
- `PATCH /api/v1/features/{id}` — accepts optional `label_source` (whitelist `user|mcp_agent`) and `override_protected` (bool, default false); returns **409** when editing name/category/description of an aqua-starred feature without the override.

---

## 6. Data Model (new)

- **`feature_grouping_runs`** — `id UUID PK`, `extraction_id` (idx), `status`, `params JSONB`, `params_hash`, `feature_count`, `group_count`, `error_message`, `created_at`, `completed_at`
- **`feature_token_index`** — `id BigInt PK`, `run_id FK`, `extraction_id` (idx), `feature_id FK`, `neuron_index`, `raw_token`, `normalized_token`, `token_rank`, `weight`, `context_tokens JSONB`; composite index `(extraction_id, normalized_token)`
- **`feature_groups`** — `id PK`, `run_id FK`, `extraction_id` (idx), `normalized_token`, `display_token`, `member_count`, `cohesion FLOAT`
- **`feature_group_members`** — PK `(group_id, feature_id)`, `similarity`, `context_snippet TEXT`
- **`agent_approval_requests`** — `id UUID PK`, `tool_name`, `payload JSONB`, `status` (`pending|approved|denied|expired`), `reason`, `steering_task_id`, `created_at`, `resolved_at`
- **Enum change:** `label_source_enum` + `mcp_agent` (additive `ALTER TYPE`)

Staleness: extractions are immutable after completion → the index never goes stale; labels/stars joined live at query time; recompute only on `force` or params change.

---

## 7. WebSocket Channels (new)

| Channel | Events | Purpose |
|---------|--------|---------|
| `extractions/{id}/feature-groups` | `feature_groups:progress` / `completed` / `failed` | Precompute job progress |
| `mcp/approvals` | `approval:created` / `approval:resolved` | Live approvals surface |

The MCP server itself never consumes WebSockets — agents poll via `get_task_status` (FR-1.5).

---

## 8. MCP Tool Catalog (~24 tools)

| Category | Tools |
|----------|-------|
| `read` | `list_extractions`, `get_extraction_summary`, `list_trainings`, `search_features`, `get_feature`, `get_feature_examples`, `get_feature_token_analysis`, `get_feature_logit_lens`, `get_feature_correlations`, `get_feature_ablation`, `get_feature_nlp_analysis` |
| `groups` | `compute_feature_groups`, `get_feature_groups`, `get_feature_group_members`, `find_features_by_token`, `find_related_features` |
| `steering` | `steering_status`, `enter_steering_mode`, `exit_steering_mode`, `steer_compare`, `steer_sweep`, `steer_combined`, `get_steering_result`, `cancel_steering_task` |
| `experiments` | `save_experiment`, `list_experiments`, `get_experiment` |
| `labeling` | `update_feature_label`, `run_enhanced_labeling`, `get_enhanced_label` |
| `jobs` | `get_task_status` |
| `admin` (off by default) | `delete_experiment`, `delete_extraction` — marked destructive |

---

## 9. Non-Goals (BRD §4.2)

- No MCP tools for dataset upload, model management, activation extraction, or SAE training (later phase)
- No hosted/cloud MCP offering — local deployment only
- No dedicated "agent activity" UI beyond the approvals surface (agent records are ordinary DB records)
- No bundled agent client — the deliverable is the server

---

## 10. Dependencies

| Dependency | Type |
|------------|------|
| Feature Discovery (004) | Provides features, examples, analysis endpoints |
| Model Steering (006) | Provides async steering API + experiments |
| Enhanced Labeling | Write-back target, aqua-star semantics |
| Task queue / Celery / WebSocket infrastructure (008) | Job pattern for precompute + progress |
| `mcp` Python SDK (new dependency) | MCP server implementation |

---

## 11. Success Criteria (BRD §13)

| Metric | Target |
|--------|--------|
| Agent produces steering-validated label for a feature group (cold start) | ≤ 10 min median on reference hardware |
| Grouping query latency (50k-feature extraction) | ≤ 10 s cold, ≤ 2 s warm |
| Agent loop completion rate without human intervention on tool errors | ≥ 90% |
| In-scope endpoint coverage as MCP tools | 100% |
| Documented client configs | Claude Code + one other MCP client |

---

## 12. Testing Requirements

- Unit: token normalization, grouping algorithm (bucketing, cohesion), auth middleware, category gating, guardrail enforcement, aqua-guard 409 paths
- Integration: precompute job end-to-end against a seeded extraction; grouping endpoints pagination/filters; approval flow (create → approve → steering task submitted); MCP tools against a live backend (start job → poll → result)
- E2E acceptance (BRD §14): fresh Claude Code session performs enumerate → group → inspect → sweep → relabel with records visible in the UI; frontend Feature Groups walkthrough; disable paths (category off, server off) leave REST/UI unaffected

---

## 13. Implementation Considerations

- Response sizing for agent context windows: summary-first payloads, `limit ≤ 100`, context snippets truncated (~160 chars)
- The grouping index reads `feature_activations` (range-partitioned by feature_id) — batch reads per feature set, never full-table scans
- TF-IDF context similarity chosen over embeddings to avoid new heavy deps; pgvector (already deployed for Neuronpedia) is the natural upgrade path
- First authenticated component in the stack — bearer check must be constant-time (`hmac.compare_digest`), mirroring `internal_api_secret`

## 14. Open Questions

1. Should approval-mode requests expire automatically (proposed: 24 h → `expired`)?
2. Should `find_related_features` blend existing `AnalysisService` correlations by default, or only on request (cost: per-feature cache warm-up)?
3. Rate limiting beyond steering (e.g., global tool-calls/min) — deferred unless abuse observed?

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [BRD](../brds/miStudio-MCP-Server-BRD.md) | [TDD](../tdds/010_FTDD|MCP_Server.md) | [TID](../tids/010_FTID|MCP_Server.md) | [Tasks](../tasks/010_FTASKS|MCP_Server.md)*
