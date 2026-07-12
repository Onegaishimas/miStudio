# Technical Design Document: MCP Server & Cross-Feature Grouping

**Document ID:** 010_FTDD|MCP_Server
**Version:** 1.1 (implemented 2026-07-12)
**Last Updated:** 2026-07-12
**Status:** Implemented
**Related PRD:** [010_FPRD|MCP_Server](../prds/010_FPRD|MCP_Server.md)

---

## 1. System Architecture

### 1.1 Component Topology

```
┌──────────────┐   MCP (streamable HTTP :8765,       ┌──────────────────────┐
│  MCP Client  │   Authorization: Bearer <token>)    │  mcp-server          │
│ (Claude Code,│ ───────────────────────────────────▶│  (backend image,     │
│  Codex, …)   │ ◀─────────────────────────────────── │  python -m           │
└──────────────┘        tool results                 │  src.mcp_server)     │
                                                     └─────────┬────────────┘
                                                               │ httpx → /api/v1 ONLY
                                                               ▼
┌──────────────┐   REST + Socket.IO                  ┌──────────────────────┐
│   Frontend   │ ───────────────────────────────────▶│  FastAPI backend     │
│ FeatureGroups│ ◀── WS: extractions/{id}/           │  + feature_groups    │
│ Panel        │      feature-groups, mcp/approvals  │  + mcp_approvals     │
└──────────────┘                                     └───┬──────────┬───────┘
                                                         │          │
                                                 ┌───────▼───┐  ┌───▼──────────────┐
                                                 │ PostgreSQL│  │ Redis → Celery    │
                                                 │ (5 new    │  │ low_priority queue│
                                                 │  tables)  │  │ feature_grouping  │
                                                 └───────────┘  └──────────────────┘
```

Key invariants:
- **BR-1.3:** the MCP server holds no DB connection and imports no service modules; it is a REST client. Provenance, validation, and job-queue behavior are identical to UI actions.
- **BR-8.6:** the frontend and MCP consume the same grouping endpoints — one source of truth.
- The MCP server never consumes WebSockets; agents poll (`get_task_status`).

### 1.2 Request Flows

**Grouping precompute:** agent/UI → `POST /extractions/{id}/feature-groups/compute` → 202 + task_queue row → Celery `feature_grouping` task (low_priority) → index/groups persisted → WS `feature_groups:completed` (UI) / polling (agent).

**Approval-mode steering:** agent → `steer_compare` tool → (approval on) MCP server `POST /api/v1/mcp/approvals` (creates `pending` row, returns request_id) → agent polls request status → operator approves in Steering panel → backend submits the stored payload to `/steering/async/compare` itself, stores `steering_task_id` on the request → agent's next poll returns `approved` + task_id → normal result polling.

---

## 2. Database Schema

### 2.1 New Tables

```sql
CREATE TABLE feature_grouping_runs (
    id UUID PRIMARY KEY,
    extraction_id VARCHAR NOT NULL,          -- ExtractionJob id
    status VARCHAR(20) NOT NULL,             -- pending|computing|completed|failed
    params JSONB NOT NULL,                   -- {context_window, similarity_threshold, top_examples, min_group_size}
    params_hash VARCHAR(64) NOT NULL,        -- sha256 of canonical params (idempotency)
    feature_count INTEGER,
    group_count INTEGER,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    completed_at TIMESTAMP
);
CREATE INDEX ix_fgr_extraction ON feature_grouping_runs (extraction_id);

CREATE TABLE feature_token_index (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES feature_grouping_runs(id) ON DELETE CASCADE,
    extraction_id VARCHAR NOT NULL,
    feature_id VARCHAR NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    neuron_index INTEGER NOT NULL,
    raw_token TEXT NOT NULL,                 -- most common raw surface form
    normalized_token TEXT NOT NULL,
    token_rank INTEGER NOT NULL,             -- 1 = feature's top token
    weight FLOAT NOT NULL,                   -- activation-weighted share
    context_tokens JSONB                     -- normalized ±5-token context bag
);
CREATE INDEX ix_fti_ext_token ON feature_token_index (extraction_id, normalized_token);
CREATE INDEX ix_fti_feature ON feature_token_index (feature_id);

CREATE TABLE feature_groups (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES feature_grouping_runs(id) ON DELETE CASCADE,
    extraction_id VARCHAR NOT NULL,
    normalized_token TEXT NOT NULL,
    display_token TEXT NOT NULL,
    member_count INTEGER NOT NULL,
    cohesion FLOAT NOT NULL                  -- mean pairwise cosine within group
);
CREATE INDEX ix_fg_ext ON feature_groups (extraction_id);
CREATE INDEX ix_fg_ext_token ON feature_groups (extraction_id, normalized_token);

CREATE TABLE feature_group_members (
    group_id UUID NOT NULL REFERENCES feature_groups(id) ON DELETE CASCADE,
    feature_id VARCHAR NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    similarity FLOAT NOT NULL,               -- cosine to group centroid
    context_snippet TEXT,                    -- highest-activation "prefix *prime* suffix", ≤160 chars
    PRIMARY KEY (group_id, feature_id)
);

CREATE TABLE agent_approval_requests (
    id UUID PRIMARY KEY,
    tool_name VARCHAR(50) NOT NULL,          -- steer_compare|steer_sweep|steer_combined
    payload JSONB NOT NULL,                  -- full steering request body
    status VARCHAR(10) NOT NULL DEFAULT 'pending',  -- pending|approved|denied|expired
    reason TEXT,
    steering_task_id VARCHAR,                -- set on approval
    created_at TIMESTAMP NOT NULL DEFAULT now(),
    resolved_at TIMESTAMP
);
```

### 2.2 Enum Migration

```sql
ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'mcp_agent';
```
Additive, no table rewrite. **Ordering pitfall:** `ADD VALUE` cannot run inside the same transaction that later uses the value — keep it in its own Alembic revision with `op.execute` outside a batch, mirroring how `enhanced_llm` was added.

### 2.3 Design decision: why not `FeatureAnalysisCache`
`FeatureAnalysisCache` is keyed per-feature; grouping is per-extraction with cross-feature rows and a run lifecycle (params, recompute, cascade delete). Dedicated tables keep queries indexable (`(extraction_id, normalized_token)`) and cleanup trivial (`run_id` cascade).

### 2.4 Staleness model
Extraction jobs are immutable once `completed` → the index never goes stale. Mutable attributes (labels, stars, favorites) are **joined live** from `features` at query time, never copied into group rows. Recompute happens only on `?force=true` or a changed `params_hash`; a successful new run deletes the previous run's rows.

---

## 3. Cross-Feature Grouping Algorithm

### 3.1 Inputs (all already persisted)
Per feature in the extraction: top N examples (default 10) from `feature_activations` — `prime_token`, `prefix_tokens`, `suffix_tokens`, `max_activation`.

### 3.2 Token Normalization (`utils/token_normalization.py`)
1. Strip BPE markers: leading `▁` (SentencePiece), `Ġ` (GPT-2 BPE), leading/trailing `##` (WordPiece)
2. Unicode NFKC normalize
3. Lowercase
4. Strip surrounding punctuation (keep interior: `don't` stays)
5. Empty result after stripping (pure punctuation) → token excluded from grouping
`display_token` = most frequent raw surface form among members.

### 3.3 Index Build
For each feature: take top N examples, compute activation-weighted token counts of `prime_token`; the top-ranked normalized token(s) (rank ≤ 3, weight ≥ 0.2) get `feature_token_index` rows with a context bag = normalized `prefix_tokens[-5:] + suffix_tokens[:5]` across examples.

### 3.4 Bucketing & Context Subgroups
1. Bucket features by `(extraction_id, normalized_token)` at rank 1.
2. Within each bucket of size ≥ 2: build per-feature context documents; sklearn `TfidfVectorizer` (pre-tokenized input, `analyzer=lambda x: x`) → cosine similarity matrix.
3. Threshold graph at `similarity_threshold` (default **0.35**) → connected components = subgroups. Buckets of size < `min_group_size` (default 2) are indexed but not materialized as groups.
4. **Cohesion** = mean pairwise cosine within the subgroup; member `similarity` = cosine to the subgroup centroid.

Complexity: buckets are small (token-conditioned); O(n²) per bucket is fine at 50k features / BR-6.1 targets. The heavy step is the batched read of `feature_activations` (partitioned by feature_id — read per feature-id batch, never a full scan).

### 3.5 Related-Features Query (on demand, no precompute needed beyond the index)
`GET /features/{id}/related` = union of: (a) features sharing any rank ≤ 3 normalized token (index lookup); (b) context-bag Jaccard ≥ threshold against (a) candidates; (c) existing `AnalysisService.calculate_correlations` results (cached). Each result carries `link_types: [shared_token|context|correlation]` and a blended `score`.

---

## 4. MCP Server Design

### 4.1 Package Layout
```
backend/src/mcp_server/
├── __init__.py
├── __main__.py          # entry: parses --stdio flag, runs server
├── config.py            # pydantic-settings, env prefix MCP_
├── server.py            # FastMCP instance, auth middleware, /health, category registration
├── client.py            # httpx.AsyncClient wrapper for /api/v1 + error translation
└── tools/
    ├── discovery.py     # read: extractions, trainings
    ├── features.py      # read: search/detail/examples/analysis
    ├── groups.py        # groups: grouping endpoints
    ├── steering.py      # steering: async wrappers + guardrails + approval path
    ├── experiments.py   # experiments CRUD
    ├── labeling.py      # labeling write-back + enhanced labeling
    ├── jobs.py          # get_task_status
    └── admin.py         # destructive (off by default)
```

### 4.2 Configuration (env, prefix `MCP_`)
| Variable | Default | Purpose |
|----------|---------|---------|
| `MCP_AUTH_TOKEN` | *(required)* | Bearer token; empty → startup refusal unless anonymous allowed |
| `MCP_ALLOW_ANONYMOUS` | `false` | Dev/stdio only |
| `MCP_PORT` | `8765` | Streamable-HTTP port, bound `0.0.0.0` (LAN default — product decision) |
| `MCP_TOOL_CATEGORIES` | `read,groups,steering,labeling,experiments,jobs` | Comma list; `admin` opt-in |
| `MCP_STEERING_MAX_CONCURRENT` | `2` | In-flight agent steering tasks |
| `MCP_STEERING_MAX_NEW_TOKENS` | `512` | Ceiling clamped in tool layer |
| `MCP_STEERING_APPROVAL` | `false` | Operator-approval mode |
| `MISTUDIO_API_URL` | `http://backend:8000` | Backend base URL |

### 4.3 Auth Middleware
Streamable-HTTP requests must carry `Authorization: Bearer <token>`; comparison via `hmac.compare_digest` (constant-time, mirroring `main.py`'s `X-Internal-Token` check). 401 JSON on failure. stdio transport bypasses HTTP auth (process-level trust) — allowed only with `MCP_ALLOW_ANONYMOUS=true` or a set token.

### 4.4 Tool Conventions (BR-1.4/1.5)
- FastMCP decorator tools with typed parameters → auto JSON schema; docstrings state ranges, GPU implications, and async behavior
- Ranges enforced before the HTTP call: `strength ∈ [-300, 300]`, `max_new_tokens ∈ [1, min(2048, MCP_STEERING_MAX_NEW_TOKENS)]`
- Async pattern: start tools return `{task_id, status: "queued", hint: "poll get_task_status"}`; backend 4xx/5xx `detail` passed through verbatim (BR-6.2: 503 labeling / circuit-breaker states surfaced with reset path)
- List responses capped (`limit ≤ 100`), snippets truncated ≤ 160 chars (context-window discipline, BR-2.7)
- Audit: one structured log line per call — `{tool, args_digest, status, duration_ms}` (BR-5.4)

### 4.5 Steering Guardrails
Tool layer maintains an in-process semaphore (`MCP_STEERING_MAX_CONCURRENT`) counting unresolved agent-submitted steering tasks; clamps `max_new_tokens`; in approval mode routes to `/api/v1/mcp/approvals` instead of `/steering/async/*` and returns the approval request_id (agent polls it — durable across reconnects, BR-6.3).

---

## 5. API Design

Endpoints, params, and response shapes as specified in FPRD §5 (single source; not duplicated here). Conventions: pagination `limit ≤ 100`/`offset`; 202 + task_queue for compute; 409 structured detail for the aqua guard; all registered in `router.py` and visible in Swagger (BR-2.5a).

### 5.1 `PATCH /features/{id}` changes
```python
class FeatureUpdate(BaseModel):
    ...existing fields...
    label_source: Literal["user", "mcp_agent"] | None = None   # whitelist
    override_protected: bool = False
```
Guard: if target has `star_color == 'aqua'` and any of name/category/description change and not `override_protected` → `409 {"detail": {"code": "PROTECTED_LABEL", "message": ..., "hint": "pass override_protected=true"}}`.

---

## 6. WebSocket Events

| Channel | Event | Payload |
|---------|-------|---------|
| `extractions/{id}/feature-groups` | `feature_groups:progress` | `{progress, stage, features_indexed, total}` |
| | `feature_groups:completed` | `{run_id, group_count, feature_count}` |
| | `feature_groups:failed` | `{error}` |
| `mcp/approvals` | `approval:created` | `{request_id, tool_name, summary}` |
| | `approval:resolved` | `{request_id, status, steering_task_id?}` |

Emitted via the standard `websocket_emitter` internal-endpoint pattern from the Celery task / approvals endpoints.

---

## 7. Frontend Design

- **`FeatureGroupsPanel.tsx`** (panels/) + `components/featureGroups/`: `ComputeIndexBanner` (no-index / computing / stale-params states), `GroupList` (expandable rows, sort/search/filters), `GroupMembersTable` (selection checkboxes, member click → Feature Detail modal), `RelatedFeaturesDrawer` (seed lookup)
- **`featureGroupsStore.ts`** (Zustand): index status, groups page, expanded group members, filters, selection set; polling fallback when WS disconnected (standard store pattern)
- **`useFeatureGroupsWebSocket(extractionId)`**: subscribe on mount, events → store
- **Steering hand-off:** "Steer selected" writes selected feature IDs into `steeringStore` (the existing pre-populate pattern) and navigates to the Steering panel
- **Approvals surface:** `ApprovalsBanner` in the Steering panel; badge count; Approve/Deny actions; live via `mcp/approvals` channel

---

## 8. Deployment

### 8.1 Docker Compose (profile `mcp`)
```yaml
mcp-server:
  image: hitsai/mistudio-backend:latest
  profiles: ["mcp"]
  command: python -m src.mcp_server
  environment:
    MISTUDIO_API_URL: http://backend:8000
    MCP_AUTH_TOKEN: ${MCP_AUTH_TOKEN}
    MCP_TOOL_CATEGORIES: ${MCP_TOOL_CATEGORIES:-read,groups,steering,labeling,experiments,jobs}
    MCP_STEERING_APPROVAL: ${MCP_STEERING_APPROVAL:-false}
  ports: ["8765:8765"]          # LAN-reachable by default (token required)
  depends_on:
    backend: { condition: service_healthy }
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8765/health"]
```
Disable = don't pass `--profile mcp`. K8s: `mistudio-mcp` Deployment (backend image, same env from Secret) + ClusterIP Service `:8765`; no Ingress shipped — operators expose deliberately. Client config: `claude mcp add --transport http mistudio http://<host>:8765/mcp --header "Authorization: Bearer $MCP_AUTH_TOKEN"`.

### 8.2 Dependency
`mcp>=1.9` added to `backend/requirements.txt` (server code ships inside the backend image; only the `mcp-server` process imports it).

---

## 9. Performance & Risk

| Concern | Design answer |
|---------|---------------|
| 50k-feature index build time | Batched partitioned reads + vectorized TF-IDF per bucket; target < 5 min on reference CPU; progress every 500 features |
| BR-6.1 query latency | All group queries hit indexed tables (`(extraction_id, normalized_token)`); warm ≤ 2 s trivially |
| Agent context blow-up | limit caps + snippet truncation + summary-first shapes |
| GPU monopolization | Semaphore + token ceiling + approval mode; steering-mode lifecycle unchanged |
| Curated-label loss | 409 aqua guard; `override_protected` explicit and logged |
| Schema drift MCP↔API | Tools are thin wrappers over the same Pydantic-validated endpoints; integration tests call tools against a live backend |
| Similarity quality (TF-IDF vs embeddings) | Accepted tradeoff; pgvector upgrade path documented (Neuronpedia Postgres already runs pgvector) |

---

## 10. Development Phases

Mirrors FTASKS: 1 Data layer → 2 Grouping service/job → 3 REST endpoints → 4 Label write-back/guard → 5 MCP foundation → 6 Tool catalog + approvals → 7 Deployment → 8 Frontend → 9 Docs/E2E. Backend phases 1–4 are independently shippable (Feature Groups UI works without the MCP server, per BRD acceptance #3).

---

*Related: [PRD](../prds/010_FPRD|MCP_Server.md) | [TID](../tids/010_FTID|MCP_Server.md) | [Tasks](../tasks/010_FTASKS|MCP_Server.md)*
