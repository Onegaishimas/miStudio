# Technical Implementation Document: MCP Server & Cross-Feature Grouping

**Document ID:** 010_FTID|MCP_Server
**Version:** 1.0
**Last Updated:** 2026-07-12
**Status:** Planned
**Related TDD:** [010_FTDD|MCP_Server](../tdds/010_FTDD|MCP_Server.md)

---

## 1. Implementation Order

### Phase 1: Data layer & migrations
- Alembic revision A: 5 new tables (feature_grouping_runs, feature_token_index, feature_groups, feature_group_members, agent_approval_requests)
- Alembic revision B (separate!): `ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'mcp_agent'`
- SQLAlchemy models + `token_normalization.py` util

### Phase 2: Grouping service + Celery job
- `feature_grouping_service.py` (index build, bucketing, cohesion)
- `feature_grouping_tasks.py` + celery route + task_queue integration + WS progress

### Phase 3: REST endpoints
- `schemas/feature_group.py`, `endpoints/feature_groups.py`, register in `router.py`

### Phase 4: Label write-back & guard
- `PATCH /features/{id}`: `label_source` whitelist + aqua 409 guard + `override_protected`

### Phase 5: MCP server foundation
- `src/mcp_server/` package: config, auth, client, health, stdio flag; `mcp>=1.9` dep

### Phase 6: Tool catalog + approval mode
- 8 tool modules; category gating; guardrails; `mcp_approvals.py` endpoints + approval flow

### Phase 7: Deployment
- compose profile, k8s manifests, `.env.example`, client-config docs

### Phase 8: Frontend
- store, API client, WS hook, FeatureGroupsPanel + sub-components, approvals banner, steering hand-off

### Phase 9: Docs & E2E
- Manual "MCP Server" section + API reference pages; BRD §14 acceptance walkthrough

---

## 2. File-by-File Implementation

### 2.1 Backend — Data Layer

#### `backend/src/utils/token_normalization.py`
```python
import unicodedata

BPE_LEADING = ("▁", "Ġ")  # ▁ (SentencePiece), Ġ (GPT-2)

def normalize_token(raw: str) -> str | None:
    """Normalize a tokenizer surface form for cross-feature matching.

    Returns None when nothing meaningful remains (pure punctuation/markers).
    """
    t = raw
    for marker in BPE_LEADING:
        if t.startswith(marker):
            t = t[len(marker):]
    t = t.removeprefix("##").removesuffix("##")   # WordPiece
    t = unicodedata.normalize("NFKC", t).lower().strip()
    t = t.strip("\"'`.,;:!?()[]{}<>-—…·")          # surrounding only
    return t or None
```
Unit-test against: `"▁Love"→"love"`, `"Ġthe"→"the"`, `"##ing"→"ing"`, `"..."→None`, `"don't"→"don't"`.

#### Models — `backend/src/models/feature_grouping.py`
One module holding `FeatureGroupingRun`, `FeatureTokenIndex`, `FeatureGroup`, `FeatureGroupMember`; plus `backend/src/models/agent_approval.py` for `AgentApprovalRequest`. Mirror existing model style (String PKs where the codebase uses prefixed ids is NOT needed here — use `UUID(as_uuid=False)` defaults like `neuronpedia` models). **Register all in `models/__init__.py`** (conftest metadata + Alembic autogenerate both depend on it — this was a review finding for NeuronpediaPushJob; don't repeat it).

#### Alembic
Two revisions, enum first-class:
```python
# revision B — enum value, own revision, no transaction-sensitive batching
def upgrade():
    op.execute("ALTER TYPE label_source_enum ADD VALUE IF NOT EXISTS 'mcp_agent'")
```
Also add `'mcp_agent'` to the enum list in `backend/tests/conftest.py`'s enum-creation block (the test DB creates enums explicitly — omitting it breaks the suite).

### 2.2 Backend — Grouping Service & Task

#### `backend/src/services/feature_grouping_service.py`
```python
class FeatureGroupingService:
    """Builds the token→feature index and context subgroups for one extraction."""

    def build_index(self, db: Session, extraction_id: str, params: GroupingParams,
                    progress_cb: Callable[[int, int, str], None]) -> str:
        run = self._create_run(db, extraction_id, params)
        features = self._load_features(db, extraction_id)          # ids + neuron_index
        for batch in chunked(features, 200):
            examples = self._load_top_examples(db, batch, params.top_examples)
            rows = self._index_rows(batch, examples, params)       # normalize + weight
            db.bulk_save_objects(rows); db.commit()
            progress_cb(...)
        self._build_groups(db, run)                                # bucket + TF-IDF + components
        return run.id
```
Key implementation notes:
- `_load_top_examples`: query `feature_activations` **per feature-id batch** ordered by `max_activation DESC LIMIT params.top_examples` — the table is range-partitioned by `feature_id`; never scan unpartitioned.
- `_index_rows`: activation-weighted `Counter` over `normalize_token(prime_token)`; keep ranks 1–3 with weight ≥ 0.2; context bag = normalized `prefix_tokens[-5:] + suffix_tokens[:5]` across kept examples.
- `_build_groups`:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components

vec = TfidfVectorizer(analyzer=lambda bag: bag, lowercase=False)   # pre-tokenized
tfidf = vec.fit_transform(context_bags)                            # one bucket at a time
sim = cosine_similarity(tfidf)
adj = sim >= params.similarity_threshold
n, labels = connected_components(adj, directed=False)
# cohesion = mean of upper-triangle sims within each component
```
- `context_snippet`: from the member's highest-activation example, `f"{prefix_tail} *{prime}* {suffix_head}"`, truncated to 160 chars.
- On success: delete prior run's rows for the extraction (`run_id` cascade), keyed by `params_hash` idempotency check before starting.

#### `backend/src/workers/feature_grouping_tasks.py`
Mirror `nlp_analysis_tasks.py` exactly (BaseTask, `get_db()`, task_queue entries):
```python
@celery_app.task(bind=True, base=BaseTask, name="src.workers.feature_grouping_tasks.compute_feature_groups")
def compute_feature_groups(self, extraction_id: str, params: dict):
    entry = task_queue_service.create_task_entry(task_type="feature_grouping",
                                                 entity_id=extraction_id, entity_type="extraction", ...)
    def progress(done, total, stage):
        emit_progress(f"extractions/{extraction_id}/feature-groups", "feature_groups:progress",
                      {"progress": round(100*done/total), "stage": stage, ...})
    ...
```
Register: add `"src.workers.feature_grouping_tasks"` to `autodiscover_tasks` and route `src.workers.feature_grouping_tasks.*` → `low_priority` in `core/celery_app.py` `task_routes`. Emit `feature_groups:completed|failed` on exit; add matching emitter helpers in `websocket_emitter.py` (`emit_feature_groups_progress(...)` family).

### 2.3 Backend — REST Endpoints

#### `backend/src/schemas/feature_group.py`
`GroupingParams` (context_window=5, similarity_threshold=0.35, top_examples=10, min_group_size=2), `FeatureGroupSummary`, `FeatureGroupMemberOut` (live-joined label fields), `FeatureGroupListResponse`, `GroupingStatusResponse`, `RelatedFeatureOut` (`link_types: list[Literal["shared_token","context","correlation"]]`).

#### `backend/src/api/v1/endpoints/feature_groups.py`
Follow `features.py` conventions (async `get_db`, structured HTTPException detail). Endpoint set per FPRD §5.1. Notable handlers:
- `compute`: 409 if a run is `computing`; 200 short-circuit if completed run with same `params_hash` and not `force`; else create task_queue entry + `compute_feature_groups.delay(...)` → 202.
- group list: SQL over `feature_groups` with `ILIKE` on `normalized_token` for `search`; member counts precomputed.
- members: join `feature_group_members` → `features` (name, category, label_source, star_color, is_favorite, stats) — **never** store label copies.
- `by-token`: `match=exact|normalized|prefix` → `=` raw / `=` normalized / `LIKE 'tok%'` on the index.
- `related`: index lookup (shared rank≤3 tokens) + context Jaccard on `context_tokens` + blend of `AnalysisService.calculate_correlations` (already cached per feature).
Register in `router.py`: `api_router.include_router(feature_groups.router, tags=["feature-groups"])` (no prefix — paths carry `/extractions/...` and `/features/...` like the features router).

#### `backend/src/api/v1/endpoints/mcp_approvals.py`
Router `prefix="/mcp/approvals"`. `POST ""` (create pending, emit `approval:created`), `GET ""`/`GET /{id}`, `POST /{id}/approve` → build httpx-free internal call: submit stored payload via the **steering service layer is forbidden for the MCP server, but this is the backend itself** — call the same code path the steering endpoints use (import the async submit helper used by `/steering/async/*`), persist `steering_task_id`, emit `approval:resolved`. `POST /{id}/deny` sets status + reason. Optional expiry: a Beat-scheduled sweep marks `pending` older than 24 h as `expired` (reuse the celery beat schedule dict).

#### `backend/src/api/v1/endpoints/features.py` (modify PATCH)
```python
PROTECTED_FIELDS = {"name", "category", "description"}
if feature.star_color == "aqua" and (set(update.model_fields_set) & PROTECTED_FIELDS) \
        and not update.override_protected:
    raise HTTPException(409, detail={"code": "PROTECTED_LABEL",
        "message": "Feature has a protected (aqua) enhanced label.",
        "hint": "Pass override_protected=true to modify."})
if update.label_source and update.label_source not in ("user", "mcp_agent"):
    raise HTTPException(422, ...)
```

### 2.4 MCP Server Package

#### `backend/src/mcp_server/config.py`
```python
from pydantic_settings import BaseSettings

class MCPSettings(BaseSettings):
    model_config = {"env_prefix": "MCP_"}
    auth_token: str = ""
    allow_anonymous: bool = False
    port: int = 8765
    tool_categories: str = "read,groups,steering,labeling,experiments,jobs"
    steering_max_concurrent: int = 2
    steering_max_new_tokens: int = 512
    steering_approval: bool = False
    api_url: str = "http://backend:8000"     # env MISTUDIO_API_URL via alias
```

#### `backend/src/mcp_server/client.py`
```python
class MiStudioClient:
    """Thin async REST client. The ONLY way tools reach the backend (BR-1.3)."""
    def __init__(self, base_url: str):
        self._http = httpx.AsyncClient(base_url=f"{base_url}/api/v1", timeout=60)

    async def request(self, method: str, path: str, **kw) -> dict:
        r = await self._http.request(method, path, **kw)
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type","").startswith("application/json") else r.text
            raise ToolError(f"backend {r.status_code}: {detail}")   # passes 503/circuit-breaker text through (BR-6.2)
        return r.json() if r.content else {}
```

#### `backend/src/mcp_server/server.py`
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mistudio", host="0.0.0.0", port=settings.port)

def enabled(category: str) -> bool:
    return category in {c.strip() for c in settings.tool_categories.split(",")}

# startup guard
if not settings.auth_token and not settings.allow_anonymous:
    raise SystemExit("MCP_AUTH_TOKEN is required (or set MCP_ALLOW_ANONYMOUS=true for stdio dev)")

# Bearer middleware on the streamable-HTTP app: hmac.compare_digest(header_token, settings.auth_token)
# /health route on the same app for container healthchecks.
# Conditionally import+register tool modules per category, then mcp.run(transport=...).
```
Tool registration idiom (per module):
```python
def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    @mcp.tool()
    async def get_feature(feature_id: str) -> dict:
        """Full feature detail: label, category, statistics, star state."""
        return await client.request("GET", f"/features/{feature_id}")
```

#### `backend/src/mcp_server/tools/steering.py` (guardrails + approval)
```python
_inflight: set[str] = set()   # steering task ids not yet resolved

@mcp.tool()
async def steer_compare(feature_ids: list[str], strengths: list[float], prompt: str,
                        max_new_tokens: int = 256) -> dict:
    """Run a steering comparison. GPU-heavy: requires steering mode; strength in [-300, 300].
    Returns task_id (poll get_steering_result) or approval_request_id in approval mode."""
    _validate_strengths(strengths)                       # [-300, 300] → ToolError
    max_new_tokens = min(max_new_tokens, settings.steering_max_new_tokens)
    if len(_inflight) >= settings.steering_max_concurrent:
        raise ToolError(f"guardrail: {settings.steering_max_concurrent} steering tasks already in flight")
    body = {...}
    if settings.steering_approval:
        req = await client.request("POST", "/mcp/approvals",
                                   json={"tool_name": "steer_compare", "payload": body})
        return {"approval_request_id": req["id"], "status": "pending_approval",
                "hint": "poll get_approval_status; an operator must approve in the UI"}
    task = await client.request("POST", "/steering/async/compare", json=body)
    _inflight.add(task["task_id"])
    return task
```

### 2.5 Frontend

#### `frontend/src/stores/featureGroupsStore.ts`
Zustand store: `indexStatus`, `groups`, `filters`, `expandedGroupId`, `members`, `selection: Set<string>`, `isWebSocketConnected` + polling fallback (copy the systemMonitorStore fallback shape). Actions: `fetchStatus`, `computeIndex(force)`, `fetchGroups`, `fetchMembers(groupId)`, `toggleSelect`, `handoffToSteering()` (writes into `steeringStore` the way the existing pre-populate path does).

#### `frontend/src/hooks/useFeatureGroupsWebSocket.ts`
Subscribe `extractions/{id}/feature-groups`; map `feature_groups:progress|completed|failed` → store. Mirror `useTokenizationWebSocket`.

#### Components
`panels/FeatureGroupsPanel.tsx` orchestrates; `featureGroups/ComputeIndexBanner.tsx` (three states), `GroupList.tsx`, `GroupMembersTable.tsx` (member click opens the existing Feature Detail modal by id), `RelatedFeaturesDrawer.tsx`. Approvals: `steering/ApprovalsBanner.tsx` + store slice + `mcp/approvals` subscription. Tailwind slate/emerald per Mock UI standards.

---

## 3. Common Pitfalls

### Pitfall 1: enum value in the same transaction
`ALTER TYPE ... ADD VALUE` cannot be used in the transaction that also inserts the value. Keep revision B isolated; never merge with data migrations.

### Pitfall 2: model not registered in `models/__init__.py`
Alembic autogenerate and the test conftest's `create_all` both rely on it. (Bit us with NeuronpediaPushJob.)

### Pitfall 3: scanning `feature_activations`
It is range-partitioned by `feature_id`. Always filter by explicit feature-id batches; a cross-partition scan over 50k features will blow BR-6.1.

### Pitfall 4: copying labels into group tables
Labels/stars change after the index is built. Join live from `features`; only immutable data (tokens, snippets, similarity) is persisted.

### Pitfall 5: token comparison timing
Auth check must be `hmac.compare_digest(provided, expected)` — not `==` (existing convention from `main.py`).

### Pitfall 6: unbounded tool output
An agent asking for 50k features must get `{total: 50000, returned: 100, hint: "use filters/offset"}` — never the raw list. Cap every list tool at 100.

### Pitfall 7: TfidfVectorizer re-tokenizing
Context bags are already token lists. Use `analyzer=lambda bag: bag` (and `lowercase=False`) or sklearn will string-join and re-split them.

---

## 4. Testing Strategy

- **Unit (backend):** `test_token_normalization.py`; `test_feature_grouping_service.py` (synthetic features → expected buckets/subgroups/cohesion); aqua-guard + whitelist tests in the features endpoint suite; approvals state machine.
- **Unit (MCP):** auth middleware (401/ok/anonymous-refusal), category gating (disabled category absent from `tools/list`), guardrail clamps, error passthrough (mock 503 detail).
- **Integration:** seeded extraction → compute → status → groups → members → by-token → related; approval flow create→approve→steering submitted; MCP tools against live backend (docker-compose profile in CI is optional — mark `pytestmark` skip when backend unavailable, matching existing conditional-skip convention).
- **Frontend:** store logic (Vitest), ComputeIndexBanner states, hand-off writes to steeringStore.
- Test DB: add `mcp_agent` to conftest enum creation; new tables come from `Base.metadata.create_all` once models are registered.

---

## 5. Performance Tips

- Bulk-insert index rows (`bulk_save_objects`, 1k chunks); one commit per chunk keeps memory flat and progress honest.
- Compute TF-IDF per bucket, not globally — buckets are tiny; a global matrix would be 50k×vocab for nothing.
- `GET /feature-groups` should read only `feature_groups` (counts/cohesion precomputed); members loaded per expanded group.
- httpx client: single AsyncClient instance for the MCP process (connection pooling to backend).

---

*Related: [PRD](../prds/010_FPRD|MCP_Server.md) | [TDD](../tdds/010_FTDD|MCP_Server.md) | [Tasks](../tasks/010_FTASKS|MCP_Server.md)*
