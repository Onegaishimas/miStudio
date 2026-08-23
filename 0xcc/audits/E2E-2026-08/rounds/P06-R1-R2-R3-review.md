# P06 — MCP server: all three rounds

**Phase:** P06 · **Date:** 2026-08-23
**Scope:** `backend/src/mcp_server/` (8 modules + 17 tool modules, 4,308 lines),
`docs/mcp-contract.md`, `backend/tests/unit/test_reachability.py`

Mutation log: `mutations/P06-mutations.md` (2 run, 0 survived — and how M21 was
killed is itself the finding).

## R1 — findings (7)

Every R1 finding was reproduced empirically by the reviewer (MockTransport,
Starlette TestClient, or a live `build_server` run) rather than read.

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-113 | P1 | The 200-HTML failure mode is fixed on `millm_client` and **not** on `client.py` — which its own comment says was never carried across |
| MIS-E2E-114 | P2 | The published contract lists three endpoints that do not exist, and the diff test **pins them as correct** |
| MIS-E2E-115 | P2 | Calling `mistudio_howto` mutates `os.environ` and permanently changes what unauthenticated `/health` advertises |
| MIS-E2E-116 | P2 | `httpx.InvalidURL` escapes the gate's handler, so a typo'd URL makes 32 "never raise" tools raise, with no negative caching |
| MIS-E2E-117 | P2 | A non-ASCII bearer token returns **500** instead of 401 on a LAN-reachable port |
| MIS-E2E-118 | P3 | A cancelled probe logs an ERROR traceback on every graceful shutdown |
| MIS-E2E-119 | P1 | *(R2)* The payload/path assertion is a hand-list covering **16 of 116** tools |

## The two most surprising, verified independently

**MIS-E2E-114.** The bogus rows are in the **committed** file:
`get_steering_samples` lists `GET kind` and `GET manifests`; `get_steering_result`
lists `GET status`. They are `dict.get("kind")` / `.get("manifests")` / `.get("status")`
calls that the AST scraper recorded as HTTP endpoints. `contract.py:68` already has
the `startswith("/")` filter that excludes them; the sibling branch at `:78` does not.
This file is one of only two the `sync-to-clean` filter **deliberately preserves** in
the public mirror.

**MIS-E2E-115.** `howto.py:589` really does `os.environ.setdefault("MILLM_API_URL",
"http://millm.invalid")` so it can enumerate the millm categories, and `server.py:163`
re-reads that variable per request. A documentation tool has a permanent global side
effect on the endpoint whose only job is to report what is true.

## R2 — the harness is strong, and its one hand-maintained part is the gap

**M20 KILLED, hard.** Unregistering the 16 `millm_circuit_*` tools — the increment's
signature defect, the thing this harness exists for — failed three tests. Shapes 1
and 2 are parametrized off the registry, cover all 116 tools, and bite.

**M21 killed, but not by the harness.** Re-pointing `get_circuit`'s path left all 31
reachability tests green; the contract regeneration diff caught it incidentally. The
class docstring claims re-pointing a path *"fails here and nowhere else"* — true for
the 16 tools in the hand-written `EXPECTED_CALLS`, false for the other 100
(MIS-E2E-119).

Worth stating plainly: this is still the best-guarded surface in the codebase. The
finding is that its coverage is 100% on registration and 14% on behaviour, and the
docstring reads as though both were the same.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 113, 114, 115, 116, 117, 118, 119 |
| **PLAUSIBLE** | none |
| **REFUTED** | none |

Unusually, everything is CONFIRMED — the reviewer reproduced each R1 finding by
execution, and 114/115 were re-verified independently here against the committed
contract and the source of `howto.py`.

**Live verification.** The MCP server this session talks to is the k8s deployment
(`http://mcp-mistudio.hitsai.local/mcp`). `mistudio_howto(topic='tools')` was called
against it and returned `tool_count: 116` across 14 categories, confirming
MIS-E2E-017 and giving every count in this phase a live denominator.

## Verified clean (from R1, for a later round to attack)

- The k8s `MCP_TOOL_CATEGORIES` manifests match `DEFAULT_CATEGORIES`, and the
  reachability guard's regex genuinely matches both quoted forms — it does **not**
  fail open.
- `@mcp.tool()` / `@gated()` decorator ordering is correct in the millm modules, and
  `functools.wraps` preserves the signature for `func_metadata`.
- `wrap_tool_with_audit` mutating `tool.fn` does take effect (`ToolManager.list_tools()`
  returns the live objects) and kwargs reach the audit digest.
- `/api/v1/system/health` exists, as `HealthGate` expects.

## Phase closed

**7 findings** (MIS-E2E-113 … 119), no P0. Mutations: 2 run, 0 survived.
Tree verified clean.

**The one sentence for the synthesis:** the best-guarded surface in this codebase is
guarded 100% on *whether* a tool exists and 14% on *what it does*, and the difference
is that the first is derived from the registry and the second is a hand-written list.
