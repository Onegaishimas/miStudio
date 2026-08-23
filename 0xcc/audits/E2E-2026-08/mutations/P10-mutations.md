# P10 — mutation control log

**Phase:** P10 Infra & supply chain · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M28 | `k8s/base/mcp.yaml` | Strip `circuits` and all `millm_*` from the **deployed** `MCP_TOOL_CATEGORIES` — the historical production outage | ✅ | **KILLED** — 2 failures in `TestTheDEPLOYMENTEnablesWhatTheCodeRegisters` |

## M28 — the deployment-vs-code guard bites

This reproduces the increment's real outage: `MCP_TOOL_CATEGORIES` in the k8s
manifest silently overrode the code's `DEFAULT_CATEGORIES`, so 19 circuit tools and
16 `millm_circuit_*` tools were unreachable in production while being registered,
tested and documented. `test_every_DEFAULT_category_is_actually_deployed` and
`test_the_circuit_categories_are_deployed` both fail when the manifest is stripped.

Worth naming what makes this one good: it asserts across the **code/deployment
boundary**, which is where the outage lived. Most guards in this codebase check one
side. This is the only test found in the whole audit that reads a k8s manifest and
compares it to the registry.

## Not mutable here

MIS-E2E-143 — the mirror publishing its full history — has no test to break and no
test that could be written locally: the defect is in what a GitHub Actions
force-push does to a remote repository. It was verified directly against the live
public GitHub API instead, which is stronger evidence than a mutation would be.
