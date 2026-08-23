# P06 — mutation control log

**Phase:** P06 MCP server · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M20 | `mcp_server/tools/__init__.py` | Remove `millm_circuits` from `MILLM_CATEGORY_MODULES` — unregister all 16 tools, the increment's signature defect | ✅ | **KILLED** — 3 failures in `test_reachability.py` |
| M21 | `mcp_server/tools/circuits.py` | Re-point `get_circuit` from `/circuits/{id}` to `/WRONG-PATH/{id}` | ✅ | **KILLED — but not by the reachability harness** → MIS-E2E-119 |

**0 of 2 survived**, and the *way* M21 was killed is the finding.

## M20 — the harness prevents exactly the defect it was written for

Unregistering the 16 `millm_circuit_*` tools failed
`test_build_server_exposes_the_circuit_tools`,
`test_the_tool_map_names_only_REAL_tools` and
`test_the_ungated_destructive_tools_are_indexed`. Shapes 1 and 2 — registry and
built-server — are parametrized off `MILLM_CATEGORY_MODULES` rather than a hand-list,
so they cover all 116 tools and they bite. This is the best-guarded surface in the
codebase and the mutation confirms it.

## M21 — the path assertion covers 16 of 116 tools

`TestCallerReachability`'s docstring says:

> *"Registration proves a tool is exposed. This proves it does something, and the
> right something: **re-pointing a path or deleting the call body fails here and
> nowhere else**."*

Re-pointing `get_circuit`'s path left **all 31 reachability tests green**. It was
caught by `test_mcp_contract_generated.py` instead — incidentally, because the
generated contract records the path, so changing it makes the doc differ from the
committed copy.

The cause: the parametrization source, `EXPECTED_CALLS`, is a **hand-written dict of
16 entries** — the one part of an otherwise registry-derived harness that is
maintained by hand. Shapes 1 and 2 derive from the registry and cover everything;
shape 3, the only one that proves a tool does the *right* thing, covers 14%.

The incidental backstop is also weaker than it looks: per MIS-E2E-114 the committed
contract already carries three bogus rows (`GET kind`, `GET manifests`, `GET status`)
that the same diff test pins as correct.

## Equivalent mutants

None.
