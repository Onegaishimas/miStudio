"""Feature 20 task 4.0 — REACHABILITY ASSURANCE.

The rule this file enforces, stated once:

    A capability is not shipped until a test FAILS when its wiring is removed.

Not "a test exists". Not "the symbol is present". A test that asserts a
mechanism EXISTS passes forever after the mechanism stops being called — which
is exactly how, across this increment, five separate mechanisms shipped that
nothing invoked:

  * `CircuitClaimRegistry.reconcile()` — written, unit-tested, zero callers
  * `ContentionDialog` / `ClaimsStrip` — exported, tested, rendered by no page
  * `all_incumbents` — added to the payload, read by nothing
  * `attached_layers()` — orphaned by the fix that replaced its only consumer
  * `TestRingPruningIsWired` — asserted an entry point existed while nothing
    called it (the name this rule cites by convention)

Three shapes are needed, because each catches what the others cannot:

  1. REGISTRY — the category is in the maps that make it selectable.
  2. BUILT SERVER — the REAL `build_server()` registers it. A hand-called
     `register()` bypasses the gating that actually failed in the audit.
  3. CALLER — each tool issues its documented method and path. Catches a tool
     that registers but calls nothing, or calls the wrong endpoint.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.mcp_server.config import DEFAULT_CATEGORIES, VALID_CATEGORIES
from src.mcp_server.tools import MILLM_CATEGORY_MODULES


# ── Shape 1: registry ──────────────────────────────────────────────────────


class TestRegistryReachability:
    """RCH-3, shape 1. A module imported but not registered fails SILENTLY —
    the import succeeds, the tools are never exposed, and nothing errors."""

    def test_the_category_is_in_the_module_registry(self):
        assert "millm_circuits" in MILLM_CATEGORY_MODULES, (
            "the module is imported but not registered — its tools are "
            "unreachable and nothing reports it"
        )

    def test_the_category_is_selectable(self):
        assert "millm_circuits" in VALID_CATEGORIES, (
            "the category cannot be enabled, so registering it is moot"
        )

    def test_it_is_NOT_a_default(self):
        """No `millm_*` category is ever default: they are functional only with
        MILLM_API_URL set, and a default that cannot work is worse than one an
        operator opted into."""
        assert "millm_circuits" not in DEFAULT_CATEGORIES

    def test_every_registered_module_actually_exposes_register(self):
        for category, modules in MILLM_CATEGORY_MODULES.items():
            for module in modules:
                assert hasattr(module, "register"), (
                    f"{category} lists {module.__name__}, which has no "
                    "register() — the server would skip it silently"
                )


# ── Shape 2: built server ──────────────────────────────────────────────────


class TestBuiltServerReachability:
    """RCH-3, shape 2. The REAL `build_server()`, not a hand-called
    `register()`.

    This distinction is the finding: the audit's circuit-category defect was
    that registration existed and the SERVER never picked it up. A test that
    calls `register()` directly proves the module works and says nothing about
    whether anything calls it.
    """

    def _build(self, categories: str, monkeypatch):
        """The REAL build path, with its REAL signature.

        `build_server(settings, stdio=)` returns `(mcp, client)`; calling it
        wrongly is itself the kind of drift this file exists to catch, so the
        helper stays thin rather than wrapping it in a mock.
        """
        from src.mcp_server.config import MCPSettings
        from src.mcp_server.server import build_server

        # NOT MCP_-prefixed: `millm_api_url` reads the bare env var.
        monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
        settings = MCPSettings(
            tool_categories=categories,
            allow_anonymous=True,
        )
        mcp, _client = build_server(settings, stdio=True)
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_build_server_exposes_the_circuit_tools(self, monkeypatch):
        names = self._build("millm_circuits", monkeypatch)

        assert "millm_circuit_status" in names, (
            "build_server() did not expose the circuit tools even with the "
            "category enabled — registration exists but nothing reaches it"
        )
        # A representative from each family, so dropping one group is caught.
        for expected in (
            "millm_activate_circuit",
            "millm_import_circuit",
            "millm_export_circuit",
            "millm_circuit_claims",
            "millm_release_circuit_claims",
            "millm_circuit_sensing_events",
        ):
            assert expected in names, f"{expected} is not reachable"

    def test_the_category_is_absent_when_not_enabled(self, monkeypatch):
        """Specificity: if the tools appeared regardless, the test above would
        prove nothing about the wiring."""
        names = self._build("read", monkeypatch)
        assert "millm_circuit_status" not in names


# ── Shape 3: caller ────────────────────────────────────────────────────────


class RecordingClient:
    """Records the method and path each tool actually issues."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    async def _record(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"success": True, "data": {}}

    async def get(self, path, **params):
        return await self._record("GET", path, **params)

    async def post(self, path, json_body=None, **params):
        return await self._record("POST", path, json_body=json_body, **params)

    async def put(self, path, json_body):
        # F20 R1-07: `json_body` is REQUIRED on the real client. Giving it a
        # default here meant a tool calling `put(path)` with no body passed
        # every test and TypeError'd in production — the recorder was more
        # forgiving than the thing it stands in for, which is the one property
        # a stand-in must never have.
        return await self._record("PUT", path, json_body=json_body)

    async def delete(self, path, **params):
        return await self._record("DELETE", path, **params)

    async def raw_get(self, path):
        # R1-08: the real `raw_get` returns the UNENVELOPED body — that is the
        # whole reason the export uses it. Returning an envelope here would let
        # a tool that wrongly unwraps `.data` pass.
        self.calls.append(("RAW_GET", path, {}))
        return {"kind": "mistudio.circuit-definition", "schema_version": "1"}


def _open_gate():
    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    return gate


def _tools_by_name():
    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import millm_circuits

    mcp = FastMCP("t")
    client = RecordingClient()
    millm_circuits.register(mcp, client, _open_gate())
    manager = mcp._tool_manager
    return manager, client


#: Each tool's DOCUMENTED call: `(method, path, tool_kwargs, expected_payload)`.
#:
#: F20 R1-04: `expected_payload` was added because the assertion below checked
#: only method and path and DISCARDED the recorded kwargs. Three mutations
#: survived 26/26 green:
#:
#:   * dropping `limit=`/`offset=` from the list call
#:   * changing the intensity body key to `{"lambda": …}` — which 422s on
#:     EVERY call in production, because the endpoint requires `intensity`
#:   * disabling the manual gate check in `millm_import_circuit`
#:
#: A harness whose stated rule is "a test FAILS when the wiring is removed"
#: was verifying the address and not the letter.
EXPECTED_CALLS = {
    "millm_circuit_status": ("GET", "/api/circuits/active", {}, {}),
    "millm_list_circuits": ("GET", "/api/circuits", {}, {"limit": 50, "offset": 0}),
    "millm_circuit_claims": ("GET", "/api/circuits/claims", {}, {}),
    "millm_activate_circuit": (
        "POST", "/api/circuits/c1/activate", {"circuit_id": "c1"},
        {"json_body": None, "acknowledge_unvalidated": "false",
         "allow_layer_overlap": "false"},
    ),
    "millm_deactivate_circuit": (
        "POST", "/api/circuits/c1/deactivate", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_set_circuit_intensity": (
        "PUT", "/api/circuits/active/intensity", {"intensity": 1.0},
        # The body KEY is the whole contract here: the endpoint requires
        # `intensity`, so any other key 422s on every call.
        {"json_body": {"intensity": 1.0, "acknowledge_unvalidated": False}},
    ),
    "millm_delete_circuit": (
        "DELETE", "/api/circuits/c1", {"circuit_id": "c1"}, {},
    ),
    "millm_import_circuit": (
        "POST", "/api/circuits/import", {"definition": {"kind": "x"}},
        {"json_body": {"kind": "x"}, "on_conflict": None},
    ),
    "millm_export_circuit": (
        "RAW_GET", "/api/circuits/c1/export", {"circuit_id": "c1"}, {},
    ),
    "millm_release_circuit_claims": (
        "POST", "/api/circuits/claims/release", {"circuit_id": "c1"},
        {"json_body": None, "circuit_id": "c1"},
    ),
    "millm_circuit_sensing_status": (
        "GET", "/api/circuit-sensing/status", {}, {},
    ),
    "millm_circuit_sensing_events": (
        "GET", "/api/circuit-sensing/events", {},
        {"circuit_id": None, "limit": 50, "since": None},
    ),
    "millm_circuit_sensing_event": (
        "GET", "/api/circuit-sensing/events/7", {"event_id": 7}, {},
    ),
    "millm_circuit_sensing_enable": (
        "POST", "/api/circuit-sensing/c1/enable", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_circuit_sensing_disable": (
        "POST", "/api/circuit-sensing/c1/disable", {"circuit_id": "c1"},
        {"json_body": None},
    ),
    "millm_circuit_sensing_clear": (
        # R1-16: scope is REQUIRED — an unscoped call is refused, not treated
        # as "delete everything".
        "DELETE", "/api/circuit-sensing/events", {"circuit_id": "c1"},
        {"circuit_id": "c1"},
    ),
}


class TestCallerReachability:
    """RCH-4, shape 3. Each tool ISSUES its documented call.

    Registration proves a tool is exposed. This proves it does something, and
    the right something: re-pointing a path or deleting the call body fails
    here and nowhere else.
    """

    @pytest.mark.parametrize("tool_name", sorted(EXPECTED_CALLS))
    def test_the_tool_issues_its_documented_call(self, tool_name):
        manager, client = _tools_by_name()
        method, path, kwargs, payload = EXPECTED_CALLS[tool_name]

        asyncio.run(manager.call_tool(tool_name, kwargs))

        assert client.calls, f"{tool_name} registered but issued NO call"
        assert len(client.calls) == 1, (
            f"{tool_name} issued {len(client.calls)} calls; asserting on the "
            "last would let a wrong first call through"
        )
        actual_method, actual_path, actual_payload = client.calls[0]
        assert (actual_method, actual_path) == (method, path), (
            f"{tool_name} called {actual_method} {actual_path}, documented as "
            f"{method} {path}"
        )
        # R1-04: the PAYLOAD, not just the address. Verifying where a tool
        # points and not what it sends is how a permanently-422ing dial ships
        # with a green suite.
        assert actual_payload == payload, (
            f"{tool_name} sent {actual_payload}, documented as {payload}"
        )

    def test_every_registered_tool_is_covered_here(self):
        """A tool added without an entry above would otherwise be exposed with
        no caller assertion at all — the gap this file exists to close."""
        manager, _ = _tools_by_name()
        registered = {t.name for t in manager.list_tools()}
        missing = registered - set(EXPECTED_CALLS)
        assert not missing, (
            f"tools with no caller assertion: {sorted(missing)} — add them to "
            "EXPECTED_CALLS or they ship unverified"
        )

    def test_export_returns_the_document_UNWRAPPED(self):
        """R1-08. The export IS the portable artifact. A tool that unwrapped
        `.data` would return None for a raw response — and the previous
        assertion checked only the method string, so it could not tell."""
        manager, client = _tools_by_name()
        result = asyncio.run(
            manager.call_tool("millm_export_circuit", {"circuit_id": "c1"})
        )
        assert "mistudio.circuit-definition" in str(result), (
            "the export did not come back as the raw document — unwrapping "
            "`.data` on a raw response yields None"
        )

    def test_export_uses_the_RAW_path(self):
        """The export IS the portable artifact. Re-wrapping it in the envelope
        changes what a round-trip produces, so it must not go through the
        enveloped `get`."""
        manager, client = _tools_by_name()
        asyncio.run(manager.call_tool("millm_export_circuit", {"circuit_id": "c1"}))
        # R2-10: `calls[0]`, not `[-1]`. R1-04 rejected last-call assertions
        # everywhere else and this one was left behind — a tool issuing a wrong
        # first call and a right second would have passed.
        assert len(client.calls) == 1
        assert client.calls[0][0] == "RAW_GET"


# ── Gate degradation ───────────────────────────────────────────────────────


class TestGateDegradation:
    """EC-20.1/20.3. A closed gate returns a STRUCTURED unavailable, never an
    exception and never an unregistration: an agent must be able to tell
    "miLLM is down" from "this tool does not exist"."""

    def test_a_closed_gate_returns_structured_unavailable(self):
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        client = RecordingClient()
        millm_circuits.register(mcp, client, gate)

        result = asyncio.run(mcp._tool_manager.call_tool("millm_circuit_status", {}))
        payload = result[0] if isinstance(result, tuple) else result
        text = str(payload)
        assert "unavailable" in text and "millm" in text
        assert not client.calls, "a closed gate still issued the HTTP call"

    def test_the_UNGATED_tool_still_checks_the_gate(self):
        """F20 R1-06. `millm_import_circuit` omits `@gated` deliberately (so
        argument validation runs first) and checks the gate BY HAND — and
        nothing tested that hand-rolled check.

        A mutation disabling it SURVIVED the whole suite: the tool would have
        issued its HTTP call with miLLM known-down, turning a clean structured
        "unavailable" into a connection error the agent cannot classify.

        `test_argument_validation_runs_BEFORE_the_gate` below only exercises
        the BAD-argument path, which returns before reaching the gate at all —
        so the one tool that hand-rolls the gate was the one tool whose gate
        was unverified. That is this feature's own named anti-pattern, in the
        feature that exists to enforce against it.
        """
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        client = RecordingClient()
        millm_circuits.register(mcp, client, gate)

        # VALID arguments, so validation passes and the gate is what must stop
        # it.
        result = asyncio.run(
            mcp._tool_manager.call_tool(
                "millm_import_circuit", {"definition": {"kind": "x"}}
            )
        )
        text = str(result)
        assert "unavailable" in text and "millm" in text, (
            "the hand-rolled gate did not refuse; the agent gets a connection "
            "error instead of a structured unavailable"
        )
        assert not client.calls, (
            "the tool issued its HTTP call with the gate CLOSED — miLLM is "
            "known-down and it called anyway"
        )

    def test_argument_validation_runs_BEFORE_the_gate(self):
        """An agent debugging its own payload must not be told "millm is
        down". A gate failure is about the SERVER; a bad argument is about the
        CALL, and conflating them sends the agent to fix the wrong thing."""
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        millm_circuits.register(mcp, RecordingClient(), gate)

        result = asyncio.run(
            mcp._tool_manager.call_tool(
                "millm_import_circuit",
                {"definition": {"kind": "x"}, "on_conflict": "nonsense"},
            )
        )
        text = str(result)
        assert "on_conflict" in text, (
            "a bad argument was reported as miLLM being unavailable"
        )


class TestDestructiveScopeIsExplicit:
    """F20 R1-16. `millm_circuit_sensing_clear` defaulted to GLOBAL scope.

    An agent asked to "clear the events for this circuit" that omitted the
    argument wiped every observation in the deployment, irreversibly. The
    destructive path was the quiet one.
    """

    def test_an_unscoped_call_is_REFUSED(self):
        manager, client = _tools_by_name()
        result = asyncio.run(manager.call_tool("millm_circuit_sensing_clear", {}))
        assert "exactly one scope" in str(result)
        assert not client.calls, (
            "an unscoped clear reached the server — every recorded observation "
            "in the deployment would be gone"
        )

    def test_BOTH_scopes_at_once_is_refused(self):
        manager, client = _tools_by_name()
        result = asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear",
                {"circuit_id": "c1", "all_circuits": True},
            )
        )
        assert "exactly one scope" in str(result)
        assert not client.calls

    def test_explicit_all_circuits_IS_allowed(self):
        """The capability still exists — it just cannot be reached by
        omission."""
        manager, client = _tools_by_name()
        asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear", {"all_circuits": True}
            )
        )
        assert client.calls, "the deliberate global clear was blocked too"
        assert client.calls[0][0] == "DELETE"

    def test_the_DESTRUCTIVE_branch_sends_no_circuit_id(self):
        """F20 R2-05: `EXPECTED_CALLS` pins only the scoped call, so the branch
        that deletes EVERY observation in the deployment was asserted by
        nothing.

        Global scope is expressed to the server by OMITTING `circuit_id`. A
        future edit that passed it through would silently narrow the delete —
        or, re-pointed the other way, widen a scoped one. R1-16's fix was
        half-pinned: the refusal was tested, the destructive success path was
        not."""
        manager, client = _tools_by_name()
        asyncio.run(
            manager.call_tool(
                "millm_circuit_sensing_clear", {"all_circuits": True}
            )
        )
        method, path, payload = client.calls[0]
        assert (method, path) == ("DELETE", "/api/circuit-sensing/events")
        assert payload == {"circuit_id": None}, (
            f"the global clear sent {payload} — global scope is expressed by "
            "OMITTING circuit_id, so anything else changes what is deleted"
        )

    def test_scope_validation_runs_BEFORE_the_gate(self):
        """A scope mistake is about the CALL. Reporting it as "millm is down"
        sends the agent to fix the wrong thing."""
        from mcp.server.fastmcp import FastMCP

        from src.mcp_server.tools import millm_circuits

        gate = MagicMock()
        gate.check = AsyncMock(return_value=(False, "connection refused"))
        mcp = FastMCP("t")
        millm_circuits.register(mcp, RecordingClient(), gate)

        client = RecordingClient()
        mcp2 = FastMCP("t2")
        millm_circuits.register(mcp2, client, gate)
        result = asyncio.run(
            mcp2._tool_manager.call_tool("millm_circuit_sensing_clear", {})
        )
        assert "exactly one scope" in str(result), (
            "a scope error was reported as miLLM being unavailable"
        )
        # R2-09: the import tool's ordering test asserts this and the
        # DESTRUCTIVE tool's did not — so a mutation returning the scope error
        # AFTER issuing the DELETE would have passed. The tool that can delete
        # everything got the weaker of the two tests.
        assert not client.calls, (
            "the scope error was returned but the DELETE went out first"
        )


class TestTheDescriptionsCarryWhatAnAgentNeeds:
    """F20 R1-17..20. The descriptions ARE the product — an agent reads them
    and relays them to a human.

    Review found the three cardinal semantics each reaching exactly ONE tool,
    while an agent typically calls one tool and sees only that third. And
    several tools stated a problem without its next step, which is how an agent
    ends up polling or guessing.

    These assert on the REGISTERED descriptions, not the source, because that
    is what `list_tools()` actually hands an agent.
    """

    def _descriptions(self):
        manager, _ = _tools_by_name()
        return {t.name: (t.description or "") for t in manager.list_tools()}

    def test_the_rung_GATE_is_stated_where_rungs_are_listed(self):
        """R1-17: an agent must know rung < 2 is refused BEFORE it tries to
        activate, and `millm_list_circuits` is where it sees rungs."""
        d = self._descriptions()["millm_list_circuits"]
        assert "GATED at 2" in d or "gated at 2" in d.lower()
        assert "never moves a rung" in d.lower() or "never raises" in d.lower(), (
            "the observation/validation boundary reaches only the sensing "
            "tools, and this is the tool an agent calls first"
        )

    def test_NO_ACTIVE_CIRCUIT_is_explained_where_it_is_hit(self):
        """R1-18: it is a 200 body, not an error, and the remedy is to
        activate something. An agent told neither will treat it as a fault."""
        d = self._descriptions()["millm_set_circuit_intensity"]
        assert "NO_ACTIVE_CIRCUIT" in d
        assert "AMBIGUOUS_ACTIVE_CIRCUIT" in d, (
            "R2-03 added a refusal for several serving circuits and the tool "
            "never mentioned it"
        )
        # R2-06: these surface as a TOOL ERROR, because the client raises on
        # any `success:false`. Describing them as bodies told the agent to
        # expect a normal result.
        assert "TOOL ERROR" in d

    def test_the_slice_dial_names_what_DOES_govern_intensity(self):
        """Telling an agent the write is inert without naming the alternative
        leaves it retrying the same call."""
        d = self._descriptions()["millm_set_circuit_intensity"]
        assert "cluster" in d.lower() and "0.5 floor" in d

    def test_enable_points_at_the_diagnostic(self):
        """R1-19: 'enabled but no events' has two causes with different
        remedies, and the tool that distinguishes them was never named."""
        d = self._descriptions()["millm_circuit_sensing_enable"]
        assert "millm_circuit_sensing_status" in d
        assert "Do not poll" in d

    def test_unsensable_edges_names_its_actionable_fields(self):
        d = self._descriptions()["millm_circuit_sensing_status"]
        assert "reason" in d and "detail" in d, (
            "the field is described as a list of names, so an agent relays a "
            "count and drops the half that tells the human what to do"
        )

    def test_status_warns_that_a_slice_is_not_the_whole_circuit(self):
        """R1-20: reporting a slice serve with the circuit's own rung claims
        evidence for something that is not being served."""
        d = self._descriptions()["millm_circuit_status"]
        assert "slice_fallback" in d
        assert "does NOT describe" in d or "not describe" in d.lower()

    def test_activate_covers_ALL_THREE_refusal_branches(self):
        """The tool an agent reaches for first must resolve each case, or it
        loops: overridable, collision (never retry), stuck claim (release)."""
        d = self._descriptions()["millm_activate_circuit"]
        assert "overridable" in d
        assert "NEVER retry" in d
        assert "millm_release_circuit_claims" in d
        assert "one model, one fixture" in d, (
            "the measured hazard's caveat must travel with it, or an agent "
            "relays a single-fixture result as a general law"
        )


class TestTheThreeFailureShapesAreDistinguishable:
    """F20 R2-07. `{"error": …}` (your call), `{"unavailable": …}` (server
    down) and a coded tool error (operation refused) need three DIFFERENT
    agent responses — and the vocabulary was documented in no tool description,
    only in a module docstring that `list_tools()` does not transport."""

    def test_the_first_tool_an_agent_meets_explains_them(self):
        manager, _ = _tools_by_name()
        d = {t.name: (t.description or "") for t in manager.list_tools()}
        status = d["millm_circuit_status"]
        assert '{"error"' in status and "unavailable" in status
        assert "REFUSED" in status
