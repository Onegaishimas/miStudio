"""
miLLM circuit tools (category: millm_circuits) — Feature 20, MCP Circuit Surface.

Circuits are multi-layer graphs over several SAEs, carrying an EVIDENCE RUNG
that says how much is actually known about them. Every tool here transports
that language verbatim and never composes it.

Three agent-facing semantics these tools exist to keep straight — each is
stated in the tool descriptions because an agent reads those, not this file:

  * OBSERVATION IS NOT VALIDATION. Edge sensing records that an upstream
    feature fired and a downstream partner fired after it. That is a
    correlation on live traffic. It NEVER raises a rung: raising one requires
    a causal intervention, which happens in miStudio, not here.
  * ABSENCE OF ROWS IS NOT ABSENCE OF FIRING. An edge with no events may be
    unsensable (its SAE is not attached, its layer is dark). Read
    `unsensable_edges` before concluding a circuit is quiet.
  * `steering: null` MEANS NOT EVALUATED, never "not steering".

No hub tools. Circuit import is inline-only (EC-20.5): a circuit references
several SAEs by id, and importing one from a remote pack without checking those
references is how an agent ends up serving a circuit against the wrong feature
basis.
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from ..health_gate import HealthGate, gated
from ..millm_client import MiLLMClient


def _validate_since(since: Optional[str]) -> Optional[dict]:
    """Reject a naive `since`. Returns an error dict, or None if fine.

    Copied deliberately from `millm_sensing.py` rather than shared: a naive
    timestamp makes the server silently assume UTC and shift the polling
    window, so both surfaces must reject it identically.
    """
    if since is None:
        return None
    from datetime import datetime

    try:
        parsed = datetime.fromisoformat(since.replace("Z", "+00:00"))
    except ValueError:
        return {"error": f"`since` is not ISO-8601: {since!r}"}
    if parsed.tzinfo is None:
        return {
            "error": "`since` must carry a UTC offset (e.g. ...T12:00:00Z) — "
            "naive timestamps shift the polling window silently"
        }
    return None


def register(mcp: FastMCP, millm: MiLLMClient, gate: HealthGate) -> None:
    # ── Lifecycle ──────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_status() -> Any:
        """Which circuits are steering RIGHT NOW, as a list.

        Several circuits can serve at once when their layers are disjoint, so
        this is never a single object. Each row carries `steering` — the
        SERVER's verdict on whether that circuit is genuinely influencing
        generation. Do NOT derive it from `is_active`: an active row can be a
        slice-fallback, unparseable or unattached circuit that steers nothing.
        `steering: null` means NOT EVALUATED, never "not steering".

        While more than one circuit serves, the per-request dial and the
        `X-miLLM-Circuit-Rung` header are BOTH suppressed — no single circuit's
        dial or evidence describes the response. Do not substitute either
        circuit's rung."""
        return await millm.get("/api/circuits/active")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_list_circuits(limit: int = 50, offset: int = 0) -> Any:
        """Imported circuits with their evidence rung.

        `rung_language` is the server's phrase, rendered from the evidence
        ladder — transport it verbatim. Never re-derive a rung locally: a
        circuit's rung is the MINIMUM over its edges, computed server-side, and
        a local guess would overclaim on the first edge it read."""
        return await millm.get("/api/circuits", limit=limit, offset=offset)

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_claims() -> Any:
        """Which circuit holds which LAYER.

        The unit of contention is the layer, not the feature: steering sums
        into one per-layer vector, so two circuits on a layer interact even
        when their features differ. This view makes an activation refusal
        intelligible BEFORE it happens — two circuits can both be active while
        contending for nothing.

        `composed: true` marks a layer carrying more than one circuit. On those
        layers the rung header is omitted."""
        return await millm.get("/api/circuits/claims")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_activate_circuit(
        circuit_id: str,
        acknowledge_unvalidated: bool = False,
        allow_layer_overlap: bool = False,
    ) -> Any:
        """Serve a circuit.

        Two refusals arrive as `success:false` bodies, not errors:

        `UNVALIDATED_CIRCUIT` — the circuit's rung is below 2, so it is not
        causally validated. Re-send with `acknowledge_unvalidated=true` only
        on explicit human instruction; report the rung phrase first.

        `CIRCUIT_LAYER_CONTENTION` — another circuit holds one of these layers.
        Read `details.overridable`:
          * `true` — the layers overlap but the features differ. Composition is
            possible with `allow_layer_overlap=true`, but SURFACE
            `details.measured_hazard` to the human first, INCLUDING its `note`
            ("one model, one fixture — indicative, not exhaustive"). Do not
            present that measurement as a general law.
          * `false` — both circuits steer the SAME feature. NEVER retry;
            `details.override_param` is absent precisely so it cannot be
            guessed. Report `details.colliding_keys` and stop.

        If the refusal names a circuit that `millm_circuit_status` does not
        list, its claim is stuck: use `millm_release_circuit_claims`, NOT a
        retry and NOT the override."""
        return await millm.post(
            f"/api/circuits/{circuit_id}/activate",
            acknowledge_unvalidated=str(bool(acknowledge_unvalidated)).lower(),
            allow_layer_overlap=str(bool(allow_layer_overlap)).lower(),
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_deactivate_circuit(circuit_id: str) -> Any:
        """Stop serving a circuit and release its layer claims.

        `cleared_steering: false` with a warning means the steering could NOT
        be cleared — the model may still be steering. Report that rather than
        treating deactivation as done."""
        return await millm.post(f"/api/circuits/{circuit_id}/deactivate")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_set_circuit_intensity(intensity: float) -> Any:
        """Dial the active circuit's global lambda.

        Refused while SEVERAL circuits serve: no single circuit's dial
        describes the response. For a SLICE-FALLBACK circuit the steering
        belongs to a cluster profile, which keeps its own intensity — this dial
        is recorded but not applied, and the response says so."""
        return await millm.put(
            "/api/circuits/active/intensity", json_body={"intensity": intensity}
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_delete_circuit(circuit_id: str) -> Any:
        """Delete an imported circuit. Deactivates it first if it is serving."""
        return await millm.delete(f"/api/circuits/{circuit_id}")

    # ── Import / export ────────────────────────────────────────────────────

    @mcp.tool()
    # NO @gated here, deliberately (millm_clusters.py precedent).
    #
    # The decorator runs the gate BEFORE the body, so an in-body argument check
    # is unreachable while miLLM is down — and an agent debugging its own
    # payload gets told "millm is down", sending it to fix the wrong thing.
    # The gate is checked manually below, AFTER validation.
    async def millm_import_circuit(
        definition: dict, on_conflict: Optional[str] = None
    ) -> Any:
        """Import a `mistudio.circuit-definition/v1` document (INLINE only).

        No hub import: a circuit references several SAEs by id, and pulling one
        from a remote pack without checking those references is how an agent
        ends up serving a circuit against the wrong feature basis.

        An incompatible definition imports as UNBOUND and refuses only at
        activation, with a per-SAE compatibility matrix. `on_conflict`:
        'rename' (default) or 'fail'."""
        # Argument validation BEFORE the gate check: an agent debugging its own
        # payload must not be told "millm is down" (millm_clusters.py
        # precedent). A gate failure is about the SERVER; this is about the
        # CALL, and conflating them sends the agent to fix the wrong thing.
        if on_conflict not in (None, "rename", "fail"):
            return {"error": "`on_conflict` must be 'rename' or 'fail'"}
        if not isinstance(definition, dict):
            return {"error": "`definition` must be the circuit document object"}
        ok, reason = await gate.check("millm")
        if not ok:
            return {"unavailable": "millm", "reason": reason}
        return await millm.post(
            "/api/circuits/import", json_body=definition, on_conflict=on_conflict
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_export_circuit(circuit_id: str) -> Any:
        """The circuit's ORIGINAL document, byte-for-byte.

        Returned RAW — not in the `{success, data}` envelope — because the
        export is the portable artifact and re-wrapping it would change what a
        round-trip produces. Unwrapping `.data` on this response yields
        `None`."""
        return await millm.raw_get(f"/api/circuits/{circuit_id}/export")

    # ── Recovery ───────────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_release_circuit_claims(circuit_id: str) -> Any:
        """Release ONE circuit's stuck layer claims.

        Use when an activation is refused naming a circuit that
        `millm_circuit_status` does not list — its claim outlived its serving.
        Retrying will not help, and `allow_layer_overlap=true` would compose
        against a circuit that is not there.

        Scoped to a single circuit deliberately; there is no "release
        everything". Releasing a circuit that IS still serving leaves it
        steering layers it no longer holds — the response warns when that
        happens."""
        return await millm.post(
            "/api/circuits/claims/release", circuit_id=circuit_id
        )

    # ── Edge sensing ───────────────────────────────────────────────────────

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_status() -> Any:
        """Edge-sensing runtime state for the armed circuit.

        `unsensable_edges` names edges that CANNOT fire an observation — their
        SAE is not attached, or their layer is dark. Read it before concluding
        a circuit is quiet: absence of rows is not absence of firing.

        `requests_sensed == 0` means NO request reached sensing at all — a
        wiring or skip condition, not quiet traffic. `requests_truncated`
        outlives `truncated_layers`: the latter describes only the last drained
        request."""
        return await millm.get("/api/circuit-sensing/status")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_events(
        circuit_id: Optional[str] = None,
        limit: int = 50,
        since: Optional[str] = None,
    ) -> Any:
        """Observed edge firings, newest first.

        Each row is a CORRELATION on live traffic: an upstream feature fired
        and its downstream partner fired within the token lag. That is an
        OBSERVATION, not a validation — it NEVER raises the edge's rung. Only a
        causal intervention does, and that happens in miStudio.

        `edge_rung_language` is denormalised AS OF OBSERVATION: it describes
        the evidence that was true when the row was written, not today's rung.
        Do not re-read a current rung against an old event.

        `since` MUST carry a UTC offset — naive timestamps shift the window
        silently."""
        bad = _validate_since(since)
        if bad is not None:
            return bad
        return await millm.get(
            "/api/circuit-sensing/events",
            circuit_id=circuit_id,
            limit=limit,
            since=since,
        )

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_event(event_id: int) -> Any:
        """One edge observation in full: both endpoints with their layer,
        feature, position and activation, the observed token lag, and the
        context window.

        The rung language on this row is as-of-observation (see
        `millm_circuit_sensing_events`).

        `event_id` is an INTEGER. F20 R1-05: this advertised `str`, so the
        schema accepted "abc" and the agent got a FastAPI 422 from the route
        instead of a usable error from the tool — and the caller test pinned a
        call shape production would have rejected."""
        return await millm.get(f"/api/circuit-sensing/events/{event_id}")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_enable(circuit_id: str) -> Any:
        """Enable edge sensing for a circuit (persists; arms when it serves).

        Enabled is operator INTENT and is reported distinctly from `armed`: a
        circuit can be enabled but not armed because it is not active, or
        because its SAE set is not attached."""
        return await millm.post(f"/api/circuit-sensing/{circuit_id}/enable")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_disable(circuit_id: str) -> Any:
        """Disable edge sensing for a circuit. Recorded events are kept."""
        return await millm.post(f"/api/circuit-sensing/{circuit_id}/disable")

    @mcp.tool()
    @gated(gate, "millm")
    async def millm_circuit_sensing_clear(circuit_id: Optional[str] = None) -> Any:
        """Delete recorded edge observations.

        Scoped to one circuit when `circuit_id` is given, otherwise all. This
        removes evidence of what was observed; it does not change any rung,
        because observations never set one."""
        return await millm.delete(
            "/api/circuit-sensing/events", circuit_id=circuit_id
        )
