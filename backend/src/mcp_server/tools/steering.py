"""
Steering tools (category: steering) — async compare/sweep/combined + mode control.

GPU-heavy operations. Guardrails (Feature 010, BR-3.4):
- max_new_tokens clamped to MCP_STEERING_MAX_NEW_TOKENS
- at most MCP_STEERING_MAX_CONCURRENT unresolved agent steering tasks
- with MCP_STEERING_APPROVAL=true, requests become operator-approval records
  instead of running immediately (poll get_approval_status)
"""

from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ..client import MiStudioClient
from ..config import MCPSettings


class SteerFeature(BaseModel):
    """One feature to steer with."""

    feature_idx: int = Field(..., ge=0, description="Feature (neuron) index in the SAE")
    layer: int = Field(..., ge=0, description="Layer the SAE was trained on")
    strength: float = Field(
        ..., ge=-300, le=300,
        description="Raw steering coefficient, Neuronpedia scale. Start ±5–20; ±100+ often collapses output",
    )


# Unresolved agent-submitted steering task ids (concurrency guardrail)
_inflight: set[str] = set()


def register(mcp: FastMCP, client: MiStudioClient, settings: MCPSettings) -> None:
    def _generation_params(max_new_tokens: int, temperature: float, top_p: float) -> dict:
        return {
            "max_new_tokens": min(max_new_tokens, settings.steering_max_new_tokens),
            "temperature": temperature,
            "top_p": top_p,
        }

    def _check_concurrency() -> None:
        if len(_inflight) >= settings.steering_max_concurrent:
            raise RuntimeError(
                f"Guardrail: {settings.steering_max_concurrent} agent steering task(s) already "
                "in flight. Poll get_steering_result (which releases the slot) or cancel one."
            )

    async def _submit(tool_name: str, endpoint: str, body: dict) -> Any:
        if settings.steering_approval:
            approval = await client.post(
                "/mcp/approvals", json_body={"tool_name": tool_name, "payload": body}
            )
            return {
                "approval_request_id": approval["id"],
                "status": "pending_approval",
                "hint": "Operator approval required — poll get_approval_status; "
                "once approved it returns the steering task_id",
            }
        _check_concurrency()
        result = await client.post(endpoint, json_body=body)
        task_id = result.get("task_id")
        if task_id:
            _inflight.add(task_id)
        return result

    @mcp.tool()
    async def steering_status() -> Any:
        """Steering service health, circuit-breaker state, and reset path."""
        return await client.get("/steering/status")

    @mcp.tool()
    async def compute_cluster_allocation(
        sae_id: str,
        members: list[dict],
        group_cohesion: Optional[float] = None,
    ) -> Any:
        """Compute the principled starting strength allocation for steering a
        CLUSTER of features (Feature 013). members: [{feature_idx, layer,
        similarity?, activation_frequency?, sign?}] (1-20, all on the SAE's layer).
        Returns {B, B_dir, G, f_eff, weights, strengths, flags, constants_used}.
        Read-only — safe without steering mode.

        HONOR THE FLAGS CONTRACT before seeding steer_combined:
        - 'low_cohesion': the cluster fails the cohesion gate — the UI refuses
          these strengths and keeps per-feature solo baselines; you should too
          unless deliberately testing the gate.
        - 'cancellation': positive members partially cancel (worst pair in
          cancellation_pair) — the blended direction is unreliable.
        - 'approximate': decoder unavailable; constant-budget fallback (G=1).
        - 'default_budget': no activation frequencies known.
        - 'nonunit_decoder': decoder columns deviate from unit norm — the law
          extrapolates; prefer running the calibration protocol first."""
        body: dict = {"sae_id": sae_id, "members": members}
        if group_cohesion is not None:
            body["group_cohesion"] = group_cohesion
        return await client.post("/steering/cluster-allocation", json_body=body)

    @mcp.tool()
    async def get_steering_mode() -> Any:
        """Is steering mode active (model+SAE pre-loaded on the GPU)?"""
        return await client.get("/steering/mode")

    @mcp.tool()
    async def enter_steering_mode() -> Any:
        """Enter steering mode — LOADS THE MODEL+SAE ONTO THE GPU. This consumes
        significant VRAM and may contend with other jobs. Exit when done."""
        return await client.post("/steering/enter-mode")

    @mcp.tool()
    async def exit_steering_mode() -> Any:
        """Exit steering mode and free the GPU memory it held."""
        return await client.post("/steering/exit-mode")

    @mcp.tool()
    async def steer_compare(
        sae_id: str,
        prompt: str,
        features: list[SteerFeature],
        model_id: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Any:
        """Run a steering comparison: 1-4 features, each generating steered output
        side-by-side with an unsteered baseline. GPU-heavy background task —
        returns task_id; poll get_steering_result. Strengths in [-300, 300]."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "selected_features": [f.model_dump() for f in features],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
            "include_unsteered": True,
        }
        return await _submit("steer_compare", "/steering/async/compare", body)

    @mcp.tool()
    async def steer_sweep(
        sae_id: str,
        prompt: str,
        feature_idx: int,
        layer: int,
        strength_values: list[float],
        model_id: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Any:
        """Dose-response sweep: one feature generated at each strength value
        (e.g. [0, 5, 20, 50]). The definitive causal-validation tool. GPU-heavy
        background task — returns task_id; poll get_steering_result."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "feature_idx": feature_idx,
            "layer": layer,
            "strength_values": [max(-300.0, min(300.0, s)) for s in strength_values],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
        }
        return await _submit("steer_sweep", "/steering/async/sweep", body)

    @mcp.tool()
    async def steer_combined(
        sae_id: str,
        prompt: str,
        features: list[SteerFeature],
        model_id: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Any:
        """Apply ALL selected features simultaneously in one generation pass
        (synergy testing). GPU-heavy background task — returns task_id."""
        body = {
            "sae_id": sae_id,
            "model_id": model_id,
            "prompt": prompt,
            "selected_features": [f.model_dump() for f in features],
            "generation_params": _generation_params(max_new_tokens, temperature, top_p),
            "include_baseline": True,
        }
        return await _submit("steer_combined", "/steering/async/combined", body)

    @mcp.tool()
    async def get_steering_result(task_id: str) -> Any:
        """Poll an async steering task. Returns status and, when completed, the
        generated outputs and metrics."""
        result = await client.get(f"/steering/async/result/{task_id}")
        # status is nested and terminal states are success/failure/revoked
        # (matching the backend's status_map) — releasing on the wrong key
        # leaked concurrency slots until the guardrail wedged (2026-07-14).
        state = (result.get("status") or {}).get("status") if isinstance(result.get("status"), dict) else result.get("status")
        if state in ("success", "failure", "revoked"):
            _inflight.discard(task_id)
        return result

    @mcp.tool()
    async def cancel_steering_task(task_id: str) -> Any:
        """Cancel a running steering task and free its guardrail slot."""
        result = await client.delete(f"/steering/async/task/{task_id}")
        _inflight.discard(task_id)
        return result

    if settings.steering_approval:
        @mcp.tool()
        async def get_approval_status(approval_request_id: str) -> Any:
            """Poll an operator-approval request. status: pending | approved
            (includes steering_task_id to poll) | denied (includes reason)."""
            return await client.get(f"/mcp/approvals/{approval_request_id}")
