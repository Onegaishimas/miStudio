"""
MiLLMClient — thin async wrapper over miLLM's management API (Feature 9,
Unified MCP; contract: miLLM repo docs/mcp-contract.md).

The ONE place the miLLM ApiResponse envelope
({success, data, error:{code,message,details}}) is unwrapped — tools must
never see or leak the wrapper. Raw (non-envelope) endpoints — cluster export —
use raw_get.
"""

import json
from typing import Any, Optional

import httpx

from .client import BackendError


class MiLLMClient:
    """Async client for a miLLM deployment's /api management surface."""

    def __init__(self, base_url: str, timeout: float = 60.0,
                 bearer_token: str = "") -> None:
        headers = {}
        if bearer_token:
            # Forward-compatible: miLLM is unauthenticated today (contract §6);
            # a future additive bearer requirement slots in here.
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> Any:
        try:
            response = await self._http.request(
                method, path, params=params, json=json_body
            )
        except httpx.HTTPError as e:
            raise BackendError(0, f"miLLM backend unreachable: {e}")

        body: Any = None
        if response.content:
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = None

        # Envelope unwrap (contract §2). The envelope is authoritative even
        # on 4xx/5xx — its error.message is written to be agent-safe.
        if isinstance(body, dict) and "success" in body:
            if body.get("success"):
                return body.get("data")
            error = body.get("error") or {}
            raise BackendError(
                response.status_code,
                {"code": error.get("code", "MILLM_ERROR"),
                 "message": error.get("message", "miLLM request failed"),
                 "details": error.get("details")},
            )

        if response.status_code >= 400:
            raise BackendError(response.status_code,
                               body if body is not None else response.text)
        return body if body is not None else {}

    async def get(self, path: str, **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("GET", path, params=clean or None)

    async def post(self, path: str, json_body: Optional[dict[str, Any]] = None,
                   **params: Any) -> Any:
        clean = {k: v for k, v in params.items() if v is not None}
        return await self.request("POST", path, params=clean or None,
                                  json_body=json_body)

    async def put(self, path: str, json_body: dict[str, Any]) -> Any:
        return await self.request("PUT", path, json_body=json_body)

    async def raw_get(self, path: str) -> Any:
        """GET an endpoint that returns a RAW document (no envelope) — the
        cluster export IS the portable artifact (contract §2)."""
        try:
            response = await self._http.get(path)
        except httpx.HTTPError as e:
            raise BackendError(0, f"miLLM backend unreachable: {e}")
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = None
        if response.status_code >= 400:
            # Export failures still arrive as envelope JSON — surface the
            # structured {code, message, details} like every other path
            # (009 R1: agents got the envelope as an escaped string blob).
            if isinstance(body, dict) and body.get("error"):
                error = body["error"]
                raise BackendError(response.status_code,
                                   {"code": error.get("code", "MILLM_ERROR"),
                                    "message": error.get("message", ""),
                                    "details": error.get("details")})
            # Non-envelope JSON (e.g. FastAPI 422 detail) stays structured
            # rather than re-stringified (009 R2).
            raise BackendError(response.status_code,
                               body if body is not None else response.text)
        if body is None:
            # 2xx non-JSON (misrouted ingress, splash page) must be a
            # structured BackendError, not a bare JSONDecodeError (009 R2).
            raise BackendError(
                response.status_code,
                f"export returned non-JSON content from {path}")
        return body
