"""
HealthGate — per-product availability cache for the unified MCP server
(Feature 9).

Tools stay registered through outages (MCP clients cache tool lists;
unregistering churns agents — contract §3): when a product is down, gated
tools return a structured {"unavailable": <product>, "reason": …} result
instead. `degraded` (any 2xx) is AVAILABLE — miLLM with no model loaded must
still accept imports and report status.
"""

import time
from typing import Callable, Optional

import httpx

PROBE_TIMEOUT_S = 3.0


class HealthGate:
    """TTL-cached availability probes, one entry per product."""

    def __init__(self, millm_url: str = "", ttl_s: float = 10.0) -> None:
        self._millm_url = millm_url.rstrip("/")
        self._ttl = ttl_s
        # product -> (checked_at_monotonic, available, reason)
        self._cache: dict[str, tuple[float, bool, str]] = {}

    async def check(self, product: str) -> tuple[bool, str]:
        """(available, reason). reason is agent-readable on refusal."""
        now = time.monotonic()
        hit = self._cache.get(product)
        if hit is not None and now - hit[0] < self._ttl:
            return hit[1], hit[2]
        available, reason = await self._probe(product)
        self._cache[product] = (now, available, reason)
        return available, reason

    def invalidate(self, product: Optional[str] = None) -> None:
        if product is None:
            self._cache.clear()
        else:
            self._cache.pop(product, None)

    async def _probe(self, product: str) -> tuple[bool, str]:
        if product != "millm":
            return False, f"unknown product '{product}'"
        if not self._millm_url:
            return False, "MILLM_API_URL is not configured"
        url = f"{self._millm_url}/api/health"
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT_S) as http:
                response = await http.get(url)
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e} ({url})"
        if response.status_code >= 400:
            return False, f"HTTP {response.status_code} from {url}"
        # Any 2xx — including a 'degraded' status body — is available.
        return True, "ok"


def gated(gate: HealthGate, product: str) -> Callable:
    """Decorator: run the gate before the tool body; on refusal return the
    uniform structured-unavailable result (never raise — pitfall 1/3)."""
    def wrap(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            ok, reason = await gate.check(product)
            if not ok:
                return {"unavailable": product, "reason": reason}
            return await fn(*args, **kwargs)

        return inner

    return wrap
