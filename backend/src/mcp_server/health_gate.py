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

# product -> health path relative to that product's base URL
_HEALTH_PATHS = {"millm": "/api/health", "mistudio": "/api/v1/system/health"}


class HealthGate:
    """TTL-cached availability probes, one entry per product."""

    def __init__(self, millm_url: str = "", ttl_s: float = 10.0,
                 mistudio_url: str = "") -> None:
        self._urls = {"millm": millm_url.rstrip("/"),
                      "mistudio": mistudio_url.rstrip("/")}
        self._ttl = ttl_s
        # product -> (checked_at_monotonic, available, reason)
        self._cache: dict[str, tuple[float, bool, str]] = {}
        # One long-lived client: a fresh AsyncClient per probe cost a TCP
        # setup every TTL window (009 R1).
        self._http = httpx.AsyncClient(timeout=PROBE_TIMEOUT_S,
                                       follow_redirects=False)

    async def aclose(self) -> None:
        await self._http.aclose()

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
        path = _HEALTH_PATHS.get(product)
        if path is None:
            return False, f"unknown product '{product}'"
        base = self._urls.get(product, "")
        if not base:
            return False, ("MILLM_API_URL is not configured"
                           if product == "millm"
                           else f"{product} URL is not configured")
        url = f"{base}{path}"
        try:
            response = await self._http.get(url)
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e} ({url})"
        # Strictly 2xx: a 3xx from an ingress fronting a dead backend must
        # not count as available (009 R1; redirects are not followed).
        if not 200 <= response.status_code < 300:
            return False, f"HTTP {response.status_code} from {url}"
        # Contract: available <=> 2xx AND status != 'unhealthy'. degraded IS
        # available (miLLM with no model loaded still accepts imports).
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("status") == "unhealthy":
                return False, f"status 'unhealthy' from {url}"
        except ValueError:
            pass  # non-JSON body on 2xx — treat as available
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
