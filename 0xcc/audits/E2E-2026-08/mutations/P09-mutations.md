# P09 — mutation control log

**Phase:** P09 Realtime · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M26 | `workers/websocket_emitter.py:128` | `except httpx.TimeoutException` → `except httpx.TransportError` (i.e. apply the MIS-E2E-137 fix) | ✅ | **SURVIVED** → MIS-E2E-142 |
| M27 | `workers/websocket_emitter.py:110` | Drop `X-Internal-Token` from the emit POST — the recorded P0 regression | ✅ | **KILLED** — `test_emit_progress_success` |

**1 of 2 survived.**

## M27 — a recorded P0's fix is genuinely pinned

The qa_engineer persona records: *"P0 — System-metrics WS emission 403s silently:
`BackgroundMonitor._emit_to_channel` omits `X-Internal-Token`; regression from the
internal-token hardening. **No integration test covers this path.**"* One does now, and
it bites: removing the header fails `test_emit_progress_success`. A gap that was
recorded as open has since been closed.

## M26 — the retry behaviour is not pinned in either direction

This mutation is unusual: it applies the **fix** for MIS-E2E-137 rather than breaking
something. Broadening the retry from `TimeoutException` to `TransportError` — which
changes whether a `ConnectError` or `RemoteProtocolError` is retried three times or
abandoned immediately — left the suite green.

So no test asserts *which* failures are retried. The narrow catch is not a deliberate,
pinned decision; it is unobserved behaviour. That matters because the events carrying
`retries=3` are the ones the code treats as must-not-lose (`steering:completed`,
`neuronpedia:push_completed`, `enhanced_labeling:completed`), and the exception a
stale pooled connection actually raises after a backend restart —
`RemoteProtocolError` — is outside the catch.

Mutating toward a fix is worth doing when a finding claims behaviour is wrong: if the
corrected behaviour also passes, neither version is pinned, and the fix will not stay
fixed.

## Equivalent mutants

None.
