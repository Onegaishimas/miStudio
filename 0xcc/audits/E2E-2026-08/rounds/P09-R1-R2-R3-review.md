# P09 — Realtime (WebSocket end to end): all three rounds

**Phase:** P09 · **Date:** 2026-08-23
**Scope:** `backend/src/workers/websocket_emitter.py` (1,614 lines),
`backend/src/core/websocket.py`, `backend/src/services/background_monitor.py`,
`backend/src/main.py` socket handlers, the 15 frontend `use*WebSocket*` hooks,
`manual/docs/reference/websocket-channels.md`

Mutation log: `mutations/P09-mutations.md` (2 run, 1 survived).

**Tree note:** this phase is the first to review the post-fix tree. The four
Feature Detail modal fixes (MIS-E2E-132…135) were merged to `main` and deployed
before this phase ran. None of them touches the realtime layer.

## R1 — findings (7)

Every finding was verified empirically by the reviewer — a minimal uvicorn repro
for the deadlock, the httpx exception hierarchy for the retry gap, a faithful
mirror of the monitor's start/stop, and live `python-socketio` 5.16.2 behaviour
for the handler overwrite.

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-136 | P1 | Sync `emit_progress` POSTs to the API's **own** event loop from `async def` handlers — reproduced: `ReadTimeout` at 5.01 s, event dropped, whole API frozen |
| MIS-E2E-137 | P1 | The retry catches only `TimeoutException`; `RemoteProtocolError` — what a stale pooled connection raises after a restart — gets **zero** retries |
| MIS-E2E-138 | P2 | Duplicate `@sio.event` registrations silently overwrite; the `subscribed`/`unsubscribed` acks never fire and the exported `ws_manager` stays empty |
| MIS-E2E-139 | P2 | A setup crash leaves `_running=True`, so the monitor refuses to restart and every `system/*` channel goes permanently silent, unlogged |
| MIS-E2E-140 | P2 | `subscribe` accepts any type and any quantity — 50,000 channels created unauthenticated in test |
| MIS-E2E-141 | P3 | `emit_system_metrics` emits `"metrics"` not `"system:metrics"` and returns `True` while delivering nothing; unlocked, never-closed HTTP client |
| MIS-E2E-142 | P2 | *(R2)* Nothing pins which emit failures are retried |

### The deadlock, and the fix that was not generalized

`emit_progress` is synchronous and POSTs to `/api/internal/ws/emit` — the backend's
own endpoint. Called from an `async def` handler under a single worker, the coroutine
blocks the only loop that could answer it. `datasets.py` calls it twice in sequence,
so ~10 s of frozen API.

`training_service.py:328` already wraps the same call in `asyncio.to_thread`, under a
comment reading *"Run emit_deletion_progress in thread pool to avoid blocking async
loop."* Counted across the three exposed files: **13 sync `emit_*` calls, zero
`to_thread`.** Fourth instance of this pattern in the audit (MIS-E2E-064, 072, 092).

The architect persona already recorded the right remedy — *"BackgroundMonitor runs
inside FastAPI yet POSTs to its own `/api/internal/ws/emit` … should call
`ws_manager.emit_event()` directly"* — and MIS-E2E-138 shows why that is not a
drop-in today: the exported `ws_manager` is the overwritten one and is always empty.
The two findings have to be fixed together.

## R2 — mutations

**M27 KILLED.** Dropping `X-Internal-Token` from the emit POST fails
`test_emit_progress_success`. The qa_engineer persona recorded this exact regression
as a P0 with *"no integration test covers this path"* — one does now, and it bites. A
recorded gap that has since been closed.

**M26 SURVIVED**, and it is an unusual mutation: it applied the **fix** for
MIS-E2E-137 rather than breaking something. Broadening the retry to `TransportError`
changed the suite not at all, so neither the current narrow behaviour nor its
correction is observed (MIS-E2E-142). Where a finding claims behaviour is wrong,
checking whether the *corrected* behaviour is also unobserved tells you whether the
fix would stay fixed.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 136, 137, 138, 139, 140, 141, 142 |
| **PLAUSIBLE** | none |
| **REFUTED** | none |

All CONFIRMED: the reviewer reproduced each by execution, and the retry gap and the
undefended-call count were re-verified here independently.

## What this phase inherits

Three realtime findings were raised in earlier phases and belong to the same picture:

- **MIS-E2E-105 (P0)** — Socket.IO accepts any origin, `connect` authenticates
  nothing, `subscribe` joins any string. MIS-E2E-140 is the validation half of the
  same handler.
- **MIS-E2E-120 (P1)** — every event fires N+1 times after N reconnects, because the
  `connect` handler re-attaches handlers socket.io never detached.
- **MIS-E2E-067 (P2)** — a failure emit uses `"error"` where the frontend reads
  `"error_message"`, so a failed extraction shows no reason.

Together with this phase's seven, the realtime layer has **ten** open findings and
one P0. The transport has no origin control, no channel validation, duplicated
handlers, an unpinned retry policy, a self-deadlocking emit path, and a frontend that
multiplies every event it receives.

## Not covered

The channel↔event cross-check against `manual/docs/reference/websocket-channels.md`
is doc-conformance and is carried into P11, which owns the documentation chain.
Live socket verification (connecting from a foreign origin to demonstrate
MIS-E2E-105) is carried into P12, which owns live journeys.

## Phase closed

**7 findings** (MIS-E2E-136 … 142). Mutations: 2 run, 1 survived. Tree verified clean.

**The one sentence for the synthesis:** the realtime layer's emit path calls its own
API over HTTP from inside the loop that must answer it, retries the one failure mode
that does not occur, and hands its events to a frontend that has registered each
listener once per reconnect.
