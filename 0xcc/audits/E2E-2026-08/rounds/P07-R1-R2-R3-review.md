# P07 — Frontend state layer: all three rounds

**Phase:** P07 · **Date:** 2026-08-23
**Scope:** `frontend/src/{stores,api,hooks,contexts,utils}/` — 20 stores (16,172
lines), 26 API modules, 21 hooks, 18 utils

Mutation log: `mutations/P07-mutations.md` (1 run, 1 survived).

## R1 — findings (7 register entries, 11 defects)

| Id | Sev | Claim |
|---|---|---|
| MIS-E2E-120 | P1 | **Every WebSocket event fires N+1 times after N reconnects** — the re-attach guards the wrong set |
| MIS-E2E-121 | P1 | A stale request nulls the newer one's abort controller — cancellation and the 5s timeout die for the session |
| MIS-E2E-122 | P1 | Rebalance reads a member's sign from an already-zeroed strength, flipping suppress to amplify |
| MIS-E2E-123 | P1 | In-flight batch state is **persisted**, so a mid-batch refresh disables Generate permanently |
| MIS-E2E-124 | P2 | Three concurrency defects: no double-submit guard on `generateCombined`; a structurally-dead recovery guard; no request sequencing on feature detail |
| MIS-E2E-125 | P2 | Polling dies on one transient error (and the caller discards the handle), and can render stale state after stopping |
| MIS-E2E-126 | P2 | An abandoned channel resubscribes on every reconnect; `duplicateFeature` drops `sae_id` |
| MIS-E2E-127 | P1 | *(R2)* The auto-baseline test file samples only where the slope cannot matter |

## MIS-E2E-014 resolved

The seeded finding — a second, parallel Socket.IO client — was filed as *"plausible;
import graph not yet traced"*. Traced: **no importers anywhere in `frontend/src`.**
`api/websocket.ts` is entirely dead. Downgraded **P2 → P3**: the risk was a second
connection with different transports opening if anything imported it, and nothing
does. Two latent defects inside it were noted and not filed, since nothing calls
them. It is a deletion task now, not a defect.

## MIS-E2E-120, verified at source

```js
// IMPORTANT: Re-attach existing handlers FIRST (for reconnections)
// This must happen before processing pending handlers to avoid double-registration
const existingHandlers = new Map(eventHandlersRef.current);
existingHandlers.forEach((handlers, event) => {
  handlers.forEach(handler => { socket.on(event, handler); });
});
```

socket.io-client does not detach handlers on disconnect (4.8.1's `onclose` clears
only acks), so these are still registered and `socket.on` adds a second. The comment
reasons explicitly about double-registration and has the direction backwards — it
protects the *pending* handlers while double-registering the *existing* ones. With
`reconnectionAttempts: Infinity`, reconnects are routine. `addCheckpoint` appends
with no dedupe, so the visible symptom is duplicated checkpoints; every other handler
behind this transport is multiplied identically. There is **no test file for this
context**.

## R2 — the mutation

**M22 SURVIVED.** `BASELINE_SLOPE` 2.6 → 2.4 left 75 tests green, including the
formula's own dedicated test file. The file samples at freq 0 (slope × 0), and at
freq 0.9/1.0 (clamped) — the slope is invisible at every point it looks, and the
arithmetic that would distinguish the two lives in a comment (MIS-E2E-127). This is
the J-Lens arc's `torch.eye` trap in the frontend.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 014 (resolved dead), 120, 127 |
| **PLAUSIBLE** | 121, 122, 123, 124, 125, 126 |
| **REFUTED** | none |

The PLAUSIBLE set are single-site reads of unambiguous code. Several (123 especially)
are reproducible in a browser and are carried into P08's live pass rather than
asserted here.

## Verified clean

`api/client.ts` is clean apart from two cosmetics the reviewer deliberately did not
file: a non-string `error.detail` stringifies as `[object Object]`, and a 200 with an
empty body throws a raw `SyntaxError` rather than an `APIError`.

## Phase closed

**8 findings** (MIS-E2E-120 … 127, plus MIS-E2E-014 resolved). Mutations: 1 run,
1 survived. Tree verified clean.

**The one sentence for the synthesis:** the state layer's defects cluster on
*sequencing* — a reconnect that re-registers, a stale response that clobbers a newer
one, a persisted flag that outlives its loop — in the one layer with no test file for
its central component.
