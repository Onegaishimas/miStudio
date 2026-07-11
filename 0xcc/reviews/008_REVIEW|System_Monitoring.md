# Feature Review: 008 — System Monitoring

**Reviewed:** 2026-07-11 (building on this session's deep Celery/Monitor review, [review_celery_monitor_operations_2026-07-10](../../.claude/context/sessions/review_celery_monitor_operations_2026-07-10.md))
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v1.0](../prds/008_FPRD|System_Monitoring.md) (2025-12-05), FTDD, FTID
**Verdict:** The **most operationally dangerous doc-drift in the app**: the FPRD describes a Celery-Beat monitoring architecture that **was deleted this session**. The implementation itself is now correct (this session fixed the P0 403 outage), but the documentation actively describes a retired design, and there are **zero tests** — which is exactly why the P0 shipped undetected.

---

## 1. Scope

Fully reviewed this session in the Celery/Monitor operations review (F1 = the system-metrics 403 outage, F8/F11–F14 = monitor-page issues). Here I add the **doc↔code accuracy** lens and confirm the current state.
**Backend:** `services/system_monitor_service.py` (405), `services/gpu_monitor_service.py` (484), `services/background_monitor.py` (asyncio monitor — the real collector), `api/v1/endpoints/system.py` (route inventory), `workers/websocket_emitter.py` (system emit funcs). **`workers/system_monitor_tasks.py` — DELETED this session.**
**Frontend:** `SystemMonitor.tsx`, `useSystemMonitorWebSocket.ts`, `systemMonitorStore.ts` (all reviewed this session).
**Tests present:** **NONE** (`test_monitor*`, `test_system*`, `test_gpu*` all absent).

---

## 2. Doc ↔ Code Accuracy

**This FPRD is the most inaccurate about *architecture* (not just reference tables) in the review — it describes a design that no longer exists.**

### 2.1 Monitoring architecture — §3.1 describes a deleted mechanism

PRD §3.1 shows the canonical flow: **`Celery Beat (every 2s) → collect_system_metrics task → WebSocket emit`**. Reality (as of this session):
- **There is no Celery Beat monitoring task.** `system_monitor_tasks.py` was **deleted** this session. The beat entry was already commented out in `celery_app.py` before that.
- Metrics are collected by **`background_monitor.py`** — an **asyncio background task started in the FastAPI app lifespan** (`BackgroundMonitor`), running in-process, not via Celery. This exists specifically because long-running Celery tasks were blocking the beat monitoring task.
- So the entire §3.1 diagram is wrong: no beat, no Celery task, different process model.

### 2.2 Key files — 1 deleted, others accurate

| PRD (§7) | Reality |
|---|---|
| `workers/system_monitor_tasks.py` (Celery Beat task) | **DELETED this session.** Replaced by `services/background_monitor.py` (asyncio). |
| `services/system_monitor_service.py`, `workers/websocket_emitter.py`, `SystemMonitor.tsx`, `useSystemMonitorWebSocket.ts`, `systemMonitorStore.ts` | Accurate ✅ (though `gpu_monitor_service.py` — the pynvml collector — is omitted from the list). |

### 2.3 API endpoints — §5 wrong

| PRD (§5) | Reality |
|---|---|
| `GET /system/metrics` | Actual: `GET /system/metrics` exists ✅ but returns a different shape. |
| `GET /system/history` | **Does not exist.** The 1-hour history is a **frontend-only rolling buffer** (`useHistoricalData`, `maxDataPoints`), never persisted or served. |
| `GET /system/gpu` | Actual granular routes: `/gpu-list`, `/gpu-metrics`, `/gpu-metrics/all`, `/gpu-info`, `/gpu-processes`. |
| (not in PRD) | `/health`, `/restart`, `/disk-usage`, `/network-rates`, `/disk-rates`, `/all`, `/resource-estimate`. |

### 2.4 WebSocket channels + payload — mostly right, but payload lacked `metric_type`

The `system/gpu/{id}`, `system/cpu`, `system/memory`, `system/disk`, `system/network` channels at 2s are accurate. **But** the PRD payload doesn't include `metric_type` — which this session **added** to `background_monitor.py` because the frontend was sniffing payload shape to route metrics (fragile; Celery-review F12). Doc should reflect the `metric_type` field now present.

### 2.5 Config — §9 partially fictional

`system_monitor_interval_seconds` (default 2) exists. `system_monitor_history_hours` (§9) — **no such setting**; history is the frontend buffer, not a backend retention config.

**Recommendation:** FPRD 008 needs the **most substantive rewrite** of any feature — not just reference tables but the core §3.1 architecture section, because it documents a deleted Celery-Beat design. This was already flagged in the PPRD §3.8/§5.1 divergence note earlier this session; 008 is the source.

---

## 3. Findings (severity-ranked)

*(The P0 — system-metrics 403 outage — was found AND fixed this session. See the Celery review F1. It is resolved in production. The findings below are the residual + doc issues.)*

### P1 — Verification gap that enabled the P0

**F1. Zero test coverage — directly responsible for the P0 shipping.**
The system-metrics WebSocket emission silently 403'd for ~2 months (since the internal-token hardening) with **no test** exercising the `BackgroundMonitor → /api/internal/ws/emit` path. A single integration test asserting a 200 from that emit (with the token wired) would have caught it immediately. Given monitoring is the app's observability backbone, this is the highest-priority gap. **Fix:** add (a) a BackgroundMonitor emit test (asserts token + 200), (b) a frontend staleness-fallback test (WS connected but silent → polling resumes — the compounding frontend bug from Celery-review F1).

### P2 — Residual from Celery review (monitor-page)

**F2. Frontend polling fallback keys on connection, not data freshness** (Celery-review F1 frontend half) — a connected-but-silent socket freezes the page. This session added a staleness watchdog fix; verify it's deployed and tested (ties to F1).

**F3. Payload-shape sniffing in the WS hook** (Celery-review F12) — `handleMetrics` distinguishes metric types by field presence. This session added `metric_type` to payloads; the frontend should switch to keying on it explicitly rather than sniffing. Half-done.

### P3 — Quality / tech-debt

**F4. `system.py` has `/health` and `/restart`** — a `POST /restart` endpoint in a monitoring router is a surprising, powerful operation (restarts services?). Verify it's authz-gated and intentional; document it (it's undocumented in the FPRD).
**F5. The deleted `system_monitor_tasks.py` left dangling references** — this session removed it and its celery_app route; confirm no stale imports remain (the merge/tests passed, so likely clean).
**F6. `gpu_monitor_service.py` (484) + `system_monitor_service.py` (405)** — reasonable split; `gpu_monitor_service` is omitted from the FPRD key-files list.

---

## 4. Test Coverage Notes

Zero coverage — and uniquely consequential here because monitoring *is* the verification layer for everything else. The P0 (2-month silent outage) is the concrete cost. Minimum viable tests: (1) BackgroundMonitor emit returns 200 with token, (2) each metric collector returns the expected schema, (3) frontend staleness fallback resumes polling. This feature should arguably be the **first** to get tests, since it guards observability of all the GPU-bound features.

---

## 5. Summary — if you fix three things

1. **Rewrite FPRD 008 §3.1 + §7** — it documents a deleted Celery-Beat architecture; reality is the asyncio `BackgroundMonitor`. Most important doc fix in the app (and the root of the PPRD §3.8/§5.1 drift).
2. **F1** — add monitoring tests (the absence directly caused the 2-month P0). Highest-value test investment in the app.
3. **F3** — finish the payload-shape → `metric_type` migration on the frontend.

**The good:** post-this-session, the implementation is correct — metrics emit with the token and `metric_type`, per-GPU/CPU/mem/disk/net channels work, pynvml/psutil collectors are clean, and the fallback has a staleness watchdog. The problem is purely (a) the documentation describes a retired design and (b) nothing tests it. Both are now well-understood.
