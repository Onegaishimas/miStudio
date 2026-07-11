# Feature PRD: System Monitoring

**Document ID:** 008_FPRD|System_Monitoring
**Version:** 1.1 (BackgroundMonitor architecture — doc refresh)
**Last Updated:** 2026-07-11
**Status:** Implemented
**Priority:** P1 (Important Feature)

---

## 1. Overview

### 1.1 Purpose
Provide real-time system resource monitoring during long-running operations like training and extraction.

### 1.2 User Problem
Researchers need visibility into system resources because:
- Training can consume significant GPU memory
- Memory leaks can cause job failures
- Resource bottlenecks affect training speed
- Temperature monitoring prevents thermal throttling

### 1.3 Solution
A comprehensive system monitoring dashboard with real-time GPU, CPU, memory, disk, and network metrics streamed via WebSocket.

---

## 2. Functional Requirements

### 2.1 GPU Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | GPU utilization percentage | Implemented |
| FR-1.2 | GPU memory used/total | Implemented |
| FR-1.3 | GPU temperature | Implemented |
| FR-1.4 | GPU power draw | Implemented |
| FR-1.5 | Per-GPU metrics (multi-GPU) | Implemented |

### 2.2 CPU Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | Overall CPU utilization | Implemented |
| FR-2.2 | Per-core utilization | Implemented |
| FR-2.3 | CPU frequency | Planned |

### 2.3 Memory Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | RAM used/total | Implemented |
| FR-3.2 | Swap used/total | Implemented |
| FR-3.3 | Memory pressure indicator | Planned |

### 2.4 I/O Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Disk read/write rates (MB/s) | Implemented |
| FR-4.2 | Network upload/download rates (MB/s) | Implemented |

### 2.5 Visualization
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-5.1 | Real-time updating charts | Implemented |
| FR-5.2 | 1-hour rolling history | Implemented |
| FR-5.3 | Combined utilization + temperature chart | Implemented |
| FR-5.4 | Grid layout for multiple metrics | Implemented |

---

## 3. Monitoring Architecture

### 3.1 Data Collection

**Note (2026-07):** monitoring runs via `BackgroundMonitor` — an **asyncio task
inside the FastAPI process** (`services/background_monitor.py`), started in the
app lifespan. The earlier Celery-Beat `system_monitor_tasks.py` collector was
dead code and has been **deleted**. `BackgroundMonitor` emits directly through
the internal WebSocket endpoint (with the `X-Internal-Token` header).

```
FastAPI app lifespan → BackgroundMonitor (asyncio, every ~2s)
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│  GPU Metrics     │     │  System Metrics  │
│  (pynvml)        │     │  (psutil)        │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └────────────┬───────────┘
                      ▼
         POST /api/internal/ws/emit  (X-Internal-Token)
                      │
    ┌─────────────┬───┴─────────┬─────────────┐
    ▼             ▼             ▼             ▼
system/gpu/{id} system/cpu  system/memory  system/disk, system/network
     (event: system:metrics, payload includes metric_type)
```

### 3.2 Fallback Pattern
- Primary: WebSocket streaming
- Fallback: HTTP polling (on disconnect)
- Auto-switch on reconnection

---

## 4. User Interface

### 4.1 System Monitor Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ System Monitor                                              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ GPU Utilization & Temp  │ │ Memory Usage                │ │
│ │ ┌─────────────────────┐ │ │ ┌─────────────────────────┐ │ │
│ │ │     ▄▄▄▄████████    │ │ │ │ RAM: ████████░░ 80%     │ │ │
│ │ │   ▄█████████████    │ │ │ │ 12.8 GB / 16 GB         │ │ │
│ │ │ ▄████████████████   │ │ │ │                         │ │ │
│ │ │ Util: 75% | Temp: 72°│ │ │ │ Swap: ███░░░░░░ 30%     │ │ │
│ │ └─────────────────────┘ │ │ │ 2.4 GB / 8 GB           │ │ │
│ └─────────────────────────┘ │ └─────────────────────────┘ │ │
│                             └─────────────────────────────┘ │
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ CPU Utilization         │ │ I/O Rates                   │ │
│ │ ┌─────────────────────┐ │ │ ┌─────────────────────────┐ │ │
│ │ │ Core 0: ████████░░  │ │ │ │ Disk R: 150 MB/s        │ │ │
│ │ │ Core 1: █████░░░░░  │ │ │ │ Disk W: 45 MB/s         │ │ │
│ │ │ Core 2: ██████████  │ │ │ │ Net ↓: 2.3 MB/s         │ │ │
│ │ │ Core 3: ███░░░░░░░  │ │ │ │ Net ↑: 0.5 MB/s         │ │ │
│ │ └─────────────────────┘ │ │ └─────────────────────────┘ │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 GPU Memory Widget (Sidebar)
```
┌──────────────────┐
│ GPU Memory       │
│ ██████████░░░░░░ │
│ 6.2 / 8.0 GB     │
└──────────────────┘
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/system/metrics` | GET | Current combined system metrics |
| `/api/v1/system/all` | GET | All monitoring data (GPU + system) in one call |
| `/api/v1/system/gpu-list` | GET | Enumerate available GPUs |
| `/api/v1/system/gpu-metrics` | GET | Metrics for the selected GPU |
| `/api/v1/system/gpu-metrics/all` | GET | Per-GPU metrics (all devices) |
| `/api/v1/system/gpu-info` | GET | GPU device info |
| `/api/v1/system/gpu-processes` | GET | Processes using the GPU |
| `/api/v1/system/disk-usage` | GET | Disk usage per mount |
| `/api/v1/system/disk-rates` | GET | Disk I/O rates |
| `/api/v1/system/network-rates` | GET | Network I/O rates |
| `/api/v1/system/resource-estimate` | POST | Estimate resource needs |
| `/api/v1/system/health` | GET | Service health |
| `/api/v1/system/restart` | POST | Restart services (admin) |

*(There is no `/system/history` endpoint — the 1-hour history is a frontend
rolling buffer, not a backend query. The multi-GPU routes here are shared with
Feature 009.)*

---

## 6. WebSocket Channels

All system channels emit the **`system:metrics`** event (namespaced). Payloads
carry a `metric_type` field (`gpu` / `cpu` / `memory` / `disk` / `network`).

| Channel | Event | Payload | Interval |
|---------|-------|---------|----------|
| `system/gpu/{id}` | `system:metrics` | `{utilization, memory, temperature, power, metric_type:"gpu", timestamp}` | ~2s |
| `system/cpu` | `system:metrics` | `{percent, count, metric_type:"cpu", timestamp}` | ~2s |
| `system/memory` | `system:metrics` | `{ram, swap, metric_type:"memory", timestamp}` | ~2s |
| `system/disk` | `system:metrics` | `{read_bytes, write_bytes, metric_type:"disk", timestamp}` | ~2s |
| `system/network` | `system:metrics` | `{sent_bytes, recv_bytes, metric_type:"network", timestamp}` | ~2s |

Frontend falls back to HTTP polling when the socket disconnects **or when no
`system:metrics` event arrives for ~10s while connected** (data-staleness watchdog).

---

## 7. Key Files

### Backend
- `backend/src/services/background_monitor.py` - **Asyncio collector loop (current)** — emits every ~2s
- `backend/src/services/system_monitor_service.py` - CPU/RAM/disk/net collection (psutil)
- `backend/src/services/gpu_monitor_service.py` - Per-GPU collection (pynvml)
- `backend/src/workers/websocket_emitter.py` - WebSocket emission helpers
- `backend/src/api/v1/endpoints/system.py` - API routes
- *(removed: `workers/system_monitor_tasks.py` — the dead Celery-Beat collector)*

### Frontend
- `frontend/src/components/SystemMonitor/SystemMonitor.tsx` - Main dashboard
- `frontend/src/components/SystemMonitor/UtilizationChart.tsx` - GPU chart
- `frontend/src/hooks/useSystemMonitorWebSocket.ts` - WebSocket hook
- `frontend/src/stores/systemMonitorStore.ts` - Zustand store

---

## 8. Metrics Collection

### 8.1 GPU Metrics (pynvml)
```python
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
temperature = pynvml.nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
```

### 8.2 System Metrics (psutil)
```python
import psutil

cpu_percent = psutil.cpu_percent(percpu=True)
memory = psutil.virtual_memory()
swap = psutil.swap_memory()
disk_io = psutil.disk_io_counters()
net_io = psutil.net_io_counters()
```

---

## 9. Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `system_monitor_interval_seconds` | Metrics collection interval | 2 |
| `system_monitor_history_hours` | History retention | 1 |

---

## 10. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Monitors during training |
| Feature Discovery | Monitors during extraction |

---

## 11. Testing Checklist

- [x] GPU metrics collection
- [x] CPU metrics per core
- [x] Memory metrics
- [x] Disk I/O rates
- [x] Network I/O rates
- [x] WebSocket streaming
- [x] HTTP polling fallback
- [x] Chart visualization
- [x] 1-hour history

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/008_FTDD|System_Monitoring.md) | [TID](../tids/008_FTID|System_Monitoring.md)*
