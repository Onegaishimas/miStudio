# Feature PRD: Multi-GPU Scalability

**Document ID:** 009_FPRD|Multi_GPU_Scalability
**Version:** 1.2 (doc refresh — accuracy corrections)
**Last Updated:** 2026-07-11
**Status:** Partially Complete — Phase 1 (monitoring) + Phase 2 (extraction GPU routing) done; Phase 3 (DDP) + training GPU selection planned
**Priority:** P2

> **Corrections 2026-07-11** — FR-1.3 overstated: **training GPU selection is NOT
> implemented** (only *extraction* has `gpu_id` routing). §5 (endpoints), §6 (data
> model), and §8 (config) describe **planned/design-intent** schema that does not
> exist yet. See the **Doc-Refresh Corrections** appendix at the end.

---

## 1. Overview

### 1.1 Purpose
Enable distributed SAE training across multiple GPUs and provide enhanced monitoring with aggregated vs. per-GPU views.

### 1.2 User Problem
Researchers with multi-GPU systems cannot fully utilize their hardware:
- ~~Training runs on single GPU only~~ → single-GPU training, but jobs routable to any GPU ✅
- ~~No visibility into per-GPU resource usage~~ → per-GPU monitoring implemented ✅
- Cannot leverage data parallelism for faster training *(DDP — still planned)*
- Memory constraints on single GPU limit model/SAE size *(DDP — still planned)*

### 1.3 Solution
Multi-GPU support with distributed training, configurable GPU selection, and enhanced monitoring views.

**Implementation status as of Dec 2025:** GPU monitoring infrastructure (Phase 1) and per-GPU job routing (Phase 2) are complete. Distributed DDP training (Phase 3) remains planned.

---

## 2. Functional Requirements

### 2.1 Distributed Training
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-1.1 | Data parallel training across GPUs | Planned |
| FR-1.2 | Gradient synchronization | Planned |
| FR-1.3 | GPU selection for training jobs | ✅ Complete (Dec 2025) |
| FR-1.4 | Automatic batch size scaling | Planned |
| FR-1.5 | Mixed precision per GPU | Planned |

### 2.2 GPU Selection
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-2.1 | List available GPUs with specs | ✅ Complete — `GPUMonitorService.get_all_gpu_info()`, `/api/v1/system/gpu-list` |
| FR-2.2 | Select GPUs for training/extraction job | ✅ Complete — `gpu_id` param wired through API → schema → worker |
| FR-2.3 | Exclude busy GPUs | Partial — GPU watchdog tracks usage; no UI exclusion yet |
| FR-2.4 | Memory-based GPU recommendation | Planned |

### 2.3 Enhanced Monitoring
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-3.1 | Toggle: Aggregated vs. Per-GPU view | ✅ Complete — GPU comparison view (commit 8cbe31c) |
| FR-3.2 | Aggregated VRAM usage (total across GPUs) | ✅ Complete |
| FR-3.3 | Aggregated utilization (average) | ✅ Complete |
| FR-3.4 | Per-GPU separate meters | ✅ Complete — per-GPU WebSocket channels `system/gpu/{gpu_id}` |
| FR-3.5 | Per-GPU temperature/power display | ✅ Complete |

### 2.4 Load Balancing
| Requirement | Description | Status |
|-------------|-------------|--------|
| FR-4.1 | Automatic workload distribution | Partial — manual `gpu_id` routing; no auto-scheduling |
| FR-4.2 | Memory-aware batch allocation | Planned |
| FR-4.3 | Straggler detection and handling | Planned (DDP dependency) |

---

## 3. Architecture Design

### 3.1 Distributed Training Pattern
```
┌─────────────────────────────────────────────────────────────┐
│                    Training Orchestrator                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Data Loader (Shared)                     │   │
│  │         Batch splitting across GPUs                   │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         ▼               ▼               ▼                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   GPU 0    │  │   GPU 1    │  │   GPU 2    │            │
│  │ SAE Replica│  │ SAE Replica│  │ SAE Replica│            │
│  │            │  │            │  │            │            │
│  │ Forward    │  │ Forward    │  │ Forward    │            │
│  │ Backward   │  │ Backward   │  │ Backward   │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │                    │
│        └───────────────┼───────────────┘                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Gradient Synchronization                 │   │
│  │           (All-Reduce via NCCL)                      │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Optimizer Step (Primary)                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Monitoring View Toggle
```
                    [Aggregated ●] [Per-GPU ○]

Aggregated View:                 Per-GPU View:
┌────────────────┐               ┌────────────────┐
│ Total VRAM     │               │ GPU 0: 6.2/8GB │
│ ████████░░░░░░ │               │ ████████░░░░░░ │
│ 18.6 / 24 GB   │               │ GPU 1: 5.8/8GB │
│                │               │ ███████░░░░░░░ │
│ Avg Util: 82%  │               │ GPU 2: 6.6/8GB │
└────────────────┘               │ █████████░░░░░ │
                                 └────────────────┘
```

---

## 4. User Interface

### 4.1 GPU Selection in Training Modal
```
┌─────────────────────────────────────────────────────────────┐
│ Start Training                                              │
├─────────────────────────────────────────────────────────────┤
│ GPU Selection                                               │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [✓] GPU 0: NVIDIA RTX 3090 (24GB) - Available          │ │
│ │ [✓] GPU 1: NVIDIA RTX 3090 (24GB) - Available          │ │
│ │ [ ] GPU 2: NVIDIA RTX 3080 (10GB) - In Use (Training)  │ │
│ └─────────────────────────────────────────────────────────┘ │
│ Selected: 2 GPUs | Effective Batch Size: 8192               │
├─────────────────────────────────────────────────────────────┤
│ [Continue to Hyperparameters →]                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Enhanced System Monitor
```
┌─────────────────────────────────────────────────────────────┐
│ System Monitor              [Aggregated ●] [Per-GPU ○]      │
├─────────────────────────────────────────────────────────────┤
│ Per-GPU View:                                               │
│ ┌─────────────────────────┐ ┌─────────────────────────────┐ │
│ │ GPU 0 (RTX 3090)        │ │ GPU 1 (RTX 3090)            │ │
│ │ Util: ████████░░ 82%    │ │ Util: ███████░░░ 75%        │ │
│ │ Mem:  ████████░░ 6.2GB  │ │ Mem:  ███████░░░ 5.8GB      │ │
│ │ Temp: 68°C | 245W       │ │ Temp: 65°C | 230W           │ │
│ └─────────────────────────┘ └─────────────────────────────┘ │
│ ┌─────────────────────────┐                                 │
│ │ GPU 2 (RTX 3080)        │                                 │
│ │ Util: █████████░ 92%    │                                 │
│ │ Mem:  █████████░ 8.9GB  │                                 │
│ │ Temp: 72°C | 280W       │                                 │
│ └─────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/system/gpus` | GET | List available GPUs with status |
| `/api/v1/system/gpus/{id}` | GET | Get specific GPU details |
| `/api/v1/trainings` | POST | Extended with `gpu_ids` field |
| `/api/v1/system/metrics?view=aggregated` | GET | Aggregated metrics |
| `/api/v1/system/metrics?view=per_gpu` | GET | Per-GPU metrics |

---

## 6. Data Model Extensions

### 6.1 Training Table Extension
```sql
ALTER TABLE trainings ADD COLUMN gpu_ids INTEGER[];  -- Selected GPU indices
ALTER TABLE trainings ADD COLUMN distributed BOOLEAN DEFAULT FALSE;
```

### 6.2 SystemMetrics Extension
```sql
CREATE TABLE gpu_metrics (
    id UUID PRIMARY KEY,
    gpu_index INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    utilization FLOAT,
    memory_used BIGINT,
    memory_total BIGINT,
    temperature FLOAT,
    power_draw FLOAT,
    job_id UUID,  -- Which job is using this GPU
    UNIQUE(gpu_index, timestamp)
);
```

---

## 7. Implementation Approach

### 7.1 PyTorch Distributed
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(sae, device_ids=[local_rank])

# Training loop with gradient sync
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradients auto-synced
    optimizer.step()
```

### 7.2 Batch Size Scaling
```python
effective_batch_size = base_batch_size * num_gpus
per_gpu_batch_size = base_batch_size
```

---

## 8. Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `multi_gpu_enabled` | Enable multi-GPU features | false |
| `default_gpu_selection` | all, available, manual | available |
| `monitor_view_default` | aggregated, per_gpu | aggregated |

---

## 9. Dependencies

| Feature | Dependency Type |
|---------|-----------------|
| SAE Training | Extends training system |
| System Monitoring | Extends monitoring views |

---

## 10. Implementation Phases

### Phase 1: Enhanced Monitoring ✅ Complete (Dec 2025)
- [x] Per-GPU metrics collection — `GPUMonitorService.get_all_gpu_metrics()`
- [x] Aggregated vs. per-GPU view toggle — GPU comparison view (commit 8cbe31c)
- [x] Per-GPU charts in dashboard — WebSocket channels `system/gpu/{gpu_id}`

### Phase 2: GPU Selection ✅ Complete (Dec 2025)
- [x] GPU availability detection — `pynvml` enumeration, `/api/v1/system/gpu-list`
- [x] GPU selection for extraction jobs — `gpu_id` param wired API → schema → worker
- [x] GPU watchdog task — monitors processes per device
- [ ] Busy GPU exclusion from UI (partial)

### Phase 3: Distributed Training ⏳ Planned
- [ ] PyTorch DDP integration
- [ ] Gradient synchronization (NCCL)
- [ ] Batch size scaling
- [ ] Multi-GPU progress tracking

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| NCCL communication overhead | Performance | Profile and optimize |
| Memory imbalance across GPUs | Training failure | Memory-aware allocation |
| Straggler GPUs | Slowdown | Load balancing |
| Single point of failure | Training loss | Checkpoint frequently |

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Training speedup (2 GPUs) | 1.8x |
| Training speedup (4 GPUs) | 3.2x |
| Memory efficiency | >90% |
| GPU utilization (distributed) | >80% |

---

## Doc-Refresh Corrections (2026-07-11)

Verified against the code — separating what's *built* from what's *planned*.

### Actually implemented ✅
- **Phase 1 monitoring:** `GPUMonitorService` (`get_all_gpu_info()`,
  `get_all_gpu_metrics()`, `get_device_count()`, `is_available()`,
  `get_gpu_metrics(gpu_id)`); per-GPU WS channels `system/gpu/{id}`; the
  aggregated-vs-per-GPU `viewMode` toggle.
- **Extraction GPU routing:** `gpu_id` on the extraction schema/task with per-GPU
  memory cleanup + index validation.
- **GPU watchdog:** `workers/gpu_watchdog_task.py`, scheduled in Celery Beat.
- **Real GPU routes** (under `/api/v1/system`): `/gpu-list`, `/gpu-metrics`,
  `/gpu-metrics/all`, `/gpu-info`, `/gpu-processes` (not `/system/gpus[/{id}]`).

### NOT implemented (doc previously implied current) ❌
- **Training GPU selection (FR-1.3):** no `gpu_id`/`gpu_ids` anywhere in the
  training schema/service/task — training runs on the default CUDA device.
- **Data model §6:** `trainings.gpu_ids`/`distributed` columns and the
  `gpu_metrics` table **do not exist** — GPU metrics are ephemeral (WebSocket
  only, never persisted). This is Phase-3 design intent.
- **Config §8:** `multi_gpu_enabled` / `default_gpu_selection` /
  `monitor_view_default` settings **do not exist**.
- Steering is hard-pinned to `CUDA_VISIBLE_DEVICES=0` (single-GPU assumption).

---

*Related: [Project PRD](000_PPRD|miStudio.md) | [TDD](../tdds/009_FTDD|Multi_GPU_Scalability.md) | [TID](../tids/009_FTID|Multi_GPU_Scalability.md)*
