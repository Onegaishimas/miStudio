---
sidebar_position: 2
title: "WebSocket Channels"
description: "Every real-time channel, its events, and the fallback behavior"
---

# WebSocket Channels

All real-time updates flow over a single **Socket.IO** connection. The frontend subscribes to per-job **channels** (Socket.IO rooms); events are namespaced `entity:event`. This page catalogs every channel the backend emits, verified against `websocket_emitter.py`.

For how emissions travel from Celery workers to your browser, see [System Architecture](/concepts/architecture#websocket-first-real-time-updates).

## Job progress channels

| Channel | Events | Emitted during |
|---------|--------|----------------|
| `datasets/{id}/progress` | `dataset:progress`, `dataset:completed`, `dataset:error` | Dataset download/processing |
| `datasets/{id}/tokenization/{tok_id}` | `tokenization:progress`, `tokenization:status` | Tokenization jobs |
| `models/{id}/progress` | `model:progress`, `model:completed`, `model:error` | Model download/quantization |
| `models/{id}/extraction` | extraction progress events | Activation extraction (Stage 1) |
| `trainings/{id}/progress` | `training:progress`, `training:completed`, `training:failed`, `training:status_changed` | SAE training (includes live loss/L0/dead-neuron metrics) |
| `trainings/{id}/checkpoints` | `checkpoint:created` | Checkpoint saves |
| `trainings/{id}/deletion` | deletion progress | Training deletion |
| `extraction/{id}` | `extraction:progress`, `extraction:failed`, `extraction:deleted`, `extraction:deletion_progress` | Feature extraction jobs (Stage 2) |
| `sae/{id}/download` | `sae:download` | SAE download from HuggingFace |
| `sae/{id}/upload` | `sae:upload` | SAE upload to HuggingFace |
| `sae/{id}/extraction` | `sae:extraction` | Per-SAE feature-extraction progress |
| `labeling/{job_id}/progress` | labeling progress events | Bulk labeling |
| `labeling/{job_id}/results` | incremental results | Bulk labeling (labels stream in as they're produced) |
| `enhanced_labeling/{job_id}` | `enhanced_labeling:progress`, `enhanced_labeling:completed`, `enhanced_labeling:failed` | Enhanced two-pass labeling |
| `steering/{task_id}` | `steering:progress`, `steering:completed`, `steering:failed` | Async steering generation |
| `neuronpedia/{job_id}/export` | `export:progress` | Neuronpedia ZIP export |
| `neuronpedia/push/{push_job_id}` | `neuronpedia:push_progress`, `neuronpedia:push_completed`, `neuronpedia:push_failed` | Direct push to local Neuronpedia |

## System monitoring

Emitted every **2 seconds** by a Celery Beat task; all use the event `system:metrics`, and payloads carry a `metric_type` discriminator.

| Channel | Payload |
|---------|---------|
| `system/gpu/{gpu_id}` | Per-GPU utilization, memory, temperature, power |
| `system/cpu` | CPU utilization |
| `system/memory` | RAM and swap usage |
| `system/disk` | Disk I/O rates |
| `system/network` | Network I/O rates |

## Client behavior

- **Subscription:** React hooks subscribe on mount, unsubscribe on unmount; handlers update Zustand stores.
- **Polling fallback:** every store detects WebSocket disconnection and falls back to HTTP polling automatically, then stops polling when the socket reconnects. A refresh is never *required* for correctness — the stores re-fetch authoritative state from REST on load.
- **Payload conventions:** progress events include `progress` (0–100) and job-specific fields; failure events include `error`.
