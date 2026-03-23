---
sidebar_position: 2
title: "System Requirements & Installation"
description: "Hardware requirements and software installation guide"
---

# System Requirements & Installation

## Hardware Requirements

| Tier | VRAM | Capability |
|------|------|-----------|
| **Minimum** | 8 GB | TinyLlama (1.1B), Phi-2, Phi-4-mini |
| **Recommended** | 16–24 GB (RTX 3090/4090) | Models up to 9B, wide SAEs (16k–131k features) |
| **Multi-GPU** | 2×24 GB+ | Dedicated inference + training partitions |

:::warning VRAM vs. System RAM
System RAM cannot compensate for low VRAM. Model weights and activations must reside on the GPU for acceptable speed. If a job exceeds VRAM, you'll get an "Out of Memory" (OOM) crash — the most common failure mode in local research.
:::

## Software Installation

miStudio is packaged as a Docker Compose project:

1. **Prerequisites:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. **Network Setup:** Add the domain to your hosts file:
   ```bash
   sudo bash -c 'echo "127.0.0.1  mistudio.mcslab.io" >> /etc/hosts'
   ```
3. **Start all services:**
   ```bash
   ./start-mistudio.sh
   ```

This launches six services:

| Service | Purpose |
|---------|---------|
| **PostgreSQL** | Stores all experiment metadata, labels, metrics, and settings |
| **Redis** | Message broker for the Celery task queue |
| **Celery Worker** | Performs GPU-intensive training, extraction, and labeling tasks |
| **Celery Beat** | Schedules periodic tasks (system monitoring, cleanup) |
| **FastAPI Backend** | API orchestrator with WebSocket support for real-time updates |
| **React Frontend** | Interactive dashboard at `http://mistudio.mcslab.io` |

:::info Why Docker?
A MechInterp environment requires exact versions of PyTorch, Transformers, spaCy, and CUDA kernels. Docker freezes these into a reproducible image — miStudio runs identically on a Jetson Orin and a datacenter server.
:::
