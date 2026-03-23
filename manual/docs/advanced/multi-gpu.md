---
sidebar_position: 3
title: "Multi-GPU Scalability"
description: "Partitioning work across multiple GPUs"
---

# Multi-GPU Scalability

For labs with multiple GPUs, miStudio partitions work:

- **Inference Partition:** One GPU stays dedicated to the base model for interactive steering
- **Training Partition:** Remaining GPUs handle SAE training via Celery workers

:::info The Celery/Redis Backbone
GPU jobs can take hours or days. Redis stores the task queue, Celery workers execute tasks. Queue 5 training runs with different L1 settings, close your browser, and check results in the morning — research continues on the backend.
:::
