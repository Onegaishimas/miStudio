---
sidebar_position: 4
title: "Exporting and Sharing"
description: "Export SAEs and push to Neuronpedia"
---

# Exporting and Sharing

## Local Export
Export SAE weights as `.safetensors` files for use in Python research environments (SAELens, TransformerLens).

## Neuronpedia Integration
Two modes for sharing discoveries with the research community:

1. **Export to ZIP:** Package SAE data, labels, and activation examples for manual upload
2. **Direct Push:** Push directly to a local Neuronpedia instance with progress tracking:
   - Async processing via Celery with WebSocket progress updates
   - Computes dashboard data (logits, feature statistics) during push
   - Proper model naming for discoverability
