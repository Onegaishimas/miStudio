---
sidebar_position: 4
title: "Trainings API"
description: "SAE training job endpoints — create, control, metrics, checkpoints"
---

# Trainings API

Prefix: `/api/v1/trainings` · UI: [SAE Training](/core-workflow/sae-training)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create + start a training job (201). Config: `model_id`, dataset tokenization(s) or cached extraction(s), layers, hook types (`residual`/`mlp`/`attention`), framework + hyperparameters |
| `GET` | `` | List trainings (paginated) |
| `GET` | `/{id}` | Get training details (status, progress, live loss/L0/dead-neuron stats) |
| `PATCH` | `/{id}` | Update training metadata |
| `DELETE` | `/{id}` | Delete training and its artifacts (204) |
| `POST` | `/{id}/control` | Control a running job — body `{"action": "pause" \| "resume" \| "stop"}` |
| `GET` | `/{id}/metrics` | Time-series metrics (per step, optionally per `layer_idx` for multi-hook runs) |
| `GET` | `/{id}/checkpoints` | List saved checkpoints |
| `GET` | `/{id}/checkpoints/best` | The lowest-loss checkpoint |

**Notes**

- There is no `/{id}/retry` — "Retry" in the UI re-`POST`s a new training with the copied config.
- Multi-dataset and cached-activation training use `dataset_ids` / `extraction_ids` arrays in the create payload — see [Multi-Dataset Training](/advanced/multi-dataset).
- Metrics rows are unique per `(training_id, step, layer_idx)`; `layer_idx = null` rows are the aggregated series.

**Progress channels:** `trainings/{id}/progress` (events `training:progress|completed|failed|status_changed`), `trainings/{id}/checkpoints` (`checkpoint:created`), and `trainings/{id}/deletion` for delete progress.
