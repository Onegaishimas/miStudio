---
sidebar_position: 2
title: "Data & Model Management"
description: "Model ingestion, quantization, hook points, and tokenization"
---

# Data & Model Management

## Model Ingestion

When you enter a HuggingFace ID (e.g., `google/gemma-2-2b`), miStudio:

1. Downloads the model weights to local cache
2. Runs **Dynamic Layer Discovery** to map every layer and hook point
3. Displays the architecture in the model detail view

**Quantization** reduces VRAM usage at the cost of precision:

| Mode | Bits | VRAM Savings | Quality Impact | Best For |
|------|------|-------------|----------------|----------|
| **FP16/BF16** | 16 | Baseline | None | Maximum precision SAE training |
| **Q8 (INT8)** | 8 | ~50% | Minimal | Good balance of speed and accuracy |
| **Q4 (NF4)** | 4 | ~75% | Moderate | Running large models on consumer GPUs |
| **Q2** | 2 | ~87% | Significant | Maximum compression, research only |

:::warning Quantization and SAE Quality
For high-precision SAE training, use FP16/BF16 if VRAM allows. Quantized models add noise to activations, which propagates into the SAE's learned features. The SAE itself is always trained in full precision — only the base model is quantized.
:::

## Understanding Hook Points

When configuring extraction or training, you must choose **where** to place probes inside the model. Each location reveals different aspects of the model's computation:

| Hook Type | What It Captures | When to Use |
|-----------|-----------------|-------------|
| **Residual Stream** (`residual`) | The "main highway" — cumulative information from all previous layers | Default choice. Best for general feature discovery. |
| **MLP Layer** (`mlp`) | Factual "lookup tables" — world knowledge, entity associations | When investigating specific facts or knowledge storage |
| **Attention Layer** (`attention`) | Relational reasoning — how tokens attend to each other | When investigating grammar, coreference, or syntactic patterns |

:::info Multi-Hook Training
You can train SAEs on **multiple layers and multiple hook types** simultaneously. For example, selecting layers [6, 12] with hooks [residual, mlp] creates 4 separate SAEs in one training job — one for each layer×hook combination.
:::

## Tokenization & Stride

Models process text as integer **tokens**, not words. When tokenizing a dataset:

- **Max Length:** The context window size (e.g., 1024 tokens). Longer = more context per sample but more VRAM.
- **Stride:** Overlap between chunks. A stride of 512 with max length 1024 means each chunk shares half its tokens with the next, preventing concepts from being split across chunk boundaries.
