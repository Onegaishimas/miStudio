---
sidebar_position: 5
title: "Auto-Labeling"
description: "Interpreting features at scale with LLM-powered labeling"
---

# Auto-Labeling — Interpreting Features at Scale

With 8,000–131,000 features, manual labeling is impractical. miStudio's auto-labeling system uses LLMs to interpret each feature from its activation examples.

![Labeling Configuration Panel — Method selection and options](/img/miStudio_Extraction_Panel-FeatureLabelConfigPanel.jpg)

## Four Labeling Methods

| Method | Cost | Speed | Privacy | Best For |
|--------|------|-------|---------|----------|
| **OpenAI** | $$$ | Fast | Cloud | Highest quality labels. GPT-4o-mini is cost-effective. |
| **OpenAI-Compatible** | Free–$ | Variable | Local/Cloud | Local models via Ollama, vLLM, miLLM, or any OpenAI-compatible API |
| **Local** | Free | Slow | Full | HuggingFace models loaded directly. Complete privacy. |
| **Manual** | Free | Slowest | Full | Human-provided labels for verification or correction |

:::info OpenAI-Compatible Endpoints
This is the most flexible option. Point it at any OpenAI-compatible API:
- **Ollama:** `http://localhost:11434/v1` (local, free)
- **miLLM:** `http://millm-backend:8000/v1` (local, free, GPU-accelerated)
- **vLLM:** `http://localhost:8000/v1` (local, high throughput)
- **Together AI, Fireworks, etc.:** Cloud providers with OpenAI-compatible APIs
:::

## Labeling Configuration

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Batch Size** | 10 | 1–100 | Features labeled in parallel. Higher = faster but may overwhelm local models. |
| **Max Examples** | 25 | 10–50 | Activation examples shown to the LLM per feature. More = better context but longer prompts. |
| **Max Tokens** | 300 | 50–8,000 | Maximum response length from LLM. Increase for reasoning models that use `<think>` tags. |
| **API Timeout** | 120s | 30–600s | Request timeout. Increase for large local models. |

:::warning Reasoning Models
Models like LFM2.5-Thinking or DeepSeek-R1 produce `<think>...</think>` tags before their answer. miStudio automatically strips these. Set `max_tokens` to 1,000–2,000 to ensure the actual answer isn't truncated after the thinking phase.
:::

## The Dual-Label System

Every feature receives two labels:

- **Semantic Label:** A descriptive name (e.g., "Legal Precedents in UK Law")
- **Category:** A high-level classification tag (e.g., "legal", "structural", "semantic")

Labels track their **provenance** — whether they came from OpenAI, a local model, or manual editing — enabling quality comparison across methods.

## Labeling Progress & Results

Once a labeling job is running, track progress in real-time:

![Labeling Job Progress — Real-time progress and labeled results](/img/miStudio_Labeling_Panel-LabelingJobProgressResults.jpg)

Browse completed labeling results across all jobs:

![Labeling Results Browser — View and compare labels across jobs](/img/miStudio_Labeling_Panel-LabelingJobResultsPanelBrowser.jpg)

## Prompt Templates

Customize how the LLM analyzes features by editing **Labeling Prompt Templates**:
- Change the "persona" of the labeling assistant
- Adjust analysis instructions for different research goals
- Add domain-specific context for specialized datasets
