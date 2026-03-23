---
sidebar_position: 100
title: "Troubleshooting"
description: "Common issues and key formulas"
---

# Troubleshooting

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM during training | SAE too wide for available VRAM | Reduce expansion factor, increase batch accumulation, or use Q4 model |
| OOM during steering | Model + SAE + KV cache exceeds VRAM | Use smaller SAE width or more aggressive quantization |
| >50% dead neurons | Sparsity too aggressive | Reduce `l1_alpha`/`sparsity_coeff`, enable sparsity warmup |
| Labels say "uncategorized" | LLM couldn't interpret the feature | Increase `max_examples`, try a larger LLM, check activation examples manually |
| Training loss spikes | Learning rate too high | Reduce by 2–5x, increase warmup steps |
| Training loss plateaus | Learning rate too low or not enough steps | Increase LR or total_steps |
| Labeling timeouts | Local model too slow for batch | Reduce `batch_size` to 1, increase `api_timeout` |
| Steering has no effect | Strength too low or wrong feature | Increase strength (try 20–50), verify feature has clear activation pattern |

## Key Formulas

| Framework | Loss Function |
|-----------|--------------|
| **Standard** | `L = MSE(x, x̂) + λ · Σ|z_i|` (L1 on activations) |
| **JumpReLU** | `L = MSE(x, x̂) + λ · Σ_i H(z_i - θ_i)` (count of active features) |
| **TopK** | `L = MSE(x, x̂) + α · aux_loss(dead_features)` (no sparsity penalty — K is structural) |
