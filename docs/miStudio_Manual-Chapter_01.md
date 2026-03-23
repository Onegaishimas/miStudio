---
sidebar_position: 1
title: "miStudio User Manual"
description: "Complete guide to MechInterp Studio - from installation to advanced feature steering"
---

# MechInterp Studio (miStudio): The Complete User Manual

## Part 1: Getting Started

### Section 1.1: Introduction to miStudio

MechInterp Studio (miStudio) is an end-to-end mechanistic interpretability platform designed to replace the fragmented tooling typically associated with AI safety research — Jupyter notebooks, custom scripts, and manually tracked experiments — with a professional, database-backed workbench.

By providing a unified environment for data management, SAE training, feature discovery, and causal intervention testing, miStudio allows researchers to move from hypothesis to proven intervention in a fraction of the time required by traditional methods.

#### The Scalability Spectrum: Edge to Cluster

miStudio is engineered with a "scale-agnostic" architecture:

- **At the Edge:** Run on an NVIDIA Jetson Orin or a laptop with a single RTX 3060. The software optimizes memory via quantization (4-bit, 8-bit) and micro-batching for 1B–3B parameter models.
- **In the Lab:** Deploy on multi-GPU workstations. miStudio detects CUDA devices and distributes extraction and training jobs across all available VRAM via its Celery/Redis task queue.
- **In the Cloud:** Deploy on GCP/AWS with Kubernetes for auto-scaling GPU access.

:::info Why Not Notebooks?
Jupyter notebooks suffer from "hidden state" — results depend on cell execution order. miStudio enforces structured workflows where every experiment is recorded in PostgreSQL with exact hyperparameters, making research reproducible by default.
:::

:::tip The Superposition Hypothesis
Modern LLMs represent more concepts than they have neurons through **superposition** — neurons become "polysemantic," firing for unrelated concepts. Sparse Autoencoders (SAEs) "unpack" these neurons into individual, monosemantic features. miStudio is the workbench for this science.
:::

### Section 1.2: System Requirements & Installation

#### Hardware Requirements

| Tier | VRAM | Capability |
|------|------|-----------|
| **Minimum** | 8 GB | TinyLlama (1.1B), Phi-2, Phi-4-mini |
| **Recommended** | 16–24 GB (RTX 3090/4090) | Models up to 9B, wide SAEs (16k–131k features) |
| **Multi-GPU** | 2×24 GB+ | Dedicated inference + training partitions |

:::warning VRAM vs. System RAM
System RAM cannot compensate for low VRAM. Model weights and activations must reside on the GPU for acceptable speed. If a job exceeds VRAM, you'll get an "Out of Memory" (OOM) crash — the most common failure mode in local research.
:::

#### Software Installation

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

### Section 1.3: Navigating the Dashboard

The interface uses a **collapsible sidebar** with grouped navigation sections. The sidebar can collapse to icon-only mode for maximum content area.

#### Panel Overview

| Section | Panel | Purpose |
|---------|-------|---------|
| **Data** | Datasets | Download from HuggingFace, upload local files, tokenize |
| **Data** | Models | Download, quantize, inspect architecture |
| **Training** | Training | Configure and run SAE training jobs |
| **Training** | Training Templates | Save/load/share training configurations |
| **Analysis** | Extractions | Run feature extraction from trained or external SAEs |
| **Analysis** | Extraction Templates | Save/load extraction configurations |
| **Analysis** | Labeling | Manage auto-labeling jobs for feature interpretation |
| **Analysis** | Labeling Templates | Customize LLM prompts for labeling |
| **Analysis** | SAEs | Browse and download external SAEs from HuggingFace |
| **Steering** | Steering | Feature intervention matrix testing |
| **Steering** | Prompt Templates | Manage reusable steering prompts |
| **System** | Settings | API keys, endpoints, display preferences |
| **System** | Monitor | Real-time GPU, CPU, memory, disk metrics |

#### Real-Time System Monitoring

The **Monitor** panel provides live WebSocket-driven metrics:

- **GPU Utilization & Temperature:** Watch for thermal throttling (usually ~85°C). If hit, clock speeds drop and training times multiply.
- **VRAM Pressure:** If VRAM is >90% before starting a job, unload unused models or use more aggressive quantization.
- **Disk I/O:** During activation extraction, miStudio writes large tensor files. High disk I/O with low GPU utilization means your storage is the bottleneck.

:::tip Interpreting Metrics
- **100% GPU Utilization** = good (GPU isn't waiting for data)
- **High Power Draw** = high heat. Laptops should be plugged in with high-wattage supply.
- **Network I/O** spikes indicate HuggingFace downloads or WebSocket activity.
:::

#### Application Settings

The **Settings** panel provides persistent configuration:

| Tab | What It Controls |
|-----|-----------------|
| **Endpoints** | Saved API endpoints for labeling (OpenAI-compatible URLs) |
| **API Keys** | Encrypted storage (AES-256-GCM) for OpenAI, HuggingFace tokens |
| **Labeling** | Default batch size, token filtering preferences |
| **Display** | Theme and UI customization |

:::info Encrypted Settings
API keys are stored with AES-256-GCM encryption in the database — never in plain text or environment variables.
:::

---

## Part 2: The Core Workflow

### Section 2.1: The Researcher's Journey

The mechanistic interpretability pipeline follows five stages:

```
Model → Dataset → Activations → SAE Training → Feature Discovery → Steering
  ↓         ↓          ↓              ↓                ↓              ↓
Select    Prepare    Extract       Disentangle      Interpret      Prove
the LLM   stimuli   internal      superposed       what each     causation
                     numbers       features         feature       with
                                                    means         intervention
```

1. **The Subject (Model):** Select an LLM — the "brain" you're dissecting
2. **The Stimuli (Dataset):** Text that "stimulates" the model to activate different concepts
3. **The Capture (Extraction):** Record internal activations as the model processes text
4. **The Disentanglement (SAE):** Train a Sparse Autoencoder to "untangle" polysemantic neurons
5. **The Proof (Steering):** Manipulate discovered features to verify causal influence

### Section 2.2: Data & Model Management

#### Model Ingestion

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

#### Understanding Hook Points

When configuring extraction or training, you must choose **where** to place probes inside the model. Each location reveals different aspects of the model's computation:

| Hook Type | What It Captures | When to Use |
|-----------|-----------------|-------------|
| **Residual Stream** (`residual`) | The "main highway" — cumulative information from all previous layers | Default choice. Best for general feature discovery. |
| **MLP Layer** (`mlp`) | Factual "lookup tables" — world knowledge, entity associations | When investigating specific facts or knowledge storage |
| **Attention Layer** (`attention`) | Relational reasoning — how tokens attend to each other | When investigating grammar, coreference, or syntactic patterns |

:::info Multi-Hook Training
You can train SAEs on **multiple layers and multiple hook types** simultaneously. For example, selecting layers [6, 12] with hooks [residual, mlp] creates 4 separate SAEs in one training job — one for each layer×hook combination.
:::

#### Tokenization & Stride

Models process text as integer **tokens**, not words. When tokenizing a dataset:

- **Max Length:** The context window size (e.g., 1024 tokens). Longer = more context per sample but more VRAM.
- **Stride:** Overlap between chunks. A stride of 512 with max length 1024 means each chunk shares half its tokens with the next, preventing concepts from being split across chunk boundaries.

### Section 2.3: SAE Training — Building the Prism

The **Training** panel is where you build the Sparse Autoencoder that decomposes polysemantic neurons into monosemantic features.

#### The Six SAE Frameworks

miStudio supports six paper-grounded SAE architectures, each with different sparsity mechanisms and trade-offs:

##### 1. Standard SAELens (Bricken et al., 2023)

The classic approach — ReLU activation with L1 sparsity penalty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l1_alpha` | 5e-4 | L1 penalty strength. Higher = sparser but risks dead features |
| `normalize_activations` | `constant_norm_rescale` | Rescales activations to unit norm before encoding |

**When to use:** General-purpose feature discovery. Good starting point for new researchers.

:::tip L1 Tuning Guide
- **Too many messy features?** Increase `l1_alpha` (try 2x)
- **Too many dead features (>50%)?** Decrease `l1_alpha` (try 0.5x)
- **Target:** L0 between 10–100 active features per token, dead neurons <20%
:::

##### 2. Standard Anthropic (Templeton et al., 2024)

Anthropic's variant with specialized normalization and higher default sparsity.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `l1_alpha` | 5.0 | Much higher than SAELens — Anthropic's normalization rescales differently |
| `normalize_activations` | `anthropic_rescale` | Anthropic-specific rescaling that changes the L1 coefficient scale |

:::danger L1 Scale Warning
The `l1_alpha` for Anthropic (default 5.0) is NOT comparable to SAELens (default 5e-4). The normalization modes change what the coefficient means. Do NOT copy L1 values between frameworks.
:::

**When to use:** When replicating Anthropic's published results or using their recommended configurations.

##### 3. JumpReLU (Rajamanoharan et al., 2024 — Gemma Scope)

High-performance architecture using learnable thresholds. Features are binary — OFF below threshold, full activation above.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `sparsity_coeff` (λ) | 1e-3 | 1e-5 to 5e-3 | L0 coefficient (paper scale). At L0=50: loss_l0 = 1e-3 × 50 = 0.05 |
| `initial_threshold` | 0.5 | 0–5.0 | Starting jump threshold per feature |
| `bandwidth` | 0.01 | 0–1.0 | STE gradient estimation bandwidth |
| `normalize_decoder` | true | — | Required: keeps decoder columns at unit norm |

**How it works:** Instead of a continuous penalty, JumpReLU uses a step function approximated by a sigmoid for gradient flow: `σ((z-θ)/ε)` where θ is a learnable per-feature threshold. Features are counted (not averaged) for the L0 loss: `L0 = Σ_i H(z_i - θ_i)` summed per sample, then averaged over the batch.

:::warning Sparsity Coefficient Scale
`sparsity_coeff` for JumpReLU is on a completely different scale than `l1_alpha`. Typical values: 1e-4 to 5e-3. Do NOT use L1 values (like 0.0005) for this parameter — they will produce zero sparsity pressure.
:::

**When to use:** Best for preventing "shrinkage" — features activate sharply rather than being penalized into small values. Preferred for production-quality SAEs.

##### 4. TopK (Gao et al., 2024 — OpenAI)

Structural sparsity — exactly K features activate per input, no penalty needed.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 50 | Exact number of active features per sample |
| `aux_loss_alpha` | 0.03125 | Auxiliary loss weight for dead feature prevention (1/32 per paper) |
| `aux_k` | `top_k × 2` | Features used in auxiliary loss computation |
| `adam_epsilon` | 6.25e-10 | Paper-specific Adam optimizer epsilon |

**How it works:** After encoding, only the top K activations are kept; all others are zeroed. An auxiliary loss encourages dead features to eventually activate.

**When to use:** When you want exact control over sparsity level. No L1/L0 tuning needed — just set K.

##### 5. Skip (Community Variant)

Standard L1 sparsity with a residual skip connection from input to decoder output.

**When to use:** When reconstruction quality is critical — the skip connection provides an "escape hatch" for information the SAE bottleneck can't capture.

##### 6. Transcoder (Dunefsky et al., 2024)

Predicts MLP output from MLP input — learns the transformation a layer performs.

**When to use:** When studying how information transforms between layers, not just what's represented at a single point.

#### Framework-Aware Configuration

When you select a framework, the UI automatically:
- Shows/hides framework-specific fields (e.g., `top_k` only appears for TopK)
- Sets paper-grounded defaults for all parameters
- Adjusts validation ranges to match the framework's expected scales

#### Activation Normalization Modes

Before feeding activations into the SAE, they can be normalized:

| Mode | Description | Used By |
|------|-------------|---------|
| `constant_norm_rescale` | Rescale to constant L2 norm | SAELens Standard |
| `anthropic_rescale` | Anthropic-specific rescaling | Standard Anthropic |
| `none` | Raw activations, no normalization | Skip, Transcoder |

:::info Why Normalization Matters
Different normalization modes change the scale of activations entering the encoder, which changes the effective meaning of sparsity coefficients. This is why `l1_alpha=5.0` for Anthropic produces similar sparsity to `l1_alpha=0.0005` for SAELens — the normalization absorbs the difference.
:::

#### Essential Hyperparameters

Beyond architecture-specific settings, these apply to all frameworks:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Learning Rate** | 3e-4 | 1e-5 to 1e-2 | If loss spikes: too high. If loss barely moves: too low. |
| **Batch Size** | 4096 | 32–65536 | Larger = smoother gradients but more VRAM |
| **Total Steps** | 30,000 | 1,000–1,000,000 | More steps = better features, longer training |
| **Warmup Steps** | 1,000 | 0–10,000 | Linear LR warmup prevents early instability |
| **Sparsity Warmup** | 5,000 | 0–50,000 | Gradually increases sparsity pressure. Critical for JumpReLU to prevent mass neuron death at initialization. |
| **Expansion Factor** | 8–32× | 2–128× | SAE width relative to model hidden dim. 8× is common (512 neurons → 4,096 features). |
| **Weight Decay** | 0.0 | 0–0.1 | L2 regularization. Usually 0 for SAEs. |
| **Gradient Clip Norm** | 1.0 | 0–10.0 | Prevents gradient explosions during training. |

#### Dead Neuron Management

Features that never activate are "dead neurons" — wasted capacity.

| Setting | Default | Description |
|---------|---------|-------------|
| `dead_neuron_threshold` | 1,000 steps | Steps of inactivity before a feature is considered dead |
| `resample_dead_neurons` | true | Automatically reinitialize dead features |
| `resample_interval` | 5,000 steps | How often to check and resample dead features |

:::tip Dead Neuron Debugging
- **>50% dead:** Your sparsity pressure is too high. Reduce `l1_alpha` or `sparsity_coeff`.
- **<5% dead:** Your sparsity might be too low — features may be polysemantic.
- **10–20% dead** is typical and healthy.
- **TopK:** Uses auxiliary loss instead of resampling — set `aux_loss_alpha` higher if too many die.
:::

#### Training Metrics

While training, miStudio streams these metrics in real-time via WebSocket:

| Metric | Target | What It Means |
|--------|--------|--------------|
| **Total Loss** | Decreasing | Combined reconstruction + sparsity loss |
| **Reconstruction Loss (MSE)** | < 0.1 | How much "truth" the SAE lost. Lower = more faithful. |
| **L0 (Sparsity)** | 10–100 | Average active features per token. Lower = easier to interpret. |
| **FVU** | < 0.1 | Fraction of Variance Unexplained. Below 0.05 is excellent. |
| **Dead Neurons %** | 10–20% | Features that never fire. See Dead Neuron Management above. |

#### Training Controls

Active training jobs support:
- **Pause/Resume:** Suspend training to free GPU for other work, then continue
- **Stop:** Gracefully end training (saves final checkpoint)
- **Checkpoints:** Saved every N steps (configurable). Each records loss, L0, and model weights. The "best checkpoint" (lowest loss) is tracked automatically.

#### Training Templates

Save any training configuration as a **template** for reproducibility:
- Export as JSON to share with colleagues
- Import templates from other researchers
- Mark favorites for quick access
- Duplicate and modify for parameter sweeps

### Section 2.4: Feature Extraction — Recording the Evidence

After training (or downloading an external SAE), run an **Extraction Job** to scan your dataset and record which features activate on which tokens.

#### Extraction Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Evaluation Samples** | 10,000 | 100–1,000,000 | Dataset samples to scan. More = better coverage but slower. |
| **Top-K Examples** | 100 | 10–1,000 | Max-activating examples saved per feature. More = richer context for labeling. |
| **Batch Size** | Auto | 8–256 | Processing batch size. Auto-detected based on available VRAM. |

#### Token Filtering

Control which tokens appear in activation examples. These filters affect both extraction and labeling:

| Filter | Default | Effect |
|--------|---------|--------|
| **Special Tokens** | ✅ On | Removes `<s>`, `</s>`, `<pad>`, etc. |
| **Single Characters** | ✅ On | Removes single-character tokens |
| **Punctuation** | ✅ On | Removes pure punctuation tokens |
| **Numbers** | ✅ On | Removes pure numeric tokens |
| **Fragments** | ✅ On | Removes BPE subwords like "tion", "ing" |
| **Stop Words** | ❌ Off | Optionally removes "the", "and", "is", etc. |

:::tip Filter Strategy
Keep most filters ON for cleaner labeling results. Only disable fragment filtering if you're specifically studying tokenization patterns. Enable stop word filtering when you want labels focused on content words.
:::

#### Context Window

Each activation example includes surrounding context for interpretation:

| Setting | Default | Description |
|---------|---------|-------------|
| **Prefix Tokens** | 25 | Tokens shown before the activating token |
| **Suffix Tokens** | 25 | Tokens shown after the activating token |

The asymmetric window (25+25=50 tokens of context) is based on research showing this window size captures sufficient context for accurate labeling.

#### Dead Feature Filtering

Features that activate too rarely are filtered:

| Setting | Default | Description |
|---------|---------|-------------|
| **Min Activation Frequency** | 0.001 (0.1%) | Features firing less than this rate are excluded as "dead" |

### Section 2.5: Auto-Labeling — Interpreting Features at Scale

With 8,000–131,000 features, manual labeling is impractical. miStudio's auto-labeling system uses LLMs to interpret each feature from its activation examples.

#### Four Labeling Methods

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

#### Labeling Configuration

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Batch Size** | 10 | 1–100 | Features labeled in parallel. Higher = faster but may overwhelm local models. |
| **Max Examples** | 25 | 10–50 | Activation examples shown to the LLM per feature. More = better context but longer prompts. |
| **Max Tokens** | 300 | 50–8,000 | Maximum response length from LLM. Increase for reasoning models that use `<think>` tags. |
| **API Timeout** | 120s | 30–600s | Request timeout. Increase for large local models. |

:::warning Reasoning Models
Models like LFM2.5-Thinking or DeepSeek-R1 produce `<think>...</think>` tags before their answer. miStudio automatically strips these. Set `max_tokens` to 1,000–2,000 to ensure the actual answer isn't truncated after the thinking phase.
:::

#### The Dual-Label System

Every feature receives two labels:

- **Semantic Label:** A descriptive name (e.g., "Legal Precedents in UK Law")
- **Category:** A high-level classification tag (e.g., "legal", "structural", "semantic")

Labels track their **provenance** — whether they came from OpenAI, a local model, or manual editing — enabling quality comparison across methods.

#### Prompt Templates

Customize how the LLM analyzes features by editing **Labeling Prompt Templates**:
- Change the "persona" of the labeling assistant
- Adjust analysis instructions for different research goals
- Add domain-specific context for specialized datasets

### Section 2.6: Model Steering — Proving Causation

Steering is the definitive proof of your research. By manipulating specific features during generation, you demonstrate that a feature causally influences model behavior — not just correlates with it.

#### Steering Modes

miStudio provides three distinct steering modes:

| Mode | What It Does | Best For |
|------|-------------|----------|
| **Individual** | One feature at multiple strengths | Understanding a single feature's dose-response curve |
| **Comparison** | Multiple features at the same strength, side-by-side | Comparing related features (e.g., "French" vs "German") |
| **Combined** | All selected features applied simultaneously | Discovering synergistic effects between features |

#### Strength Values

Steering strengths are **raw coefficients** added to the model's residual stream, compatible with Neuronpedia's scale:

| Range | Effect | Example |
|-------|--------|---------|
| **0** | No intervention (baseline) | Unsteered output |
| **0.07 – 5** | Subtle influence | Slight shift in topic or tone |
| **5 – 50** | Moderate effect | Clear behavioral change |
| **50 – 100** | Strong effect | Dominant feature influence |
| **100 – 200** | Very strong | Feature overwhelms other signals |
| **200 – 300** | Extreme | Often causes repetition or collapse |
| **Negative** | Suppression | Inhibits the feature's concept |

:::warning Strength Calibration
The effective range depends on the SAE and layer. Start with strengths around **5–20** and increase gradually. Values above ±100 frequently cause the model to "collapse" into repetitive or incoherent output.
:::

:::tip Multi-Strength Testing
Each feature supports up to 3 **additional strengths** tested simultaneously. Set a primary strength and additional values to see the dose-response curve in one generation pass. For example: primary=10, additional=[5, 20, 50].
:::

#### Generation Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| **Max Tokens** | 100 | 1–2,048 | Length of generated text |
| **Temperature** | 0.7 | 0–2.0 | Randomness. 0 = deterministic, 1.0 = creative |
| **Top-P** | 0.9 | 0–1.0 | Nucleus sampling threshold |
| **Top-K** | 50 | 0–500 | Vocabulary restriction per token |
| **Repetition Penalty** | 1.15 | 0.5–2.0 | Penalizes repeated tokens. Increase if output loops. |
| **Seed** | — | Optional | Set for reproducible results across runs |

#### The Matrix Testing Workflow

Unlike tools with a single slider, miStudio uses a **grid approach**:

1. **Select Features:** Add up to 4 features (e.g., "Honesty" + "French Language")
2. **Add Prompts:** Multiple test prompts processed in batch
3. **Configure Strengths:** Set primary and additional strengths per feature
4. **Include Baseline:** Toggle "Include Unsteered" to compare against the natural output
5. **Execute:** miStudio runs all combinations, presenting results in a structured comparison view

:::info Combined Mode Synergies
Steering multiple features simultaneously is NOT the same as running them separately. "Scientific Tone" + "Excitement" combined may produce different text than either alone. Combined mode reveals **circuit behavior** where features interact.
:::

---

## Part 3: Advanced Usage

### Section 3.1: The Template Ecosystem

miStudio uses JSON templates for scientific reproducibility across three systems:

| Template Type | What It Saves | Use Case |
|--------------|--------------|----------|
| **Training Templates** | All SAE hyperparameters, architecture, layer/hook config | Share exact training recipes |
| **Extraction Templates** | Sample count, token filters, context window settings | Standardize extraction methodology |
| **Labeling Prompt Templates** | LLM persona, analysis instructions, output format | Consistent labeling across teams |

All templates support: create, edit, duplicate, export (JSON), import, and favorites.

### Section 3.2: External SAE Integration

The **SAEs** panel lets you download pre-trained SAEs from HuggingFace:

- **Multi-select downloads:** Browse SAE repositories and select multiple SAEs at once
- **Grouped preview:** Downloads are organized by directory structure
- **Compatibility check:** miStudio validates SAE dimensions against the loaded model
- **Attach to model:** Once downloaded, attach an SAE to the corresponding model layer for extraction or steering

### Section 3.3: Multi-GPU Scalability

For labs with multiple GPUs, miStudio partitions work:

- **Inference Partition:** One GPU stays dedicated to the base model for interactive steering
- **Training Partition:** Remaining GPUs handle SAE training via Celery workers

:::info The Celery/Redis Backbone
GPU jobs can take hours or days. Redis stores the task queue, Celery workers execute tasks. Queue 5 training runs with different L1 settings, close your browser, and check results in the morning — research continues on the backend.
:::

### Section 3.4: Exporting and Sharing

#### Local Export
Export SAE weights as `.safetensors` files for use in Python research environments (SAELens, TransformerLens).

#### Neuronpedia Integration
Two modes for sharing discoveries with the research community:

1. **Export to ZIP:** Package SAE data, labels, and activation examples for manual upload
2. **Direct Push:** Push directly to a local Neuronpedia instance with progress tracking:
   - Async processing via Celery with WebSocket progress updates
   - Computes dashboard data (logits, feature statistics) during push
   - Proper model naming for discoverability

### Section 3.5: Multi-Dataset Training

Train SAEs on data from multiple sources:
- Select multiple datasets per training job
- Use cached activations from previous extractions
- Combine diverse text sources for more robust feature discovery

---

## Appendix A: Troubleshooting

### Common Issues

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

### Key Formulas

| Framework | Loss Function |
|-----------|--------------|
| **Standard** | `L = MSE(x, x̂) + λ · Σ|z_i|` (L1 on activations) |
| **JumpReLU** | `L = MSE(x, x̂) + λ · Σ_i H(z_i - θ_i)` (count of active features) |
| **TopK** | `L = MSE(x, x̂) + α · aux_loss(dead_features)` (no sparsity penalty — K is structural) |
