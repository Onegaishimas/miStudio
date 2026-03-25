---
sidebar_position: 1
title: "The Template Ecosystem"
description: "JSON templates for scientific reproducibility"
---

# The Template Ecosystem

miStudio uses JSON templates for scientific reproducibility across four systems:

| Template Type | What It Saves | Use Case |
|--------------|--------------|----------|
| **Extraction Templates** | Sample count, token filters, context window settings | Standardize extraction methodology |
| **Training Templates** | All SAE hyperparameters, architecture, layer/hook config | Share exact training recipes |
| **Labeling Prompt Templates** | LLM persona, analysis instructions, output format | Consistent labeling across teams |
| **Steering Prompt Templates** | Reusable prompt series for steering experiments | Reproducible steering tests |

All templates support: create, edit, duplicate, export (JSON), import, and favorites.

## Extraction Templates

![Extraction Templates — Standardize extraction methodology](/img/miStudio_Template_Panel-Browse-Extraction_Templates.jpg)

Extraction templates capture sample counts, token filtering settings, context window configuration, and other extraction parameters for consistent methodology across experiments.

## Training Templates

![Training Templates — Save and reuse training configurations](/img/miStudio_Template_Panel-Browse-Training_Templates.jpg)

Save any training configuration as a template for reproducibility. Export as JSON to share with colleagues, import templates from other researchers, and mark favorites for quick access.

## Labeling Prompt Templates

![Labeling Prompt Templates — Customize LLM analysis prompts](/img/miStudio_Template_Panel-Browse-Labeling_Templates.jpg)

Customize how the LLM analyzes features by editing labeling prompt templates. Change the "persona" of the labeling assistant, adjust analysis instructions, and add domain-specific context.

## Steering Prompt Templates

![Creating a Steering Prompt Template](/img/miStudio_Template_Panel-CreateSteeringPromptTemplate.jpg)

Steering prompt templates store reusable series of prompts for batch steering experiments. Create a template with multiple prompts, then apply it across different features and strength configurations for systematic testing.
