---
sidebar_position: 6
title: "Features & Labeling API"
description: "Feature browsing, analysis, bulk labeling, and enhanced labeling endpoints"
---

# Features & Labeling API

These routers are mounted **without a prefix** — paths sit directly under `/api/v1`. UI: [Feature Extraction](/core-workflow/feature-extraction), [Auto-Labeling](/core-workflow/auto-labeling), [Enhanced Labeling](/core-workflow/enhanced-labeling).

## Extraction jobs & feature browsing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/extractions` | List feature-extraction jobs |
| `DELETE` | `/extractions/{id}` | Delete an extraction job and its features |
| `GET` | `/extractions/{id}/features` | Features from one extraction (paginated, filterable) |
| `GET` | `/trainings/{tid}/features` | Features by source training |
| `GET` | `/features/{id}` | Feature detail |
| `GET` | `/trainings/{tid}/features/by-index/{idx}` | Look up a feature by neuron index (trained SAE) |
| `GET` | `/saes/{sae_id}/features/by-index/{idx}` | Look up a feature by neuron index (external SAE) |

## Feature detail & curation

| Method | Path | Description |
|--------|------|-------------|
| `PATCH` | `/features/{id}` | Edit name, category, description, notes |
| `POST` | `/features/{id}/favorite` | Toggle favorite |
| `POST` | `/features/{id}/star` | Set star color — `?star_color=yellow\|purple\|aqua` (aqua marks completed enhanced labels and is protected from bulk overwrite) |
| `GET` | `/features/{id}/examples` | Top activating examples with per-token activations |
| `GET` | `/features/{id}/token-analysis` | Aggregated token statistics |
| `GET` | `/features/{id}/logit-lens` | Promoted/suppressed vocabulary tokens |
| `GET` | `/features/{id}/correlations` | Correlated features |
| `GET` | `/features/{id}/ablation` | Ablation analysis |

## NLP analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/extractions/{id}/analyze-nlp` | Run NLP analysis across an extraction's features |
| `POST` | `/extractions/{id}/cancel-nlp` / `.../reset-nlp` | Cancel / reset that analysis |
| `POST` | `/features/{id}/analyze-nlp` | Analyze a single feature |
| `GET` | `/features/{id}/nlp-analysis` | Retrieve stored analysis |
| `POST` | `/analysis/cleanup` | Clean up orphaned analysis artifacts |

## Bulk labeling

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/labeling` | Start a bulk labeling job (LLM labels many features) |
| `POST` | `/extractions/{id}/label` | Start labeling scoped to one extraction |
| `GET` | `/labeling` | List labeling jobs |
| `GET` | `/labeling/{job_id}` | Job status + results |
| `POST` | `/labeling/{job_id}/cancel` | Cancel a running job |
| `DELETE` | `/labeling/{job_id}` | Delete a job |
| `GET` | `/labeling/models/available` | Models served by the configured local endpoint |
| `POST` | `/labeling/models/openai` | List models available to your OpenAI API key |

Returns **503** when the labeling endpoint has no model loaded — see [troubleshooting](/troubleshooting#labeling).

## Enhanced labeling (per-feature, two-pass)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/features/{id}/label/enhanced` | Start the two-pass analysis (parallel per-example summaries → synthesis) |
| `GET` | `/features/{id}/label/enhanced/latest` | Latest enhanced-labeling job + result for the feature |

**Progress channels:** `extraction/{id}` (feature extraction), `labeling/{job_id}/progress` + `/results` (bulk), `enhanced_labeling/{job_id}` (events `enhanced_labeling:progress|completed|failed`).
