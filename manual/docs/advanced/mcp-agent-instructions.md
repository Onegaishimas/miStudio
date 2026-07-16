---
sidebar_position: 8
title: "MCP Agent Instructions"
description: "Drop-in instruction set for a Claude Code agent operating the miStudio MCP server"
---

# Agent Instructions: Operating miStudio via MCP

This page is written **for the agent, not the human**. Operators: paste it into your agent's context (e.g. a `CLAUDE.md`, a system prompt, or just tell the agent to fetch this URL) after connecting it to the miStudio MCP server. Everything below assumes the `mistudio` MCP server is connected and authenticated.

```bash
# Operator setup (once):
claude mcp add --transport http mistudio http://<host>:8765/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

---

## 1. What You Are Connected To

miStudio is a mechanistic-interpretability workbench. The objects you operate on:

| Object | What it is | Key id format |
|--------|-----------|---------------|
| **Extraction** | One harvest of features from a trained SAE — your top-level working set | `extr_…` |
| **Feature** | One SAE latent: a candidate concept, with top activating examples as evidence | `feat_…` |
| **Group** | Features sharing a normalized top activating token with similar context — a candidate concept *cluster* | UUID |
| **Steering task** | A GPU generation run that causally tests a feature by injecting it at a strength | UUID |
| **Experiment** | A saved steering result — durable evidence you can cite in labels | `exp_…`/UUID |

Your job is usually some slice of this loop:

```
find features (search / groups / by-token)
  → interpret (examples, token stats, logit lens)
    → validate causally (steering sweep/compare)
      → write back labels with evidence (update_feature_label + save_experiment)
```

A label is a **hypothesis**; steering is the **proof**. Prefer steering-validated claims over correlation-only claims, and say which kind you're making.

## 2. Golden Rules

1. **Never enumerate everything.** List tools cap at `limit ≤ 100` and return `total` counts. Narrow with filters/search instead of paging through tens of thousands of features. Fetch detail only for features you're actively analyzing.
2. **Start-tools return ids; poll — don't re-submit.** Any tool that starts work returns a `task_id` (or `run_id`/`approval_request_id`). Poll with the matching status tool. Ids are durable database records: they survive your disconnects, so record them and resume.
3. **Poll with backoff.** Grouping on a large extraction takes minutes; steering generations take tens of seconds to minutes. Poll every 10–30 s, not in a tight loop.
4. **GPU hygiene.** `enter_steering_mode` loads a model onto a shared GPU. Enter once, run your batch of steering calls, then **always `exit_steering_mode`** when done — even after errors.
5. **Respect guardrails, don't fight them.** Concurrency caps, token ceilings, protected labels, and approval mode are operator policy. When blocked, do what the error's `hint` says or report to the user — never retry-loop against a guard.
6. **Preserve provenance.** Your label writes are automatically tagged `mcp_agent`. Always attach evidence to notes (format in §6). Never overwrite a human's work silently — the 409 guard exists for exactly this.
7. **End with a summary.** Finish sessions with a compact report: features examined, groups found, experiments run (with ids), labels changed (before → after), and anything you could not validate.

## 3. Tool Reference

Categories can be disabled by the operator — if a tool is absent from your tool list, that capability is off; don't ask for workarounds.

### Discovery & feature reading (`read`)
| Tool | Use for |
|------|---------|
| `list_extractions` / `get_extraction_summary` | Find your working set. Prefer `status: completed` extractions with high `features_extracted` |
| `list_trainings` | SAE training runs behind extractions |
| `search_features(extraction_id, search=, category=, is_favorite=, sort_by=…)` | Targeted feature lookup. `search` matches name/description |
| `get_feature(feature_id)` | Label, category, description, notes, stats, `star_color`, `label_source` |
| `get_feature_examples(feature_id, limit=20)` | **Primary evidence** — top activating snippets with per-token activations. Read these before believing any label |
| `get_feature_token_analysis` / `get_feature_logit_lens` / `get_feature_nlp_analysis` | Aggregated token stats · promoted/suppressed vocabulary · stored NLP analysis (may be empty) |
| `get_feature_correlations` / `get_feature_ablation` | Slower analyses; cached server-side after first call |

### Cross-feature clustering (`groups`)
| Tool | Use for |
|------|---------|
| `compute_feature_groups(extraction_id, force=False)` | Build the token→feature index (background job, CPU-only, minutes). Idempotent: re-running with same params short-circuits |
| `get_grouping_status(extraction_id)` | `none` → not built (build it) · `computing` → poll · `completed` → proceed · `failed` → report `error_message` |
| `get_feature_groups(extraction_id, sort_by="size"\|"cohesion"\|"token", search=, token=)` | Browse clusters. **Cohesion** = context similarity within the group (≥0.5 tight, ≤0.3 loose — treat loose groups as token coincidences until proven otherwise) |
| `get_feature_group_members(extraction_id, group_id, has_label=, star_color=)` | Members with live labels + a `context_snippet` (`prefix *token* suffix`) each |
| `find_features_by_token(extraction_id, token, match="normalized")` | All features firing on a token. `normalized` ignores case/BPE markers (`▁`, `Ġ`, `##`); `exact` matches the raw surface form; `prefix` for stems |
| `find_related_features(feature_id)` | Seed-feature expansion. Each result carries `link_types`: `shared_token` / `context` / `correlation` — weight `context`+`shared_token` links above bare `correlation` |

### Steering (`steering`) — GPU-heavy
| Tool | Use for |
|------|---------|
| `steering_status` / `get_steering_mode` | Health + circuit-breaker state; whether the model is loaded |
| `enter_steering_mode` / `exit_steering_mode` | Load/free the GPU. Enter before a steering batch; exit after |
| `steer_compare(sae_id, prompt, features=[{feature_idx, layer, strength}], …)` | 1–4 features side-by-side vs unsteered baseline |
| `steer_sweep(sae_id, prompt, feature_idx, layer, strength_values=[…])` | **The causal validator**: one feature across strengths. Default recipe: `[0, 10, 30]`; add `-20` to test suppression |
| `steer_combined(…)` | All features applied simultaneously (group-synergy test) |
| `get_steering_result(task_id)` | Poll; releases your concurrency slot on terminal states |
| `cancel_steering_task(task_id)` | Free a stuck slot |

**Strength semantics** (raw residual-stream coefficients, ±300 max): 0 = baseline · 5–20 = start here · 50–100 = strong · >100 = usually degrades into repetition (that's evidence of *influence*, not of *meaning*). Negative = suppression. `max_new_tokens` is clamped to the operator's ceiling — don't request more.

### Experiments (`experiments`), labeling (`labeling`), jobs (`jobs`)
| Tool | Use for |
|------|---------|
| `save_experiment(name, experiment_type, data, description)` | Persist a validated steering result BEFORE citing it in labels |
| `list_experiments` / `get_experiment` | Retrieve prior evidence |
| `update_feature_label(feature_id, name=, category=, description=, notes=, override_protected=False)` | Write back labels (§6 rules) |
| `run_enhanced_labeling(feature_id)` → `get_enhanced_label(feature_id)` | Trigger the two-pass LLM labeler (background job; requires operator-configured labeling backend) |
| `get_task_status(task_queue_id)` or `get_task_status(list_active=True)` | Generic background-job polling (e.g. grouping jobs) |

## 4. Standard Workflows

### W1 — Orient in a fresh session
```
list_extractions(status_filter="completed")
get_extraction_summary(<newest or user-specified>)
get_grouping_status(extraction_id)        # build index if "none"
```

### W2 — Find and interpret a concept cluster
```
compute_feature_groups(extraction_id)               # if status was "none"
get_task_status(list_active=True) … until done      # poll 15–30 s
get_feature_groups(extraction_id, sort_by="cohesion", min_group_size=3)
get_feature_group_members(extraction_id, group_id)
for 3–5 representative members:
    get_feature_examples(feature_id, limit=10)
    get_feature_logit_lens(feature_id)
→ write a one-sentence hypothesis for the group's shared meaning
```
Interpretation heuristics: the examples' *contexts* matter more than the shared token itself; check whether unlabeled members (`has_label=false`) fit the pattern of labeled ones; a high-cohesion group whose members have inconsistent existing labels is a relabeling opportunity.

### W3 — Causally validate the hypothesis
```
get_steering_mode → enter_steering_mode (if not active)
steer_sweep(sae_id, neutral_prompt, feature_idx, layer, strength_values=[0, 10, 30])
get_steering_result(task_id) … poll
# judge: does output shift toward the hypothesis as strength rises, monotonically?
# optional: steer_combined on 2–4 group members for synergy
save_experiment(name="<concept> validation", experiment_type="sweep", data=<result summary + task_id>)
exit_steering_mode   # ALWAYS
```
Use a **neutral prompt** the concept could plausibly bend (e.g. `"The report said that"`), not one already containing the concept. Confirmed = dose-dependent shift in the hypothesized direction; if outputs shift somewhere else, that *is* the finding — revise the hypothesis, don't force it.

### W4 — Write back with evidence
```
update_feature_label(
  feature_id,
  name="expressions of affection",
  description="Fires on affection-context 'love/heart' tokens; steering shifts tone warmly (dose-dependent).",
  notes="<existing notes>\n[MCP 2026-07-12] evidence: experiment <id> — sweep 0/10/30 showed monotonic affection shift; baseline neutral."
)
```
Repeat W2→W4 across groups. Then produce the session summary (§2 rule 7).

## 5. Async Pattern Cheat Sheet

| You called | Poll with | Terminal states |
|------------|-----------|-----------------|
| `compute_feature_groups` | `get_grouping_status(extraction_id)` | `completed` / `failed` |
| `steer_*` | `get_steering_result(task_id)` | `completed` / `failed` / `cancelled` |
| `steer_*` under approval mode | `get_approval_status(approval_request_id)` → then `get_steering_result(steering_task_id)` | `approved` / `denied` / `expired` |
| `run_enhanced_labeling` | `get_enhanced_label(feature_id)` | job status inside the response |
| anything else with a task id | `get_task_status(task_queue_id)` | `completed` / `failed` |

## 6. Label Write-Back Rules

- **Provenance is automatic** — your edits are stored as `label_source: mcp_agent`. Don't claim to be a human.
- **Evidence convention** (append, never replace, existing notes):
  `[MCP <YYYY-MM-DD>] evidence: experiment <id> — <one-line summary>`
- **Aqua-starred features are protected** (completed enhanced labels curated by the researcher). Editing their name/category/description returns `409 PROTECTED_LABEL`. Policy:
  1. Default: leave the label; append your findings to `notes` only (notes are always allowed).
  2. Use `override_protected=True` **only** when you hold steering-validated evidence that the existing label is wrong, and say so explicitly in the notes.
- Only claim causation for steering-validated labels; phrase correlation-only labels tentatively ("appears to…", "fires on…").

## 7. Error Handling

| Error | Meaning | Correct reaction |
|-------|---------|-----------------|
| `401` | Bad/missing bearer token | Stop; tell the operator to fix the connection config |
| `409 ALREADY_COMPUTING` | Grouping run in flight | Poll status; don't resubmit |
| `409 NO_INDEX` | by-token/groups queried before index exists | `compute_feature_groups` first |
| `409 PROTECTED_LABEL` | Aqua-starred feature | Follow §6 policy |
| `Guardrail: N steering task(s) already in flight` | Concurrency cap | Poll/cancel existing tasks; don't queue more |
| `pending_approval` response | Operator-approval mode is on | Poll `get_approval_status`; if it stays pending, tell the user a human must approve in the Steering panel |
| `backend 503 …` | Dependent service down (e.g. labeling LLM has no model loaded) | Report the detail verbatim; skip that capability, continue elsewhere |
| Steering circuit-breaker open (via `steering_status`) | Prior GPU failures | Report; the reset path is in the status payload — don't hammer steering |
| `backend unreachable` | miStudio backend down | Stop and report; nothing will work |

## 8. Context-Window Discipline

- Summarize as you go; keep raw tool outputs out of your final report.
- For a 30k-feature extraction, work group-by-group (50 groups/page), reading full examples for at most a handful of members per group.
- When the user asks something global ("what concepts exist here?"), answer from `get_feature_groups` summaries + `total` counts — never by fetching all features.

## 9. Worked Micro-Example

> **User:** "Find a solid concept cluster in the newest extraction and validate it."

```
1. list_extractions(status_filter="completed", limit=5)        → pick extr_A (32k features)
2. get_grouping_status(extr_A)                                 → "completed" (2,528 groups)
3. get_feature_groups(extr_A, sort_by="cohesion", min_group_size=5)
                                                               → "copyright" ×29, cohesion 0.68
4. get_feature_group_members(extr_A, <group_id>)               → snippets all attribution-like
5. get_feature_examples on 3 members + logit_lens on 1         → promotes ©, "rights", "reserved"
   Hypothesis: "source-attribution / rights boilerplate"
6. enter_steering_mode
7. steer_sweep(sae_id, "The article about the festival", idx=17204, layer=12,
               strength_values=[0, 10, 30])                    → task_id T
8. get_steering_result(T) … completed → 30-strength output injects attribution phrasing
9. save_experiment("copyright-attribution validation", "sweep", {...})  → exp_9f2
10. exit_steering_mode
11. update_feature_label(3 unlabeled members, name="source attribution boilerplate",
    notes="[MCP 2026-07-12] evidence: experiment exp_9f2 — dose-dependent attribution phrasing")
12. Report: 1 group validated (29 members, 3 relabeled, evidence exp_9f2); labeled
    members left untouched (consistent with hypothesis).
```

---

*Operator-facing setup, configuration, and guardrail policy: [MCP Server](/advanced/mcp-server). REST equivalents of every tool: [API Reference](/reference/api/features-labeling).*
