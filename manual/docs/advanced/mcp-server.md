---
sidebar_position: 7
title: "MCP Server"
description: "Agentic access — let Claude Code and other MCP clients drive miStudio"
---

# MCP Server: Agentic Access

miStudio ships an optional **MCP (Model Context Protocol) server** that exposes the post-extraction workflow — feature analysis, [cross-feature grouping](/core-workflow/feature-groups), steering, and label write-back — as tools for agentic AI clients like Claude Code. An agent can run the full *analyze → group → steer → relabel* loop autonomously, with every action flowing through the same REST API and appearing in the UI like any other work.

## Enabling the Server

The MCP server is **off by default** and runs as its own container.

**Docker Compose:**

```bash
# 1. Set a token in .env (required — the port is LAN-reachable)
echo "MCP_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env

# 2. Start with the mcp profile
docker compose --profile mcp up -d
```

**Kubernetes:** add `mcp-auth-token` to the `mistudio-secrets` Secret and apply the `mistudio-mcp` Deployment/Service included in `k8s/mistudio-deployment.yaml`. The Service is ClusterIP-only — no Ingress is shipped; expose it deliberately (e.g. `kubectl port-forward svc/mistudio-mcp 8765:8765`).

**Verify:** `curl http://<host>:8765/health` returns the enabled tool categories.

:::warning Network exposure
The server binds `0.0.0.0:8765` so agents on other LAN machines can connect (bearer token always required). If your agents run on the same host only, firewall port 8765.
:::

## Connecting a Client

**Claude Code:**

```bash
claude mcp add --transport http mistudio http://<host>:8765/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

**Any MCP client:** streamable-HTTP transport, URL `http://<host>:8765/mcp`, header `Authorization: Bearer <token>`. A stdio mode exists for local development: `python -m src.mcp_server --stdio` (inside the backend container/venv).

## Configuration

| Env variable | Default | Purpose |
|--------------|---------|---------|
| `MCP_AUTH_TOKEN` | *(required)* | Bearer token; startup refused if empty |
| `MCP_TOOL_CATEGORIES` | `read,groups,steering,labeling,experiments,jobs` | Which tool categories are exposed; add `admin` to enable destructive deletes |
| `MCP_STEERING_MAX_CONCURRENT` | `2` | Max in-flight agent steering tasks |
| `MCP_STEERING_MAX_NEW_TOKENS` | `512` | Ceiling on generation length for agent steering |
| `MCP_STEERING_APPROVAL` | `false` | Route agent steering through operator approval (see below) |

Disabled categories simply don't appear in the agent's tool list. Disabling the whole server (omit the compose profile / scale the deployment to 0) never affects the frontend or REST API.

## Tool Catalog (33 tools)

| Category | Tools |
|----------|-------|
| **read** | `list_extractions`, `get_extraction_summary`, `list_trainings`, `search_features`, `get_feature`, `get_feature_examples`, `get_feature_token_analysis`, `get_feature_logit_lens`, `get_feature_correlations`, `get_feature_ablation`, `get_feature_nlp_analysis` |
| **groups** | `compute_feature_groups`, `get_grouping_status`, `get_feature_groups`, `get_feature_group_members`, `find_features_by_token`, `find_related_features` |
| **steering** | `steering_status`, `get_steering_mode`, `enter_steering_mode`, `exit_steering_mode`, `steer_compare`, `steer_sweep`, `steer_combined`, `get_steering_result`, `cancel_steering_task` (+ `get_approval_status` in approval mode) |
| **experiments** | `save_experiment`, `list_experiments`, `get_experiment` |
| **labeling** | `update_feature_label`, `run_enhanced_labeling`, `get_enhanced_label` |
| **jobs** | `get_task_status` |
| **admin** *(off by default)* | `delete_experiment`, `delete_extraction` — destructive |

Long-running operations follow the platform's async pattern: the start tool returns a task id, and the agent polls `get_task_status` / `get_steering_result`. Task ids are durable database records, so they survive agent reconnects.

## Guardrails & Provenance

- **Label provenance:** agent label edits carry `label_source: mcp_agent`, so you can always tell agent work from human work.
- **Protected labels:** aqua-starred features (completed enhanced labels) return a 409 when an agent tries to edit their name/category/description; the agent must pass an explicit `override_protected` flag. Notes remain appendable — the documented convention is `[MCP <date>] evidence: experiment <id> — <summary>`.
- **Steering limits:** concurrency cap and generation-length ceiling are enforced before the GPU is touched.
- **Operator-approval mode:** with `MCP_STEERING_APPROVAL=true`, agent steering calls become pending requests instead of running. They appear as a banner in the **Steering panel** with Approve/Deny buttons; on approval the backend submits the stored request itself, and the agent's poll picks up the resulting task id.
- **Audit:** every tool call is logged (tool name, argument digest, status, duration).

## Worked Example: the Analyze → Group → Steer → Relabel Loop

A Claude Code session pointed at the server, instructed with natural language only:

1. *"List extractions and build feature groups for the newest one"* → `list_extractions`, `compute_feature_groups`, `get_task_status` until complete
2. *"Show me the biggest groups"* → `get_feature_groups(sort_by="size")` — say it finds a 12-member `"love"` group with cohesion 0.81
3. *"What do the members have in common?"* → `get_feature_group_members`, then `get_feature_examples` + `get_feature_logit_lens` per member → agent hypothesizes "expressions of affection"
4. *"Validate that by steering"* → `enter_steering_mode`, `steer_sweep(feature_idx=4821, strength_values=[0, 10, 30])`, `get_steering_result` → steered outputs turn affectionate; baseline doesn't
5. *"Save the evidence and update the labels"* → `save_experiment`, then `update_feature_label(name="expressions of affection", notes="[MCP 2026-07-12] evidence: experiment exp_… — sweep shows dose-dependent affection shift")`
6. Every record — the group index, the experiment, the relabeled features with `mcp_agent` provenance — is immediately visible in the miStudio UI.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Server exits at startup: "MCP_AUTH_TOKEN is required" | Set the token in `.env` (or `MCP_ALLOW_ANONYMOUS=true` for stdio dev only) |
| 401 from every call | Client isn't sending `Authorization: Bearer <token>`, or tokens don't match |
| Tools error "backend unreachable" | The `mcp-server` container can't reach `backend:8000` — check both are on the same network and the backend is healthy |
| `steer_*` returns a guardrail message | Concurrency cap hit — poll or cancel existing tasks, or raise `MCP_STEERING_MAX_CONCURRENT` |
| Steering tools return `pending_approval` | Approval mode is on — approve the request in the Steering panel |
