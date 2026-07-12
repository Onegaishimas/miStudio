# Business Requirements Document

## miStudio MCP Server: Agentic Access to Feature Analysis and Steering

| | |
|---|---|
| **Document ID** | BRD-MIS-MCP-001 |
| **Version** | 0.1 (Draft) |
| **Date** | July 11, 2026 |
| **Product** | miStudio (MechInterp Studio) |
| **Enhancement** | MCP Server for Agentic AI Integration |
| **Status** | Draft — for review and incorporation into project PRD |

---

## 1. Executive Summary

miStudio is a mature, end-to-end mechanistic interpretability platform with a well-documented REST API (`/api/v1`) that already exposes everything the frontend does: dataset management, SAE training, feature extraction, LLM-powered labeling, and causal steering. Today, the only clients of that API are the miStudio frontend and hand-written scripts.

This enhancement adds a **Model Context Protocol (MCP) server** to miStudio so that agentic AI coding assistants — Claude Code, Codex, and any other MCP-capable agent — can operate the miStudio backend directly. The primary use cases are:

1. **Cross-feature analysis** — querying extracted, labeled features at scale to find groups of features that are semantically linked (e.g., features that fire on the same top activating token with similar surrounding context), and reasoning about what those features mean.
2. **Agent-driven steering** — running steering experiments (compare, sweep, combined) programmatically to causally validate and refine feature interpretations, feeding results back into improved feature labels.

The strategic value is **iteration speed**: an agent can run hundreds of query → hypothesize → steer → validate → relabel loops in the time a human researcher runs a handful, turning miStudio from an interactive workbench into a platform for semi-autonomous interpretability research.

---

## 2. Background and Current State

### 2.1 What exists today

- **Backend API**: A full REST API rooted at `/api/v1`, with Swagger documentation, consistent conventions (202-accepted async job pattern, WebSocket progress channels, structured errors, pagination).
- **Feature analysis surface**: Per-feature endpoints for top activating examples with per-token activations, aggregated token statistics, logit-lens output, correlated features, ablation analysis, and NLP analysis.
- **Labeling surface**: Bulk LLM labeling jobs, per-feature two-pass enhanced labeling, and feature curation (edit name/category/description/notes, favorite, star).
- **Steering surface**: Async compare/sweep/combined generation with strength coefficients in [-300, +300], steering mode management (GPU pre-load), and saved experiments.
- **Data layer**: Every dataset, training run, feature, label, and steering experiment is a database record with full provenance.

### 2.2 The gap

There is no first-class way for an agentic AI to use miStudio. An agent could in principle call the REST API with raw HTTP, but this is slow to set up, error-prone, and forces the agent to rediscover conventions (async job polling, WebSocket channels, pagination) in every session. More importantly, the API is organized around single resources — there is no efficient primitive for the cross-feature correlation queries an agent needs ("give me all features whose top activating token is X, grouped by context similarity"), so an agent must page through thousands of features and reconstruct that view itself, which is prohibitively slow and token-expensive.

---

## 3. Business Objectives

| # | Objective | Rationale |
|---|-----------|-----------|
| BO-1 | Enable MCP-capable agents (Claude Code, Codex, etc.) to control miStudio's feature analysis and steering capabilities without custom HTTP scripting. | Removes integration friction; makes miStudio agent-native. |
| BO-2 | Reduce the time to discover and validate a group of related features from hours (manual UI workflow) to minutes (agent-driven loop). | Iteration speed is the core value proposition. |
| BO-3 | Improve feature label quality by closing the loop: analysis → hypothesis → steering validation → label refinement, executed autonomously. | Steering proves causation; agents can afford to steer every hypothesis. |
| BO-4 | Preserve full provenance for agent-initiated work, on par with UI-initiated work. | Auditability and reproducibility are existing platform commitments. |
| BO-5 | Keep the deployment self-hosted and local, consistent with miStudio's "your hardware, your data" posture. | No new external data flows introduced by this feature. |

---

## 4. Scope

### 4.1 In scope

1. An **MCP server component** shipped with miStudio (deployable alongside the existing Docker Compose / Kubernetes stack) that exposes backend capabilities as MCP tools.
2. **Read/query tools** over extractions, features, labels, token analysis, examples, logit-lens, correlations, and ablation data.
3. **New cross-feature analytics capability, delivered as first-class backend REST endpoint(s)** under `/api/v1`, supporting queries across all extracted features in an extraction or training: group by top activating token, filter by activation context similarity, filter by label/category/star status, and return feature clusters. This capability is a required deliverable of this enhancement — if the backend cannot answer these queries today, the endpoint(s) to enable it are in scope and must be built.
4. **Frontend enablement of cross-feature grouping**: a UI surface in the feature-analysis area that exposes the same grouping capability to human researchers (browse groups by top activating token, inspect group members, jump to feature detail and steering), so the capability is usable with and without an agent.
5. **Steering tools** wrapping the async steering API: compare, sweep, combined generation; steering mode enter/exit/status; task polling; saved experiment CRUD.
6. **Label write-back tools**: update feature name, category, description, and notes; trigger enhanced labeling; record steering-derived evidence in feature notes or experiment records.
7. **Long-running job handling** adapted to the MCP interaction model (agents poll; the server translates the 202 + WebSocket pattern into MCP-friendly status tools).
8. **Configuration, authentication/authorization, and safety controls** appropriate for exposing a GPU-backed backend to autonomous agents.
9. Documentation: MCP server setup guide, tool catalog, example agent workflows, new REST endpoint reference, and frontend usage documentation added to the existing manual.

### 4.2 Out of scope

- MCP tools for dataset upload, model management, activation extraction, or SAE training (candidate for a later phase; this phase targets the post-extraction workflow: analyze → correlate → steer → label).
- Any hosted/cloud MCP offering; the server runs locally with the rest of the stack.
- Frontend changes beyond the cross-feature grouping surface defined in Section 7.8 (agent-created records already appear in the UI because they are ordinary database records; no dedicated "agent activity" UI is required in this phase).
- Building or bundling a specific agent; the deliverable is the server, not the client.

---

## 5. Stakeholders and Users

| Role | Interest |
|------|----------|
| Interpretability researcher (primary user) | Delegates feature-group discovery and steering validation to an agent; reviews and approves resulting labels. |
| Agentic AI client (Claude Code, Codex, other MCP clients) | Consumes the MCP tool surface; the "user" from the protocol's perspective. |
| miStudio maintainers | Own the new component; need it to reuse existing services and conventions, not fork them. |
| Platform operator (self-hoster) | Needs simple deployment, clear resource controls (GPU contention), and the ability to restrict what agents may do. |

---

## 6. User Stories

**US-1 — Find linked feature groups.**
As a researcher using Claude Code, I want the agent to query all labeled features from an extraction and return groups of features that fired on the same top activating token with similar surrounding context, so that I can see candidate "concept clusters" without manually browsing thousands of features.

**US-2 — Interpret a feature group.**
As a researcher, I want the agent to pull each group member's top activating examples, token statistics, and logit-lens data, and synthesize a proposed shared meaning for the group.

**US-3 — Validate by steering.**
As a researcher, I want the agent to run steering comparisons and strength sweeps on candidate features (individually and combined) and evaluate whether the generated text shifts in the direction the hypothesized label predicts.

**US-4 — Refine labels with evidence.**
As a researcher, I want the agent to update feature names, descriptions, and notes with the steering-validated interpretation, and to save the supporting steering runs as experiments linked in the notes, so that provenance is preserved.

**US-5 — Iterate fast.**
As a researcher, I want the full loop (US-1 → US-4) to run over many groups in a single agent session with minimal human intervention, with a summary I can review at the end.

**US-6 — Stay in control.**
As an operator, I want to configure which tool categories the MCP server exposes (e.g., read-only vs. read/write vs. steering) and require confirmation or limits on GPU-consuming operations.

**US-7 — Explore groups in the UI.**
As a researcher working in the miStudio frontend, I want to browse the same cross-feature groups the agent uses — grouped by top activating token with context similarity — inspect group members, and jump directly to a member's feature detail or into steering, so that agent findings and my own exploration share one view of the data.

---

## 7. Business Requirements

### 7.1 MCP server foundation

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-1.1 | miStudio SHALL provide an MCP server that exposes backend capabilities as MCP tools, connectable by any MCP-compliant client (stdio and/or HTTP transport per MCP spec). | Must |
| BR-1.2 | The MCP server SHALL be deployable as part of the standard Docker Compose and Kubernetes deployments, and independently disableable. | Must |
| BR-1.3 | The MCP server SHALL communicate with the backend exclusively through the existing `/api/v1` REST API (or shared service layer), not by bypassing it, so that provenance, validation, and job-queue behavior are identical to UI-initiated actions. | Must |
| BR-1.4 | Tool descriptions SHALL be self-documenting (parameters, ranges, async behavior) so agents can operate correctly without out-of-band documentation. Documented parameter ranges (e.g., steering strength [-300, +300], max_new_tokens [1, 2048]) SHALL be enforced and surfaced in tool schemas. | Must |
| BR-1.5 | The MCP server SHALL translate the backend's async-job pattern (202 + WebSocket progress) into an agent-friendly pattern: start-tool returns a task/job ID; a status/result tool supports polling; errors return the backend's structured detail. | Must |

### 7.2 Feature query and cross-feature analysis

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-2.1 | Tools SHALL exist to list extractions, list/filter features within an extraction or training (by label presence, category, star/favorite status, activation statistics), and fetch full feature detail. | Must |
| BR-2.2 | Tools SHALL expose per-feature analysis data: top activating examples with per-token activations, aggregated token statistics, logit-lens promoted/suppressed tokens, correlated features, ablation analysis, and stored NLP analysis. | Must |
| BR-2.3 | A **cross-feature grouping capability** SHALL be provided: given an extraction (or training), return groups of features that share the same (or normalized-equivalent) top activating token, optionally refined by similarity of activation context (e.g., embedding or n-gram similarity of surrounding tokens in top examples). | Must |
| BR-2.4 | The grouping capability SHALL support querying by token (all features whose top activating token matches token X or pattern) and by feature (all features linked to feature Y through shared tokens/context and/or existing correlation data). | Must |
| BR-2.5 | The grouping capability SHALL be implemented as **new first-class REST endpoint(s)** under `/api/v1`, following existing API conventions (pagination, structured errors, async 202 pattern if precomputation is required), e.g., `GET /extractions/{id}/feature-groups`, `GET /extractions/{id}/features/by-token`, and `GET /features/{id}/related`. The MCP tools and the frontend SHALL both consume these same endpoints; the capability SHALL NOT exist only inside the MCP layer. | Must |
| BR-2.5a | The new endpoint(s) SHALL be documented in the API reference alongside the existing Features & Labeling endpoints and exposed in the Swagger UI. | Must |
| BR-2.5b | Queries SHALL be executed server-side and return compact, consumer-ready results — the design SHALL avoid forcing the agent (or frontend) to page through the full feature set to build groups itself. Where grouping requires precomputation (token→feature index, context embeddings), it SHALL run as a standard background job with progress reporting, with results persisted for reuse. | Must |
| BR-2.6 | Group results SHALL include, per member: feature ID, neuron index, current label/category, top token(s), representative context snippets, and summary activation statistics, plus a group-level cohesion measure. | Should |
| BR-2.7 | Results SHALL be pageable/limitable and bounded in size so responses fit agent context windows. | Must |

### 7.3 Steering

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-3.1 | Tools SHALL wrap async steering compare (one feature at multiple strengths, or several features side-by-side), sweep (dose-response across a strength range), and combined (multiple features simultaneously) generation, plus result polling and task cancellation. | Must |
| BR-3.2 | Tools SHALL manage steering mode: query state, enter, exit, and report steering service health; tool documentation SHALL make GPU-load implications explicit to the agent. | Must |
| BR-3.3 | Tools SHALL support saved-experiment CRUD so agents can persist validated steering results and reference them as evidence. | Must |
| BR-3.4 | The server SHALL apply configurable guardrails on steering usage by agents (e.g., max concurrent tasks, max_new_tokens ceiling, optional operator-approval mode) to protect shared GPU resources. | Should |

### 7.4 Labeling and write-back

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-4.1 | Tools SHALL allow updating a feature's name, category, description, and notes. | Must |
| BR-4.2 | Tools SHALL allow triggering per-feature enhanced labeling and retrieving its latest result, and (configurably) starting/monitoring bulk labeling jobs. | Should |
| BR-4.3 | Agent-written label updates SHALL be distinguishable in provenance (e.g., recorded source/attribution such as "MCP agent") and SHALL NOT silently overwrite protected curation states (e.g., aqua-starred completed enhanced labels), consistent with existing platform protections. | Must |
| BR-4.4 | A recommended workflow convention SHALL be documented for recording steering evidence alongside label changes (e.g., experiment IDs in feature notes). | Should |

### 7.5 Security, access control, and safety

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-5.1 | The MCP server SHALL support authentication for its transport (at minimum a locally configured token/secret) and SHALL NOT be exposed beyond the local deployment by default. | Must |
| BR-5.2 | Operators SHALL be able to configure tool exposure by category: read-only analysis; label write-back; steering execution; job control. Default SHOULD be read + steering with label write-back enabled, destructive operations (deletes) disabled. | Must |
| BR-5.3 | Destructive tools (deleting extractions, labeling jobs, experiments) SHALL be opt-in and clearly marked as destructive in tool metadata. | Must |
| BR-5.4 | All agent-initiated actions SHALL be logged with enough detail to audit what an agent session did. | Should |

### 7.6 Performance and reliability

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-6.1 | Read/query tools SHALL respond within interactive latencies for typical extractions (target: < 2 s for feature queries; < 10 s for cross-feature grouping over an extraction of ~50k features, with caching permitted). | Should |
| BR-6.2 | The MCP server SHALL degrade gracefully when backend dependencies are unavailable (e.g., labeling LLM not loaded → surface the backend's 503 detail; steering circuit-breaker open → surface status and the reset path). | Must |
| BR-6.3 | Long-running steering/labeling tasks SHALL remain trackable across agent reconnects (task IDs are durable backend records). | Must |

### 7.7 Documentation

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-7.1 | The manual SHALL gain an "MCP Server" section: installation/enabling, client configuration examples (Claude Code, generic MCP client), full tool catalog, and at least one end-to-end worked example of the analyze → group → steer → relabel loop. | Must |
| BR-7.2 | The API reference SHALL document the new cross-feature grouping endpoint(s) (BR-2.5), and the Core Workflow section SHALL document the new frontend feature-groups experience (Section 7.8). | Must |

### 7.8 Frontend enablement of cross-feature grouping

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-8.1 | The miStudio frontend SHALL provide a **Feature Groups** view within the feature-analysis workflow that consumes the new grouping endpoint(s) (BR-2.5) and displays groups of features sharing a top activating token with similar activation context. | Must |
| BR-8.2 | The view SHALL support the same query dimensions as the API: group by token, search by token/pattern, seed-feature "find related features," and filters for label presence, category, and star/favorite status. | Must |
| BR-8.3 | Each group SHALL display its shared token(s), a group cohesion measure, and member rows showing feature ID/index, current label, and representative context snippets; selecting a member SHALL navigate to the existing feature detail view. | Must |
| BR-8.4 | From a group (or selected members), the user SHALL be able to launch steering directly (pre-populating the selected features in the steering panel) to validate the group's hypothesized meaning. | Should |
| BR-8.5 | If grouping requires a precomputation job (BR-2.5b), the frontend SHALL surface job start, live progress (via the standard WebSocket pattern), and completion state, consistent with other background jobs in the UI. | Must |
| BR-8.6 | Groups discovered or annotated via MCP agents and via the UI SHALL be the same underlying records/queries — there SHALL be one source of truth, with no agent-only or UI-only grouping results. | Must |

---

## 8. Proposed Capability Map (illustrative, non-binding)

The following tool groupings illustrate scope; final tool names and schemas are a PRD/design concern.

- **Discovery/context**: `list_extractions`, `list_trainings`, `get_extraction_summary`
- **Feature query**: `search_features` (filters: label, category, star, activation stats), `get_feature`, `get_feature_examples`, `get_feature_token_analysis`, `get_feature_logit_lens`, `get_feature_correlations`, `get_feature_ablation`
- **Cross-feature analysis (new)**: `group_features_by_token`, `find_related_features` (seed feature → linked set via shared tokens/context/correlations), `compare_feature_contexts` — thin wrappers over the new REST endpoints required by BR-2.5, which also serve the frontend Feature Groups view (Section 7.8)
- **Steering**: `steering_status`, `enter_steering_mode`, `exit_steering_mode`, `steer_compare`, `steer_sweep`, `steer_combined`, `get_steering_result`, `cancel_steering_task`
- **Experiments**: `save_experiment`, `list_experiments`, `get_experiment`
- **Labeling**: `update_feature_label`, `run_enhanced_labeling`, `get_enhanced_label`, (configurable) `start_bulk_labeling`, `get_labeling_job`

---

## 9. Assumptions

1. The existing REST API is stable and sufficiently complete that most MCP tools are thin wrappers; the principal new engineering work is the cross-feature grouping endpoint(s) (BR-2.3–2.6) and the frontend Feature Groups view (Section 7.8).
2. Per-token activation data for top examples is already persisted per feature, making token/context grouping feasible without re-running extraction.
3. Deployments are single-tenant or trusted-team; fine-grained per-user authorization is not required in this phase.
4. Agents run on the same network as (or with secure access to) the miStudio deployment.

## 10. Constraints

1. GPU memory is shared between steering mode and other workloads; agent steering must respect the existing steering-mode lifecycle and resilience mechanisms.
2. MCP responses must be sized for LLM context windows; large analytical outputs need summarization/pagination at the server.
3. The solution must not send data off-host, consistent with miStudio's self-hosted posture.

## 11. Dependencies

- MCP specification and reference SDKs (server implementation).
- Existing miStudio backend services: features/labeling router, steering service, task queue, database.
- New backend analytics endpoint(s) or materialized query support for cross-feature grouping.

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Agents monopolize GPU via unbounded steering | Blocks human/UI use | BR-3.4 guardrails: concurrency caps, token ceilings, approval mode |
| Cross-feature grouping too slow at scale | Kills the iteration-speed objective | Server-side implementation with indexing/caching (BR-2.5); precompute per-extraction token→feature maps |
| Agent overwrites curated labels | Loss of researcher work | BR-4.3 protections + attribution; respect existing protected-star semantics |
| Tool schema drift vs. backend API | Broken agent sessions | Generate tool schemas from the same OpenAPI/source of truth where possible |
| Oversized responses blow agent context | Unusable tools | BR-2.7 pagination + summary-first response design |

## 13. Success Metrics

| Metric | Target |
|--------|--------|
| Time for an agent to produce a steering-validated label for a feature group (from cold start) | ≤ 10 minutes median on reference hardware |
| Cross-feature grouping query latency (50k-feature extraction) | ≤ 10 s (warm cache ≤ 2 s) |
| Agent workflow completion rate (analyze → group → steer → relabel loop without human intervention on tool errors) | ≥ 90% |
| Coverage: fraction of the target workflow (features/labeling + steering API surface) exposed as MCP tools | 100% of in-scope endpoints |
| Adoption signal | Documented, reproducible Claude Code and one other MCP client configuration |

## 14. Acceptance Criteria (business level)

1. From a fresh Claude Code session pointed at a configured miStudio MCP server, a user can, via natural-language instruction only: enumerate extractions; retrieve groups of features sharing a top activating token with similar context; inspect group members' examples and token statistics; run a steering sweep on a group member; and write an updated, evidence-annotated label — with all resulting records visible in the miStudio UI.
2. A user in the miStudio frontend can open the Feature Groups view for an extraction, browse groups by top activating token, filter by label/category/star, open a member's feature detail, and launch steering on selected group members — powered by the same REST endpoint(s) the MCP tools use.
3. Disabling the MCP server (or a tool category) via configuration takes effect without impacting the frontend or REST API — including the Feature Groups view, which depends only on the REST endpoint(s), not the MCP server.
4. All agent actions appear in logs/provenance with agent attribution.
5. Manual documentation exists and is sufficient for a new operator to enable and use the server without reading source code, and covers the new endpoint(s) and Feature Groups view.

## 15. Downstream Artifacts

Per the project's document chain, this BRD feeds:

1. **PRD amendment** — a new feature section ("MCP Server for Agentic Access") deriving product requirements, tool schemas, and UX/operator-experience decisions from Sections 7–8.
2. **Technical design** — MCP transport choice, backend analytics endpoint design (token→feature indexing, context-similarity method, precompute job), auth mechanism, deployment manifests.
3. **Frontend design** — Feature Groups view (Section 7.8): information architecture within the feature-analysis workflow, group/member presentation, steering hand-off interaction.
4. **Documentation plan** — new manual sections per BR-7.1 and BR-7.2.

---

## Appendix A — Terminology

| Term | Meaning in miStudio |
|------|---------------------|
| Feature | A learned SAE latent (neuron) representing a candidate concept in the subject model. |
| Extraction | A job that harvests features (and their activating examples) from a trained or external SAE. |
| Top activating token | The token on which a feature fires most strongly, taken from the feature's top activating examples with per-token activations. (Referred to informally as the "prime token" in the source request.) |
| Activation context | The surrounding tokens in the examples where a feature fires; used here as the similarity basis for grouping related features. |
| Labeling / Enhanced labeling | LLM-generated feature descriptions; enhanced labeling is a two-pass per-feature analysis (per-example summaries → synthesis). |
| Steering | Adding a feature's decoder direction into the residual stream at a chosen strength ([-300, +300]) during generation to causally test the feature's meaning. |
| MCP | Model Context Protocol — the open standard by which agentic AI clients (Claude Code, Codex, etc.) discover and call external tools. |
