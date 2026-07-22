<!-- GENERATED FILE — DO NOT EDIT BY HAND.

Regenerate with:
    python -c "from src.mcp_server.contract import write_contract; write_contract()"

`backend/tests/unit/test_mcp_contract_generated.py` regenerates this and fails
if it differs from what is committed, so an edit here is reverted by the next
run rather than silently kept.
-->

# miStudio MCP contract

Every tool the miStudio MCP server registers, its category, and the backend
endpoint it calls. Derived from the live registry — see
`src/mcp_server/contract.py`.

**Agents: call `mistudio_howto` instead of reading this.** A table cannot carry
ordering constraints, GPU-lock contention, id namespaces, or the failure modes
that mislead. This document is the inventory; that tool is the guidance.

Categories are gated by `MCP_TOOL_CATEGORIES`. The `millm_*` categories also
require `MILLM_API_URL` and are never enabled by default.


**97 tools across 13 categories.**


## `admin` (2 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `delete_experiment` ⚠️ | `DELETE /steering/experiments/{…}` | DESTRUCTIVE: permanently delete a saved steering experiment. |
| `delete_extraction` ⚠️ | `DELETE /extractions/{…}` | DESTRUCTIVE: permanently delete an extraction job AND every feature, label, and activation example derived from it. |

## `circuits` (24 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `build_circuit_from_discovery` | `POST /circuit-discovery/{…}/build-circuit` | Build a circuit from a discovery run's candidates — PREFER THIS over create_circuit when the circuit came from discovery. |
| `calibrate_circuit_strength` | `POST /circuits/{…}/calibration` | Calibrate a circuit's usable steering STRENGTH and clamp its served dial to it (Feature 20). |
| `create_circuit` | `POST /circuits` | Create a circuit from hand assembly. |
| `delete_circuit` ⚠️ | `DELETE /circuits/{…}` | Delete a circuit permanently (its manifests survive — they are first-class records). |
| `export_circuit_definition` | `GET /circuits/{…}/export` | Export the portable circuit-definition JSON (lossless: rungs, edge types, attribution, manifest refs, provenance all travel). |
| `export_circuit_slices` | `POST /circuits/{…}/export-slices` | Export per-layer cluster-definition/v1 slices (BR-014) for today's single-SAE consumers (miLLM). |
| `get_circuit` | `GET /circuits/{…}` | One circuit: members by layer, typed edges with full evidence (statistics, attribution, validation manifest refs), budget, faithfulness, rung + rung_language +  |
| `get_discovery_results` | `GET /circuit-discovery/{…}` | A discovery run + its report (null-model summary, FDR discipline, held-out replication RATE, stage counts, caps, uncovered seeds, lag-0 disclosure) + ranked can |
| `get_steering_samples` | `GET /validation-manifests`<br>`GET kind`<br>`GET manifests` | Fetch recorded steering-sample transcripts for a circuit: per prompt, the unsteered baseline output plus the steered output at each recorded dial — the raw mate |
| `get_validation_manifest` | `GET /validation-manifests/{…}` | A validation manifest — the SELF-CONTAINED, reproducible record of a validation run (intervention config, baseline, prompts, seeds, null summary, per-edge effec |
| `import_circuit_definition` | `POST /circuits/import` | Import a mistudio.circuit-definition/v1 document (the BR-013 round-trip). |
| `list_circuit_captures` | `GET /circuit-capture` | List capture runs (status, corpus, layers, split, size, stale flag). |
| `list_circuits` | `GET /circuits` | List circuits with rung + rung_language on every row. |
| `list_validation_manifests` | `GET /validation-manifests` | List validation manifests for a discovery run or a circuit. |
| `promote_circuit` | `POST /circuits/{…}/promote` | Promote a circuit into a loadable multi-layer steering profile — or unpromote it (promoted=false). |
| `record_steering_samples` | `POST /circuits/steering-samples` | Record (dial, prompt, unsteered_output, steered_output) transcripts on the GPU for a circuit, cluster, or ad-hoc feature set — the raw material for a SEPARATE,  |
| `reproduce_calibration` | `POST /circuits/calibration-manifests/{…}/reproduce` | Re-run a calibration from its manifest — the SAME probes and seed — and record a reproduction manifest with the band-delta verdict (does the re-measured onset/s |
| `reproduce_validation` | `POST /validation-manifests/{…}/reproduce` | Re-execute an edge_batch manifest from its payload and compare — the test that a rung-2 claim is reproducible, not a one-off. |
| `run_attribution_pass` | `POST /circuit-discovery/{…}/attribution` | Tier-2 gradient-attribution pass over a discovery run's candidates: re-ranks the shortlist before 017's causal validation and gates rung-1 (attribution_supporte |
| `run_circuit_discovery` | `POST /circuit-discovery` | Mine a completed capture store for candidate cross-layer edges. |
| `run_circuit_faithfulness` | `POST /circuits/{…}/faithfulness` | Faithfulness-test a circuit (rung 3 — the HIGHEST tier): suppress its members and measure how much of the behavior they drive is NECESSARY (ablating them collap |
| `start_circuit_capture` | `POST /circuit-capture` | Start a circuit-capture run (per-token multi-layer SAE activations with FIRST-CLASS positions + error norms). |
| `update_circuit` | `PATCH /circuits/{…}` | Edit a circuit (rename, fix a narrative, drop a bad edge, adjust members, switch granularity) — the agent review loop is not create-only. |
| `validate_circuit_edges` | `POST /circuit-discovery/{…}/validate` | Causally validate the top-K edges of a discovery run (rung 2): suppress the upstream feature, run the model, measure the downstream effect size vs a shuffled-no |

## `experiments` (3 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_experiment` | `GET /steering/experiments/{…}` | Get one saved experiment with its full stored data. |
| `list_experiments` | `GET /steering/experiments` | List saved steering experiments (paginated, limit ≤ 100). |
| `save_experiment` | `POST /steering/experiments` | Save a steering result as a persistent experiment record. |

## `groups` (6 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `compute_feature_groups` | `POST /extractions/{…}/feature-groups/compute` | Start the grouping precompute job for an extraction (background job). |
| `find_features_by_token` | `GET /extractions/{…}/features/by-token` | Features whose top activating token matches. |
| `find_related_features` | `GET /features/{…}/related` | Features related to a seed feature via shared tokens, context overlap, and cached correlations. |
| `get_feature_group_members` | `GET /extractions/{…}/feature-groups/{…}` | Members of one group with current labels, stars, stats, and a context snippet each (labels are live — never stale). |
| `get_feature_groups` | `GET /extractions/{…}/feature-groups` | List feature groups (features sharing a top activating token with similar context). |
| `get_grouping_status` | `GET /extractions/{…}/feature-groups/status` | State of the grouping index: none \| pending \| computing \| completed \| failed, with progress, params, and counts. |

## `jobs` (1 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_task_status` | `GET /task-queue`<br>`GET /task-queue/active`<br>`GET /task-queue/{…}` | Poll background jobs. |

## `labeling` (3 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_enhanced_label` | `GET /features/{…}/label/enhanced/latest` | Latest enhanced-labeling job + synthesized label for a feature. |
| `run_enhanced_labeling` | `POST /features/{…}/label/enhanced` | Trigger two-pass enhanced LLM labeling for one feature (background job; uses the labeling backend configured in Settings). |
| `update_feature_label` | `PATCH /features/{…}` | Update a feature's label. |

## `millm_circuits` (16 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_circuit` | `POST /api/circuits/{…}/activate` | Serve a circuit. |
| `millm_circuit_claims` | `GET /api/circuits/claims` | Which circuit holds which LAYER. |
| `millm_circuit_sensing_clear` ⚠️ | `DELETE /api/circuit-sensing/events` | Delete recorded edge observations. |
| `millm_circuit_sensing_disable` | `POST /api/circuit-sensing/{…}/disable` | Disable edge sensing for a circuit. |
| `millm_circuit_sensing_enable` | `POST /api/circuit-sensing/{…}/enable` | Enable edge sensing for a circuit (persists; arms when it serves). |
| `millm_circuit_sensing_event` | `GET /api/circuit-sensing/events/{…}` | One edge observation in full: both endpoints with their layer, feature, position and activation, the observed token lag, and the context window. |
| `millm_circuit_sensing_events` | `GET /api/circuit-sensing/events` | Observed edge firings, newest first. |
| `millm_circuit_sensing_status` | `GET /api/circuit-sensing/status` | Edge-sensing runtime state for the armed circuit. |
| `millm_circuit_status` | `GET /api/circuits/active` | Which circuits are steering RIGHT NOW, as a list. |
| `millm_deactivate_circuit` | `POST /api/circuits/{…}/deactivate` | Stop serving a circuit and release its layer claims. |
| `millm_delete_circuit` ⚠️ | `DELETE /api/circuits/{…}` | Delete an imported circuit permanently. |
| `millm_export_circuit` | `GET /api/circuits/{…}/export` | The circuit's ORIGINAL document, byte-for-byte. |
| `millm_import_circuit` | `POST /api/circuits/import` | Import a `mistudio.circuit-definition/v1` document (INLINE only). |
| `millm_list_circuits` | `GET /api/circuits` | Imported circuits with their evidence rung. |
| `millm_release_circuit_claims` | `POST /api/circuits/claims/release` | Release ONE circuit's stuck layer claims. |
| `millm_set_circuit_intensity` | `PUT /api/circuits/active/intensity` | Dial the active circuit's global lambda. |

## `millm_clusters` (6 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_cluster` | `POST /api/clusters/{…}/activate` | Activate an imported cluster (applies all members at sign x strength x lambda; hard compatibility gate server-side). |
| `millm_deactivate_cluster` | `POST /api/clusters/{…}/deactivate` | Deactivate a cluster (clears its live steering). |
| `millm_export_cluster` | `GET /api/clusters/{…}/export` | Re-export a cluster's lossless original v1 document (byte-honest — survives unknown additive fields). |
| `millm_hub_search` | `GET /api/clusters/hub/search` | Search public Hugging Face cluster packs (repos tagged mistudio-cluster-definition), optionally narrowed to a base model. |
| `millm_import_cluster` | `POST /api/clusters/hub/import`<br>`POST /api/clusters/import` | Import a mistudio.cluster-definition/v1 into miLLM. |
| `millm_list_clusters` | `GET /api/clusters` | List clusters imported into miLLM: bound state, warnings, current lambda intensity, active flag. |

## `millm_runtime` (5 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_profile` | `POST /api/profiles/{…}/activate` | Activate a miLLM profile by id — replaces the live steering. |
| `millm_deactivate_profile` | `POST /api/profiles/{…}/deactivate` | Deactivate a miLLM profile (clears live steering when it is the active one). |
| `millm_list_profiles` | `GET /api/profiles` | List miLLM steering profiles (manual rows and imported clusters — source_kind discriminates). |
| `millm_set_intensity` | `PUT /api/clusters/active/intensity` | Set the ACTIVE cluster's persistent intensity dial (lambda). |
| `millm_status` | `GET /api/health/detailed` | miLLM runtime status in one call: loaded model, attached SAE, inference backend, circuit breakers, and the ACTIVE steering profile (id/name/source_kind/intensit |

## `millm_sensing` (5 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_sensing_config` | `PUT /api/sensing/{…}/config` | SET a cluster's sensing quorum (members that must co-fire for an event) — this tool WRITES config; use millm_sensing_status to read. |
| `millm_sensing_disable` | `POST /api/sensing/{…}/disable` | Disable sensing for a cluster (disarms live if armed). |
| `millm_sensing_enable` | `POST /api/sensing/{…}/enable` | Enable sensing for a cluster (persists; arms live when that cluster is active with an SAE attached). |
| `millm_sensing_events` | `GET /api/sensing/events` | Co-activation events newest-first. |
| `millm_sensing_status` | `GET /api/sensing/status` | Sensing runtime status: armed cluster, quorum/threshold mode, per-request overhead, retention limits, and which clusters have the persistent sensing toggle enab |

## `profiles` (4 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `export_cluster_definition` | `GET /cluster-profiles/{…}/export` | Export a profile as portable `mistudio.cluster-definition/v1` JSON — the consumer-neutral artifact (never contains secrets or local paths). |
| `get_cluster_profile` | `GET /cluster-profiles/{…}` | Get one cluster profile: members with tuned strengths, budget snapshot (013 allocation), narrative, provenance. |
| `list_cluster_profiles` | `GET /cluster-profiles` | List saved cluster profiles (newest first), optionally filtered by SAE or a name/display-token substring. |
| `save_cluster_profile` | `POST /cluster-profiles` | Save a cluster profile (durable, decoupled from recomputable groups). |

## `read` (12 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_extraction_summary` | `GET /extractions/{…}` | Get one extraction job's detail: SAE, status, feature counts, config. |
| `get_feature` | `GET /features/{…}` | Full feature detail: label, category, description, notes, statistics, star state. |
| `get_feature_ablation` | `GET /features/{…}/ablation` | STATISTICAL-ESTIMATE ablation impact for a feature (method= "statistical_estimate" in the response) — scored from activation frequency/magnitude/consistency, NO |
| `get_feature_correlations` | `GET /features/{…}/correlations` | Features correlated with this one (token overlap + activation-stat similarity). |
| `get_feature_examples` | `GET /features/{…}/examples` | Top activating examples with per-token activations — the primary evidence for what a feature detects. |
| `get_feature_logit_lens` | `GET /features/{…}/logit-lens` | Logit-lens: vocabulary tokens this feature promotes/suppresses when active. |
| `get_feature_nlp_analysis` | `GET /features/{…}/nlp-analysis` | Stored NLP analysis (prime-token distribution, POS tags, context patterns, semantic clusters), if it has been computed for this feature. |
| `get_feature_token_analysis` | `GET /features/{…}/token-analysis` | Aggregated token statistics for a feature (ranked token frequencies). |
| `list_extractions` | `GET /extractions` | List feature-extraction jobs (the entry point for feature analysis). |
| `list_trainings` | `GET /trainings` | List SAE training runs (id, model, status, hyperparameters, metrics). |
| `mistudio_howto` | — | START HERE for circuit/steering work — workflow guidance an agent cannot infer from tool signatures. |
| `search_features` | `GET /extractions/{…}/features` | Search/filter features within an extraction. |

## `steering` (10 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `cancel_steering_task` | `DELETE /steering/async/task/{…}` | Cancel a running steering task and free its guardrail slot. |
| `compute_cluster_allocation` | `POST /steering/cluster-allocation` | Compute the principled starting strength allocation for steering a CLUSTER of features (Feature 013) — or a MULTI-LAYER circuit (Feature 015). |
| `enter_steering_mode` | `POST /steering/enter-mode` | Enter steering mode — LOADS THE MODEL+SAE ONTO THE GPU. |
| `exit_steering_mode` | `POST /steering/exit-mode` | Exit steering mode and free the GPU memory it held. |
| `get_steering_mode` | `GET /steering/mode` | Is steering mode active (model+SAE pre-loaded on the GPU)? |
| `get_steering_result` | `GET /steering/async/result/{…}`<br>`GET status` | Poll an async steering task. |
| `steer_combined` | `POST /steering/async/combined` | Apply ALL selected features simultaneously in one generation pass (synergy testing). |
| `steer_compare` | `POST /steering/async/compare` | Run a steering comparison: 1-4 features, each generating steered output side-by-side with an unsteered baseline. |
| `steer_sweep` | `POST /steering/async/sweep` | Dose-response sweep: one feature generated at each strength value (e.g. |
| `steering_status` | `GET /steering/status` | Steering service health, circuit-breaker state, and reset path. |

⚠️ = destructive or irreversible.
