# BRD — Multi-SAE Cross-Layer Steering & Automatic Circuit Discovery (miStudio increment)

> **Incremental BRD.** This is an enhancement BRD for the existing miStudio product, not a greenfield
> project BRD. It is the direct successor to `BRD-MIS-CLUSTERS-001` and consumes its deliverables (cluster
> profiles, the strength-budget model, and the `mistudio.cluster-definition/v1` interchange contract). It is
> the direct input to an update of the Project PRD (`000_PPRD|miStudio.md`) and to one or more new Feature
> PRDs. Per the XCC workflow, the next step is `0xcc/instruct/002_create-project-prd.md`.
>
> **Scope decision (locked):** This BRD covers the **miStudio-side** work only — multi-SAE cross-layer
> steering, the evidence-capture pass, automatic circuit discovery, circuit review/promotion, and the new
> portable `mistudio.circuit-definition/v1` contract with a per-layer projection to today's
> cluster-definition/v1. The downstream **miLLM multi-SAE serving runtime** (attaching multiple SAEs,
> serving circuits live, circuit-aware intensity dial, edge-level co-activation sensing) is captured here as
> **future considerations / vision** and will be specified in a **separate follow-on BRD**
> (BRD-MILLM-CIRCUITS-001 or similar) once this increment lands — mirroring how BRD-MIS-CLUSTERS-001
> deferred the cluster runtime that became BRD-MILLM-CLUSTERS-001. The projection export (BR-014) keeps
> today's single-SAE miLLM usefully in the loop meanwhile.
>
> **Locked decisions (product owner, 2026-07-18):** (1) causal validation is a **badge, not a gate** —
> circuits can be promoted to steering profiles before validation, prominently marked "unvalidated";
> (2) discovery ships with **both seeded mode** (mining around the existing curated cluster library) **and
> open-corpus mode** as first-class; (3) the user-facing term is **"Circuits"**, a new artifact class beside
> Clusters; (4) **miStudio-only** this increment, per the scope decision above.

```yaml
brd:
  metadata:
    brd_id: BRD-MIS-CIRCUITS-001
    project_name: "miStudio — Multi-SAE Cross-Layer Steering & Automatic Circuit Discovery"
    version: "0.1"
    author: "Sean (transcribed direction, 2026-07-18)"
    last_updated: "2026-07-18"
    status: "draft"
    increment_of: "miStudio (000_PPRD|miStudio.md)"
    successor_to: "BRD-MIS-CLUSTERS-001"

  business_context:
    problem_statement: >
      miStudio can steer a model with a tuned cluster of features — but only within one SAE at a time.
      Steering hooks are already registered per-layer and features carry a layer field, yet every hook
      shares the single loaded SAE's decoder, so cross-layer steering is an illusion: features placed on
      other layers would be steered with directions from the wrong layer's feature basis. Separately, every
      cluster in the library was hand-curated: a human searched labels, judged coherence, and assembled
      members within one extraction. The system has no way to FIND the cross-layer structure users actually
      care about — e.g. an L10 feature that reliably drives an L13 feature — because (a) extraction discards
      the per-token SAE activation matrix, keeping only top-k example snippets, so co-activation across
      layers cannot be computed from existing data; (b) the feature-correlations surface measures lexical
      similarity, not co-firing; and (c) the existing ablation surface fabricates its numbers from stored
      statistics without ever running the model, so causal claims cannot be tested. The unit of mechanistic
      meaning is the cross-layer circuit; today it can be neither discovered nor steered.
    vision_statement: >
      Make miStudio discover circuits, not just host them: the system automatically mines cross-layer
      feature relationships from real per-token activations, validates the promising ones with real causal
      interventions, and presents them as evidence-backed candidate circuits a user can review, name, and
      promote into loadable multi-layer steering profiles — steered with genuine multi-SAE combined
      influence and exported through the same portable interchange family that already carries clusters
      across the miStudio↔miLLM ecosystem.
    primary_objectives:
      - "Enable true multi-SAE steering: features from multiple layers, each applied through its own layer's SAE, in one generation."
      - "Extend the principled strength-budget model across layers with an honest v1: independent per-layer budgets governed by a single global intensity, preserving the 'never start from a useless point' guarantee."
      - "Build the missing evidence base: a per-token, multi-layer feature-activation capture pass over a shared evaluation corpus."
      - "Automatic circuit discovery: mine cross-layer co-activation (seeded around the existing cluster library AND open-corpus), rank candidates with independent evidence signals, and validate top candidates with real ablation."
      - "Retire fabricated causal claims: no heuristic estimate is ever presented as causal evidence."
      - "Make discovered circuits first-class artifacts: reviewable, promotable to steering profiles, portable via a new versioned interchange kind, and projectable to today's single-SAE consumers."
    success_criteria:
      - "A user can steer a multi-layer circuit in one generation and verify that every member feature at every layer contributed through its own layer's SAE."
      - "Selecting a circuit for steering yields an immediately-useful per-layer starting allocation with zero manual tuning, governed by one global intensity dial."
      - "The system proposes ranked candidate circuits automatically — in seeded mode (partners of existing curated clusters) and open-corpus mode — each disclosing the evidence behind it."
      - "Top-ranked candidate edges can be submitted to real ablation validation, and the survival rate is reported as a first-class number."
      - "A reviewed circuit can be named, given a narrative, promoted to a loadable steering profile (with a prominent 'unvalidated' badge when applicable), exported as a portable circuit definition, and re-imported with full fidelity."
      - "Today's miLLM can consume a per-layer projection of any circuit with zero runtime changes."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers / power users hunting mechanistic structure in the miStudio workbench."
      - "AI/agent operators (e.g. Claude Code via the MCP server) driving automated circuit discovery, validation, and promotion."
    secondary_users:
      - "Downstream miLLM operators who will serve circuits in a later increment (out of scope here, noted for contract design) and who can consume per-layer projections today."
      - "Future circuit-definition consumers/traders (marketplace vision — future)."
    stakeholders:
      - "Product owner (Sean) — direction and acceptance."
      - "miStudio maintainers."

  scope_definition:
    in_scope:
      - "Multi-SAE loading and combined steering in miStudio: features grouped by (SAE, layer), each layer's hook steering through its own SAE's decoder."
      - "Per-layer strength budgets computed by the established model (freq-budget/sim-alloc@1 per layer) + a single global intensity control + a new self-describing formula id for the multi-layer composition."
      - "Cross-layer interaction hazard detection (compounding and cancellation across layers), surfaced to the user — detection, not auto-correction."
      - "Per-token multi-layer feature-activation capture over a shared evaluation corpus: sparse/thresholded storage, user-configured layers/corpus/threshold, cost estimate before launch, managed as a long-running task."
      - "Automatic circuit discovery: co-activation mining over captured activations, in BOTH seeded mode (around existing curated cluster members) and open-corpus mode; ranking that combines co-activation statistics with a model-weight-derived cross-layer alignment prior; a triviality filter for residual-stream echo edges."
      - "Real causal edge validation: suppress the upstream feature in an actual forward pass, measure the downstream feature's response; validation status recorded per edge."
      - "Remediation of the existing heuristic ablation surface: relabeled or deprecated so no fabricated number is presented as causal evidence."
      - "Circuit review UX: full evidence disclosure (member labels/examples, co-activation statistics, weight-prior score, validation status), member editing, naming, narrative."
      - "Promotion of a reviewed circuit into a loadable multi-layer steering profile (validation badge, not gate)."
      - "New portable contract: mistudio.circuit-definition/v1 (new kind; per-layer SAE refs; members keyed to a layer/SAE; edges with evidence + validation status; discovery provenance; vendored JSON schema; lossless round-trip)."
      - "Projection export: render a circuit as one or more valid single-layer mistudio.cluster-definition/v1 artifacts, marked as partial renderings, so current single-SAE consumers work unchanged."
      - "MCP tool surface for the new capabilities (capture, discovery, validation, review, promotion, export) — discovery is an agent-native workflow."
    out_of_scope:
      - "Any miLLM code or runtime changes (multi-SAE attach, circuit serving, circuit-aware dial, edge-level sensing) — explicitly excluded; belongs to the follow-on BRD."
      - "Gradient-based attribution graphs / attribution patching (future tier; requires new backprop-through-SAE infrastructure)."
      - "Exhaustive all-pairs causal validation (combinatorially prohibitive; validation is sampled over top-ranked candidates)."
      - "Cross-model circuit portability (a circuit is bound to one model's SAE set this increment)."
      - "Automatic labeling/naming of circuits beyond the existing per-feature labels."
      - "Circuit marketplace / trading / commercialization (vision only)."
    future_considerations:
      - "Follow-on BRD (BRD-MILLM-CIRCUITS-001 or similar): miLLM multi-SAE serving runtime — attach multiple SAEs, import+serve circuit definitions, circuit-aware per-request intensity dial."
      - "miLLM edge-level co-activation sensing: extend cluster sensing to circuit EDGES (record when an upstream member firing is followed by its downstream partner), turning validated circuits into live monitors."
      - "Gradient-based attribution tier for discovery (attribution patching, integrated gradients through SAE codes)."
      - "Joint cross-layer budget calibration (empirical, γ-style fitting across layers) once per-layer v1 has field data."
      - "Circuit library / sharing / marketplace across the ecosystem."
      - "Seeding discovery from imported third-party cluster definitions, not just locally-curated ones."
    dependencies:
      - "Cluster profiles + strength-budget model + interchange contract (BRD-MIS-CLUSTERS-001 / Features 012–014)."
      - "Steering hook machinery with dynamic layer discovery (hooks already register per-layer; the SAE-per-hook change rides it)."
      - "Multi-layer raw activation capture infrastructure (activation service: per-layer memmap stores) and the layer-agnostic SAE encode path."
      - "Logit-lens weight access (basis for the W_dec(Li)·W_enc(Lj) cross-layer alignment prior)."
      - "Trained per-layer SAEs (L0–L15, 8k latents) with completed labeled extractions."
      - "Existing MCP server as the agent surface."
    assumptions:
      - "Per-layer SAEs share the residual stream at aligned token positions, so same-token (lag-0) alignment is a valid basis for cross-layer co-activation; small positional lags are a discovery-time refinement, not a blocker."
      - "Sparse above-threshold storage makes per-token capture tractable on the current single-GPU deployment; cost is bounded by user-selected corpus/layer scope, not unbounded by design."
      - "The existing 20-member cap is a workable default per layer; whether caps apply per layer or per circuit is an open question (see next_steps)."
      - "The evaluation corpus can be drawn from existing dataset management (the extractions' shared dataset is the natural default)."
      - "The 30 existing curated cluster profiles across L13/L14 are viable discovery seeds (known-good semantic territory)."

  business_requirements:
    - id: BR-001
      text: "The system SHALL support combined steering with features drawn from multiple SAEs at different layers simultaneously in a single generation, with each feature's influence applied through the SAE trained on its own layer. Steering configurations spanning layers SHALL NOT be silently served through a single SAE's decoder."
    - id: BR-002
      text: "For any multi-layer combined run, the user SHALL be able to verify that every member feature at every layer contributed its assigned strength through its own layer's SAE, and per-prompt results SHALL be labeled by the circuit/profile identity — extending the trust guarantees of BRD-MIS-CLUSTERS-001 (BR-002/BR-003) across layers."
    - id: BR-003
      text: "When a multi-layer circuit is selected for steering, the system SHALL compute a principled starting allocation with zero manual tuning required: a per-layer influence budget derived by the established strength model applied independently per layer, governed by a single global intensity control, with the composition method identified by a self-describing formula id. Joint cross-layer calibration is explicitly NOT required this increment."
    - id: BR-004
      text: "The system SHALL detect and surface cross-layer interaction hazards in a steering configuration — at minimum compounding (an upstream steered feature amplifying a downstream steered member, double-counting influence) and cancellation — so the user is warned before or during generation rather than surprised. Detection SHALL NOT silently alter the user's configuration."
    - id: BR-005
      text: "The system SHALL be able to capture per-token feature activations for a user-selected set of layers over a shared evaluation corpus and persist them in a form sufficient for cross-layer co-activation analysis. Capture SHALL be resource-bounded and configurable (layers, corpus, activation threshold), SHALL present a cost estimate before launch, and sparse above-threshold storage SHALL be the default so cost is a user decision, not an accident."
    - id: BR-006
      text: "Capture, discovery, and validation runs SHALL be long-running managed tasks: schedulable, progress-reporting, cancellable, and subject to the existing GPU guardrails, consistent with the established steering-mode discipline."
    - id: BR-007
      text: "The system SHALL automatically propose candidate cross-layer circuits (upstream→downstream feature chains or small graphs) from captured activations, ranked by strength of association, without requiring the user to hand-assemble members. Discovery SHALL offer two first-class modes: SEEDED (mining upstream/downstream partners of members of existing cluster profiles) and OPEN-CORPUS (unseeded mining across the captured layers)."
    - id: BR-008
      text: "Candidate ranking SHALL combine at least two independent evidence signals — statistical co-activation from real per-token data and a model-weight-derived cross-layer alignment prior — and each surfaced candidate SHALL disclose which signals support it and how strongly."
    - id: BR-009
      text: "The system SHALL validate candidate circuit edges by real causal intervention: suppressing the upstream feature in an actual forward pass and measuring the downstream feature's response. Heuristic or fabricated estimates SHALL NOT be presented as causal evidence anywhere in the product; the pre-existing heuristic ablation surface SHALL be relabeled as an estimate or deprecated within this increment."
    - id: BR-010
      text: "The system SHALL filter or de-rank trivial cross-layer edges — the same signal merely persisting through the residual stream — so that surfaced circuits represent meaningful feature interaction, and the filtering policy applied to each candidate SHALL be disclosed."
    - id: BR-011
      text: "Discovered circuits SHALL be presented for human review with their full evidence — member labels and examples, co-activation statistics, weight-prior score, and causal-validation status — and the user SHALL be able to edit members, name the circuit, and attach a narrative before promotion."
    - id: BR-012
      text: "The user SHALL be able to promote a reviewed circuit into a loadable multi-layer steering profile whose starting allocation is computed per BR-003 and which is immediately usable in multi-SAE combined steering (BR-001). Causal validation SHALL NOT gate promotion; a circuit not yet validated SHALL carry a prominent 'unvalidated' badge wherever it appears, upgraded when validation passes."
    - id: BR-013
      text: "Promoted circuits SHALL be exportable as a portable, standardized, versioned circuit definition (new kind: mistudio.circuit-definition/v1) carrying members with per-layer SAE references, edges with their evidence and validation status, and discovery provenance (method, corpus, thresholds, dates); the artifact SHALL round-trip through export→import with fidelity. The new kind SHALL NOT alter or break the existing mistudio.cluster-definition/v1 contract or its consumers."
    - id: BR-014
      text: "The system SHALL be able to project a multi-layer circuit into one or more valid single-layer mistudio.cluster-definition/v1 artifacts so that current single-SAE consumers (miLLM today) can use per-layer slices of a circuit without any runtime change, with each projection clearly marked as a partial rendering of the parent circuit."

  success_metrics:
    quantitative_metrics:
      - "100% of member features in a multi-layer combined run verifiably contribute through their own layer's SAE (extension of the existing member-contribution verification)."
      - "Causal survival rate: of the top-K mined candidate edges submitted to validation, the % whose downstream response to upstream suppression exceeds the agreed noise threshold is MEASURED AND REPORTED as a first-class number; ~50% for the top-ranked tier is the calibration target (a materially lower rate is a finding about the miner, not a hidden failure)."
      - "≥3 promoted circuits demonstrate dual-shift evidence: (a) a measurable shift in the downstream member's activation and (b) a measurable output change vs baseline on identical prompts via the compare workflow — matching the causal-validation bar set by the validated hand-curated profiles."
      - "Capture cost envelope: a full capture run over the reference corpus and chosen layer set completes within an agreed storage budget (GB) and wall-clock bound; actuals reported per run."
      - "Circuit-definition export→import round-trip reproduces member set, per-member strengths, edges, and provenance (semantic equality)."
      - "Contract safety: the existing cluster-definition/v1 validation suite passes unchanged; projected per-layer exports validate against the vendored v1 schema."
    qualitative_indicators:
      - "A reviewer can tell, for any surfaced circuit, WHY the system believes in it — the evidence disclosure feels sufficient to judge."
      - "Discovered circuits feel non-trivial — not the same feature echoing through adjacent layers."
      - "Multi-layer steering results feel as trustworthy as single-layer cluster results; no fabricated causal number remains presented as real."
      - "The 'unvalidated' badge is unmissable — a user cannot reasonably mistake a mined-but-unvalidated circuit for a causally validated one."
    measurement_methods:
      - "End-to-end multi-SAE steering verification (drive a multi-layer circuit run; confirm per-layer member application)."
      - "Discovery→validation pipeline run over the seeded and open modes with survival-rate reporting."
      - "Compare-workflow validation of ≥3 promoted circuits (activation shift + output shift)."
      - "Export→import round-trip test for circuit definitions; schema validation for v1 projections."
      - "Resource accounting on capture runs (storage + wall-clock vs estimate)."
      - "Product-owner acceptance on evidence disclosure, badge prominence, and result trust."

  feature_themes:
    core_features:
      - "Multi-SAE cross-layer combined steering (per-layer SAE application, member verification, circuit-titled results)."
      - "Per-layer budget composition + global intensity + cross-layer hazard detection."
      - "Per-token multi-layer activation capture (managed, sparse, cost-estimated)."
      - "Automatic circuit discovery (seeded + open-corpus co-activation mining, weight prior, triviality filter, ranked candidates with evidence disclosure)."
      - "Real ablation-based edge validation + heuristic-ablation remediation."
      - "Circuit review, naming, narrative, promotion to steering profiles (badge-not-gate)."
      - "mistudio.circuit-definition/v1 contract + vendored schema + per-layer v1 projection export."
      - "MCP tools for the full discovery→validation→promotion→export loop."
    secondary_features:
      - "Capture-artifact reuse across discovery runs on the same model/corpus."
      - "Discovery presets (recommended layer subsets, threshold defaults)."
      - "Circuit visualization (edge graph rendering) in the review UX."
    future_features:
      - "miLLM multi-SAE serving + circuit import + circuit-aware dial (follow-on BRD)."
      - "miLLM edge-level co-activation sensing on validated circuits."
      - "Gradient-based attribution discovery tier."
      - "Joint cross-layer budget calibration from field data."
      - "Circuit library / marketplace (vision)."

  considerations:
    budget_constraints: "TBD"
    timeline_expectations: >
      Incremental, sequenced: miStudio work first (this BRD) through the circuit-definition contract and
      projection export. The miLLM runtime follows in a separate BRD once this increment lands, mirroring
      the CLUSTERS-001 sequencing that worked.
    regulatory_or_policy_drivers: []
    technical_constraints:
      - "Single-GPU deployment (RTX 3090, 24GB): multi-SAE steering must load only the layers referenced by the active circuit; capture runs contend with steering for the GPU and must respect the existing guardrails."
      - "Per-token capture must default to sparse above-threshold storage; dense capture across 16 layers × 8k latents is out of budget by design."
      - "The strength model consumes already-available member stats; the multi-layer composition must remain self-describing (formula id) like freq-budget/sim-alloc@1."
      - "The additive-only evolution rule of the interchange family: circuit support arrives as a NEW kind, never as a mutation of cluster-definition/v1 (old consumers must cleanly reject unknown kinds rather than silently mis-serve)."
    integration_requirements:
      - "The circuit-definition contract must be designed for cross-product consumption (reviewed against the anticipated miLLM multi-SAE runtime before freezing), even though only miStudio produces/consumes it this increment."
      - "The per-layer projection must emit artifacts indistinguishable from ordinary v1 cluster definitions to today's consumers, apart from the partial-rendering marker carried in display-only metadata."
    scalability_expectations: >
      Discovery must remain tractable at 16 layers × 8k latents via seeding, thresholds, and sampled
      validation — never exhaustive pairwise search. Circuits may span multiple layers with up to the
      existing per-layer member cap; capture artifacts are reusable across discovery runs.

  risks:
    - id: RSK-001
      description: "Correlational false positives: co-activation mining proposes confounded, non-causal circuits; promoting them pollutes the profile library with untrue mechanistic claims."
      impact: "high"
      likelihood: "high"
      mitigation: "Badge-not-gate promotion keeps velocity while the mandatory prominent 'unvalidated' badge (BR-012), per-candidate evidence disclosure (BR-008/BR-011), and the first-class survival-rate metric keep claims honest."
    - id: RSK-002
      description: "Capture cost blowout: per-token capture across many layers over a meaningful corpus explodes GPU time and storage."
      impact: "high"
      likelihood: "medium"
      mitigation: "Sparse above-threshold storage by default, user-selected layer subsets and corpus size, pre-launch cost estimate (BR-005), retention policy, and the cost-envelope metric."
    - id: RSK-003
      description: "Trivial residual-stream echoes: adjacent-layer SAEs on a shared residual stream 'discover' the same feature persisting layer-to-layer; the top of the ranking is all identity edges."
      impact: "medium"
      likelihood: "high"
      mitigation: "BR-010 triviality filter (weight-prior + token-identity heuristics), semantic-distinctness cues in review, and tracking the filter's effect in metrics."
    - id: RSK-004
      description: "Cross-layer over-steering compounding: steering an upstream feature amplifies downstream circuit partners that are also being steered; influence double-counts and output quality collapses faster than single-layer intuition predicts."
      impact: "high"
      likelihood: "medium"
      mitigation: "BR-004 hazard detection, conservative default global intensity, per-layer budgets rather than a naively summed budget, compare-mode validation before promotion."
    - id: RSK-005
      description: "Contract churn for miLLM: a wrong contract call (mutating cluster-definition/v1 vs a new kind) either breaks the additive-only promise or strands the runtime."
      impact: "medium"
      likelihood: "medium"
      mitigation: "New kind + BR-014 projection means zero miLLM changes now; review the circuit schema against the anticipated miLLM multi-SAE runtime before freezing, as was done for v1."
    - id: RSK-006
      description: "Credibility debt from the existing fabricated ablation surface: left unlabeled next to real validation, it undermines trust in every causal claim the product makes."
      impact: "medium"
      likelihood: "high"
      mitigation: "BR-009's explicit prohibition; relabel the heuristic surface as an estimate or deprecate it in the same increment."
    - id: RSK-007
      description: "Multi-SAE resource ceiling: loading many SAEs simultaneously for steering may exceed VRAM/latency budgets on the current GPU deployment."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Load only the layers referenced by the active circuit; document a supported layer-count envelope; guardrail messaging when exceeded."
    - id: RSK-008
      description: "Discovery yield risk: at 8k latents on a 1.2B model, validated, interpretable, steerable circuits may be rare — the feature could ship 'working' but empty-feeling."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Seeded mode around the 30 curated profiles' members (known-good semantic territory) de-risks yield; success is framed on the N=3 promoted-circuit bar and honest survival-rate reporting, never volume promises."

  next_steps:
    open_questions:
      - "Edge-validation criterion: what downstream response counts as a validated edge — activation-delta threshold, effect size vs a measured noise floor, sign consistency — and what is the default?"
      - "Evaluation corpus: which corpus (the extractions' shared dataset? a new curated one?) defines co-activation ground truth, and at what size? The corpus biases every discovered circuit."
      - "Triviality policy nuance: is 'concept X persists L10→L13' ever a circuit worth surfacing (e.g. for robustness steering), or always noise to filter?"
      - "Member caps: does the existing 20-member cap apply per layer or per circuit for multi-layer profiles?"
      - "Global intensity semantics: does the 0–2.0 λ dial from the cluster world carry the same meaning applied across layers (relevant to the future miLLM circuit dial)?"
      - "Capture-artifact retention: how long are per-token capture stores kept, and under what policy are they invalidated (new SAE version, new corpus)?"
      - "Positional alignment: is lag-0 (same-token) co-activation sufficient for v1, or do small positive lags (upstream fires at t, downstream at t+1..k) need first-class support?"
    recommended_actions:
      - "Proceed to 002_create-project-prd.md: extend 000_PPRD|miStudio.md with the new feature rows/sections (multi-SAE steering, evidence capture + discovery, circuit review/promotion/portability), keeping the miLLM circuit-runtime vision visible."
      - "Update the ADR with: the multi-SAE steering decision, the per-layer budget composition (new formula id), the circuit-definition/v1 contract decision (new kind + projection), and the real-ablation/heuristic-remediation decision."
      - "Then author the Feature PRDs — likely: (a) multi-SAE cross-layer steering + budgets + hazards, (b) capture + automatic discovery + validation, (c) circuit review/promotion + circuit-definition contract + projection. Exact feature count to be finalized during PRD."
    priority_for_clarification:
      - "The edge-validation criterion and noise floor — highest leverage; blocks BR-009 and the survival-rate metric."
      - "The evaluation corpus choice — blocks BR-005/BR-007 and biases all discovery results."
      - "The circuit-definition/v1 schema (members/edges/provenance keying) — blocks BR-013/BR-014 and the follow-on miLLM arc."
```
