# BRD — Circuit Discovery Rigor Supplement: Attribution, Faithfulness & Cross-Position Structure (miStudio increment)

> **Supplemental BRD.** This document amends and extends `BRD-MIS-CIRCUITS-001`. It does not replace it.
> The two documents are consumed **together** as the input to the Project PRD update: CIRCUITS-001 defines
> the feature surface (multi-SAE steering, capture, discovery, review, promotion, contract); this supplement
> (CIRCUITS-002) closes eight identified methodological gaps so the discovery pipeline produces circuits
> whose causal claims survive contact with the mechanistic-interpretability literature and with real
> inference-time use. Per the XCC workflow, the next step remains `0xcc/instruct/002_create-project-prd.md`,
> executed once against the union of CIRCUITS-001 + CIRCUITS-002.
>
> **Why a supplement exists:** an external methodology review (2026-07-18) of CIRCUITS-001 found the
> increment directionally correct but exposed to eight specific risks: (1) lag-0 same-token mining
> preferentially finds the trivial edges the triviality filter then removes, leaving little signal;
> (2) the weight prior is structurally biased toward residual echoes — the same edges BR-010 filters;
> (3) gradient attribution was mis-costed as heavy future infrastructure when it is one forward + one
> backward pass; (4) ablation semantics (suppression method, SAE error handling, baseline) were
> unspecified; (5) validation was edge-level only, with no circuit-level faithfulness check;
> (6) co-activation statistics lacked a null model and held-out replication; (7) mining ran at feature
> granularity where feature splitting fragments true circuits; (8) cross-layer hazard detection was
> unconnected to the validation data that quantifies the hazard. This BRD converts each finding into
> business requirements, resolves four open questions CIRCUITS-001 left open, and adds the evidence-ladder
> claims discipline that keeps the product honest as capability tiers accumulate.
>
> **Audience note (agentic implementation):** the primary implementer of this increment is an AI coding
> agent (Claude Code via the XCC chain). Appendix A is therefore normative supporting material, not
> commentary: it defines the mathematics, algorithms, and vocabulary the requirements refer to, so the
> agent implements the intended computation rather than a plausible guess. Where a BR and Appendix A
> overlap, the BR states WHAT must be true; Appendix A states HOW it is computed.
>
> **Locked decisions (product owner direction, 2026-07-18):**
> (1) The evidence ladder (Appendix A.1) is the product's single claims vocabulary — every surfaced
> circuit and edge displays its rung, and no UI or MCP surface may present a lower rung using a higher
> rung's language. (2) Attribution re-ranking (Tier 2) is **in scope this increment**, positioned between
> mining and ablation validation. (3) Cross-position/attention-aware discovery is scoped as a **named,
> designed-for tier** (Tier 2.5): the capture store and circuit-definition contract MUST carry token-position
> and attention-mediation fields from day one, while the mining implementation of Tier 2.5 may land in a
> fast-follow. (4) Persistence edges are **typed, not hidden**. (5) Member caps apply **per layer**.

```yaml
brd:
  metadata:
    brd_id: BRD-MIS-CIRCUITS-002
    project_name: "miStudio — Circuit Discovery Rigor Supplement: Attribution, Faithfulness & Cross-Position Structure"
    version: "0.1"
    author: "Sean (product owner) with external methodology review, 2026-07-18"
    last_updated: "2026-07-18"
    status: "draft"
    increment_of: "miStudio (000_PPRD|miStudio.md)"
    supplement_to: "BRD-MIS-CIRCUITS-001"
    successor_to: "BRD-MIS-CLUSTERS-001 (via BRD-MIS-CIRCUITS-001)"
    consumed_with: "BRD-MIS-CIRCUITS-001 — the two BRDs are one planning unit for the PPRD update"

  business_context:
    problem_statement: >
      CIRCUITS-001 gives miStudio the machinery to mine, review, promote, steer, and export cross-layer
      circuits — but as specified, the pipeline's evidence has three structural weaknesses. First, its
      discovery signal (lag-0 same-token co-activation, ranked partly by a direct residual weight prior)
      is biased toward exactly the trivial residual-echo edges its own filter removes, while the
      non-trivial circuits users actually want — attention-mediated, cross-token-position computation —
      are invisible to it by construction. Second, its causal claims rest on an underspecified ablation
      (no defined suppression semantics, no SAE-error handling, no noise floor, edge-level only), so
      "validated" would not mean what a reader of the interpretability literature expects it to mean.
      Third, its statistics lack a null model and replication discipline, so the top of the ranking will
      be dominated by high-base-rate feature pairs and corpus artifacts. Left unaddressed, these gaps
      produce a workbench that ships circuits which feel discovered but are not mechanistically real —
      the precise credibility failure BR-009 of CIRCUITS-001 was written to prevent.
    vision_statement: >
      Complete the arc the product owner has set: miStudio as the environment where the meanings of a
      model's features are discovered (traditionally via labels and examples, and dynamically via mined
      structure), managed and refined as first-class curated artifacts, and then converted into real-world
      leverage — validated causal handles on a model's inference direction and response, usable in live
      inference. This supplement supplies the missing rigor layer: a graded evidence ladder from
      correlation to attribution to intervention to circuit-level faithfulness, statistics that survive a
      null model and a held-out corpus, discovery granularities (feature-level and cluster-level) that
      match how meaning actually organizes, and a capture/contract foundation ready for the cross-position
      structure where the model's most interesting computation lives. The result is a workbench whose
      discovered circuits are trustworthy enough to steer production inference and defensible enough to
      publish.
    primary_objectives:
      - "Install the evidence ladder as the product's claims discipline: every edge and circuit carries exactly one rung (mined → attribution-supported → causally validated → faithfulness-tested), displayed everywhere the artifact appears."
      - "Make co-activation mining statistically sound: base-rate-corrected association (PMI/lift), minimum-support thresholds, multiple-comparison discipline, and held-out-corpus replication as a reported number."
      - "Add cluster-level (supernode) mining as a first-class discovery granularity alongside feature-level mining, converting the existing curated cluster library into the unit of circuit structure."
      - "Insert a Tier-2 gradient-attribution pass (one forward + one backward per prompt, stop-gradient through SAE reconstruction error) that re-ranks mined candidates before expensive ablation validation, and measure its survival-rate uplift."
      - "Fully specify interventional validation: suppression semantics, SAE-error preservation, counterfactual baseline, standardized effect size against a shuffled-pair null, and sign-consistency requirements — resolving CIRCUITS-001's highest-priority open question."
      - "Add circuit-level faithfulness testing at promotion time: whole-circuit ablation and circuit-only sufficiency, not just per-edge checks."
      - "Repair the weight-prior/triviality-filter tension: the direct residual alignment prior becomes an echo DETECTOR inside the filter and participates in ranking only in combination with distinctness signals; persistence edges become a typed, visible edge class."
      - "Design capture storage and the circuit-definition contract to carry token-position and attention-mediation information from day one, so Tier-2.5 cross-position discovery is an implementation fast-follow, not a schema migration."
      - "Wire cross-layer steering hazard detection to measured validated edge effect sizes instead of a separate heuristic."
    success_criteria:
      - "No surfaced edge or circuit anywhere in the product (UI, MCP, exports) presents a claim above its evidence rung; rung labels are machine-readable in the contract and human-unmissable in the UX."
      - "Top-of-ranking candidates are no longer dominated by high-base-rate pairs: the null-model comparison is computed and reported for every discovery run."
      - "Attribution re-ranking measurably raises the ablation survival rate of the submitted tier versus co-activation ranking alone, and the uplift is reported per run."
      - "'Causally validated' means: specified suppression semantics, error-preserving intervention, standardized effect size exceeding the shuffled null at the agreed threshold, with sign consistency — reproducible from the recorded validation manifest."
      - "A promoted circuit can display a faithfulness score (behavior retained under circuit-only operation / behavior lost under whole-circuit ablation) alongside its per-edge validations."
      - "Cluster-level mining produces circuit candidates whose members are existing curated clusters, reviewable in the same UX, exportable in the same contract."
      - "The capture store and mistudio.circuit-definition/v1 schema carry position/attention fields (populated or explicitly null) such that enabling Tier-2.5 requires no migration of existing artifacts."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers / power users who need evidence they can defend."
      - "AI/agent operators (Claude Code via MCP) executing the full discovery→attribution→validation→faithfulness→promotion loop autonomously."
    secondary_users:
      - "Downstream miLLM operators consuming circuit projections and, later, live circuits — inheriting the evidence-rung vocabulary."
      - "External consumers of exported circuit definitions, for whom the rung labels are the trust interface."
    stakeholders:
      - "Product owner (Sean) — direction and acceptance."
      - "miStudio maintainers."

  scope_definition:
    in_scope:
      - "Evidence-ladder claims discipline across UI, MCP, and contract (GAP-CLAIMS; BR-026)."
      - "Statistical soundness of co-activation mining: PMI/lift with base-rate correction, minimum support, multiple-comparison control, held-out replication (GAP-6; BR-015)."
      - "Cluster-level (supernode) mining granularity, feature-level mining retained (GAP-7; BR-016)."
      - "Fully specified interventional semantics and validation criterion with recorded validation manifests (GAP-4; BR-017, BR-018)."
      - "Circuit-level faithfulness testing at promotion (GAP-5; BR-019)."
      - "Weight-prior role change + typed persistence edges (GAP-2; BR-020, BR-021)."
      - "Tier-2 gradient-attribution re-ranking of mined candidates (GAP-3; BR-022)."
      - "Tier-2.5 readiness: position- and attention-aware capture schema and contract fields; mediation-analysis design documented (GAP-1; BR-023). Implementation of Tier-2.5 mining itself is a fast-follow feature PRD, not blocked on this BRD."
      - "Hazard detection driven by validated effect sizes with heuristic fallback clearly labeled (GAP-8; BR-024)."
      - "Resolution of CIRCUITS-001 open questions: per-layer member caps (BR-025), persistence-edge policy (BR-021), edge-validation criterion (BR-018), lag-0 sufficiency (BR-023)."
      - "Amendments to mistudio.circuit-definition/v1 BEFORE freeze: evidence-rung field, validation manifest reference, edge type (computed|persistence|attention-mediated), position/attention fields, attribution scores. Additive design within the not-yet-frozen v1 — no second kind."
    out_of_scope:
      - "Full attribution-graph construction (complete linearized replacement-model graphs à la circuit tracing) — future tier beyond attribution re-ranking."
      - "Cross-layer transcoder training — SAar-per-layer remains the substrate this increment; transcoders are a research option recorded in future_considerations."
      - "Tier-2.5 mining implementation (in-scope to DESIGN and schema-carry; implementation is the named fast-follow)."
      - "miLLM runtime changes (unchanged from CIRCUITS-001 scope lock)."
      - "Automated circuit narration/labeling beyond existing per-feature labels."
    future_considerations:
      - "Tier-3 full attribution graphs with error nodes and frozen-attention linearization, per the circuit-tracing methodology; candidate substrate: per-layer transcoders replacing or complementing SAEs."
      - "Tier-2.5 implementation feature PRD: attention-mediated cross-position edge mining (see Appendix A.6 design sketch)."
      - "Aggregation of per-prompt attribution across corpora to find prompt-general circuits (attribution graphs are prompt-specific; generality comes from aggregation)."
      - "Cross-model circuit correspondence once multiple models are hosted."
      - "Faithfulness-metric hardening (noising vs denoising baselines, task-metric-specific faithfulness) as field standards settle."
      - "SUBSTRATE PILOT (see Addendum B): a bounded evaluation of multi-layer dictionary architectures (MLSAE / crosscoder / cross-layer transcoder) against the incumbent per-layer SAEs, run as a research track BESIDE this increment, never blocking it. Its output is a go/no-go seed for BRD-MIS-SUBSTRATE-001; a 'go' would make multi-layer substrate the follow-on arc after the miLLM circuits runtime."
    dependencies:
      - "Everything CIRCUITS-001 depends on, plus:"
      - "Backward-pass capability through the host model with SAE encode/decode in the graph (verified feasible: 1.2B model, single RTX 3090 24GB; see technical_constraints)."
      - "Capture-store schema authority (this BRD amends it before first implementation)."
      - "The 30 curated cluster profiles (supernode mining seeds and units)."
      - "circuit-definition/v1 NOT yet frozen (this BRD's contract amendments land pre-freeze; if freeze has occurred, amendments arrive as additive optional fields per the interchange family's evolution rule)."
    assumptions:
      - "One forward + one backward pass per prompt is affordable at corpus scale on the current GPU (attribution batch is comparable to or cheaper than the capture pass)."
      - "SAE reconstruction error terms are computable and storable per (layer, token) at capture time at acceptable cost when stored as norms + optional sparse residuals."
      - "The existing evaluation-corpus choice (extraction dataset) can be split into discovery and held-out partitions without invalidating CIRCUITS-001's corpus assumption."
      - "Curated cluster profiles are semantically coherent enough that cluster-level co-activation is meaningful (cohesion scores exist and can gate eligibility)."

  business_requirements:
    # Numbering continues from BRD-MIS-CIRCUITS-001 (BR-001..BR-014).
    - id: BR-015
      gap: GAP-6
      amends: "BR-007, BR-008 (CIRCUITS-001)"
      text: >
        Co-activation association SHALL be computed with base-rate correction (pointwise mutual
        information and/or lift as defined in Appendix A.3), not raw co-occurrence counts. Candidate
        edges SHALL satisfy a minimum-support threshold (minimum co-firing token count) before ranking.
        Discovery runs SHALL apply a multiple-comparison discipline appropriate to the candidate-pair
        count and disclose it. Every discovery run SHALL evaluate its surfaced candidates against a
        held-out corpus partition and report the held-out replication rate as a first-class number
        alongside the survival rate of CIRCUITS-001.
    - id: BR-016
      gap: GAP-7
      amends: "BR-007 (adds a granularity; seeded/open modes unchanged)"
      text: >
        Discovery SHALL support two granularities as first-class: FEATURE-level (individual SAE latents,
        as in CIRCUITS-001) and CLUSTER-level (existing curated cluster profiles as supernodes, with
        cluster activation defined per Appendix A.4). Cluster-level candidates SHALL be reviewable in the
        same review UX and exportable in the same contract, with members recorded as cluster references
        resolvable to their feature membership. Cluster-level mining SHALL be the recommended default for
        seeded mode.
    - id: BR-017
      gap: GAP-4
      amends: "BR-009 (specifies the intervention BR-009 requires)"
      text: >
        Interventional validation SHALL use defined, recorded suppression semantics. The default
        intervention is DIRECTIONAL SUBTRACTION: subtract the feature's realized contribution
        (activation × decoder direction) from the residual stream at the hook point, leaving the SAE
        reconstruction error term untouched (Appendix A.5). Full-reconstruction-swap intervention, if
        offered, SHALL preserve and re-add the reconstruction error term. The counterfactual baseline
        (zero vs corpus-mean activation) SHALL be a recorded run parameter with zero as default. Every
        validation run SHALL persist a validation manifest (intervention type, baseline, prompts, seeds,
        thresholds, measured values) sufficient to reproduce the result.
    - id: BR-018
      gap: GAP-4
      resolves_open_question: "CIRCUITS-001 next_steps: edge-validation criterion"
      text: >
        An edge is CAUSALLY VALIDATED when, under BR-017 intervention on the upstream feature: (a) the
        standardized effect size on the downstream feature's activation (Appendix A.5) exceeds the
        threshold calibrated against a shuffled-pair null distribution measured on the same corpus; and
        (b) the effect's sign is consistent across at least the agreed minimum number of evaluation
        prompts. Both the threshold and the null distribution summary SHALL be recorded in the validation
        manifest. Edges failing (a) or (b) SHALL be recorded as tested-and-failed, not silently dropped.
    - id: BR-019
      gap: GAP-5
      amends: "BR-009, BR-012 (adds circuit-level evidence at promotion)"
      text: >
        At promotion time the system SHALL offer circuit-level faithfulness testing: (a) NECESSITY —
        ablate all circuit members simultaneously (per BR-017 semantics) and measure the loss of the
        circuit's associated behavior; (b) SUFFICIENCY — where tractable in the current single-GPU
        envelope, measure behavior retention when non-circuit features at the circuit's layers are
        ablated (Appendix A.7 defines the tractable v1 approximation). The resulting faithfulness scores
        SHALL be displayed on the circuit alongside per-edge validation status and recorded in the
        contract. Faithfulness, like validation, is a badge, not a gate.
    - id: BR-020
      gap: GAP-2
      amends: "BR-008, BR-010 (redefines the weight prior's role)"
      text: >
        The direct cross-layer weight alignment prior (W_dec(Li)·W_enc(Lj)) SHALL function primarily as
        an ECHO DETECTOR within the triviality filter: high weight alignment combined with high
        co-activation and high token-identity overlap classifies an edge as persistence-type. In candidate
        RANKING, the weight prior SHALL NOT act as a standalone booster; it participates only in
        combination with semantic-distinctness signals, and the ranking function disclosure required by
        BR-008 SHALL state this composition. High co-activation with LOW weight alignment SHALL be
        recognized as the signature of a computed (MLP-mediated) edge and SHALL NOT be penalized for the
        low prior.
    - id: BR-021
      gap: GAP-2
      resolves_open_question: "CIRCUITS-001 next_steps: triviality policy nuance"
      text: >
        Every surfaced edge SHALL carry a machine-readable type: COMPUTED, PERSISTENCE, or (schema-ready
        for Tier-2.5) ATTENTION-MEDIATED. Persistence edges SHALL NOT be silently removed: they are
        de-ranked out of default views but remain queryable, visibly typed, and usable (e.g. for
        robustness steering). The classification signals for each edge's type SHALL be disclosed with the
        edge.
    - id: BR-022
      gap: GAP-3
      amends: "CIRCUITS-001 out_of_scope (gradient attribution moves IN scope as re-ranking)"
      text: >
        Between mining and interventional validation the system SHALL provide a Tier-2 ATTRIBUTION pass:
        for each candidate edge, a gradient-based attribution score computed with one forward and one
        backward pass per evaluation prompt, with stop-gradient through the SAE reconstruction error
        (Appendix A.6). Attribution scores SHALL re-rank the candidate list before ablation sampling, be
        disclosed per candidate as a third evidence signal (extending BR-008), and be recorded in the
        contract. Each discovery run SHALL report the ablation survival rate of the attribution-re-ranked
        tier versus the co-activation-only ranking, so the uplift is a measured, first-class number.
    - id: BR-023
      gap: GAP-1
      resolves_open_question: "CIRCUITS-001 next_steps: positional alignment / lag-0 sufficiency"
      text: >
        Lag-0 same-token co-activation is RECOGNIZED as insufficient for discovering attention-mediated
        cross-position circuits and is scoped as the deliberately-limited Tier-1 signal. This increment
        SHALL: (a) store token positions in the capture store such that cross-position co-activation and
        attention-pattern joins are computable without recapture; (b) carry attention-mediation fields
        (edge type, source/target position relationship, mediating head set when known) in the
        circuit-definition contract as schema-ready fields; (c) document the Tier-2.5 mining design
        (Appendix A.8) as the named fast-follow. UI and documentation SHALL disclose the Tier-1
        limitation wherever discovery results are presented.
    - id: BR-024
      gap: GAP-8
      amends: "BR-004 (grounds hazard detection in measured data)"
      text: >
        Cross-layer steering hazard detection SHALL consume validated edge effect sizes as its primary
        signal: when a steering configuration simultaneously boosts an upstream feature and a downstream
        feature connected by a validated positive edge, the compounding warning SHALL quantify the
        expected double-counting using the measured effect size; analogously for cancellation on negative
        edges. Where no validated edge exists between co-steered members, heuristic hazard signals remain
        permitted but SHALL be labeled heuristic, per the evidence-ladder discipline.
    - id: BR-025
      resolves_open_question: "CIRCUITS-001 next_steps: member caps"
      text: >
        The existing 20-member cap applies PER LAYER within a circuit, not per circuit. The contract and
        UX SHALL enforce and display the cap per layer; deep circuits across many layers are thereby not
        artificially capped in total membership.
    - id: BR-026
      gap: GAP-CLAIMS
      amends: "BR-008, BR-011, BR-012, BR-013 (claims vocabulary across all surfaces)"
      text: >
        The product SHALL implement the evidence ladder as its single claims vocabulary. Rungs: (0)
        MINED — statistical association only; (1) ATTRIBUTION-SUPPORTED — Tier-2 gradient attribution
        agrees; (2) CAUSALLY VALIDATED — BR-018 satisfied; (3) FAITHFULNESS-TESTED — BR-019 satisfied at
        circuit level. Every edge carries its rung; a circuit's displayed rung is the MINIMUM rung of its
        constituent edges plus its own faithfulness status. Rungs SHALL be machine-readable in
        mistudio.circuit-definition/v1, rendered in every UI surface where the artifact appears, returned
        by every MCP tool that returns circuits or edges, and included in exports. No surface may
        describe a rung-0/1 artifact with causal language. This subsumes and strengthens the
        'unvalidated' badge of CIRCUITS-001 BR-012.

  success_metrics:
    quantitative_metrics:
      - "Attribution uplift: ablation survival rate of the attribution-re-ranked top-K exceeds the co-activation-only top-K on the same run; both numbers reported per run (target: relative uplift ≥ +15 percentage points on the calibration corpus; a null result is a reportable finding, not a hidden one)."
      - "Held-out replication rate reported for 100% of discovery runs; calibration target ≥60% of surfaced (rung ≥1) candidates replicate direction and significance on the held-out partition."
      - "Null-model separation: the reported distribution of surfaced-candidate association scores versus the shuffled-null distribution shows the disclosed threshold's percentile for every run."
      - "Echo classification: ≥90% of a hand-labeled audit set of known persistence edges are typed PERSISTENCE by the BR-020/BR-021 classifier; misclassification of the audit set's computed edges ≤10%."
      - "Faithfulness coverage: 100% of circuits promoted after this increment lands carry either a faithfulness score or an explicit 'not yet faithfulness-tested' rung display; ≥3 promoted circuits reach rung 3."
      - "Validation manifests: 100% of causal claims (rung ≥2) are backed by a stored manifest sufficient to reproduce the measurement."
      - "Contract: position/attention/rung/attribution fields present in the schema; export→import round-trip preserves rung, edge types, attribution scores, and manifest references."
      - "Backward-pass envelope: the Tier-2 attribution pass over the reference corpus completes within an agreed wall-clock multiple (target ≤1.5×) of the Tier-1 capture pass on the same corpus and layers."
    qualitative_indicators:
      - "A reviewer can state, for any circuit, which rung it is on and what exactly would move it up one rung."
      - "The top of a default-ranked candidate list reads as semantically interesting computed edges, not base-rate noise or echoes."
      - "Steering hazard warnings feel grounded ('validated edge, effect size X') rather than vague."
      - "Product-owner acceptance that the ladder vocabulary is consistent everywhere: UI, MCP responses, exports, docs."
    measurement_methods:
      - "Per-run discovery reports carrying: null-model summary, held-out replication, attribution uplift, survival rate, echo-filter effect."
      - "A curated audit set of known-echo and known-computed edges maintained as a regression fixture."
      - "Manifest reproduction test: re-run a stored validation manifest and reproduce the recorded effect within tolerance."
      - "Contract round-trip and schema-validation suites extended for the new fields."
      - "Product-owner acceptance review against the qualitative indicators."

  feature_themes:
    core_features:
      - "Evidence-ladder rung model: data model, contract fields, UI badges, MCP response fields (BR-026)."
      - "Statistically sound mining: PMI/lift + support + multiple-comparison discipline + discovery/held-out corpus partitioning and replication reporting (BR-015)."
      - "Cluster-level supernode mining with cluster-activation definition and cluster-reference members (BR-016)."
      - "Intervention engine v2: directional-subtraction semantics, error-term preservation, baselines, validation manifests (BR-017)."
      - "Edge-validation criterion: standardized effect size vs shuffled null + sign consistency; tested-and-failed recording (BR-018)."
      - "Circuit faithfulness runner: whole-circuit necessity ablation + tractable sufficiency approximation at promotion (BR-019)."
      - "Edge typing and ranking recomposition: echo detector, persistence type, computed-edge recognition (BR-020, BR-021)."
      - "Tier-2 attribution pass: stop-gradient SAE attribution, re-ranking, uplift reporting (BR-022)."
      - "Tier-2.5 readiness: position-carrying capture schema, attention-mediation contract fields, documented mining design (BR-023)."
      - "Hazard detection v2 wired to validated effect sizes (BR-024)."
      - "Per-layer cap enforcement (BR-025)."
    secondary_features:
      - "Discovery-report artifact (per-run methods + numbers page) rendered in UI and returned via MCP."
      - "Audit-set fixture management for the echo classifier."
      - "Manifest browser in the review UX."
    future_features:
      - "Tier-2.5 attention-mediated mining implementation (named fast-follow feature PRD)."
      - "Tier-3 full attribution graphs; transcoder substrate evaluation."
      - "Cross-corpus circuit generality aggregation."

  considerations:
    budget_constraints: "TBD"
    timeline_expectations: >
      This supplement is absorbed into the same PPRD update and feature-PRD wave as CIRCUITS-001 — it
      re-shapes that work rather than following it. Sequencing inside the increment (see
      recommended_actions): contract/schema amendments and the evidence ladder land FIRST (they gate the
      freeze), then statistics + mining granularity, then intervention/validation/faithfulness, then
      attribution, then hazard v2. Tier-2.5 implementation follows as its own feature PRD.
    regulatory_or_policy_drivers: []
    technical_constraints:
      - "Single RTX 3090 (24GB): the Tier-2 backward pass through a 1.2B model with SAE modules attached must fit; gradient checkpointing and per-layer-subset attribution are the pressure valves; the wall-clock envelope metric bounds cost."
      - "Capture-store growth from position indices and error-term norms must stay within the CIRCUITS-001 sparse-storage budget philosophy: positions are index columns on already-stored events, error terms stored as norms by default (full residual vectors optional per run)."
      - "All new contract content lands inside mistudio.circuit-definition/v1 BEFORE freeze; if v1 has frozen, additive-optional fields only, per the interchange family rule."
      - "The evidence ladder must be implemented as one shared enum/model consumed by UI, MCP, and contract code — divergence between surfaces is itself a defect class."
    integration_requirements:
      - "MCP tools from CIRCUITS-001 extend to expose: rung fields on every returned circuit/edge, attribution scores, validation-manifest retrieval, faithfulness runs, per-run discovery reports, and edge-type filters — keeping the agent loop fully self-serve."
      - "The projection export (CIRCUITS-001 BR-014) carries the parent circuit's rung and a partial-rendering marker so single-layer consumers cannot mistake a slice for a validated whole."
    scalability_expectations: >
      The statistical discipline (support thresholds, null models) and the granularity choice (cluster-level
      default for seeded mode) are the primary tractability controls at 16 layers × 8k latents. Attribution
      is O(prompts) per candidate set, not O(candidates); ablation remains the sampled, expensive tier and
      is now spent on an attribution-enriched shortlist.

  risks:
    - id: RSK-009
      description: "Backward-pass memory pressure: attaching SAE modules with gradients through a 1.2B model may exceed 24GB with naive implementation."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Stop-gradient design keeps SAE parameters frozen (activations-only gradients); gradient checkpointing; per-layer-subset attribution; envelope metric catches regressions."
    - id: RSK-010
      description: "Rigor tax: the added statistics/validation machinery slows the discovery loop enough that the workbench feels heavy."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Ladder is badge-not-gate at every rung; Tier-1 mining remains instant-feedback; expensive tiers are explicit user/agent choices with cost estimates; per-run reports make the cost buy visible credibility."
    - id: RSK-011
      description: "Shuffled-null miscalibration: a badly constructed null (e.g. shuffling that breaks sparsity structure) sets thresholds that pass noise or block everything."
      impact: "high"
      likelihood: "medium"
      mitigation: "Null construction is specified (Appendix A.3: within-feature circular shift preserving marginal rates), disclosed per run, and sanity-checked against the audit fixture."
    - id: RSK-012
      description: "Cluster-activation definition risk: a poor supernode activation definition (BR-016) makes cluster-level mining incoherent."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Definition specified in Appendix A.4 with max-over-members default and cohesion-score eligibility gating; feature-level mining remains available as the fallback granularity."
    - id: RSK-013
      description: "Schema-readiness theater: Tier-2.5 fields land in the contract but the fast-follow never ships, leaving dead schema."
      impact: "low"
      likelihood: "medium"
      mitigation: "Fields are nullable/optional and cheap; the Tier-2.5 design doc (Appendix A.8) is a deliverable of THIS increment so the fast-follow starts specified, not from scratch."
    - id: RSK-014
      description: "Sufficiency testing intractability: circuit-only sufficiency (ablate everything else) is ill-defined/expensive at v1 scale."
      impact: "medium"
      likelihood: "high"
      mitigation: "BR-019 scopes sufficiency to the tractable approximation defined in Appendix A.7 (top-k non-member ablation at circuit layers) and allows necessity-only faithfulness with the limitation disclosed on the badge."

  next_steps:
    open_questions:
      - "Shuffled-null parameters: number of shuffles per run and the percentile threshold default (99th?) — calibrate on the audit fixture during TDD."
      - "Held-out partition ratio and whether the split is per-document or per-token-window (per-document recommended to avoid leakage)."
      - "Minimum sign-consistency prompt count for BR-018 (proposal: ≥8 of 10 evaluation prompts agree in sign)."
      - "Behavior metric for faithfulness (BR-019): which output measure defines 'the circuit's associated behavior' — steering-target logit set, KL to baseline, or the compare-workflow's existing shift measures? (Recommendation: reuse compare-workflow output-shift measures for continuity.)"
      - "Whether attribution scores use raw grad×activation or integrated-gradients-lite (2-4 interpolation steps) as default (Appendix A.6 defines both; proposal: raw for re-ranking, IG-lite available on demand)."
    recommended_actions:
      - "Proceed to 0xcc/instruct/002_create-project-prd.md consuming CIRCUITS-001 + CIRCUITS-002 together: the PPRD's circuit rows absorb the ladder, statistics, attribution, faithfulness, and Tier-2.5-readiness scope."
      - "ADR updates (0xcc/instruct/003_create-adr.md): (a) evidence-ladder as product-wide claims model; (b) intervention semantics (directional subtraction + error preservation) as the causal standard; (c) weight-prior role change; (d) contract amendments pre-freeze; (e) attribution-tier architecture (stop-gradient through SAE, activations-only backward)."
      - "Feature PRD wave (004): fold this supplement into the CIRCUITS-001 PRD split as amended scope — likely (a) multi-SAE steering + budgets + hazards-v2 [BR-001..004, BR-024]; (b) capture + statistics + mining granularities + attribution [BR-005..008, BR-015, BR-016, BR-022, BR-023 capture-side]; (c) intervention engine + validation + faithfulness + remediation [BR-009, BR-017, BR-018, BR-019]; (d) review/promotion/ladder + contract + projection [BR-010..014, BR-020, BR-021, BR-025, BR-026]. Then TDDs (005), TIDs (006), tasks (007), execution (008)."
      - "Author the Tier-2.5 design doc from Appendix A.8 as a deliverable of feature PRD (b)."
    priority_for_clarification:
      - "Faithfulness behavior metric — blocks BR-019 TDD."
      - "Null-model parameters — blocks BR-015/BR-018 thresholds."
      - "Contract amendment review against the anticipated miLLM runtime BEFORE v1 freeze — same discipline as CLUSTERS-001."
```

---

# Appendix A — Implementation Primer (normative for the implementing agent)

This appendix defines the computations the business requirements reference. It exists so an agentic
implementer (Claude Code executing the XCC chain) builds the intended mathematics. Where a TDD later
refines a formula, the TDD wins; until then, this appendix is the reference.

## A.1 The evidence ladder (BR-026)

Mechanistic claims come in strictly increasing strength. The product encodes this as an enum:

| Rung | Name | Meaning | Produced by |
|---|---|---|---|
| 0 | `mined` | Statistical association on the discovery corpus survived the null model and support thresholds | Tier-1 mining (BR-015) |
| 1 | `attribution_supported` | Gradient attribution agrees in sign and magnitude percentile | Tier-2 pass (BR-022) |
| 2 | `causally_validated` | Real intervention satisfied BR-018 | Ablation validation (BR-017/018) |
| 3 | `faithfulness_tested` | Circuit-level necessity (and, where run, sufficiency) measured per BR-019 | Faithfulness runner |

Rules: an edge's rung is the highest rung it has *passed* (a failed rung-2 test does not remove rung 0/1;
it records `tested_and_failed` at rung 2). A circuit's displayed rung = min over member edges' rungs,
with faithfulness status shown separately. Language mapping (enforced in UI copy and MCP descriptions):
rung 0–1 → "associated / suggested"; rung 2 → "causally validated (edge)"; rung 3 → "faithfulness-tested
(circuit)". The word "causal" is forbidden below rung 2.

## A.2 Notation

- Model: decoder-only transformer, residual stream `x_l(t)` at layer `l`, token position `t`.
- Per-layer SAE at layer `l`: encoder `f_l(x) = act_fn(W_enc,l · x + b_enc,l)`, decoder
  `x̂ = W_dec,l · f + b_dec,l`. Feature `i`'s activation at token `t`: `a_{l,i}(t)`.
- Reconstruction error (error term): `ε_l(t) = x_l(t) − x̂_l(t)`. The true stream always satisfies
  `x_l(t) = x̂_l(t) + ε_l(t)`. **Every intervention and attribution below must preserve `ε_l(t)`** —
  this is the single most common implementation mistake; dropping it measures SAE artifact, not model
  mechanism.
- Candidate edge: upstream feature `u = (l_u, i)` → downstream feature `d = (l_d, j)`, `l_u < l_d`.

## A.3 Tier-1 statistics (BR-015)

Binarize firings with the capture threshold θ: `F_u(t) = 1[a_u(t) > θ]`. Over N captured tokens with
counts `n_u, n_d, n_{ud}`:

- `PMI(u,d) = log( (n_{ud} · N) / (n_u · n_d) )`; `lift = (n_{ud}·N)/(n_u·n_d)` (same quantity, linear).
- Optionally weight by activation magnitude (continuous co-activation: Pearson/Spearman over tokens
  where either fires) as a secondary column — report both, rank by PMI by default.
- **Minimum support:** discard pairs with `n_{ud} < s_min` (default proposal: 20) before any ranking.
- **Null model:** for each shuffle, apply an independent random *circular shift* to each feature's
  binary firing sequence *within each document* (preserves each feature's marginal rate and burstiness,
  destroys cross-feature alignment). Compute the PMI distribution over shuffled pairs; the run's
  significance threshold is a high percentile (default 99th) of this null. Do **not** use naive
  permutation across the whole corpus — it breaks document structure and inflates significance.
- **Multiple comparisons:** with P candidate pairs after support filtering, apply Benjamini–Hochberg FDR
  at q=0.05 on empirical p-values from the null, or equivalently raise the null percentile; disclose
  which in the run report.
- **Held-out replication (per run):** split the corpus per-document into discovery/held-out (default
  80/20) before mining. A surfaced candidate *replicates* if its held-out PMI exceeds the held-out null
  threshold with the same sign. Report `replication_rate = replicated / surfaced`.

## A.4 Cluster-level (supernode) mining (BR-016)

A curated cluster `C` at layer `l` with members `{i_1..i_m}` gets a supernode activation per token:

- Default: `A_C(t) = max_k a_{l,i_k}(t)` (max preserves sparsity semantics and is robust to member
  count). Alternative (recorded per run): mean over members.
- Binarize with the same θ; run A.3 statistics on supernode pairs. The candidate space collapses from
  (8k)² per layer-pair to (~clusters)², making exhaustive cross-layer supernode search tractable.
- Eligibility: only clusters whose stored cohesion score ≥ the configured floor participate (incoherent
  clusters make incoherent supernodes).
- A cluster-level edge that passes rungs is stored with `member_kind: cluster_ref` and resolves to
  feature membership at steering/projection time. Feature-level refinement of a promising cluster edge
  (which member pairs carry it) is a review-UX drill-down running A.3 restricted to the two clusters'
  members.

## A.5 Intervention semantics and validation criterion (BR-017, BR-018)

**Directional subtraction (default intervention)** on upstream `u` at layer `l_u`, all token positions
(v1) during a forward pass:

```
x'_{l_u}(t) = x_{l_u}(t) − (a_u(t) − a_base) · W_dec,l_u[:, i]
```

with `a_base = 0` (zero-ablation default) or the corpus-mean activation of `u` (mean-ablation option).
This touches only feature `u`'s realized contribution; `ε_{l_u}(t)` is untouched by construction
because we never re-decode. (If a full reconstruction-swap intervention is implemented — replace
`x_{l_u}` with `x̂` computed from a modified code — it MUST add `ε_{l_u}(t)` back.)

**Measurement:** run matched forward passes (same prompts, seeds, and sampling disabled/greedy for the
measurement pass) with and without the intervention; record downstream activations `a_d(t)` on both.
Define per-prompt `Δ_p = mean_t[a_d(t) | clean] − mean_t[a_d(t) | intervened]` over tokens where the
clean pass had `F_u(t)=1` (measuring where the edge could act).

**Standardized effect size:** `ES = mean_p(Δ_p) / σ_d`, where `σ_d` is the std of `a_d` over the clean
corpus capture (from the capture store — no extra passes). **Validated (rung 2)** iff `|ES|` exceeds the
shuffled-pair null ES threshold (same null machinery as A.3, applied to ES via random non-edge pairs)
AND sign(Δ_p) agrees on ≥ the configured fraction of prompts (proposal: 8/10). Persist everything in the
validation manifest (BR-017).

## A.6 Tier-2 gradient attribution (BR-022)

Goal: a cheap causal *estimate* for every candidate edge from one forward + one backward pass per
prompt, following the sparse-feature-circuits recipe adapted to per-layer SAEs:

1. Forward pass with SAEs attached in *pass-through* mode: at each captured layer compute `f_l`, `x̂_l`,
   `ε_l`, and **rewrite the stream as** `x_l = x̂_l + stopgrad_free(ε_l)` — i.e. the stream is expressed
   through the SAE code plus the error term, both carried forward so the computation is numerically
   identical to the clean model. SAE weights are frozen (`requires_grad=False`); gradients flow to the
   *activations* `f_l` and `ε_l`, not to SAE parameters.
2. Choose the downstream scalar `m`: for edge attribution, `m = mean_t a_d(t)` (the downstream
   feature's activation); for output attribution (used in faithfulness context), `m` = the behavior
   metric.
3. One backward pass: `g_u = ∂m/∂a_u` (per token, from autograd since `a_u` is in the graph).
4. **Attribution score:** `attr(u→d) = Σ_t g_u(t) · a_u(t)` (linear approximation of the effect of
   zero-ablating `u` on `m`). Optional IG-lite: average `g_u` at 2–4 interpolation points
   `α·a_u, α∈{¼,½,¾,1}` before the product, for better accuracy on saturated features.
5. Aggregate over the evaluation prompt set (mean and sign-consistency fraction). Rung 1
   (`attribution_supported`) requires sign agreement with the mined association and attribution
   magnitude above the run's percentile floor.
6. Cost: candidates at the same downstream target share the backward pass; batch prompts. Total cost is
   O(prompts × distinct downstream targets), independent of upstream-candidate count — this is why
   Tier 2 belongs *before* ablation sampling.

Memory notes (RSK-009): activations-only gradients + frozen SAEs + gradient checkpointing over
transformer blocks keeps a 1.2B model in the 24GB envelope; attribute layer-subsets per pass if needed.

## A.7 Circuit-level faithfulness (BR-019)

For a circuit `G` with member set `M` and behavior metric `B` (see open question; default = the
compare-workflow output-shift measure on the circuit's evaluation prompts):

- **Necessity:** `necessity = [B(clean) − B(ablate M)] / [B(clean) − B(ablate-all-features-at-G's-layers)]`
  using A.5 semantics for every member simultaneously. Closer to 1 → the circuit carries the behavior.
- **Sufficiency (tractable v1 approximation):** full circuit-only operation is intractable per-token;
  approximate by ablating the top-k *non-member* features by mean activation at each of the circuit's
  layers (default k=256/layer) and measuring behavior retention:
  `sufficiency ≈ [B(ablate top-k non-members) − B(ablate all)] / [B(clean) − B(ablate all)]`.
  Disclose k on the badge; allow necessity-only runs with the sufficiency field marked untested.
- Both scores, the metric identity, prompts, and parameters go in the manifest and contract.

## A.8 Tier-2.5 design sketch (attention-mediated cross-position edges) — deliverable design doc seed

Why: information moves across token positions only through attention; lag-0 mining cannot see e.g. a
subject-token feature at L10 driving a verb-token agreement feature at L13. Design outline for the
fast-follow:

1. **Capture side (landed THIS increment, BR-023):** store `(layer, feature, doc, token_pos, activation)`
   — position is already required; additionally persist per-layer attention patterns `A_h(t_q, t_k)` for
   selected heads/layers as an *optional* capture artifact (top-k keys per query to stay sparse).
2. **Candidate generation:** for feature pair `(u@l_u, d@l_d)`, compute *cross-position* co-activation:
   `u` fires at `t_k` and `d` fires at `t_q ≠ t_k` where some head `h` in a layer `l_u < l_h ≤ l_d` has
   `A_h(t_q, t_k)` above a mass threshold. The attention pattern is the join key — this replaces fixed
   lags entirely.
3. **Statistics:** same A.3 machinery on the joined events; the null additionally shuffles the
   attention join (shift `t_k` assignments) to control for positional artifacts.
4. **Attribution check:** A.6 extends naturally — attribute `a_d(t_q)` to `a_u(t_k)` with attention
   patterns *frozen* (treat `A_h` as constants in the backward pass), the standard linearization of the
   attribution-graph literature; record the mediating head set as edge evidence.
5. **Validation:** A.5 intervention on `u` restricted to source positions `t_k`, measuring `d` at the
   attention-linked `t_q`.
6. **Contract:** edge type `attention_mediated`, with `source_position_role`, `target_position_role`,
   `mediating_heads[]` — fields already present (nullable) from BR-023.

## A.9 Weight prior and edge typing (BR-020, BR-021)

`W_prior(u,d) = cos(W_dec,l_u[:,i], W_enc,l_d[j,:])` measures *direct residual-path* alignment only —
it is blind to MLP/attention computation between layers, hence biased toward echoes. Edge-type
classifier (v1, disclosed per edge):

- `persistence` if: `W_prior` ≥ high threshold AND token-identity overlap of top-activating contexts ≥
  overlap threshold AND label-embedding similarity ≥ similarity threshold (any 2 of 3 → persistence).
- `computed` otherwise (Tier-1/2), with the note that low `W_prior` + high association is the
  *expected signature* of MLP-mediated computation and must not be down-ranked for the low prior.
- `attention_mediated` reserved for Tier-2.5 evidence.
Persistence edges: default-view de-ranked, queryable, steerable, exportable, always typed.

## A.10 References (for the implementing agent's context)

- Marks et al., *Sparse Feature Circuits* — per-layer-SAE attribution with stop-gradients; the direct
  template for A.6.
- Ameisen et al. (Anthropic), *Circuit Tracing* / attribution graphs — error nodes, frozen attention,
  replacement-model linearization; the template for A.8 step 4 and the Tier-3 future.
- Conmy et al., *ACDC* — iterative patching-based discovery; origin of the faithfulness framing in A.7.
- Syed et al., *Attribution Patching* — gradient approximation of activation patching; cost model
  behind BR-022.
- `decoderesearch/circuit-tracer` (open source) — reference implementation patterns for feature-level
  intervention and graph pruning.

## A.11 XCC chain execution instructions (for the agent)

1. Read `BRD-MIS-CIRCUITS-001` and this document as one unit; conflicts resolve in favor of THIS
   document (it amends by design; every amendment names its target BR).
2. Run `0xcc/instruct/002_create-project-prd.md`: update `000_PPRD|miStudio.md` with the amended
   circuit scope (ladder, statistics, granularities, attribution, intervention v2, faithfulness,
   Tier-2.5 readiness). Ask only questions neither BRD answers; this BRD's `open_questions` are the
   expected question set.
3. Run `003_create-adr.md` for the five ADR entries listed in `recommended_actions`.
4. Run `004_create-feature-prd.md` per the four-feature split in `recommended_actions`, mapping every
   BR (001–026) to exactly one owning feature PRD; include the Tier-2.5 design doc as a deliverable of
   feature (b).
5. Run `005`/`006`/`007`/`008` per feature in dependency order (d's contract amendments and the ladder
   model land first — they gate the v1 freeze; then b; then c; then a's hazard-v2 which depends on c's
   validation data).
6. Throughout: every causal-language string in UI/MCP/docs is checked against A.1's language mapping;
   every stored causal number traces to a manifest.

---

# Addendum B — Substrate Pilot Evaluation (future consideration; seed for BRD-MIS-SUBSTRATE-001)

> **Status:** future consideration, not a requirement of this increment. Nothing in CIRCUITS-001/002 is
> blocked by, sequenced after, or redesigned for this pilot. It exists so the substrate question is
> answered with data rather than re-litigated by intuition each planning cycle.

**B.1 Question.** The workbench's substrate is per-layer SAEs (16 layers × 8k latents). Multi-layer
dictionary architectures — multi-layer SAEs (one dictionary shared across depth), crosscoders (per-layer
encoder/decoder vectors per feature), and cross-layer transcoders (features that replace MLP computation,
writing to all subsequent layers) — would absorb residual-echo edges by construction, make cross-layer
structure a property of single features, natively encode per-layer steering allocations, and open the
Tier-3 attribution-graph path. They would also invalidate every trained artifact, extraction, label, and
curated cluster, and weaken feature localization in the evidence-disclosure UX. The pilot decides whether
that trade is worth a substrate migration arc.

**B.2 Shape.** One pilot dictionary — recommended: a **crosscoder** over a contiguous 4–6-layer window
containing the curated-cluster layers (L11–L15), trained on the same corpus as the incumbent extractions,
at matched dictionary width and sparsity (L0). Crosscoder is the recommended candidate because it is the
strongest test of the two properties this product actually monetizes (echo absorption + native multi-layer
steering); an MLSAE fallback is acceptable if crosscoder training proves unstable on the single-GPU
envelope. CLTs are explicitly deferred to the Tier-3 decision, not this pilot.

**B.3 Evaluation gates.** The pilot reports five numbers/judgments, each against the incumbent per-layer
SAEs on the same corpus, layers, and evaluation prompts:

1. **Echo absorption:** fraction of the incumbent pipeline's PERSISTENCE-typed edges (BR-021) whose
   endpoint features map onto a *single* pilot feature active at both layers (mapping via decoder-direction
   similarity + activation correlation). Target: majority absorbed.
2. **Reconstruction parity:** per-layer FVU / CE-loss-recovered within an agreed margin of the incumbent
   SAEs at matched L0.
3. **Interpretability parity:** blinded label-quality review on a sampled feature set (existing labeling
   workflow, same rubric); pilot features must not be materially worse.
4. **Steering quality:** for ≥3 behaviors with existing validated single/multi-layer profiles, steer via
   the pilot feature's native multi-layer decoder directions; compare output-shift vs coherence-degradation
   curves (compare-workflow measures) against the incumbent profiles. Target: equal-or-better shift at
   equal degradation, with zero manual per-layer allocation.
5. **Localization cost:** qualitative — can the evidence-disclosure UX still answer "where does this
   feature live?" acceptably (per-layer decoder-norm profile as the proposed answer)?

**B.4 Decision rule.** GO (draft BRD-MIS-SUBSTRATE-001 as a full migration arc) iff gates 1 and 4 pass
and gates 2, 3, 5 show parity-or-acceptable. NO-GO records the numbers and re-queues the question only on
a material trigger (new architecture, new model, Tier-3 commitment). Either way the pilot's capture,
training, and evaluation harness is retained — it is the seed infrastructure for any future substrate
work.

**B.5 Chain placement.** The pilot is a research task, not an XCC feature: no PPRD row, no contract
changes. Its only XCC artifact is the resulting seed/decision record in `0xcc/prds/` (see
`BRD-MIS-SUBSTRATE-001.seed.md`), which — on GO — is expanded via `001_generate-brd.md` into the full
substrate BRD, sequenced after BRD-MILLM-CIRCUITS-001.
