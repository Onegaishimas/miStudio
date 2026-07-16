# BRD — Feature Clusters, Combined-Strength Steering & Portable Cluster Definitions (miStudio increment)

> **Incremental BRD.** This is an enhancement BRD for the existing miStudio product, not a greenfield
> project BRD. It is the direct input to an update of the Project PRD (`000_PPRD|miStudio.md`) and to one or
> more new Feature PRDs. Per the XCC workflow, the next step is `0xcc/instruct/002_create-project-prd.md`.
>
> **Scope decision (locked):** This BRD covers the **miStudio-side** work only — up to and including the
> portable JSON *cluster definition* export. The downstream MILLM import, the unified MILLM+miStudio MCP
> server, and the Open WebUI runtime integration are captured here as **future considerations / vision** so the
> ever-tightening miStudio↔MILLM interdependency stays on record, and will be specified in a **separate follow-on
> BRD** (BRD-MIS-CLUSTERS-002 or similar) once this increment lands. This BRD deliberately contains **no MILLM
> implementation requirements** — only the miStudio deliverables and the portable contract that makes the
> ecosystem integration possible.

```yaml
brd:
  metadata:
    brd_id: BRD-MIS-CLUSTERS-001
    project_name: "miStudio — Feature Clusters & Portable Combined-Strength Steering"
    version: "0.1"
    author: "Sean (transcribed verbal description, 2026-07-15)"
    last_updated: "2026-07-15"
    status: "draft"
    increment_of: "miStudio (000_PPRD|miStudio.md)"

  business_context:
    problem_statement: >
      miStudio can steer a model with individual SAE features and, recently, with a blended set of features.
      But the unit of meaning users actually reason about is a *cluster* of features that fire together and
      appear to share a meaning — not a lone feature. The current product (a) calls these "Feature Groups,"
      which under-sells them and reads as mere organization rather than a steering primitive; (b) does not let
      a user capture a tuned cluster (its members, their per-feature strengths, a name, and a narrative) as a
      first-class, reusable, portable object; and (c) gives low-trust feedback during combined steering — the
      per-prompt result is labeled only with the single top feature's index, which makes users doubt that all
      the cluster's features are actually being combined. There is also no grounded, non-arbitrary way to set
      the combined influence strength across a cluster; today it is guessed, so every experiment starts from a
      useless point.
    vision_statement: >
      Turn the miStudio workbench into the place where users *discover, tune, name, and understand* clusters of
      features that work together — then capture each tuned cluster as a portable, standardized definition
      (members + a principled per-feature strength allocation + name + narrative) that can be exported and,
      in later increments, carried across the miStudio↔MILLM ecosystem to influence real chat sessions. The
      workbench is the laboratory; the exported cluster definition is the mobile, tradeable artifact.
    primary_objectives:
      - "Rename the 'Feature Groups' concept to 'Clusters' in the user experience, making clusters the primary steering primitive."
      - "Validate and make trustworthy that combined ('Blended') steering truly sums the influence of every feature in a cluster."
      - "Replace the misleading single-top-feature result label with a cluster-identifying label users can trust."
      - "Introduce a grounded, non-arbitrary combined-strength model: a total influence budget derived from the cluster's activation frequencies, allocated per-feature by similarity score, with budget-preserving manual rebalancing."
      - "Let users name a cluster, attach a narrative/description, and capture its tuned member strengths."
      - "Export a tuned cluster (or a group of clusters) as a portable, standardized JSON *cluster definition* file."
    success_criteria:
      - "Users see and speak of 'Clusters' throughout the steering/labeling/SAE workbench UI; 'Feature Groups' no longer appears in the interface."
      - "For any Blended run over a cluster, users can verify that all member features contributed (not just the top one), and the per-prompt result is labeled by the cluster, not a lone feature index."
      - "Selecting a cluster for steering yields an immediately-useful starting strength allocation with zero manual tuning required, derived from the cluster's frequencies and members' similarity scores."
      - "Editing one member's strength automatically rebalances the others so the cluster's total influence budget is preserved."
      - "A user can export a named, described, tuned cluster to a JSON file and re-import it into miStudio to reproduce the same steering configuration."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers / power users tuning clusters in the miStudio workbench."
      - "AI/agent operators (e.g. Claude Code via the MCP server) driving cluster discovery and experimentation."
    secondary_users:
      - "Downstream MILLM operators who will import cluster definitions in a later increment (out of scope here, noted for contract design)."
      - "Future 'cluster definition' consumers/traders (marketplace vision — future)."
    stakeholders:
      - "Product owner (Sean) — direction and acceptance."
      - "miStudio maintainers."

  scope_definition:
    in_scope:
      - "UI/UX terminology change: 'Feature Groups' → 'Clusters' across the user-facing interface (nav, labels, panels, copy)."
      - "Validation + correctness of combined-strength steering: confirm every feature in a selected cluster contributes its assigned strength in Blended/combined mode."
      - "Cluster-identifying result labels on steering output/prompt result boxes (e.g. the cluster's name / shared common token), replacing the single top-feature index."
      - "A principled combined-strength model: total influence budget as a function of the cluster's aggregate activation frequency; per-feature allocation as a function of each member's similarity score relative to the aggregate; equal sim → equal strength."
      - "Budget-preserving manual rebalance: adjusting one member's strength redistributes the remaining budget across the other members."
      - "Cluster metadata authoring: name a cluster, attach a narrative/description, persist the tuned member strengths."
      - "Portable, standardized JSON *cluster definition* export (single cluster and multi-cluster), and import back into miStudio."
      - "Carry cluster member statistics (similarity, activation_frequency, max_activation) through selection into steering so the strength model can compute (extends the existing hand-off)."
    out_of_scope:
      - "Any MILLM code or MILLM steering implementation (explicitly excluded; belongs to the follow-on BRD)."
      - "Backend/API/data-model rename of feature_groups tables/endpoints (UI terminology only this increment)."
      - "The unified MILLM+miStudio MCP server (future)."
      - "Open WebUI runtime integration / live chat dial control (future)."
      - "Cluster-definition marketplace / trading / commercialization (vision only)."
      - "MILLM cluster-scoped combined activation sensing/recording (future — see future_considerations)."
    future_considerations:
      - "Follow-on BRD: extend MILLM to IMPORT the portable cluster definition and use it in static combined steering."
      - "Unify the MCP server so a single server talks to BOTH miStudio and MILLM back ends, with per-product health checks that enable/disable product-specific tools when the corresponding back end is absent (a MILLM-only or miStudio-only host still gets a coherent, self-describing tool set)."
      - "Open WebUI integration: expose an imported cluster as a live, dial-controlled influence in real prompt/response chat sessions (off / min / max comparison against identical prompts)."
      - "MILLM cluster-scoped combined activation sensing: record ONLY the moments when ALL of a cluster's features fire together (ignoring unrelated concurrent activations), with a side-channel that distinguishes 'this cluster alone fired' from 'this cluster fired within a larger activation set' — to learn what patterns to monitor for."
      - "A 'cluster definition' marketplace enabling users to trade/share tuned clusters across models and SAE sets; potential commercialization."
      - "Standardize the exported JSON as the interchange format that makes clusters mobile across the miStudio↔MILLM ecosystem and any future consumer."
    dependencies:
      - "Existing Feature Grouping capability (Feature 010): cluster discovery, members with similarity/activation_frequency/max_activation, MCP grouping tools."
      - "Existing Steering UX (Feature 011): Blended|Compare toggle, up to 20 features, frequency auto-baseline, combined-strength generation, Groups→Steering hand-off."
      - "Existing MCP server (Feature 010) as the experimentation surface for AI-driven cluster tuning."
    assumptions:
      - "Cluster members already carry a similarity score, an activation frequency, and a max activation (verified present) — sufficient inputs for the strength model."
      - "The combined/Blended steering endpoint already sums N features server-side; the trust gap is in feedback labeling and verification, not (necessarily) the math."
      - "The portable JSON is a self-contained, model/SAE-referencing definition; portability across different models/SAEs is a later concern, not required for round-tripping within miStudio this increment."
      - "'Clusters' is the settled user-facing term; note the naming collision with the existing NLP 'semantic_clusters' concept must be disambiguated in the UI copy."

  business_requirements:
    - id: BR-001
      text: "The user-facing interface SHALL refer to feature groups as 'Clusters' everywhere they appear (navigation, steering, labeling, SAE, and related panels); the term 'Feature Groups' SHALL NOT appear in the UI."
    - id: BR-002
      text: "When steering a cluster in combined ('Blended') mode, the system SHALL apply the combined influence of ALL member features, and the user SHALL be able to verify that every member contributed its assigned strength (not only the top-listed feature)."
    - id: BR-003
      text: "Combined-steering result labels (per-prompt result boxes) SHALL identify the cluster (e.g. by cluster name or the cluster's shared common token) rather than the index of a single top feature, so results are trustworthy; the underlying member features SHALL remain discoverable via the cluster."
    - id: BR-004
      text: "When a cluster is selected for steering, the system SHALL compute a starting combined-strength allocation with no manual tuning required. The cluster's TOTAL influence budget SHALL be a principled mathematical function of the aggregate of the members' activation frequencies. Each member's share of that budget SHALL be a principled mathematical function of its individual similarity score relative to the aggregate of members' similarity scores, such that equal similarity scores yield equal strength allocation."
    - id: BR-005
      text: "The starting allocation SHALL be grounded in expected steering outcome and non-arbitrary — the system SHALL NOT require the user to begin from a guessed or useless strength. (The exact closed-form is to be validated in the PRD/TDD; see open_questions.)"
    - id: BR-006
      text: "When the user manually changes one member's strength, the system SHALL redistribute the remaining budget across the other members so the cluster's total influence budget is preserved."
    - id: BR-007
      text: "The user SHALL be able to name a cluster and attach a narrative/description that captures its discovered meaning, and the system SHALL persist the cluster's tuned member strengths alongside this metadata."
    - id: BR-008
      text: "The user SHALL be able to export a tuned cluster — and a set of multiple clusters — as a portable, standardized JSON 'cluster definition' file containing members, per-member strength allocation, name, narrative, and the source model/SAE references."
    - id: BR-009
      text: "The user SHALL be able to import a previously-exported cluster definition JSON back into miStudio and reproduce the same steering configuration (round-trip fidelity)."
    - id: BR-010
      text: "The exported JSON format SHALL be standardized and self-describing so that, in a later increment, other ecosystem components (MILLM, a unified MCP server, downstream consumers) can consume the same definition without miStudio-specific coupling."
    - id: BR-011
      text: "Cluster member statistics required by the strength model (similarity, activation frequency, max activation) SHALL be carried through cluster selection into the steering configuration."

  success_metrics:
    quantitative_metrics:
      - "0 occurrences of 'Feature Group(s)' in the shipped UI (copy audit)."
      - "100% of member features in a Blended cluster run verifiably contribute (verification check passes)."
      - "Cluster definition export→import round-trip reproduces identical member set and per-member strengths (byte/semantic equality on the reproduced config)."
      - "Starting strength allocation requires 0 manual edits to be 'usable' by the user's own judgment on first selection."
    qualitative_indicators:
      - "Users trust the combined-steering result labels (no longer suspect only the top feature is applied)."
      - "Users report the starting cluster strength 'makes sense' rather than feeling arbitrary."
      - "The exported cluster feels like a portable, self-contained artifact the user could hand to someone else."
    measurement_methods:
      - "UI copy audit / automated grep of the built frontend."
      - "End-to-end steering verification (drive a cluster Blended run; confirm all members applied)."
      - "Export→import round-trip test."
      - "User acceptance (product owner) on label trust and starting-allocation sensibility."

  feature_themes:
    core_features:
      - "Clusters rename (UI/UX terminology)."
      - "Combined-strength verification + trustworthy cluster result labeling."
      - "Principled combined-strength budget model (frequency-derived budget, similarity-weighted allocation, budget-preserving rebalance)."
      - "Cluster authoring (name + narrative + persisted tuned strengths)."
      - "Portable cluster-definition JSON export/import (single + multi-cluster)."
    secondary_features:
      - "Multi-cluster export bundles."
      - "Cluster narrative enrichment (auto-suggested from analysis)."
    future_features:
      - "MILLM import of cluster definitions + static combined steering (follow-on BRD)."
      - "Unified miStudio+MILLM MCP server with per-product health-gated tools (follow-on BRD)."
      - "Open WebUI dial-controlled live influence (off/min/max) in chat sessions."
      - "MILLM cluster-scoped combined activation sensing (record only when all cluster features co-fire)."
      - "Cluster-definition marketplace / trading / commercialization (vision)."

  considerations:
    budget_constraints: "TBD"
    timeline_expectations: >
      Incremental, sequenced: miStudio work first (this BRD), through cluster-definition export. A breather/review
      after specing the full chain, then execution. Cross-product integration follows in a separate BRD.
    regulatory_or_policy_drivers: []
    technical_constraints:
      - "UI-only rename this increment; backend/API/data 'feature_group' names may remain to limit churn."
      - "Must disambiguate 'Clusters' from the pre-existing NLP 'semantic_clusters' concept in the code/UI."
      - "The strength model must consume already-available member stats (similarity, activation_frequency, max_activation)."
    integration_requirements:
      - "This increment produces the standardized cluster-definition JSON contract that later miStudio↔MILLM integration depends on; the contract must be designed for cross-product consumption even though only miStudio produces/consumes it now."
    scalability_expectations: >
      Clusters may contain up to the current 20-feature steering cap; the strength model and export format should
      handle multi-cluster bundles.

  risks:
    - id: RSK-001
      description: "Combined steering may not actually be summing all members correctly (the user's trust concern), i.e. a real math bug hides behind the mislabeling."
      impact: "high"
      likelihood: "medium"
      mitigation: "Make verification a first-class requirement (BR-002); prove member-wise contribution end-to-end before shipping the new labels."
    - id: RSK-002
      description: "The 'right' strength formula is under-determined; a poorly-chosen closed-form could produce unhelpful starting points and undermine the whole 'never start from a useless point' goal."
      impact: "high"
      likelihood: "medium"
      mitigation: "State the mathematical intent as the requirement (BR-004/005); validate candidate formulas against steering outcomes in the PRD/TDD before committing; keep manual override + rebalance."
    - id: RSK-003
      description: "'Clusters' collides with existing NLP 'semantic_clusters'; user confusion or developer ambiguity."
      impact: "low"
      likelihood: "medium"
      mitigation: "Reserve 'Cluster' for the steering primitive in the UI; keep NLP semantic clusters as a distinct, differently-labeled concept."
    - id: RSK-004
      description: "Designing the JSON contract only for today's miStudio needs could force a breaking redesign when MILLM/Open WebUI consume it."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Design the format as self-describing and consumer-neutral now (BR-010); review it against the anticipated MILLM import before finalizing."
    - id: RSK-005
      description: "UI-only rename leaves backend 'feature_group' naming, creating a lasting vocabulary split for developers."
      impact: "low"
      likelihood: "high"
      mitigation: "Document the UI↔backend term mapping; record the full rename as a future consideration."

  next_steps:
    open_questions:
      - "Exact closed-form for the budget: what monotonic function of aggregate activation frequency yields the total budget (and in what units — the existing raw steering coefficient)? Must be validated against steering outcomes, not guessed."
      - "Exact closed-form for per-feature allocation from similarity scores (e.g. normalized share sim_i / Σsim; softmax; other) — which best matches desired outcomes while satisfying 'equal sim → equal strength'?"
      - "How does the frequency-derived TOTAL budget reconcile with the existing per-feature frequency auto-baseline (Feature 011)? Is the cluster budget a replacement for, or a layer above, per-feature baselines?"
      - "Where is a cluster's 'name' and 'narrative' authored and stored — extend the existing cluster (feature group) record, or a new steering-cluster entity?"
      - "Precise JSON schema for the cluster definition (fields, versioning, model/SAE reference identifiers, how members are keyed)."
      - "Result-label content: cluster name vs shared common token vs both — and behavior when a cluster is unnamed."
    recommended_actions:
      - "Proceed to 002_create-project-prd.md: extend 000_PPRD|miStudio.md to fold these enhancements in as if planned from the start (new feature rows + sections), keeping the miStudio↔MILLM integration vision visible."
      - "Then author the needed Feature PRD(s) — likely: (a) Clusters rename + trustworthy combined labeling, (b) principled combined-strength budget model + rebalance, (c) cluster authoring + portable JSON export/import. Exact feature count to be finalized during PRD."
      - "Update the ADR with the strength-model decision and the cluster-definition JSON contract, then produce TDDs/TIDs/task lists per feature."
    priority_for_clarification:
      - "The two strength closed-forms (budget + allocation) — highest leverage, blocks BR-004/005."
      - "The JSON cluster-definition schema — blocks BR-008/009/010 and the whole ecosystem-integration arc."
```
