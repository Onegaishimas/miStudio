# BRD SEED — Multi-Layer Dictionary Substrate Migration (miStudio future increment)

> **Seed document, not a BRD.** This file pre-stages `BRD-MIS-SUBSTRATE-001`. It is NOT an input to the
> XCC chain and authorizes no implementation work. It becomes a BRD only if the substrate pilot defined
> in `BRD-MIS-CIRCUITS-002` Addendum B returns **GO**, at which point this seed is expanded via
> `0xcc/instruct/001_generate-brd.md` with the pilot's measured numbers filled into §4. Sequencing on GO:
> after BRD-MILLM-CIRCUITS-001 (the miLLM circuits runtime), so the migration lands on a stable
> two-product circuit ecosystem rather than a moving one.
>
> **Standing constraints inherited regardless of outcome:** the interchange family's additive-only rule
> (substrate-backed artifacts arrive as NEW kinds, never mutations of cluster-definition/v1 or
> circuit-definition/v1); the evidence-ladder claims discipline of CIRCUITS-002 BR-026; badge-not-gate;
> single-GPU (RTX 3090, 24GB) deployment envelope unless separately revised.

```yaml
brd_seed:
  metadata:
    seed_for: BRD-MIS-SUBSTRATE-001
    project_name: "miStudio — Multi-Layer Dictionary Substrate Migration"
    status: "seed / gated on pilot GO (BRD-MIS-CIRCUITS-002 Addendum B)"
    author: "Sean (direction) with external methodology review, 2026-07-19"
    depends_on_outcome_of: "Substrate pilot (Addendum B, gates B.3.1–B.3.5)"
    sequenced_after: "BRD-MILLM-CIRCUITS-001"

  problem_statement_draft: >
    The per-layer-SAE substrate forces the product to reconstruct cross-layer meaning from outside:
    circuits are mined as edges between independently-trained dictionaries, residual echoes must be
    detected and typed away, per-layer steering allocations must be computed by a budget model, and
    attribution must bridge dictionaries that never saw each other. Multi-layer dictionaries invert
    this: depth-persistent concepts become single features, cross-layer steering directions are learned
    at training time, and echo edges cease to exist as a category. The pilot has demonstrated (numbers
    in §4) that this inversion is achievable at parity on reconstruction and interpretability. The cost
    is a substrate migration: retraining, re-extraction, re-labeling, artifact continuity, and contract
    evolution across the miStudio↔miLLM ecosystem.

  vision_statement_draft: >
    Move the workbench's unit of meaning from "a feature at a layer" to "a feature through depth":
    dictionaries whose features natively span layers, steer through their own learned per-layer
    directions, and expose cross-layer computation as structure WITHIN the dictionary rather than
    edges mined around it — while preserving every trust guarantee (evidence ladder, manifests,
    verification, portable contracts) and carrying forward, not orphaning, the curated knowledge
    (labels, clusters, validated circuits) accumulated on the per-layer substrate.

  candidate_scope_themes:
    - "Substrate training pipeline: production-grade multi-layer dictionary training (architecture per pilot outcome — crosscoder default) over the full layer window, with acceptance gates lifted from the pilot harness."
    - "Extraction & labeling v2: feature pages that render a per-layer activity profile (decoder-norm / activation-by-layer) as the answer to 'where does this feature live'; labeling workflow reuse."
    - "Artifact continuity ('the bridge'): a mapping layer from incumbent features/clusters/circuits to pilot-substrate features (decoder-similarity + activation-correlation matching, as in pilot gate B.3.1), so curated labels, the 30 clusters, and validated circuits are MIGRATED with provenance and a rung reset policy — validated claims do not silently transfer across substrates; they re-validate cheaply via the existing manifest machinery."
    - "Steering v2: native multi-layer steering through a feature's own per-layer decoder directions; the strength-budget model recast as scaling a learned allocation rather than composing per-layer budgets; hazard detection simplification (intra-feature compounding is gone by construction; inter-feature hazards remain)."
    - "Discovery v2: CIRCUITS-001/002 pipeline re-based — persistence typing retires as a category; mining/attribution/validation/faithfulness target genuinely-computed structure between multi-layer features; expected simplification of the triviality filter documented and measured."
    - "Contract evolution: new interchange kind(s) for substrate-backed artifacts (e.g. mistudio.mlfeature-definition/v1, and a circuit-definition revision carrying multi-layer member refs), plus a PROJECTION to the existing per-layer kinds so current consumers (miLLM as shipped) keep working unchanged — same discipline as CIRCUITS-001 BR-014."
    - "Dual-substrate operation window: incumbent and new substrate co-hosted during migration with explicit substrate identity on every artifact; no silent mixing."
    - "MCP surface extension for substrate identity, bridge queries, and re-validation runs."

  candidate_out_of_scope:
    - "Cross-layer transcoders / full attribution-graph substrate (remains the Tier-3 decision; this migration is dictionary-substrate only unless the pilot outcome says otherwise)."
    - "Cross-model substrates (one model's dictionary; correspondence across models stays future)."
    - "miLLM runtime changes beyond consuming projections (any native multi-layer serving is its own follow-on, mirroring the established sequencing pattern)."

  success_criteria_draft:
    - "Echo category retired: persistence-typed edges in default discovery output drop to a negligible reported fraction, replaced by single features with multi-layer activity profiles."
    - "Zero-allocation steering: selecting a multi-layer feature steers through its learned per-layer directions with quality ≥ the incumbent validated profiles (pilot gate B.3.4 reproduced at production scale)."
    - "No orphaned knowledge: 100% of curated labels/clusters/validated circuits either bridge with recorded mapping confidence or are explicitly recorded as non-bridging; re-validation of bridged causal claims is a batch operation, not a manual redo."
    - "Contract safety: all existing kinds validate unchanged; projections keep shipped consumers working; substrate identity is unambiguous on every artifact."
    - "Evidence ladder intact: every claim on the new substrate carries a rung and a manifest from day one."

  known_risks_draft:
    - "Migration fatigue: re-extraction/re-labeling at 16-layer scale dwarfs the pilot; mitigation — bridge-first migration (labels transfer with confidence scores; human review prioritized by bridge uncertainty)."
    - "Rung-reset trust cliff: users see previously-validated circuits demoted pending re-validation; mitigation — batch re-validation tooling before the substrate becomes default."
    - "Localization regressions in review UX (pilot gate B.3.5 was qualitative); mitigation — per-layer activity profile as a first-class feature-page element, acceptance-reviewed."
    - "Single-GPU training ceiling for full-window dictionaries; mitigation — layer-window sharding strategy from the pilot harness, or a revised hardware envelope decision taken explicitly in the BRD."

  open_questions_for_brd_expansion:
    - "Architecture lock: crosscoder vs MLSAE per pilot numbers; dictionary width and L0 at production scale."
    - "Rung reset policy details: does attribution (rung 1) evidence bridge, or only mined (rung 0)?"
    - "Dual-substrate window length and the default-substrate flip criterion."
    - "Whether circuit-definition evolves as v2 of the existing kind or a parallel kind (interchange-family review required before freeze, as always)."
    - "Hardware envelope: does production training justify revisiting the single-3090 constraint?"

  expansion_instructions:
    - "On pilot GO: run 0xcc/instruct/001_generate-brd.md with this seed + the pilot report as inputs; fill §4 of the problem statement with measured gate numbers; resolve open questions with the product owner; then proceed down the standard chain (002 PPRD update → 003 ADRs → 004 feature PRDs)."
    - "On pilot NO-GO: retain this seed unchanged; append the pilot report reference and the re-queue triggers recorded per Addendum B.4."
```
