# BRD — J-Space Capability: Jacobian Lens Artifacts, Workspace Analysis & Runtime Monitoring Handoff (miStudio increment)

> **New BRD, new capability family.** This document opens the `JSPACE` family. It does not amend
> `BRD-MIS-CIRCUITS-001/002` or `BRD-MIS-CLUSTERS-001`, but it is adjacent to all three and must be read
> against them: J-space is a *second dictionary substrate* that arrives alongside the SAE substrate rather
> than replacing it, and it annotates the existing one. Per the XCC workflow the next step is
> `0xcc/instruct/002_create-project-prd.md`.
>
> **Version 0.3 — revised on the model-agnosticism directive.** v0.2 was written assuming the reference
> model would be `gemma-2-2b` and that lens artifacts would usually be *downloaded*. The product owner has
> directed (2026-07-31) that the capability MUST work with **any model the workbench can load**, and that
> **`LFM2.5-1.2B-Instruct` is the reference model**. That single decision invalidates four v0.2 positions.
> They are corrected here rather than quietly patched, in the manner the conformance assessment
> established.
>
> **Changes in v0.3:**
> 1. **BR-031 inverted — CONSTRUCTION is the primary path.** v0.2 said "prefer ACQUISITION over
>    CONSTRUCTION" and called fitting "the exception". Pre-fitted lenses exist for 36 models; LFM2 is not
>    among them, and neither is most of what this workbench runs. Acquisition is retained as an
>    opportunistic optimization when a conformant lens happens to exist for the exact weights in use, but
>    the product SHALL NOT depend on it. Fitting is a first-class, supported path.
> 2. **Reference model is `LFM2.5-1.2B-Instruct`**, not `gemma-2-2b`. Note the workbench holds
>    `gemma-2-2b-**it**` — the instruction-tuned variant, whose weights differ from the base model the
>    upstream lens is fitted for, so that artifact would not have been valid regardless. This retires
>    RSK-015's premise rather than mitigating it.
> 3. **Hybrid architectures are first-class, and they break two recipe assumptions.** LFM2 interleaves
>    10 `conv` layers with 6 `full_attention` layers over 16 total. The frozen-Q/K recipe variant
>    (Appendix A.2 choice 2) is UNDEFINED on a conv layer, and the attention-broadcast metrics
>    (Appendix A.5) are computable on 6 of 16 layers only. Both SHALL be recorded per layer as applicable
>    or inapplicable — never averaged over whatever layers happened to qualify. See new BR-032.
> 4. **The layer-subsampling open question is RESOLVED for small models.** A 16-layer model is analyzed in
>    full; subsampling to 25 evenly spaced layers is meaningless below 25 layers. The question survives
>    only for models larger than the subsampling target, and is no longer blocking for BR-002 at the
>    reference scale.
> 5. **New BR-032** — model-agnostic construction through miStudio's own architecture discovery.
> 6. **BR-027 broadened to FULL MCP parity** (product-owner direction, 2026-07-31). Every J-space
>    capability reachable in the workbench must be reachable by an agent, with tools delivered
>    alongside the feature that creates them rather than batched into the final feature, and every
>    tool covered by the reachability harness that exists because 16 implemented-but-unregistered
>    tools once shipped green.
> 7. **Envelope arithmetic is model-derived, not constant** (Appendix A.1). For LFM2 the required artifact
>    is ~134 MB against ~4.3 GB materialized — a ratio of ~32×, not the ~111× quoted for gemma, because
>    LFM2's vocabulary is four times smaller. The rule holds; the constant does not.
>
> **Version 0.2 — revised against verified upstream source.** v0.1 was written from the source paper and
> product manuals alone. v0.2 incorporates `0xcc/brds/neuronpedia-jlens-conformance.md`, a Phase 0
> conformance assessment verified directly against `anthropics/jacobian-lens` @ HEAD and
> `hijohnnylin/neuronpedia` @ HEAD (webapp, inference server, Prisma schema, migration
> `20260619032021_add_jlens`, `utils/neuronpedia-utils/neuronpedia_utils/jlens/`). Several v0.1 assumptions
> were wrong and are corrected here rather than quietly patched.
>
> **Changes in v0.2 (all traceable to the conformance assessment §7):**
> 1. **BR-022 restructured into two independent tracks.** There is *no upload path for J-lens data into
>    Neuronpedia*, local or hosted — its entire database footprint is two tables persisting shared analysis
>    sessions. J-lens is compute-on-demand from a mounted artifact. Track A (artifact supply) is a mounted
>    directory and one environment variable; Track B (SAE workspace annotation) is the only track that uses
>    the existing feature upload path. Building a J-lens ingestion API is now an explicit **non-goal**.
> 2. **RSK-001 downgraded and narrowed.** Neuronpedia ships lens support down to `pythia-70m`,
>    `gpt2-small`, and `gemma-3-270m`. The scale question is no longer "does the lens work at 2B" — it
>    demonstrably does — but "does the full workspace claim set replicate at 2B", which is a finding to
>    characterize rather than a gate that can strand the increment.
> 3. **Corpus-size guidance corrected** (Appendix A.2). v0.1 said to plan at ~10 sequences. Both reference
>    implementations disagree: usable floor is **100**, with convergence-based early stopping.
> 4. **Artifact dtype corrected to fp16** (Appendix A.1). v0.1 said bf16.
> 5. **Phase 0 rescoped from build to run-and-verify** (BR-001). The evaluation harness, six evaluation
>    sets, eleven experiment fixtures, fitter, applier, and slice visualization all already exist upstream.
> 6. **Three new requirements:** BR-029 (mirror Neuronpedia's lens stream wire format), BR-030 (artifact
>    validation suite — required because Neuronpedia's lens loading fails *silently*), BR-031 (acquire
>    pre-fitted lenses before fitting anything; contribute upstream for uncovered models).
> 7. **BR-006 upgraded from reasoned to externally validated** — neither upstream codebase materializes the
>    token dictionary.
> 8. **BR-017 gains a small-model default** — coordinate swaps oversteer at small scale and need fewer
>    layers selected.
> 9. **Two new risks:** RSK-014 (silent artifact failure) and RSK-015 (acquisition assumption failure).
>    **Two open questions resolved.**
>
> **Process note.** `001_generate-brd.md` requires clarifying questions to be answered before the BRD is
> written. That gate was not run as a separate turn: the source material was unusually complete and the
> product owner asked for the BRD directly. Every item a clarifying round would have surfaced is carried
> explicitly as `"TBD"` in-place and enumerated in `next_steps.open_questions` /
> `priority_for_clarification`. v0.2 resolves two of v0.1's six blocking questions from verified source.
>
> **Source research.** Gurnee, Sofroniew, et al., *"Verbalizable Representations Form a Global Workspace
> in Language Models,"* Transformer Circuits Thread, 2026-07-06. Numeric targets deriving from the paper
> are marked with their origin, because they were measured on Claude 4.5-family production models. §9.1 of
> the paper states that whether smaller models have an equally rich workspace is open. v0.2 narrows but does
> not eliminate that uncertainty — see RSK-001.
>
> **Audience note (agentic implementation).** As with CIRCUITS-002, the primary implementer is an AI coding
> agent executing the XCC chain. Appendix A is normative supporting material, not commentary. Where a BR and
> Appendix A overlap, the BR states WHAT must be true; Appendix A states HOW it is computed.
>
> **Locked decisions (product owner direction, 2026-07-29; amended 2026-07-30):**
> (1) **Phase 0 is a GO/NO-GO gate, not a milestone** — now scoped as run-and-verify against existing
> upstream assets rather than a build.
> (2) **Logit lens first.** The readout surface, analysis suite, and intervention primitives are built and
> validated against the logit lens (`J_ℓ = I`) before any Jacobian artifact exists. Independently
> corroborated: Neuronpedia ships `LOGIT_LENS` as a peer tab requiring no artifact, with a `DIFF` mode
> against `JACOBIAN_LENS` that serves as a free regression surface.
> (3) **Never materialize `W_U J_ℓ`.** Storage is the per-layer `d_model × d_model` matrix only.
> (4) **J-space is additive to the SAE substrate.** New interchange kinds; no mutation of
> `cluster-definition/v1` or `circuit-definition/v1`; projections keep miLLM-as-shipped working unchanged.
> (5) **Readout is not a causal claim.** J-space evidence enters the existing evidence ladder at defined
> rungs; a readout may never be presented in intervention language. Badge-not-gate throughout.
> (6) **Two fields, not one, for SAE J-alignment.** Motor features share high lens-kurtosis with workspace
> features; geometric and behavioral classification are separate.
> (7) **Sequencing.** miStudio lands first; the runtime consumer follows as `BRD-MILLM-JSPACE-001`.
> (8) **Claims discipline on experiential findings.** The source paper explicitly takes no position on
> phenomenal consciousness. Neither does this product. Stated requirement (BR-024), not a disclaimer.
> (9) **NEW — Buy before build on lens artifacts.** Pre-fitted lenses exist for 36 models. Fitting is the
> exception, not the default path.
> (10) **NEW — Conform, don't parallel.** miStudio's readout transport mirrors Neuronpedia's lens stream
> wire format exactly, so contract compatibility is structural rather than a translation layer.

```yaml
brd:
  metadata:
    brd_id: BRD-MIS-JSPACE-001
    project_name: "miStudio — J-Space Capability: Jacobian Lens Artifacts, Workspace Analysis & Runtime Monitoring Handoff"
    version: "0.3"
    author: "Sean (product owner) with source-paper analysis, design review, and verified upstream conformance assessment"
    last_updated: "2026-07-31"
    status: "draft"
    increment_of: "miStudio (000_PPRD|miStudio.md)"
    opens_family: "JSPACE"
    adjacent_to: "BRD-MIS-CLUSTERS-001, BRD-MIS-CIRCUITS-001, BRD-MIS-CIRCUITS-002 (shared evidence ladder, shared interchange family, shared capture store)"
    sequenced_before: "BRD-MILLM-JSPACE-001 (runtime monitoring consumer)"
    source_research: "Gurnee, Sofroniew, et al., 'Verbalizable Representations Form a Global Workspace in Language Models', Transformer Circuits Thread, 2026-07-06"
    conformance_assessment: "0xcc/brds/neuronpedia-jlens-conformance.md — Phase 0 deliverable satisfying BR-022; its section 7 is the change log for this revision"
    verified_against: "anthropics/jacobian-lens @ HEAD (reference implementation, evaluation sets, experiment fixtures); hijohnnylin/neuronpedia @ HEAD (webapp jlens components, inference server lens loader, Prisma schema, migration 20260619032021_add_jlens, neuronpedia_utils/jlens fitter)"
    reference_assets: "Pre-fitted lenses for 36 models at neuronpedia/jacobian-lens (HuggingFace); miStudio J-Space readout panel reference implementation (React/Tailwind/recharts, conforming to the upstream lens wire format)"

  business_context:
    problem_statement: >
      miStudio's unit of meaning is a learned SAE latent, discovered by dictionary training, named by an
      LLM labeler reading top-activating examples, and validated causally through steering. That substrate
      has a blind spot the source paper makes concrete: a large fraction of a model's representational
      content sits outside the small, privileged subset of directions the model can actually report,
      modulate, and reason with — and the workbench currently cannot tell the two apart. Three consequences
      are already costing us. First, labeling from top-activating examples silently mislabels exactly the
      abstract-assessment features that matter most for safety work: the paper documents an SAE feature
      whose activating contexts are technical prose (neutron capture, a conference announcement, a Hubble
      constant measurement) and which any example-driven labeler would call "technical exposition," but
      whose verbalizable content is fake / fraud / suspicious — it is a fabricated-content detector, and
      suppressing it cut the model's stated recognition of an artificial scenario from 28/50 to 10/50.
      Second, the workbench has no cheap readout of what a model is about to be able to say: every
      interpretive question routes through dictionary training, whose cost gates experimentation. Third,
      and most commercially significant, miLLM serves models in real deployments with no way to observe
      whether the concepts driving a response are present in the model's reportable workspace — no runtime
      signal for evaluation-awareness, deceptive intent, or silently-assembled situational judgments, all
      of which the paper shows surface in the workspace at decision positions before any output token is
      produced.
    vision_statement: >
      Give miStudio a second, token-indexed dictionary substrate that requires no training: a per-layer
      linear map from which any vocabulary token's residual-stream direction can be synthesized on demand.
      Use it to (a) read what a model is poised to say at every layer and token position, (b) annotate the
      existing SAE dictionary with whether each learned feature lives in the reportable workspace or
      outside it, (c) interpret model components — SAE decoders, transcoder encoder/decoder pairs, attention
      weight matrices — directly from weights with no activations at all, and (d) emit a small, cheap,
      contract-stable watchlist artifact that lets miLLM turn interpretability into runtime observability:
      per-token probes against named concepts, streamed alongside inference, at the cost of an inner product.
      Preserve every existing trust guarantee — evidence ladder, manifests, verification, additive-only
      portable contracts, badge-not-gate — and treat the reportable/non-reportable distinction as a new
      dimension of the workbench rather than a replacement for anything in it.
    primary_objectives:
      - "Stand up the J-lens artifact as a first-class, provenance-tracked miStudio artifact class alongside SAEs, with the storage, recipe, and on-disk-conformance decisions locked (Appendix A.1–A.2, A.10)."
      - "Answer, as an explicit gated deliverable, whether the full workspace claim set replicates at the model scales miStudio serves — and publish the answer either way."
      - "Deliver a position × layer readout surface that makes 'what is the model poised to say here' a routine question in the workbench, not a research project."
      - "Annotate the existing SAE dictionaries with workspace membership so accumulated curated knowledge (labels, the 30 clusters, validated circuits) gains a reportability dimension without re-training anything."
      - "Extend the intervention engine with the primitives the causal claims require — projective ablation, dynamic top-k workspace ablation, lens-coordinate swap, and paired-run clamping — which the current static steering config cannot express."
      - "Emit the artifacts the runtime consumer needs under an additive interchange contract, so BRD-MILLM-JSPACE-001 starts from a frozen surface."
      - "Serve a conformant lens artifact to a local Neuronpedia instance and export SAE workspace annotations through the existing feature upload path, conforming to upstream representations rather than paralleling them."
    success_criteria:
      - "Phase 0 replication: the six methodological-evaluation distributions from the source paper (multihop n=50, multilingual n=54, order-of-operations n=55, poetry n=52, typo n=96, association n=50) run in miStudio against the vendored upstream evaluation data, and the J-lens beats the logit-lens baseline on normalized pass@k AUC on the reference model — or the failure to do so is characterized and reported."
      - "Phase 0 gate: a workspace-band report exists for at least one served open model, with the five layer-band metrics plus occupancy and excess-FVE, and a recorded GO/NO-GO on whether that model exhibits the full workspace claim set."
      - "Storage discipline holds: the J-lens artifact for the reference model occupies the order of hundreds of MB, not tens of GB (Appendix A.1); a materialized-dictionary regression is caught by an envelope check in CI."
      - "Every J-lens readout, probe score, decomposition, and intervention in the product is reproducible from its recorded provenance alone — recipe variant, corpus, aggregation, solver parameters, and control seeds included."
      - "The existing cluster-definition/v1 and circuit-definition/v1 validation suites pass unchanged, and miLLM-as-shipped consumes projections of the new kinds without a runtime change."
      - "A reviewer looking at any surfaced workspace claim can tell which evidence rung it sits on and cannot mistake a readout for an intervention result."
      - "Track A conformance: a miStudio-supplied lens artifact mounted via the local instance's lens-source override is loaded and served, verified by an explicit request-time test rather than a clean boot log."
      - "Track B conformance: exported SAE workspace annotations validate against the local instance's feature schema and round-trip with semantic equality."

  stakeholders_users:
    primary_users:
      - "Interpretability researchers using miStudio as their workbench — the existing primary user, who gains a training-free readout and a reportability dimension on their dictionaries."
      - "Alignment / safety auditors running evaluation suites who need to see silent strategic and situational cognition at the token positions where decisions are made."
      - "miLLM operators who need runtime observability over what a served model is poised to say."
    secondary_users:
      - "Automated auditing agents consuming the readout through the MCP surface (source paper §A.22 demonstrates an agent equipped with a top-25 lens readout scoring comparably to natural-language-autoencoder tooling and better than SAE tooling, at far lower cost)."
      - "Operators of the local Neuronpedia instance, as consumers of the mounted lens artifact and the uploaded workspace annotations."
      - "The Neuronpedia community and upstream maintainers, as recipients of contributed lenses for uncovered models and of the SAE workspace annotation as a schema proposal."
      - "Downstream miStudio features — Circuit Discovery, Clusters, Steering — which gain workspace-membership filtering on their inputs."
    stakeholders:
      - "Product owner (Sean) — scope, gate decisions, claims discipline."
      - "Human in the Stream, LLC — commercial positioning of runtime observability."
      - "External methodology review — as with CIRCUITS-002, contract and statistics review before v1 freeze."
      - "Upstream maintainers (Neuronpedia) — schema conformance and lens contribution. Note the reference implementation repository is explicitly unmaintained and not accepting contributions; Neuronpedia is actively soliciting pre-fitted lenses."

  scope_definition:
    in_scope:
      - "Phase 0 — Replication & Gate: vendor the upstream reference implementation and evaluation data at a recorded commit; RUN its existing harness on a served open model; produce the workspace-band report; record GO/NO-GO."
      - "Logit-lens readout path (J_ℓ = I) as the first-delivered, zero-artifact readout, and as the substitution point for the Jacobian artifact."
      - "J-lens artifact class: acquisition (pre-fitted first), construction where necessary, storage, versioning, provenance, envelope reporting, on-disk conformance, and lifecycle integration with the existing checkpoint-lifecycle machinery."
      - "Artifact validation suite: structural, naming, envelope, semantic, cross-implementation, and round-trip checks (Appendix A.10.6), run before any artifact is handed to a consumer."
      - "Three distinct readout modes: full-vocabulary ranked readout; single-direction probe; sparse non-negative decomposition by gradient pursuit."
      - "Readout transport conforming to Neuronpedia's lens stream wire format, so miStudio's own readout stream and any Neuronpedia-sourced stream are interchangeable."
      - "Position × layer readout viewer: hoverable top-1 grid, pinned-token rank heatmap, per-position and per-layer full readouts, rank-vs-layer trajectories, lens-mode switching including a diff view."
      - "Structural analysis suite: the five layer-band metrics, cross-layer CKA, occupancy, excess fraction-of-variance-explained, and the weight-space broadcast metrics (MLP gain; attention gain + label preservation; broadcast-head identification)."
      - "SAE dictionary annotation: per-feature lens kurtosis, motor/workspace/outside classification, and a label-disagreement flag where the auto-label and the lens readout diverge."
      - "Weight-space component readouts for SAE decoder directions, transcoder encoder/decoder pairs, and attention Q/K/V/O matrices."
      - "Intervention primitives: additive steering along a lens direction; projective ablation; dynamic top-k J-space ablation with clean-pass exclusion; lens-coordinate swap; and the paired-run-with-clamping execution mode all of the above mediation analyses require."
      - "Evidence-ladder integration: rung assignment for J-space evidence types and enforcement that readout evidence is never presented in causal language."
      - "Interchange contract: new additive kinds for the J-lens artifact, the workspace annotation, and the runtime watchlist, plus projections to existing kinds."
      - "Track A — J-lens artifact supply to a local Neuronpedia instance: conformant on-disk layout mounted via the instance's local-source override."
      - "Track B — SAE workspace annotation export through the existing Neuronpedia feature/explanation upload path."
      - "Watchlist authoring and export: named concept sets (including a validated evaluation-awareness score) that miLLM can evaluate per token; this BRD owns the authoring/export side, miLLM owns evaluation."
      - "MCP surface extension for readout queries, workspace annotation queries, and watchlist management."
      - "Template lens (multi-token concepts) as a scoped, forward-passes-only extension — see BR-023; may land as a fast-follow but its contract fields are day-one."
    out_of_scope:
      - "A J-lens ingestion API against Neuronpedia. NEW EXPLICIT NON-GOAL in v0.2: no such path exists upstream — J-lens is compute-on-demand from a mounted artifact, and Neuronpedia's entire J-lens database footprint is two tables persisting shared analysis sessions. Building one would mean building a Neuronpedia feature that does not exist."
      - "The oracle lens (source paper §A.9.2). It requires fine-tuning two auxiliary models including an RL stage; the cost is out of proportion to this increment and outside the single-GPU envelope."
      - "Any runtime evaluation of watchlists or streaming of readouts during inference — that is BRD-MILLM-JSPACE-001. This increment emits the artifacts and the contract only."
      - "Counterfactual reflection training (source paper §7). It requires gradient updates to the served model; miStudio trains dictionaries, not models. The evaluation harness for such training is a legitimate future increment; the training itself is not this product."
      - "Training a workspace-aligned SAE (a J-space-aware dictionary objective). Deferred pending Phase 0 and the template-lens outcome."
      - "Multi-layer / crosscoder substrate work — remains BRD-MIS-SUBSTRATE-001, gated separately."
      - "J-lens attribution graphs (source paper §A.24.2) beyond a feasibility spike; see future_considerations."
      - "Cross-model workspace correspondence."
      - "Any product claim, UI copy, or documentation asserting or implying phenomenal consciousness or subjective experience in a served model."
    future_considerations:
      - "J-lens attribution graphs: nodes from sparse decomposition, non-J-space content collected into remainder nodes, edges by one-layer backprop with attention and normalization frozen. Requires no dictionary training and every node is labeled by construction — a material shortcut over the transcoder path. Deferred because the source paper is candid that most output influence flows through remainder nodes and the greedy decomposition can land on near-miss correlates."
      - "Workspace emergence across pretraining: the same pipeline pointed at a public checkpoint series (Pythia, OLMo) answers a question the source paper names as open (§9.1). Materially cheaper than v0.1 assumed, since Neuronpedia's model map already covers pythia-70m-deduped and the reference implementation's layout auto-detection handles the GPT-NeoX family. Strong candidate for the next JSPACE increment."
      - "Workspace-aligned dictionary training, if Phase 0 is GO and the template lens proves insufficient on multi-token concepts."
      - "Reflection-training evaluation harness."
      - "Quantifying the abstract-vs-syntactic split across a full labeled dictionary at scale — cheap, publishable, and only possible because miStudio holds labels at scale."
      - "Contributing pre-fitted lenses for models Neuronpedia lacks, as an ongoing relationship rather than a one-off."
    dependencies:
      - "anthropics/jacobian-lens — reference fitter, applier, slice visualization, six evaluation sets, eleven experiment fixtures. Vendor at a recorded commit; the repository is explicitly unmaintained and not accepting contributions, so expect no upstream fixes."
      - "neuronpedia/jacobian-lens (HuggingFace) — pre-fitted lenses for 36 models, largest Llama 70B. Primary acquisition path (BR-031)."
      - "hijohnnylin/neuronpedia — local instance: inference server lens loader, its local-source override, the np_model_to_hf model map, lens stream wire format, and the existing feature/explanation upload path."
      - "Existing miStudio activation-capture pipeline, background job runner, and provenance/DB layer."
      - "Existing evidence-ladder implementation (CIRCUITS-002 BR-026) as the claims vocabulary."
      - "Existing Neuronpedia export service, package format, and dashboard payload (the dashboard table already carries a logit-lens payload field, the natural extension point for Track B)."
      - "miLLM's residual-stream hook points and Socket.IO activation-monitoring channel, as the consumer-side integration surface. Read-only dependency this increment."
      - "A served open model of sufficient scale. gemma-2-2b appears in Neuronpedia's model map, so the reference model is supported upstream."
    assumptions:
      - "REVISED v0.3 — The reference model for Phase 0 is LFM2.5-1.2B-Instruct (d_model 2048, 16 layers, vocab 65536, hybrid: 10 conv + 6 full_attention). It is NOT in Neuronpedia's pre-fitted set, so Phase 0 includes fitting. Its artifact is ~134 MB (16 x 2048^2 x 2 bytes) against ~4.3 GB if the dictionary were materialized. In fp16 the model is ~2.4 GB and fits the local RTX 3080 Ti (12 GB) with ample room for the backward pass, so lens work does not contend with the cluster GPU serving miLLM."
      - "REVISED v0.3 — Two GPUs are now available: the cluster RTX 3090 (24 GB, occupied by miLLM serving) and a local RTX 3080 Ti (12 GB) on mcs-lnxgpu02. Reference-model lens work targets the local card. Where fitting is necessary, it parallelizes by splitting the corpus across runs and merging, not by splitting the model."
      - "Lens fitting, where required, uses convergence-based stopping with a floor of 100 prompts — NOT the ~10-sequence figure assumed in v0.1. See Appendix A.2."
      - "Sparse decomposition is non-unique by construction; reproducibility comes from pinned solver parameters and control seeds, not from mathematical uniqueness."
      - "Neuronpedia's feature-centric data model accommodates the SAE workspace annotation (Track B). To be verified against the running local instance's actual schema before building, since the feature surface has moved since miStudio's export was written."
      - "Neuronpedia's lens loading is best-effort and fails silently at request time rather than at startup, so miStudio must validate artifacts before handover (BR-030)."

  business_requirements:
    # New family; numbering starts at BR-001. BR-029..031 added in v0.2.

    # --- Phase 0: replication and gate ---
    - id: BR-001
      theme: "Replication & gate"
      revised_in: "v0.2 — rescoped from build to run-and-verify"
      text: >
        The system SHALL reproduce the source paper's methodological evaluation on a served open model
        before any new product surface is built. The reference implementation already ships the fitter,
        the applier, the six evaluation distributions, and eleven experiment fixtures reproducing the
        paper's headline results, so this requirement is RUN AND VERIFY, not build. Vendor at a recorded
        commit and report, per lens (logit / J-lens / tuned where available): normalized pass@k AUC for
        intermediate-concept recovery across the six distributions, ablation KL divergence, and
        lens-coordinate swap success rate. Results SHALL be recorded as a first-class report artifact with
        full provenance, and SHALL be published whether favourable or not.
    - id: BR-002
      theme: "Replication & gate"
      text: >
        The system SHALL produce a per-model WORKSPACE BAND REPORT comprising: J-lens top-k next-token
        agreement by layer; excess kurtosis of the readout distribution by layer; top-1 readout
        autocorrelation across positions against a position-shuffled null; effective linear dimensionality
        of the lens dictionary; cross-layer CKA of the lens geometry; sparse-decomposition occupancy; and
        fraction-of-variance-explained in excess of a size-matched random-direction control. The report
        SHALL derive that model's own sensory / workspace / motor layer boundaries. Layer bands from the
        source paper (~L38–92 on a reindexed 0–100 scale, Sonnet 4.5) SHALL NOT be ported to any other
        model, and the product SHALL make porting them impossible by construction.
    - id: BR-003
      theme: "Replication & gate"
      revised_in: "v0.2 — gate question narrowed"
      text: >
        The workspace-band report SHALL terminate in an explicit recorded GO / NO-GO / GO-AT-LARGER-SCALE
        decision. The question is NOT whether the lens produces usable readouts at the served scale — that
        is settled upstream, where lens support ships for models as small as 70M parameters — but whether
        the full WORKSPACE CLAIM SET (band structure, selectivity, flexible generalization, capacity
        limits) replicates. Product surface work beyond the logit-lens readout viewer SHALL NOT begin until
        that decision is recorded. A NO-GO SHALL be a supported terminal outcome producing a publishable
        negative result, not a blocked project.
    - id: BR-004
      theme: "Replication & gate"
      text: >
        Next-token agreement with the model's own output distribution SHALL NOT be used as a quality
        metric for the J-lens anywhere in the product, in CI, or in any report. The source paper
        establishes that the J-lens is deliberately worse on this measure than the logit lens through most
        of the network, and that the directions best predicting the output are not the directions that
        best expose or causally drive the computation producing it. Any dashboard, gate, or acceptance
        test that rewards next-token agreement is a defect.

    # --- Readout substrate ---
    - id: BR-005
      theme: "Readout substrate"
      text: >
        The system SHALL implement the logit-lens readout path (`J_ℓ = I`) as a complete, shippable
        capability requiring no precomputed artifact, and SHALL build the readout viewer, the structural
        analysis suite, the decomposition routines, and the intervention primitives against it FIRST. The
        Jacobian artifact SHALL substitute at a single call site with no change to any consumer. This
        sequencing is independently corroborated upstream, where the logit lens ships as a peer lens mode
        requiring no artifact, alongside a diff mode against the Jacobian lens that serves as a free
        regression surface.
    - id: BR-006
      theme: "Readout substrate"
      revised_in: "v0.2 — externally validated"
      text: >
        The system SHALL construct and store the J-lens artifact as one `d_model × d_model` matrix per
        analyzed layer. It SHALL NOT materialize or persist the full token-indexed dictionary
        (`W_U J_ℓ`). Individual token directions SHALL be synthesized on demand and cached by working set.
        Rationale and arithmetic are normative in Appendix A.1; for the reference model the difference is
        roughly 276 MB versus roughly 31 GB, and it recurs worse at larger scale. This is now VERIFIED
        rather than reasoned: neither the reference implementation nor Neuronpedia's inference server
        materializes the dictionary — the server's transport step is one matmul per layer with the model's
        own unembedding applied afterwards. An envelope check SHALL fail CI if artifact size crosses a
        configured multiple of the expected per-layer size.
    - id: BR-007
      theme: "Readout substrate"
      revised_in: "v0.2 — dtype corrected to fp16"
      text: >
        Every J-lens artifact SHALL record its full construction recipe as provenance, sufficient to
        rebuild it bit-for-bit: target layer (final vs penultimate residual stream), attention-pattern
        gradient treatment (full vs frozen Q/K), target-position scope (self-only / future-only / all
        subsequent), aggregation estimator (per-element mean vs median) and any outlier-exclusion
        thresholds, corpus identity and sampling, sequence count and length, convergence criterion, and
        library/commit versions. Defaults SHALL be the source paper's best-performing readout recipe
        (penultimate target, mean aggregation) with frozen-Q/K available as a selectable variant, per
        Appendix A.2. Emitted artifacts SHALL be fp16, matching the reference implementation's serialized
        form — NOT bf16 as stated in v0.1. Acquired artifacts SHALL carry their upstream recipe as
        provenance so a recipe/use mismatch is visible rather than latent.
    - id: BR-008
      theme: "Readout substrate"
      text: >
        The system SHALL expose THREE distinct readout modes and SHALL NOT substitute one for another:
        (a) FULL RANKED readout over the vocabulary, for the position × layer grid and top-token lists;
        (b) PROBE, scoring a residual-stream activation against one named token direction without ranking
        the vocabulary, for threshold detection; (c) SPARSE DECOMPOSITION, solving for a sparse
        non-negative combination of at most k lens directions by gradient pursuit, for a discrete
        inventory of active concepts. Occupancy, variance, and J-space/non-J-space split figures SHALL be
        computed from (c) only. Top-k by inner product SHALL NOT be used as a substitute for (c); the
        source paper is explicit that on an overcomplete non-orthogonal frame these give different and
        typically more redundant answers. Where (b) and (a) disagree because of the data-dependent
        normalization factor, the canonical mode SHALL be recorded per analysis.
    - id: BR-009
      theme: "Readout substrate"
      text: >
        Sparse decomposition SHALL record its sparsity level k, solver identity and parameters, iteration
        count, convergence criterion, and the seed and construction of any random-direction control set,
        as provenance on every derived figure. The decomposition is non-unique by construction, so
        reproducibility is a provenance property, not a mathematical one. Occupancy and excess-FVE figures
        without a recorded control seed SHALL be treated as invalid.

    # --- Readout surface ---
    - id: BR-010
      theme: "Readout surface"
      revised_in: "v0.2 — lens-mode control and reference implementation added"
      text: >
        The system SHALL provide a POSITION × LAYER readout view as a first-class panel, distinct from the
        existing feature-centric dashboards. It SHALL present: the prompt with per-token affordances; a
        grid of the top-ranked token at each (position, layer) cell with hover for the full top-k; a
        heatmap of user-pinned tokens' rank across all cells; the full ranked readout at a selected
        position across layers and at a selected layer across positions; rank-vs-layer trajectory charts
        for pinned tokens; and a lens-mode control offering Jacobian, logit, and diff views. The view SHALL
        visually mark the model's sensory / workspace / motor bands from that model's own band report
        (BR-002), and SHALL indicate that early-layer readouts are expected to be uninterpretable rather
        than presenting them as findings. A reference implementation of this panel exists as a design
        artifact and SHALL be treated as the interaction specification.
    - id: BR-011
      theme: "Readout surface"
      text: >
        The system SHALL surface, alongside any readout, an explicit interpretability caveat drawn from the
        source paper's own limitations: readouts are restricted to concepts with single-token names, and a
        non-trivial fraction of workspace-layer readouts resist interpretation for reasons that may be
        averaging noise, multi-token concepts, or unrecognized genuine content. A user SHALL NOT be led to
        treat an uninterpretable readout as a null result.

    # --- Dictionary annotation ---
    - id: BR-012
      theme: "Dictionary annotation"
      text: >
        The system SHALL annotate every feature of every managed SAE dictionary with a WORKSPACE
        CLASSIFICATION derived from projecting the feature's decoder direction through the lens. The
        classification SHALL use at least two independent fields: (a) a GEOMETRIC field — the excess
        kurtosis of the projected vocabulary distribution, with J-alignment thresholded against a
        covariance-matched random null; and (b) a BEHAVIORAL field distinguishing MOTOR features from
        WORKSPACE features, since high kurtosis alone does not separate them. Motor classification SHALL
        follow the source paper's operational definition: a feature is motor if, over its strongest
        activations, the model's actual next token appears among the feature's top lens tokens more often
        than a configured rate (paper default: 10%). A single scalar "workspace score" SHALL NOT be
        shipped as the whole annotation.
    - id: BR-013
      theme: "Dictionary annotation"
      text: >
        The system SHALL raise a LABEL DISAGREEMENT flag on any feature whose existing auto-generated
        label and whose lens readout are semantically divergent, and SHALL make disagreement a filterable,
        sortable, reviewable queue. This requirement exists because the failure mode is documented and
        consequential: example-driven labeling misidentified a fabricated-content detector as technical
        exposition, and features of exactly that kind are the ones safety work depends on. Divergence
        detection method is TBD (candidates: LLM judge over label vs top-k readout; embedding distance;
        human triage of a sampled tier).
    - id: BR-014
      theme: "Dictionary annotation"
      text: >
        The system SHALL validate its annotation implementation against the source paper's reported
        distributional findings on the reference dictionary — that only a modest fraction of features are
        J-aligned once motor features are excluded, that non-aligned features are dominated by low-level
        syntactic and bookkeeping content, and that J-aligned features fire more strongly but less often.
        Material divergence SHALL be reported as either a finding about the model or a defect in the
        implementation, and SHALL be resolved before the annotation is exported.
    - id: BR-015
      theme: "Dictionary annotation"
      text: >
        The system SHALL support projecting arbitrary WEIGHT-SPACE directions through the lens and
        presenting the resulting ranked tokens as a component interpretation, for at minimum: SAE decoder
        directions, transcoder encoder and decoder directions (as separate readouts, so the input-to-output
        transformation a feature implements is legible), and attention head query, key, value, and output
        matrices. This capability requires no activations and no corpus, and is therefore expected to be
        among the cheapest high-value additions in this increment.

    # --- Intervention engine ---
    - id: BR-016
      theme: "Intervention engine"
      text: >
        The system SHALL implement PAIRED-RUN EXECUTION WITH CLAMPING as a capability of the intervention
        engine: a clean reference forward pass whose per-position results parameterize a second intervened
        pass, and the ability to hold specified lens coordinates at their clean-pass values at every
        position and layer for the duration of that pass. This is the highest-priority new engine
        capability in this increment. Without it the product can execute interventions but cannot reproduce
        a single one of the source paper's mediation analyses — which is where that paper's evidential
        weight sits, and therefore where the product's credibility on causal claims will sit.
    - id: BR-017
      theme: "Intervention engine"
      revised_in: "v0.2 — scale-aware swap default added"
      text: >
        The system SHALL implement four intervention primitives over lens directions, each with recorded
        semantics: (a) ADDITIVE steering along a named token direction at specified layers and positions;
        (b) PROJECTIVE ABLATION, removing the activation's component along a direction entirely, recorded
        as distinct from negative-strength additive steering; (c) DYNAMIC TOP-K WORKSPACE ABLATION,
        identifying the k most strongly active lens directions per position across a layer band and zeroing
        their projections, while EXCLUDING any token present in the clean pass's top output candidates so
        the intervention targets internal reasoning rather than report; (d) LENS-COORDINATE SWAP,
        exchanging two named concepts' coordinates while leaving the component orthogonal to their span
        untouched, clamped across positions. Ablation strength SHALL be expressible as a layer band, and
        the product SHALL ship the paper's light / medium / heavy band presets as named starting points
        rather than raw layer numbers, rederived for the target model. Swap layer-count defaults SHALL be
        SCALE-AWARE: upstream reports that on smaller models swaps oversteer easily and require selecting
        FEWER layers to land the intended result, so band presets SHALL NOT be uniform across model scales.
    - id: BR-018
      theme: "Intervention engine"
      text: >
        Every intervention run SHALL execute against a size-matched random-direction control at the same
        layers and positions, and SHALL report control results alongside intervened results by default.
        A run reported without its control SHALL be treated as invalid. Rationale: the source paper's
        interpretations rest throughout on the gap between an intervention and its matched control, and
        several of its headline effects are only meaningful as that difference.

    # --- Claims discipline ---
    - id: BR-019
      theme: "Claims discipline"
      text: >
        J-space evidence SHALL be assigned rungs on the product's existing evidence ladder, and the ladder
        SHALL remain the single claims vocabulary. At minimum: a READOUT (a concept appearing in a lens
        readout at a position) is the lowest rung and is explicitly NOT a causal claim; a PROBE
        THRESHOLD CROSSING is a readout with a stated detection criterion; a DECOMPOSITION MEMBERSHIP is a
        structural claim; an INTERVENTION RESULT WITH MATCHED CONTROL is causal; and a MEDIATION RESULT
        (intervention plus complementary-component clamping) is the highest rung. No UI, report, export, or
        MCP surface may present a lower rung using a higher rung's language. The ladder remains
        badge-not-gate: low-rung evidence is surfaced with its rung, not suppressed.
    - id: BR-020
      theme: "Claims discipline"
      text: >
        The system SHALL NOT present workspace evidence as evidence of comprehensive coverage. The source
        paper is explicit that sufficiently automatic or well-practiced computation proceeds without
        engaging the workspace, and that a concept without a single-token name may not surface even when
        represented. Any monitoring, auditing, or screening surface built on this capability SHALL state
        that absence of a signal is not evidence of absence of the underlying cognition.

    # --- Contracts, export, runtime handoff ---
    - id: BR-021
      theme: "Contracts & export"
      text: >
        The system SHALL extend the interchange family ADDITIVELY with new kinds for: the J-lens artifact
        and its recipe provenance; the per-feature workspace annotation; the position × layer readout
        record; and the runtime watchlist. Existing kinds (`mistudio.cluster-definition/v1`,
        `mistudio.cluster-bundle/v1`, `mistudio.circuit-definition/v1`) SHALL NOT be mutated. Where a new
        kind carries content a shipped consumer could use, a PROJECTION to an existing kind SHALL be
        provided so miLLM-as-shipped works unchanged, each projection marked as a partial rendering — the
        same discipline as CIRCUITS-001 BR-014. All new kinds SHALL carry day-one nullable fields for
        multi-token concept references (BR-023) even if the template-lens implementation lands as a
        fast-follow, and SHALL be reviewed against the anticipated miLLM runtime BEFORE v1 freeze.
    - id: BR-022
      theme: "Contracts & export"
      revised_in: "v0.2 — restructured into two independent tracks"
      text: >
        Neuronpedia conformance SHALL be delivered as TWO INDEPENDENT TRACKS, which share nothing but a
        name and MAY ship in either order. TRACK A — J-LENS ARTIFACT SUPPLY: miStudio produces a conformant
        lens artifact (on-disk layout, checkpoint schema, and reproducibility config per Appendix A.10)
        which a local Neuronpedia instance mounts through its local-source override. There is NO upload,
        NO API call, and NO database write in this track; J-lens is compute-on-demand from a mounted
        artifact and Neuronpedia's entire J-lens database footprint is two tables persisting shared
        analysis sessions. TRACK B — SAE WORKSPACE ANNOTATION: the per-feature classification of BR-012 is
        feature-shaped data and SHALL be exported through the EXISTING Neuronpedia feature/explanation
        upload path that miStudio already implements, carried either as explanation-adjacent metadata or as
        a sibling to the existing logit-lens dashboard payload. Track B target placement SHALL be confirmed
        against the running local instance's actual schema before building. Where an upstream
        representation cannot carry something miStudio needs, the gap SHALL be recorded and proposed
        upstream rather than worked around locally.
    - id: BR-023
      theme: "Contracts & export"
      text: >
        The system SHALL implement a TEMPLATE LENS path for multi-token concepts: for each entry in a
        configurable vocabulary, generate contexts in which that word or phrase is the natural
        continuation, average the residual stream at the final position, mean-center against a baseline
        set, and whiten by the regularized inverse covariance to obtain a direction usable for readout,
        probing, steering, and coordinate swap identically to a lens direction. This path is FORWARD
        PASSES ONLY and therefore fits the existing activation-capture pipeline without new machinery. The
        implementation SHALL carry the source paper's three documented failure modes as first-class
        limitations: a tendency to skip ahead to the eventual answer in early layers; unreliable
        final-layer next-word agreement; and a small set of words appearing spuriously across many
        transcripts, for which a filter list is an accepted mitigation. Vocabulary size and composition are
        TBD.
    - id: BR-024
      theme: "Claims discipline"
      text: >
        Product copy, UI labels, documentation, and export metadata SHALL NOT assert, imply, or invite the
        inference that a served model has subjective experience or phenomenal consciousness. The source
        paper takes no position on this and describes the philosophical implications as unclear and likely
        controversial; the product takes the same position explicitly. Where the capability surfaces
        experiential-report phenomena, framing SHALL be functional and mechanistic. This is a shipping
        requirement subject to review, not a disclaimer: mis-framing here is the fastest available route to
        having serious researchers discount the tool.
    - id: BR-025
      theme: "Runtime handoff"
      text: >
        The system SHALL author, validate, version, and export WATCHLISTS: named sets of concept directions
        with per-concept detection thresholds and a stated scoring definition, intended for per-token
        evaluation at inference time by miLLM. A watchlist SHALL be an exportable artifact under the
        contract of BR-021, SHALL reference its J-lens artifact and layer band by identity, and SHALL
        refuse to validate against a model or artifact it was not authored for. This BRD owns authoring,
        validation, and export; runtime evaluation and streaming are BRD-MILLM-JSPACE-001.
    - id: BR-026
      theme: "Runtime handoff"
      text: >
        The system SHALL ship at least one VALIDATED reference watchlist as an existence proof and a
        template, implementing the source paper's evaluation-awareness score: the mean lens log-probability
        of a small set of evaluation-awareness tokens minus the mean over a control set of unrelated
        concrete nouns, averaged across workspace layers. Validation SHALL reproduce the paper's four
        supporting checks to the extent the served model permits: discrimination against an independent
        method's judgments; monotonic response to contrastive steering; reduction under realism edits to
        evaluation prompts; and separation between evaluation-suite and real-traffic distributions.
        Reference thresholds SHALL be rederived for the target model and SHALL NOT be inherited.
    - id: BR-027
      theme: "Runtime handoff"
      revised_in: "v0.3 — broadened to FULL parity on product-owner direction (2026-07-31)"
      text: >
        The MCP surface SHALL reach FULL PARITY with the J-space product surface, not a selected
        subset. Every capability a user can reach in the workbench SHALL be reachable by an agent:
        readout (all three modes), artifact listing and validation status, band-report retrieval,
        workspace-annotation queries, weight-space component readouts, intervention execution with
        its mandatory matched control, and watchlist management. Tools SHALL be delivered ALONGSIDE
        the feature that creates the capability rather than batched at the end of the increment —
        a readout tool is useful the moment a readout exists.
        EVERY tool SHALL be covered by the reachability harness at `tests/unit/test_reachability.py`,
        asserting presence in the LIVE registry and the built server, with payload and call count
        asserted — never that the tool module imports. This is not optional diligence: this repo
        shipped 16 fully-implemented, unit-tested, documented `millm_circuit_*` tools that were
        never registered with the server, so every test passed by importing the module directly
        while no agent could call the feature. J-space SHALL NOT repeat it.
        The surface SHALL expose readout queries, workspace-annotation queries, and watchlist
        management, and SHALL include a readout tool shaped for automated auditing agents: top-k lens
        tokens at a queried position and layer, accompanied by normative interpretation guidance
        instructing the consumer to treat the output as a bag of related ideas rather than prose, to look
        for token families rather than exact terms, to discount single-layer noise in favour of content
        recurring across layers or nearby positions, and to cite a specific layer line verbatim so claims
        can be verified. The guidance text is part of the deliverable, not documentation.
    - id: BR-028
      theme: "Envelope"
      text: >
        The system SHALL report a cost envelope for every J-space operation class — artifact acquisition or
        construction, readout, decomposition, annotation sweep, intervention run, template-lens vocabulary
        build — in wall-clock and peak memory against the deployment envelope, and SHALL surface estimates
        before a user or agent commits to an expensive run. Operations that cannot fit the single-GPU
        envelope at the chosen model scale SHALL fail with a stated reason rather than degrade silently.

    # --- NEW IN v0.2 ---
    - id: BR-029
      theme: "Contracts & export"
      added_in: "v0.2"
      text: >
        miStudio's readout transport SHALL mirror Neuronpedia's lens stream wire format rather than
        defining a parallel one: a meta message carrying model identity, lens types, per-type layer lists,
        top-N, and prompt length; per-position token messages carrying the token, its id, a generated flag,
        and one slice per lens type with `top_tokens[layer][k]` and `top_probs[layer][k]` as DECODED
        STRINGS; and terminal done / error messages. Conforming makes a miStudio readout stream and a
        Neuronpedia readout stream interchangeable at the client, so the viewer of BR-010 can be driven by
        either without a translation layer, and satisfies part of BR-021's projection obligation
        structurally rather than by adaptation code.
    - id: BR-030
      theme: "Contracts & export"
      added_in: "v0.2"
      text: >
        The system SHALL validate every J-lens artifact before it is handed to any consumer, against the
        acceptance suite defined in Appendix A.10.6: STRUCTURAL (deserializes with weights-only loading;
        required keys present; every Jacobian square of side d_model; layer keys coercible), NAMING
        (filename and slug convention; exactly one lens file per mounted directory), ENVELOPE (size within
        tolerance of the expected per-layer arithmetic — the BR-006 guard), SEMANTIC (a fixture prompt with
        a known unspoken intermediate recovers that intermediate in the top-k at a mid-band layer),
        CROSS-IMPLEMENTATION (same prompt, layer, and top-k agree between miStudio's readout and the local
        Neuronpedia instance, in both Jacobian and logit modes), and ROUND-TRIP (mount, serve, issue a
        Jacobian-lens request, confirm a non-empty readout). The round-trip check SHALL be explicit rather
        than inferred from a clean startup log, because upstream lens loading is best-effort and a
        malformed artifact fails silently at request time. This validator is the reusable asset of the
        track: it guards every artifact regardless of whether it was downloaded, fitted locally, or
        contributed.
    - id: BR-031
      theme: "Readout substrate"
      added_in: "v0.2"
      revised_in: "v0.3 — INVERTED. Construction is the primary path; acquisition is opportunistic."
      text: >
        The system SHALL treat lens CONSTRUCTION as the primary, first-class path and acquisition as an
        opportunistic optimization taken only when a conformant pre-fitted lens exists for the EXACT
        weights in use. v0.2 had this backwards. Pre-fitted lenses cover 36 models upstream; the reference
        model LFM2.5-1.2B-Instruct is not among them, and neither is most of what this workbench loads.
        A capability that works only for pre-fitted models is not the capability the product owner asked
        for. The system SHALL therefore be able to fit a lens for any model it can load, and SHALL NOT
        make acquisition a precondition of any downstream feature. Where a pre-fitted lens IS available
        for the exact weights, using it is preferred on cost grounds alone, and its upstream recipe SHALL
        be carried as provenance (BR-007) so a recipe/use mismatch is visible. Note the trap that motivated
        this: the workbench holds gemma-2-2b-IT, whose weights differ from the gemma-2-2b BASE model the
        upstream lens is fitted for, so acquisition would have silently supplied an invalid artifact.
        Fitting SHALL use convergence-based stopping from a floor of 100 prompts, SHALL exploit
        corpus-slice parallelism with merge rather than splitting the model, and SHALL emit the conformant
        directory layout natively so Track A layout conformance is free. Lenses fitted for models upstream
        lacks SHALL be offered for contribution.

    - id: BR-032
      theme: "Readout substrate"
      added_in: "v0.3"
      text: >
        Lens construction, readout, and analysis SHALL be MODEL-AGNOSTIC BY CONSTRUCTION, resolving model
        structure through miStudio's own architecture discovery
        (`ml/layer_discovery.discover_transformer_structure`) rather than through any architecture
        whitelist, name match, or an upstream fitter's layout auto-detection. This is not a new principle:
        miStudio already deleted its SUPPORTED_ARCHITECTURES whitelist and replaced every hardcoded
        architecture branch with dynamic discovery, and J-space SHALL NOT reintroduce one. The residual
        stream SHALL be captured at the DECODER LAYER OUTPUT (`structure.layers_module[L]`), never at a
        discovered normalization module: on LFM2 the module a naive search identifies as "residual" is a
        post-attention RMSNorm, and a vector applied there is renormalized away — the failure is silent
        and produces plausible-looking numbers with no signal.
        Because layer KINDS differ within a single model, every artifact and every band report SHALL
        declare, PER LAYER, which computations were applicable: frozen-Q/K variant, attention-broadcast
        metrics, and MLP gain. The reference model interleaves 10 convolutional layers with 6 attention
        layers over 16 total, so frozen-Q/K is undefined on 10 of them and the attention-broadcast metrics
        are computable on 6. Inapplicable metrics SHALL be recorded as ABSENT AND LABELLED, never as zero,
        and SHALL NOT be averaged over the subset that happened to qualify.

  success_metrics:
    quantitative_metrics:
      - "Phase 0 replication: J-lens normalized pass@k AUC exceeds the logit-lens baseline on the reference model across the six evaluation distributions, or the deviation is characterized. Reported per distribution, not pooled."
      - "Phase 0 causal replication: lens-coordinate swap success rate on the multihop set is MEASURED AND REPORTED as a first-class number. The source paper's 54–70% range across model scales is context, not a target."
      - "Artifact envelope: J-lens artifact for the reference model is within the order of magnitude predicted by Appendix A.1 (hundreds of MB), verified by CI check."
      - "Acquisition-first holds: the fraction of served models whose lens was acquired rather than fitted is reported; fitting effort is spent only on upstream-uncovered models."
      - "Construction cost, where fitting is required: completes within a recorded wall-clock and peak-memory bound on a single RTX 3090 using convergence-based stopping."
      - "Artifact validation: 100% of artifacts handed to a consumer pass all six acceptance classes of Appendix A.10.6, including the explicit round-trip check."
      - "Wire-format conformance: the BR-010 viewer renders identically when driven by a miStudio stream and by a Neuronpedia stream for the same model and prompt, with no adaptation layer."
      - "Annotation coverage: 100% of features in managed dictionaries carry both workspace annotation fields, with recorded null-control seed."
      - "Label-disagreement yield: the disagreement rate across at least one fully labeled dictionary is measured and reported — the first such number published at dictionary scale."
      - "Mediation capability: at least three mediation analyses (intervention plus complementary-component clamping) execute end to end and reproduce the qualitative direction of the source paper's corresponding result, each with matched control."
      - "Contract safety: existing cluster-definition/v1 and circuit-definition/v1 validation suites pass unchanged; projections validate against vendored schemas; miLLM-as-shipped consumes projections with zero runtime change."
      - "Track B round-trip: exported workspace annotations validate against the local instance's feature schema and re-import with semantic equality."
      - "Watchlist reference implementation: the evaluation-awareness score achieves reported discrimination against an independent judgment source on the target model, with the number published whatever it is."
      - "Reproducibility audit: a sampled set of shipped figures is rebuilt from recorded provenance alone and matches."
    qualitative_indicators:
      - "A researcher can answer 'what is this model poised to say here' in the workbench without writing code."
      - "A reviewer looking at any workspace claim can tell which rung it sits on, and cannot mistake a readout for an intervention result."
      - "The label-disagreement queue changes how the team labels features — i.e. it is used, not merely present."
      - "An auditor can take an evaluation transcript, find the decision positions, and see what was in the workspace there, without bespoke tooling."
      - "The J-space annotation makes the existing dictionary more trustworthy rather than casting doubt on it."
      - "Framing discipline holds under adversarial reading: no reviewer can quote product copy as an overclaim about model experience."
      - "The local Neuronpedia instance and miStudio agree, and disagreement between them is diagnostic rather than mysterious."
    measurement_methods:
      - "Phase 0 evaluation harness over the vendored upstream evaluation data, run as a versioned regression suite in CI, not a one-off."
      - "Artifact acceptance suite (Appendix A.10.6) executed as a gate on artifact publication."
      - "Cross-implementation comparison against the local Neuronpedia instance in both Jacobian and logit modes, using the diff view as the visual regression surface."
      - "Per-run report artifacts with full provenance, stored as first-class DB records consistent with existing miStudio practice."
      - "Envelope metrics captured per operation and regression-checked against recorded baselines."
      - "Contract validation suites executed against vendored schemas on every contract change."
      - "Round-trip export/import equality tests against a local Neuronpedia instance."
      - "Provenance replay audit: scheduled rebuild of sampled figures from recorded provenance."
      - "Claims-discipline review as an acceptance gate on UI and export copy, with an explicit reviewer sign-off."

  feature_themes:
    core_features:
      - "Phase 0 replication harness and gate: vendored upstream implementation and evaluation data, run-and-verify, workspace-band report, GO/NO-GO record."
      - "Logit-lens readout path and the position × layer readout viewer with Jacobian / logit / diff modes."
      - "J-lens artifact class: acquisition-first lifecycle, recipe provenance, storage discipline, envelope reporting, on-disk conformance."
      - "Artifact validation suite (six acceptance classes), used as a publication gate."
      - "Three readout modes including gradient-pursuit sparse decomposition."
      - "Readout stream conforming to the upstream lens wire format."
      - "Structural analysis suite: layer-band metrics, CKA, occupancy, excess FVE, MLP gain, attention broadcast metrics, broadcast-head identification."
      - "SAE dictionary workspace annotation with dual geometric/behavioral classification and the label-disagreement queue."
      - "Weight-space component readouts (SAE decoders, transcoder pairs, attention Q/K/V/O)."
      - "Intervention engine extension: paired-run with clamping, projective ablation, dynamic top-k workspace ablation, scale-aware lens-coordinate swap, mandatory matched controls."
      - "Evidence-ladder rung integration and claims-discipline enforcement."
      - "Additive interchange kinds plus projections."
      - "Track A: conformant artifact supply to a local Neuronpedia instance via mounted directory."
      - "Track B: SAE workspace annotation export through the existing feature upload path."
      - "Watchlist authoring, validation, export, and the reference evaluation-awareness watchlist."
    secondary_features:
      - "Template lens for multi-token concepts (contract fields day-one; implementation may be fast-follow)."
      - "MCP surface extension including the auditing-agent readout tool and its interpretation guidance."
      - "Cost-envelope estimation and pre-commit warnings for expensive operations."
      - "Distributional validation of the annotation against the source paper's reported findings."
      - "Local lens fitting for upstream-uncovered models, with corpus-slice parallelism and merge."
      - "Upstream contributions: fitted lenses for uncovered models; the SAE workspace annotation as a schema proposal."
    future_features:
      - "J-lens attribution graphs (training-free, self-labeling nodes) — feasibility spike only this increment."
      - "Workspace emergence across a public pretraining checkpoint series."
      - "Workspace-aligned dictionary training."
      - "Reflection-training evaluation harness."
      - "Dictionary-scale quantification of the abstract-vs-syntactic split."

  considerations:
    budget_constraints: >
      No incremental hardware budget assumed; the single-RTX-3090 (24 GB) envelope stands unless the
      Phase 0 gate returns GO-AT-LARGER-SCALE, in which case a hardware decision is taken explicitly
      rather than absorbed. v0.2 materially lowers the expected compute line: pre-fitted lenses exist for
      36 models including the reference model's family, so artifact construction may cost nothing. Where
      fitting is required, it uses convergence-based stopping from a floor of 100 prompts and parallelizes
      by corpus slice. The dominant remaining cost risk is the template-lens vocabulary build, which needs
      a few hundred forward passes per entry and scales linearly with vocabulary size. Precise budget TBD.
    timeline_expectations: >
      TBD. Phasing is fixed even where dates are not: Phase 0 (acquire, validate, run-and-verify, gate)
      precedes everything; the logit-lens readout surface and weight-space component readouts follow as the
      cheapest high-value work and can proceed against a local Neuronpedia instance before any artifact
      exists; annotation and the intervention-engine extension follow the gate; contracts and export freeze
      only after the miLLM-runtime contract review; the template lens and the runtime handoff artifacts land
      last on the miStudio side, immediately before BRD-MILLM-JSPACE-001 opens.
    regulatory_or_policy_drivers:
      - "No external regulatory driver identified."
      - "Internal claims-integrity policy (BR-019, BR-020, BR-024): a tool that surfaces what a model is 'thinking' is unusually easy to overclaim with, and the reputational exposure is asymmetric."
      - "Research-integrity expectation that negative and unfavourable results from Phase 0 are published, not buried."
      - "Upstream licence and attribution obligations on vendored reference code and acquired artifacts."
    technical_constraints:
      - "Single RTX 3090, 24 GB, unless explicitly revised."
      - "Readouts are restricted to concepts with single-token names until the template lens lands; this is the source paper's own central stated limitation and it bounds what the product can claim to detect."
      - "Sparse decomposition is non-unique on an overcomplete correlated frame; solver parameters and control seeds are load-bearing provenance."
      - "The workspace is treated as a flat collection of independently active concepts. Binding, role assignment, and relational structure are not represented, by the source method's own admission."
      - "No mechanistic account exists of what causes a representation to ENTER the workspace; the product reads contents, it does not explain arrival."
      - "Artifact storage discipline (BR-006) is a hard constraint, not an optimization."
      - "Serialized artifacts are fp16; consumers cast on load. Byte-level conformance matters for Track A."
      - "Upstream lens loading is best-effort and fails silently at request time, so validation is miStudio's responsibility, not the consumer's."
      - "The reference implementation is unmaintained and not accepting contributions; vendored code will not receive upstream fixes."
    integration_requirements:
      - "Local Neuronpedia instance, Track A: a mounted directory referenced by the instance's local lens-source override, wired as a volume in the existing deployment manifest rather than a push job."
      - "Local Neuronpedia instance, Track B: the existing feature/explanation upload path, with target placement confirmed against the running instance's actual schema."
      - "Upstream model map: miStudio must resolve Neuronpedia model ids and the corresponding HuggingFace model ids to construct conformant paths and filenames."
      - "miLLM: residual-stream hook points and the Socket.IO activation-monitoring channel are the consumer-side integration surface. This increment is read-only against miLLM and must not require a miLLM change to ship. Contract review against the anticipated runtime is required before v1 freeze."
      - "Existing miStudio subsystems: activation capture, background job runner, provenance/DB, evidence ladder, checkpoint lifecycle, MCP server, export service."
    scalability_expectations: >
      Readout and probe operations must be cheap enough to be routine: a probe is an inner product per
      (layer, position, watched concept), which is what makes the runtime monitoring case viable at all.
      Full ranked readouts add one d_model-square matrix-vector product over an unembedding call.
      Annotation sweeps and structural analyses are batch operations sized to the dictionary and reported
      per run. v0.2 removes model scale as a primary scaling risk for the readout itself — upstream serves
      lenses from 70M to 70B — leaving the open question as whether the full workspace claim set holds at
      the served scale, which is a research finding rather than a scaling constraint.

  risks:
    - id: RSK-001
      revised_in: "v0.2 — downgraded and narrowed"
      description: "The full workspace claim set (band structure, selectivity, flexible generalization, capacity limits) may not replicate at the model scale miStudio serves, even though the lens itself produces usable readouts there. Narrowed from v0.1, which framed this as whether the phenomenon exists at all: upstream ships lens support for models as small as 70M and hosts working J-lens interfaces across 12 models and four families, so readout viability at 2B is settled. What remains genuinely open is the paper's own §9.1 question about whether smaller models have an equally rich workspace, a proportionally smaller one, or a less reliable one."
      impact: "medium"
      likelihood: "low"
      mitigation: "Phase 0 remains a gate (BR-003) but its question is narrowed to claim-set replication, and a partial result is a characterization rather than a blocker. Fallback to a larger open model is pre-identified with the hardware consequence taken explicitly. The logit-lens viewer and weight-space readouts deliver value independent of the outcome."
    - id: RSK-002
      revised_in: "v0.2 — availability confirmed, maintenance risk added"
      description: "The vendored reference implementation is explicitly unmaintained and not accepting contributions, so defects found during Phase 0 will not be fixed upstream and miStudio inherits maintenance of anything it depends on."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Availability is confirmed and the v0.1 risk of the assets not existing is retired: the fitter, applier, six evaluation sets, and eleven experiment fixtures are all present. Vendor at a recorded commit, treat the code as forked-on-arrival, and keep the dependency surface minimal — Neuronpedia's own inference server deliberately does not import the reference package and loads the tensors directly, which is the pattern to follow."
    - id: RSK-003
      revised_in: "v0.2 — likelihood lowered"
      description: "Storage blowout: a naive implementation materializes the token-indexed dictionary, turning a hundreds-of-MB artifact into tens of GB and breaking the deployment envelope."
      impact: "high"
      likelihood: "low"
      mitigation: "BR-006 makes non-materialization a requirement with normative arithmetic in Appendix A.1 and a CI envelope check, now reinforced by BR-030's envelope acceptance class. Likelihood lowered from medium because both upstream implementations demonstrate the correct pattern and are available to copy."
    - id: RSK-004
      description: "Motor/workspace conflation: shipping a single kurtosis-derived 'workspace score' presents output-driving features as workspace features, producing a misleading annotation on precisely the dimension users would trust it for."
      impact: "medium"
      likelihood: "high"
      mitigation: "BR-012 requires two independent fields with the behavioral motor classifier specified; BR-014 requires distributional validation against the source paper's reported findings before the annotation is exported."
    - id: RSK-005
      description: "Rung creep: readout evidence gets described in causal language somewhere in UI, export, or MCP output, and the product's claims discipline erodes exactly where it is most load-bearing."
      impact: "high"
      likelihood: "medium"
      mitigation: "BR-019 binds J-space evidence to the existing ladder with an explicit prohibition; claims-discipline review is an acceptance gate with named sign-off; the ladder is already implemented and enforced for circuits, so this extends a working control rather than creating one."
    - id: RSK-006
      description: "Mediation capability slips: paired-run-with-clamping is the hardest engine change and the easiest to defer, and deferring it yields a product that can perform interventions but cannot substantiate causal claims about them."
      impact: "high"
      likelihood: "medium"
      mitigation: "BR-016 names it the highest-priority engine capability and states the consequence; at least three end-to-end mediation analyses are a quantitative success metric, so the capability cannot be quietly dropped without failing acceptance."
    - id: RSK-007
      revised_in: "v0.2 — blast radius reduced by BR-029"
      description: "Contract freeze regret: new kinds freeze before the miLLM runtime requirements are understood, forcing either a breaking change or an awkward projection layer."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Contract review against the anticipated runtime before v1 freeze is a requirement (BR-021) and a named clarification priority. Additive-only rule plus day-one nullable multi-token fields limit the blast radius. BR-029's adoption of the upstream wire format removes one whole class of contract invention."
    - id: RSK-008
      revised_in: "v0.2 — reframed to Track B only"
      description: "Track B divergence: the local instance's feature schema cannot carry the workspace annotation as miStudio models it, and the export path forks into a parallel local representation. Track A is no longer exposed to this risk, since its conformance surface is a file layout verified against source."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Target placement is confirmed against the running instance before building (BR-022); two placement options are pre-identified, one of which extends an existing payload field and needs no schema negotiation; gaps are proposed upstream rather than worked around."
    - id: RSK-009
      description: "Overclaiming on experiential findings: the capability's most attention-getting results concern the model's reports of its own experience, and product framing drifts into implying consciousness, costing credibility with the research audience the tool is built for."
      impact: "high"
      likelihood: "medium"
      mitigation: "BR-024 makes framing a shipping requirement with review sign-off; the source paper's own explicit neutrality is the standard; qualitative success indicator is adversarial-reading resistance."
    - id: RSK-010
      description: "Reproducibility gap: non-unique decompositions and unrecorded control seeds produce figures that cannot be rebuilt, undermining the provenance guarantee that differentiates miStudio."
      impact: "medium"
      likelihood: "medium"
      mitigation: "BR-009 makes solver parameters and control seeds mandatory provenance and invalidates figures lacking them; scheduled provenance-replay audit is a measurement method."
    - id: RSK-011
      description: "Absence-as-evidence misuse: a monitoring or auditing surface is read as certifying that no concerning cognition occurred, when the method is known to miss automatic computation and multi-token concepts."
      impact: "high"
      likelihood: "medium"
      mitigation: "BR-020 requires the coverage limitation to be stated on every monitoring, auditing, and screening surface; the limitation is carried into the runtime handoff contract so BRD-MILLM-JSPACE-001 inherits it rather than restating it optionally."
    - id: RSK-012
      description: "Template-lens quality shortfall: the multi-token path inherits documented pathologies and lands weak enough to be untrustworthy while still appearing in the product."
      impact: "medium"
      likelihood: "medium"
      mitigation: "BR-023 carries the three failure modes as first-class limitations with the accepted filter-list mitigation; the path is scoped as secondary with day-one contract fields so it can be delayed without stranding schema; template-derived readouts carry a distinct provenance marker."
    - id: RSK-013
      description: "Dictionary-annotation churn: the annotation depends on the J-lens artifact recipe, so a recipe change silently invalidates every annotation and every downstream disagreement flag."
      impact: "medium"
      likelihood: "medium"
      mitigation: "Annotations reference their artifact by identity and refuse to validate against a different one; recipe change produces a new artifact version rather than mutating one; re-annotation is a batch operation on the existing job runner. To be confirmed against the checkpoint-lifecycle machinery during TDD."
    - id: RSK-014
      added_in: "v0.2"
      description: "Silent artifact failure: upstream lens loading is best-effort — a malformed or mismatched artifact does not fail at deploy time, it fails at request time inside the webapp, potentially long after handover and with a misleading symptom."
      impact: "medium"
      likelihood: "high"
      mitigation: "BR-030 makes validation miStudio's responsibility with six acceptance classes, and requires the round-trip check to be explicit rather than inferred from a clean startup log. The cross-implementation class additionally catches structurally valid but semantically wrong artifacts (wrong model, wrong layer indexing) that structural checks alone would pass."
    - id: RSK-015
      added_in: "v0.2"
      description: "Acquisition assumption failure: the reference model turns out not to be covered by the pre-fitted upstream set, or the available lens was fitted with a recipe incompatible with miStudio's intended use (for example readout-tuned where intervention work wants frozen-Q/K), silently reintroducing the fitting cost the v0.2 budget removed."
      impact: "low"
      likelihood: "medium"
      mitigation: "Coverage check is the first task of Phase 0, before dependent planning. Acquired artifacts carry their upstream recipe as provenance (BR-007), so a recipe mismatch is visible rather than latent, and BR-031 retains the local fitting path with corpus-slice parallelism for exactly this case."

  next_steps:
    open_questions:
      - "Layer subsampling: the source paper reports on 25 evenly spaced layers reindexed to 0–100, while the upstream stream format carries an explicit per-type layer list. Does miStudio analyze all layers of a 26-layer model, or adopt a subsampling convention for comparability with published results? Affects every band figure. BLOCKS BR-002 TDD."
      - "Label-disagreement detection method (BR-013): LLM judge over label vs top-k readout, embedding distance, or sampled human triage — and what is the acceptable false-positive rate for a review queue to stay usable? BLOCKS BR-013 TDD."
      - "Sparsity level k policy: the source paper varies k by analysis (16 for concept vectors, 25 for probes and occupancy). Does the product fix k per analysis type, expose it, or infer it from occupancy? BLOCKS BR-008/BR-009 TDD."
      - "Watchlist scoring definition (BR-025/BR-026): is the canonical score the paper's mean-log-probability-minus-control-mean, cosine similarity, or a decomposition coefficient — and is the threshold per-concept, per-watchlist, or per-model? Determines the runtime contract. BLOCKS BR-025 TDD and the miLLM handoff."
      - "Track B placement (BR-022): explanation-adjacent metadata or a sibling to the existing logit-lens dashboard payload — resolved by inspecting the running local instance, not by design debate."
      - "Interchange kind naming and versioning for the new kinds; whether the readout record is a kind at all or a report artifact only, given BR-029 adopts an upstream wire format for transport."
      - "Template-lens vocabulary: size, source, and whether phrase entries are in v1 or words only. Drives the dominant remaining compute cost in the increment."
      - "Whether the annotation sweep runs eagerly on every managed dictionary or lazily on request, given the envelope."
      - "Ablation and swap band presets: how are the paper's light/medium/heavy bands and the scale-aware swap layer counts (BR-017) rederived for a target model — proportionally from the band report, or re-tuned against a local multihop set?"
      - "Whether J-lens attribution-graph feasibility is a spike in this increment or deferred wholesale."
      - "RESOLVED IN v0.2 — Phase 0 reference model: gemma-2-2b is present in the upstream model map and lens support extends well below its scale, so it is viable as the Phase 0 target and the hardware envelope does not reopen on scale grounds. Confirm pre-fitted coverage as the first Phase 0 task."
      - "RESOLVED IN v0.2 — Neuronpedia conformance surface: two independent tracks, artifact-mount and feature-upload, specified in BR-022 and Appendix A.10. No ingestion API is required or permitted."
    recommended_actions:
      - "Confirm pre-fitted lens coverage for the reference model as the FIRST Phase 0 task, before any dependent planning (BR-031, RSK-015). If covered, Track A collapses to acquire → validate → mount."
      - "Stand up the local Neuronpedia instance with logit-lens mode working first: zero miStudio dependency, validates the whole serving path before any artifact exists, and gives BR-005's logit-first sequencing a live target."
      - "Implement the BR-030 acceptance suite early — it is the reusable asset of Track A and guards every artifact thereafter, however acquired."
      - "Run 0xcc/instruct/002_create-project-prd.md against this BRD to absorb J-space into the PPRD as a new capability row adjacent to (not inside) the circuits row."
      - "ADR updates via 0xcc/instruct/003_create-adr.md: (a) J-lens as a second, training-free dictionary substrate and its relationship to the SAE substrate; (b) artifact storage discipline — synthesize-on-demand, never materialize — as an architectural rule; (c) paired-run-with-clamping execution model in the intervention engine; (d) J-space rungs as an extension of the evidence ladder; (e) interchange additions, the projection obligation, and the decision to adopt the upstream wire format for readout transport; (f) recipe-provenance schema including solver parameters and control seeds; (g) NEW — artifact-mount rather than upload as the Neuronpedia integration architecture for Track A."
      - "Feature PRD wave via 0xcc/instruct/004_create-feature-prd.md, proposed split: (a) Phase 0 acquisition + validation suite + replication run + band report + gate [BR-001..004, BR-030, BR-031]; (b) readout substrate + artifact class + three modes + wire format [BR-005..009, BR-029]; (c) readout viewer + interpretability framing [BR-010, BR-011]; (d) dictionary annotation + disagreement queue + weight-space component readouts [BR-012..015]; (e) intervention engine extension [BR-016..018]; (f) claims discipline + ladder integration + framing review [BR-019, BR-020, BR-024]; (g) contracts + two-track Neuronpedia conformance + template lens [BR-021..023]; (h) runtime handoff: watchlists + reference evaluation-awareness watchlist + MCP + envelope [BR-025..028]. Then TDDs (005), TIDs (006), tasks (007), execution (008)."
      - "Treat the J-Space readout panel reference implementation as the interaction specification for feature PRD (c), and drive it from a live logit-lens stream as the first integration milestone."
      - "Open BRD-MILLM-JSPACE-001 as a seed document once the watchlist and readout contracts are drafted but before they freeze, so the runtime review required by BR-021 has a document to review against — mirroring the SUBSTRATE seed pattern."
    priority_for_clarification:
      - "Watchlist scoring definition — blocks BR-025 TDD and is the single field the miLLM runtime contract is built around. Now the highest-priority open item, v0.1's top two having been resolved."
      - "Layer subsampling convention — blocks BR-002 and silently affects every band, occupancy, and FVE figure in the increment."
      - "Sparsity-level policy — blocks BR-008/BR-009 and determines comparability of occupancy figures with published results."
      - "Label-disagreement detection method — blocks BR-013, the increment's clearest near-term value to existing users."
      - "Contract review against the anticipated miLLM runtime BEFORE v1 freeze — same discipline as CLUSTERS-001 and CIRCUITS-002."
```

---

# Appendix A — Implementation Primer (normative for the implementing agent)

This appendix defines the computations and decisions the business requirements reference. It exists so an
agentic implementer building from the XCC chain implements the intended mathematics rather than a
plausible guess. Where a TDD later refines a formula, the TDD wins; until then this appendix is the
reference. Notation follows the source paper: `h_ℓ` is the residual stream at layer `ℓ`, `W_U` the
unembedding matrix, `d_model` the residual width, `n_vocab` the vocabulary size.

**v0.3 changes:** A.1 envelope arithmetic made model-derived (the constant is not portable); A.2
attention-gradient variant marked undefined on non-attention layers; A.5 broadcast metrics scoped to
attention layers with mandatory disclosure. All three follow from BR-032.

**v0.2 changes:** A.1 dtype corrected to fp16 and the storage claim upgraded to verified; A.2 corpus-size
guidance corrected and parallelism added; A.7 swap default made scale-aware; A.10 added (Neuronpedia
conformance — normative for BR-022, BR-029, BR-030, BR-031).

## A.1 The storage decision (normative — BR-006, RSK-003)

The J-lens artifact at layer `ℓ` is a single `d_model × d_model` matrix `J_ℓ`, defined as the expectation
over source position, subsequent positions, and a prompt corpus of the Jacobian of the late-layer residual
stream with respect to `h_ℓ`.

The *token dictionary* is the rows of `W_U J_ℓ` — one `d_model` direction per vocabulary entry. It is
tempting to precompute and store this, and it is the wrong decision by one to two orders of magnitude:

| | per layer (fp16) | reference model, all layers |
|---|---|---|
| `J_ℓ` (required) | `d_model²·2` ≈ **10.6 MB** | ≈ **276 MB** |
| `W_U J_ℓ` materialized (prohibited) | `n_vocab·d_model·2` ≈ **1.18 GB** | ≈ **30.7 GB** |

Figures for `d_model = 2304`, `n_vocab ≈ 256k`, 26 layers. The ratio worsens with width: at
`d_model = 3584` and 42 layers the comparison is roughly 1.1 GB against 77 GB.

**v0.3 — the constant is model-derived, not fixed.** For the reference model LFM2.5-1.2B-Instruct
(`d_model = 2048`, `n_vocab = 65536`, 16 layers) the comparison is **134 MB required against 4.3 GB
materialized** — a ratio of ~32×, not the ~111× above, because that vocabulary is four times smaller.
The RULE is unchanged and still decisive; the ARITHMETIC is not portable. The CI envelope check
required by BR-006 SHALL derive its bound from the model's own `d_model`, `n_vocab` and `n_layers`,
and SHALL NOT compare against a hardcoded size.

Materialization buys nothing:

- **Full ranked readout** is `softmax(W_U · norm(J_ℓ · h_ℓ))` — one `d_model²` matrix-vector product, then
  the model's own normalization and unembedding call. Cost over a logit-lens readout is a single extra
  matvec.
- **A single token direction** `v_t` is row `t` of `W_U J_ℓ`, i.e. `W_U[t,:] · J_ℓ` — one vector-matrix
  product. Synthesize on demand; cache the small working set actually being steered with or probed against.
- **Probes, swaps, and ablations** touch a handful of directions each, never the dictionary.

**v0.2 — externally validated.** Neither the reference implementation nor Neuronpedia's inference server
materializes the dictionary. The server keeps per-layer Jacobians on CPU, moves them to the compute device
lazily per layer, and transports a residual with a single matmul, applying the model's own unembedding
afterwards. This is now a verified pattern to copy, not an inference to defend.

**Serialization dtype is fp16**, correcting v0.1's bf16. The reference implementation defaults to fp16 on
save with the stated reasoning that entries are O(1) so range is not a constraint and fp16's extra mantissa
bits are the better trade here; consumers cast to fp32 on load. Size arithmetic above is unchanged.

Implementation rule: no code path may allocate an `n_vocab × d_model` array derived from `J_ℓ`. The CI
envelope check required by BR-006 asserts artifact size against
`k · d_model² · sizeof(dtype) · n_layers` for a configured tolerance `k`.

## A.2 Construction recipe and its degrees of freedom (normative — BR-007, BR-031)

`J_ℓ` is estimated by backpropagating from a late-layer residual stream to `h_ℓ` and averaging the
resulting Jacobians over token positions and over a prompt corpus. Cotangents are summed over target
positions and then averaged over source positions. Reverse mode with batched cotangents fills the whole
(layer, source-position) grid from one backward sweep per output dimension; the practical cost driver is
the model's own backward pass.

Four independent choices must be recorded because they change the artifact:

1. **Target layer.** Final vs penultimate residual stream. **Default: penultimate.** Including the last
   block increases noisy artifacts in readouts, plausibly because that block is specialized for calibrating
   next-token probabilities and carries less semantic content.
2. **Attention-pattern gradients.** Full backward vs frozen Q/K. Frozen Q/K slightly reduces readout
   quality but can *increase* the causal effect of the resulting directions. Expose as a selectable
   variant; record per artifact.
   **v0.3 — UNDEFINED on layers without attention.** The reference model interleaves 10 convolutional
   layers with 6 attention layers; "freeze Q/K" has no meaning on a conv block. The variant SHALL be
   recorded PER LAYER as applied or inapplicable (BR-032), and an artifact SHALL NOT be described as
   "frozen-Q/K" wholesale when the treatment reached only a subset of its layers. For artifacts built for intervention work rather than readout work,
   frozen Q/K is the recommended variant — and an acquired artifact's recipe must be checked against
   intended use (RSK-015).
3. **Target-position scope.** Self-only, future-only, or all subsequent positions. **Default: all
   subsequent.**
4. **Aggregation.** Per-element mean or median across prompts, with optional exclusion of outlier-norm
   positions and of the first few positions of each sequence. **Default: mean.**

**Corpus size — corrected in v0.2.** v0.1 advised planning at the paper's ~10-sequence floor. Both
reference implementations disagree, and they are the operative authority:

- The reference implementation states the paper's lenses use 1000 sequences of 128 tokens, that quality
  saturates quickly, and that **~100 prompts is usable**.
- Neuronpedia's production fitter defaults to 1000 prompts over wikitext-103-raw-v1 (train split, max 2000
  chars, max sequence length 128, bfloat16 compute), with a **hard minimum of 100 prompts**,
  convergence-based early stopping at a delta of 2e-3 over a 10-prompt window, and reporting levels at
  1e-2 / 5e-3 / 1e-3.

**Operating guidance: convergence-based stopping with a floor of 100 prompts.** The ten-prompt figure is
the paper's demonstration that quality saturates early, not an operating recommendation.

**Parallelism.** Fitting parallelizes by running over disjoint corpus slices and merging the resulting
lenses, not by splitting the model. This is the mechanism that keeps fitting inside a single-GPU envelope
(BR-031).

## A.3 The three readout modes (normative — BR-008)

**(a) Full ranked.** `softmax(W_U · norm(J_ℓ · h_ℓ))`, sorted. Populates the position × layer grid and all
top-token lists.

**(b) Probe.** The pre-softmax logits are determined by the inner products `⟨v_t, h_ℓ⟩`, but only
*approximately* — up to a data-dependent normalization factor. So probe scores and full-ranking positions
can disagree. Pick one as canonical per analysis and record which (BR-008). The probe form is what makes
runtime monitoring cheap: one inner product per (layer, position, watched concept), no vocabulary ranking.

**(c) Sparse decomposition.** Solve for a sparse non-negative combination of at most `k` lens directions
that best reconstructs `h_ℓ`, by gradient pursuit. This is **not** the same as taking the top `k` by inner
product: because the directions are overcomplete and non-orthogonal, pursuit returns a different and
typically less redundant active set. Occupancy, J-space/non-J-space splits, and excess-FVE figures come
from (c) exclusively.

**J-space component and remainder.** For a given activation (or steering vector, or SAE decoder direction),
the pursuit solution is its J-space component; the difference is its non-J-space component. Two calibration
figures from upstream, useful as sanity checks and *not* as targets on a different model: a concept vector's
J-space component carried a median 6–7% of its variance while accounting for most of its effect on verbal
report; an inferred-intermediate probe's J-space component carried roughly 10–15% of variance while
carrying most of its causal effect.

## A.4 Structural metrics (normative — BR-002)

Compute per layer, on the model under analysis, never inherited:

- **Top-k next-token agreement** — fraction of positions where any top-k readout token matches the model's
  top-1 prediction. Near zero early, rising through the workspace band, jumping steeply in the final
  layers. The steep late rise marks the **motor** boundary. Diagnostic only — see BR-004.
- **Excess kurtosis** of the readout logit distribution. Near zero early; rising marks the **workspace
  onset**; falls in the last few layers.
- **Top-1 autocorrelation across positions** against a position-shuffled null. High values indicate
  abstract content persisting across the token stream; low values indicate token-local content.
- **Effective linear dimensionality** — fraction of residual dimensions needed to capture a given share of
  variance across the lens dictionary. Small early, rising sharply at onset, rising again at the motor
  transition as `J_ℓ` approaches identity.
- **Cross-layer CKA** on the matrices of pairwise similarities among lens directions. Expect block
  structure: an early block, a long middle block, a small late block. Blocks may be less sharp or contain
  sub-blocks on other models, and apparent sharpness is exaggerated by layer subsampling — which is why
  the subsampling convention is a blocking open question.
- **Occupancy** — the `k` at which marginal reconstruction improvement from a `k`-direction pursuit falls
  below a size-matched random-direction control. Upstream plateau was around 25 in the median case.
- **Excess FVE** — fraction of variance explained by the top-`k` directions at `k` = median occupancy, in
  excess of the random control. Upstream never exceeded 10%; a much larger figure is a red flag for a
  control-construction bug.

The random control is load-bearing in the last two: its construction and seed are mandatory provenance
(BR-009).

## A.5 Broadcast metrics (normative — BR-002, weight-space)

Corpus-free, computed from weights alone:

- **MLP gain** — output norm of the next MLP block applied to a unit direction, normalized by the median
  over isotropic random directions. Upstream, lens directions sat near 1 before workspace onset, rose to
  roughly 10× through the band, and fell in final layers, while individual neuron output directions stayed
  near 1.
- **Attention broadcast** — computable ONLY on layers that have attention. On a hybrid model this is a
  strict subset (6 of 16 on the reference model), and the report SHALL say so rather than presenting a
  figure averaged over the qualifying layers as if it covered the network (BR-032). Per head, (i) *gain*, mean `‖W_OV v‖` over a direction population normalized by
  the head's gain on random directions; (ii) *label preservation*, the mean reciprocal rank of
  `cos(W_OV v_i, v_i)` among `{cos(W_OV v_i, v_j)}_j`, contrasted against the same statistic on random
  directions so heads that copy indiscriminately score zero. **Broadcast heads** are the top 1% of
  workspace-layer heads by worse-rank across both metrics.
- Broadcast-head ablation is a validation target, not a product feature: upstream, ablating them dropped
  readout recall@25 at mid-workspace layers to 0.67 against 0.86 for a layer-matched random control, while
  perturbing top-1 next-token prediction at only 5% of positions against 2%.

## A.6 Dictionary annotation (normative — BR-012, BR-014)

For each SAE feature, project its decoder direction through the lens and measure how sharply peaked the
resulting vocabulary distribution is (excess kurtosis, `κ`). High `κ` means the direction loads
disproportionately onto few lens directions, which is the relevant criterion for presence in a *sparse
frame* — the J-space is not a subspace, so subspace-style alignment measures are the wrong tool.

**High `κ` is necessary but not sufficient.** Motor features — those that fire immediately before emitting
a specific token — also have high `κ`. The separating test is behavioral, not geometric: classify a feature
as MOTOR if, over its strongest activations, the model's actual next token appears among the feature's top
lens tokens more than a configured fraction of the time (upstream default 10%). Hence three classes:
WORKSPACE (high `κ`, not next-token-predictive), MOTOR (high `κ`, next-token-predictive), OUTSIDE (low `κ`).

Thresholding: J-alignment against `1.5 ×` the maximum `κ` of a covariance-matched random null. Upstream,
roughly 15% of a dictionary passed once motor features were excluded. Distributional checks for BR-014:
the `κ` distribution should be heavy-tailed with most mass low; OUTSIDE features should be dominated by
syntactic and bookkeeping content whose lens readouts are unrelated to their apparent meaning; J-aligned
features should fire more strongly but less often than average.

## A.7 Intervention primitives (normative — BR-016, BR-017)

- **Additive steering:** `h ← h + α·v_t` at specified layers and positions.
- **Projective ablation:** remove the component of `h` along `v_t` entirely. Record as distinct from
  negative `α`; they are different operations with different failure modes.
- **Dynamic top-k workspace ablation:** at each position, across a layer band, identify the `k` most
  strongly activated lens directions and zero the residual stream's projection onto each — **excluding any
  token appearing among the clean pass's top output candidates**. Requires the paired run. Upstream band
  presets, to be *rederived* not ported: light, medium, and heavy differed only in layer-band width, with
  multihop accuracy degrading 68% → 26% → 6% while ordinary next-token agreement held at 87% → 75% → 65%,
  against a random-direction control at the medium band that left both essentially untouched (98% / 96%).
  That control gap is the result; the absolute numbers are not portable.
- **Lens-coordinate swap:** form `V = [v_s  v_t]`, read coordinates `c = V⁺h`, set
  `h ← h + V(σ(c) − c)` where `σ` exchanges the two entries, optionally scaled. The component of `h`
  orthogonal to `span{v_s, v_t}` is unchanged. Clamp across positions. **Scale-aware default (v0.2):**
  on smaller models swaps oversteer readily and land the intended result only when applied across FEWER
  layers. Swap layer-count defaults are therefore a function of model scale, not a constant (BR-017).
- **Paired run with clamping:** a clean pass whose per-position results parameterize the intervened pass,
  plus the ability to hold named coordinates at clean-pass values at every position and layer. This is what
  makes mediation testable.

Every run pairs with a size-matched random-direction control at the same layers and positions (BR-018).

## A.8 Watchlist and evaluation-awareness score (normative — BR-025, BR-026)

A watchlist is a named set of concept directions with a scoring definition and per-concept thresholds,
bound by identity to a specific J-lens artifact, layer band, and model.

The reference implementation is the upstream evaluation-awareness score: the mean lens log-probability of a
small set of evaluation-awareness tokens, minus the mean over roughly a hundred unrelated concrete nouns,
averaged across the workspace layers. Upstream validation, to be reproduced as far as the served model
permits (BR-026):

1. Discrimination against an independent method's judgments on the same positions (upstream AUC 0.853).
2. Monotonic response to contrastive steering toward "test" versus "ordinary conversation".
3. Reduction when evaluation prompts are rewritten to read like real usage, on essentially every suite.
4. Separation between evaluation-suite transcripts and real deployment traffic.

The upstream authors describe this two-token classifier as a proof of concept and note that a larger
readout-token set with post-hoc LLM filtering would do better. Treat the two-token form as the validated
baseline and the filtered form as the improvement path.

Thresholds are model-specific and MUST be rederived. A watchlist that validates against a different
artifact or model than it was authored for is invalid, not merely stale (BR-025).

## A.9 Template lens for multi-token concepts (normative — BR-023)

For a word or phrase `w`: generate short passages in which `w` is the natural continuation, ending just
before `w` appears and never using `w` in the passage; run the model over them; average the residual stream
at the final position to get `μ_w(ℓ)`. Then centre against the mean over all other entries and whiten:

`t_w(ℓ) = (Σ_ℓ + λI)⁻¹ (μ_w(ℓ) − μ(ℓ))`

where `μ(ℓ)` and `Σ_ℓ` are the residual stream's mean and covariance over the same passages and `λ` is a
small ridge term. This is the linear discriminant direction separating contexts where the model is about to
say `w` from contexts where it is not, and — via Stein's lemma, under Gaussian assumptions that do not hold
exactly — an approximation to the corresponding lens direction.

Use identically to a lens direction for readout, steering, and coordinate swap, **omitting the unembedding
step**. Forward passes only. Cost is a few hundred forward passes per entry, which is the increment's
dominant remaining compute line and the reason vocabulary size is a blocking open question.

Carry the three documented pathologies as first-class limitations (BR-023, RSK-012): a tendency to skip
ahead to the eventual answer in early layers; unreliable final-layer agreement with the model's actual next
word (upstream: in the top ten only about 67% of the time); and a small set of words appearing spuriously
across many transcripts, mitigated upstream by filtering them from the vocabulary — effective but
unprincipled, and recorded as such.

Template-derived readouts carry a distinct provenance marker and are never presented as lens readouts.

## A.10 Neuronpedia conformance (normative — BR-022, BR-029, BR-030, BR-031) — NEW IN v0.2

Full detail, with source citations, is in `0xcc/brds/neuronpedia-jlens-conformance.md`. This section
carries the normative essentials.

### A.10.1 The integration model

J-lens is **compute-on-demand from a mounted artifact**, not stored per-feature data. Neuronpedia's entire
J-lens database footprint is two tables persisting shared analysis sessions (UI restore state plus a URL to
a gzipped token-stream blob). There is no lens table, no readout table, and no ingestion endpoint.
Therefore **Track A involves no upload**, and building an ingestion API is a non-goal (BR-022).

### A.10.2 On-disk layout (Track A)

```
<exports-dir>/<np_model_id>/jlens/<dataset>/
├── <slug>_jacobian_lens.pt     # the artifact the server loads
├── <slug>_checkpoint.pt        # resumable fit state (not served)
├── <slug>_convergence.csv      # convergence curve (not served)
└── config.yaml                 # reproducibility record
```

`<slug>` derives from the **HuggingFace** model id: take the segment after the last `/`, replace each run
of characters outside `[0-9A-Za-z._-]` with a single `-`, strip leading/trailing `-`. The consuming server
reimplements this to reconstruct remote paths, so miStudio must reproduce it exactly. The
`*_jacobian_lens.pt` suffix is load-bearing — local resolution globs for it, and multiple matches take the
first with a warning. One lens per directory, always.

`config.yaml` records dataset identity, config, split, text field, max chars, prompt count, sequence
length, dtype, device map, and convergence settings, so that the lens is reproducible from the file alone —
the same standard BR-007 sets. Fields miStudio records that the upstream schema lacks go under a namespaced
key rather than a renamed core key.

### A.10.3 Checkpoint schema

A plain `torch.save` dict, loaded with weights-only deserialization:

```python
{
    "J":             {layer_index: Tensor[d_model, d_model]},   # required
    "source_layers": [int, ...],                                 # optional, defaults []
    "n_prompts":     int,                                        # optional, defaults 0
    "d_model":       int,                                        # REQUIRED, no default
}
```

Emitted fp16 (A.1). Layer keys are coerced with `int()`. Absence of `"J"` raises.

### A.10.4 Local instance wiring

Resolution order at server startup: a local override **directory** if set, otherwise download from a
HuggingFace repo at `<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt` into a cache directory. Model
identity resolves through a flat `{np_model_id: hf_model_id}` map at the workspace root; an explicit model
id argument always wins.

The local override directory is the whole of miStudio's runtime contract for Track A. It should be a
mounted volume in the existing deployment manifest, not a push job.

**Two operational properties to design around:**

- **Loading is best-effort.** A failed load never crashes startup; it makes Jacobian-lens requests error at
  request time. Validation is miStudio's responsibility (BR-030, RSK-014).
- **Logit-lens mode needs no artifact.** The lens-mode tabs are Jacobian / logit / diff, so a local
  instance renders logit-lens readouts with zero miStudio involvement, and diff mode is a free visual
  regression surface against any supplied lens.

### A.10.5 Readout stream wire format (BR-029)

miStudio's readout transport mirrors the upstream lens stream:

- **meta** — model id, lens types, per-type layer lists, top-N, prompt length, and prefix-reuse count.
- **token**, one per position — position, decoded token, token id, generated flag, and one slice per lens
  type where each slice carries `top_tokens[layer][k]` and `top_probs[layer][k]`. **All token references
  are decoded strings, never ids.**
- **done** — sequence length, prompt length, vocabulary size, completion text.
- **error** — an error string.

Conforming means a miStudio stream and a Neuronpedia stream are interchangeable at the client.

### A.10.6 Acceptance suite (BR-030)

| Class | Assertion |
|---|---|
| **Structural** | Deserializes weights-only; `"J"` and `"d_model"` present; every Jacobian square of side `d_model`; layer keys coerce to int; `source_layers`, if present, equals the sorted key set |
| **Naming** | Filename matches `<slug>_jacobian_lens.pt` with the slug computed from the HF model id; exactly one lens file in the mounted directory |
| **Envelope** | Size within tolerance of `n_layers × d_model² × 2 bytes` — the BR-006 guard |
| **Semantic** | A fixture prompt with a known unspoken intermediate recovers that intermediate in the top-k at a mid-band layer; catches wrong-model and wrong-layer-indexing artifacts that pass structural checks |
| **Cross-implementation** | Same prompt, layer, and top-k agree between miStudio and the local instance, in **both** Jacobian and logit modes; logit-mode disagreement isolates a readout bug from an artifact bug |
| **Round-trip** | Mount, serve, issue a Jacobian-lens request, confirm non-empty readout. **Explicit**, never inferred from a clean startup log |

Cross-implementation is the highest-value class: it is a free independent implementation to check against,
and it makes BR-001's replication substantially cheaper.

### A.10.7 Acquisition (BR-031)

Preference order: (1) pre-fitted upstream artifacts — 36 models available, largest 70B, with the reference
model's family covered; (2) the reference implementation's fitter; (3) the upstream batch fitter, which
emits the A.10.2 layout natively and is therefore the right choice whenever fitting is actually necessary.
Lenses fitted for uncovered models should be offered upstream, which is an open invitation.
