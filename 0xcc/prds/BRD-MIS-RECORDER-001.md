# BRD: Steered Transcript Recorder

**Document ID:** BRD-MIS-RECORDER-001
**Status:** Implemented (3-round-reviewed; hardware E2E at close-out)
**Related:** PADR IDL-38 · extends Feature 20 Circuit Strength Calibration (IDL-37) · uses the calibration GPU steering core + the validation-manifest store

---

## 1. Why

Feature 20 calibration finds a circuit's usable strength band by having an LLM
judge grade steered outputs. Its real-hardware run exposed the limit: a small
served model (LFM2.5-1.2B) is too weak to grade its own factual correctness, so
calibration correctly reported `judge_unreliable` rather than fabricating a band.
That is the right behaviour, but it strands the *automated* cliff-finding whenever
the only judge available is weak.

The reframe (user): the run is the **instrument**, not the judge. Steering already
generates exactly the raw material worth analysing — a prompt, its unsteered
baseline, and the steered response at each dial. The valuable analysis —
*"semantically, what did steering this artifact DO to the outputs, and how did it
change across the dial?"* — should be done by a strong model (Opus 4.8), driven
by the agent, **after** the run, as interpretation. Not a pass/fail gate inside a
search loop.

The gap that blocked this: the calibration manifest recorded the *verdict* per
(dial, prompt) but not the *generated text* — so a consumer could see "at dial 0.6
the answer was broken" but never *what the answer was*, which is what a
meaning-analysis pass needs.

## 2. What

- **A general recorder** over MCP: pick a **circuit, a cluster profile, or an
  ad-hoc feature set**, a set of dials, and a set of prompts; miStudio generates
  on the GPU and records `(dial, prompt, unsteered_output, steered_output)`
  transcripts; the agent reads them back and hands them to Opus for a qualitative
  reading. Judge-free.
- **First-class transcripts on the calibration record** too — every calibration
  manifest now carries the generated text, so it is directly analysis-ready.

## 3. Requirements

- **BR-R1 Unified steering core.** One GPU generation core drives all three
  artifact types via per-type member resolvers; the circuit path (calibration's
  32-findings-hardened `_build_generation_fns`) is refactored to use it, byte-
  identical. Standardized on the residual hook target — so a recorded transcript
  matches the calibrated band. The recorder owns the dial multiply.
- **BR-R2 Recorder + store.** `record_samples(artifact, dials, prompts,
  max_tokens, seed)` records the transcripts to a new `steering_samples` manifest
  kind carrying the actual generated TEXT. Caps bound the one-GPU job: ≤8 dials,
  ≤8 prompts, ≤200 tokens, each dial ≤2.0, prompts×(1+dials) ≤64. Config is
  validated UP FRONT (per-kind required refs incl. one-SAE-per-layer) so a
  malformed request 422s before the GPU lock is taken.
- **BR-R3 Calibration transcripts.** The search trace stores the generated text
  per (dial, prompt) incl. the shipped sweet-spot re-judge; the calibration
  manifest carries a required `transcripts` field. Back-compat: `validate_payload`
  runs only on write, so legacy manifests (no transcripts) stay readable +
  reproducible.
- **BR-R4 GPU lifecycle.** The recorder holds the single GPU like calibration.
  A dedicated `steering_record_runs` marker table (uniform across all three
  artifact types, since cluster/feature jobs have no circuit row) is checked by
  the single-GPU guard and reclaimed by the stuck-run cleanup.
- **BR-R5 Reachability.** `record_steering_samples` + `get_steering_samples` MCP
  tools, every parameter described, in the howto index; behaviour-based
  reachability tests; the Celery task autodiscovered + GPU-routed.

## 4. Non-goals

- No change to the live feature/cluster SERVE path (only the recorder standardizes
  the hook target).
- No re-running of cluster allocation (uses persisted tuned strengths).
- The Opus meaning-analysis prompt/format is downstream (agent-driven); miStudio's
  job ends at analysis-ready transcripts.
- No frontend UI this increment (MCP-driven).

## 5. Acceptance

1. Record on a circuit, a cluster profile, AND an ad-hoc feature set (via MCP) →
   each transcript carries a non-empty `unsteered_output` + one `steered_output`
   per dial, steered≠unsteered at higher dials, on real k8s hardware.
2. Malformed artifact / caps → 422 up front; a second concurrent record → 409.
3. Calibration is unregressed (its manifest now carries `transcripts`; a re-run
   still reaches `judge_unreliable` on the weak k8s judge, unchanged).
4. 3-round review + mutation controls; the unified-core refactor byte-identical.
