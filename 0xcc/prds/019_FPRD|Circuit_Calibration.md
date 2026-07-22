# Feature PRD: Circuit Strength Calibration

**Document ID:** 019_FPRD|Circuit_Calibration
**Version:** 1.0
**Status:** Planned
**Related:** 000_PPRD §3.20 (row 20) · PADR IDL-37 · extends the circuits arc (016 discovery / 017 validation+faithfulness / 018 portability) · consumes a promoted circuit, writes back a calibration block on `mistudio.circuit-definition/v1` · grounded in the served-circuit finding on `crc_124fd83d1f2a` (2026-07-21)

---

## 1. Overview

### 1.1 Purpose
Find, automatically, the steering strength at which a circuit is actually *usable* when served — the band
between **onset** (the least influence that does anything) and the **correctness cliff** (the most
influence before the answer stops being true) — and ship that band inside the circuit contract so a served
dial cannot reach the nonsense zone.

### 1.2 User Problem
The circuits arc discovers, validates, and makes a circuit portable, but the per-member steering strengths
it ships are **uncalibrated placeholders**. Taking the first real circuit to serving proved the cost: a
single-prompt strength sweep declared a "usable ceiling" that, served, produced **fluent-but-false**
output — the model confidently asserted an Irish wedding "honors the deceased." The strength that reads as
on-theme sits well past the strength at which the answer stops being *true*, and no perplexity- or
theme-based metric can see that line. Worse, the usable band was narrow (~0.4–0.6 effective) and
off-center from where a fixed sweep sampled, so the sweep stepped over it entirely and reported only the
collapsed region. Every circuit will have this problem, with a different band each time.

### 1.3 Solution
- **Two-detector search:** onset by output-drift-vs-baseline (a difference test, no judge); correctness
  cliff by an LLM judge scoring generations "still true" against falsifiable probes (a property test).
- **Adaptive bisection**, not a fixed grid — finds the band wherever it sits and at whatever width.
- **Probes generated from the circuit's feature labels**, targeting *neutral factual topics the steering
  should not touch*, so degradation shows as the circuit's tint corrupting unrelated facts.
- **Ship the band, not a point:** write `{onset, sweet_spot, cliff}` + evidence into the contract, clamp
  `intensity_range` to `[onset, cliff]`, default `intensity` to the sweet-spot. Badge, not gate.

## 2. User Stories
- **US-1:** I run calibration on a promoted circuit; it returns onset, sweet-spot, and cliff dial values,
  each with the generation and judge verdict that established it, and writes them onto the circuit.
- **US-2:** After calibration the circuit's `intensity_range` is clamped to `[onset, cliff]` and its
  default `intensity` is the sweet-spot, so when it is served the dial cannot be pushed into the band that
  produces confidently-false output.
- **US-3:** Calibration authored no prompts — the probes were generated from the circuit's feature labels,
  on neutral topics, and I can see each probe, its expected-correct answer, and the per-step verdict.
- **US-4:** The resulting band is marked **provisional** because the probes were generated (not authored)
  and it was measured on the discovery-plane model; the probe set travels in the exported contract so a
  serve-time re-verify is one cheap pass.
- **US-5:** I can re-run a stored calibration manifest and reproduce the band within tolerance.
- **US-6 (agent):** Via MCP I calibrate a circuit, read `{onset, sweet_spot, cliff, provisional}`, inspect
  the probe set and per-step verdicts, and fetch/reproduce the manifest — the full loop without the UI.

## 3. Functional Requirements

### 3.1 Onset detection (min influence above none)
1. **Difference test, no judge.** Onset = the smallest dial at which steered output diverges from the
   unsteered baseline past a **noise floor**. Divergence metric: embedding distance (default) or
   output-distribution shift between the steered and matched-unsteered generation on the same probe/seed.
2. The noise floor is established from **baseline-vs-baseline** variation (two unsteered generations at
   different seeds) so onset is "above the model's own sampling noise," not an absolute constant.
3. Search from 0 upward (coarse steps) until divergence crosses the floor; report the crossing dial and
   the measured divergence at it.

### 3.2 Correctness-cliff detection (max before facts break)
4. **Property test via LLM judge.** The cliff = the largest dial at which every probe generation is still
   scored **correct** by the judge; the first dial with a **broken** verdict is above the cliff.
5. **Judge contract:** per (probe, dial) the judge returns `correct | degrading | broken` with a one-line
   reason, scoring the generation against the probe's expected-correct answer. `degrading` is treated as
   past the usable point for the *sweet-spot*, `broken` for the *cliff* — both recorded.
6. **Collapse shortcut:** a cheap repetition/perplexity signal may flag obvious collapse ("par par par")
   to bound the search ceiling *before* spending judge calls — it may raise the ceiling, never decide the
   cliff (the cliff is judge-decided only; this is the mistake the placeholder sweep made).
7. Perplexity/theme metrics are **rejected** as cliff detectors and this is asserted by test — the cliff
   sat between two adjacent dial steps a perplexity delta could not separate.

### 3.3 Adaptive search
8. **Bisection, not a fixed grid.** Onset by upward walk to the floor crossing; cliff by binary search
   between the last **correct** dial and the first **broken** dial. Sweet-spot = the highest **correct**
   dial with a comfortable margin below the cliff (default: midpoint of onset..cliff, biased toward
   correctness — a recorded, tunable policy).
9. **Step budget** (default proposal ~10 generations/probe-set) is disclosed in the manifest; the search
   reports how many steps it used and whether it converged or hit the budget.
10. Search assumes the correctness property is **roughly monotone** in strength (observed universally:
    tint grows, then breaks). A non-monotone result (a `broken` below a `correct`) is flagged in the
    manifest, not silently smoothed — recorded as a search-robustness follow-up.

### 3.4 Probe generation
11. **Generated from feature labels, on neutral topics.** The generator produces 2–3 probes whose subject
    is a **falsifiable factual topic the circuit should NOT influence** — NOT the circuit's own concept
    (whose "right answer" is not falsifiable). It derives the neutrality target from the member labels
    (e.g. a humor circuit → probes on unrelated factual distinctions).
12. Each probe carries its **expected-correct answer** (or key facts) for the judge to score against.
13. Probes generated (not authored) ⇒ the resulting band is marked **`provisional: true`**. (Author-supplied
    probes are a recorded future enhancement; this feature ships the generated path only, per IDL-37.)

### 3.5 Contract carriage & clamping
14. **New nullable `calibration` block** on `mistudio.circuit-definition/v1`:
    `{onset, sweet_spot, cliff, probe_set, judge_metric_id, step_budget, provisional, manifest_ref} | null`
    — additive, nullable, no migration of existing documents (IDL-33 faithfulness-field discipline).
15. On a completed calibration: **clamp** `budget.intensity_range` to `[onset, cliff]` and **default**
    `budget.intensity` to the sweet-spot. The clamp is what makes a served dial unable to reach the
    nonsense zone.
16. **Badge, not gate:** calibration never blocks promotion, steering, or export; a circuit with
    `calibration: null` serves exactly as today.
17. **Manifest** persisted per run (first-class row, `vman_`-style id, kind `calibration`): probe set,
    seeds, every (dial, probe) generation + verdict, divergence measurements, step budget, convergence —
    sufficient to **reproduce** the band. Reproduction is an acceptance test.

### 3.6 Provisional / cross-plane honesty
18. The band is a **starting point**, not a cross-plane guarantee: strength measured on miStudio's model
    instance may differ from what miLLM serves. The **probe set travels in the exported contract** so a
    one-shot serve-time re-verify is cheap. Cross-plane re-verification is **recorded tech debt**, not a
    miLLM deliverable this increment, and not presented as verified.

## 4. User Interface
- **Circuits panel — Calibration (on the promotion/circuit surface, 018):** "Calibrate strength" action;
  a band readout `onset — sweet-spot — cliff` on a dial track with the usable zone shaded; per-probe,
  per-step verdict table (dial, probe, generation excerpt, `correct|degrading|broken`, reason); a
  **provisional** badge with its rationale tooltip; manifest drawer with a reproduce action.
- **Circuit card:** shows the calibrated band and default intensity; a circuit steered outside the clamped
  range is visibly impossible (the dial control's min/max follow `intensity_range`).

## 5. API / Integration
- `POST /api/v1/circuit-calibration` (circuit id, optional step budget / probe count / seed) → task id;
  `GET` results on the circuit; `GET /api/v1/validation-manifests/{id}` serves the calibration manifest
  (reuses 017's manifest store, `kind=calibration`); `POST /{id}/reproduce`.
- MCP (`circuits` category): `calibrate_circuit_strength`, `get_calibration` (or fold into `get_circuit`),
  `reproduce_calibration` — every returned band carries `provisional`. Parameter descriptions per the
  MCP-discoverability gate (every param described; dial semantics + neutral-probe rationale in docstrings).
- WS progress channel per house pattern (`circuit-calibration/{id}`), emitting per-step dial + verdict.

## 6. Data / Types
- `CircuitCalibration` pydantic model + the nullable `calibration` field on `CircuitDefinitionV1`
  (`schemas/circuit_definition.py`); vendored JSON schema regenerated + re-synced to miLLM with the
  pydantic-sync discipline (the schema-sync guard must stay green — this is the failure that has bitten
  this repo before).
- Calibration manifests in the existing `validation_manifests` table (`kind=calibration`), first-class,
  referenced by id.
- `CircuitCalibrationService` mirroring `CircuitFaithfulnessService`'s run/persist shape.
- Alembic migration only if a new column is needed (single-head check); the JSONB circuit record likely
  absorbs the block without DDL — confirmed in FTDD.

## 7. Dependencies
- **018:** the circuit record + contract to write the `calibration` block onto; `intensity_range` /
  `intensity` on `CircuitBudget` (already present) are what the clamp targets.
- **017:** the manifest store (reused, new kind) and the faithfulness service's run/persist/reproduce
  pattern (mirrored).
- **§3.6 steering:** `generate_strength_sweep` is the raw generation engine the search drives.
- **§2.3 enhanced labeling:** the LLM-judge / OpenAI-compatible plumbing scores the cliff.
- **GPU + model:** calibration runs the served model on the discovery host, same envelope as validation.

## 8. Success Criteria
1. On `crc_124fd83d1f2a`, calibration reproduces the hand-found result: onset above 0, a sweet-spot near
   the ~0.4–0.5 dial (effective), and a cliff at/below the ~0.6 dial where the served facts broke — with
   the per-step verdicts as evidence. (Regression pin of the empirical finding this feature exists for.)
2. A perplexity/theme-only calibrator run on the same circuit is shown by test to MISS the cliff (it
   passes 0.6) — proving the judge is load-bearing, not decorative.
3. A completed calibration clamps `intensity_range` to `[onset, cliff]` and sets `intensity` to the
   sweet-spot; a serve-time attempt to dial outside the range is refused (verified end-to-end against the
   live miLLM circuit gate).
4. Probes are generated with zero human authoring, land on neutral topics, and the band is marked
   `provisional`; the probe set is present in the exported contract.
5. Manifest reproduction reproduces the band within tolerance (automated test).
6. Contract round-trips losslessly with the new nullable block; the vendored schema re-syncs and the
   schema-sync guard stays green; existing (calibration-null) documents remain valid.
7. MCP agent completes calibrate → read band → inspect probes/verdicts → reproduce, without the UI, and
   the calibration tool passes the reachability + every-parameter-described gates.

## 9. Non-Goals
- **Author-supplied probes** (generated-only this increment; authored path recorded as future).
- **Serve-time re-verification in miLLM** (the probes travel and the band is provisional; the re-verify
  pass itself is tech debt / a future miLLM increment — NO miLLM code this feature).
- **Joint multi-circuit / compositional calibration** (single circuit; compounding bands are future).
- **Non-monotone search hardening** beyond flagging (recorded).
- **Cross-model transfer** of a band (per-model re-measure; explicitly not promised).
- **Task-metric-specific correctness** beyond the neutral-fact judge (future hardening).

## 10. Testing Requirements
- **Unit:** onset noise-floor arithmetic (baseline-vs-baseline); bisection convergence on a synthetic
  monotone correctness function (finds onset/cliff within tolerance; hits budget gracefully); sweet-spot
  policy; non-monotone flagging; probe-generator neutrality (generated probes are NOT about the circuit's
  own concept — asserted); `CircuitCalibration` schema round-trip + the nullable-block back-compat;
  clamp logic (`intensity_range`←`[onset,cliff]`, `intensity`←sweet-spot); manifest completeness.
- **The judge is load-bearing (negative control):** a fixture where perplexity is flat across the cliff
  but the judge flips `correct→broken` — the calibrator must place the cliff where the JUDGE flips, and a
  perplexity-only variant must be shown to miss it. (Directly pins the mistake this feature corrects.)
- **Integration (GPU):** one real calibration on a small circuit — forward passes asserted, band
  returned, deterministic repeat within tolerance.
- **API:** scope/seed/budget validation; manifest retrieval + reproduce; hostile inputs.
- **Contract/schema:** schema-sync guard green; vendored-copy identity green; a pre-existing
  calibration-null circuit-definition still validates against the regenerated schema.
- **MCP:** reachability (removing the calibrate tool's registration turns the suite red); every parameter
  described; docstring names the dial semantics + neutral-probe rationale + `provisional`.
- **Copy audit:** the shared causal-language grep suite still passes (calibration copy adds no rung claims).
- **E2E:** calibrate `crc_124fd83d1f2a` → band written → export carries it → import to miLLM → attempt an
  out-of-range dial is refused → an in-range served generation is correct; screenshot
  `0xcc/caps/miStudio_Circuit_Calibration_<date>.png`.
- **Mutation controls (repo discipline):** break the clamp (range not narrowed), break the judge wiring
  (cliff always = ceiling), break onset (floor = 0) — each must turn a test red; run as negative controls.

## 11. Traceability

| Source | Covered by |
|---|---|
| PPRD §3.20 / row 20 (usable-band calibration, band shipped in contract) | §1, §3, §8 |
| IDL-37 decision 1 (two detectors, perplexity rejected for cliff) | §3.1, §3.2, §8.2 |
| IDL-37 decision 2 (adaptive bisection not grid) | §3.3 |
| IDL-37 decision 3 (generated neutral-topic falsifiable probes; provisional) | §3.4, §3.6 |
| IDL-37 decision 4 (nullable `calibration` block; clamp; badge not gate) | §3.5, §6 |
| IDL-37 decision 5 (provisional cross-plane; probes travel; tech debt) | §3.6, §9 |
| IDL-37 decision 6 (reuse sweep engine / faithfulness pattern / label-judge) | §7, FTDD |
| Empirical finding on `crc_124fd83d1f2a` (fluent-but-false at placeholder strength) | §1.2, §8.1 |
