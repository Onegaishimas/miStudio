# Technical Design Document: Circuit Strength Calibration

**Document ID:** 019_FTDD|Circuit_Calibration
**Version:** 1.0
**Status:** Planned
**Related:** 019_FPRD · IDL-37 · consumes 018 (circuit record + contract) + 017 (manifest store, faithfulness run/persist pattern) + §3.6 steering (`generate_strength_sweep`) + §2.3 enhanced-labeling judge · writes the `calibration` block onto `mistudio.circuit-definition/v1`

---

## 1. Search core

```
calibrate(circuit, cfg{step_budget=10, seed=0, probe_count=3, margin=0.15}):
  probes  = generate_probes(circuit)                 # §4, neutral-topic falsifiable
  lo, hi  = intensity_range floor/ceiling of the circuit's authored range
  floor   = noise_floor(circuit, probes, seed)       # §2, baseline-vs-baseline
  onset   = walk_up_to_floor(lo..hi, floor)          # §2
  ceiling = collapse_ceiling(hi, cheap_signal)       # §3, may LOWER hi before judging
  cliff   = bisect_cliff(onset..ceiling, probes)     # §3, JUDGE-decided
  sweet   = highest CORRECT dial ≤ cliff − margin, else midpoint(onset, cliff) biased low
  band    = {onset, sweet, cliff, provisional:true, step_budget, judge_metric_id, probe_set}
  clamp_and_persist(circuit, band, manifest)         # §5
```

- One "dial" = the global `intensity` multiplier on the circuit's per-member strengths (effective =
  dial × Σ member strengths). Calibration tunes the DIAL; member strengths are not touched.
- `generate_strength_sweep` (steering service) is the raw generation engine — called per (dial, probe,
  seed), matched greedy where a baseline is needed. No new generation code.

## 2. Onset (difference test, no judge)

```
noise_floor(circuit, probes, seed):
  for probe: g0 = gen(dial=0, seed);  g0' = gen(dial=0, seed+1)   # unsteered, two seeds
  floor = qtile_0.95( divergence(g0, g0') over probes )           # the model's own sampling noise
divergence(a, b) = 1 − cos(embed(a), embed(b))                    # embedding drift (default)
walk_up_to_floor(range, floor):
  for dial in coarse_steps(range):                                # e.g. 6 steps low→high
    d = mean_probe divergence( gen(dial), gen(dial=0) )           # steered vs unsteered, same seed
    if d > floor: return dial                                     # first crossing = onset
  return range.lo                                                 # inert circuit → onset at floor
```

- Embedding model = the same OpenAI-compatible embeddings endpoint the platform already configures;
  fallback to output-token-distribution KL if no embeddings endpoint (recorded, degrades gracefully).

## 3. Cliff (property test, JUDGE-decided)

```
collapse_ceiling(hi, signal):        # CHEAP, may only LOWER the search ceiling, never set the cliff
  raise if repetition_ratio(gen(hi)) > 0.5 or ppl(gen(hi)) > K·ppl(gen(0))
  → binary-search DOWN for the first non-collapsed dial; that becomes the judge's upper bound
bisect_cliff(lo..hi, probes):
  invariant: verdict(lo)=CORRECT, verdict(hi)=BROKEN            # seed the ends; widen if needed
  while steps_left and (hi − lo) > tol:
    mid = (lo + hi)/2
    v = worst_over_probes( judge(gen(mid, probe), probe.expected) )   # correct<degrading<broken
    if v == CORRECT: lo = mid ; last_correct = mid
    else:            hi = mid                                    # degrading OR broken ⇒ above sweet/cliff
  cliff = first dial judged BROKEN ; sweet uses last CORRECT
```

- **Judge** = enhanced-labeling LLM plumbing (OpenAI-compatible). Prompt: "Here is a question, its
  correct answer, and a model response. Is the response CORRECT / DEGRADING / BROKEN? one-line reason."
  Verdict + reason stored per (dial, probe).
- **Perplexity is NOT a cliff detector** — only a ceiling shortcut. The cliff is whatever dial the JUDGE
  flips at. Asserted by the load-bearing-judge negative-control test (FPRD §10, §8.2).
- Monotonicity: a BROKEN below a CORRECT during bisection ⇒ `non_monotone:true` in the manifest, band
  still reported from the lowest break (conservative).

## 4. Probe generation (neutral-topic, falsifiable)

```
generate_probes(circuit):
  concept = summarize(member labels)                 # e.g. "humor / parody / comedians"
  ask the label-LLM: "Give 3 general-knowledge questions on topics UNRELATED to {concept},
     each with a short factual correct answer. The questions must have a verifiable right answer."
  → [{prompt, expected} ...]   ; store verbatim in the manifest AND the contract probe_set
```

- **Why neutral, not on-concept:** a probe *about* humor has no falsifiable answer, so the judge cannot
  detect the cliff. A neutral factual probe DOES — degradation shows as the circuit's tint corrupting
  unrelated facts (the empirical signature: "an Irish wedding honors the deceased"). Asserted by test:
  generated probes must not be about the circuit's own concept.
- Generated ⇒ `provisional:true` on the band. Author-supplied probes = recorded future (FPRD §9).

## 5. Contract carriage, clamp, persist

```
CircuitCalibration(BaseModel):                       # schemas/circuit_definition.py, nullable field
  onset: float; sweet_spot: float; cliff: float
  provisional: bool = True
  probe_set: list[{prompt, expected}]
  judge_metric_id: str; step_budget: int; manifest_ref: str | None
  non_monotone: bool = False
CircuitDefinitionV1.calibration: Optional[CircuitCalibration] = None      # additive, no migration

clamp_and_persist(circuit, band, manifest):
  circuit.budget.intensity_range = [band.onset, band.cliff]     # served dial CANNOT exceed the band
  circuit.budget.intensity       = band.sweet_spot              # default to the usable point
  circuit.calibration            = band
  persist manifest (validation_manifests, kind="calibration")   # 017 store, reused
  # badge, not gate: a circuit with calibration=null is unchanged; clamp only on a completed run
```

- **Schema discipline (the repo's recurring failure mode):** regenerate `circuit-definition-v1.json`,
  re-sync the vendored copy to miLLM, keep the schema-sync + vendored-identity guards green. A
  pre-existing `calibration:null` document must still validate (nullable/additive verified by test).

## 6. Architecture / types

```
CircuitCalibrationService — probe gen, sweep-driven search, judge orchestration, clamp+persist (GPU task)
  mirrors CircuitFaithfulnessService.run/persist/reproduce shape
reuses: SteeringService.generate_strength_sweep (generation); EnhancedLabeling judge client; embeddings client
endpoints: circuit_calibration.py (POST run, GET results, POST /{id}/reproduce); WS circuit-calibration/{id}
manifests: validation_manifests (kind="calibration") — 017 store, no new table
schema: CircuitCalibration + nullable CircuitDefinitionV1.calibration; regenerated + vendored JSON
MCP (circuits): calibrate_circuit_strength, reproduce_calibration (+ band surfaced in get_circuit)
frontend: Calibration section on CircuitsPanel promotion surface; band-on-dial readout; verdict table; manifest drawer
```

## 7. Risks

| Risk | Mitigation |
|---|---|
| Perplexity/theme used as the cliff (the exact placeholder-sweep mistake) | cheap signal is a CEILING shortcut only; cliff is judge-decided; load-bearing-judge negative control pins it |
| Generated probes not falsifiable (on-concept) | neutral-topic generation + expected-answer required + "not about the concept" assertion test; band marked provisional |
| Judge is wrong / flaky | worst-over-probes (conservative); reason stored for audit; reproduce re-runs; provisional band; serve-time re-verify path |
| Non-monotone correctness vs strength | flagged in manifest, conservative lowest-break cliff; denser sampling recorded as follow-up |
| Cross-plane strength drift (miStudio vs miLLM) | band is a starting point, provisional; probe_set travels; re-verify = recorded tech debt, not claimed verified |
| Schema divergence / stale vendored copy | regenerate + re-sync + keep schema-sync & vendored-identity guards green; back-compat test for calibration=null |
| GPU cost of the search | bounded step budget (disclosed); cheap ceiling shortcut avoids judging collapsed dials; badge-not-gate ⇒ skippable |
| Contract change crosses to miLLM | additive nullable field only; reviewed against the miLLM circuits runtime before the schema re-sync (IDL-33 discipline) |
