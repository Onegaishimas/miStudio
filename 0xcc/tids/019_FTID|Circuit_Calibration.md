# Technical Implementation Document: Circuit Strength Calibration

**Document ID:** 019_FTID|Circuit_Calibration
**Version:** 1.0
**Status:** Planned
**Related:** 019_FPRD · 019_FTDD · IDL-37 · reuses 017 (manifest store, faithfulness pattern) + §3.6 steering (`generate_strength_sweep`) + §2.3 enhanced-labeling judge

---

## 1. Implementation Order

1. `CircuitCalibration` schema + nullable `CircuitDefinitionV1.calibration`; regenerate + re-vendor JSON; back-compat test (calibration=null still valid). **Schema first — the contract change gates everything and is the repo's known failure point.**
2. Probe generator (label→neutral-topic prompts + expected answers) — pure/unit, no GPU.
3. Onset (noise floor + upward walk) against the sweep engine — GPU-light on a small circuit.
4. Judge client + cliff bisection — the load-bearing piece; negative-control test alongside.
5. Clamp + persist + manifest (reuse 017 store, kind="calibration"); reproduce skeleton.
6. Service assembly + task + endpoints + WS.
7. MCP tools (+ band in `get_circuit`); reachability + every-param-described gates.
8. Frontend calibration surface; GPU integration; reproduction test; E2E.

## 2. File-by-file

### 2.1 `backend/src/schemas/circuit_definition.py` (extend)
- `class CircuitCalibration(BaseModel)` per FTDD §5; `CircuitDefinitionV1.calibration: Optional[...] = None`.
- **Additive + nullable ONLY.** Regenerate `docs/schemas/circuit-definition-v1.json` via the existing generator; re-sync the vendored copy to miLLM; keep `test_circuit_definition_schema_sync` + the vendored-identity guard green. Add a test: a `calibration:null` (and calibration-absent) document validates against the regenerated schema.
- **Do NOT use `alias` on any new field** — `validation_alias` if an input synonym is ever needed (the serialization-rename trap that invalidated exports before; see the pydantic-alias lesson).

### 2.2 `backend/src/services/circuit_calibration_service.py` (NEW) + `workers/circuit_calibration_tasks.py`
- Mirror `CircuitFaithfulnessService`'s run/persist/reproduce shape (member/param resolution → GPU work → scores + manifest → write-back).
- **Generation:** call `SteeringService.generate_strength_sweep` per (dial, probe, seed) — do NOT write a new generation loop. One dial = the global `intensity` multiplier; member strengths untouched.
- **Onset** (FTDD §2): noise floor from two unsteered seeds per probe (0.95 qtile of baseline-vs-baseline divergence); walk coarse steps low→high until steered-vs-unsteered divergence crosses it. Divergence = 1−cos(embed) via the configured embeddings endpoint; KL-on-token-dist fallback if none.
- **Cliff** (FTDD §3): cheap collapse ceiling first (repetition ratio / ppl multiple) — may only LOWER the judged range; then bisect between last-CORRECT and first-BROKEN using the judge. `worst_over_probes` is conservative. Record every (dial, probe) generation + verdict + reason. Flag `non_monotone` if a BROKEN sits below a CORRECT.
- **Sweet-spot:** highest CORRECT dial ≤ cliff−margin (default margin 0.15), else midpoint biased low. Policy is a named constant, unit-tested.
- **Clamp + persist:** `budget.intensity_range = [onset, cliff]`, `budget.intensity = sweet_spot`, `calibration = band`, write through **`CircuitService.update`** (contract validators + version precondition — never mutate the JSONB column directly; same 018 hand-off discipline 017 followed). Persist manifest via the 017 store.

### 2.3 `backend/src/services/probe_generator.py` (NEW)
- `generate_probes(member_labels) -> [{prompt, expected}]` via the enhanced-labeling LLM client. Prompt asks for general-knowledge questions on topics UNRELATED to the concept, each with a short verifiable answer.
- **Neutrality assertion** is a unit test: generated probes must not be about the circuit's concept (keyword/embedding check against member labels). Generated ⇒ band `provisional:true`.

### 2.4 Judge — reuse `EnhancedLabelingService` client (no new provider plumbing)
- `judge(generation, expected) -> {verdict: correct|degrading|broken, reason}`; OpenAI-compatible, same endpoint/key config as labeling. Deterministic settings (temp 0) for reproducibility. `judge_metric_id` recorded in the manifest + contract.

### 2.5 Manifest (reuse 017 `validation_manifests`, kind="calibration")
- Payload = probes, seeds, every (dial, probe) generation + verdict + reason, divergence measurements, step budget, convergence/non-monotone flags — self-contained. `POST /{id}/reproduce` re-runs from payload, stores a `reproduction` manifest with band deltas + tolerance verdict (acceptance test).

### 2.6 Endpoints + MCP + WS
- `backend/src/api/v1/endpoints/circuit_calibration.py`: `POST` (circuit id, optional budget/probe_count/seed) → task; `GET` results on circuit; `POST /{id}/reproduce`. WS `circuit-calibration/{id}` emits per-step dial + verdict.
- MCP in `tools/circuits.py`: `calibrate_circuit_strength`, `reproduce_calibration`; surface the band in `get_circuit`. **Every parameter gets a `Field(description=...)`** (the discoverability gate); docstrings state dial semantics, neutral-probe rationale, and `provisional`. **Reachability test:** removing the tool registration must turn the suite red.

### 2.7 Frontend
- CircuitsPanel calibration section on the promotion/circuit surface: "Calibrate strength" action; band-on-a-dial readout (`onset — sweet — cliff`, usable zone shaded); per-(dial,probe) verdict table (excerpt, chip, reason); **provisional** badge + tooltip; `CalibrationManifestDrawer` reusing the 017 manifest-drawer pattern. Circuit card's dial min/max follow the clamped `intensity_range`.

### 2.8 Manual
- `circuits.md` += a Calibration section: the two thresholds, why perplexity can't find the cliff, why probes are neutral-topic, and that the band is provisional + travels for serve-time re-verify.

## 3. Pitfalls

- **Perplexity is a ceiling shortcut, not the cliff.** The cliff is whatever dial the JUDGE flips at. Using a cheap signal to DECIDE the cliff is the exact placeholder-sweep mistake — pinned by the load-bearing-judge negative control.
- **Probes must be NEUTRAL-topic** — a probe about the circuit's concept has no falsifiable answer and the cliff becomes undetectable. Assert it.
- **Additive nullable schema only** — never a rename/removal; regenerate + re-vendor + keep the sync guards green; a pre-existing calibration-null document must stay valid. This contract crosses to miLLM.
- **Never mutate the circuit JSONB directly** — clamp writes go through `CircuitService.update` (validators + version precondition), same as 017's edge writes.
- **Clamp only on a COMPLETED run** — a partial/failed search must not narrow `intensity_range` (would silently disable a working circuit). Badge, not gate.
- **`worst_over_probes`** for verdicts — one probe passing while another breaks means the dial is past the cliff. Do not average verdicts.
- **Reuse the sweep engine and the label-judge client** — do not fork generation or add a second LLM-provider path (the labeling plumbing already handles reasoning models, keys, endpoints).
- **Band is provisional / cross-plane** — do not present a miStudio band as serve-verified; the probe_set travels so re-verify is cheap, but that pass is not this feature.

## 4. Testing

- **Unit:** noise-floor arithmetic; bisection on a synthetic monotone correctness fn (finds onset/cliff in tolerance; graceful at budget); sweet-spot policy; non-monotone flagging; probe neutrality assertion; `CircuitCalibration` round-trip + calibration-null back-compat; clamp logic; manifest completeness.
- **Load-bearing-judge negative control:** fixture with flat perplexity across the cliff but a judge flip correct→broken — calibrator places the cliff at the JUDGE flip; a perplexity-only variant misses it (asserted). Pins the mistake this feature corrects.
- **Integration (GPU):** one real calibration on a small circuit — forward passes asserted, band returned, deterministic repeat within tolerance.
- **API:** scope/seed/budget validation; manifest retrieval + reproduce; hostile inputs.
- **Contract/schema:** schema-sync + vendored-identity green; calibration-null document still validates.
- **MCP:** reachability (unregister → red); every-parameter-described; docstring content.
- **Copy audit:** shared causal-language suite still green (no new rung claims).
- **Mutation controls:** break the clamp (range not narrowed) / judge wiring (cliff=ceiling) / onset (floor=0) — each turns a test red; run as negative controls, `git diff` clean after.
- **E2E:** calibrate `crc_124fd83d1f2a` → band written → export carries it → import to miLLM → out-of-range dial refused → in-range served generation is correct; screenshot `0xcc/caps/miStudio_Circuit_Calibration_<date>.png`.
