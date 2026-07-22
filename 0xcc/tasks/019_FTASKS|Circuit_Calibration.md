# Task List: Circuit Strength Calibration

**Document ID:** 019_FTASKS|Circuit_Calibration
**Version:** 1.0
**Status:** ✅ IMPLEMENTED + 3-ROUND-REVIEWED + HARDWARE-VERIFIED + UI (2026-07-22). All phases done incl. the REAL GPU `_build_generation_fns` and the Calibration UI. Ran end-to-end on k8s (crc_124fd83d1f2a, k8s miLLM judge): the pipeline works, the judge-sanity gate + greedy generation + badge-not-gate all verified on hardware. **Review:** 3 rounds (15+8+6 findings) + a hardware round (3 findings) = 32 total, all fixed; the R2 round caught a FATAL get_hookable_module arg-order crash that only the "runs on hardware" framing surfaced. **One tracked follow-up (not a blocker):** proving the band lands near ground-truth needs a stronger judge than the 1.2B model this k8s deployment serves — the feature correctly reports `judge_unreliable` rather than fabricating a band. Commits: schema/probes/search 33e5b07, service/task/endpoint/MCP 65cb42c, R1 582ab86, R2 12a7a62, R3 f73d4cd, hardware-R4 d592ac0, UI ac86c6e.
**Source:** 019_FPRD · 019_FTDD · 019_FTID · IDL-37 · grounded in the served-circuit finding on `crc_124fd83d1f2a`

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Contract (schema + clamp) | 2 tasks | ❌ |
| Phase 2: Probe generation | 1 task | ❌ |
| Phase 3: Onset + cliff search | 2 tasks | ❌ |
| Phase 4: Service + manifest + reproduce | 2 tasks | ❌ |
| Phase 5: API + MCP + UI | 3 tasks | ❌ |
| Phase 6: Verification + acceptance | 2 tasks | ❌ |

---

## Phase 1: Contract (gates everything — the schema change is the repo's known failure point)

### Task 1.1: `CircuitCalibration` schema + nullable field + re-vendor
- [ ] `CircuitCalibration(BaseModel)` (onset/sweet_spot/cliff/provisional/probe_set/judge_metric_id/step_budget/manifest_ref/non_monotone) + `CircuitDefinitionV1.calibration: Optional[...] = None`. **Additive + nullable ONLY.** Regenerate `docs/schemas/circuit-definition-v1.json`; re-sync vendored copy to miLLM. **No `alias`** on any new field (validation_alias only if ever needed — the serialization-rename trap). Tests: schema-sync guard green, vendored-identity green, a `calibration:null`/absent document still validates.

### Task 1.2: Clamp + write discipline
- [ ] `clamp_and_persist`: on a COMPLETED run set `budget.intensity_range=[onset,cliff]`, `budget.intensity=sweet_spot`, `calibration=band` — written EXCLUSIVELY via `CircuitService.update` (validators + version precondition; never raw JSONB). Badge-not-gate: calibration=null unchanged; a partial/failed search never narrows the range. Tests: clamp math; completed-only clamp; write-through routes through update().

## Phase 2: Probe generation

### Task 2.1: Neutral-topic falsifiable probe generator
- [ ] `probe_generator.generate_probes(member_labels) -> [{prompt, expected}]` via the enhanced-labeling LLM client — general-knowledge questions on topics UNRELATED to the concept, each with a verifiable short answer. Generated ⇒ band `provisional:true`. **Neutrality assertion test:** generated probes are NOT about the circuit's concept (keyword/embedding check vs member labels).

## Phase 3: Onset + cliff search

### Task 3.1: Onset (difference test, no judge)
- [ ] Noise floor from two unsteered seeds/probe (0.95 qtile of baseline-vs-baseline divergence); upward coarse walk until steered-vs-unsteered divergence crosses it → onset. Divergence = 1−cos(embed) via the configured embeddings endpoint; KL-token-dist fallback if none. Generation via `SteeringService.generate_strength_sweep` (no new loop). Tests: floor arithmetic; inert circuit → onset at floor.

### Task 3.2: Cliff (property test, JUDGE-decided) + the load-bearing-judge control
- [ ] Cheap collapse ceiling (repetition/ppl) may only LOWER the judged range. Judge client (reuse `EnhancedLabelingService`, temp 0) returns correct|degrading|broken + reason per (dial,probe). Bisect between last-CORRECT and first-BROKEN; `worst_over_probes`; sweet-spot = highest CORRECT ≤ cliff−margin (default 0.15) else midpoint biased low; `non_monotone` flag. **NEGATIVE CONTROL test:** fixture with flat perplexity across the cliff but a judge flip correct→broken — calibrator places the cliff at the JUDGE flip, and a perplexity-only variant is shown to MISS it (pins the placeholder-sweep mistake). Bisection-convergence unit test on a synthetic monotone fn.

## Phase 4: Service + manifest + reproduce

### Task 4.1: `CircuitCalibrationService` + task
- [ ] Mirror `CircuitFaithfulnessService` run/persist shape: probes → sweep-driven onset+cliff → sweet-spot → clamp+persist. GPU task + `circuit-calibration/{id}` WS emitting per-step dial+verdict. Every (dial,probe) generation+verdict+reason recorded.

### Task 4.2: Manifest (reuse 017 store) + reproduce
- [ ] Persist a self-contained manifest in `validation_manifests` (kind="calibration"): probes, seeds, all generations+verdicts, divergences, budget, convergence/non-monotone. `POST /{id}/reproduce` re-runs from payload → reproduction manifest with band deltas + tolerance verdict (acceptance test). No new table.

## Phase 5: API + MCP + UI

### Task 5.1: Endpoints
- [ ] `circuit_calibration.py`: `POST` (circuit id, optional budget/probe_count/seed) → task; `GET` results on circuit; `POST /{id}/reproduce`. Scope/seed/budget validation; hostile-input tests.

### Task 5.2: MCP tools (+ discoverability gates)
- [ ] `calibrate_circuit_strength`, `reproduce_calibration` in `tools/circuits.py`; surface band in `get_circuit`. **Every parameter `Field(description=...)`**; docstrings state dial semantics + neutral-probe rationale + `provisional`. **Reachability test:** unregister the tool → suite red. Every-parameter-described gate green.

### Task 5.3: Calibration UI
- [ ] CircuitsPanel calibration section: "Calibrate strength" action; band-on-dial readout (onset—sweet—cliff, usable zone shaded); per-(dial,probe) verdict table (excerpt, chip, reason); provisional badge + tooltip; `CalibrationManifestDrawer` (reuse 017 drawer). Circuit-card dial min/max follow the clamped `intensity_range`.

## Phase 6: Verification + acceptance

### Task 6.1: GPU integration + E2E
- [ ] Real calibration on a small circuit (forward passes asserted; deterministic repeat within tolerance). E2E: calibrate `crc_124fd83d1f2a` → band written (sweet-spot near the hand-found ~0.4–0.5, cliff ≤ ~0.6) → export carries the block → import to miLLM → out-of-range dial REFUSED → in-range served generation is correct. Cap `0xcc/caps/miStudio_Circuit_Calibration_<date>.png`.

### Task 6.2: Acceptance (per instruct 007)
- [ ] FPRD §8 criteria 1–7 verified (incl. the `crc_124fd83d1f2a` regression pin, the load-bearing-judge control, the clamp-refuses-out-of-range serve, calibration-null back-compat, reproduction, reachability). Mutation controls run (clamp / judge wiring / onset floor — each a red, `git diff` clean). Suites green; manual section; CLAUDE.md + PPRD row 20 status update.

---

## Relevant Files

| File | Purpose |
|------|---------|
| `backend/src/schemas/circuit_definition.py` (extend) + `docs/schemas/circuit-definition-v1.json` (regen) + vendored copy | `calibration` block; contract |
| `backend/src/services/circuit_calibration_service.py` (NEW) + `workers/circuit_calibration_tasks.py` (NEW) | search + clamp + persist (GPU task) |
| `backend/src/services/probe_generator.py` (NEW) | neutral-topic falsifiable probes |
| `backend/src/services/circuit_service.py` (extend) | `update`-only clamp write-through |
| `backend/src/api/v1/endpoints/circuit_calibration.py` (NEW) + WS | REST + progress |
| `backend/src/mcp_server/tools/circuits.py` (extend) | `calibrate_circuit_strength`, `reproduce_calibration`, band in `get_circuit` |
| `backend/tests/unit/test_calibration_*.py` (NEW) | schema back-compat, clamp, probe neutrality, bisection, **load-bearing-judge control**, reachability |
| `frontend/src/components/panels/CircuitsPanel.tsx` (calibration section) + `CalibrationManifestDrawer.tsx` (NEW) | UI |
| `manual/docs/**` (circuits.md) | docs |
| reused: `steering_service.generate_strength_sweep`, `EnhancedLabelingService` (judge), 017 `manifest`/`validation_manifests` | no new generation/judge/manifest engines |

## Coverage audit (instruct 007)
- Data ✅ (Ph1 schema, no new table — 017 store reused) · API ✅ (Ph5) · MCP ✅ (Ph5, reachability + param-described) ·
  UI/State ✅ (Ph5) · Tests ✅ (schema back-compat, clamp, probe neutrality, bisection, **judge negative control**,
  reproduction, GPU integration, mutation controls, E2E) · Docs ✅ (Ph5–6) · Acceptance ✅ (Ph6).
  Security: hostile-input tests; manifests self-contained, no secrets/paths.
- **Contract-crosses-to-miLLM:** additive nullable field, reviewed against the miLLM circuits runtime before the schema re-sync; schema-sync + vendored-identity guards are acceptance-blocking.
