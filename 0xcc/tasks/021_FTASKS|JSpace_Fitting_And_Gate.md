# Task List: J-Space Fitting, Artifact Validation & the Phase-0 Gate

**Document ID:** 021_FTASKS|JSpace_Fitting_And_Gate
**Version:** 1.0
**Status:** ⏳ In progress — Phases 2, 3 and 5 implemented and review-round-1 clean; Phases 1, 4, 6, 7 outstanding
**Related:** 021_FPRD · 021_FTDD · 021_FTID · PPRD §3.22 · PADR IDL-40..46

| Phase | Delivers | Gates |
|---|---|---|
| 1 | Data layer | everything |
| 2 | Fitter (model-agnostic, per-layer applicability) | validation |
| 3 | Validation suite (six classes) | publish |
| 4 | Lifecycle service + endpoints + readout binding | the UI's Jacobian mode |
| 5 | Band report + gate | all downstream product surface (BR-003) |
| 6 | MCP parity | agent reachability (BR-027) |
| 7 | UI surface | user-visible delivery |
| 8 | Verification + hardware acceptance | close-out |

---

## Phase 1: Data layer

- [ ] 1.1 ORM: `jlens_artifacts`, `jlens_validation_results`, `jlens_band_reports`, `jlens_gate_decisions`.
- [ ] 1.2 Migration; `NO_GO` representable as a first-class enum value.
- [ ] 1.3 Per-layer applicability persisted with inapplicable stored as **NULL**, never false.

## Phase 2: Fitter

- [x] 2.1 `discover_transformer_structure()` for all structure; **no architecture name in the module**.
- [x] 2.2 Capture at `structure.layers_module[L]` (resid_post) — **never** `residual_norm_module`.
- [x] 2.3 Backward pass with attention patterns and normalisation statistics frozen.
- [x] 2.4 Per-layer applicability recorded; inapplicable = absent.
- [x] 2.5 Convergence stopping, floor 100 prompts; convergence measured on the change in accumulated
      `J`, **not** on any readout-quality proxy.
- [x] 2.6 Corpus sharding + merge; never split the model.
- [x] 2.7 fp16; store only `d_model × d_model` per layer; **never** form `W_U J`.
- [x] 2.8 Conformant on-disk layout: `<slug>_jacobian_lens.pt` + `config.yaml`.
- [x] 2.9 **Negative control**: fit at `residual_norm_module`, assert measurable degradation.

## Phase 3: Validation suite (BR-030)

- [x] 3.1 STRUCTURAL · 3.2 NAMING · 3.3 ENVELOPE (model-derived bound) · 3.4 SEMANTIC
- [x] 3.5 CROSS-IMPLEMENTATION check written; an unreachable consumer is NOT_RUN, never PASS.
      **Wiring to a live instance is Phase 4.**
- [x] 3.6 ROUND-TRIP check written; an empty served readout is a FAIL, not an empty pass.
      **Wiring to a live instance is Phase 4.**
- [x] 3.7 Each class independently fails against its own violation.

## Phase 4: Lifecycle, endpoints, readout binding

- [x] 4.1 Fit → validate → publish; **publish only after all six classes pass**. Stage-then-commit,
      so a half-written artifact is never mounted.
- [ ] 4.2 Acquisition path: adopt a conformant lens only when **weight identity** matches.
- [ ] 4.3 Celery task on the correct queue — routes match the TASK NAME, so a short name silently
      uses the default queue.
- [x] 4.4 Artifact list + validate endpoints. Band-report and gate endpoints wait on Phase 4.5.
- [ ] 4.5 Bind `POST /jlens/readout` for `JACOBIAN_LENS`; the 501 goes away.

## Phase 5: Band report and gate

- [x] 5.1 Seven metrics (BR-002).
- [x] 5.2 Position-shuffled null for autocorrelation; size-matched random-direction control for
      excess FVE, with `control_seed` recorded.
- [x] 5.3 Boundaries derived from this model's own profile; **no default BandReport anywhere**.
- [x] 5.4 Agreement REPORTED as a layer profile, never SCORED (BR-004).
- [ ] 5.5 Replication report (BR-001), vendored at a recorded commit, published either way.
- [x] 5.6 GO / NO-GO / GO-AT-LARGER-SCALE recorded with evidence.

## Phase 6: MCP parity (BR-027)

- [x] 6.1 Reachability test written **first**.
- [x] 6.2 Tools shipped for what EXISTS: `list_jlens_artifacts`, `validate_jlens_artifact`. Fit,
      band-report, replication and gate tools land with their endpoints — a tool calling a route
      that does not exist is the same defect in a new place.
- [x] 6.3 Registered with the server; presence asserted in the **live registry** and in the real
      `build_server()`, not a hand-called `register()`.
- [x] 6.4 Payload and call count asserted — "was called" passes against wrong arguments.

## Phase 7: UI

- [ ] 7.1 Artifacts surface in J-Lens: fit, progress, per-check validation results.
- [ ] 7.2 Band report + gate rendered, `NO_GO` included.
- [ ] 7.3 Jacobian/Diff light up via `meta.types` — **no change to the readout panel**.
- [ ] 7.4 Band shading appears for a model with a report and nowhere else.

## Phase 8: Verification and acceptance

- [ ] 8.1 Two architectures, one hybrid and one dense, both producing valid artifacts.
- [x] 8.2 Source guard: no architecture name in the fit/readout modules.
- [ ] 8.3 No `n_vocab × d_model` allocation on either path.
- [x] 8.4 A test that fails if next-token agreement enters a scoring or gating path — AST guards
      over both `jlens_validation` and `jlens_band_report`.
- [ ] 8.5 Mutation controls, each red: hook the norm module; `False` for inapplicable; materialise
      `W_U J`; hardcode the envelope; score the gate on agreement; skip ROUND-TRIP; publish before
      validation; drop an MCP registration; accept a weight-identity mismatch.
- [ ] 8.6 **Hardware acceptance**: fit the reference model on the local 3080 Ti, validate, serve a
      real Jacobian readout, and confirm Diff shows the two lenses genuinely differing in early
      layers.
- [ ] 8.7 Three rounds of security-review and review; all findings fixed and re-verified.

**Acceptance:** a validated artifact exists for two architectures; every validation class passes
including a live round-trip; a band report derives that model's own boundaries and bands appear only
there; the gate decision is recorded with evidence and a NO-GO is representable; nothing scores the
lens on next-token agreement; and every capability is reachable by an agent under the reachability
harness.

---

## Relevant Files

| file | purpose |
|---|---|
| `backend/src/ml/jlens_fitter.py` | fit `J` per layer; shard + merge |
| `backend/src/ml/jlens_metrics.py` | band metrics + null controls |
| `backend/src/services/jlens_validation.py` | the six BR-030 classes |
| `backend/src/services/jlens_artifact_service.py` | lifecycle |
| `backend/src/services/jlens_band_report.py` | boundaries + gate |
| `backend/src/workers/jlens_fit_tasks.py` | Celery task |
| `backend/src/api/v1/endpoints/jlens.py` | extended; binds the readout |
| `backend/src/mcp_server/tools/jlens.py` | MCP parity, registered |
| `frontend/src/components/jlens/ArtifactsPanel.tsx` | fit + validation UI |

---

## Coverage audit

| FPRD requirement | Phase |
|---|---|
| §3.1 model-agnostic construction | 2.1, 2.2 |
| §3.2 per-layer applicability | 1.3, 2.4 |
| §3.3 fitting | 2.5–2.8 |
| §3.4 acquisition (weight identity) | 4.2 |
| §3.5 validation before handover | 3 |
| §3.6 band report | 5.1–5.3 |
| §3.7 agreement never a quality metric | 5.4, 8.4 |
| §3.8 replication report | 5.5 |
| §3.9 the gate | 5.6 |
| §3.10 MCP parity | 6 |

---

## Carried open questions

- **Sparsity `k`** for decomposition occupancy — `TBD (BRD open question)`. Blocks the metric's
  absolute values, not its comparison against the control.
- **Layer subsampling** — resolved at reference scale (16 layers analysed in full); survives only for
  models larger than the subsampling target.


---

## Progress record (2026-07-31)

**Implemented and review-round-1 clean:** the fitter (`jlens_fitter.py`), the six-class validation
suite (`jlens_validation.py`), the band metrics with their null controls (`jlens_metrics.py`), and
the band report + gate (`jlens_band_report.py`). 52 tests. Backend suite 2138, green.

**Round-1 findings, all mine, all fixed:**

1. **The fitter could not have run.** `jacobian_of` took one forward-mode pass per input dimension —
   d_model x n_layers x n_prompts is millions of passes on the reference model. Freezing makes the
   map AFFINE, so the columns come from one batched call per chunk: `J[:, i] = fn(e_i) - fn(0)`.
   The jvp version is kept as `jacobian_by_jvp` and a test asserts the two agree, so the assumption
   the fast path rests on is verified rather than asserted.
2. **The affine assumption was unchecked.** If a norm or an attention pattern escapes the freeze the
   extracted matrix is a local linearisation of nothing in particular — and is the right shape and
   size, so it passes STRUCTURAL, NAMING and ENVELOPE. `affine_residual` measures the departure and
   the fit is REFUSED above the limit. Fit time is the only point where this is detectable.
3. **Downstream layers were replayed with hidden states alone.** A real decoder layer needs position
   ids, an attention mask and rotary embeddings; calling it without them either raises or silently
   takes a default path and fits a lens for a model that was never run that way. Kwargs are now
   recorded during the reference forward and replayed.
4. **The envelope allowance was wrong at both ends.** The check compares a FILE size against a
   TENSOR size, so the container's own bytes sit between them — fine at 134 MB, dominant at test
   scale. A flat allowance fixed that and then exceeded a small model's ENTIRE materialised
   dictionary, blinding the check exactly where the numbers are smallest. Now bounded at both ends,
   and the cap never binds at real scale.
5. **`_norm_modules` matched "norm" anywhere in a class name**, capturing a decoder block named
   `NormedBlock`. Freezing a decoder block replaces it with an elementwise rescaling. Tightened to
   `endswith("norm")` plus an independent tensor-output guard.

**Two mutations survived their first run** — the affine guard and the norm-name rule — and both are
now pinned by regressions re-verified as negative controls. Nine mutation controls verified biting
across the fitter; six across the band report and gate.

**Also implemented since:** `jlens_artifact_service.py` — the artifact lifecycle. **The filesystem is
the registry, not a database table** (PADR IDL-46): a J-lens artifact is consumed by MOUNTING a
conformant directory and there is no upload path, so a DB row as source of truth would invent a
second registry that can disagree with the one the consumer actually reads — silently. Stage-then-
commit; `commit` refuses anything short of a full pass; `load_for_readout` refuses to serve without
a passing report. 20 tests, 6 mutation controls.

That removes Phase 1 (ORM + migration) from the critical path — it is now optional bookkeeping over
a filesystem that is already authoritative.

**Also implemented since:** the artifact list/validate endpoints and the `jlens` MCP category
(`list_jlens_artifacts`, `validate_jlens_artifact`), registered and covered by a reachability harness
written before the tools. Five mutation controls, including unregistering the category — the exact
defect that once shipped 16 tools nobody could call.

**Outstanding for this feature:** Phase 4.2-4.5 (acquisition weight-identity check, Celery task,
endpoints, readout binding — **the 501 stays until then**), Phase 6 (MCP tools + reachability),
Phase 7 (UI), Phase 8.1/8.6 (two-architecture and hardware acceptance), and review rounds 2-3.

**The cross-implementation and round-trip checks are written but not yet wired to a live consumer.**
They are correct and independently tested; until Phase 4 they cannot run against a real instance,
and `ValidationReport.passed` fails closed on a class that never ran, so nothing can be published on
their absence.
