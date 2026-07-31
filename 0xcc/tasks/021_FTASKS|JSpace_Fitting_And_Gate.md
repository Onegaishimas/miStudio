# Task List: J-Space Fitting, Artifact Validation & the Phase-0 Gate

**Document ID:** 021_FTASKS|JSpace_Fitting_And_Gate
**Version:** 1.0
**Status:** Planned
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

- [ ] 2.1 `discover_transformer_structure()` for all structure; **no architecture name in the module**.
- [ ] 2.2 Capture at `structure.layers_module[L]` (resid_post) — **never** `residual_norm_module`.
- [ ] 2.3 Backward pass with attention patterns and normalisation statistics frozen.
- [ ] 2.4 Per-layer applicability recorded; inapplicable = absent.
- [ ] 2.5 Convergence stopping, floor 100 prompts; convergence measured on the change in accumulated
      `J`, **not** on any readout-quality proxy.
- [ ] 2.6 Corpus sharding + merge; never split the model.
- [ ] 2.7 fp16; store only `d_model × d_model` per layer; **never** form `W_U J`.
- [ ] 2.8 Conformant on-disk layout: `<slug>_jacobian_lens.pt` + `config.yaml`.
- [ ] 2.9 **Negative control**: fit at `residual_norm_module`, assert measurable degradation.

## Phase 3: Validation suite (BR-030)

- [ ] 3.1 STRUCTURAL · 3.2 NAMING · 3.3 ENVELOPE (model-derived bound) · 3.4 SEMANTIC
- [ ] 3.5 CROSS-IMPLEMENTATION against the local Neuronpedia instance, both modes.
- [ ] 3.6 ROUND-TRIP: mount, serve, request, confirm non-empty. **Explicit, never assumed.**
- [ ] 3.7 Each class independently fails against its own violation.

## Phase 4: Lifecycle, endpoints, readout binding

- [ ] 4.1 Fit → validate → publish; **publish only after all six classes pass**.
- [ ] 4.2 Acquisition path: adopt a conformant lens only when **weight identity** matches.
- [ ] 4.3 Celery task on the correct queue — routes match the TASK NAME, so a short name silently
      uses the default queue.
- [ ] 4.4 Artifact + report endpoints.
- [ ] 4.5 Bind `POST /jlens/readout` for `JACOBIAN_LENS`; the 501 goes away.

## Phase 5: Band report and gate

- [ ] 5.1 Seven metrics (BR-002).
- [ ] 5.2 Position-shuffled null for autocorrelation; size-matched random-direction control for
      excess FVE, with `control_seed` recorded.
- [ ] 5.3 Boundaries derived from this model's own profile; **no default BandReport anywhere**.
- [ ] 5.4 Agreement REPORTED as a layer profile, never SCORED (BR-004).
- [ ] 5.5 Replication report (BR-001), vendored at a recorded commit, published either way.
- [ ] 5.6 GO / NO-GO / GO-AT-LARGER-SCALE recorded with evidence.

## Phase 6: MCP parity (BR-027)

- [ ] 6.1 Reachability test written **first**.
- [ ] 6.2 Tools: fit, validate, list artifacts, band report, replication report, gate decision.
- [ ] 6.3 Registered with the server; presence asserted in the **live registry**.
- [ ] 6.4 Payload and call count asserted — "was called" passes against wrong arguments.

## Phase 7: UI

- [ ] 7.1 Artifacts surface in J-Lens: fit, progress, per-check validation results.
- [ ] 7.2 Band report + gate rendered, `NO_GO` included.
- [ ] 7.3 Jacobian/Diff light up via `meta.types` — **no change to the readout panel**.
- [ ] 7.4 Band shading appears for a model with a report and nowhere else.

## Phase 8: Verification and acceptance

- [ ] 8.1 Two architectures, one hybrid and one dense, both producing valid artifacts.
- [ ] 8.2 Source guard: no architecture name in the fit/readout modules.
- [ ] 8.3 No `n_vocab × d_model` allocation on either path.
- [ ] 8.4 A test that fails if next-token agreement enters a scoring or gating path.
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
