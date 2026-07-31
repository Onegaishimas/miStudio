# Task List: J-Space Readout Substrate & Wire Format

**Document ID:** 022_FTASKS|JSpace_Readout_Substrate
**Version:** 1.0
**Status:** Planned
**Related:** 022_FPRD · 022_FTDD · 022_FTID · PPRD §3.23 · PADR IDL-40..42, IDL-45

| Phase | Delivers | Gates |
|---|---|---|
| 1 | Wire-format schemas | the frontend and every later phase |
| 2 | Structure resolution + applicability | all capture |
| 3 | Residual capture + hook negative control | correctness of every readout |
| 4 | Transport interface + logit path | the substitution point |
| 5 | Three modes + provenance | figures being valid at all |
| 6 | Envelope guard + endpoints | publication and consumption |
| 7 | Verification + acceptance | — |

---

## Phase 1: Wire-format schemas

- [ ] 1.1 `schemas/jlens.py`: `LensMetaMessage`, `LensTokenMessage`, `LensTypeSlice` matching the
      upstream shape exactly.
- [ ] 1.2 Type `top_tokens` as `List[List[str]]` — decoded strings, not ids.
- [ ] 1.3 `ReadoutRequest`, `DecompositionResult`, `JLensArtifact`, `LayerApplicability`.
- [ ] 1.4 Test: a slice whose `top_tokens` are ints fails validation.

## Phase 2: Structure resolution + applicability

- [ ] 2.1 Resolve structure via `discover_transformer_structure`; no architecture branch.
- [ ] 2.2 Build the per-layer `LayerApplicability` map from the model config's layer types.
- [ ] 2.3 Inapplicable metrics are represented as **absent**, never `false`/`0`.
- [ ] 2.4 Test: a hybrid fixture reports 6 attention-applicable of 16 layers; a dense fixture reports
      all layers applicable.

## Phase 3: Residual capture + hook negative control

- [ ] 3.1 Capture at `structure.layers_module[L]`.
- [ ] 3.2 **Negative control**: capture at `residual_norm_module` and assert measurable degradation.
- [ ] 3.3 Capture device chosen explicitly, not inherited.
- [ ] 3.4 Test: capture shape is `[positions, d_model]` per layer for both architectures.

## Phase 4: Transport interface + logit path

- [ ] 4.1 `LensTransport` interface; `IdentityTransport` (`J = I`).
- [ ] 4.2 `W_U` via `analysis_service.load_unembedding_matrix` — do not instantiate the model.
- [ ] 4.3 Readout pinned to CPU.
- [ ] 4.4 Test: logit readout succeeds with **no artifact present**.
- [ ] 4.5 Test: no `.cuda()` / `device="cuda"` on the readout path.

## Phase 5: Three modes + provenance

- [ ] 5.1 Full ranked → sorted vocabulary readout.
- [ ] 5.2 Probe → single-direction score; record which mode is canonical per analysis.
- [ ] 5.3 Sparse decomposition by gradient pursuit.
- [ ] 5.4 Mandatory provenance: `k`, solver id/params, iterations, convergence, control seed.
- [ ] 5.5 Validation **rejects** a decomposition result lacking a control seed.
- [ ] 5.6 Test: pursuit and top-k return different active sets on a correlated frame.
- [ ] 5.7 Confirm no next-token-agreement metric exists anywhere in the feature.

## Phase 6: Envelope guard + endpoints

- [ ] 6.1 Envelope bound computed from the model's own `d_model` / `n_vocab` / `n_layers`.
- [ ] 6.2 Assert no `n_vocab × d_model` allocation on any path.
- [ ] 6.3 `/readout`, `/probe`, `/decompose`; register in `api/v1/router.py`.
- [ ] 6.4 Stream emits `meta` → `token`* → `done`, with `error` carrying a reason.
- [ ] 6.5 Test: changing a fixture's `n_vocab` moves the envelope bound.

## Phase 7: Verification + acceptance

- [ ] 7.1 Two-architecture readout (hybrid + dense), no code change between.
- [ ] 7.2 Guard: no architecture name in the service module outside comments.
- [ ] 7.3 Full backend suite green against baseline; no new failures.
- [ ] 7.4 **Mutation controls**, each must go red: hook the norm module; hardcode the envelope;
      emit ids; top-k for pursuit; `frozen_qk=false` on a conv layer; accept a missing control
      seed; move the readout to CUDA.
- [ ] 7.5 Three rounds of security-review + review; all findings fixed and re-verified.

**Acceptance:** a logit-lens readout streams for two architectures with no artifact and no
architecture branch; `JacobianTransport` is the only diff needed when an artifact arrives; every
occupancy figure carries a control seed; the envelope guard fails on a materialised dictionary.

---

## Relevant Files

| file | purpose |
|---|---|
| `backend/src/schemas/jlens.py` | wire-format and artifact types |
| `backend/src/services/jlens_readout_service.py` | transports, capture, three modes |
| `backend/src/api/v1/endpoints/jlens.py` | readout / probe / decompose |
| `backend/src/api/v1/router.py` | registration |
| `backend/tests/unit/test_jlens_readout.py` | modes, wire conformance, provenance |
| `backend/tests/unit/test_jlens_model_agnostic.py` | two-architecture, hook control, name guard |
| `backend/tests/unit/test_jlens_envelope.py` | model-derived envelope |

Reused unchanged: `ml/layer_discovery.py`, `ml/forward_hooks.py`,
`services/analysis_service.py` (`load_unembedding_matrix`, `resolve_snapshot_dir`).

---

## Coverage audit (instruct 007)

| FPRD requirement | Phase |
|---|---|
| §3.1 logit-lens first, single substitution point | 4 |
| §3.2 model-agnostic; decoder-layer capture | 2, 3 |
| §3.3 three modes; decomposition-only figures | 5 |
| §3.4 storage discipline; model-derived envelope | 6 |
| §3.5 recipe provenance; per-layer applicability | 2, 5 |
| §3.6 decomposition provenance; invalid without seed | 5 |
| §3.7 wire format conformance | 1, 6 |
| §3.8 no next-token-agreement metric | 5.7, 7 |

---

## Recorded follow-up debt

- `JacobianTransport` cannot be end-to-end tested until doc chain 021 produces an artifact; until
  then it is unit-tested against a synthetic `J` and the substitution is proven structurally.
- Caching policy for synthesised token directions is deliberately simple (working-set only) and
  should be revisited once probe workloads are real.
