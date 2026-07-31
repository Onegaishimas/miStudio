# Technical Implementation Document: J-Space Readout Substrate & Wire Format

**Document ID:** 022_FTID|JSpace_Readout_Substrate
**Version:** 1.0
**Status:** Planned
**Related:** 022_FPRD · 022_FTDD · PADR IDL-40..42, IDL-45

---

## 1. Implementation Order

1. **Schemas first** (`schemas/jlens.py`). The wire format gates the frontend, and getting
   `top_tokens` typed as `List[List[str]]` up front prevents the ids-vs-strings divergence.
2. **`W_U` loading + CPU discipline** — thin wrapper over the existing
   `analysis_service.load_unembedding_matrix`; no new loader.
3. **Structure resolution + applicability map** — `discover_transformer_structure`, then classify
   each layer.
4. **Residual capture** — hooks on `layers_module[L]`. Write the negative-control test *here*, not
   after; it is the check that makes step 3 meaningful.
5. **`LensTransport` + `IdentityTransport`** — the logit path end to end.
6. **Full ranked readout → stream assembly** — first thing that produces a `meta`/`token` sequence.
7. **Probe**, then **sparse decomposition** with mandatory provenance.
8. **Envelope guard** — model-derived bound.
9. **Endpoints + router registration.**
10. **`JacobianTransport`** last — it is the substitution proof, and it should be the only diff when
    an artifact arrives.

---

## 2. File-by-file

| file | contents |
|---|---|
| `backend/src/schemas/jlens.py` | `LensMetaMessage`, `LensTokenMessage`, `LensTypeSlice`, `ReadoutRequest`, `DecompositionResult`, `JLensArtifact`, `LayerApplicability` |
| `backend/src/services/jlens_readout_service.py` | `LensTransport` / `IdentityTransport` / `JacobianTransport`, `ReadoutService` |
| `backend/src/api/v1/endpoints/jlens.py` | `/readout`, `/probe`, `/decompose` |
| `backend/src/api/v1/router.py` | register with prefix `/jlens` |
| `backend/tests/unit/test_jlens_readout.py` | modes, wire conformance, provenance rejection |
| `backend/tests/unit/test_jlens_model_agnostic.py` | two-architecture + hook negative control + no-architecture-name guard |
| `backend/tests/unit/test_jlens_envelope.py` | model-derived envelope, no `n_vocab × d_model` allocation |

---

## 3. Pitfalls

Each of these has already cost this project time, in this repo, on this hardware.

1. **`residual_norm_module` is not the residual stream.** On LFM2 it is a post-attention RMSNorm.
   Hooking it produced byte-identical steered/unsteered output in the steering work — a *silent*
   failure. Hook `structure.layers_module[L]`. See `steering_core.py:230`.

2. **Do not instantiate the model to get `W_U`.** The logit-lens path previously loaded all 8B
   parameters to read one tensor and then OOM'd the moment the GPU was busy. Use
   `load_unembedding_matrix` — it reads `lm_head.weight` from the shard and falls back to tied
   `embed_tokens`.

3. **Do not put the readout on CUDA.** It is ~0.4 GFLOP. Running it on the GPU is how it broke
   before: it competed with serving for VRAM and failed at request time.

4. **The envelope constant is not portable.** LFM2's ratio is ~32×, gemma's ~111×, because the
   vocabularies differ fourfold. A guard hardcoded to either passes on the other. Derive from
   `d_model`, `n_vocab`, `n_layers` at runtime.

5. **`top_tokens` are strings.** Emitting ids type-checks fine and renders as unreadable cells.

6. **`layers_by_type` drives the client's layer axis.** The reference panel hardcodes 21 layers at
   0,5,…,100; LFM2 has 16. Anything that assumes a count or spacing is wrong for the next model.

7. **Sparse decomposition ≠ top-k.** They give different active sets on a correlated overcomplete
   frame. Occupancy computed from top-k is inflated and will not match published figures.

8. **Frozen-Q/K is undefined on conv layers.** Recording it as `false` is wrong — it is
   *inapplicable*. `false` averages; absent does not.

9. **A missing control seed invalidates a figure.** Not a lint warning — validation must reject it,
   because the decomposition is non-unique and the figure is unreproducible without it.

10. **Do not add a next-token-agreement metric anywhere**, however natural it looks as a sanity
    check. BR-004 makes it a defect: the lens is deliberately worse on it than the logit lens.

---

## 4. Testing

**Model-agnosticism** — run the readout against a hybrid (LFM2) and a dense (granite) model and
assert both produce well-formed streams. Guard test: `grep -riE "lfm2|gemma|llama|granite"` over
`jlens_readout_service.py` returns nothing outside comments. A single-architecture suite is exactly
how the removed whitelist survived as long as it did.

**Hook negative control** — capture at `residual_norm_module` instead of `layers_module[L]` and
assert the readout degrades measurably. This is the mutation that proves pitfall 1 is guarded.

**Envelope** — assert the bound is computed from model config (change `n_vocab` in a fixture and the
bound must move), and that no `n_vocab × d_model` array is allocated.

**Wire conformance** — `top_tokens[layer][k]` are `str`; `len(layers_by_type[type])` equals the slice
arity; `done` terminates; `error` carries a reason.

**Provenance** — a `DecompositionResult` without a control seed is rejected by validation.

**Mutation controls to run** (each must go red):
- hook the norm module instead of the layer output
- hardcode the envelope bound to a constant
- emit ids instead of decoded strings
- substitute top-k for pursuit in occupancy
- record `frozen_qk=false` on a conv layer instead of inapplicable
- accept a decomposition result with no control seed
- move the readout to CUDA
