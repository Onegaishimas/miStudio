# Technical Design Document: J-Space Readout Substrate & Wire Format

**Document ID:** 022_FTDD|JSpace_Readout_Substrate
**Version:** 1.0
**Status:** Planned
**Related:** 022_FPRD · PADR IDL-40..42, IDL-45 · BRD-MIS-JSPACE-001 v0.3 Appendix A.1–A.3

---

## 1. Resolving model structure without knowing the model

`discover_transformer_structure(model)` returns a `TransformerStructure` carrying `layers_path`,
`layers_module` (the actual `nn.ModuleList`), `num_layers`, and the discovered `attention_module` /
`mlp_module` / `residual_norm_module` names plus an `architecture_hint`. It already carries LFM2's
patterns (`conv`, `ffn_norm`, `operator_norm`), and it is what removed this codebase's
`SUPPORTED_ARCHITECTURES` whitelist.

The readout uses it for two things only: **how many layers there are**, and **which module to hook**.

### 1.1 The hook point is `layers_module[L]`, not `residual_norm_module`

This is the design decision most likely to be "corrected" by someone reading the field name.
`residual_norm_module` sounds like the residual stream. On LFM2 it resolves to a **post-attention
RMSNorm**, and this project has already paid for that confusion once: in steering, a vector applied
there was renormalised away and steered output was byte-identical to unsteered at every dial
(IDL-38, `steering_core.py:230`).

A readout taken at a normalisation module fails the same way and is harder to notice — it returns
plausibly-shaped numbers with the signal scaled out. So:

```
hook target := structure.layers_module[L]          # resid_post
NOT           structure.residual_norm_module
```

The FPRD requires a negative control that hooks the norm and asserts measurable degradation
(§10). Without that control this design decision is unenforced.

### 1.2 Layer kind is per-layer state

`layer_types` in a hybrid config (`['conv','conv','full_attention',...]`) means layer capability is
**not** a model-level property. The design carries a per-layer applicability map:

```
LayerApplicability = { layer: int, has_attention: bool,
                       frozen_qk_applicable: bool,
                       broadcast_metrics_applicable: bool }
```

Consumers read this map rather than assuming homogeneity. An inapplicable metric is **absent and
labelled**, never zero — a zero would be averaged and would silently understate.

---

## 2. The readout itself

Logit lens is the degenerate case `J = I`:

```
full ranked:  softmax(W_U · norm(h_ℓ))
jacobian:     softmax(W_U · norm(J_ℓ · h_ℓ))
```

One extra `d_model²` matrix-vector product distinguishes them. That is the **single substitution
point** IDL-40 requires: a `LensTransport` interface with two implementations, `IdentityTransport`
and `JacobianTransport`, selected by lens type. No consumer branches on lens type.

### 2.1 `W_U` without the model

`analysis_service.load_unembedding_matrix(model_dir, device)` reads `lm_head.weight` directly out of
the safetensors shard, falling back to the tied `model.embed_tokens.weight`. It exists because the
previous logit-lens implementation instantiated all 8B parameters to reach one tensor and then
failed outright when the GPU was occupied.

Reuse it verbatim. For the reference model `W_U` is `65536 × 2048` — about 268 MB in fp16.

### 2.2 CPU, deliberately

The readout is a matvec per (layer, position) — fractions of a GFLOP. It gains nothing from CUDA and
everything it touches is contended: the same fix that introduced `load_unembedding_matrix` also
pinned logit lens to CPU precisely because a small analysis competing with serving for VRAM took the
whole feature down.

Residual **capture** needs the model resident and is the one GPU-touching step; it is a separate,
bounded operation and its device is chosen explicitly rather than inherited.

---

## 3. Storage: why the dictionary is never materialised

The token dictionary is the rows of `W_U J_ℓ`. Precomputing it is the single most likely
implementation error, and the arithmetic is decisive but **not portable**:

| model | `J` per layer | all layers | materialised | ratio |
|---|---|---|---|---|
| LFM2.5-1.2B (d 2048, v 65k, 16L) | 8.4 MB | **134 MB** | 4.3 GB | ~32× |
| gemma-2-2b (d 2304, v 256k, 26L) | 10.6 MB | 276 MB | 30.7 GB | ~111× |

The **ratio scales with vocabulary**, so a guard written against a constant would pass on a
small-vocab model while missing a real materialisation. The envelope check therefore derives its
bound from the loaded model's `d_model`, `n_vocab` and `n_layers`.

Materialisation buys nothing: a full ranked readout is one matvec plus the model's own unembedding
call; a single token direction is `W_U[t,:] · J_ℓ`, one vector-matrix product, synthesised on demand
and cached for the small working set actually in use.

---

## 4. Three modes are three different questions

**Full ranked** and **probe** disagree by a data-dependent normalisation factor, so one is canonical
per analysis and that choice is recorded.

**Sparse decomposition** is not top-k. On an overcomplete non-orthogonal frame, gradient pursuit
returns a different and typically less redundant active set than the k largest inner products.
Occupancy, J-space/non-J-space splits and excess-FVE come from pursuit **exclusively**; using top-k
there would systematically over-report correlated directions and inflate occupancy.

Pursuit is non-unique by construction. Reproducibility is therefore a **provenance property, not a
mathematical one**: solver identity, parameters, iteration count, convergence criterion, and the
random-control seed are mandatory on every derived figure, and a figure lacking the seed is invalid
rather than merely undocumented.

---

## 5. Wire format

Adopting Neuronpedia's shape rather than inventing one removes a class of contract invention and
makes the reference panel drivable without adaptation:

```
meta  { model, types, layers_by_type, top_n, prompt_len }
token { position, token, id, is_generated, results: slice[] }
slice { type: 'JACOBIAN_LENS'|'LOGIT_LENS', top_tokens[layer][k], top_probs[layer][k] }
done | error
```

Two details are load-bearing:

- **`top_tokens` are decoded strings.** Emitting ids would be a silent divergence that only shows up
  as unreadable cells in a client expecting text.
- **`layers_by_type` is per lens type.** The layer axis is model-derived; a client must not assume a
  fixed count or spacing. The reference panel currently hardcodes 21 layers and must be driven from
  this field instead.

---

## 6. Architecture / types

```
services/jlens_readout_service.py
  LensTransport            (interface)      ← the single substitution point
  IdentityTransport        (J = I)
  JacobianTransport        (loads J_ℓ)
  ReadoutService
    .full_ranked(...)  .probe(...)  .decompose(...)
    ._capture_residuals(...)   → hooks layers_module[L]
    ._applicability(...)       → LayerApplicability[]

schemas/jlens.py
  LensMetaMessage · LensTokenMessage · LensTypeSlice
  ReadoutRequest · DecompositionResult · JLensArtifact

api/v1/endpoints/jlens.py   → /readout /probe /decompose
```

---

## 7. Risks

| risk | mitigation |
|---|---|
| Hook resolved to the norm module; readout silently meaningless | negative-control test asserting degradation when hooked at the norm (§1.1) |
| Envelope guard written against a constant; passes on small-vocab models | bound derived from the loaded model's own config (§3) |
| Top-k substituted for pursuit; occupancy inflated | pursuit-only enforcement plus a test asserting the two differ on a correlated frame |
| Architecture branch creeps back in | guard test: no architecture name in the readout module outside comments |
| Ids emitted instead of strings | conformance test on `top_tokens` element type |
| Readout allocates GPU memory and contends with serving | CPU-only assertion on the readout path |
| Hybrid metrics averaged over qualifying layers only | applicability map consumed rather than assumed; inapplicable is absent, not zero |
