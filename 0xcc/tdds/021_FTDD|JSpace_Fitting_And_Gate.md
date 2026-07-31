# Technical Design Document: J-Space Fitting, Artifact Validation & the Phase-0 Gate

**Document ID:** 021_FTDD|JSpace_Fitting_And_Gate
**Version:** 1.0
**Status:** Planned
**Related:** 021_FPRD · PADR IDL-40..46 · BRD-MIS-JSPACE-001 v0.3 Appendix A (normative)

---

## 1. What the Jacobian lens is, in one paragraph

The logit lens reads a residual `h_ℓ` by projecting it straight through the model's final norm and
unembedding — implicitly assuming the identity map carries `h_ℓ` to the final layer. The Jacobian
lens replaces that identity with `J_ℓ`, the Jacobian of the final residual with respect to `h_ℓ`,
computed by one backward pass with attention patterns and normalisation statistics **frozen**. The
readout is then `softmax(W_U · norm(J_ℓ h_ℓ))`.

`J_ℓ` is `d_model × d_model` and is the only thing stored. `W_U J_ℓ` — the materialised token
dictionary — is never formed; the readout applies `J_ℓ` first and `W_U` second, which is why the
existing readout service substitutes the transport at exactly one call site.

## 2. Module layout

```
backend/src/ml/jlens_fitter.py            fit J per layer; corpus sharding + merge
backend/src/ml/jlens_metrics.py           the seven band-report metrics + the null controls
backend/src/services/jlens_artifact_service.py   lifecycle: fit -> validate -> publish
backend/src/services/jlens_validation.py         the six BR-030 check classes
backend/src/services/jlens_band_report.py        boundaries derived per model + the gate
backend/src/workers/jlens_fit_tasks.py           Celery task
backend/src/api/v1/endpoints/jlens.py            extended: artifacts, reports; binds readout
backend/src/mcp_server/tools/jlens.py            MCP parity, registered
backend/src/models/jlens_artifact.py             ORM + migration
```

`jlens_readout_service.JacobianTransport` already exists and already casts `J` once per instance —
this feature supplies the tensors it consumes and nothing about that class changes.

## 3. Model-agnostic construction

```python
structure = discover_transformer_structure(model)
for idx in range(structure.num_layers):
    module = structure.layers_module[idx]      # resid_post — the WHOLE decoder layer
```

**Not** `structure.residual_norm_module`. On the reference model that is a post-attention RMSNorm,
and a vector applied there is renormalised away. That cost this repository a full increment and the
failure mode was byte-identical output rather than an error, so it is guarded by a negative control
(§7) rather than by a comment.

Layer kind comes from the same discovery, per layer:

```python
applicability = LayerApplicability(
    layer=idx,
    has_attention=is_attention_layer(structure, idx),
    frozen_qk_applicable=True if is_attention_layer(...) else None,   # None = INAPPLICABLE
    broadcast_metrics_applicable=True if is_attention_layer(...) else None,
)
```

`None`, never `False`. A `False` is averaged downstream and silently understates; an absent value
forces the consumer to decide. The schema in doc chain 022 already enforces this shape.

## 4. Fitting

Convergence-based stopping with a floor of **100 prompts** (Appendix A.2 — v0.1's ~10-sequence figure
was wrong and both reference implementations disagree with it).

Corpus sharding: split prompts across runs, accumulate a running mean per layer, merge. Never split
the model — a 12 GB card holds the reference model in fp16 with room for the backward pass.

Convergence is measured on the **change in the accumulated `J`**, not on a downstream readout quality
score, because §3.7 forbids scoring on agreement and any readout-quality proxy drifts toward it.

Output: `<slug>_jacobian_lens.pt` + `config.yaml` in the conformant layout, so Track A conformance is
free. `<slug>` comes from the HF id per the conformance spec's slug function.

## 5. Validation — six classes, all before handover

| class | checks | why it is not optional |
|---|---|---|
| STRUCTURAL | weights-only load; required keys; each `J` square of side `d_model`; layer keys coercible | a malformed artifact loads and reads as an empty result |
| NAMING | filename/slug; exactly one lens file per mounted directory | the loader picks silently among several |
| ENVELOPE | size within tolerance of `d_model² · 2 · n_layers` **from this model's config** | a constant bound passes on one model and misses a real materialisation on another |
| SEMANTIC | fixture prompt with a known unspoken intermediate recovers it in the top-k at a mid-band layer | structure can be perfect and content absent |
| CROSS-IMPLEMENTATION | same prompt/layer/top-k agrees with the local Neuronpedia instance, both modes | our reader and theirs can diverge while each looks fine |
| ROUND-TRIP | mount, serve, request Jacobian, confirm non-empty | **the consumer fails at request time without raising** |

ROUND-TRIP is explicit rather than assumed precisely because the failure is silence. RSK-014 exists
for this.

## 6. Band report and the gate

Seven metrics (BR-002 §3.6). Two carry null controls that are part of the metric, not a sanity extra:

- top-1 autocorrelation is measured **against a position-shuffled null**;
- fraction-of-variance-explained is reported **in excess of a size-matched random-direction control**,
  whose seed is recorded — the schema already makes `control_seed` required, because a figure without
  it cannot be reproduced or believed.

Boundaries are derived from this model's own metric profile. There is no constant to fall back to and
no default `BandReport`, which is what makes porting impossible by construction rather than by
policy.

The gate is a stored decision — `GO` / `NO_GO` / `GO_AT_LARGER_SCALE` — with its evidence attached.
`NO_GO` is a first-class value that renders and exports; a gate that cannot say no is not a gate.

## 7. Risks

| risk | mitigation |
|---|---|
| Wrong hook target; artifact plausible and inert | negative control: fit at `residual_norm_module` and assert measurable degradation |
| Architecture name creeps into the fit path | source guard over the new modules; two architectures exercised |
| Envelope bound hardcoded | bound derived from the model's own config; a second model with a different vocab in the suite |
| Agreement becomes a quality metric by drift | a test that fails if agreement appears in a scoring or gating path |
| Silent consumer failure | ROUND-TRIP against a live instance, explicit |
| MCP tools implemented but unregistered | reachability harness, registry-level, payload and call count |
| Fixtures agree by construction | every fixture varies the dimension under test — the trap that let two mutations survive in doc chain 023 |

## 8. Open questions carried forward

- **Sparsity `k`** for decomposition occupancy — `TBD (BRD open question)`. Blocks the occupancy
  metric's absolute values, not its comparison against the control.
- **Layer subsampling** is resolved at reference scale (16 layers analysed in full) and survives only
  for models larger than the subsampling target.
