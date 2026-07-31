# Technical Implementation Document: J-Space Fitting, Validation & Gate

**Document ID:** 021_FTID|JSpace_Fitting_And_Gate
**Version:** 1.0
**Status:** Planned
**Related:** 021_FPRD · 021_FTDD

---

## 1. Implementation order

1. **ORM + migration** — artifact, validation result, band report, gate decision.
2. **`jlens_fitter.py`** — discovery, capture, backward pass, shard/merge. The riskiest piece; build
   it first and put the hook-target negative control on it immediately.
3. **`jlens_validation.py`** — the six classes, each independently runnable and independently failing.
4. **`jlens_artifact_service.py`** — lifecycle, publish only after validation passes.
5. **`jlens_metrics.py` + `jlens_band_report.py`** — metrics, null controls, boundary derivation, gate.
6. **Endpoints**, and bind the readout's Jacobian path.
7. **MCP tools**, registered, with the reachability test written *before* the tools.
8. **UI** — artifacts surface; Jacobian/Diff light up through `meta.types` with no panel change.

## 2. Pitfalls

Each of these produces an artifact or a report that *looks* right.

1. **Hook `structure.layers_module[L]`, never `residual_norm_module`.** The discovered `"residual"`
   module on the reference model is a post-attention RMSNorm. This is the single most expensive
   lesson in this repository and it was learned on this exact model. Symptom: no error at all.

2. **`None`, not `False`, for inapplicable.** 10 of the reference model's 16 layers are conv;
   `frozen_qk_applicable=False` there gets averaged and reads as "we checked and it did not apply",
   which is a different claim from "undefined here".

3. **Never form `W_U J`.** Apply `J` then `W_U`. Assert no `n_vocab × d_model` allocation on the
   path — the same shape as the existing memory test.

4. **Derive the envelope from the model's config**, never from a constant. The ratio is ~32× on the
   reference model and ~111× on a 256k-vocab model; a constant tuned on one passes while missing a
   real materialisation on the other.

5. **Do not converge on a readout-quality proxy.** Any such proxy drifts toward next-token
   agreement, which BR-004 forbids as a quality metric. Converge on the change in accumulated `J`.

6. **Agreement may be REPORTED, never SCORED.** The band report includes it as a layer profile. The
   gate, CI and any dashboard must not rank on it. Write the test that fails if it appears in a
   scoring path.

7. **ROUND-TRIP must actually mount and serve.** A test that constructs the artifact in-process and
   reads it back proves nothing about the consumer, whose loader fails at request time without
   raising.

8. **Vendor at a recorded commit.** The reference repository is unmaintained and not accepting
   contributions; an unpinned dependency here cannot be fixed upstream.

9. **Weight identity is part of acquisition.** An instruction-tuned variant is not the base model.
   Checking only the model *name* adopts a lens fitted for different weights, silently.

10. **Two architectures in the suite, always.** One hybrid, one dense. A suite that only ever sees
    one architecture is exactly how the old whitelist survived.

11. **Vary the dimension under test in every fixture.** Two mutation controls survived in doc chain
    023 because fixtures agreed by construction — one axis shared across lens types, an evenly
    spaced axis where two positioning modes coincide. Build fixtures that can disagree.

12. **Write the reachability test before the MCP tools.** Sixteen tools once shipped implemented,
    unit-tested and documented while registered nowhere, and every test passed by importing the
    module directly.

## 3. Testing

**Hook target** — fit at `residual_norm_module`, assert measurable degradation. Negative control, not
a comment.

**Model-agnosticism** — `grep -riE "lfm2|gemma|llama|granite|qwen"` over the new modules returns
nothing outside comments; both architectures produce well-formed artifacts.

**Envelope** — asserted against `d_model² · 2 · n_layers` from each model's own config; no
`n_vocab × d_model` allocation on fit or readout.

**Validation** — each class fails against a violation: a corrupted tensor (STRUCTURAL), two lens
files in one directory (NAMING), a truncated artifact (ENVELOPE), a shuffled `J` (SEMANTIC), a
divergent reader (CROSS-IMPLEMENTATION), an unmounted artifact (ROUND-TRIP).

**BR-004** — a test that fails if next-token agreement is referenced in a scoring, ranking or gating
path.

**Gate** — `NO_GO` is constructible, storable, renderable and exportable end to end.

**MCP** — tools present in the live registry; payload and call count asserted; deleting the
registration turns the suite red.

**Mutation controls to run** (each must go red): hook the norm module; return `False` for an
inapplicable layer; materialise `W_U J`; hardcode the envelope bound; score the gate on agreement;
skip ROUND-TRIP; publish before validation; drop an MCP registration; accept an artifact whose
weight identity differs.
