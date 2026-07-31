# Neuronpedia J-Space Conformance Specification

> **Status:** Phase 0 deliverable satisfying `BRD-MIS-JSPACE-001` BR-022 (conformance assessment before
> contract design commits). **Verified against source**, not inferred: `anthropics/jacobian-lens` @ HEAD,
> `hijohnnylin/neuronpedia` @ HEAD (webapp, inference server, Prisma schema, migration
> `20260619032021_add_jlens`, `utils/neuronpedia-utils/neuronpedia_utils/jlens/`), and the Neuronpedia
> release post of 2026-07-10.
>
> **Headline finding — the premise needs correcting.** There is no upload path for J-lens data into
> Neuronpedia, local or hosted. J-lens is **compute-on-demand from a mounted artifact**, not stored
> per-feature data. Nothing about the SAE upload model (model → source → feature index, package push, job
> polling) applies. The integration surface for a local instance is one environment variable pointing at
> one directory. This makes the work *smaller* than the BRD assumed, but on a different axis than planned,
> and it splits into two independent tracks that share nothing but a name.
>
> **Amendments this document forces on `BRD-MIS-JSPACE-001`** are collected in §7. Four of them are
> material; one materially reduces the increment's headline risk.

---

## 1. The corrected integration model

Neuronpedia treats J-lens and SAE features as fundamentally different kinds of object.

| | SAE features (existing miStudio path) | J-lens (this work) |
|---|---|---|
| Unit | Feature, keyed `model → source → index` | Per-layer matrix `J_ℓ`, keyed by model |
| Storage | Postgres rows + dashboard payloads | A single `.pt` file on disk |
| Ingestion | Package build → API push → async job | Mount a directory; server reads it at boot |
| Persisted per feature | Yes — activations, explanations, dashboards | **Nothing** |
| Computed | Offline, uploaded | **Live, per request, on the inference server** |
| DB tables | `Feature`, `Explanation`, `Source`, … | `JlensShare`, `JlensSharePutRequest` — *shared analysis sessions only* |

The entire J-lens footprint in Neuronpedia's database is two tables that persist **shared links**: UI
restore state (locked tokens, selected positions, active lens-mode tab, top-N, temperature) plus a URL to
a gzipped JSON blob of the token stream. Neither table stores lens data, feature data, or readouts as
first-class records. Confirmed in `migration.sql` and the `JlensShare` / `JlensSharePutRequest` Prisma
models — there is no third table.

**Consequence.** "Upload J-lens results from miStudio to our local Neuronpedia" is not a thing the system
does, and building it would mean building a Neuronpedia feature that does not exist. What miStudio can do
is *supply the lens artifact that a local instance serves from* — which is Track A below, and is roughly a
day of work rather than an export-pipeline project.

### Two tracks

- **Track A — J-lens artifact supply.** miStudio produces a conformant lens artifact; the local
  Neuronpedia inference server mounts it. No API, no DB, no upload. §2–§4.
- **Track B — SAE workspace annotation.** The per-feature workspace classification from BR-012 *is*
  feature-shaped data and *does* go through the existing upload path miStudio already implements (FPRD
  007). This is where "Neuronpedia-compliant upload" is the right frame. §5.

These are independent. Track A can ship without Track B and vice versa.

---

## 2. Track A — artifact format (normative)

### 2.1 On-disk layout

A fit run emits a directory:

```
<exports-dir>/<np_model_id>/jlens/<dataset>/
├── <slug>_jacobian_lens.pt        # the artifact the server loads
├── <slug>_checkpoint.pt           # resumable fit state (not served)
├── <slug>_convergence.csv         # per-prompt convergence curve (not served)
└── config.yaml                    # full reproducibility record
```

`<slug>` is derived from the **HuggingFace** model id, not the Neuronpedia one, by taking the segment after
the last `/`, replacing every run of characters outside `[0-9A-Za-z._-]` with a single `-`, and stripping
leading/trailing `-`. The inference server reimplements this function to reconstruct remote paths, so
miStudio must reproduce it exactly. For `google/gemma-2-2b` the slug is `gemma-2-2b` and the filename is
`gemma-2-2b_jacobian_lens.pt`.

The `*_jacobian_lens.pt` **suffix is load-bearing**: local resolution globs `*_jacobian_lens.pt` in the
mounted directory. Multiple matches log a warning and take the lexicographically first — so one lens per
directory, always.

### 2.2 Checkpoint contents

A plain `torch.save` dict, loaded with `weights_only=True`:

```python
{
    "J":             {layer_index: Tensor[d_model, d_model]},   # required
    "source_layers": [int, ...],                                 # optional, defaults to []
    "n_prompts":     int,                                        # optional, defaults to 0
    "d_model":       int,                                        # REQUIRED — no default
}
```

Absence of `"J"` raises with the offending key list. `d_model` is the only other key read without a
fallback. Layer keys are coerced with `int()`, so string keys survive, but emit integers.

**dtype: fp16.** The reference implementation defaults to fp16 on save, explicitly reasoning that entries
are O(1) so the range is not a constraint and fp16's extra mantissa bits beat bf16 here. The server casts
to fp32 on load and keeps Jacobians on CPU, moving them to the compute device lazily per layer.
`BRD-MIS-JSPACE-001` Appendix A.1 says bf16 — see §7, amendment A2.

**Storage discipline is confirmed by both codebases.** Neither the reference implementation nor
Neuronpedia's server ever materializes `W_U J_ℓ`. The server's transport step is `residual @ J_bar.T` —
one matmul per layer, with the model's own unembedding applied afterwards. BR-006 is correct and is now
externally validated rather than merely reasoned.

### 2.3 `config.yaml`

Written by the fitter as a comment header (GPU inventory, exact command line, attribution) plus a nested
body recording dataset identity, config, split, text field, max chars, prompt count, sequence length,
dtype, device map, and convergence settings. Its stated purpose is that the lens be *fully reproducible
from the file alone* — the same standard BR-007 sets for miStudio provenance. Emit it, and treat any
field miStudio records that Neuronpedia's schema lacks as an additive extension under a namespaced key
rather than a renamed core key.

---

## 3. Track A — the local instance wiring

The inference server resolves the lens at startup in this order:

1. **`JLENS_SOURCE`** — a local override *directory*. If set, the server globs it for
   `*_jacobian_lens.pt` and loads the first match. **This is the miStudio integration point.**
2. Otherwise, download from `JLENS_HF_REPO` (default `neuronpedia/jacobian-lens`) at
   `<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt`, cached under `JLENS_CACHE_DIR`
   (default `/tmp/neuronpedia-jlens-cache`).

Relevant environment / CLI surface: `JLENS_SKIP`, `JLENS_SOURCE`, `JLENS_DATASET`, `JLENS_HF_REPO`,
`JLENS_HF_PATH`, `JLENS_CACHE_DIR`, plus a separate token limit for lens endpoints independent of the
general `--token_limit`.

Model identity resolves through `np_model_to_hf.json` at the workspace root — a flat
`{np_model_id: hf_model_id}` map the operator copies into place. An explicit `--NEURONPEDIA_MODEL_ID`
always wins.

**Two operational properties worth designing around:**

- **Loading is best-effort.** A failed lens load never crashes startup; it just makes `JACOBIAN_LENS`
  requests return an error. A malformed artifact therefore fails *quietly*, at request time, in the
  webapp — not at deploy time. miStudio must validate before handing over (§6), because Neuronpedia will
  not.
- **`LOGIT_LENS` needs no artifact at all.** The webapp's lens-mode tabs are
  `JACOBIAN_LENS | LOGIT_LENS | DIFF`. So a local instance renders logit-lens readouts with zero miStudio
  involvement, and the `DIFF` tab gives a free visual regression check on any lens you supply. This
  independently corroborates the BRD's logit-lens-first sequencing (BR-005) and gives Phase 0 a working
  comparison surface on day one.

### Minimum viable integration

```bash
# miStudio writes:  /srv/mistudio/jlens-exports/gemma-2-2b/jlens/wikitext/
#                     gemma-2-2b_jacobian_lens.pt
#                     config.yaml
# Local Neuronpedia inference server:
JLENS_SOURCE=/srv/mistudio/jlens-exports/gemma-2-2b/jlens/wikitext
```

That is the whole of Track A's runtime contract. It should be a mounted volume in the existing K8s
manifest for the local Neuronpedia deployment, not a push job.

---

## 4. Track A — what miStudio should actually do

**Do not write a fitter first.** Three sources of pre-fitted lenses exist, in preference order:

1. **`neuronpedia/jacobian-lens` on HuggingFace — 36 pre-fitted models**, largest Llama 70B, with
   Neuronpedia actively soliciting contributions for models they lack. `np_model_to_hf.json` maps
   `gemma-2-2b → google/gemma-2-2b`, so miStudio's reference model is very likely already covered.
2. **`anthropics/jacobian-lens`** — the reference fitter, `jlens.fit(...)` plus `lens.save(...)`, with
   `JacobianLens.merge()` for combining disjoint prompt slices. That last one matters on a single 3090:
   fitting parallelizes by splitting the corpus, not by splitting the model.
3. **Neuronpedia's own batch fitter** (`run-all-fit-lens.py` + `fit_lens.py`) — wraps the reference
   implementation and produces the exact directory layout above, including `config.yaml`. If you are
   fitting anything, fit with this, because layout conformance is then free.

So the realistic Track A sequence is: check the HF repo for `gemma-2-2b`; if present, download, validate
(§6), mount, done. Fit only for models the repo lacks — and then contribute the result upstream, which is
an open invitation and a cheap way to establish the collaboration BR-022 anticipates.

**Reference fit parameters** (Neuronpedia's production defaults, all recorded in `config.yaml`):
wikitext-103-raw-v1 train split, 1000 prompts, max 128 tokens, max 2000 chars, bfloat16 compute, CUDA,
convergence stop at delta 2e-3 with a 10-prompt window and reporting levels 1e-2/5e-3/1e-3, and a
**minimum of 100 prompts**.

That minimum contradicts guidance I gave earlier and which the BRD encodes — see §7, amendment A3.

---

## 5. Track B — SAE workspace annotation (the real upload path)

This is where miStudio's existing Neuronpedia export machinery applies unchanged. The BR-012 annotation —
lens kurtosis `κ`, motor/workspace/outside classification, and the BR-013 label-disagreement flag — is
per-feature data keyed exactly as Neuronpedia keys features.

Two placement options, in preference order:

1. **Explanation-adjacent metadata**, carried alongside the existing per-feature explanation records. Best
   fit if the annotation is to be visible and filterable in the feature UI.
2. **Dashboard payload extension.** miStudio's export dashboard table already carries a
   `logit_lens_data` JSONB column. A sibling J-space payload is the lowest-friction extension and needs no
   schema negotiation with upstream.

Confirm which against the running local instance's schema before building; the webapp's feature surface
has moved since miStudio's export was written, and this is a five-minute check that prevents a rebuild.

**This is the genuine upstream contribution opportunity.** Neuronpedia hosts J-lens readouts and hosts SAE
features, but nothing currently joins them — nobody has published, at dictionary scale, which learned
features live in the reportable workspace. miStudio holds labeled dictionaries and would be first. Offer
it as a schema proposal rather than a local fork, per BR-022.

---

## 6. Acceptance tests

miStudio must validate before handover, because the server's best-effort loading will not.

**A1 — Structural.** `torch.load(path, weights_only=True)` succeeds; `"J"` present; `"d_model"` present and
integer; every value in `"J"` is a square 2-D tensor of side `d_model`; layer keys coerce to `int`;
`source_layers`, if present, equals the sorted key set.

**A2 — Naming.** Filename matches `<slug>_jacobian_lens.pt` where `<slug>` is the slug function applied to
the **HF** model id. Exactly one `*_jacobian_lens.pt` in the mounted directory.

**A3 — Envelope.** Artifact size within tolerance of `n_layers × d_model² × 2 bytes`. For gemma-2-2b:
~10.6 MB per layer, ~276 MB for 26 layers. An artifact one to two orders larger means someone materialized
the token dictionary — this is BR-006's CI guard and the single most likely implementation error.

**A4 — Semantic.** Load the artifact in miStudio, run a readout at a mid-band layer on a fixture prompt
with a known unspoken intermediate (the reference implementation ships evaluation sets for multihop,
multilingual, order-of-operations, poetry, typo, and association), and assert the intermediate appears in
the top-k. Catches a structurally valid but wrong-model or wrong-layer-indexing artifact.

**A5 — Cross-implementation.** Same prompt, same layer, same top-k, compared between miStudio's readout
and the local Neuronpedia webapp's `JACOBIAN_LENS` tab. Then flip to `LOGIT_LENS` and confirm both
implementations agree there too — the logit lens needs no artifact, so disagreement isolates a readout
bug from an artifact bug.

**A6 — Round trip.** Mount, boot the local inference server, issue a `JACOBIAN_LENS` request, confirm no
error and non-empty readout. Because load failure is silent at startup, this must be an explicit test,
not an assumption from a clean boot log.

A5 is the highest-value test in the set: it is a free, independent implementation to check against, and
Phase 0's replication work (BR-001) gets substantially cheaper by leaning on it.

---

## 7. Amendments required to `BRD-MIS-JSPACE-001`

**A1 — BR-022 splits in two. (Material.)** Current text assumes one conformance surface. Replace with:
(a) J-lens artifact conformance — file layout, checkpoint schema, `config.yaml`, `JLENS_SOURCE` mount; no
upload path exists and none should be built; (b) SAE workspace annotation conformance via the existing
feature/explanation upload path. Add the acceptance tests of §6. Add an explicit non-goal: *miStudio shall
not build a J-lens ingestion API against Neuronpedia*.

**A2 — Appendix A.1 dtype correction.** Says bf16; the reference implementation saves fp16 with stated
reasoning, and Neuronpedia's loader casts to fp32 on load. Size arithmetic is unchanged (both 2 bytes) but
emitted artifacts must be fp16 for byte-level conformance.

**A3 — Appendix A.2 corpus-size correction. (Material.)** The BRD says to plan at the paper's §A.7 floor of
~10 sequences and warns against budgeting from the 1000-prompt Methods figure. Both implementations
disagree: the reference README says ~100 prompts is usable, and Neuronpedia's production fitter defaults to
1000 with a hard minimum of 100 and convergence-based early stopping at delta 2e-3. **Correct guidance:
convergence-based stopping with a floor of 100 prompts, not a fixed 10.** The ten-prompt figure is the
paper's demonstration that quality saturates early, not an operating recommendation, and I over-read it.

**A4 — RSK-001 downgrade. (Material, and the good news.)** The BRD rates the scale risk high-impact /
medium-likelihood on the basis that the paper studied only large production models. Neuronpedia's model map
ships lens support down to **pythia-70m-deduped, gpt2-small, and gemma-3-270m**, and the release post
reports J-lens working across 12 hosted models spanning the Gemma, Llama, GPT-OSS, and Qwen families, with
36 pre-fitted. The *readout* is demonstrably viable far below miStudio's reference scale. Recommended
revision: downgrade likelihood to low, and **narrow the risk statement** — what remains uncertain at 2B is
not whether the lens produces usable readouts but whether the full workspace *claim set* (selectivity,
flexible generalization, capacity structure) replicates. That is a finding to characterize in Phase 0, not
a gate that can strand the increment.

Retain one specific caveat from the release post: on smaller models, coordinate swaps oversteer easily and
required selecting **fewer layers** to land the intended result. Fold this into BR-017 as a small-model
default rather than leaving band presets uniform across scales.

**A5 — BR-005 corroborated, no change.** Neuronpedia's own UI ships `LOGIT_LENS` as a peer tab requiring no
artifact, with a `DIFF` mode against `JACOBIAN_LENS`. The logit-lens-first sequencing is now the
independently-arrived-at industry pattern, and `DIFF` is a free regression surface.

**A6 — Phase 0 scope reduction.** BR-001 assumes vendoring an implementation and standing up an evaluation
harness. Both exist: `anthropics/jacobian-lens` ships the fitter, the applier, the slice visualization, and
the six evaluation sets plus eleven experiment fixtures reproducing the paper's headline results. Phase 0
becomes *run and verify* rather than *build*. Note the repo is explicitly unmaintained and not accepting
contributions — vendor at a recorded commit and expect no upstream fixes, which is RSK-002's mitigation
holding rather than failing.

---

## 8. Recommended sequence

1. Check `neuronpedia/jacobian-lens` on HuggingFace for `gemma-2-2b`. If present, Track A collapses to
   download → validate → mount, and no fitting is needed.
2. Stand up the local Neuronpedia instance with `LOGIT_LENS` working first — zero miStudio dependency,
   and it validates the whole serving path before any artifact exists.
3. Implement the §6 acceptance suite in miStudio as the artifact validator. This is the reusable piece:
   it guards every future lens, whether downloaded, fitted, or contributed.
4. Mount via `JLENS_SOURCE`, run A6, then A5 against miStudio's own readout — which is simultaneously the
   cheapest available execution of BR-001's replication.
5. Fit only for uncovered models, using Neuronpedia's fitter so layout conformance is free. Contribute
   results upstream.
6. Track B separately, after the annotation exists, against the running instance's actual feature schema.
7. Apply the §7 amendments to the BRD before it enters `002_create-project-prd.md`, so the PPRD absorbs
   the corrected scope rather than the assumed one.
