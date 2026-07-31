# Feature PRD: J-Space Readout Substrate & Wire Format

**Document ID:** 022_FPRD|JSpace_Readout_Substrate
**Version:** 1.0
**Status:** Planned
**Related:** 000_PPRD §3.23 (row 23) · PADR IDL-40, IDL-41, IDL-42, IDL-45 · BRD-MIS-JSPACE-001 v0.3 BR-005..009, BR-029, BR-032 · consumed by Feature 024 (viewer, doc chain 023) · substrate for Features 025–029

---

## 1. Overview

### 1.1 Purpose

Give miStudio a training-free way to ask *"what is this model poised to say at this layer and
position?"* — cheaply enough to be routine, and in a transport a second implementation already
speaks.

### 1.2 The problem this closes

Every interpretive question in the workbench currently routes through dictionary training. An SAE
must exist before a feature can be named, and dictionary training is expensive enough that it gates
experimentation. There is no cheap readout of what a model is *about* to be able to say.

The Jacobian lens supplies one: a per-layer `d_model × d_model` map from which any vocabulary
token's residual-stream direction can be synthesised on demand. Its degenerate case — `J = I`, the
logit lens — needs **no artifact at all**, which makes it the correct first delivery. Everything
built above it (viewer, decomposition, interventions) substitutes the Jacobian later at a single
call site.

### 1.3 Scope boundary

This feature delivers the **substrate and its transport**. It does not deliver the viewer
(Feature 024, doc chain 023), the artifact fitting pipeline (Feature 022 in PPRD terms — doc chain
021), the dictionary annotation, or any intervention.

---

## 2. User Stories

- As an **interpretability researcher**, I want a ranked readout at a chosen layer and position so
  I can see what content is present without training a dictionary first.
- As an **alignment auditor**, I want to score an activation against one named concept without
  ranking the whole vocabulary, so per-token monitoring is affordable.
- As a **researcher comparing methods**, I want a discrete inventory of concurrently active
  concepts, not a top-k list that over-reports correlated directions.
- As **any consumer of the stream**, I want the same wire format Neuronpedia emits, so one viewer
  serves both and disagreement between implementations is diagnostic rather than mysterious.
- As a **reviewer**, I want every derived figure to name the solver parameters and control seed that
  produced it, because the decomposition is not unique.

---

## 3. Functional Requirements

### 3.1 Logit-lens path first, no artifact (BR-005)

The system SHALL implement `J = I` as a complete, shippable readout requiring no precomputed
artifact, and SHALL expose it through the same interface the Jacobian will use. Substituting the
Jacobian SHALL change exactly one call site and no consumer.

### 3.2 Model-agnostic structure resolution (BR-032, IDL-41)

Layer enumeration and residual access SHALL resolve through
`ml/layer_discovery.discover_transformer_structure`. The readout path SHALL contain no architecture
whitelist, no model-name branch, and no dependency on an upstream fitter's layout detection.

The residual stream SHALL be captured at the **decoder-layer output**
(`structure.layers_module[L]`), never at a discovered normalisation module.

### 3.3 Three readout modes, not interchangeable (BR-008)

| mode | output | used for |
|---|---|---|
| **Full ranked** | `softmax(W_U · norm(J·h))` sorted over the vocabulary | the position × layer grid, top-token lists |
| **Probe** | scalar score of one named direction against an activation | threshold detection, runtime monitoring |
| **Sparse decomposition** | sparse non-negative combination of ≤ k lens directions by gradient pursuit | occupancy, variance splits, concept inventory |

Occupancy, J-space/non-J-space splits and excess-FVE figures SHALL be computed from **sparse
decomposition only**. Top-k by inner product SHALL NOT be substituted — on an overcomplete
non-orthogonal frame it returns a different, more redundant active set.

Where probe scores and full-ranking positions disagree because of the data-dependent normalisation
factor, the canonical mode SHALL be recorded per analysis.

### 3.4 Storage discipline (BR-006, IDL-42)

The artifact SHALL be one `d_model × d_model` matrix per analysed layer. No code path SHALL
allocate an `n_vocab × d_model` array derived from `J`. Token directions are synthesised on demand
and cached by working set.

An envelope check SHALL fail CI when artifact size exceeds a configured multiple of
`d_model² · sizeof(dtype) · n_layers`, **computed from the model's own config** — never a constant.

### 3.5 Recipe provenance (BR-007)

Every artifact SHALL record enough to rebuild it: target layer (final vs penultimate), attention-
gradient treatment, target-position scope, aggregation estimator and outlier thresholds, corpus
identity and sampling, sequence count and length, convergence criterion, and library/commit
versions. Emitted artifacts SHALL be **fp16**.

Where a recipe choice is **inapplicable to a layer** — frozen-Q/K on a convolutional layer — it
SHALL be recorded per layer as inapplicable, never as a value and never silently omitted (BR-032).

### 3.6 Decomposition provenance (BR-009)

Sparse decomposition SHALL record sparsity level `k`, solver identity and parameters, iteration
count, convergence criterion, and the seed and construction of any random-direction control, on
every derived figure. Occupancy and excess-FVE figures without a recorded control seed SHALL be
treated as **invalid**.

### 3.7 Wire format conformance (BR-029, IDL-45)

The readout stream SHALL mirror Neuronpedia's lens stream:

```
meta  = { model, types, layers_by_type, top_n, prompt_len }
token = { position, token, id, is_generated, results: slice[] }
slice = { type, top_tokens[layer][k], top_probs[layer][k] }
```

`top_tokens` entries SHALL be **decoded strings**, not token ids. Terminal `done` / `error` messages
SHALL be emitted. A miStudio stream and a Neuronpedia stream SHALL be interchangeable at the client
with no adaptation layer.

### 3.8 Next-token agreement is not a quality metric (BR-004)

No dashboard, gate, acceptance test or report in this feature SHALL reward next-token agreement.
The Jacobian lens is deliberately worse on that measure than the logit lens through most of the
network; a check that rewards it is a defect.

---

## 4. User Interface

None. This feature is substrate and transport; the surface is Feature 024 (doc chain 023).

---

## 5. API / Integration

- `POST /api/v1/jlens/readout` — prompt + model + lens types + layer selection + top-n → the §3.7
  stream.
- `POST /api/v1/jlens/probe` — activation reference + named directions → scores.
- `POST /api/v1/jlens/decompose` — activation reference + `k` + solver params → active set with
  provenance.

Registered in `api/v1/router.py`. MCP exposure is Feature 029.

---

## 6. Data / Types

- `JLensArtifact` — per-layer matrices, recipe provenance, per-layer applicability map, envelope
  record.
- `ReadoutRequest` / `LensMetaMessage` / `LensTokenMessage` / `LensTypeSlice` — the §3.7 shapes,
  shared verbatim with the frontend client.
- `DecompositionResult` — active set, coefficients, residual, and mandatory solver/control
  provenance.

---

## 7. Dependencies

- `ml/layer_discovery.discover_transformer_structure` — structure resolution.
- `ml/forward_hooks` — residual capture.
- `services/analysis_service.load_unembedding_matrix` / `resolve_snapshot_dir` — `W_U` without
  instantiating the model, and the CPU-only discipline that fix established.
- Existing provenance/DB layer and background job runner.

---

## 8. Success Criteria

- A readout returns for **two architecturally different models** (a hybrid conv/attention model and
  a dense transformer) with no code change between them.
- The logit-lens path produces a well-formed stream with **no artifact present**.
- Envelope check fails CI on a deliberately materialised dictionary.
- The viewer renders identically when driven by a miStudio stream and a Neuronpedia stream for the
  same model and prompt.
- Every occupancy figure carries a control seed; figures without one are rejected by validation, not
  by convention.
- Readout runs on CPU and does not allocate GPU memory.

---

## 9. Non-Goals

- Fitting a Jacobian artifact (doc chain 021).
- The viewer (doc chain 023).
- Any intervention, annotation, or watchlist.
- Materialising `W_U J` under any circumstance.
- Next-token-agreement optimisation.

---

## 10. Testing Requirements

- **Two-architecture test** — the readout runs against a hybrid and a dense model; a suite that
  only ever sees one architecture is how the old `SUPPORTED_ARCHITECTURES` whitelist survived.
- **Hook negative control** — capture at the normalisation module instead of the layer output and
  assert the readout degrades measurably. Without it, a wrong-hook readout looks plausible.
- **Envelope guard** — assert no `n_vocab × d_model` allocation, bound derived from model config.
- **Wire-format conformance** — `top_tokens` are strings; `layers_by_type` matches slice arity;
  terminal messages present.
- **Provenance rejection** — a decomposition result missing its control seed fails validation.
- **CPU-only assertion** — no `.cuda()` / `device="cuda"` on the readout path.

---

## 11. Traceability

| Source | Covered by |
|---|---|
| PPRD §3.23 row 23 | §1–§9 |
| PADR IDL-40 (logit-first, single substitution point) | §3.1 |
| PADR IDL-41 (model-agnostic discovery; decoder-layer capture; per-layer applicability) | §3.2, §3.5 |
| PADR IDL-42 (synthesise on demand; model-derived envelope) | §3.4 |
| PADR IDL-45 (adopt upstream wire format) | §3.7 |
| BR-004 (agreement is not a metric) | §3.8 |
| BR-005 (logit lens first, no artifact) | §3.1 |
| BR-006 (never materialise) | §3.4 |
| BR-007 (recipe provenance, fp16) | §3.5 |
| BR-008 (three modes, decomposition-only figures) | §3.3 |
| BR-009 (decomposition provenance, invalid without seed) | §3.6 |
| BR-029 (wire format) | §3.7 |
| BR-032 (model-agnostic; per-layer applicability) | §3.2, §3.5 |
