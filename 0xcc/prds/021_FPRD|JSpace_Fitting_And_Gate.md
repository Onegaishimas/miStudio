# Feature PRD: J-Space Fitting, Artifact Validation & the Phase-0 Gate

**Document ID:** 021_FPRD|JSpace_Fitting_And_Gate
**Version:** 1.0
**Status:** Planned
**Related:** 000_PPRD §3.22 (row 22) · PADR IDL-40, IDL-41, IDL-42, IDL-46 · BRD-MIS-JSPACE-001 v0.3 BR-001, BR-002, BR-003, BR-004, BR-030, BR-031, BR-032 · consumed by doc chains 022 (substrate) and 023 (viewer)

---

## 1. Overview

### 1.1 Purpose

Produce a **validated J-lens artifact** for any model this workbench holds, and answer — on the
record — whether the workspace claim set replicates at the scale we run.

Three things depend on this feature and none of them can be faked:

- `POST /jlens/readout` currently returns **501**; this feature binds model resolution and artifact
  loading behind it.
- The J-Lens panel's **Jacobian and Diff modes** are disabled with a stated reason until an artifact
  exists.
- **No band shading appears anywhere in the product** until this feature emits a band report.

### 1.2 Construction is the primary path

BRD v0.3 inverted BR-031. Pre-fitted lenses exist for 36 models upstream; the reference model is not
among them, and neither is most of what this workbench runs. **Fitting is a first-class supported
path**, and acquisition is an optimisation to take when a conformant lens happens to exist for the
exact weights in use — not a dependency.

### 1.3 A NO-GO is a result, not a failure

The gate asks whether the **full workspace claim set** replicates at this scale — band structure,
selectivity, flexible generalisation, capacity limits. Whether the lens produces usable readouts at
all is settled upstream, where lens support ships for models as small as 70M parameters.

A **NO-GO is a supported terminal outcome** producing a publishable negative result. A feature that
can only conclude "yes" is not a gate.

---

## 2. User Stories

- As an **interpretability researcher**, I want to fit a J-lens for a model I hold, so the analysis
  surface is not restricted to the 36 models someone else fitted.
- As a **researcher on a hybrid model**, I want the artifact to record *per layer* what was
  computable, so I am never shown an average over whatever layers happened to qualify.
- As **anyone consuming an artifact**, I want it validated before it reaches me, because the
  downstream consumer fails **silently** — a bad artifact surfaces as an empty readout, not an error.
- As a **reviewer**, I want the Phase-0 decision recorded with its evidence, including a NO-GO.

---

## 3. Functional Requirements

### 3.1 Model-agnostic construction (BR-032)

Fitting SHALL discover model structure through `discover_transformer_structure()`. It SHALL NOT
contain an architecture name, a layer-count assumption, or a whitelist. This workbench deleted its
`SUPPORTED_ARCHITECTURES` whitelist once already; J-space SHALL NOT reintroduce one.

Residual capture SHALL hook the **whole decoder layer output** (`structure.layers_module[L]`,
resid_post). It SHALL NOT hook the discovered `residual_norm_module`: on the reference model that is
a post-attention RMSNorm, and this repository has already lost a full increment to it — a vector
applied there was renormalised away, and the failure was byte-identical output, not an error.

### 3.2 Per-layer applicability (BR-032)

Every artifact and every report SHALL record, **per layer**, what was computable: frozen-Q/K
applicability, attention-broadcast metrics, MLP gain.

Inapplicable SHALL be recorded as **absent**, never as zero or false. On the reference model
frozen-Q/K is undefined on 10 of 16 layers; an artifact SHALL NOT be described as "frozen-Q/K"
wholesale when the treatment reached a subset.

### 3.3 Fitting (BR-031)

Fitting SHALL use convergence-based stopping with a **floor of 100 prompts**. It SHALL parallelise by
splitting the corpus and merging, never by splitting the model. It SHALL emit fp16, and SHALL store
only the per-layer `d_model × d_model` matrix.

`W_U J` SHALL NEVER be materialised. The envelope check SHALL derive its bound from **this model's
own** `d_model`, `n_layers` and `n_vocab` — the required-vs-materialised ratio scales with vocabulary
(~32× at the reference model, ~111× at a 256k-vocab model), so a constant passes on one model while
missing a real materialisation on another.

### 3.4 Acquisition (BR-031)

Where a conformant pre-fitted lens exists **for the exact weights in use**, the system SHALL support
adopting it. Weight identity is part of the check: an instruction-tuned variant is not the base model
a base-model lens was fitted for, and adopting one for the other is a silent correctness failure.

### 3.5 Artifact validation before any consumer sees it (BR-030)

Every artifact SHALL pass, before handover: **STRUCTURAL** (weights-only deserialisation; required
keys; every Jacobian square of side `d_model`; layer keys coercible), **NAMING** (filename/slug
convention; exactly one lens file per mounted directory), **ENVELOPE** (size within tolerance of the
model-derived arithmetic), **SEMANTIC** (a fixture prompt with a known unspoken intermediate recovers
that intermediate in the top-k at a mid-band layer), **CROSS-IMPLEMENTATION** (same prompt, layer and
top-k agree with the local Neuronpedia instance in both modes), and **ROUND-TRIP** (mount, serve,
issue a Jacobian request, confirm a non-empty readout).

The round-trip check SHALL be explicit rather than assumed. The consumer's lens loading is
best-effort and fails at request time without raising, so an unvalidated artifact presents as a
feature that quietly returns nothing.

### 3.6 The band report (BR-002)

The system SHALL produce a per-model band report comprising: J-lens top-k next-token agreement by
layer; excess kurtosis of the readout distribution by layer; top-1 readout autocorrelation across
positions against a **position-shuffled null**; effective linear dimensionality of the lens
dictionary; cross-layer CKA of the lens geometry; sparse-decomposition occupancy; and
fraction-of-variance-explained **in excess of a size-matched random-direction control**.

It SHALL derive that model's **own** sensory / workspace / motor boundaries.

Boundaries from the source paper SHALL NOT be ported to any other model, and the product SHALL make
porting them impossible by construction. Doc chain 023 already holds this line on the UI side: there
is no band constant in the panel, and bands render only from a report.

### 3.7 Next-token agreement is never a quality metric (BR-004)

Next-token agreement with the model's own output distribution SHALL NOT be used as a quality metric
for the J-lens anywhere — not in the product, not in CI, not in a report.

The J-lens is *deliberately worse* on this measure than the logit lens through most of the network.
The directions that best predict the output are not the directions that best expose the computation
producing it. **Any gate, dashboard or acceptance test that rewards next-token agreement is a
defect**, and this feature SHALL carry a test that fails if one appears.

Agreement is still *reported* in the band report as a descriptive layer profile (§3.6). Reporting it
and scoring on it are different acts.

### 3.8 The replication report (BR-001)

The system SHALL reproduce the reference evaluation on a served model, vendored at a recorded commit,
and report per lens (logit / J-lens / tuned where available): normalised pass@k AUC for
intermediate-concept recovery across the six distributions, ablation KL divergence, and
lens-coordinate swap success rate.

Results SHALL be a first-class artifact with full provenance and SHALL be published **whether
favourable or not**.

### 3.9 The gate (BR-003)

The band report SHALL terminate in an explicit recorded **GO / NO-GO / GO-AT-LARGER-SCALE** decision,
with its evidence attached.

Product surface work beyond the logit-lens readout viewer SHALL NOT begin until that decision is
recorded.

### 3.10 MCP parity (BR-027 as broadened)

Every capability this feature creates SHALL be reachable by an agent as well as by the UI: fit,
validate, list artifacts, fetch the band report, fetch the replication report, and read the gate
decision.

Tools ship **with this feature**, not batched into a later one, and each SHALL be covered by the
reachability harness — the one that exists because 16 implemented, unit-tested and documented tools
once shipped green while registered nowhere.

---

## 4. User Interface

An **Artifacts** surface within J-Lens: fit a lens (model, corpus, recipe), watch progress, see
validation results per check, read the band report and the gate decision.

Disabled Jacobian/Diff tabs in the readout viewer become enabled when a validated artifact exists for
the selected model — driven by `meta.types`, which doc chain 023 already derives from the stream.

---

## 5. API / Integration

- `POST /api/v1/jlens/artifacts` — start a fit
- `GET /api/v1/jlens/artifacts` · `GET /api/v1/jlens/artifacts/{id}`
- `POST /api/v1/jlens/artifacts/{id}/validate`
- `GET /api/v1/jlens/artifacts/{id}/band-report`
- `GET /api/v1/jlens/reports/replication`
- Binds `POST /api/v1/jlens/readout` (currently 501) for `JACOBIAN_LENS` requests.

---

## 6. Data

New tables for the artifact, its validation results, the band report and the gate decision. The
artifact schema types already exist (`JLensArtifact`, `JLensArtifactRecipe`, `LayerApplicability` in
`backend/src/schemas/jlens.py`, doc chain 022).

---

## 7. Dependencies

- Doc chain 022 (readout service, wire format, envelope guard) — **shipped**.
- `layer_discovery.discover_transformer_structure()`, `forward_hooks.py`.
- `anthropics/jacobian-lens` vendored at a recorded commit; the repository is unmaintained and not
  accepting contributions, so expect no upstream fixes.
- Local RTX 3080 Ti (12 GB) — the reference model is ~2.4 GB in fp16, leaving room for the backward
  pass, so lens work does not contend with the cluster GPU serving miLLM.

---

## 8. Success Criteria

- A validated artifact exists for at least **two architectures**, one hybrid and one dense.
- Per-layer applicability is recorded and inapplicable layers are absent, not zero.
- All six validation classes pass, round-trip included, against a live consumer.
- A band report derives that model's own boundaries; bands appear in the UI for that model and
  nowhere else.
- The gate decision is recorded with evidence, and a NO-GO is representable end-to-end.
- No test, metric or report scores the J-lens on next-token agreement.
- Every capability has an MCP tool covered by the reachability harness.

---

## 9. Non-Goals

- The oracle lens (requires fine-tuning two auxiliary models including an RL stage).
- A J-lens ingestion API against Neuronpedia — no such upstream path exists.
- Workspace-aligned dictionary training.
- Runtime evaluation or streaming of readouts — that is the miLLM-side BRD.

---

## 10. Testing Requirements

- Fit and readout paths contain **no architecture name** — a guard over the module source.
- Both architectures produce a well-formed artifact; a suite that only ever sees one is how the old
  whitelist survived.
- **Negative control on the hook target**: capture at `residual_norm_module` instead of
  `layers_module[L]` and assert the readout degrades measurably. Without it, a wrong-hook artifact
  looks plausible.
- Envelope asserted against arithmetic derived from the model's own config, and **no
  `n_vocab × d_model` allocation** on the fit or readout path.
- Each validation class fails when its condition is violated — including a deliberately corrupt
  artifact for STRUCTURAL and a truncated one for ENVELOPE.
- A test that **fails if next-token agreement is used as a scoring metric** anywhere.
- A NO-GO decision is representable and rendered.
- MCP tools present in the **live registry**, with payload and call count asserted.

---

## 11. Traceability

| Source | Covered by |
|---|---|
| BR-001 (replication, published either way) | §3.8 |
| BR-002 (band report; boundaries never ported) | §3.6 |
| BR-003 (GO/NO-GO/GO-AT-LARGER-SCALE) | §3.9 |
| BR-004 (agreement is never a quality metric) | §3.7 |
| BR-030 (six validation classes before handover) | §3.5 |
| BR-031 (construction primary, acquisition opportunistic) | §3.3, §3.4 |
| BR-032 (model-agnostic; per-layer applicability) | §3.1, §3.2 |
| BR-027 broadened (full MCP parity) | §3.10 |
| PADR IDL-41 (model-agnostic construction) | §3.1 |
| PADR IDL-42 (synthesize-on-demand storage) | §3.3 |
| PADR IDL-46 (artifact mount, not upload) | §3.5 |
