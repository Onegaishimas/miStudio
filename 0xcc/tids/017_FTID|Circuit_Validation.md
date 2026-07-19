# Technical Implementation Document: Intervention, Validation & Faithfulness

**Document ID:** 017_FTID|Circuit_Validation
**Version:** 1.0
**Status:** Planned
**Related:** 017_FPRD · 017_FTDD · IDL-34 · CIRCUITS-002 A.5/A.7 normative

---

## 1. Implementation Order

1. Suppression hook variant + ε-invariance pins (GPU-light unit on toy tensors).
2. Manifest table/service/endpoints (+ reproduce skeleton).
3. Edge validation service + task (ES, shuffled-non-edge null, sign gate, batch orchestration).
4. Faithfulness service + task (member expansion, metric reuse).
5. Remediation (relabels + shared copy-audit suite).
6. MCP tools + Validation tab UI + manifest drawer.
7. GPU integration + reproduction test + E2E.

## 2. File-by-file

### 2.1 `backend/src/services/steering_service.py` (extend) or `circuit_intervention_hooks.py` (NEW)
- `_create_suppression_hook(sae, feature_idx, a_base, positions=None)` per FTDD §1 — house hook
  conventions (mirror `_create_steering_hook`); same-pass encode for `a_u(t)` via the existing layer
  capture pattern.
- **Pins:** ε-invariance (recompute ε pre/post — identical); subtraction exactness on toy tensors;
  positions=None ⇒ clean-fire positions only (from the measurement pass mask, not the store).

### 2.2 `backend/src/models/validation_manifest.py` (NEW) + Alembic
- `validation_manifests` per FTDD §4 (`vman_`, kind, payload JSONB, parent refs). Single-head check.

### 2.3 `backend/src/services/manifest_service.py` (NEW)
- persist/get/list-by-parent; `reproduce(id)` re-executes from payload via the intervention service and
  stores a `reproduction` manifest with per-value deltas + tolerance verdict.

### 2.4 `backend/src/services/circuit_intervention_service.py` (NEW) + `workers/circuit_validation_tasks.py`
- Prompt-window reconstruction from the 016 store + the SAME tokenization (manifest corpus ref) — reuse
  extraction's context-decode helpers; strongest-firing selection.
- Matched greedy passes (seeds fixed); Δ_p over clean-fire tokens; `ES = mean/σ_d` with σ_d from the
  store manifest stats (add σ_d to 016's manifest counts if not already there — coordinate).
- Shuffled-non-edge null: support-matched random same-layer upstreams against the same downstream;
  reuse 016's shuffle RNG conventions; separate null per batch, summarized into the manifest.
- Batch orchestration: per-ordering scopes; survival per tier per ordering; uplift number into BOTH the
  batch result and 016's discovery-run report (fills the placeholder).
- Results write into discovery-run candidates (`validation` field + rung history) and, when the edge
  belongs to a promoted circuit, into 018's circuit edges.

### 2.5 `backend/src/services/circuit_faithfulness_service.py` (NEW) + task
- Member expansion (cluster_ref → features via profile membership); simultaneous multi-feature
  suppression = one hook per layer subtracting the sum of member contributions (extend the hook to take
  a feature LIST — cheaper than N hooks).
- `B(*)` via the compare workflow's output-shift measures (import, don't fork — `metric_id` recorded);
  "ablate-all-at-layers" proxy = per-layer top-N by mean activation (N recorded, default 1024).
- Scores persisted on the circuit record + manifest.

### 2.6 Remediation
- `analysis_service.calculate_ablation` docstring + `method: "statistical_estimate"` response field;
  `FeatureDetailModal` retitle + caveat; MCP `get_feature_ablation` docstring.
- NEW shared test `backend/tests/unit/test_causal_language_audit.py`: greps built surfaces/copy for
  "causal|ablation-tested|proven" outside rung-2+ contexts (015–018 shared; wire into each feature's
  acceptance).

### 2.7 Endpoints + MCP + WS
- `circuit_validation.py`, `circuit_faithfulness.py`, `validation_manifests.py` (+ reproduce);
  WS `validation/{id}`, `faithfulness/{id}`. MCP: 4 tools in `tools/circuits.py` (extend 016's file;
  same category).

### 2.8 Frontend
- CircuitsPanel **Validation tab**: scope picker (ordering + K), config, per-edge results table
  (ES vs threshold, status chip, manifest link), batch banner (survival per ordering + uplift).
- `ManifestDrawer.tsx` (NEW): payload view + Reproduce button + reproduction verdict.
- FeatureDetailModal retitle (remediation).

### 2.9 Manual
- `circuits.md` += validation & faithfulness sections (ladder framing: what moves an edge to rung 2, a
  circuit to rung 3); feature-detail page note on the estimate relabel.

## 3. Pitfalls

- **Never re-decode** in the suppression path — ε dies silently and the measurement becomes SAE
  artifact (the A.2 warning; pinned).
- σ_d must come from the SAME capture store the candidates came from (manifest ref chain) — a different
  corpus's σ silently rescales ES.
- Prompt windows: same tokenization as capture (drift breaks position alignment) — assert tokenization
  id match.
- The uplift number needs BOTH orderings' survival measured on the SAME edge set size K — don't compare
  a K=20 attr tier against a K=50 coact tier.
- Faithfulness multi-feature hook: subtract the SUM per layer in one hook — N separate hooks reorder
  float ops and add overhead.
- Manifest payloads must be self-contained (no live refs that can drift) — reproduction is the test.
- Rung enum imports from 018's shared model — do NOT define a local status enum (IDL-35 divergence
  defect class).

## 4. Testing

- Unit: ε-invariance + subtraction exactness; baseline variants; ES arithmetic; support-matched null;
  sign gate; tested-and-failed recording; necessity/sufficiency formulas (synthetic metrics); manifest
  completeness validator; uplift arithmetic.
- Integration (GPU): planted-obvious edge validates (forward passes asserted, repeat-deterministic);
  reproduction within tolerance; one necessity run on a small circuit.
- API: scopes/thresholds validation, manifest retrieval/reproduce, hostile inputs.
- Frontend: results table, banner, manifest drawer, retitles.
- Copy audit: shared suite green across 015–018 surfaces.
- E2E: validate top-K of a real seeded run → manifests → faithfulness → scores displayed; screenshot
  `0xcc/caps/miStudio_Circuit_Validation_<date>.png`.
