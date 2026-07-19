# Feature PRD: Intervention, Validation & Faithfulness

**Document ID:** 017_FPRD|Circuit_Validation
**Version:** 1.0
**Status:** Planned
**Related:** BRD-MIS-CIRCUITS-001 (BR-009) as specified by BRD-MIS-CIRCUITS-002 (BR-017, BR-018, BR-019) · 000_PPRD §3.18 · IDL-34 · normative math: CIRCUITS-002 A.5/A.7 · consumes 016 candidates; feeds 018 rungs

---

## 1. Overview

### 1.1 Purpose
Make "causally validated" mean what the interpretability literature expects: defined intervention
semantics, error-term preservation, effect sizes against a proper null, reproducible manifests — at edge
level AND circuit level (faithfulness) — and retire every fabricated causal number from the product.

### 1.2 User Problem
CIRCUITS-001 required real intervention but left it underspecified: unstated suppression semantics, no
SAE-error handling, no baseline, no noise floor, edge-level only. Meanwhile the existing ablation surface
fabricates its numbers with no forward pass. A "validated" badge over either would be the exact
credibility failure BR-009 exists to prevent.

### 1.3 Solution
- **Intervention engine v2 (BR-017):** directional subtraction preserving the SAE reconstruction error;
  recorded baselines; reproducible validation manifests.
- **Edge criterion (BR-018):** standardized effect size vs a shuffled-pair null + sign consistency;
  tested-and-failed recorded, never dropped.
- **Faithfulness (BR-019):** whole-circuit necessity + tractable sufficiency at promotion; badge, not
  gate.
- **Remediation:** the heuristic surface relabeled "impact estimate (statistical — no inference)".

## 2. User Stories
- **US-1:** I select the attribution-re-ranked top-20 candidates from a discovery run and launch
  validation; each edge returns validated / tested-and-failed with its measured ES, and the batch reports
  survival rate — for BOTH orderings, so I can see whether attribution re-ranking earned its keep.
- **US-2:** Every validated claim has a manifest; I can open it, see prompts/seeds/thresholds/null
  summary, and re-run it to reproduce the number.
- **US-3:** At promotion time I run faithfulness on a circuit: necessity (behavior lost when the whole
  circuit is ablated) and sufficiency (behavior kept under top-k non-member ablation), and the scores
  appear on the circuit.
- **US-4:** The old ablation panel now says "impact estimate" with a no-inference caveat — nowhere does a
  fabricated number read as causal.
- **US-5 (agent):** Via MCP I validate edges, fetch manifests, run faithfulness, and read survival/uplift
  numbers — the full loop without the UI.

## 3. Functional Requirements

### 3.1 Intervention engine v2 (BR-017)
1. Default intervention = **directional subtraction**: `x'(t) = x(t) − (a_u(t) − a_base)·W_dec[:,i]` at
   the hook point, all positions v1; the SAE reconstruction error is untouched by construction (never
   re-decode). Any reconstruction-swap variant MUST re-add ε.
2. Baseline `a_base`: zero (default) | corpus-mean (from the 016 store) — a recorded run parameter.
3. Measurement passes are matched and deterministic (same prompts, seeds, greedy); Δ measured over
   tokens where the clean pass fired upstream.
4. **Validation manifest** persisted per run: intervention type, baseline, prompts, seeds, thresholds,
   null summary, measured values — sufficient to reproduce; manifests are first-class records retrievable
   by id.

### 3.2 Edge validation criterion (BR-018)
5. **Standardized effect size** `ES = mean_p(Δ_p)/σ_d`, σ_d from the capture store (no extra passes).
6. Validated (rung 2) iff |ES| exceeds the **shuffled-non-edge-pair null** ES threshold AND sign
   consistency ≥ configured fraction (default proposal 8/10; config).
7. Failures recorded **tested-and-failed** on the edge (rung history per IDL-35) — never silently
   dropped.
8. Batch results report **survival rate per rank tier and per ordering** (attribution-re-ranked vs
   co-activation-only → the BR-022 uplift number, reported here).

### 3.3 Circuit-level faithfulness (BR-019)
9. **Necessity:** ablate all circuit members simultaneously (BR-017 semantics);
   `necessity = [B(clean) − B(ablate M)] / [B(clean) − B(ablate all at circuit layers)]`.
10. **Sufficiency (tractable v1):** ablate top-k non-member features by mean activation at the circuit's
    layers (default k=256/layer, disclosed); necessity-only runs allowed with sufficiency marked
    untested.
11. Behavior metric default = the compare-workflow output-shift measures (continuity with the existing
    validation bar); metric identity recorded in the manifest. Cluster-granularity circuits ablate
    member clusters' features.
12. Faithfulness scores + parameters persist on the circuit and in its contract record; **badge, not
    gate**.

### 3.4 Heuristic remediation (BR-009)
13. The statistical ablation surface is relabeled "Impact estimate (statistical — no model inference)"
    everywhere (API `method` field, Feature Detail modal retitle, MCP docstring); its docstring no
    longer claims inference; the word "causal" never appears below rung 2 (copy-audit shared with
    015/016/018).

## 4. User Interface
- **Circuits panel — Validation tab (new):** candidate selection (top-K by either ordering), threshold/
  sign-consistency config, run; per-edge results (ES, null threshold, status chip); batch banner with
  survival rates per ordering + the uplift number; manifest viewer (drawer with reproduce action).
- **Faithfulness:** "Run faithfulness" on a circuit (promotion surface, 018); scores + k + metric
  displayed on the circuit card.
- **Feature Detail modal:** ablation section retitled with caveat.

## 5. API / Integration
- `POST /api/v1/circuit-validation` (run id, candidate scope, ordering, thresholds) · `GET` results ·
  `GET /api/v1/validation-manifests/{id}` · `POST /{id}/reproduce`.
- `POST /api/v1/circuit-faithfulness` (circuit id, mode necessity|both, k, metric) · results on the
  circuit.
- MCP (`circuits` category): `validate_circuit_edges`, `get_validation_manifest`,
  `reproduce_validation`, `run_circuit_faithfulness`.
- WS progress channels per house pattern.

## 6. Data / Types
- `validation_manifests` table (`vman_` ids, JSONB payload, kind edge|faithfulness|reproduction,
  parent refs); edge validation results write into 016's discovery-run candidates (and 018's circuit
  edges once promoted); faithfulness results on 018's circuit records.
- Alembic migration (single-head check).

## 7. Dependencies
- 016: candidates + both orderings, capture store (σ_d, corpus-mean baselines, upstream firing
  positions, prompt-window reconstruction), stats-service null machinery (reused for ES nulls).
- 015: suppression hook = steering-hook variant; steering worker lifecycle fixes.
- 018: rung model (IDL-35 shared enum), circuit records for faithfulness; sequencing — 018's ladder
  model lands first (it's the shared enum), then 017 populates rungs 2–3.

## 8. Success Criteria
1. The ε-preservation property is pinned by test (directional subtraction leaves ‖ε‖ unchanged at the
   hook point).
2. A validation batch over a seeded candidate set produces per-edge ES + status + manifests; survival
   rates per ordering reported, uplift number computed; ≥1 edge reaches rung 2 end-to-end.
3. Manifest reproduction: re-running a stored manifest reproduces the recorded ES within tolerance
   (automated test).
4. Faithfulness on ≥1 circuit yields necessity (+ sufficiency with k disclosed) using the
   compare-workflow metric; scores render on the circuit.
5. Copy audit passes: no causal language below rung 2 anywhere; the heuristic surface carries the
   no-inference caveat.
6. MCP agent completes validate → manifest fetch → reproduce → faithfulness without the UI.

## 9. Non-Goals
- Position-restricted intervention (arrives with Tier-2.5); noising/denoising baseline variants and
  task-metric-specific faithfulness (future hardening); exhaustive validation; edge typing/rung display
  surfaces (018); miLLM anything.

## 10. Testing Requirements
- Unit: directional-subtraction math (ε untouched — the canonical-mistake pin), baseline variants,
  ES arithmetic, shuffled-non-edge null, sign-consistency gate, tested-and-failed recording,
  necessity/sufficiency formulas on synthetic behavior metrics, manifest completeness validator.
- Integration (GPU): one real edge validation on a planted-obvious edge (forward passes asserted;
  deterministic repeat within tolerance); one faithfulness run (necessity path) on a small circuit.
- API: scope/threshold validation, manifest retrieval/reproduce, hostile inputs.
- Frontend: results table, survival/uplift banner, manifest drawer, faithfulness display.
- Copy audit: shared grep suite (015/016/017/018) for causal-language discipline.
- E2E: validate top-K from a real discovery run → manifests → faithfulness on a promoted circuit;
  screenshot `0xcc/caps/miStudio_Circuit_Validation_<date>.png`.

## 11. BRD Traceability

| BRD req | Covered by |
|---|---|
| BR-009 (real intervention; no fabricated causal claims) | §3.1, §3.4 |
| BR-017 (002: suppression semantics, ε preservation, baselines, manifests) | §3.1 |
| BR-018 (002: ES-vs-null + sign-consistency criterion; tested-and-failed) | §3.2 |
| BR-019 (002: circuit necessity + tractable sufficiency at promotion) | §3.3 |
| BR-022 partial (uplift REPORTING lives here; the pass itself is 016) | §3.2 item 8 |
