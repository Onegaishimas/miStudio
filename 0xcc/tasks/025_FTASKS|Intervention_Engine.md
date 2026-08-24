# Task List: Intervention Engine Extension

**Document ID:** 025_FTASKS|Intervention_Engine · **Version:** 1.0
**Status:** ⏳ Core implemented (2026-08-01) — 6 mutation controls verified biting

## Phase 1: The control is not optional (BR-018)
- [x] 1.1 `InterventionResult` takes `control` and `control_outcome` POSITIONALLY — no default,
      no Optional. A run without its control cannot be CONSTRUCTED, let alone reported.
- [x] 1.2 The signature itself is asserted, so adding a default is caught immediately.
- [x] 1.3 `excess_over_control` is the finding; the raw outcome is not one.
- [x] 1.4 Control size-matched (`k > 0`) and reconstructible from a recorded seed.

## Phase 2: Paired run with clamping (BR-016)
- [x] 2.1 `ClampSpec` keyed by (position, layer) — per-position-only leaks the effect through the
      unclamped positions and reports it as the clamped quantity.
- [x] 2.2 An empty clamp is distinguishable from no clamp.
- [ ] 2.3 Wire clean-pass capture and the intervened pass to the existing circuits engine.

## Phase 3: Four primitives (BR-017)
- [x] 3.1 ADDITIVE · 3.2 PROJECTIVE ABLATION (normalises an unnormalised direction, which would
      otherwise scale the ablation by its own magnitude and look like a stronger effect)
- [x] 3.3 DYNAMIC TOP-K, EXCLUDING clean-pass coordinates — without the exclusion it ablates
      ordinary behaviour and reports the consequence as an intervention effect.
- [x] 3.4 COORDINATE SWAP, source undisturbed.
- [x] 3.5 Primitive and parameters recorded with every result.
- [x] 3.6 Swap layer default DERIVED from `n_layers` (BR-017 v0.2) — a constant tuned on a large
      model oversteers a small one, which is the amendment's reason for existing.

## Phase 4: Outstanding
- [ ] 4.1 Endpoints + Celery task (model-bound: queue and poll, per the 021 hardware finding).
- [ ] 4.2 MCP tools + reachability harness.
- [ ] 4.3 UI surface.
- [ ] 4.4 Review rounds 2 and 3.

**Acceptance:** a result cannot exist without its control; clamping holds every named
(position, layer); dynamic top-k never ablates clean-pass behaviour; the swap default scales with
the model; and every run records the primitive that produced it.

## Relevant Files

> **Added 2026-08-24 (MIS-E2E-153).** This file had no `## Relevant Files`
> section, and neither did the four beside it — the five FTASKS with the most
> implementation and the least traceability. The PPRD marked their rows
> "Planned" and `CLAUDE.md`'s Document Inventory had no entry for them at all,
> while their own boxes ran 68–100% checked and the code below exists. A shipped
> feature with no doc→code join is one nobody can review, and the framework
> names this section as the join.
>
> Paths verified to exist at the time of writing; `test_task_docs_traceability`
> fails if one stops resolving.

| File | Purpose |
|---|---|
| `backend/src/services/jlens_intervention.py` | Paired-run execution with clamping, projective ablation, coordinate swap |
| `backend/src/workers/jlens_intervention_tasks.py` | GPU-bound Celery entry point for the above |
| `backend/src/services/jlens_causal.py` | Causal-claim helpers shared with the claims-discipline surface |
