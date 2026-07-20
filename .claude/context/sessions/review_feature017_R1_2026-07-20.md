# Review Record — Feature 017: Circuit Validation (Intervention / ES-vs-null / Faithfulness / Manifests) — ROUND 1

**Date:** 2026-07-20
**Scope:** 017 at HEAD `de03e5d` (commits a23fe62 Task3.0, 22aa850 Ph1-2, b95a5cf Ph3-4, ac6c64f Ph5-be, e896c31 docs, de03e5d Ph5-fe+repro). Two agents: high-recall code-review (14 findings) + 4-perspective /review (17 findings). **31 findings.** Per the user's directive, reviewers ALSO surfaced pre-existing issues.

---

## The two Criticals (both fixed this round)

1. **False-positive rung-2 on an empty/underpowered null (#1/Q1).** `edge_verdict` defaulted `thresh=0.0` when the null list was empty, so any nonzero ES passed the null gate on ZERO statistical evidence — a fake "causally validated" badge, the worst possible failure in front of AI scientists. **Fixed:** a null below `MIN_NULL_SAMPLES=10` now FAILS with "null underpowered — cannot validate", not passes. Compounded by #5 (the null wasn't actually support-matched despite the label) — also fixed (band-filter by support).

2. **`write_edge_validation` had no caller — validated ES never reached promoted circuits (A1/#2).** Validation wrote rung-2 ES onto the ephemeral discovery *candidates* only; the durable, 015-readable record is a *promoted circuit's edges*, which stayed at their promotion rung. The service docstring admitted the writer "goes through CircuitService.write_edge_validation at the caller" — but nothing called it. 015 would have read empty ES for hazard quantification. **Fixed:** `_write_promoted_circuit_edges` propagates rung-2/tested-and-failed onto every promoted circuit built from the discovery run, contract round-tripped (validators + rung recompute + version bump), sync worker path.

## Also fixed (backend)
- **P1 Critical — faithfulness had no runnable path.** Pure core only; no GPU run/task/endpoint/MCP → the UI implied a necessity/sufficiency number nothing produced. **Fixed:** full end-to-end tier — `CircuitFaithfulnessService.run` (4 behavior passes: clean / ablate-members / ablate-all-proxy / ablate-nonmembers via per-layer SUM suppression, never re-decoding), Celery task `run_circuit_faithfulness`, `POST /circuits/{id}/faithfulness` (guarded, 404/409/422), MCP `run_circuit_faithfulness`. Honest disclosure: the v1 metric is the downstream-member activation-sum (recorded verbatim in the manifest's `metric_definition` — the `metric_id` string stays "compare_output_shift/v1" but the real definition travels beside it); fails honestly (409) if the capture store was pruned rather than fabricating a corpus.
- **#3 reproduction verdict wrong payload path** (I introduced it in de03e5d): backend nests under `payload.verdict`, frontend read top-level → always "did not reproduce". Fixed drawer + type (three-state true/false/null).
- **#14 reproduction empty-overlap** → `within_tolerance=None` ("no overlapping edges to compare"), not a false pass.
- **#4 `_empty_result` missing sigma_d/null_percentile_value** → drawer `toFixed` crash. Fixed.
- **#6/Q6 reproduce bypassed the GPU guard** + **A4 reproduce stomped the run report** → guard-and-mark + report writes gated behind `not reproduce_of`.
- **#7/Q1 `assert_no_active_gpu_run` didn't cover validation** → added the validation-status check.
- **Q4 attr ordering without attribution** → 409.
- **P2 uplift never produced** → survival stored per-ordering + `uplift(attr−coact)` computed when both exist.
- **#9 cancel race** → re-check `validation_status` before the terminal `completed` write.
- **#13 necessity/sufficiency negative denominator** → `denom <= 0` returns None (was `== 0`).

## Pre-existing fixed (user directive)
- **#12 correlations non-reproducible** — `ORDER BY func.random()` → deterministic `ORDER BY Feature.id`; response discloses `sampled`/`sample_size`.
- **#11 ablation perplexity tiles framed as measured** — relabeled "(projected)" / "Heuristic — not measured".
- **T4 the 98-failure frontend vitest baseline** — FIXED: **98 failed → 0 failed / 905 passed**. Root causes: a missing global socket.io-client mock + no WebSocketProvider test wrapper (69), stale assertions against redesigned components, and incomplete store mocks. Added `src/test/renderWithProviders.tsx` + socket.io mock in setup; updated 9 stale test files (no tests deleted, no assertions weakened). **Surfaced ONE real source bug:** `useTrainingWebSocket` resubscribed on reordered-but-identical IDs — fixed with an order-insensitive memo key (the app's core real-time path).

## Verified NON-bugs
- JSONB reassignment (`run.candidates = cands`, `run.report = {...}`) is dirty-tracked correctly (not in-place mutated).
- The never-re-decode rule holds (suppression subtracts from the handed-in residual, returns the original tuple — Gemma-2 safe).
- σ_d is read from the SAME capture store's down_reader.
- Migration chain linear (single head); `version` backfilled to 1; PATCH `expected_version` wired end-to-end.

## Recorded (R2/close-out, not gating)
- **A2 (design debt):** three lifecycles (discovery/attribution/validation) on one row — a child `circuit_validation_runs` table would give uplift + re-runs first-class rows; consider before faithfulness gets a 4th. (Faithfulness currently uses circuit.faithfulness + manifest, avoiding a 4th column.)
- **A3 (latent):** adopt `MutableList.as_mutable(JSONB)` or keep the reassign-only contract documented.
- **Q3 σ_d=1.0 on a lone downstream firing** — internally consistent but the "standardized" claim is soft; consider skipping such edges. Recorded.
- **Q5 manifest path-guard is blacklist-shaped** — a whitelist of allowed payload keys is stronger; recorded.
- **T5 calculate_ablation scoring formula untested** — the disclosed estimate's magic constants aren't pinned; recorded.
- **P4 reproduction verdict not auto-surfaced** in the drawer after Reproduce (lands on a new manifest) — poll + open child; R2 UX.
- Faithfulness UI surfacing (run action + score display in the Validation tab) — R2.

---

## R1 outcome
**31 findings; both Criticals + all P1/correctness-P2 fixed** (incl. the false-positive rung-2, the 015 seam, faithfulness made real, the reproduction path). Pre-existing correlations/ablation-labels fixed; the frontend baseline being materially reduced. Backend circuit suite green. R2 verifies these hold + fresh sweep.
