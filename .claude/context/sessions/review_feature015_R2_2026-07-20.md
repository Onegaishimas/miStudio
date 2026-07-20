# Review Record — Feature 015: Multi-SAE Cross-Layer Steering — ROUND 2

**Date:** 2026-07-20
**Scope:** post-R1 at HEAD 66e2fe3. Code-review (9, F1–F9) + 4-perspective /review (13). **22 findings.** All R1 fixes verified HOLDING. No CRITICAL/HIGH from the /review; the code-review found 2 P1s. Pre-existing addressed per the user directive.

## THE ARC IS CLOSED (definitive)
Both agents traced the full browser chain end-to-end: discover (016) → validate rung-2 (017, ES→Circuit.edges) → promote (018) → "Steer this circuit" → loadCircuitIntoSteering (per-layer sae_id) → circuit_id → _load_circuit_edges → detect_hazards → "compounding (validated, ES=0.80)" in the HazardBanner → per-layer budget bars → Blended-generate through each member's own-layer SAE → circuit-titled result. **The arc's payoff — a hazard QUANTIFIED from a causally-validated effect size — is visibly, legibly demonstrated in the browser, API, and MCP.** The increment can close.

## PART A — R1 fixes: all HOLD.

## The two P1s (both fixed)
1. **F1 — `strict=False` masked garbage weights.** The CPU loader created the SAE then `load_state_dict(strict=False)`; an arch mismatch (JumpReLU checkpoint into a "standard" class) would leave the decoder at RANDOM INIT → G + weight_prior computed from noise, presented as principled numbers. **Fixed:** verify the decoder/encoder loaded with the right d_model orientation, else RAISE → the endpoint degrades to approximate G=1 (honest) instead of trusting init noise.
2. **F2 — missing-layer SAE → wrong-decoder fallback → 422 on BOTH allocate and generate.** loadCircuitIntoSteering fell back to the primary (wrong-layer) SAE for a member whose layer had no saes[] entry, which then failed the own-layer rule everywhere. **Fixed:** skip members whose layer has no SAE (warn), never mis-route to the primary.

## Also fixed
- **F5 (pre-existing, now one-click reachable):** single-layer allocation still GPU-loaded the SAE; the "Steer this circuit" button makes this a browser action, so routed single-layer through load_sae_weights_cpu too (multi-layer already was). The read-only allocate never touches the GPU now.
- **F4:** strength=0 vs nullish — an explicit parked-at-0 member is honored (was overwritten with the default).
- **F3:** steerCircuit gates on SAEStatus.READY with a clear message (was a confusing downstream 400).
- **PROD-1:** circuit load clears the prior run's combinedResults/title (was leaving stale output mounted).
- **QA-2:** cap-drop notice — loading a >20-member circuit surfaces "only the first N loaded" (was silent tail-layer loss).
- **F7:** constants_by_sae → constants_by_layer (name matched the key).
- **F8:** assert ml.strengths is 1:1 with request members (order-preservation guard for hazard sign detection).
- **F9/TEST-1:** endpoint integration test — a circuit_id with a rung-2 edge produces a `validated:ES=` hazard (the arc-payoff seam, previously unpinned). + CPU-loader correctness test (skips gracefully where the community-save helper is absent).

## Recorded (R3/close-out, not gating — tech debt)
- **ARCH-1 (top debt):** cross-layer hazards + steering are FEATURE-LEVEL only in v1; cluster-granularity circuits dead-end or lose cluster→cluster hazards. Document explicitly in close-out.
- Two-SAE GPU generation run (§8.1) + VRAM<200MB (§8.6) — the only unproven success criteria; GPU close-out.
- F6 reorderFeatures doesn't clear cluster provenance (data-safe today, order-independent keys) — consistency nit.
- Pre-existing DatasetsPanel unhandled-rejection line (017-era) — separate cleanup so the suite is genuinely clean.

## R2 outcome
**22 findings, both P1s + all correctness/UX fixed.** The arc is verified closed end-to-end in UI+API+MCP. Backend steering suite green; FE 916 tests + build green. R3 is the final gate of the whole circuits arc.
