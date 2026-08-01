# Feature PRD: Runtime Handoff — Watchlists, MCP Parity & the Cost Envelope

**Document ID:** 028_FPRD|Runtime_Handoff
**Version:** 1.0 · **Status:** Planned
**Related:** PPRD §3.29 (row 29) · BRD-MIS-JSPACE-001 v0.3 BR-025..028 · closes the arc

---

## 1. Purpose

Emit what miLLM needs to evaluate concepts per token at inference, and make every J-space capability
reachable by an agent and honest about what it costs.

**This increment emits the artifact and the contract only.** Runtime evaluation is miLLM's plane
(`BRD-MILLM-JSPACE-001`).

## 2. Watchlists (BR-025)

Named sets of concept directions with **per-concept detection thresholds** and a **stated scoring
definition**, exportable under BR-021's contract, referencing the artifact they were built against.

A threshold without its scoring definition is not portable: the consumer applies it to a
differently-computed score and gets a different detector while believing it has the same one.

## 3. One validated reference watchlist (BR-026)

Ship at least one **validated** watchlist as an existence proof and a template, implementing the
source paper's **evaluation-awareness score**: the mean lens log-probability of a small set of
evaluation-awareness tokens **minus** the mean over a control set of unrelated concrete nouns,
averaged across layers.

The subtraction is the score. Reporting the raw mean would make every prompt look evaluation-aware,
because common tokens have high log-probability everywhere.

## 4. Full MCP parity (BR-027, as broadened)

Every capability a user can reach in the workbench SHALL be reachable by an agent — readout in all
three modes, artifact listing and validation, fitting, band report, gate, annotation, interventions,
watchlists.

Tools ship **with the feature that creates them**, and each is covered by the reachability harness.

## 5. Cost envelope (BR-028)

Every J-space operation class SHALL report **wall-clock and peak memory** against the deployment
envelope, and SHALL surface an **estimate before** a user or agent commits to a run.

Operation classes: artifact construction, readout, decomposition, annotation sweep, intervention run,
template-lens vocabulary build.

This is not decoration. An annotation sweep over a 32k-feature dictionary and a single readout differ
by orders of magnitude, and an agent with no estimate cannot tell them apart before starting.

## 6. Non-Goals

- Runtime evaluation or streaming during inference — miLLM's plane.
- A watchlist that scores without a recorded definition.

## 7. Testing Requirements

- A watchlist without a scoring definition is refused.
- The evaluation-awareness score SUBTRACTS its control; the raw mean is not the score.
- Estimates exist for every operation class, and an unestimated class is refused rather than
  defaulted to cheap.
- Every J-space MCP tool is in the live registry with payload and call count asserted.

## 8. Traceability

| Source | Covered by |
|---|---|
| BR-025 (watchlists: thresholds + scoring definition) | §2 |
| BR-026 (one validated reference watchlist) | §3 |
| BR-027 (full MCP parity) | §4 |
| BR-028 (cost envelope, estimate before commit) | §5 |
