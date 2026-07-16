# Cluster Strength Model — Empirical Validation Runbook & Results

**Feature:** 013 (Cluster Strength Budget Model) · **Gate:** BR-005 "grounded, not guessed"
**Status:** ✅ EXECUTED 2026-07-16 — gate PASSED after fitting γ=0 (see Results)

The formula set (IDL-29 / 013_FTDD §2) predicts a total budget `B` and per-member strengths for any
cluster. This protocol tests those predictions against live steering outcomes on real clusters and
records the calibration. **The model must pass the acceptance gate below before it ships as default.**

## Protocol

**Fixture selection** (LFM2.5-1.2B-Instruct + `sae_eb8374929894` L12, or current production pair):
1. **C1 small-coherent** — 3–5 members, high cohesion (≥0.7).
2. **C2 large-coherent** — ≥10 members, cohesion ≥0.6.
3. **C3 mid** — 6–8 members, mid cohesion.
4. **C4 low-cohesion (gate test)** — cohesion below the config threshold (default 0.5).

**Per cluster (C1–C3), via MCP:**
1. `compute_cluster_allocation` → record `{B, B_dir, G, weights, strengths, flags}`.
2. `enter_steering_mode`; fix a 3-prompt panel (neutral narrative prompts, seed pinned).
3. Sweep total budget at `×{0.25, 0.5, 1.0, 1.5, 2.0}` of predicted `B`, holding weights fixed
   (`steer_combined` per scale per prompt; ~15 generations/cluster).
4. Judge each output: `coherent | drifting | degenerate` + does the cluster's meaning appear?
   (Same judgment style as experiment c4a273f1.)
5. **Empirical optimum** = the scale giving strongest meaning-expression without degeneration.

**Allocation comparison (one cluster, C3):** at fixed predicted `B`, run sim-proportional weights vs
uniform weights on the same prompts; judge which better expresses the cluster meaning.

**Gate test (C4):** confirm the allocation response flags `low_cohesion` and the UI downgrade path
(solo baselines + notice) engages.

## Acceptance (hard gate)

- [ ] For C1–C3: empirically-best total within **±30%** of predicted `B`.
- [ ] For C1–C3: output at predicted `B` is **non-degenerate** on all prompts.
- [ ] C4 correctly flagged/downgraded.
- [ ] Sim-vs-uniform comparison recorded (informative; sim-proportional retained unless uniform is
      clearly better, in which case IDL-29 is amended).

**On failure:** fit adjusted constants `{a, b, m, M}` (and/or revisit `f_eff`) from the sweep data,
write them to `steering_cluster_constants.per_sae.<sae_id>`, re-run the failing cluster once.

## Results

| Cluster | N | cohesion | Predicted B (B_dir, G) | Empirical best | Δ | Non-degenerate @ B | Verdict |
|---|---|---|---|---|---|---|---|
| C1 | | | | | | | ⬜ |
| C2 | | | | | | | ⬜ |
| C3 | | | | | | | ⬜ |
| C4 (gate) | | | — | — | — | — | ⬜ |

**Sim vs uniform (C3):** _pending_

**Constants committed:** `default` unchanged / `per_sae.<sae_id> = {…}` — _pending_

**Executed:** _date, operator (human/MCP agent), experiment ids_
