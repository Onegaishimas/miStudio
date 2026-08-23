# P03 — ML / GPU: all three rounds

**Phase:** P03 · **Date:** 2026-08-23
**Scope:** `backend/src/ml/` (9 files, 5,142 lines) plus the jlens services that
own the band report, validation and artifact lifecycle.

Mutation log: `mutations/P03-mutations.md` (6 run, 3 survived, 3 killed).

## R1 — discovery

`/code-review high` over the six substantive `ml/` modules. **15 findings, 8 of them
verified by execution** against the project venv (torch 2.9.1).

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-079** | **P0** | The documented `affine_residual` freeze-leak gate **does not exist** — the threshold is stored and never read |
| MIS-E2E-080 | P1 | "Converged" measures the 1/N shrinkage of a running mean, not stabilisation of J |
| MIS-E2E-081 | P1 | The published artifact writes a positional **spread** under keys named `linearisation_residual_*` |
| MIS-E2E-082 | P1 | A raise during norm patching leaks the process-wide SDPA patch and never releases `_FREEZE_LOCK` |
| MIS-E2E-083 | P1 | Circuit capture and attribution run the SAE **off-distribution** — `_load_sae` drops `normalize_activations` |
| MIS-E2E-084 | P1 | Three generic-caller contract breaks in the SAE classes, all reproduced (TypeError, silent bias offset, IndexError) |
| MIS-E2E-088 | P1 | Band metrics: rank-deficient QR inflates FVE 4.5×; two controls have no caller; `derive_boundaries` does not implement "sustained" |
| MIS-E2E-085 | P2 | `anthropic_rescale` is arithmetically identical to `constant_norm_rescale` (2.4e-7) — two of six "frameworks" are one |
| MIS-E2E-086 | P2 | Raw-space MSE vs normalized-space sparsity; ghost gradient encodes the wrong tensor |
| MIS-E2E-087 | P2 | Layer discovery returns the alphabetically-first norm, discarding the documented preference |
| MIS-E2E-089 | P3 | Two candidates **REFUTED** by execution and recorded as such |

### The two that contradict documented capability claims

**MIS-E2E-079** was re-verified independently. `CLAUDE.md` states as shipped:
*"`affine_residual` refuses a fit whose freeze leaked… fit time is the only point
where it is detectable."* `max_affine_residual` has exactly two occurrences in the
whole tree — a constructor parameter and its assignment. It is never read.
`affine_residual` appears only in docstrings; it is never computed. The single
detection point for an incomplete freeze is absent, so a leaked-freeze lens is
fittable, validatable, publishable and mountable, and nothing would say otherwise.

**MIS-E2E-080** was re-verified at source. The update is
`accumulated += (mat - accumulated) / seen` and the stop criterion is the relative
change of that mean. A running mean's increment is `O(σ/n)`, so the delta shrinks
because the denominator grows — convergence is declared at `n ≈ σ/δ`, proportional
to per-prompt variance and independent of whether J stabilised. The reviewer's
simulation (518 / 1050 / 2030 for σ = 0.5 / 1.0 / 2.0) brackets the two real fits
`CLAUDE.md` reports as *"paper-aligned converged lenses (gemma 634, LFM2 1097)"*.

## R2 — mutation controls

**3 kills, and they are the good news:** the `endswith("norm")` lesson is pinned by a
test named for the `NormedBlock` trap (M10); BR-002's AST guard fires on a
straightforward reintroduction of the published boundaries (M11); and BR-030's
fail-closed property holds — a `NOT_RUN` class cannot pass, exactly as its docstring
promises (M15). Three previous rounds' fixes that survive mutation.

**3 survivals**, yielding two findings:

- **MIS-E2E-090** — BR-002 is stated as *"no band constant anywhere"* and checked in
  a hardcoded two-module tuple. A literal `40`/`90` in `jlens_band_service.py`, a
  sibling service in the same package, is not scanned (M13). The obfuscation evasion
  (M12, `4 * 10`) is recorded as unrealistic; the scope gap is the finding.
- **MIS-E2E-091** — `weights_only=True`, the only thing stopping a downloaded
  artifact from executing pickled code in the GPU worker, has no test at any of its
  three sites (M14).

## R3 — verification & closure

**Verdicts**

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 079, 080, 090, 091 |
| **PLAUSIBLE** (8 of these execution-verified by the reviewer) | 081, 082, 083, 084, 085, 086, 087, 088 |
| **REFUTED** | 089 (both sub-items) |

**Refutations recorded so no later round re-raises them (MIS-E2E-089):**
`torch.quantile`'s historical 2^24-element limit no longer applies in torch 2.9, so
`calibrate_thresholds` is fine at 134M elements; and `torch_dtype=` remains
back-compatible in transformers 5.15.1 despite the deprecation. Both checked by
execution against the project venv rather than assumed — the reviewer discarded them
before reporting, which is the behaviour this audit wants.

**On coverage of this phase.** R1 ran `/code-review` over `ml/`; the security surface
of this layer was largely covered from P02 (`torch.load weights_only` and
`trust_remote_code` were both found there, and the `ollm_server` RCE-by-design path
is recorded for P10). No separate `/security-review` pass was run for P03 and none of
its findings is security-classed except by inheritance — stated plainly rather than
implied by an empty section.

**No live GPU verification.** The k8s host is reachable over HTTP and the MCP server,
but `ssh` access for `kubectl`/`nvidia-smi` is unavailable in this session
(BASELINE §1). The GPU-dependent claims here — MIS-E2E-082's patch leak, MIS-E2E-083's
off-distribution capture — are the ones this repo's history says only hardware
finds, and they are recorded as PLAUSIBLE for that reason, not as CONFIRMED.

## Phase closed

**13 findings** (MIS-E2E-079 … 091), **1 P0**. Mutations: 6 run, 3 survived.
Tree verified clean after every mutation.

**The one sentence for the synthesis:** this phase's defects are concentrated in the
words — "converged", "residual", "refuses", "anywhere", "six frameworks" — each of
which names something the code does not do, in a subsystem whose entire output is
evidence.
