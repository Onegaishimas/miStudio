# P03 — mutation control log

**Phase:** P03 ML / GPU · **Round:** 2 · **Date:** 2026-08-23

Same discipline: back up → one edit → **confirm it landed** → run the suite →
restore → verify `git diff` clean. No reading agent ran concurrently.

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M10 | `ml/jlens_fitter.py:326` | `type(m).__name__.lower().endswith("norm")` → `"norm" in …` — reverts the recorded `NormedBlock` lesson | ✅ | **KILLED** — `test_norm_discovery_does_not_capture_a_block_merely_named_for_a_norm` |
| M11 | `services/jlens_band_report.py` | Inject literal `{"workspace_start": 40, "motor_start": 90}` — the forbidden BR-002 constants | ✅ | **KILLED** — `test_no_band_constant_exists_in_the_derivation_module` |
| M12 | `services/jlens_band_report.py` | The **same constants** as `4 * 10` and `int("90")` | ✅ | **SURVIVED** → MIS-E2E-090 |
| M13 | `services/jlens_band_service.py` | Literal `40`/`90` in a **sibling jlens module** | ✅ | **SURVIVED** → MIS-E2E-090 |
| M14 | `services/jlens_artifact_service.py:396` | `weights_only=True` → `False` | ✅ | **SURVIVED** → MIS-E2E-091 |
| M15 | `services/jlens_validation.py:118` | Let a `NOT_RUN` validation class pass the report — disable fail-closed | ✅ | **KILLED** — `test_report_fails_closed_when_a_class_never_ran` |

**3 of 6 survived.**

## The three kills are genuinely good guards

- **M10** — the `endswith("norm")` lesson (a substring match once captured a decoder
  block named `NormedBlock`, and freezing a decoder block replaces it with an
  elementwise rescaling) is pinned by a test named for exactly that trap.
- **M11** — BR-002's by-construction guard is a real AST scan and it fires on a
  straightforward reintroduction of the published boundaries.
- **M15** — BR-030's fail-closed property holds: a class that never ran cannot pass.
  The docstring's claim (*"'we did not check' and 'we checked and it was fine' must
  never produce the same verdict"*) is enforced, not asserted.

Three previous rounds' fixes that survive mutation. That is worth as much signal as
the failures.

## M12 + M13 — BR-002 is guarded in two modules, not "anywhere"

BR-002 is described in `CLAUDE.md` as *"no band constant **anywhere**, by
construction"*. The guard is an AST walk over `inspect.getsource` of exactly two
modules — `jlens_metrics` and `jlens_band_report` — rejecting `ast.Constant` values
in `(38, 40, 90, 92)`.

Two evasions, both of which left the suite green:

- **M12 (obfuscation, unrealistic):** `4 * 10` and `int("90")` are not
  `ast.Constant` numeric nodes with those values. Recorded for completeness; nobody
  writes this by accident.
- **M13 (scope, realistic):** a literal `40`/`90` in `jlens_band_service.py` — a
  sibling service in the same package, a plausible place for a default — is not
  scanned at all, because the module list is hardcoded.

M13 is the finding. It is the "derive from the registry, not a hand-list" lesson
applied to a guard: the rule is package-wide and the check is two-module.

## M14 — an RCE-prevention control with no test

`weights_only=True` on `torch.load` is what stops a downloaded J-lens artifact from
executing pickled code — and the code says so, with an explicit comment at
`jlens_artifact_service.py:382` that *"an artifact is an untrusted file"*. Flipping
it to `False` leaves 58 jlens tests green.

## Equivalent mutants

None chased. M12 is arguably an unrealistic mutant rather than an equivalent one —
recorded as such rather than counted as a finding on its own; M13 carries the weight.
