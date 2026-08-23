# P08 — Frontend UI: all three rounds

**Phase:** P08 · **Date:** 2026-08-23
**Scope:** `frontend/src/components/**` (23 panels, 15,127 lines in `panels/` alone),
`frontend/src/config/panels.ts`, `App.tsx`, `Sidebar.tsx`

Mutation log: `mutations/P08-mutations.md` (2 run, 0 survived).

## R1 — findings (4 register entries, 10 defects)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-128** | **P0** | The prune dialog says *"report on"* and the toast says *"Dry-run queued"* while the task **permanently deletes** |
| MIS-E2E-129 | P1 | The Diff view shades **agreement** as disagreement, reports every rank off by one, and contradicts its own correct badge |
| MIS-E2E-130 | P2 | Settings: the labeling model is swept into the endpoints list with a delete button; five unawaited mutations; cross-card reversion |
| MIS-E2E-131 | P2 | Circuits: a poll dead on arrival under StrictMode; a hidden stale id submitted after deselection; `NaN` seeds; a silent export in Firefox |

### MIS-E2E-128, verified at source

Both the confirm text and the success toast read `preview.policy.dry_run` — a
snapshot from when the preview was fetched — while `prune_checkpoints.py:147,237`
re-reads the setting live at execution. Preview under dry-run, untick and Save, then
prune: the dialog says *"This will **report on** 12 checkpoint file(s)"*, the toast
says *"Dry-run prune queued"*, and twelve files are permanently deleted. The
`'PERMANENTLY DELETE'` branch exists; it is chosen from stale state.

This is the second time the same boolean has produced a finding. Feature 21's
recorded lesson was *"a boolean setting must fail to its DEFAULT, not to False (for
`dry_run`, False means delete)"* — the backend fixed that; the confirmation dialog
reads a stale copy of it.

## A correction to my own earlier finding

**MIS-E2E-023 was over-stated and is now corrected.** I claimed at P00 that
`ReadoutGrid`'s Rules-of-Hooks violation crashes the panel when the user switches to
a lens type with no fitted layers. It cannot, today: the selector is populated from
`meta.types`, and the backend emits `types` and `layers_by_type` **from the same
tuple on adjacent lines** (`jlens_readout_service.py:550-551`), so every offered type
has a non-empty axis by construction.

The hook violation is real and eslint reports it; the crash is held off by an
invariant maintained in one backend expression. **Downgraded P1 → P2** and recorded
as a latent crash. A register that overstates is as useless as one that misses.

## R2 — mutations, and a refuted hypothesis

Both mutations were killed. M24 is the interesting one: I expected it to survive.

P04 and P06 had each found a guard whose coverage was narrower than its docstring —
a janitor test named for one janitor, a payload assertion covering 16 of 116 tools.
`App.jlens.test.tsx` reads like the same shape. It is not: removing a *different*
panel id failed `Sidebar.jlens.test.tsx > panel registry > routes every nav entry`,
which iterates **every** nav entry despite the filename. The panel registry's
coverage is registry-derived and general.

Recorded as a refutation. Three phases in a row looking for hand-maintained lists is
exactly the frame that produces a false positive, and this one did not hold.

## R3 — verification

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 128 |
| **PLAUSIBLE** | 129, 130, 131 |
| **CORRECTED** | 023 (P1 → P2, crash unreachable) |
| **REFUTED** | the hypothesis that panel-registry coverage is panel-specific |

The three PLAUSIBLE entries are all browser-observable and were not driven live: the
frontend is not running locally (BASELINE §1) and the k8s deployment serves the built
bundle, so reproducing them means a Playwright pass against
`k8s-mistudio.hitsai.local`. That is carried into P12, which owns the live journeys,
rather than asserted here.

## Verified clean

`config/panels.ts`, `App.tsx` and `Sidebar.tsx` came out clean under review **and**
under mutation: all 13 `PANEL_IDS` are reachable from `navItems`/`bottomNavItems`,
rendered in `AppContent`, and the `localStorage` restore is narrowed through
`isActivePanel`. This is the one place in the audit where a previously-recorded bug
(the triplicated panel list) has a guard that survives both a targeted and an
off-target mutation.

## Phase closed

**4 findings** (MIS-E2E-128 … 131), **1 P0**, plus one earlier finding corrected.
Mutations: 2 run, 0 survived. Tree verified clean.

**The one sentence for the synthesis:** the UI's worst defect is a confirmation
dialog rendered from a stale snapshot of the setting the action will actually read —
the product telling the user it is about to do the opposite of what it does.
