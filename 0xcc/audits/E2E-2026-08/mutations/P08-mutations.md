# P08 — mutation control log

**Phase:** P08 Frontend UI · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M23 | `frontend/src/config/panels.ts` | Remove `'jlens'` from `PANEL_IDS` | ✅ | **KILLED** — `App.jlens.test.tsx > restores the panel from localStorage on reload` |
| M24 | `frontend/src/config/panels.ts` | Remove `'circuits'` from `PANEL_IDS`, full suite | ✅ | **KILLED** — `Sidebar.jlens.test.tsx > panel registry > routes every nav entry` |

**0 of 2 survived.**

## The panel registry is properly guarded — and my hypothesis about it was wrong

M23 killed, which was expected: the registry exists precisely because the panel list
was once triplicated and a missed third copy produced a panel that worked when
clicked and vanished on reload. That bug is pinned.

M24 was the follow-up, and I expected it to survive. The pattern established in P04
(a janitor fix tested by `test_cleanup_stuck_circuit_runs_pending.py`, named for one
janitor) and P06 (`EXPECTED_CALLS`, a hand-list of 16) suggested the same shape here:
a test file named `App.jlens.test.tsx` should cover jlens and nothing else.

**It does not.** Removing a *different* panel id failed
`Sidebar.jlens.test.tsx > panel registry > routes every nav entry — a nav id outside
PANEL_IDS is dropped on reload`, which despite its filename iterates **every** nav
entry. The coverage is registry-derived and general.

Recorded as a refutation rather than quietly dropped: the file naming reads
panel-specific and the assertion inside is not, and that distinction is exactly what
the P04 and P06 findings turned on. Here the naming misleads in the *safe* direction.

## Equivalent mutants

None.
