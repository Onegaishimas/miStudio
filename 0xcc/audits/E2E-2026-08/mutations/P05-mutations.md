# P05 — mutation control log

**Phase:** P05 REST API & schemas · **Round:** 2 · **Date:** 2026-08-23

| # | Target | Mutation | Landed | Result |
|---|---|---|---|---|
| M18 | `api/v1/endpoints/trainings.py:534` | Neutralise the checkpoint-delete parent guard (`if checkpoint.training_id != training_id`) | ✅ | **SURVIVED** (277 tests green) → MIS-E2E-112 |
| M19 | `schemas/cluster_profile.py:184` | `validation_alias=AliasChoices(...)` → plain `alias="sae_id"` — the recorded serialisation trap | ✅ | **KILLED** — both schema-sync tests failed |

**1 of 2 survived.**

## M18 — the API's only IDOR guard is untested

Of the 11 routes with two or more path parameters, ten bind the child to the parent
inside the query itself. The eleventh —
`DELETE /trainings/{tid}/checkpoints/{cid}` — fetches by child id alone and relies
on a single post-fetch comparison, whose comment states the invariant plainly:

```python
if checkpoint.training_id != training_id:
    # Never allow a checkpoint to be deleted via an unrelated training's URL.
```

Neutralising that line left **277 tests green**. The reviewer had flagged it in its
"checked and clean" section as *"correct, but the whole route rests on that one line,
so it is worth a mutation control"* — and the control says nothing holds it there.

## M19 — the alias guard works, and its scope is the finding

Reintroducing the plain-alias trap inside a contract module failed both
`test_circuit_definition_schema_sync.py` and `test_cluster_definition_schema_sync.py`.
The protection is real and it bites.

That is precisely what makes MIS-E2E-107 a **scope** finding rather than a missing
guard: `schemas/metadata.py` carries the same construct and sits outside the swept
set, so the trap the project documented in two places and guarded in one is live in a
third.

## Equivalent mutants

None.
