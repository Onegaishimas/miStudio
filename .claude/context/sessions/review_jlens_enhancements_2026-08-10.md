# Session: J-Lens enhancement arc — three-round multi-agent review

**Session ID:** review_jlens_enhancements_20260810
**Date:** 2026-08-10
**Type:** review
**Scope:** `5903930..1344e36` — coordinate_swap, ranked side-by-side readouts,
Steer/Swap on a token, layer range, interventions.json portability, MCP parity

## Session Context

**Working On:** BRD-MIS-JSPACE-001, the J-Lens readout viewer and the rung-2
intervention surface behind it.
**Phase:** Post-implementation review of the enhancement arc.
**Mode:** comprehensive — three rounds, four agent perspectives, mutation
controls on every fix.

## Session Goals

1. Three rounds of review, each round's findings fixed before the next begins.
2. Every fix pinned by a test that FAILS when the fix is removed, with the edit
   confirmed to have landed before concluding anything about a mutation.
3. The enhancements reachable from MCP, not only from the browser.

## Headline

**40 findings across three rounds. 33 mutation controls verified red.**

Two of the findings mean previously recorded evidence was *wrong* rather than
merely missing, and both were in code that had already passed a review round:

- **The control was not norm-matched (BR-018).** `build_control` returns
  unit-norm random directions; the intervened arm used a raw unembedding row,
  whose norm varies several-fold across tokens. An additive run pushed
  `strength·‖W_U[t]‖` against a control pushing `strength·1`, and the two were
  compared as though the only difference between them were semantic — under a
  report reading *"against a matched-norm random control"*. On a token with a
  large row, the intervals separate on magnitude alone.
- **One prompt could never separate.** Below four trials no outcome separates:
  a perfect intervened arm against a perfect null control still produces
  overlapping Wilson intervals. Both UI paths sent a single prompt, so *"no
  effect was demonstrated here"* was the only verdict the product could ever
  produce — reported as a fact about the direction when it was a fact about the
  sample size.

## Round-by-round

### Round 1 — Architect (6 findings, 7 mutation controls)

A task that died *after* its first progress report sat at "running 42%" on an
idle GPU forever. The previous round had taught the janitor to sweep `queued`
rows; it deliberately never closes `running` ones, because `looks_abandoned`
returns False for a terminal Celery state — a finished task is not an orphan. So
the fix for the easier half read as complete. Tasks now own their own failure
and carry the real reason.

Also: `_recipe_key` ignored `target_token` and `positions`; `interventions.json`
was written non-atomically; `primitive` was free text, so a typo took a GPU slot
behind a possible 45-minute fit; `fullSpan` was not persisted beside the
`layerRange` it bounds.

### Round 2 — QA + Product + Test Engineer (24 findings, 19 mutation controls)

The two headline findings above, plus:

- A one-click one-prompt Steer had the same recipe key as a 50-prompt agent run
  and **deleted** it from the file built to travel to HuggingFace.
- `prompts` entries had no length bound and the worker budgeted against
  `prompt`, which it discards whenever `prompts` is present.
- Duplicate `layers` registered duplicate hooks, each perturbing the output of
  the one before: `[9,9,9]` at strength 1.0 applied 3.0 and recorded 1.0.
- `restore_superseded` parked a live artifact under `.swap`, a suffix discovery
  did not skip, and `rmtree`'d it on the next call — the recovery operation
  destroyed what it was recovering.
- `InterventionCard` read three keys the task had not returned since the rung-2
  rewrite, so its SUCCESS path called `.toFixed` on `undefined` and took the
  panel down — **on success, and only on success**, which is why nothing noticed.
- A ranked click hooked every layer the token appeared at: 26 on gemma for a
  common token.
- A logit-lens click credited the Jacobian artifact.

**Nine test findings**, each a load-bearing line no test protected. The worst:
the layer-range picker could be unmounted entirely with 143/143 green, because
its own test renders the component directly and the store test calls `setState`
— the exact shape of the 16 MCP tools that shipped unregistered.

### Round 3 — reconciliation (10 findings, 11 mutation controls)

Aimed at rounds 1 and 2's *fixes*. **Both headline fixes survived their
mutations.** Deleting the unit-norm scaling left 63 tests green: the fixture's
`W_U` is `torch.eye(VOCAB, D_MODEL)`, every row of which already has unit norm,
so normalising and not normalising are the same operation on it — *the
fixtures-agree-by-construction trap, inside the test written to pin the fix for
it*. Replacing `separation_attainable()` with `return True` was also green,
because the frontend tests that appeared to cover it **mock** the value.

The three-state verdict went to the panel and not to the card, so the card kept
printing the sentence the change exists to remove — one commit after writing
"fixed one representative, never generalized" into the record. And the new
message said *"add prompts"* when no UI could send any: `JLensRequest.prompts`
existed and neither call site populated it.

## Agent Status

**Product Engineer:** Found the claim/behaviour divergences — the rung-1 badge
over a rung-2 measurement, `artifact_id` documented as changing the experiment
when it only changes provenance, and the n=1 dead end that made the user's
stated goal ("experiment with them to determine how they can be used in causal
influence") unreachable from the product.

**QA Engineer:** Found the norm mismatch, the unbounded `prompts` DoS, the
duplicate-hook multiplication, and the `.swap` destruction. Strongest lens on
this arc.

**Architect:** Found the task-ownership gap and the record-integrity defects.

**Test Engineer:** Found nine unprotected lines by mutating rather than reading,
including three whose stated MUTATION CONTROL comments were false as written.

## Durable lessons

1. **A fix is the most dangerous code in a review.** Round 3 existed only to
   re-review rounds 1 and 2, and it found that both headline fixes were
   unprotected and one was only half-applied. Without it the arc would have
   shipped believing the opposite.

2. **A test written to pin a fix can inherit the very trap the fix addresses.**
   The unit-norm test could not fail, because the fixture's `W_U` was already
   unit-norm. When writing a regression test, ask what fixture makes the two
   behaviours *differ*, and assert that it does.

3. **A source-scrape guard fails OPEN.** The `owns_its_failure` reachability
   check used a regex allowing at most one decorator between `@celery_app.task`
   and `def`; a task carrying a second decorator matched nothing, never entered
   the checked list, and would have shipped green. The same shape appeared in
   the atomicity test (`".replace(" in inspect.getsource(...)`). Read the live
   registry, or kill the operation and assert what survived.

4. **Verify a mutation LANDED before concluding it survived.** Several
   candidate mutations here did not apply — wrong anchor text, a value added to
   a `Literal` rather than the constraint removed — and each would have been
   recorded as a passing capability.

5. **Never run a mutating agent concurrently with a reading one.** A reviewer
   read `return false && all.length ? (` out of the working tree and reported it
   as a committed defect; it was another agent's in-flight mutation.

6. **A statistic can have a floor below which its own question is unanswerable.**
   `separated_from_control` cannot be true below four trials. Reporting the
   false as a null was arithmetically guaranteed and read as a finding.

## Suites

| | Before | After |
|---|---|---|
| Backend unit | 2426 | **2461** |
| Frontend | 1134 | **1149** |
| tsc | clean | clean |

Mutation controls: **33**, each verified to turn the suite red with the edit
confirmed landed. One equivalent mutant recorded and not chased (`>=` for `>` in
`separation_attainable` — the two Wilson bounds are never exactly equal).

## Commits

- `9e436e3` — tasks own their own failure; record-integrity fixes (round 1)
- `e6f7fdd` — norm-matched control; the four-trial floor; 24 findings (round 2)
- `d65994c` — both headline fixes were untested; the card kept the old verdict (round 3)
- `1344e36` — MCP parity: four tool descriptions that were no longer true

## Outstanding

- **Hardware acceptance:** a real Swap on k8s writing `primitive: coordinate_swap`
  into `interventions.json` for a run that actually swapped. Not yet done.
- `linearisation_residual_mean/max` in `config.yaml` is populated by a
  source-position spread, not by `linearisation_residual()`, which has zero
  callers. Pre-existing; recorded, not fixed.
- `bandReport` is dead state; the band-report API is unwired, so every band
  branch is unreachable. Nine other backend routes have no client function.
- gemma's L12–L17 junk band remains unexplained.
