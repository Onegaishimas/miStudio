# P11 — Documentation chain conformance: all three rounds

**Phase:** P11 · **Date:** 2026-08-23
**Deliverable:** `TRACEABILITY.md` (doc chain ↔ code ↔ test matrix)
**Scope:** all of `0xcc/` (197 files), `manual/` (47 pages), `docs/`, `README.md`, `CLAUDE.md`

## R1 — findings (16: MIS-E2E-149…164)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-149** | **P1** | The sentence that cost a real SAE is **still live**, verbatim, in the second manual |
| **MIS-E2E-150** | **P1** | The manual's fix for a startup refusal **removes authentication** from a LAN-bound MCP server |
| MIS-E2E-151 | P1 | Dataset cancel is documented as conservative and deletes **unrelated** tokenizations |
| MIS-E2E-152 | P1 | The K8s install guide's four `sed` steps match nothing — and one renames the database to the password |
| MIS-E2E-153 | P1 | Five shipped features have **no** doc→code traceability by any documented path |
| MIS-E2E-154 | P2 | 22 `Relevant Files` entries point at files that were **never written** |
| MIS-E2E-155 | P2 | CLAUDE.md's instruction references are off by one and point at the **wrong action** |
| MIS-E2E-156 | P2 | IDL-5 is **inverted**, and the error is propagated across five documents |
| MIS-E2E-157 | P2 | IDL-16's three claims about the schema guard are all false |
| MIS-E2E-158 | P2 | IDL-1/IDL-12 document channel and event conventions the code does not use |
| MIS-E2E-159 | P2 | IDL-11's DLQ and backoff do not exist; the exemplar task overrides `acks_late` |
| MIS-E2E-160 | P2 | The PIN gates one tab of five, and not the destructive one |
| MIS-E2E-161 | P2 | MCP docs omit a default-enabled category holding a GPU intervention tool |
| MIS-E2E-162 | P2 | README's startup path cannot work for a fresh clone |
| MIS-E2E-163 | P3 | CLAUDE.md contradicts itself on status, test counts and paths |
| MIS-E2E-164 | P3 | Four smaller: README's panel list, an asserted verification, a phantom env var, IDL-13's consumer |

## The two verified independently, because they are the sharpest

**MIS-E2E-149.** `docs/miStudio_Manual.md:349` reads
`- **Stop:** Gracefully end training (saves final checkpoint)` — the exact sentence
`CLAUDE.md` records as *"factually wrong and… what cost a real run"*. The sibling page
`manual/docs/core-workflow/sae-training.md:183` now carries
`:::warning Stop does not save an importable SAE`. **The fix reached one of two
manuals.** The stale one is indexed in `.understand-anything/knowledge-graph.json`, so
an agent querying the repo's own knowledge graph can be served the uncorrected text.

**MIS-E2E-150.** `server.py:102` is
`if not settings.auth_token and not (settings.allow_anonymous or stdio)` — so
`MCP_ALLOW_ANONYMOUS` **alone** satisfies the guard on the HTTP transport. The manual
calls it *"stdio dev only"*, and so does the guard's own `SystemExit` message. Two
places state a restriction the code does not enforce, on a server the same page says
binds `0.0.0.0` with a bearer token "always required".

## R2 — the adversarial pass

R2 for a documentation phase is not mutation — there is nothing to break whose
failure a suite could observe. The equivalent discipline is **checking the claims the
docs make about being checked**, and re-deriving the recorded divergences rather than
accepting them:

- **MIS-E2E-011 was an undercount.** It recorded 6 contradicting PPRD rows. Re-derived:
  **13**. PPRD §2.1 marks "Planned" *every* feature shipped since 2026-07-19. Corrected
  in `TRACEABILITY.md` §4.
- **MIS-E2E-050 gains a detail.** `data-model.md:9` states the page is *"verified
  against the ORM models"*. It asserts a verification it does not have.
- **MIS-E2E-012 confirmed exactly** — 348 unchecked, 193 in six files, none with the
  join key.
- **MIS-E2E-010 confirmed, plus one more** (`0xcc/research_context.json`), and with a
  severity note: `0xcc/project-specs/reference-implementation/Mock-embedded-interp-ui.tsx`
  is named **five times** in CLAUDE.md, designated `PRIMARY REFERENCE`, and declared
  binding — *"All implementation MUST match the Mock UI specification exactly"*. It was
  deleted in `c9aac7f`. **The authority the project declares binding on all UI work does
  not exist.**

## R3 — verification

All 16 CONFIRMED; none refuted. Every finding was checked at source, and the two
highest-severity were re-verified here independently of the review.

**Coverage recorded for legibility.** IDLs sampled and found **clean**: IDL-19 (OpenAI
SDK, exact), IDL-25 (PIN backend, exact — only the manual's scope claim is wrong),
IDL-26 (architecture clean), IDL-39 (all eight points, including `dry_run` failing to
its default), IDL-20, 22, 23, 42. On the manual side, **all ~140 documented endpoint
rows were diffed against every `@router.*` decorator and no documented method+path is
missing from the backend** — the omissions run the other way.

## The structural cause, and the cheap fix

Three findings share one mechanism: **a correction is applied to the document under
review and not propagated.** The `system_monitor_tasks.py` deletion was fixed in
`008_FPRD` and left standing in five other documents. The Stop sentence was fixed in
`manual/docs/` and left standing in `docs/`. The instruction renumbering was never
reflected in CLAUDE.md at all.

A cross-document grep of the corrected phrase, run at fix time, catches all three.
That is the cheapest durable remediation this phase produces, and it belongs in the
remediation tasklist as a process item rather than as N documentation edits.

## Phase closed

**16 findings** (MIS-E2E-149…164). Deliverable `TRACEABILITY.md` written.

**The one sentence for the synthesis:** the document chain is not stale uniformly — it
is stale exactly where a correction was made, because corrections are applied to the
file under review and the other copies are never grepped for.
