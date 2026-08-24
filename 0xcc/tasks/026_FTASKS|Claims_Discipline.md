# Task List: Claims Discipline & Evidence-Ladder Integration

**Document ID:** 026_FTASKS|Claims_Discipline · **Version:** 1.0 · **Status:** ✅ Implemented (2026-08-01) — 6 mutation controls, one survivor fixed
**Related:** 026_FPRD · 026_FTDD · 026_FTID · PPRD §3.27 · PADR IDL-44

## Phase 1: The mapping
- [x] 1.1 `schemas/jspace_claims.py`: J-space evidence kinds → `EvidenceRung`. **No new enum.**
- [x] 1.2 One definition of each required caveat string; duplicates drift.
- [x] 1.3 Assert the mapping against `EvidenceRung` members, so a rung rename breaks it.

## Phase 2: Causal-language audit (BR-019)
- [x] 2.1 J-space coverage DISCOVERED from the filesystem, never listed.
- [x] 2.2 Audit user-facing STRING LITERALS; exclude comments, docstrings and `0xcc/`.
- [x] 2.3 Cover backend modules, frontend jlens components, and the MCP tool module.
- [x] 2.4 Negative control: a planted causal string on a rung-0 surface fails.

## Phase 3: Absence caveat (BR-020)
- [x] 3.1 Surfaces reporting a negative carry the caveat.
- [x] 3.2 Both mechanisms stated: automatic computation, and no single-token name.
- [x] 3.3 No surface claims comprehensive coverage.

## Phase 4: Consciousness audit (BR-024)
- [x] 4.1 Corpus = all shipped text: UI copy, API/MCP descriptions, export metadata, the manual.
- [x] 4.2 Negative control: a planted phrase in the manual fails.
- [x] 4.3 Exclusions asserted, so the corpus cannot silently narrow.

## Phase 5: Verification
- [x] 5.1 Full suite green.
- [x] 5.2 Mutation controls: hard-code the coverage list; audit identifiers; drop the manual;
      downgrade to a warning; duplicate the caveat.
- [x] 5.3 Three review rounds; all findings fixed and re-verified.

**Acceptance:** J-space evidence sits on the existing ladder with no second enum; the audit
discovers its own coverage; a lower rung cannot ship in higher-rung language; negatives carry the
absence caveat; and no shipped text implies subjective experience — each enforced by a failing
build rather than by review.


---

## Review record

**One finding on PRE-EXISTING code, and it cut both ways.** The causal-language audit joined wrapped
lines for docstrings but scanned CONSECUTIVE COMMENT LINES individually. So

    # A concept appearing in a readout is not
    # a causal claim.

was flagged — the second line reads as the bare phrase "a causal claim" — while the identical
sentence in a docstring passed. That is a false positive that pushes authors to reword the very
caveats BR-019 requires until the audit stops objecting, and a false NEGATIVE in the other
direction: an overclaim split so neither line holds a whole claim was invisible. Both are now pinned.

I hit the false positive myself while writing `jspace_claims.py`, and my first instinct was to loosen
the `_DENIAL` patterns. That would have been wrong — the file explicitly warns against loosening
them, and the guard is tight on purpose. Fixed the line-joining instead and reworded one comment.

**One mutation survived its first run** (reverting the comment-joining) because I had reworded
around it rather than testing it. Now pinned by two regressions — a split denial that must pass and a
split overclaim that must fail — and re-verified as a negative control.

## Relevant Files

> **Added 2026-08-24 (MIS-E2E-153).** This file had no `## Relevant Files`
> section, and neither did the four beside it — the five FTASKS with the most
> implementation and the least traceability. The PPRD marked their rows
> "Planned" and `CLAUDE.md`'s Document Inventory had no entry for them at all,
> while their own boxes ran 68–100% checked and the code below exists. A shipped
> feature with no doc→code join is one nobody can review, and the framework
> names this section as the join.
>
> Paths verified to exist at the time of writing; `test_task_docs_traceability`
> fails if one stops resolving.

| File | Purpose |
|---|---|
| `backend/src/schemas/jspace_claims.py` | J-space rung assignment on the existing evidence ladder |
| `frontend/src/config/jspaceClaims.ts` | The frontend's copy of the rung vocabulary and its non-causal phrasing |
| `backend/tests/unit/test_causal_language_audit.py` | Enforces that a readout is never presented in causal language |
