# Task List: Contracts & Two-Track Neuronpedia Conformance

**Document ID:** 027_FTASKS|Contracts_And_Conformance · **Version:** 1.0
**Status:** ⏳ Contract kinds implemented (2026-08-01) — 4 mutation controls verified biting

## Phase 1: Additive kinds (BR-021)
- [x] 1.1 Four new kinds: artifact, workspace annotation, readout record, watchlist.
- [x] 1.2 Version in the KIND IDENTIFIER, so an unknown version is rejected rather than read as an
      older one with missing fields.
- [x] 1.3 `Literal` on `kind`, so a wrong kind is refused rather than defaulted.
- [x] 1.4 **No aliases anywhere** — an alias renames on OUTPUT and once republished a schema
      without its wire field, invalidating every exported document.
- [x] 1.5 Existing shipped kinds asserted untouched.
- [x] 1.6 Provenance travels with every kind; per-layer applicability rather than a model-level claim.

## Phase 2: Two independent tracks (BR-022)
- [x] 2.1 Track A describes a MOUNTED directory and carries no weights — there is no ingestion API
      upstream, so a document embedding tensors describes a transfer nobody performs.
- [x] 2.2 Track A and Track B are separate kinds; one object would couple independent releases.
- [x] 2.3 Absent validation means NOT RUN, matching the suite's own fail-closed rule.

## Phase 3: Template lens (BR-023)
- [x] 3.1 Contract fields DAY-ONE, compute path optional.
- [x] 3.2 Uncomputed is ABSENT, not `False` — `False` reads as "no direction exists".

## Phase 4: Outstanding
- [ ] 4.1 Export/import services for the new kinds + round-trip tests.
- [ ] 4.2 miLLM mirror update + schema-sync guard extension.
- [ ] 4.3 Template-lens compute path.
- [ ] 4.4 Review rounds 2 and 3.

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
| `backend/src/schemas/jspace_contracts.py` | Additive interchange kinds for the lens artifact, annotation and watchlist |
| `backend/src/services/jlens_artifact_service.py` | Stage → validate → commit → serve for a mounted artifact |
| `docs/schemas/` | The frozen interchange schemas these validate against |
