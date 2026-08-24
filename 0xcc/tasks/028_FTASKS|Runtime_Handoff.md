# Task List: Runtime Handoff — Watchlists, MCP Parity & the Cost Envelope

**Document ID:** 028_FTASKS|Runtime_Handoff · **Version:** 1.0
**Status:** ⏳ Core implemented (2026-08-01) — 4 mutation controls verified biting

## Phase 1: Watchlists (BR-025)
- [x] 1.1 A watchlist is a DETECTOR DEFINITION: directions, thresholds and the scoring definition
      travel together or none of them mean anything.
- [x] 1.2 Missing scoring definition REFUSED at construction — a threshold applied to a
      differently computed score is a different detector and the consumer cannot notice.
- [x] 1.3 Missing artifact reference REFUSED — lens coordinates are artifact-specific.
- [x] 1.4 An empty watchlist is refused: it would export cleanly and detect nothing.

## Phase 2: Evaluation-awareness score (BR-026)
- [x] 2.1 The score is a DIFFERENCE; the subtraction lives inside the function, because a caller
      who must remember to subtract will eventually not.
- [x] 2.2 A missing control is refused rather than treated as zero — that silently yields the raw
      mean, which is high for common tokens in any prompt.
- [x] 2.3 Layers average the DIFFERENCES, not the means; they coincide only when every layer
      contributes equally.
- [ ] 2.4 Ship the validated reference watchlist itself (needs a fitted artifact).

## Phase 3: Cost envelope (BR-028)
- [x] 3.1 An estimate for every operation class, each carrying its BASIS.
- [x] 3.2 An unknown class RAISES — a cheap-looking default invites the run it should warn about,
      and an agent cannot tell "cheap" from "unmeasured".
- [x] 3.3 Labelled order-of-magnitude; false precision invites planning against a number nobody
      measured.
- [x] 3.4 The intervention estimate INCLUDES its mandatory control, so a run is never priced at half.

## Phase 4: Outstanding
- [ ] 4.1 MCP tools for annotation, interventions, watchlists + reachability (BR-027 full parity).
- [ ] 4.2 Surface estimates before a run in the UI and in tool descriptions.
- [ ] 4.3 Review rounds 2 and 3.

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
| `backend/src/mcp_server/tools/jlens.py` | The J-space MCP surface — full parity with the workbench |
| `backend/src/api/v1/endpoints/jlens.py` | REST surface for readout, artifacts, band report and gate |
| `backend/tests/unit/test_jlens_reachable_mcp.py` | Reachability harness: a tool is not shipped until removing its registration fails |
