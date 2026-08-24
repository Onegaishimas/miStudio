# Task List: Dictionary Annotation & Weight-Space Readouts

**Document ID:** 024_FTASKS|Dictionary_Annotation · **Version:** 1.0
**Status:** ⏳ Core implemented (2026-08-01) — 5 mutation controls verified biting

## Phase 1: Projection & fields
- [x] 1.1 Project a weight-space direction via the READOUT's `LensTransport`, never a second path.
- [x] 1.2 GEOMETRIC field: excess kurtosis of the projected vocabulary distribution.
- [x] 1.3 BEHAVIOURAL field, independent of it — `UNKNOWN` without a band report, never guessed.
- [x] 1.4 `is_j_aligned` requires BOTH fields; nothing downstream can rebuild it from kurtosis.
- [x] 1.5 Absent, never zero, for every optional field.

## Phase 2: Disagreement queue (BR-013)
- [x] 2.1 A sortable SCORE, not just a flag — a flag gives a reviewer no place to start.
- [x] 2.2 Filterable `has_disagreement` alongside it.
- [x] 2.3 Nothing to compare is NOT disagreement.
- [x] 2.4 Never auto-resolved: the lens is rung 0 and does not overrule a label.

## Phase 3: Distributional shape check (BR-014)
- [x] 3.1 Motor features EXCLUDED from the denominator, per the published finding.
- [x] 3.2 An implausible sweep is a RESULT, returned as a verdict rather than raised.
- [x] 3.3 `UNKNOWN` reported, never folded into a measured bucket.
- [x] 3.4 An empty sweep is refused: "0 aligned out of nothing" reads as a finding.

## Phase 4: Outstanding
- [ ] 4.1 Weight-space readouts for transcoder ENCODER and DECODER as separate surfaces (BR-015).
- [ ] 4.2 Attention Q/K/V/O readouts.
- [ ] 4.3 Persistence + the queue as a filter over the existing feature list.
- [ ] 4.4 Sweep as a bounded, queued job with a cost estimate (BR-028).
- [ ] 4.5 MCP tools + reachability harness.
- [ ] 4.6 Review rounds 2 and 3.

**Acceptance:** a motor feature is never classified workspace on kurtosis alone; the behavioural
field is absent without a band report; annotation and readout share one projection; disagreement is
a sortable, filterable queue that resolves nothing automatically; and a sweep calling most features
workspace is reported implausible.

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
| `backend/src/services/jlens_annotation.py` | Dual geometric/behavioral classification of dictionary features through the lens |
| `backend/src/services/jlens_watchlist.py` | Named concept sets with detection thresholds — the runtime handoff artifact |
| `backend/src/schemas/jlens.py` | Annotation and watchlist request/response shapes |
