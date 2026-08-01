# TDD: Claims Discipline & Evidence-Ladder Integration

**Document ID:** 026_FTDD|Claims_Discipline · **Version:** 1.0 · **Status:** Planned
**Related:** 026_FPRD · PADR IDL-44 · BR-019, BR-020, BR-024

---

## 1. One ladder, extended by mapping — not by a second enum

`EvidenceRung` already exists (`backend/src/schemas/evidence_ladder.py`, mirrored in
`frontend/src/types/evidenceLadder.ts`, pinned to each other by a sync test). J-space does not add
rungs; it MAPS its evidence kinds onto the existing ones.

```
READOUT            -> MINED                    (rung 0) — not a causal claim
PROBE_CROSSING     -> MINED                    (rung 0) — a readout with a threshold
ATTRIBUTION        -> ATTRIBUTION_SUPPORTED    (rung 1)
INTERVENTION+CTRL  -> CAUSALLY_VALIDATED       (rung 2) — the first causal rung
```

A `JSpaceEvidence` enum with its own numbering would be a second ladder. Two ladders means a
reviewer must hold both, and the whole point of the ladder is that one vocabulary spans the product.

## 2. The audit must DISCOVER, not enumerate

`test_causal_language_audit.py` holds a `SURFACES` list. That list was hand-maintained at 5 files
while 16 circuit modules shipped unaudited — the audit was green and the coverage was fiction.

So J-space coverage is derived:

```
jspace_modules = every backend module matching src/**/jlens*.py
              + every frontend file under components/jlens/ and the J-Lens panel
              + the jlens MCP tool module
```

Adding `jlens_watchlist.py` next week puts it in scope automatically. The test that would have
caught the original defect is the one that asks the FILESYSTEM, not the one that reads a list.

## 3. Three audits, three different corpora

| audit | corpus | catches |
|---|---|---|
| CAUSAL LANGUAGE | J-space modules' user-facing strings | "causes", "drives", "makes the model…" on rung-0 evidence |
| ABSENCE CAVEAT | surfaces that report a negative result | a "not found" with no statement of what it does not mean |
| CONSCIOUSNESS | all shipped text incl. MCP descriptions, manual, export metadata | "experiences", "is aware", "feels", "conscious" |

The third has the widest corpus deliberately: the likeliest place for that language is a
well-meaning paragraph in the manual, not a variable name.

## 4. What counts as "shipped text"

UI copy, API and MCP tool descriptions, exported document metadata, and the manual. NOT code
comments and NOT `0xcc/` documents — those explain *why* a rule exists and often must quote the
forbidden phrasing to do so. An audit that cannot tell an explanation from an assertion forces the
explanation out of the codebase, which is how the reasoning gets lost.

## 5. Risks

| risk | mitigation |
|---|---|
| The audit's coverage silently narrows again | coverage is discovered from the filesystem; a test asserts the discovered set is non-empty AND contains known modules |
| Over-matching flags a legitimate explanation | comments and docstrings excluded; only user-facing string literals audited |
| A new surface ships unaudited | discovery, plus a test that fails if a J-space module exports user-facing text and is not reachable by the audit |
| The rung mapping drifts from the circuits ladder | mapping asserted against `EvidenceRung` members, so removing or renaming a rung breaks it |
