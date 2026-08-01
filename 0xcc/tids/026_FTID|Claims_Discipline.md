# TID: Claims Discipline & Evidence-Ladder Integration

**Document ID:** 026_FTID|Claims_Discipline · **Version:** 1.0 · **Status:** Planned

## Implementation order

1. `schemas/jspace_claims.py` — the evidence-kind → rung mapping and the required caveat strings.
2. Extend the causal-language audit with DISCOVERED J-space coverage.
3. The absence-caveat audit.
4. The consciousness audit over shipped text.
5. Wire the caveats into the surfaces that report negatives.

## Pitfalls

1. **Do not add a new rung enum.** Map onto `EvidenceRung`. A second enum is a second ladder.
2. **Discover coverage; never list it.** The listed-coverage version of this audit shipped green
   over 16 unaudited modules.
3. **Audit STRINGS, not identifiers.** `causal_validation_service` is a module name, not a claim.
4. **Exclude comments and docstrings.** They must be free to quote what they forbid — otherwise the
   explanation leaves the code, which is the outcome this project has been avoiding all arc.
5. **Include MCP tool descriptions.** They are read by agents and are shipped text; the contract
   file is generated from them.
6. **Include the manual.** The likeliest home for a consciousness implication is a friendly
   paragraph, not a variable.
7. **Fail the build, do not warn.** This feature exists because advice does not hold.
8. **The caveat must be ONE definition.** Two copies drift, and the drifted one is the one that
   ships on the surface nobody re-read.

## Testing

- The rung mapping covers every J-space evidence kind and adds no enum member.
- The audit's discovered corpus is non-empty and contains the known J-space modules.
- A planted causal string in a J-space surface fails the audit (negative control).
- A planted consciousness phrase in the manual fails the audit (negative control).
- A "not found" surface without the caveat fails.
- Comments and 0xcc documents are NOT audited (asserted, so the exclusion cannot silently widen).

**Mutation controls:** replace discovery with a hard-coded list; audit identifiers as well as
strings; drop the manual from the consciousness corpus; downgrade a failure to a warning; duplicate
the caveat string.
