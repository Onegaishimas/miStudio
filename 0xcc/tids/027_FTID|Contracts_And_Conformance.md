# TID: Contracts & Two-Track Neuronpedia Conformance

**Document ID:** 027_FTID|Contracts_And_Conformance · **Version:** 1.0

## Pitfalls
1. **Never add a field to a shipped kind.** New kind, new version. The test enumerates fields.
2. **No `alias` on an existing kind.** It renames on OUTPUT and has already invalidated every
   exported document once.
3. **Version lives in the kind identifier**, so an unknown version is REJECTED rather than read as
   an older one with missing fields.
4. **Track A and Track B share nothing.** One object for both couples independent releases.
5. **No ingestion API.** It does not exist upstream; building one builds a feature Neuronpedia
   does not have.
6. **Template-lens fields day-one**, compute optional and absent-when-not-run.
7. **`extra="forbid"` is unsafe with validation aliases** — the combination is what produced the
   earlier republish defect.

## Testing
- Each existing kind round-trips byte-identically.
- No existing kind's field carries an alias.
- A planted field on an existing kind fails (negative control).
- New kinds carry version-in-identifier and a provenance block.
- Template fields exist and are absent-when-not-run.

**Mutation controls:** add a field to `cluster-definition/v1`; add an alias to an existing field;
move the version out of the kind identifier; merge Track A and Track B.
