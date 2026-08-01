# TDD: Contracts & Two-Track Neuronpedia Conformance

**Document ID:** 027_FTDD|Contracts_And_Conformance · **Version:** 1.0

## 1. Additive means the OLD kinds are byte-identical

The guard is not "we intended not to change them". It is a round-trip test over each existing kind
that fails if its serialised shape moves at all.

Two specific mechanisms have already broken this in this codebase and both get a test:

- **A pydantic `alias` renames on OUTPUT.** It republished a schema without its wire field and
  invalidated every exported document. So: no `alias` on any existing kind's field, asserted.
- **miLLM holds a HAND-WRITTEN mirror.** Re-vendoring without updating it silently drops the new
  field. The schema-sync guard catches it, and new kinds must be added there deliberately.

## 2. Version in the kind, not beside it

`mistudio.jlens-artifact/v1`. Version in the identifier means a consumer that does not understand v2
rejects it rather than reading it as v1 with missing fields — which is what a separate `version`
field permits.

## 3. Track A is a directory, Track B is the existing upload path

Track A shares nothing with Track B but a name. Modelling them as one "conformance" object would
couple two things that ship independently and fail independently.

Track A's shape is already implemented by `jlens_artifact_service` (feature 021): `<slug>/`
containing `<slug>_jacobian_lens.pt` and `config.yaml`. This feature adds the CONTRACT DOCUMENT that
describes it, so a consumer can validate before mounting.

## 4. Template-lens fields are day-one

The compute path may be a fast-follow; the FIELDS are not. Adding a field to a shipped kind later is
exactly the change §1 forbids, so the kind carries them from the start with the compute path
optional and absent-when-not-run.

## 5. Risks

| risk | mitigation |
|---|---|
| An existing kind's shape moves | round-trip test per kind; fails on any difference |
| An alias renames a field on output | asserted absent on existing kinds |
| A new field is added to a shipped kind | test enumerates each kind's fields |
| The mirror drifts | schema-sync guard; new kinds added deliberately |
| Template fields retrofitted later | present day-one, absent-when-not-run |
