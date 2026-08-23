# P02 R2 + R3 — adversarial re-review, mutations, verification & closure

**Phase:** P02 Backend services · **Date:** 2026-08-23

Mutation log: `mutations/P02-mutations.md` (4 run, 3 survived, 1 killed).

## R2 — attacks on R1's "verified clean" list

### 1. "`_SENSITIVE_KEYS` forces encryption server-side" — **UNDERMINED**
R1's `/security-review` listed this under verified clean and described it correctly.
Mutation M7 removed `openai_api_key` from the set: **40 tests green**. The control is
right and nothing keeps it right (MIS-E2E-077). Reading could establish the first
half and never the second.

### 2. "The expunge-before-mask fix is present and correctly ordered" — **HOLDS**
Mutation M6 removed `db.expunge(setting)`:
`TestUpsertPreservesEncryptedValue::test_single_upsert_keeps_ciphertext_in_db` and
`::test_re_save_does_not_progressively_truncate` both failed. A previous round's fix
that is genuinely pinned. Stated as plainly as the failures.

### 3. "Both steering implementations hook the correct target" — **TRUE AND UNTESTED**
R1's `/review` refuted the hypothesis that the hook-target fix reached only one path
— both are correct. R2 then asked the harder question: is either *protected*?
Mutations M8 and M9 regressed the target on each. **84 tests green on
`steering_service`, 200 on `steering_core`** (MIS-E2E-078). The failure mode of that
regression is `steered == unsteered at every dial`, it took a hardware round to find
originally, and `steering_core.py:229-236` documents the trap in prose while nothing
enforces it.

### 4. "The `/bulk` settings path" — R1 flagged this as **unverified**; verified in R2
`PUT /settings/bulk` duplicates the expunge fix (with its explanatory comment) and
**omits the URL validation**. One protection carried across, one not
(MIS-E2E-073). Recorded because R1 explicitly declined to claim it either way.

### 5. "`slug_for` permitting `.` is safe only because every delete site suffixes it"
R1 flagged this as its own weakest clean call. Re-read and **upheld** — every
destructive site builds `f"{slug}.superseded"` / `.staging` / `.swap`, so a `".."`
slug collapses to a harmless `"...superseded"` name, and the one bare-slug `rename`
would fail EINVAL renaming a parent into its own child. Latent, not live. Not filed;
recorded here so R3 of a later phase does not re-derive it.

## R3 — verification

### The `tmp_api` sweep (MIS-E2E-072) — run, and it came back clean
The proposed remediation was "sweep and rotate". The sweep was performed: **27 files
across three labeling runs, 9 Postman collections, zero `Authorization` headers** —
no match for the string anywhere. Those runs used a keyless local endpoint, so the
`if self.api_key and …` guard skipped the header. The code defect is confirmed and
fires the first time a real OpenAI key is used with `save_requests_for_testing`; it
**has not fired on this machine**. The finding stands at P0 on severity-if-triggered
and its urgency is lower than first implied. Recorded rather than quietly left
pending, because a register that overstates is as useless as one that misses.

### Verdicts for P02

| Verdict | Ids |
|---|---|
| **CONFIRMED** | 055, 056*, 062, 063, 069, 070, 071, 072†, 073, 076, 077, 078 |
| **PLAUSIBLE** (read-only, strong) | 057, 058, 059, 060, 061, 064, 065, 066, 067, 068, 074, 075 |
| **REFUTED** | none; one sub-hypothesis refuted inside 076 (the hook-target fix *did* generalize) |

\* 056's mechanism is confirmed; the transmission to OpenAI is traced, not executed.
† 072 confirmed in code, not materialised on disk — see above.

## Phase closed

**24 findings** (MIS-E2E-055 … 078), of which **4 are P0**. Mutations: 4 run, 3
survived. Tree verified clean after every mutation.

**The one sentence for the synthesis:** every P0 in this phase is a guard that
exists, is correctly built, and is not on the path that needs it —
`validate_llm_endpoint_url` (two call sites, neither credential-bearing),
`resolve_user_path` (one caller, not the `rmtree` sites), and the
"never write the bearer token to disk" rule (applied to cURL, not to the Postman
branch sixty lines below it in the same function).
