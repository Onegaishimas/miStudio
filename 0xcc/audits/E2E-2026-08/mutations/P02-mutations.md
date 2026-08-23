# P02 — mutation control log

**Phase:** P02 Backend services · **Round:** 2 · **Date:** 2026-08-23

Same discipline as P01: back up → edit one line → **confirm it landed** → run the
suite → restore → verify `git diff` clean. No reading agent ran concurrently.
`rc=0` means the suite stayed green, i.e. the mutation **survived** = a test finding.

| # | Target | Mutation | Suite | Landed | Result |
|---|---|---|---|---|---|
| M6 | `api/v1/endpoints/settings.py` | Remove `db.expunge(setting)` before masking — the historically-recorded ciphertext-corruption defect | `-k 'setting or encrypt or app_setting'` | ✅ | **KILLED** — 2 failures |
| M7 | `services/app_setting_service.py:22` | Remove `openai_api_key` from `_SENSITIVE_KEYS`, so encryption is no longer forced server-side | same (40 tests) | ✅ | **SURVIVED** → MIS-E2E-077 |
| M8 | `services/steering_service.py:1117` | Regress the hook target from the whole decoder layer to a norm submodule | `-k 'steer or hook or layer'` (84 tests) | ✅ | **SURVIVED** → MIS-E2E-078 |
| M9 | `services/steering_core.py:236` | The **same** regression, on the unified core | `-k 'steer or recorder or calibration or core'` (200 tests) | ✅ | **SURVIVED** → MIS-E2E-078 |

**3 of 4 survived.**

## M6 — the one kill, and it matters

Removing the expunge fix failed
`TestUpsertPreservesEncryptedValue::test_single_upsert_keeps_ciphertext_in_db` and
`::test_re_save_does_not_progressively_truncate`. This is the defect this project
recorded historically — the upsert committing a masked display string back over the
ciphertext — and its fix **is** properly pinned, on both the single and bulk paths.
A previous round's fix that holds up under mutation. Worth stating as plainly as the
failures.

## M7 — a control R1 verified clean, and could not have verified

The `/security-review` pass listed `_SENSITIVE_KEYS` under "verified clean",
describing it accurately: *"forces encryption server-side regardless of the client's
`is_sensitive` flag (blocking a plaintext downgrade)"*. The control is correct. It is
also completely unprotected: drop `openai_api_key` from the set and 40 tests stay
green, after which a client sending `is_sensitive: false` stores the operator's
OpenAI key in **plaintext**.

This is the argument for mutation testing in one example. Reading established the
control exists and is right. Only breaking it established that nothing keeps it
right.

## M8 + M9 — the hardware-only fix is unpinned on both paths

The Recorder increment's headline finding (commit 91b5a6c) was that additive
steering must hook the whole decoder layer — `structure.layers_module[L]`, resid_post
— and **not** the discovered `"residual"` module, which on LFM2 is a post-attention
RMSNorm that renormalises the steering vector away. Failure mode: **steered ==
unsteered at every dial**. Four static review rounds and the entire unit suite missed
it; a hardware round found it.

Regressing that exact target:

- on `steering_service` (the user-facing compare/sweep/combined path) — 84 tests green
- on `steering_core` (the unified core the recorder and calibration use) — 200 tests green

`steering_core.py:229-236` carries a detailed comment explaining precisely why the
RMSNorm target is wrong and that hooking the layer output *"survives, so the recorded
transcript matches what miLLM serves"*. **The trap is documented in prose and
enforced by nothing.** The standing rule — *mutate the previous round's fix; if it
does not fail loudly, that round produced an unpinned fix* — applies exactly.

That both paths behave identically here is itself informative: it is the one steering
property the two implementations (MIS-E2E-076) agree on, and neither tests it.

## Equivalent mutants

None. M7 changes a real storage decision; M8/M9 change the module a hook attaches to.
