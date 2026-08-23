# P02 R1 — /code-review high

**Phase:** P02 Backend services · **Round:** 1 · **Date:** 2026-08-23
**Target:** `core/encryption.py`, `core/config.py`, `services/app_setting_service.py`,
`api/v1/endpoints/settings.py`, `services/labeling_service.py`

Scoped to the credential/secrets slice first, because P02 is 42,682 lines and this
is where the product's only stored credentials live. The remaining services are
covered by later passes in this round.

## Findings (7 register entries, 10 defects)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-055** | **P0** | The Settings PIN can be **read, rewritten and deleted** through the generic settings routes it exists to guard |
| **MIS-E2E-056** | **P0** | After a key change, `decrypt_value`'s fail-open sends **raw ciphertext to api.openai.com as a bearer token** |
| MIS-E2E-057 | P1 | Cancelling a labeling job can never be observed — identity-mapped read on `expire_on_commit=False` |
| MIS-E2E-059 | P1 | The documented no-template fallback raises `UnboundLocalError` at three sites |
| MIS-E2E-058 | P2 | A cancelled job is written FAILED, under a comment asserting it is CANCELLED |
| MIS-E2E-060 | P2 | Per-job `max_tokens` is silently overwritten by the template's default of 50 |
| MIS-E2E-061 | P3 | Three one-liners: `finally` masks the real error; `"batch_size": null` → `TypeError`; `mask_value` reveals 4–7 char secrets whole |

## The two P0s, verified at source

Both were re-read line by line rather than taken from the review, because a P0 that
turns out to be wrong is worse than one not found.

**MIS-E2E-055.** Three independent bypasses, all confirmed:
`_hash_pin` output is stored `is_sensitive=False` (`:100`) and masking is conditional
on that flag, so `GET /settings/settings_pin_hash` returns the PBKDF2 salt+hash in
the clear; `PUT /settings` (`:138`) validates only membership in a two-element
`_URL_VALIDATED_KEYS` set and will upsert the PIN key like any other; `DELETE
/settings/{key}` (`:195`) removes it, after which `/pin/set`'s `if existing and not
bypass` guard at `:88` is skipped entirely.

This is **not** absorbed by the accepted no-app-auth posture. The PIN's whole threat
model is to gate the credential panel from someone who already has network access —
exactly the population nginx admits. Route (1) needs no write at all.

**MIS-E2E-056.** This is the consequence MIS-E2E-004 recorded the mechanism for but
did not trace: the fail-open on `InvalidTag` means a rotated or regenerated key turns
every stored API key into base64 ciphertext that is then transmitted to a third party
in an `Authorization` header. Three findings now root in that one swallow.

## Verified clean — R2 must attack these

- **The masked-value-over-ciphertext trap is genuinely fixed.** `upsert_setting:156-159`
  expunges the row from the session *before* mutating `setting.value` for the
  response, with a comment naming the exact failure it prevents. This is the defect
  this project recorded historically; the fix is present and correctly ordered.
- **`_verify_pin` uses `hmac.compare_digest`** and PBKDF2-SHA256 at 600,000
  iterations with a random per-PIN salt. The primitive is well built — every problem
  in MIS-E2E-055 is about where the output is *stored*, not how it is computed.
- **URL validation is wired on the two endpoint keys** (`ollama_url`,
  `openai_compatible_endpoint`) in `PUT /settings`. R2 should check whether
  `PUT /bulk` applies the same validation — this round did not.
- **`encrypt_value` is correct**: AESGCM, a fresh 96-bit `os.urandom(12)` nonce per
  call, `b64(nonce‖ct‖tag)`. Only decryption is at fault.

## Not covered by this pass

The other 72 service modules, including the six largest untested ones
(`steering_service.py` 2,993, `extraction_service.py` 2,116,
`jlens_acquire_service.py` 1,180, `nlp_analysis_service.py` 1,109,
`circuit_capture_service.py` 1,075, `neuronpedia_local_service.py` 1,043) —
MIS-E2E-015. Those are the next passes in this round.
