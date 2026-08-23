# P02 R1 — /security-review

**Phase:** P02 Backend services · **Round:** 1 · **Date:** 2026-08-23
**Scope:** `backend/src/services/` (77 files, 42,682 lines), `backend/src/utils/`
(13 files), `backend/services/ollm_server/` (the unwired second app)

## Findings (7 — four of them P0)

| Id | Sev | Claim |
|---|---|---|
| **MIS-E2E-069** | **P0** | An unauthenticated POST to `/labeling/models/openai` sends the **decrypted OpenAI key** to any host the body names |
| **MIS-E2E-070** | **P0** | `extraction_ids: ["../../../../etc"]` → `rmtree` outside the data root; the delete sits **outside** the DB guard and reports success |
| **MIS-E2E-071** | **P0** | `raw_path`/`file_path` are user-writable and reach `rmtree`; `POST` + `DELETE` destroys `/data` |
| **MIS-E2E-072** | **P0** | The plaintext OpenAI key is written to disk in Postman collections — on the **default** `export_format` |
| MIS-E2E-074 | P1 | `trust_remote_code=True` hardcoded in three services, overriding the user's defaulted-off download choice |
| MIS-E2E-073 | P2 | `PUT /settings/bulk` skips the URL validation the single-key route applies |
| MIS-E2E-075 | P2 | `judge_endpoint` unvalidated; the calibration manifest reserves a slot for a credential |

Every P0 was re-read at source before recording. Three of the four are two-request
exploits requiring no special conditions.

## The pattern across the four P0s

They are not four unrelated bugs. Each is a **guard that exists and is not on the
path**:

- `validate_llm_endpoint_url` exists and has two call sites; the credential-bearing
  path is not one of them (069), and neither is `/settings/bulk` (073) or
  `judge_endpoint` (075).
- `resolve_user_path` exists, is correctly built, and has **one** production caller;
  the `rmtree` sites use `resolve_data_path`, which does not contain (070, 071) —
  the same gap MIS-E2E-006 recorded before its consequences were known.
- The "never write the bearer token to disk" rule exists as an implemented fix and
  a comment; the branch sixty lines below in the same function does not follow it
  (072).

The remediation that actually closes this class is not seven fixes. It is moving
each guard from the caller into the thing being guarded — validation into
`AppSettingService.upsert`, containment into the deletion sink, key-attachment into
a host allow-list.

## Verified clean — R2 must attack these

**SSRF call-site sweep.** Every HTTP client in the audited paths was traced to its
URL's origin: `schemas/labeling.py:167` genuinely validates and the whole labeling-job
path inherits it; `enhanced_labeling.py:112` uses a stored value that passed
validation on write; `utils/millm_utils.py` re-uses an already-validated URL;
`background_monitor.py:185` uses config, not input; `utils/hf_utils.py:170` builds
from a module constant with `repo_id` gated by an anchored regex, and the HF token
only ever goes to `huggingface.co`; `neuronpedia_local_service.py` has no HTTP client
at all. DNS rebinding was considered and rejected — the validator permits every
private range anyway, so rebinding buys nothing.

**Credential handling.** `AppSettingService` is correct: `_SENSITIVE_KEYS` forces
encryption server-side regardless of the client's flag (blocking a plaintext
downgrade), and all three read paths `expunge` before masking. Every `logger.*` in
`src/services/` was grepped for key/token/secret/password — **no call interpolates a
credential value**. No credential appears in any `HTTPException` detail read.
`resolve_hf_token` returns `None` rather than `""`. The only place a decrypted secret
leaves the process other than an `Authorization` header is MIS-E2E-072.

**Filesystem — 27 sites enumerated and traced.** Clean:
`training_finalize_service.py` (`_prune_stale_layer_dirs` only deletes children
matching `_LAYER_DIR_RE`), `circuit_capture_service.py` (all six sites on
generated-id dirs), `checkpoint_service.py:286` (`rmdir` correctly guarded on
`not any(dir.iterdir())`), `neuronpedia_export_service.py:649`,
`jlens_artifact_service.py` (every destructive site **suffixes** the slug, so a
`..` slug collapses to a harmless `"...superseded"` name). `sae_manager_service.py:543`
correctly uses `resolve_user_path`. **`file_utils.delete_directory` and
`is_safe_path` have zero production callers** — dead code, so their weaknesses (no
containment; `Path.resolve()` follows symlinks) are unreachable today.

> The reviewer flagged its own weakest clean call: `slug_for` permits `.`, so
> `slug_for("x/..")` returns `".."`. It is safe **only** because every delete site
> suffixes it. R2 should treat that as a latent hazard, not a settled clean.

**Export / manifest writers.** `manifest_service.validate_payload` +
`_assert_no_paths` walks to depth 12 rejecting `/data/`- and `/home/`-prefixed
strings, with a correctly-scoped `_TEXT_KEYS` exemption that still walks a dict
hiding under `prompt`. The J-Lens publish path is tight: `PUBLISHED_FILES` is a
two-name allow-list, `validation.json`/`acquisition.json` are deliberately excluded,
and `interventions.json` records `prompts_sha256` rather than prompt text, with an
explicit comment saying user prompts must not reach HuggingFace. The Neuronpedia
export carries no credentials and no absolute paths. Steering-sample manifests do
carry prompts and generated text — that is the recorder's documented contract
(BRD-MIS-RECORDER-001) and they stay in the DB.

**The unwired `ollm_server`.** Verified genuinely not run: no compose service
(`docker-compose.yml:336` is a comment recording its removal), no `k8s/` reference,
no CI reference. Only `stop-mistudio.sh:88` still tries to stop a `mistudio-ollm`
container. Its `allowed_origins` is a **fixed three-entry allow-list**, not `"*"`.
**But note for P10:** `inference.py:287,318` does `trust_remote_code=True` on a model
name taken straight from an unauthenticated `POST /api/pull` body — RCE by design,
unreachable only because nothing deploys it. If any deployment of this app is found,
that is a P0, not the CORS setting.

**subprocess.** `nlp_analysis_service.py:56` is the only call in the audited paths.
argv is the fixed list `[sys.executable, "-m", "spacy", "download", "en_core_web_sm"]`
— no `shell=True`, no interpolation, 300s timeout. Clean.

**Deserialisation.** All three `torch.load` sites pass `weights_only=True`. No
`pickle.load`, `joblib.load` or `yaml.load` anywhere. Clean.
