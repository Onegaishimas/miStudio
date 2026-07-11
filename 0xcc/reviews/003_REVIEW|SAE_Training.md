# Feature Review: 003 — SAE Training

**Reviewed:** 2026-07-11
**Reviewer:** deep code review (doc↔code accuracy + bug/correctness + quality/tech-debt)
**Feature docs:** [FPRD v2.0](../prds/003_FPRD|SAE_Training.md) (2026-03-21), FTDD, FTID
**Verdict:** The **ML core is the strongest code in the app so far** — all 6 SAE architectures are paper-grounded and correct, including the subtle JumpReLU STE and count-based L0. No P0/P1 *bugs*. Findings are concentrated in **input validation gaps**, **internal doc-drift in the ML code**, and the usual FPRD reference staleness. The two training-loop bugs I flagged in the earlier Celery review (F5 pause-race, F6 per-step DB query) are **already fixed** in the committed code.

---

## 1. Scope

**Backend read in full:** `ml/sparse_autoencoder.py` (1484 — all 6 architectures + STE autograd fns), `services/training_service.py` (708, key methods), `api/v1/endpoints/trainings.py` (387), `models/training.py` (118), `models/training_metric.py` (72), `schemas/training.py` (443, key sections). `workers/training_tasks.py` (2000) — reviewed in depth earlier this session (Celery review) + re-verified the loop's status/throttle fixes and hook-type handling here.
**Frontend checked:** `config/frameworkConfigs.ts` (250), `stores/trainingsStore.ts` (retry path), `types/training.ts` (status enum), `TrainingCard.tsx` (status coverage). `TrainingPanel.tsx` (2081) and `TrainingCard.tsx` (1414) confirmed present but not read line-by-line (very large; UI).
**Tests present (strong):** `test_training_service`, `test_training_tasks`, `test_training_schemas`, `test_training_validator`, `test_checkpoint_service`, `test_sae_converter`, `test_multi_sae_import`, `test_checkpoint_deletion` (×2), integration `test_training_workflow`.

---

## 2. Doc ↔ Code Accuracy

FPRD v2.0's §3 (the 6 frameworks) is **excellent and accurate** — the math matches the code. But the mechanical reference sections drift as usual:

### 2.1 Key files — 2 wrong

| PRD claims (§9) | Reality |
|---|---|
| `ml/jumprelu_sae.py` | **Does not exist.** All 6 architectures (Standard, Skip, Transcoder, TopK, JumpReLU + the STE autograd `Function`s) live in the single `ml/sparse_autoencoder.py` (1484 lines). |
| `components/training/StartTrainingModal.tsx` | **Does not exist.** The start-training modal is embedded in `TrainingPanel.tsx` (2081 lines). |

### 2.2 Data model — PK type, status enum, metric field names

| PRD (§7) | Reality |
|---|---|
| `trainings.id UUID` | `String` = `train_{uuid[:8]}`. |
| status `pending, running, completed, failed, stopped` | Enum = `pending, initializing, running, paused, completed, failed, **cancelled**` (no `stopped`; adds initializing/paused/cancelled). |
| `layer INTEGER`, single `dataset_id` | Real: `training_layers` + `dataset_ids` (JSONB arrays) + `extraction_ids` (JSONB), plus `total_steps`, `celery_task_id`, `checkpoint_dir`, `logs_path`. |
| `training_metrics`: `l0`, `l1`, `reconstruction_loss`, `id UUID`, `UNIQUE(training_id, step)` | Real columns: `l0_sparsity`, `l1_sparsity`, `loss_reconstructed`, `id BigInteger autoincrement`, **no unique constraint on (training_id, step)**, plus `layer_idx`, `grad_norm`, `gpu_memory_used_mb`, `samples_per_second`, `fvu`. |
| `training_templates.usage_count` | (Templates are a separate feature file; not verified here.) |

### 2.3 API endpoints — control consolidated, "retry" isn't an endpoint

| PRD (§6) | Reality |
|---|---|
| `POST /{id}/stop` | Actual: `POST /{id}/control` with `{action: pause\|resume\|stop}`. No separate `/stop`. |
| `POST /{id}/retry` | **No retry endpoint exists.** Frontend "retry" ([trainingsStore.ts:422](../../frontend/src/stores/trainingsStore.ts#L422)) simply re-POSTs a new training with the copied config. Functionally fine, but the documented endpoint is fictional. |
| (not in PRD) | Real extras: `PATCH /{id}`, `GET /{id}/checkpoints/best`. |

### 2.4 Hook-type naming mismatch

PRD §2.7 FR-7.5 lists hook types `residual, mlp_out, attn_out`. The schema Literal ([training.py:52](../../backend/src/schemas/training.py#L52)) and training loop use `residual, mlp, attention`. Code is source of truth; **PRD names are wrong**.

### 2.5 WebSocket channels — naming (same pattern as 001/002)

PRD `training/{id}` + events `progress/completed/failed/checkpoint`; real channel `trainings/{id}/progress` + namespaced `training:progress` etc.

**Recommendation:** FPRD 003 §3 (frameworks) can stay — it's good. Rewrite §6/§7/§2.7 hook-names/channels. Note the single-file architecture (no `jumprelu_sae.py`).

---

## 3. Findings (severity-ranked)

*(No P0/P1 bugs. The earlier-session Celery review's P1 training-loop findings — F5 pause/cancel race, F6 per-step DB query — are confirmed **fixed** in the committed code: guarded status write at [training_tasks.py:85](../../backend/src/workers/training_tasks.py#L85), throttled check `status_check_interval = min(25, max(1, log_interval))`.)*

### P2 — Correctness / robustness

**F1. `create_training` performs no existence/compatibility validation**
[training_service.py:36-94](../../backend/src/services/training_service.py#L36) — creates the Training row directly from the request: no check that `model_id` exists, that `dataset_ids`/`extraction_ids` resolve, or that tokenizations match across datasets. FR-6.5 ("Validate all datasets have matching tokenizations") is marked **Implemented** in the PRD, but I found no such validation in the create path — the validation, if any, happens later in the Celery task (which then fails the whole job at runtime instead of rejecting at API time). A user pointing a training at a missing/mismatched dataset gets a queued job that dies mid-run rather than a 400. (There is a `test_training_validator.py` — worth checking whether that validator is actually wired into `create_training`; from the service code it is not.)
**Fix:** validate FKs + tokenization compatibility in `create_training` and return 400 on mismatch.

**F2. `TrainingMetric` has no `UNIQUE(training_id, step)` constraint**
The PRD claims it (§7.2) and it's the natural key, but the model ([training_metric.py](../../backend/src/models/training_metric.py)) only indexes `training_id` and `step` separately. With multi-hook training writing per-`layer_idx` metric rows at the same step, plus retries/resumes re-emitting steps, duplicate `(training_id, step, layer_idx)` rows can accumulate silently, skewing charts. Either add a composite unique constraint (with `layer_idx`) or document that duplicates are expected and dedup on read.

### P3 — Quality / tech-debt

**F3. `JumpReLUSAE.__init__` docstring describes the OLD (wrong) L0 formulation.**
[sparse_autoencoder.py:974-976](../../backend/src/ml/sparse_autoencoder.py#L974) says `sparsity_coeff` is "Applied to L0 fraction (active features / d_sae)... Default 0.4... Typical range: 0.1 to 1.0." The **actual default is `1e-3`** and the code correctly uses **count-based** L0 (`l0_differentiable.sum(dim=-1).mean()` at line 1234, matching Gemma Scope). This is the exact fraction→count fix from Feb 2026 — the docstring was never updated. Dangerously misleading for anyone tuning `sparsity_coeff` (off by ~d_sae in scale). **Fix the docstring to match the count-based reality** (λ ≈ 6e-4–1e-3).

**F4. Three near-identical `normalize()`/`denormalize()` implementations.** `SparseAutoencoder`, `TopKSAE`, and `JumpReLUSAE` each re-implement the same `constant_norm_rescale` / `anthropic_rescale` / `none` logic verbatim. Extract to a shared mixin or module function — three copies will drift (they already differ trivially: base uses `x[:, :1]` for the `none` ones-tensor, JumpReLU uses `x[..., :1]`).

**F5. `SparseAutoencoder.forward` computes reconstruction loss in *denormalized* space** ([line 225](../../backend/src/ml/sparse_autoencoder.py#L225): `F.mse_loss(x_reconstructed, x)`) while the L1 penalty is on normalized-space `z`. This is defensible (matches SAELens which reports recon in raw space) but means the recon/sparsity trade-off `l1_alpha` operates across two scales; worth a one-line comment noting the intent, since it's easy to "fix" mistakenly. (Not a bug — flagging so a future reader doesn't break it.)

**F6. `Training.model_id` is `ondelete="RESTRICT"`** but `ModelService.delete_model` (Feature 002) deletes activation extractions and the model without checking for trainings that reference it. A delete of a model still used by a training will raise a raw FK IntegrityError → 500 rather than a friendly 409. Cross-feature (002+003) — worth a pre-check in model delete.

**F7. `control_training` f-string bug in error detail.** [trainings.py:271](../../backend/src/api/v1/endpoints/trainings.py#L271) — `detail="Failed to {action} training"` is a plain string, not an f-string, so the literal `{action}` is returned. Cosmetic (only on the 500 path), but sloppy. (The log line above it *is* correct.)

**F8. `create_training` reads `hyperparameters_dict['total_steps']` with bracket access** ([training_service.py:72](../../backend/src/services/training_service.py#L72)). Safe today because the schema requires `total_steps` (`Field(...)`), but it's a latent `KeyError`→500 if the hyperparameters schema ever makes it optional. Use `.get()` with a guard or rely on the validated model attribute.

---

## 4. Test Coverage Notes

The **strongest test suite in the review so far**: service, tasks, schemas, a dedicated validator test, checkpoint service, SAE converter, multi-SAE import, and two checkpoint-deletion suites plus a workflow integration test. Gaps:
- **No test asserts `create_training` rejects a missing model / mismatched tokenization** (F1) — which is why the FR-6.5 validation gap is invisible. If `test_training_validator.py` tests a validator that isn't wired into the create path, that's a false sense of coverage.
- No test for duplicate metric rows at the same step (F2).
- No unit test pins the JumpReLU `sparsity_coeff` *scale* (count-based), which would have caught the stale docstring (F3) if it asserted expected loss magnitude.

---

## 5. Summary — if you fix three things

1. **F1** — wire real validation into `create_training` (model/dataset/extraction existence + tokenization match) so bad configs 400 at API time instead of dying mid-training. This is the biggest correctness gap (and contradicts a PRD "Implemented" claim).
2. **F3** — fix the `JumpReLUSAE` docstring (says fraction-based/default 0.4; code is count-based/default 1e-3). Actively misleads sparsity tuning.
3. **F2** — add the `(training_id, step, layer_idx)` unique constraint (or document + dedup) to stop silent metric duplication under multi-hook/resume.

**The good news:** the mechanistic-interpretability core — all 6 architectures, the STE autograd functions, decoder normalization, threshold calibration, count-based L0 — is correct and paper-faithful. This is the code that matters most for research validity, and it holds up.

**Cross-feature patterns (001+002+003):** stale FPRD reference sections (data model / endpoints / channels) in all three; a documented endpoint that doesn't exist (003's `/retry`, like 002's `/preview`); and internal doc-drift now appearing in code docstrings too (F3). The FPRD refresh is looking like a single batch task across the whole feature set.
