# Technical Design Document: Training Finalization & Checkpoint Lifecycle

**Document ID:** 020_FTDD|Checkpoint_Lifecycle
**Version:** 1.0
**Status:** Implemented
**Related:** 020_FPRD · PADR IDL-39 · PPRD §3.21

---

## 1. Rebuilding an SAE from a checkpoint

The training loop holds live `nn.Module`s when it exports. Finalize has only
files, so it must reconstruct equivalent modules.

```
resolve step        -> newest complete, or the caller's explicit step
resolve layer dirs  -> read DISK: layer_{idx}_{hook}/ (or legacy layer_{idx}/)
per (layer, hook):
    d_in, d_sae  <- checkpoint tensor shapes        # NOT hyperparameters
    arch         <- safetensors metadata["architecture"]  # NOT hyperparameters
    model        <- create_sae(arch, d_in, d_sae, ...)
    load_checkpoint(file, model, device="cpu")      # strict load_state_dict
```

**Why shapes, not hyperparameters.** `training_tasks` inspects the activation
file and overwrites `hp['hidden_dim']` in memory when it disagrees, logging
`HIDDEN_DIM MISMATCH DETECTED` — and never writes the corrected value back. A
rebuild from `hp` can therefore construct the wrong input width and fail
`load_state_dict`. The tensors cannot lie.

**Why not `load_multilayer_checkpoint`.** It is legacy-only (int keys,
`layer_{idx}/` paths) and raises `FileNotFoundError` against every checkpoint
the current loop writes. It has no callers. The inline fallback from
`training_tasks` is used instead.

**Why not the weights-only converter.** `migrate_mistudio_to_community`
hard-requires `encoder.weight`; JumpReLU — the architecture in production use —
stores `W_enc`. It would fail on exactly the checkpoints this feature exists to
rescue.

## 2. Config fidelity

`save_multilayer_community_checkpoint` builds a per-layer `cfg.json` via
`CommunityStandardConfig.from_training_hyperparams`, which reads `hidden_dim`,
`latent_dim` and `architecture_type` **from the hyperparameters dict it is
handed**. Passing the stored `hp` therefore emits a config that can contradict
the weights sitting beside it.

The resolved values are written into an `export_hp` copy before the call. The
per-layer resolutions are collapsed with a consistency check; a divergence is
logged loudly because one `cfg.json` cannot describe two shapes.

## 3. Export integrity

```
staging = community_format.tmp-{step}/
write all layers into staging          # failure -> rmtree staging, previous export untouched
community_format/ -> community_format.prev-{step}/
staging           -> community_format/
rmtree prev
```

Writing in place meant a per-layer failure (empty `state_dict`, ENOSPC, a
too-small file) left this step's `layer_7` beside the previous step's
`layer_14`/`layer_18` — a chimeric SAE spanning two training steps that
`sae_manager_service` would happily scan. The swap also makes a stale-layer
prune unnecessary: the new directory only ever contains this step's layers.

**Completeness.** The expected `(layer, hook)` set comes from the checkpoint
rows' `extra_metadata`, falling back to hyperparameters. If the newest step is
missing layers and no step was named, finalize walks back to the newest complete
step. A database error while reading that metadata **raises** — returning "no
metadata" would be indistinguishable from the absent case and would silently
disable the guard.

## 4. Retention

```
group rows by step
keep = newest keep_last steps
     ∪ steps containing an is_best row (when keep_best)
     ∪ steps whose youngest row is newer than min_age_hours
prunable = all steps - keep
```

Grouping is the whole design. One row exists per `(step, layer, hook)` sharing a
single `checkpoint_{step}/` directory, so row-granular selection keeps an
arbitrary subset of a step's layers and produces an unloadable checkpoint —
which then poisons finalize, because the completeness check derives its expected
layer set from the *surviving* rows and would call the torn step complete.

Execution mirrors selection: a step's files are all unlinked, and only then are
all of its rows deleted and committed together.

## 5. Delete ordering

Unlink first; delete + commit the row only on success.

Committing the row over a failed unlink strands the file permanently: planning
is row-driven, so with no row no future prune can ever plan it again, while the
run reports "0.00 GB freed" as though nothing needed doing. The inverse residual
risk — a crash between unlink and commit — leaves a row whose file is gone,
which the next pass can detect. Both delete paths (the pruner and the API's
`delete_checkpoint`) follow this ordering; they previously had opposite
invariants while claiming to match.

## 6. Architecture / types

| Component | Responsibility |
|---|---|
| `services/training_finalize_service.py` | discovery, rebuild, atomic export (no Celery, no HTTP) |
| `services/checkpoint_retention.py` | policy; a **session-free** pure core plus thin sync/async wrappers |
| `workers/training_finalize_tasks.py` | Celery task; guarded status write; terminal WS events |
| `workers/prune_checkpoints.py` | scheduled sweep + single-training prune |
| `services/checkpoint_service.py` | `delete_checkpoint_files` (sync, shared), `is_best` guard |

`RetentionPolicy`, `PrunePlan` are dataclasses. `plan_from_checkpoints` takes
plain objects so the policy is testable without a database, and so the sync
worker and the async endpoint cannot drift.

`finalized_from_step: Optional[int]` on `Training`; `TrainingControlRequest`'s
Literal widened with `stop_and_finalize` (the schema gates the request — without
it the endpoint 422s before the handler runs).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Finalize races the SIGTERMed trainer's failure handler | guarded status write: refuse to promote a non-terminal run |
| A prune deletes files a finalize is reading | active-status guard; min-age; single-training prune is explicit |
| Operator arms deletion by accident | ships disabled + dry-run; restrictive settings saved before `enabled`; unparseable values fall back to the safe default |
| Orphan checkpoint directories with no row | **known gap** — retention is row-driven; recorded as follow-up debt |
| Transient double disk during the atomic swap | one export's worth; acceptable against a chimeric SAE |
