# Research: Hugging Face as the Marketplace for miStudio Cluster Definitions

**Date:** 2026-07-16
**Scope:** How `mistudio.cluster-definition/v1` / `mistudio.cluster-bundle/v1` JSON artifacts (see `/home/x-sean/app/miStudio/docs/schemas/cluster-definition-v1.json`) fit natively on the Hugging Face Hub, and a recommended publishing convention.
**Status:** Research only — no BRD commitments implied.

---

## TL;DR Recommendation

Publish cluster definitions as **Model repositories** (not Datasets, not Spaces), one repo per *(base model × SAE)* "cluster pack," with:

- individual `clusters/<slug>.cluster.json` files as the source of truth (git-native, diffable, KB-sized — no LFS involved),
- a flat `manifest.jsonl` index for humans and cheap programmatic listing,
- model-card YAML carrying `base_model: <hf id>` (+ `base_model_relation: adapter`) so the Hub's model tree links each pack to its base model,
- the custom tag **`mistudio-cluster-definition`** (plus `mistudio`) as the marketplace discovery key — filterable at `https://huggingface.co/models?other=mistudio-cluster-definition` and via `HfApi.list_models(filter=...)`,
- Hub git tags (`create_tag`) for content versioning; `schema_version` inside the JSON strictly for parser compatibility.

This matches how the SAE community already publishes (Gemma Scope, SAELens-format repos are model repos with per-hook-point folders + small JSON configs) and completely sidesteps the dataset-viewer's poor handling of free-form nested JSON.

---

## 1. Repo type: Datasets vs Models vs Spaces

### Findings

**The three repo types** ([hub docs](https://huggingface.co/docs/hub/repositories)): Models, Datasets, Spaces. All are git repos with the same storage/limit rules ([storage limits](https://huggingface.co/docs/hub/en/storage-limits)); the differences are (a) which card metadata fields exist, (b) whether the dataset viewer runs, (c) which search index/filters the repo appears in.

**Key asymmetry — `base_model` only exists on model cards.** The model-card spec ([model-cards docs](https://huggingface.co/docs/hub/en/model-cards)) supports `base_model: <hub id>` with an inferred or explicit relation (`adapter`, `finetune`, `quantized`, `merge` via `base_model_relation`). This renders the "model tree" on the base model's page and lets users filter "adapters of google/gemma-2-2b". The dataset-card spec ([datasetcard.md](https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1), [datasets-cards docs](https://huggingface.co/docs/hub/en/datasets-cards)) has **no** `base_model` or model-linking field — only `source_datasets`, `tags`, `task_categories`, etc. A steering artifact is definitionally "attached to a specific base model + SAE," and only a model repo can express that attachment in Hub-native metadata.

**Precedents — SAE community assets are model repos:**

- **Gemma Scope** ([google/gemma-scope](https://huggingface.co/google/gemma-scope), [google/gemma-scope-2b-pt-res](https://huggingface.co/google/gemma-scope-2b-pt-res)): model repos, one repo per *(model × site family)* (`-res`, `-att`, `-mlp`, transcoders), folders per layer/width/L0 (e.g. `layer_12/width_16k/average_l0_82/`), each leaf holding weights + a small JSON config with the SAE's parameters.
- **SAELens-format repos** ([upload tutorial](https://github.com/decoderesearch/SAELens/blob/main/tutorials/uploading_saes_to_huggingface.ipynb), example [tommmcgrath/gpt2-mlp-out-saes](https://huggingface.co/tommmcgrath/gpt2-mlp-out-saes)): model repos; `upload_saes_to_huggingface()` writes one folder per SAE, keyed by hook point, each containing `cfg.json` + `sae_weights.safetensors` + `sparsity.safetensors`. An SAE is addressed as *(repo_id, folder path)* — exactly the shape of miStudio's `sae.source_hint: "hf:repo/path"`. SAELens additionally keeps a central `pretrained_saes.yaml` registry mapping release names → repos.
- **Neuronpedia** publishes bulk data exports via S3 and some datasets on HF ([docs.neuronpedia.org/api](https://docs.neuronpedia.org/api), `neuronpedia-org/*` dataset repos exist, e.g. `neuronpedia-org/sae-evals`), but its per-feature JSON is served from its own API, not a Hub convention. Community derivative datasets exist (e.g. `hbe/neuronpedia-sae-concepts`, tagged with mechanistic-interpretability/SAE tags).

**Where Datasets *would* fit:** large tabular/JSONL corpora meant for `load_dataset()` + the viewer. A cluster definition is a *config artifact* (like an adapter's `adapter_config.json`), not training data. The Hub's own guidance for datasets asks for viewer-friendly formats (Parquet/JSONL) and discourages shapes the viewer can't handle ([storage limits — sharing large datasets](https://huggingface.co/docs/hub/en/storage-limits)).

**Spaces:** wrong primitive for artifact storage (it's app hosting). Optionally useful *later* as a browse/preview UI over the tag namespace — not as the artifact home.

### Verdict

**Model repo.** It carries `base_model` linkage, matches SAE-community precedent, gives file-browser JSON pretty-printing with zero viewer friction, and steering artifacts are semantically adapter-like.

---

## 2. JSON handling on the Hub (viewer behavior, JSONL vs JSON, sizes)

### Findings

- **Dataset viewer & free-form nested JSON:** the viewer ([configure docs](https://huggingface.co/docs/hub/en/datasets-viewer-configure)) auto-converts data files to Parquet/Arrow. JSON works when it is **JSONL (one object per line)** or a **top-level array of homogeneous objects**; `datasets` explicitly calls JSONL "the most efficient format" ([loading docs](https://huggingface.co/docs/datasets/en/loading)). It **chokes** on: a single top-level JSON object that isn't an array (needs `field=` workarounds), heterogeneous rows (the `oneOf` definition-vs-bundle shape), inconsistent nested field types (`ArrowTypeError: Expected bytes, got a 'list' object` — [datasets #7116](https://github.com/huggingface/datasets/issues/7116), [forum thread](https://discuss.huggingface.co/t/load-dataset-fail-for-custom-json-format/30350)). A `.cluster.json` file (one object, nullable nested blocks, `constants` as open dict) is exactly the shape that breaks Arrow schema inference. The viewer can be disabled with `viewer: false` in the card YAML — but in a **model repo the question never arises**; there is no viewer, and the file browser pretty-prints JSON files directly.
- **One-file-per-definition vs JSONL manifest:** not either/or — do both. Individual files preserve the artifact exactly as exported/imported (`kind` + `schema_version` intact, byte-stable for hashing, one-URL download, clean git diffs per cluster). A generated `manifest.jsonl` (one flattened row per definition: slug, name, path, model hf_id, sae layer/n_features/source_hint, member count, budget formula_id, exported_at) gives cheap listing without downloading every file. The Hub's own recommendation to "merge json files into a single jsonl" ([storage limits](https://huggingface.co/docs/hub/en/storage-limits)) is about the 100k-file ceiling — irrelevant at ≤ hundreds of KB-sized files — so keeping individual files is fully within recommendations.
- **Size limits** (see constraints table below): git (non-LFS) files must be ≤10 MiB; huggingface_hub HTTP uploads auto-route bigger files to LFS/Xet; hard cap 500 GB/file, recommended <200 GB; <100k files/repo, <10k entries/folder ([storage limits](https://huggingface.co/docs/hub/en/storage-limits), [repo recommendations](https://huggingface.co/docs/hub/repositories-recommendations)). A definition is typically 5–30 KB (narrative ≤10k chars, ≤20 members); a bundle ≤50 defs stays ≲1.5 MB. **Everything stays in plain git — no LFS, fully diffable, instantly renderable.** JSON is not in the default `.gitattributes` LFS patterns, which is what we want.

### Verdict

Individual `*.cluster.json` files as source of truth + generated `manifest.jsonl` index. If a dataset-repo mirror is ever wanted for viewer browsing, the *manifest* (flat JSONL) is the thing to mirror, never the raw nested definitions.

---

## 3. Discoverability: cards, YAML tags, Hub search, API

### Findings

- **Custom tags are first-class.** Any string in the card's `tags:` list becomes a filterable Hub tag ([model-cards FAQ](https://huggingface.co/docs/hub/en/model-cards): "you can add any tag you want"). Web filtering uses the `?other=` query param — `https://huggingface.co/models?other=mistudio-cluster-definition` — the same mechanism ecosystems like fiftyone use (`datasets?other=fiftyone`) ([search guide](https://huggingface.co/docs/huggingface_hub/guides/search)).
- **`base_model` produces auto-tags** on model repos (`base_model:adapter:<id>` etc.) which power "adapters of X" filtering and the model-tree widget ([model-cards docs](https://huggingface.co/docs/hub/en/model-cards)).
- **`library_name`** is intended for registered integrations (it drives the "Use in <library>" button). `library_name: mistudio` is harmless and self-documenting but unregistered — do **not** rely on it for discovery; the plain `mistudio` tag is the reliable key. (Registering miStudio as a Hub library via [huggingface.js model-libraries](https://github.com/huggingface/huggingface.js) is a possible future step.)
- **Programmatic browsing** ([HfApi reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api), [search guide](https://huggingface.co/docs/huggingface_hub/guides/search)):

  ```python
  from huggingface_hub import HfApi
  api = HfApi()  # no token needed for public listing

  # All cluster packs on the Hub:
  packs = api.list_models(filter="mistudio-cluster-definition", full=True)

  # Packs for a specific base model (auto-tag from base_model metadata):
  packs = api.list_models(
      filter=["mistudio-cluster-definition", "base_model:adapter:google/gemma-2-2b"]
  )

  # Plus free-text: api.list_models(search="emotional tone", filter="mistudio-cluster-definition")
  ```

  `list_models`/`list_datasets` accept `filter` (tags, AND-combined), `author`, `search`, `full`; results carry `tags`, `downloads`, `likes`, `lastModified` — enough to render a marketplace list with popularity sorting inside miStudio.
- **Collections** (hub feature) let the miStudio org curate a "featured cluster packs" gallery without owning the repos — a lightweight storefront layer.

---

## 4. Programmatic push/pull

### Findings ([upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload), [download guide](https://huggingface.co/docs/huggingface_hub/guides/download))

**Publish (miStudio already has an encrypted `hf_token` and pushes SAEs — same client, same auth):**

```python
from huggingface_hub import HfApi
api = HfApi(token=decrypted_hf_token)
api.create_repo(repo_id, repo_type="model", exist_ok=True)          # private=True supported
api.upload_folder(repo_id=repo_id, folder_path=staging_dir,
                  commit_message="Add cluster pack vN")
# or atomic multi-file commit: api.create_commit(repo_id, operations=[CommitOperationAdd(...), ...])
# card metadata: huggingface_hub.ModelCard / metadata_update() for tags + base_model
```

`upload_folder` auto-splits oversized commits; irrelevant at our file counts. Write scope on the token is required (fine-grained tokens can be limited to specific repos).

**Import:**

```python
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
files = list_repo_files(repo_id)                                     # discover *.cluster.json
p = hf_hub_download(repo_id, "clusters/formal-tone.cluster.json",
                    revision="v2")                                    # tag/branch/sha pinning
d = snapshot_download(repo_id, allow_patterns=["clusters/*.cluster.json",
                                               "manifest.jsonl", "README.md"])
```

**Anonymous read:** public repos need **no token** for listing, card fetch, or download — the import/browse side of the marketplace works for users who never configured HF credentials. Downloads are CDN-served and locally cached (`HF_HOME` cache, dedup by commit hash).

---

## 5. Versioning & provenance: Hub revisions vs `schema_version`

### Findings

Two orthogonal axes; keep them orthogonal:

| Axis | Mechanism | Governs |
|---|---|---|
| **Format compatibility** | `schema_version: "1"` + `kind` inside the JSON | Can this parser read this file at all? Bumped only on breaking schema change (→ `/v2`). |
| **Content versioning** | Hub git history + tags (`HfApi.create_tag(repo_id, tag="v3")`, [hf_api reference](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)) | Which iteration of this author's tuned strengths/narrative? Every push is a commit; tags mark releases; any `revision=` (tag/branch/sha) is fetchable forever. |

Recommended interplay:

1. Publisher pushes edits freely to `main`; cuts a repo tag (`v1`, `v2`, …) when strengths/narrative change materially. Optionally `new_version:` card field if a whole pack is superseded by a new repo.
2. Importer pins: miStudio records `repo_id + commit_sha + file path` at import time — the schema's `provenance.source_note` (≤500 chars) can carry `hf:<repo_id>@<sha>/clusters/<slug>.cluster.json` today, zero schema change needed. (A v2 schema could add a structured `provenance.hub_ref`.)
3. Never re-purpose `schema_version` for content revisions — content identity lives on the Hub, format identity lives in the file. This mirrors how `cfg.json` inside SAELens repos versions format while the repo versions content.

---

## 6. Emerging standards for steering/interp artifacts on HF

### Findings

- **No established Hub format for steering vectors exists.** The [steering-vectors library](https://github.com/steering-vectors/steering-vectors) (CAA/RepE lineage) trains and applies vectors in-process and defines **no Hub serialization convention**. Scattered individual repos upload raw vector tensors ad hoc. There is nothing to conflict with — and an opportunity for `mistudio.cluster-definition/v1` to be an early well-formed convention.
- **The de facto interp conventions are SAELens's and Gemma Scope's** (see §1): model repos, folder-per-artifact, small JSON config beside weights, addressing = *(repo, path)*. miStudio's `sae.source_hint: "hf:repo/path"` already aligns with this addressing scheme — keep it, and when the referenced SAE is itself SAELens-format, `source_hint` should point at the SAE's hook-point folder exactly as SAELens IDs do.
- **Neuronpedia** ([API/exports docs](https://docs.neuronpedia.org/api), [github](https://github.com/hijohnnylin/neuronpedia)) exports features/explanations as JSON/Parquet dumps (primarily S3, some HF datasets under `neuronpedia-org`). Its per-feature JSON is a *description* format, not a *steering-recipe* format — complementary, not competing. A future nicety: include Neuronpedia feature URLs inside `narrative` markdown for evidence links.
- **Community tags in live use** on interp assets: `sae`, `sparse-autoencoder`, `mechanistic-interpretability`, `interpretability` (e.g. on `hbe/neuronpedia-sae-concepts`). Adopting them alongside the `mistudio*` tags places cluster packs in searches interp researchers already run.
- Because `kind` is a namespaced string (`mistudio.cluster-definition`), other tools (miLLM import is already planned) can consume the format without naming collision — the vendor-prefixed `kind` + published JSON Schema (`$id: https://mistudio.hitsai.net/schemas/cluster-definition-v1.json`) is the interop contract; the Hub is just transport.

---

## Recommended Publishing Convention

### Repo type & granularity

- **Repo type:** `model`
- **Granularity:** one repo per *(base model × SAE)* cluster pack. Definitions for different SAEs never share a repo (members' `feature_idx` are meaningless across SAEs). A pack may hold 1–N definitions.
- **Repo naming (suggested, not enforced):** `<owner>/mistudio-clusters-<model-short>-l<layer>-<sae-short>` e.g. `seanm/mistudio-clusters-gemma-2-2b-l12-res-16k`. Detection relies on tags + file `kind`, never on repo name.

### File layout (worked example)

```
seanm/mistudio-clusters-gemma-2-2b-l12-res-16k        (model repo)
├── README.md                          # model card — YAML below + human overview
├── manifest.jsonl                     # generated index, 1 flat row per definition
├── clusters/
│   ├── formal-tone.cluster.json       # kind=mistudio.cluster-definition, schema_version=1
│   ├── medical-terminology.cluster.json
│   └── sycophancy-damping.cluster.json
├── bundles/
│   └── pack.bundle.json               # optional convenience: kind=mistudio.cluster-bundle (≤50)
└── schema/
    └── cluster-definition-v1.json     # vendored schema copy (offline validation)
```

Rules:
- `clusters/*.cluster.json` — exactly one `ClusterDefinitionV1` per file; filename = slugified `name`; files are the canonical artifacts (import = download one file).
- `manifest.jsonl` — regenerated on every publish; row shape (flat, viewer/Arrow-safe if ever mirrored to a dataset repo): `{"slug", "path", "name", "display_token", "model_hf_id", "sae_layer", "sae_n_features", "sae_source_hint", "n_members", "budget_formula_id", "intensity", "exported_at", "mistudio_version"}`.
- `bundles/` optional; a bundle is a convenience download, never the source of truth (avoids merge conflicts between definitions).

### Model-card YAML template

```yaml
---
base_model: google/gemma-2-2b
base_model_relation: adapter
library_name: mistudio
license: apache-2.0
tags:
- mistudio
- mistudio-cluster-definition
- steering
- steering-vectors
- sae
- sparse-autoencoder
- mechanistic-interpretability
- sae-layer:12
- sae-source:google/gemma-scope-2b-pt-res
---
# Gemma-2-2B L12 cluster pack — tone & domain steering

3 cluster definitions (mistudio.cluster-definition/v1) for
[google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) with the
Gemma Scope residual SAE (layer 12, 16k,
`hf:google/gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82`).

| Cluster | Members | λ | Summary |
|---|---|---|---|
| formal-tone | 14 | 1.0 | ... |

Import in miStudio: Clusters → Import → paste this repo id.
```

Notes: `sae-layer:12` / `sae-source:<repo>` are plain custom tags (the `:` is legal in tags) enabling narrower filters later; the required discovery key is `mistudio-cluster-definition`. Include a real `license` — cards without one get a warning banner and are un-filterable by license.

### Size/format constraints table

| Constraint | Hub limit | Cluster-definition reality | Consequence |
|---|---|---|---|
| Non-LFS git file | ≤10 MiB (pre-receive hook) | definition ~5–30 KB; bundle ≤ ~1.5 MB | plain git; diffable; browser-renderable; no LFS ever |
| LFS/Xet file | hard 500 GB, rec. <200 GB | n/a | n/a |
| Files per repo | rec. <100k | ≤ ~100 | fine |
| Entries per folder | max 10k | ≤ ~100 | fine |
| Commit size (HTTP API) | rec. <100 files/commit (`upload_folder` auto-splits) | ≤ ~100 | single commit |
| Repo total / account storage | public: best-effort free tier; PRO 10 TB | KBs–MBs | negligible |
| Dataset-viewer JSON | JSONL / homogeneous top-level arrays only; nested-heterogeneous or single-object JSON errors | `oneOf` + nullable nested blocks + open `constants` dict | **model repo avoids viewer entirely**; only `manifest.jsonl` is viewer-shaped |
| Narrative | — | ≤10k chars markdown inside JSON | also surfaced in README by the publisher for human browsing |

Sources: [storage-limits](https://huggingface.co/docs/hub/en/storage-limits), [repositories-recommendations](https://huggingface.co/docs/hub/repositories-recommendations), [datasets-viewer-configure](https://huggingface.co/docs/hub/en/datasets-viewer-configure), [datasets #7116](https://github.com/huggingface/datasets/issues/7116).

---

## Implications for a Future BRD (marketplace = HF)

What miStudio would add — all thin layers over `huggingface_hub`, which is already a dependency (SAE push):

1. **"Publish to Hugging Face" on a cluster profile / selection of profiles.** Flow: pick definitions (same SAE enforced) → repo id suggestion + private/public toggle → stage export tree (clusters/, manifest.jsonl, generated README from name/narrative/budget table, card YAML with `base_model` from `model.hf_id` and sae tags from `sae` block) → `create_repo` + `upload_folder` with the stored encrypted `hf_token` → optional `create_tag("v1")`. Re-publish = new commit + next tag.
2. **HF browse/import panel.** `list_models(filter="mistudio-cluster-definition" [+ base_model auto-tag for the currently loaded model])`, render cards with likes/downloads/lastModified; on select, fetch `manifest.jsonl` for the cluster list; import downloads the single `.cluster.json` (anonymous — no token requirement for public packs), runs the existing import matrix/validation (schema check against vendored `cluster-definition-v1.json`, SAE compatibility gate on `n_features`/`d_model`/`layer`), records `hf:<repo>@<sha>/<path>` provenance.
3. **Tag convention freeze** (goes in the BRD as a normative appendix): required `mistudio`, `mistudio-cluster-definition`; recommended `steering`, `sae`, `sparse-autoencoder`, `mechanistic-interpretability`, `sae-layer:<n>`, `sae-source:<repo>`; card must carry `base_model` when `model.hf_id` is known.
4. **Schema touch (v1-compatible):** none required. Nice-to-have for v2: structured `provenance.hub_ref {repo_id, revision, path}`; until then `provenance.source_note` string carries it.
5. **Deliberately out of scope:** custom marketplace backend, ratings/comments (Hub likes + community tab exist), payment, moderation (Hub reporting applies). A curated **HF Collection** under a miStudio org + optional registry file (SAELens `pretrained_saes.yaml` precedent) covers "featured" without infrastructure. Registering `mistudio` as a Hub library integration (huggingface.js PR) is a later polish item that would light up a "Use in miStudio" button on repo pages.
6. **miLLM alignment:** miLLM's future import (per BRD-MIS-CLUSTERS-001 future_considerations) consumes the identical repos — the convention above is consumer-neutral, keyed on file `kind`, not on anything miStudio-specific in the repo.

Risks/watch items: HF free public storage is "best-effort" beyond a few GB (irrelevant at our sizes, but ToS-watch); `?other=` tag filtering is stable-but-undocumented UI behavior (the API `filter=` param is the contract to build on); anonymous unauthenticated API calls are rate-limited — cache browse results.

---

## Sources

- https://huggingface.co/docs/hub/en/storage-limits
- https://huggingface.co/docs/hub/repositories-recommendations
- https://huggingface.co/docs/hub/en/datasets-viewer-configure
- https://huggingface.co/docs/hub/en/datasets-cards
- https://github.com/huggingface/hub-docs/blob/main/datasetcard.md
- https://huggingface.co/docs/hub/en/model-cards
- https://huggingface.co/docs/datasets/en/loading
- https://github.com/huggingface/datasets/issues/7116
- https://discuss.huggingface.co/t/load-dataset-fail-for-custom-json-format/30350
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api
- https://huggingface.co/docs/huggingface_hub/guides/search
- https://huggingface.co/docs/huggingface_hub/guides/upload
- https://huggingface.co/docs/huggingface_hub/guides/download
- https://huggingface.co/google/gemma-scope
- https://huggingface.co/google/gemma-scope-2b-pt-res
- https://github.com/decoderesearch/SAELens/blob/main/tutorials/uploading_saes_to_huggingface.ipynb
- https://huggingface.co/tommmcgrath/gpt2-mlp-out-saes
- https://docs.neuronpedia.org/api
- https://github.com/hijohnnylin/neuronpedia
- https://github.com/steering-vectors/steering-vectors
- https://huggingface.co/api/datasets?search=neuronpedia (live check: `neuronpedia-org/*` dataset repos, `hbe/neuronpedia-sae-concepts` tags)
