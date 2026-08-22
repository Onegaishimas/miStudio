# Dependency review — 2026-08-22

Triggered by `google/gemma-4-12B-it` failing to load with
`model type 'gemma4_unified' but Transformers does not recognize this
architecture`, then widened to every pinned version and the open Dependabot
alerts.

## The gemma-4 failure

`transformers==5.5.4` knows `gemma4`, `gemma4_text`, `gemma4_vision` and
`gemma4_audio` — but not `gemma4_unified`, which is what
`google/gemma-4-12B-it` declares (`Gemma4UnifiedForConditionalGeneration`, with
audio + text + vision sub-configs). Verified against the transformers source:
`src/transformers/models/gemma4_unified/` is present at `v5.15.1` and absent at
`v5.5.4`.

Note the failure was in the LOAD, not the download. `download_and_load_model`
loads the model to read its architecture for the database, so an architecture
transformers does not know fails the whole job even though the weights are on
disk.

## Changed

| package | from | to | why |
|---|---|---|---|
| transformers | 5.5.4 | 5.15.1 | `gemma4_unified` |
| safetensors | 0.7.0 | 0.8.0 | **required** by transformers 5.15 (`>=0.8.0`) |
| huggingface-hub | 1.24.0 | 1.28.0 | current 1.x; transformers wants `>=1.5,<2.0` |
| cryptography | 48.0.1 | 50.0.0 | GHSA high |
| aiohttp | 3.14.1 | 3.14.3 | GHSA high + 2 medium |
| setuptools | 80.9.0 | 83.0.0 | GHSA medium |
| frontend lockfile | — | — | `npm audit fix`, 5 high → 0, `package.json` untouched |
| @docusaurus/* | 3.9.2 | 3.10.2 | minor, within v3 |

`tokenizers` stays at 0.22.2: transformers 5.15 requires `>=0.22.0,<=0.23.0`,
so 0.22.2 satisfies it and 0.23.1 (latest) would violate the upper bound.

## Verified, not assumed

* Backend suite **2695 passed / 0 failed** on transformers 5.15.1.
* AES-256-GCM settings encryption round-trips on cryptography 50, with a fresh
  nonce per call — checked directly, because the `-k "encrypt or settings"`
  filter I first reached for collected **zero tests** and would have "passed"
  without running anything.
* Frontend: tsc clean, 1211 tests, build clean.
* Manual: builds on Docusaurus 3.10.2.

## Accepted, with no fix available

**`image-size` (high, ×18 alerts).** All eighteen trace to one root, reached
only through `@docusaurus/mdx-loader`. `image-size@2.0.2` **is** the latest
published version, and two advisories cover `<= 2.0.2` with
`firstPatchedVersion: NONE`. There is nothing to upgrade to.

Exposure is a build-time denial of service in a documentation generator: an
infinite loop when parsing a malformed ICNS, JXL or HEIF image. Reaching it
requires a crafted image inside this repository's own docs, committed by
someone who already has write access. Not shipped to any user at runtime — the
docs build produces static HTML.

Revisit when `image-size` publishes a patched release.

## Deferred

`torch` 2.9.1 (two low alerts, fixes at 2.10.0 / 2.13.0). Deferred as before:
torch is coupled to the CUDA and triton builds this deployment runs on, and a
low-severity advisory does not justify moving that floor. Tracked, not ignored.
