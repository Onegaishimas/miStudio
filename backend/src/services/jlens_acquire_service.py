"""
Adopt a J-lens that someone else fitted.

miStudio can otherwise only obtain a lens by fitting one — a long GPU job per
model — while a large body of pre-fitted lenses is already published. The
conformance spec's own recommended sequence (§8) puts "check the repo, download,
validate, mount" FIRST and fitting last; this is that missing front half.

WHAT THIS MODULE MUST NOT DO IS INVENT PROVENANCE. The fit worker's
`_config_yaml` records a recipe — attention-gradient treatment, position scope,
aggregation, differentiation mode, corpus, sequence length, convergence
threshold — because it *performed* those choices. For an acquired lens miStudio
performed none of them. Writing them anyway, even as defaults, puts a recipe
miStudio invented into the file whose stated purpose (BR-007, spec §2.3) is that
the lens be reproducible from it alone, and `ProvenanceStrip` then renders it as
fact. That file already carries a scar of exactly this shape: `target_layer` was
hardcoded to "final" while the real parameter was threaded and dropped, so a
penultimate fit published a recipe claiming otherwise.

So every field below is DERIVED FROM SOMETHING MEASURABLE — the tensors, the
loaded model, or an explicit statement in the publisher's own config — and a
field that cannot be derived is OMITTED rather than defaulted. The readers treat
absence as unknown (`_config_bool`: "None is NOT False"), which is the truth.

UPSTREAM PROVENANCE GOES IN `acquisition.json`, NEVER INTO `config.yaml`. The
config readers are line scanners: they `partition(":")` and match on
`name.strip()`, returning on the first hit AT ANY INDENTATION. A nested
`acquired:` block would therefore not be namespaced at all — verified, a nested
`layer_scales:` is read as real and yields a FABRICATED per-layer rescale that
`JacobianTransport` applies to every probe and intervention magnitude, invisible
in ranked readouts because the model's final norm divides a positive scalar back
out. `acquisition.json` is ignored by `_ref_for` and by `check_naming`, exactly
as `validation.json` and `interventions.json` are.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

#: Sidecar carrying everything about the TRANSFER rather than about the lens.
#:
#: Named like `validation.json` / `interventions.json` so a reader of the
#: upstream conformance layout ignores it; `_ref_for` requires only the `.pt`
#: and looks for `config.yaml`, and `check_naming` rejects only extra `.pt`s.
ACQUISITION_FILE = "acquisition.json"

#: How close to the identity a matrix must be to count as degenerate.
#:
#: The lens AT its own target layer is the identity by construction — there is no
#: transport to perform. Everything else is a measurement, so this is a
#: tolerance on a float comparison rather than a threshold anyone chose: the
#: fitter writes exact identities and fp16 round-trip perturbs them by ~1e-3.
IDENTITY_TOLERANCE = 5e-3


class WeightIdentity(str, Enum):
    """Whether the lens can be shown to belong to these weights."""

    #: The publisher named the model and it is the one we are attaching to.
    VERIFIED = "verified"
    #: The publisher named a DIFFERENT model. A hard refusal, never a warning.
    MISMATCH = "mismatch"
    #: No config, or no model named in it. Common for community repos; the
    #: user asserts the pairing and the artifact records that they did.
    UNVERIFIED = "unverified"


class AcquisitionRefused(RuntimeError):
    """Raised rather than adopting a lens we cannot honestly describe."""


@dataclass
class IdentityVerdict:
    state: WeightIdentity
    detail: str
    declared: Optional[str] = None
    expected: Optional[str] = None


@dataclass
class LayerVerdict:
    """What the tensors themselves say about indexing and the target layer."""

    fitted: List[int]
    d_model: int
    #: `final` / `penultimate` / None when it cannot be derived.
    target_layer: Optional[str]
    #: Layers whose matrix is the identity — measured, not declared.
    degenerate: List[int]
    #: `||J - I||_F` per layer, the evidence behind `degenerate` and the target.
    identity_distance: Dict[int, float] = field(default_factory=dict)


def check_weight_identity(
    upstream_config: Optional[Dict[str, Any]], repo_id: str
) -> IdentityVerdict:
    """Does the publisher's own config name the model we are attaching to?

    THE FIRST CHECK IN THIS PROJECT AGAINST EVIDENCE INSIDE THE FILE. The
    existing weight-identity check compares a caller-supplied slug against the
    slug of the model already loaded — two views derived from the same live
    record, which cannot detect a lens whose CONTENTS were fitted for other
    weights. A published `config.yaml` states `hf_model_name`, and that is a
    claim by the party who ran the fit.

    A mismatch is a REFUSAL, not a flag: a lens fitted for different weights
    "produces a complete, plausible readout that is wrong".
    """
    from .jlens_artifact_service import slug_for

    declared = None
    if upstream_config:
        declared = upstream_config.get("hf_model_name") or upstream_config.get("model")
    if not declared:
        return IdentityVerdict(
            WeightIdentity.UNVERIFIED,
            "the source names no model, so the pairing rests on the caller's "
            "assertion alone",
            expected=repo_id,
        )

    if str(declared).strip() == repo_id or slug_for(str(declared)) == slug_for(repo_id):
        return IdentityVerdict(
            WeightIdentity.VERIFIED,
            f"the source declares {declared!r}, which is these weights",
            declared=str(declared),
            expected=repo_id,
        )
    return IdentityVerdict(
        WeightIdentity.MISMATCH,
        f"the source was fitted for {declared!r}, not {repo_id!r}. A lens fitted "
        "for different weights produces a complete, plausible readout that is wrong",
        declared=str(declared),
        expected=repo_id,
    )


def inspect_layers(
    payload: Dict[int, torch.Tensor], n_layers: int, d_model: int
) -> LayerVerdict:
    """Read the indexing convention and the target layer OFF THE TENSORS.

    WHY THIS EXISTS RATHER THAN TRUSTING THE CONFIG. `check_semantic`
    deliberately scans EVERY fitted layer, so an artifact using a different
    layer-index convention — 1-based, or counting from the output — still finds
    the expected token somewhere and passes. Semantic discrimination cannot
    catch the failure most likely on a third-party artifact, and this can:

    * a key at or above `n_layers` is impossible for these weights;
    * the matrix at the target layer is the identity by construction, so the
      minimum of `||J - I||_F` locates the target independently of any claim.
    """
    if not payload:
        raise AcquisitionRefused("the lens contains no layers")

    fitted = sorted(payload)
    out_of_range = [l for l in fitted if l < 0 or l >= n_layers]
    if out_of_range:
        raise AcquisitionRefused(
            f"layers {out_of_range} are outside 0..{n_layers - 1}, so this lens "
            f"does not index the same stack as the model ({n_layers} layers). A "
            "different indexing convention still passes a semantic check, which "
            "scans every fitted layer"
        )

    widths = {int(t.shape[0]) for t in payload.values()}
    if widths != {d_model}:
        raise AcquisitionRefused(
            f"the lens matrices are {sorted(widths)} wide but the model's d_model "
            f"is {d_model}; these are not the same weights"
        )

    distance: Dict[int, float] = {}
    eye = torch.eye(d_model, dtype=torch.float32)
    for layer, matrix in payload.items():
        distance[layer] = float(torch.linalg.norm(matrix.float() - eye))

    degenerate = sorted(l for l, d in distance.items() if d <= IDENTITY_TOLERANCE)

    # THE TARGET IS DERIVED, NEVER ASSUMED. The fitted set runs up to the block
    # the Jacobian was taken TO, so its maximum locates the target — and that is
    # what `_target_index` and therefore the coverage gate reads back.
    top = fitted[-1]
    if top == n_layers - 1:
        target = "final"
    elif top == n_layers - 2:
        target = "penultimate"
    else:
        # Omitted rather than guessed. `target_layer()` returns None, and
        # `_coverage_delta` then fails closed when REPLACING an artifact — which
        # is the correct outcome for a lens whose extent we cannot describe.
        target = None
        logger.info(
            "Top fitted layer %d is neither final (%d) nor penultimate (%d); "
            "target_layer omitted",
            top,
            n_layers - 1,
            n_layers - 2,
        )

    return LayerVerdict(
        fitted=fitted,
        d_model=d_model,
        target_layer=target,
        degenerate=degenerate,
        identity_distance=distance,
    )


def _upstream_int(config: Optional[Dict[str, Any]], *path: str) -> Optional[int]:
    node: Any = config or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return int(node)
    except (TypeError, ValueError):
        return None


def _upstream_float(config: Optional[Dict[str, Any]], *path: str) -> Optional[float]:
    node: Any = config or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def derive_converged(upstream_config: Optional[Dict[str, Any]]) -> Optional[bool]:
    """Did the publisher's own numbers say the fit converged?

    DERIVED FROM TWO RECORDED VALUES, or omitted. The upstream fitter stops early
    when the mean relative change falls under `stop_at_delta`, and records both,
    so "converged" is a comparison rather than a claim. When either is missing
    this returns None — and `_config_bool` is explicit that "None is NOT False".

    Defaulting to True would let an unconverged third-party lens silently
    displace a converged local fit through the quality gate; defaulting to False
    would block a good one. Absent is the honest third answer.
    """
    delta = _upstream_float(upstream_config, "fit", "stop_at_delta")
    reached = _upstream_float(upstream_config, "results", "final_mean_rel_change")
    if delta is None or reached is None:
        return None
    return reached <= delta


def derive_n_prompts(upstream_config: Optional[Dict[str, Any]]) -> Optional[int]:
    """How many prompts the fit ACTUALLY ran, not how many were requested.

    `fit.n_prompts` is the cap the operator asked for; `results.prompts_fitted`
    is what ran before the convergence stop fired. On the published gemma-2-2b-it
    lens those are 1000 and 337. `_quality_regression` compares this number
    against the incumbent's, so taking the request would let a 337-prompt fit
    claim to be a 1000-prompt one and displace a genuinely larger local fit.
    """
    ran = _upstream_int(upstream_config, "results", "prompts_fitted")
    if ran is not None:
        return ran
    # A bare `n_prompts` at the top level is what OUR OWN configs write, and it
    # means "prompts seen". Fall back to it, never to `fit.n_prompts`.
    return _upstream_int(upstream_config, "n_prompts")


def config_yaml_for_acquired(
    *,
    repo_id: str,
    layers: LayerVerdict,
    n_vocab: int,
    n_layers: int,
    dtype: str,
    upstream_config: Optional[Dict[str, Any]],
) -> str:
    """The recipe file for a lens miStudio did not fit.

    ONLY WHAT WAS MEASURED OR EXPLICITLY READ. Compare the fit worker's
    `_config_yaml`, which additionally writes the treatment, position scope,
    aggregation, differentiation mode, corpus, sequence length and convergence
    threshold — every one of which describes a choice miStudio made while
    fitting. Here miStudio made none of them, and a defaulted value in this file
    is indistinguishable from a measured one to every reader downstream.
    """
    lines = [
        "# J-lens ACQUIRED, not fitted here.",
        "# Only fields miStudio could measure from the artifact or read from the",
        "# publisher's own config appear below. The transfer itself — source,",
        f"# revision, digests and identity verdicts — is in {ACQUISITION_FILE}.",
        f"model: {repo_id}",
        f"d_model: {layers.d_model}",
        f"n_layers: {n_layers}",
        f"n_vocab: {n_vocab}",
        f"dtype: {dtype}",
        f"fitted_layers: {layers.fitted}",
        f"degenerate_layers: {layers.degenerate}",
    ]

    # OMITTED WHEN UNDERIVABLE. `target_layer()` accepts only the two literals
    # and returns None otherwise, and `_coverage_delta` fails closed on None —
    # which is the right behaviour for a lens whose extent we cannot state.
    if layers.target_layer is not None:
        lines.append(f"target_layer: {layers.target_layer}")

    n_prompts = derive_n_prompts(upstream_config)
    if n_prompts is not None:
        lines.append(f"# the publisher's figure, not a fit miStudio ran")
        lines.append(f"n_prompts: {n_prompts}")

    converged = derive_converged(upstream_config)
    if converged is not None:
        lines.append(f"converged: {str(converged).lower()}")

    # NO `layer_scales:` BLOCK. An absent block reads as "no rescale to undo",
    # which is correct: the published artifacts store raw fp16 whose entries are
    # O(1), so nothing was scaled down to survive the cast. Writing 1.0s would
    # be equivalent arithmetically and would assert knowledge of a convention
    # the publisher never stated.
    return "\n".join(lines) + "\n"


def file_digest(path: Path) -> str:
    """Streaming SHA-256, matching `_lens_digest`'s chunking."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_acquisition_record(
    directory: Path,
    *,
    source_repo: str,
    source_path: str,
    revision: Optional[str],
    upstream_sha256: Optional[str],
    local_sha256: str,
    identity: IdentityVerdict,
    layers: LayerVerdict,
    upstream_config: Optional[Dict[str, Any]],
) -> Path:
    """Everything about the TRANSFER, beside the artifact and outside the recipe.

    Two questions the six check classes do not ask, both properties of the
    transfer rather than of the file's conformance:

    * **weight identity** — did the publisher say these are the same weights?
    * **byte identity** — is what we serve bit-for-bit what they published?

    Keeping them here rather than inventing a seventh check class matters: the
    classes answer "is this artifact conformant", and stretching them to cover
    provenance is how a suite starts meaning two things at once.
    """
    record = {
        "source": {
            "repo": source_repo,
            "path": source_path,
            # A REVISION, OR THE STATEMENT IS NOT REPRODUCIBLE. Without it
            # "acquired from <repo>" names a moving target.
            "revision": revision,
        },
        "bytes": {
            "upstream_sha256": upstream_sha256,
            "local_sha256": local_sha256,
            "identical": bool(upstream_sha256) and upstream_sha256 == local_sha256,
        },
        "weight_identity": {
            "state": identity.state.value,
            "detail": identity.detail,
            "declared": identity.declared,
            "expected": identity.expected,
        },
        "layers": {
            "fitted": layers.fitted,
            "target_layer": layers.target_layer,
            "degenerate": layers.degenerate,
            # Rounded for legibility; the full float is not evidence anyone reads.
            "identity_distance": {
                str(k): round(v, 6) for k, v in sorted(layers.identity_distance.items())
            },
        },
        # THE PUBLISHER'S OWN CONFIG, VERBATIM AND QUARANTINED. It is the richest
        # provenance available and it must not reach `config.yaml`, whose line
        # scanners would read its nested keys as miStudio's own.
        "upstream_config": upstream_config,
    }
    target = directory / ACQUISITION_FILE
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def read_acquisition_record(directory: Path) -> Optional[Dict[str, Any]]:
    """The transfer record, or None when the artifact was fitted locally."""
    path = directory / ACQUISITION_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001 - reported
        logger.warning("Could not read %s: %s", path, exc)
        return None


def parse_upstream_config(text: str) -> Dict[str, Any]:
    """The publisher's `config.yaml`, as a nested dict.

    A REAL PARSER, unlike the line scanners that read miStudio's own configs.
    Those are deliberately narrow because they read a file this project writes
    in a known flat shape; this reads a file written by someone else, whose
    nesting is exactly what carries the fields worth having
    (`results.prompts_fitted`, `fit.stop_at_delta`).
    """
    import yaml

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # noqa: BLE001 - absent config is a real state
        logger.warning("Upstream config did not parse as YAML: %s", exc)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def dtype_of(payload: Dict[int, torch.Tensor]) -> str:
    """The dtype actually on disk, read off a tensor rather than declared."""
    dtypes = {str(t.dtype).replace("torch.", "") for t in payload.values()}
    if len(dtypes) != 1:
        raise AcquisitionRefused(
            f"the lens mixes dtypes {sorted(dtypes)}; a consumer casts on load "
            "and would silently promote part of it"
        )
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(
        next(iter(dtypes)), next(iter(dtypes))
    )

# ---------------------------------------------------------------- the source


#: Free space that must remain AFTER a download, on every volume it touches.
#:
#: Mirrors `circuit_capture_service.MIN_FREE_DISK_BYTES`, the only such guard
#: this repo had. No download path checked disk at all — and a J-lens for a 70B
#: model is multiple GB, on a volume already at 83%.
MIN_FREE_DISK_BYTES = 5 * 2**30

#: Extensions that could hold a lens. NOT `*_jacobian_lens.pt`: that glob is the
#: conformant naming, and community repos publish `qwen3_8b_lens.pt`,
#: `gemma2_9b_jlens.pt` and worse. Listing only conformant names would make the
#: generic path useless for exactly the repos it exists to reach.
CANDIDATE_SUFFIXES = (".pt", ".pth", ".safetensors", ".bin")


@dataclass
class RemoteFile:
    path: str
    size_bytes: Optional[int]
    sha256: Optional[str]
    #: Sits beside a `config.yaml` — i.e. probably a self-describing artifact
    #: whose weight identity can be checked rather than asserted.
    has_config: bool
    #: Sits beside a `*_convergence.csv`.
    has_convergence: bool


@dataclass
class RepoPreview:
    repo_id: str
    revision: str
    candidates: List[RemoteFile]


def preview_repo(
    repo_id: str, revision: Optional[str] = None, token: Optional[str] = None
) -> RepoPreview:
    """List the files in a repo that could be a lens, with sizes.

    READ-ONLY, AND THE POINT IS TO SPEND A REQUEST INSTEAD OF A DOWNLOAD. A
    mistyped path would otherwise cost a multi-GB fetch and a slot on the
    single-GPU queue before anything noticed.

    THE REVISION IS RESOLVED HERE. `hf_hub_download` without one takes `main`,
    which moves — so "acquired from <repo>" would not be a reproducible
    statement. The caller passes the resolved sha back when it downloads, so the
    file that was previewed is the file that arrives.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, revision=revision, repo_type="model", files_metadata=True)
    resolved = info.sha or revision or "main"

    siblings = list(getattr(info, "siblings", None) or [])
    all_paths = {s.rfilename for s in siblings}
    dirs_with_config = {
        p.rsplit("/", 1)[0] if "/" in p else "" for p in all_paths if p.endswith("config.yaml")
    }
    dirs_with_csv = {
        p.rsplit("/", 1)[0] if "/" in p else ""
        for p in all_paths
        if p.endswith("_convergence.csv")
    }

    candidates: List[RemoteFile] = []
    for sibling in siblings:
        name = sibling.rfilename
        if not name.endswith(CANDIDATE_SUFFIXES):
            continue
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        lfs = getattr(sibling, "lfs", None)
        candidates.append(
            RemoteFile(
                path=name,
                # LFS carries the real size; `size` is the pointer file's.
                size_bytes=(getattr(lfs, "size", None) if lfs else None)
                or getattr(sibling, "size", None),
                sha256=getattr(lfs, "sha256", None) if lfs else None,
                has_config=parent in dirs_with_config,
                has_convergence=parent in dirs_with_csv,
            )
        )
    candidates.sort(key=lambda c: (not c.has_config, c.path))
    return RepoPreview(repo_id=repo_id, revision=resolved, candidates=candidates)


def check_free_space(*paths: Path, needed_bytes: int) -> None:
    """Refuse a download that would not fit, BEFORE fetching a byte.

    EVERY VOLUME IT TOUCHES. The HuggingFace cache and the artifact registry can
    be different mounts, and the file lands on both — once as a cached blob and
    once as the staged artifact. Checking only the destination passes and then
    fills the cache volume instead.
    """
    import shutil as _shutil

    for path in paths:
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = _shutil.disk_usage(probe).free
        if free < needed_bytes + MIN_FREE_DISK_BYTES:
            raise AcquisitionRefused(
                f"{path} has {free / 2**30:.1f} GiB free; this needs "
                f"{needed_bytes / 2**30:.1f} GiB plus a "
                f"{MIN_FREE_DISK_BYTES / 2**30:.0f} GiB floor"
            )


def fetch_file(
    repo_id: str,
    path_in_repo: str,
    revision: str,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download one file at a PINNED revision, into the HF cache.

    NOT INTO THE ARTIFACT REGISTRY. `list_artifacts` excludes only `.staging`,
    `.superseded` and `.swap` — a scratch directory anywhere else under the root
    holding a conformant `*_jacobian_lens.pt` would be discovered and SERVED as
    a second artifact under a bogus slug. The registry is the filesystem, so
    anything written there is published.

    The cache also gives resume, dedup and etag validation for free.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            revision=revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def fetch_optional(
    repo_id: str,
    path_in_repo: str,
    revision: str,
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """A sibling file that may not exist. Absence is a real state, not an error.

    Community repos ship a bare `.pt` with no config and no convergence trace,
    and that is a lens miStudio can still adopt — as UNVERIFIED, which is the
    honest record.
    """
    try:
        return fetch_file(repo_id, path_in_repo, revision, token, cache_dir)
    except Exception as exc:  # noqa: BLE001 - absence is expected, not a failure
        logger.info("No %s in %s@%s (%s)", path_in_repo, repo_id, revision[:8], exc)
        return None


def sibling_paths(lens_path: str) -> Dict[str, str]:
    """Where the config and convergence trace sit relative to a lens file.

    Spec §2.1 puts all three in one directory, so this is a directory join
    rather than a search. The convergence file's stem follows the LENS file's,
    not the directory's — `gpt2-small/` holds `gpt2_convergence.csv` — because
    upstream derives both from the HuggingFace id while the directory carries
    the publisher's own model name.
    """
    parent = lens_path.rsplit("/", 1)[0] if "/" in lens_path else ""
    stem = lens_path.rsplit("/", 1)[-1]
    for suffix in ("_jacobian_lens.pt", ".pt", ".pth", ".safetensors", ".bin"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    join = (lambda name: f"{parent}/{name}") if parent else (lambda name: name)
    return {
        "config": join("config.yaml"),
        "convergence": join(f"{stem}_convergence.csv"),
    }

