"""
J-lens artifact lifecycle: discover, load, validate, publish.

THE FILESYSTEM IS THE REGISTRY, not a database table (PADR IDL-46). A J-lens
artifact is consumed by MOUNTING a conformant directory — there is no upload
path, and Neuronpedia's entire J-lens database footprint is two tables
persisting shared analysis sessions. Making a DB row the source of truth would
invent a second registry that can disagree with the one the consumer actually
reads, and the consumer's disagreement is silent.

    <root>/<slug>/<slug>_jacobian_lens.pt
    <root>/<slug>/config.yaml

PUBLISH ONLY AFTER VALIDATION (BR-030). `load_for_readout` refuses an artifact
whose validation did not pass every class, because the failure downstream is an
empty readout rather than an error — indistinguishable from a real readout with
no content, which is the same reason `/jlens/readout` refuses to fabricate one.

STAGE, THEN COMMIT. A fit writes to a staging directory and is moved into place
only once it validates. A half-written artifact in the mounted directory is
served: the loader is best-effort and says nothing about what it found.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .jlens_validation import (
    CheckClass,
    CheckResult,
    CheckStatus,
    ValidationReport,
    check_envelope,
    check_naming,
    check_structural,
)

logger = logging.getLogger(__name__)

STAGING_SUFFIX = ".staging"

#: Where the artifact a commit REPLACED is kept. One slot, overwritten each
#: time: enough to undo the last mistake without letting 276 MB artifacts
#: accumulate silently. Excluded from discovery like staging is.
SUPERSEDED_SUFFIX = ".superseded"

#: Verdict recorded beside the artifact at publish time. Named so it cannot be
#: mistaken for part of the conformance layout — a consumer reading the upstream
#: format ignores it, and `_ref_for` does not require it.
VALIDATION_FILE = "validation.json"


def slug_for(repo_id: str) -> str:
    """The consumer's slug for a HuggingFace id.

    Mirrors the conformance spec's slug function. Kept here rather than inlined
    so the one place it is defined is the one place it can drift.
    """
    slug = repo_id.split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9._-]+", "-", slug).strip("-")
    if not slug:
        raise ValueError(f"{repo_id!r} produces an empty slug")
    return slug


@dataclass
class ArtifactRef:
    """A located artifact. Existence is not validity — see `validate`."""

    slug: str
    directory: Path
    lens_path: Path
    config_path: Optional[Path]

    @property
    def size_bytes(self) -> int:
        return self.lens_path.stat().st_size if self.lens_path.exists() else 0


class ArtifactNotValidated(RuntimeError):
    """Raised rather than serving an artifact that has not passed the suite."""


class ArtifactCoverageLoss(RuntimeError):
    """Raised rather than destroying layers the replacement does not cover."""


class JLensArtifactService:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ---------------------------------------------------------- discovery

    def list_artifacts(self) -> List[ArtifactRef]:
        """Every conformant artifact under the root.

        Staging directories are excluded: an artifact still being written is
        not an artifact, and the whole point of staging is that it is invisible
        until it commits.
        """
        if not self.root.is_dir():
            return []
        found: List[ArtifactRef] = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if directory.name.endswith((STAGING_SUFFIX, SUPERSEDED_SUFFIX)):
                continue
            ref = self._ref_for(directory)
            if ref is not None:
                found.append(ref)
        return found

    def find(self, repo_id: str) -> Optional[ArtifactRef]:
        directory = self.root / slug_for(repo_id)
        return self._ref_for(directory) if directory.is_dir() else None

    def _ref_for(self, directory: Path) -> Optional[ArtifactRef]:
        lens_files = [p for p in directory.glob("*_jacobian_lens.pt")]
        if len(lens_files) != 1:
            # Zero is "not an artifact"; more than one is ambiguous, and the
            # consumer picks among them without saying which.
            return None
        config = directory / "config.yaml"
        return ArtifactRef(
            slug=directory.name,
            directory=directory,
            lens_path=lens_files[0],
            config_path=config if config.exists() else None,
        )

    # --------------------------------------------------------- validation

    def validate(
        self,
        ref: ArtifactRef,
        d_model: int,
        expected_layers: Sequence[int],
        n_vocab: int,
        semantic_result: Optional[CheckResult] = None,
        cross_impl_result: Optional[CheckResult] = None,
        round_trip_result: Optional[CheckResult] = None,
    ) -> ValidationReport:
        """Run every class. The three that need a live consumer are INJECTED.

        They are parameters rather than internal calls because they cannot be
        performed from here: SEMANTIC needs a loaded model, and
        CROSS_IMPLEMENTATION and ROUND_TRIP need a running consumer. Passing
        `None` records them as NOT_RUN, and `ValidationReport.passed` is
        fail-closed — so an artifact validated without them can never be
        published, rather than appearing to have passed a suite it never ran.
        """
        results: List[CheckResult] = [check_naming(ref.directory)]

        payload = self._load_payload(ref)
        if payload is None:
            results.append(
                CheckResult(
                    CheckClass.STRUCTURAL,
                    CheckStatus.FAIL,
                    f"{ref.lens_path} did not deserialize with weights-only loading",
                )
            )
        else:
            results.append(check_structural(payload, d_model, expected_layers))

        results.append(
            check_envelope(
                ref.size_bytes,
                d_model=d_model,
                n_layers=len(list(expected_layers)),
                n_vocab=n_vocab,
            )
        )

        for supplied, check in (
            (semantic_result, CheckClass.SEMANTIC),
            (cross_impl_result, CheckClass.CROSS_IMPLEMENTATION),
            (round_trip_result, CheckClass.ROUND_TRIP),
        ):
            results.append(
                supplied
                if supplied is not None
                else CheckResult(
                    check,
                    CheckStatus.NOT_RUN,
                    "not supplied; this check requires a loaded model or a live consumer",
                )
            )

        return ValidationReport(results)

    def _load_payload(self, ref: ArtifactRef) -> Optional[Dict[Any, Any]]:
        """Weights-only deserialisation.

        `weights_only=True` is not a nicety: an artifact is an untrusted file on
        disk that this process is about to load, and the unrestricted loader
        executes pickled code.
        """
        try:
            return torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - reported as a FAIL, not swallowed
            logger.warning("J-lens artifact %s failed to load: %s", ref.lens_path, exc)
            return None

    # ------------------------------------------------------------ publish

    def staging_dir(self, repo_id: str) -> Path:
        return self.root / f"{slug_for(repo_id)}{STAGING_SUFFIX}"

    def write_staged(
        self, repo_id: str, jacobians: Dict[int, torch.Tensor], config_yaml: str
    ) -> ArtifactRef:
        """Write a fit into staging, where nothing will serve it."""
        slug = slug_for(repo_id)
        staging = self.staging_dir(repo_id)
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        lens_path = staging / f"{slug}_jacobian_lens.pt"
        # SAVE ON CPU, ALWAYS. `torch.save` records each tensor's device, and a
        # fit runs on the GPU — so an artifact written straight from a fit is
        # tagged cuda:0 and raises "Attempting to deserialize object on a CUDA
        # device" for any consumer without one. A J-lens artifact is a PORTABLE
        # DOCUMENT whose whole purpose is to be mounted and read elsewhere; a
        # file that only loads on the machine that produced it is not one.
        #
        # Our own loader passes map_location="cpu" and would never have noticed.
        torch.save(
            {int(k): v.detach().to("cpu") for k, v in jacobians.items()}, lens_path
        )
        (staging / "config.yaml").write_text(config_yaml)

        return ArtifactRef(
            slug=slug,
            directory=staging,
            lens_path=lens_path,
            config_path=staging / "config.yaml",
        )

    def commit(
        self,
        repo_id: str,
        report: ValidationReport,
        allow_coverage_loss: bool = False,
    ) -> ArtifactRef:
        """Move a staged artifact into the mounted directory.

        REFUSES on anything short of a full pass. The mounted directory is read
        by a consumer that reports no errors, so this is the last point at which
        a bad artifact can be stopped by anything at all.
        """
        if not report.passed:
            raise ArtifactNotValidated(
                f"refusing to publish {repo_id}: {report.summary()}. "
                "The consumer fails silently, so an unvalidated artifact "
                "presents as a feature that quietly returns nothing."
            )

        staging = self.staging_dir(repo_id)
        if not staging.is_dir():
            raise FileNotFoundError(f"nothing staged for {repo_id} at {staging}")

        # REFUSE A SILENT LOSS OF COVERAGE. A refit is not automatically an
        # upgrade: the artifact this destroyed covered 16 layers on 120 prompts
        # and the replacement covered 9 on 400 — neither dominates, and nothing
        # told the user they were about to lose seven layers. Losing coverage
        # must be a DECISION, so it is refused unless asked for by name.
        lost, out_of_scope = self._coverage_delta(repo_id, staging)
        if lost and not allow_coverage_loss:
            raise ArtifactCoverageLoss(
                f"refusing to publish {repo_id}: the existing artifact covers "
                f"layers {lost} that this fit does not. Publishing would "
                "destroy them. Re-run with allow_coverage_loss=true if that is "
                "what you want, or fit the missing layers as well."
            )
        if out_of_scope:
            # NOT A LOSS — A RECIPE CHANGE. A layer above the new target has no
            # Jacobian to that target: the path is zero by causality, and the
            # fitter refuses to fit it at all. The old artifact holds those
            # layers only because it targeted a higher block.
            #
            # This used to raise. The refusal told the user to "fit the missing
            # layers as well", which is IMPOSSIBLE under the new target, so a
            # penultimate refit over a final-target artifact could not be
            # published by any action the message suggested. The previous
            # artifact is archived to `.superseded`, so nothing is unrecoverable.
            logger.info(
                "Publishing %s drops layers %s, which are above its %s target and "
                "cannot be fitted under this recipe. The previous artifact is "
                "archived, not deleted.",
                repo_id,
                out_of_scope,
                self.target_layer(self._ref_for(staging)) or "declared",
            )

        final = self.root / slug_for(repo_id)
        if final.exists():
            # ARCHIVE, DO NOT DELETE. This used to be `shutil.rmtree(final)`,
            # and it destroyed a full 16-layer LFM2 lens when a later 9-layer
            # fit published over it — nine minutes of GPU and the reference
            # model's only full-stack artifact, gone with no warning and no way
            # back. One slot, overwritten each time: enough to undo the last
            # mistake without letting 276 MB artifacts pile up unnoticed.
            archive = self.root / f"{slug_for(repo_id)}{SUPERSEDED_SUFFIX}"
            if archive.exists():
                shutil.rmtree(archive)
            final.rename(archive)
            logger.info("Archived the previous %s artifact to %s", repo_id, archive)
        staging.rename(final)

        ref = self._ref_for(final)
        if ref is None:
            raise RuntimeError(f"published {final} is not a conformant artifact directory")

        # RECORD THE VERDICT WITH THE ARTIFACT. The filesystem is the registry
        # (PADR IDL-46), so the report belongs beside the file it describes and
        # not in a database that could disagree with it.
        #
        # Without this the fit's validation was DISCARDED and the readout
        # re-validated from scratch with its own hard-coded fixture — one that
        # assumes a mid-stack fit, so a legitimately validated PARTIAL artifact
        # was refused at read time. The caller chose a fixture appropriate to
        # the layers they fitted; substituting a different one and overruling
        # them is not a stricter check, it is a different question.
        self._write_report(ref, report)
        logger.info("Published J-lens artifact %s", final)
        return ref

    @staticmethod
    def _lens_digest(path: Path) -> str:
        """Content hash of the lens file.

        SIZE AND MTIME ARE NOT AN IDENTITY. A replacement with the same layer
        shapes has the same size, and mtime granularity is coarse enough that a
        file rewritten immediately keeps its timestamp — verified by the test
        below, which failed against the size+mtime version of this check. The
        thing being guarded is "are these the weights that were validated", so
        the guard has to look at the weights.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _coverage_delta(self, repo_id: str, staging: Path) -> Tuple[List[int], List[int]]:
        """Layers the CURRENT artifact has that the staged one does not, SPLIT.

        Returns `(lost, out_of_scope)`:

          * `lost` — layers the new fit could have covered and did not. Real
            loss, and the thing worth refusing over.
          * `out_of_scope` — layers ABOVE the staged fit's target. Their
            Jacobian to that target is zero by causality and the fitter refuses
            to fit them, so their absence is the recipe, not a gap.

        Both empty when there is no current artifact or either side is
        unreadable. Unreadable is treated as "nothing to lose" deliberately: a
        guard that blocks publishing because it could not parse the old file
        turns a corrupt artifact into a permanent obstruction.
        """
        current = self.find(repo_id)
        if current is None:
            return [], []
        staged = self._ref_for(staging)
        if staged is None:
            return [], []
        old = self._load_payload(current)
        new = self._load_payload(staged)
        if not isinstance(old, dict) or not isinstance(new, dict):
            return [], []
        missing = sorted({int(k) for k in old} - {int(k) for k in new})
        if not missing:
            return [], []

        ceiling = self._target_index(staged)
        if ceiling is None:
            # No recipe to appeal to, so nothing may be excused. Fail closed:
            # treating an unreadable recipe as permission to drop layers is how
            # a guard becomes decorative.
            return missing, []
        lost = [l for l in missing if l <= ceiling]
        out_of_scope = [l for l in missing if l > ceiling]
        return lost, out_of_scope

    def _target_index(self, ref: ArtifactRef) -> Optional[int]:
        """Highest layer this artifact's recipe COULD cover, from its own config.

        `None` when either the target or the layer count is unreadable — the
        caller must then excuse nothing.
        """
        target = self.target_layer(ref)
        n_layers = self._config_int(ref, "n_layers")
        if target is None or n_layers is None:
            return None
        return n_layers - 2 if target == "penultimate" else n_layers - 1

    def _config_int(self, ref: ArtifactRef, key: str) -> Optional[int]:
        """One integer field from config.yaml, or None if absent/unparseable."""
        if ref.config_path is None or not ref.config_path.is_file():
            return None
        try:
            for raw in ref.config_path.read_text().splitlines():
                name, _, value = raw.partition(":")
                if name.strip() == key:
                    return int(value.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001
            logger.warning("Could not read %s from %s: %s", key, ref.config_path, exc)
        return None

    def _write_report(self, ref: ArtifactRef, report: ValidationReport) -> None:
        stat = ref.lens_path.stat()
        payload = {
            "lens_file": ref.lens_path.name,
            "size_bytes": stat.st_size,
            "sha256": self._lens_digest(ref.lens_path),
            "summary": report.summary(),
            "passed": report.passed,
            "serviceable": report.serviceable,
            "results": [
                {"check": r.check.value, "status": r.status.value, "detail": r.detail}
                for r in report.results
            ],
        }
        (ref.directory / VALIDATION_FILE).write_text(json.dumps(payload, indent=2))

    def stored_report(self, ref: ArtifactRef) -> Optional[Dict[str, Any]]:
        """The verdict recorded when THIS EXACT FILE was published, if any.

        Returns None when the lens file has changed since — size and mtime are
        compared, so an artifact swapped on disk is revalidated rather than
        served on a verdict that described different weights. Serving a lens
        fitted for other weights is the failure this gate exists to prevent, so
        it must not be possible to smuggle one past by leaving a stale JSON
        file beside it.
        """
        path = ref.directory / VALIDATION_FILE
        if not path.is_file():
            return None
        try:
            stored = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            logger.warning("Unreadable validation report at %s: %s", path, exc)
            return None

        if stored.get("sha256") != self._lens_digest(ref.lens_path):
            logger.info(
                "Validation report for %s describes different weights; revalidating",
                ref.slug,
            )
            return None
        return stored

    # There is deliberately no `discard_staged`. It existed, and its only caller
    # deleted a converged artifact the moment a fixture failed — the expensive
    # half of the work thrown away to save a directory that `write_staged`
    # clears anyway on the next fit. A failed validation leaves the staged fit
    # in place so it can be re-validated for free.

    # ------------------------------------------------------------- serving

    def load_for_readout(
        self, repo_id: str, report: Optional[ValidationReport] = None
    ) -> Dict[int, torch.Tensor]:
        """Tensors for `JacobianTransport`, only if validation is SERVICEABLE.

        `report=None` is refused rather than defaulted to trusting the file.
        Serving an unvalidated artifact is precisely the failure BR-030 exists
        for, and it surfaces as an empty readout rather than an error.

        Gated on `serviceable`, not `passed`: the two consumer-interop classes
        need a live external consumer, so requiring them here would make the
        Jacobian path unreachable from miStudio itself. `commit` still requires
        the full `passed` before anything is published for handover.
        """
        if report is None or not report.serviceable:
            # NAME THE FAILING CLASS. "No serviceable validation report" is true
            # and useless: it does not distinguish a missing report from a
            # failed check, and it took a log dive plus two wrong diagnoses to
            # learn which one had happened. A refusal the user cannot act on is
            # only half a guard.
            why = ""
            if report is not None:
                detail = getattr(report, "failing_detail", None)
                why = f" Failing: {detail() if callable(detail) else report.summary()}."
            raise ArtifactNotValidated(
                f"{repo_id} has no serviceable validation report; refusing to "
                f"serve it.{why} Run the validation suite first — an "
                "unvalidated artifact reads out plausible nonsense rather "
                "than failing."
            )
        ref = self.find(repo_id)
        if ref is None:
            raise FileNotFoundError(f"no J-lens artifact for {repo_id}")
        payload = self._load_payload(ref)
        if payload is None:
            raise ArtifactNotValidated(f"{ref.lens_path} did not deserialize")
        return {int(k): v for k, v in payload.items()}

    def fitted_layers(self, ref: ArtifactRef) -> List[int]:
        """Layers this artifact covers, from config.yaml — no tensor load.

        Falls back to the `layer_scales` keys, which carry one entry per fitted
        layer, so artifacts written before `fitted_layers` existed still answer.
        An empty list means "unknown", not "none": the caller must not render
        that as an artifact covering nothing.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return []
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() != "fitted_layers":
                    continue
                inner = value.strip().strip("[]")
                if not inner:
                    return []
                return sorted(int(p) for p in inner.split(",") if p.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001 - reported
            logger.warning("Unreadable fitted_layers in %s: %s", ref.config_path, exc)
            return []
        return sorted(self.layer_scales(ref))

    def degenerate_layers(self, ref: ArtifactRef) -> List[int]:
        """Layers where the fitted J is the identity — the logit lens, exactly.

        Empty means "none recorded", which for an artifact written before this
        was tracked is genuinely unknown rather than a claim that none exist.
        Consumers must not read empty as "every layer is informative".
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return []
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() != "degenerate_layers":
                    continue
                inner = value.strip().strip("[]")
                if not inner:
                    return []
                return sorted(int(p) for p in inner.split(",") if p.strip())
        except (OSError, ValueError) as exc:  # noqa: BLE001 - reported
            logger.warning("Unreadable degenerate_layers in %s: %s", ref.config_path, exc)
        return []

    def target_layer(self, ref: ArtifactRef) -> Optional[str]:
        """Which block the Jacobian was taken TO, from config.yaml.

        The coverage strip needs it: with a `penultimate` target a COMPLETE fit
        covers 0..N-2, so comparing against the model's layer count would render
        a full artifact as "25/26" and colour it amber — reporting a recipe
        choice as a defect.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return None
        try:
            for raw in ref.config_path.read_text().splitlines():
                key, _, value = raw.partition(":")
                if key.strip() == "target_layer":
                    got = value.strip()
                    return got if got in ("final", "penultimate") else None
        except OSError as exc:  # noqa: BLE001
            logger.warning("Could not read %s: %s", ref.config_path, exc)
        return None

    def layer_scales(self, ref: ArtifactRef) -> Dict[int, float]:
        """Per-layer factors the stored matrices were divided by, from config.yaml.

        Empty when the artifact predates the scale being recorded, or declares
        none. Empty means "no rescale to undo", which is the correct reading:
        `_to_storage_dtype` leaves the scale at 1.0 whenever the matrix fits
        fp16 without help, and that is the common case.

        Parsed with a narrow reader rather than a YAML dependency — the file is
        written by `_config_yaml` as flat `  <layer>: <float>` under a
        `layer_scales:` key, and pulling in a parser to read two lines would be
        a larger surface than the thing it reads.
        """
        if ref.config_path is None or not ref.config_path.is_file():
            return {}
        scales: Dict[int, float] = {}
        in_block = False
        try:
            for raw in ref.config_path.read_text().splitlines():
                if raw.strip() == "layer_scales:":
                    in_block = True
                    continue
                if in_block:
                    if raw[:1] not in (" ", "\t") or not raw.strip():
                        break
                    key, _, value = raw.strip().partition(":")
                    try:
                        scales[int(key)] = float(value)
                    except ValueError:
                        # A malformed entry is skipped, not guessed at: a wrong
                        # scale is worse than none, because it silently changes
                        # every magnitude read through this layer.
                        logger.warning(
                            "Unreadable layer scale %r in %s", raw.strip(),
                            ref.config_path,
                        )
        except OSError as exc:  # noqa: BLE001 - reported, not swallowed
            logger.warning("Could not read %s: %s", ref.config_path, exc)
            return {}
        return scales
