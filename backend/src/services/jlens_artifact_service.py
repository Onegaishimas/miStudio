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

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
            if directory.name.endswith(STAGING_SUFFIX):
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

    def commit(self, repo_id: str, report: ValidationReport) -> ArtifactRef:
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

        final = self.root / slug_for(repo_id)
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)

        ref = self._ref_for(final)
        if ref is None:
            raise RuntimeError(f"published {final} is not a conformant artifact directory")
        logger.info("Published J-lens artifact %s", final)
        return ref

    def discard_staged(self, repo_id: str) -> None:
        staging = self.staging_dir(repo_id)
        if staging.exists():
            shutil.rmtree(staging)

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
            raise ArtifactNotValidated(
                f"{repo_id} has no serviceable validation report; refusing to "
                "serve it. Run the validation suite first — an unvalidated "
                "artifact reads out plausible nonsense rather than failing."
            )
        ref = self.find(repo_id)
        if ref is None:
            raise FileNotFoundError(f"no J-lens artifact for {repo_id}")
        payload = self._load_payload(ref)
        if payload is None:
            raise ArtifactNotValidated(f"{ref.lens_path} did not deserialize")
        return {int(k): v for k, v in payload.items()}
