"""
Artifact validation (BR-030, RSK-014).

WHY THIS EXISTS RATHER THAN A TRY/EXCEPT AT THE CONSUMER. The downstream lens
loader is best-effort and fails AT REQUEST TIME WITHOUT RAISING. A bad artifact
therefore presents as a feature that quietly returns nothing — an empty readout
is indistinguishable from a real readout with no content, which is the same
reason `/jlens/readout` refuses to fabricate one. Validation runs BEFORE
handover because after handover there is nothing left to detect.

SIX CLASSES, and each catches something the others cannot:

  STRUCTURAL          it deserializes and has the right shapes
  NAMING              exactly one lens file, named as the consumer expects
  ENVELOPE            its size matches THIS MODEL's arithmetic (BR-006)
  SEMANTIC            it actually recovers a known unspoken intermediate
  CROSS_IMPLEMENTATION our reader and the consumer's agree
  ROUND_TRIP          mounted and served, a Jacobian request returns content

Structure can be perfect while content is absent, which is why SEMANTIC is not
implied by STRUCTURAL. Both readers can be self-consistent while disagreeing
with each other, which is why CROSS_IMPLEMENTATION is not implied by SEMANTIC.
And everything can pass in-process while the mounted artifact is never picked
up, which is why ROUND_TRIP is explicit rather than assumed.

NOTHING HERE SCORES NEXT-TOKEN AGREEMENT (BR-004). The J-lens is deliberately
worse on that measure than the logit lens through most of the network, so a
validation check that rewarded it would reject good artifacts and accept bad
ones. See `test_jlens_validation.py::test_no_check_scores_next_token_agreement`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class CheckClass(str, Enum):
    STRUCTURAL = "structural"
    NAMING = "naming"
    ENVELOPE = "envelope"
    SEMANTIC = "semantic"
    CROSS_IMPLEMENTATION = "cross_implementation"
    ROUND_TRIP = "round_trip"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    # Distinct from FAIL on purpose: a check that could not run has not passed,
    # and collapsing the two either blocks a good artifact or — far worse —
    # counts an unrun check as a pass.
    NOT_RUN = "not_run"


@dataclass
class CheckResult:
    check: CheckClass
    status: CheckStatus
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS


@dataclass
class ValidationReport:
    results: List[CheckResult]

    @property
    def passed(self) -> bool:
        """FAIL-CLOSED: every class must have run and passed.

        A missing class is not a pass. The whole point of the suite is that the
        consumer's failure is silence, so "we did not check" and "we checked and
        it was fine" must never produce the same verdict.
        """
        seen = {r.check for r in self.results}
        if seen != set(CheckClass):
            return False
        return all(r.passed for r in self.results)

    @property
    def missing(self) -> List[CheckClass]:
        return sorted(set(CheckClass) - {r.check for r in self.results}, key=lambda c: c.value)

    def summary(self) -> str:
        parts = [f"{r.check.value}={r.status.value}" for r in self.results]
        for m in self.missing:
            parts.append(f"{m.value}=not_run")
        return ", ".join(parts)


# Consumer-facing filename convention. Anchored at both ends: an unanchored
# pattern accepts `not_a_lens.pt.bak` and this project has already shipped a
# regex that lost an anchor.
LENS_FILENAME = re.compile(r"^([a-z0-9][a-z0-9._-]*)_jacobian_lens\.pt$")


def check_naming(directory: Path) -> CheckResult:
    """Exactly one conformant lens file in the mounted directory.

    "Exactly one" is the check, not "at least one": the consumer picks among
    several without saying which, so two artifacts in a directory is a
    non-deterministic serve, not a convenience.
    """
    if not directory.is_dir():
        return CheckResult(
            CheckClass.NAMING, CheckStatus.FAIL, f"{directory} is not a directory"
        )

    lens_files = sorted(p.name for p in directory.glob("*.pt"))
    conformant = [n for n in lens_files if LENS_FILENAME.match(n)]

    if not conformant:
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            f"no file matching <slug>_jacobian_lens.pt in {directory}",
            {"found": lens_files},
        )
    if len(conformant) > 1:
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            "more than one lens file; the consumer picks among them silently",
            {"found": conformant},
        )
    if len(lens_files) > len(conformant):
        return CheckResult(
            CheckClass.NAMING,
            CheckStatus.FAIL,
            "non-conformant .pt files share the mounted directory",
            {"found": lens_files},
        )
    return CheckResult(
        CheckClass.NAMING, CheckStatus.PASS, conformant[0], {"file": conformant[0]}
    )


def check_structural(payload: Any, d_model: int, expected_layers: Sequence[int]) -> CheckResult:
    """Required keys present, every Jacobian square of side d_model.

    `payload` is what weights-only deserialisation returned. A non-square
    matrix, or one of the wrong side, still loads and still produces a readout
    — of the wrong thing.
    """
    if not isinstance(payload, dict):
        return CheckResult(
            CheckClass.STRUCTURAL,
            CheckStatus.FAIL,
            f"payload is {type(payload).__name__}, expected a mapping of layer -> matrix",
        )

    coerced: Dict[int, Any] = {}
    for key, value in payload.items():
        try:
            coerced[int(key)] = value
        except (TypeError, ValueError):
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer key {key!r} is not coercible to an integer",
            )

    missing = [layer for layer in expected_layers if layer not in coerced]
    if missing:
        return CheckResult(
            CheckClass.STRUCTURAL,
            CheckStatus.FAIL,
            f"missing layers {missing}",
            {"missing": missing},
        )

    for layer, matrix in sorted(coerced.items()):
        shape = tuple(getattr(matrix, "shape", ()))
        if len(shape) != 2 or shape[0] != shape[1]:
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer {layer} has shape {shape}; a Jacobian must be square",
            )
        if shape[0] != d_model:
            return CheckResult(
                CheckClass.STRUCTURAL,
                CheckStatus.FAIL,
                f"layer {layer} has side {shape[0]}, model d_model is {d_model}",
            )

    return CheckResult(
        CheckClass.STRUCTURAL,
        CheckStatus.PASS,
        f"{len(coerced)} layers, all {d_model}x{d_model}",
        {"layers": sorted(coerced)},
    )


def check_envelope(
    size_bytes: int,
    d_model: int,
    n_layers: int,
    n_vocab: int,
    dtype_bytes: int = 2,
    tolerance: float = 1.5,
) -> CheckResult:
    """Size within tolerance of THIS MODEL's arithmetic (BR-006, IDL-42).

    Both bounds are derived, never constants. The required-vs-materialised
    ratio scales with vocabulary — about 32x at a 65k vocab and 111x at 256k —
    so a bound tuned on one model passes on another while missing a real
    materialisation. `n_vocab` is taken as an argument for exactly that reason:
    it is what makes the "did someone materialise W_U J" question answerable.
    """
    required = d_model * d_model * dtype_bytes * n_layers
    materialised = n_vocab * d_model * dtype_bytes * n_layers
    ceiling = int(required * tolerance)

    evidence = {
        "size_bytes": size_bytes,
        "required_bytes": required,
        "materialised_bytes": materialised,
        "ceiling_bytes": ceiling,
        "ratio": round(materialised / required, 1) if required else None,
    }

    if size_bytes > ceiling:
        looks_materialised = size_bytes >= materialised * 0.5
        return CheckResult(
            CheckClass.ENVELOPE,
            CheckStatus.FAIL,
            (
                f"{size_bytes} bytes exceeds the ceiling of {ceiling} "
                + (
                    "and is within range of a MATERIALISED dictionary "
                    f"({materialised} bytes) — W_U J must never be formed"
                    if looks_materialised
                    else "for this model's dimensions"
                )
            ),
            evidence,
        )
    if size_bytes <= 0:
        return CheckResult(
            CheckClass.ENVELOPE, CheckStatus.FAIL, "artifact is empty", evidence
        )
    # A too-SMALL artifact is a truncation, and truncation loads fine.
    if size_bytes < required * 0.5:
        return CheckResult(
            CheckClass.ENVELOPE,
            CheckStatus.FAIL,
            f"{size_bytes} bytes is far below the {required} required — truncated?",
            evidence,
        )
    return CheckResult(
        CheckClass.ENVELOPE, CheckStatus.PASS, f"{size_bytes} within envelope", evidence
    )


def check_semantic(
    readout: Callable[[str, int, int], Sequence[str]],
    prompt: str,
    layer: int,
    expected_intermediate: str,
    top_k: int = 8,
) -> CheckResult:
    """A known UNSPOKEN intermediate is recovered at a mid-band layer.

    Deliberately an intermediate that appears in neither the prompt nor the
    output: a token present in the prompt can be recovered by an artifact that
    encodes nothing at all, so it would pass against a broken lens.
    """
    if expected_intermediate.strip() and expected_intermediate.strip() in prompt:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.FAIL,
            (
                f"fixture is invalid: {expected_intermediate!r} appears in the "
                "prompt, so recovering it proves nothing"
            ),
        )
    try:
        top = list(readout(prompt, layer, top_k))
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return CheckResult(
            CheckClass.SEMANTIC, CheckStatus.FAIL, f"readout raised: {exc}"
        )

    normalised = [t.strip().lower() for t in top]
    if expected_intermediate.strip().lower() in normalised:
        return CheckResult(
            CheckClass.SEMANTIC,
            CheckStatus.PASS,
            f"recovered {expected_intermediate!r} at layer {layer}",
            {"top": top},
        )
    return CheckResult(
        CheckClass.SEMANTIC,
        CheckStatus.FAIL,
        f"{expected_intermediate!r} absent from the top-{top_k} at layer {layer}",
        {"top": top},
    )


def check_cross_implementation(
    ours: Sequence[str], theirs: Optional[Sequence[str]], top_k: int = 5
) -> CheckResult:
    """Our reader and the consumer's agree on the same prompt/layer/top-k.

    `theirs is None` means the comparison could not be made, which is NOT_RUN,
    not PASS. Treating an unreachable consumer as agreement is how a check
    designed to catch silent divergence becomes silent itself.
    """
    if theirs is None:
        return CheckResult(
            CheckClass.CROSS_IMPLEMENTATION,
            CheckStatus.NOT_RUN,
            "consumer unreachable; comparison not made",
        )
    a = [t.strip() for t in list(ours)[:top_k]]
    b = [t.strip() for t in list(theirs)[:top_k]]
    if a == b:
        return CheckResult(
            CheckClass.CROSS_IMPLEMENTATION,
            CheckStatus.PASS,
            f"top-{top_k} identical",
            {"top": a},
        )
    return CheckResult(
        CheckClass.CROSS_IMPLEMENTATION,
        CheckStatus.FAIL,
        f"top-{top_k} differs: ours={a} theirs={b}",
        {"ours": a, "theirs": b},
    )


def check_round_trip(served_readout: Optional[Sequence[str]]) -> CheckResult:
    """Mounted, served, and a Jacobian request came back with content.

    THE CHECK THAT CANNOT BE INFERRED FROM THE OTHERS. Everything upstream can
    pass in-process while the mounted artifact is never picked up, and the
    consumer says nothing about it. An empty result here is a FAIL, not an
    empty pass.
    """
    if served_readout is None:
        return CheckResult(
            CheckClass.ROUND_TRIP,
            CheckStatus.FAIL,
            "served request returned nothing; the artifact was not picked up",
        )
    if not [t for t in served_readout if t.strip()]:
        return CheckResult(
            CheckClass.ROUND_TRIP,
            CheckStatus.FAIL,
            "served readout is empty — indistinguishable from an unmounted artifact",
        )
    return CheckResult(
        CheckClass.ROUND_TRIP,
        CheckStatus.PASS,
        f"served {len(served_readout)} tokens",
        {"top": list(served_readout)},
    )
