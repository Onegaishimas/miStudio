"""
Computing a band report from a live model (BR-002), and persisting the gate.

WHERE THE REPORT LIVES. Beside the artifact, in the mounted registry, as
`band-report.json` — not in a database table. Same reasoning as the artifact
itself (PADR IDL-46): the report describes a specific artifact's geometry, so it
travels WITH the artifact rather than in a table that can outlive or contradict
it. A mounted artifact carries its own provenance.

WHAT MAKES A REPORT HONEST. Two of the seven metrics are defined against null
controls, and the controls are part of the metric rather than a sanity extra:
top-1 autocorrelation is meaningless without a position-shuffled null, and
fraction-of-variance-explained is meaningless without a size-matched
random-direction control. The seed for the latter is recorded, because the
EXCESS is the finding and a control nobody can reconstruct cannot be checked.

WHAT IS NOT HERE. No band constant, no default boundaries, no fallback. When
this model's profile does not support a three-way split the report says so and
the product draws nothing (BR-002). And next-token agreement appears in the
per-layer profile as a DESCRIPTION and in no gate condition — the J-lens is
deliberately worse on that measure, so scoring it would fail good models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..ml.jlens_metrics import (
    LayerProfile,
    cross_layer_cka,
    effective_dimensionality,
    excess_kurtosis,
    shuffled_null_autocorrelation,
    top1_autocorrelation,
)
from .jlens_band_report import BandReport, GateDecision, GateRecord, build_band_report

logger = logging.getLogger(__name__)

BAND_REPORT_FILENAME = "band-report.json"
GATE_FILENAME = "gate.json"


def compute_band_report(
    readout_service: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    control_seed: int,
    model_id: str,
) -> BandReport:
    """Measure this model's per-layer profile and derive its own boundaries.

    `control_seed` is a required parameter, not a default: the autocorrelation
    null is drawn from it and a report whose control cannot be reproduced is
    not evidence.
    """
    from .jlens_readout_service import IdentityTransport

    if not prompts:
        raise ValueError("a band report needs at least one prompt")

    # top-1 id per (layer, position), accumulated across prompts.
    per_layer_top1: Dict[int, List[int]] = {layer: [] for layer in layers}
    per_layer_residuals: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for prompt in prompts:
        captured = readout_service._capture_residuals(  # noqa: SLF001 - same concern
            readout_service.tokenizer(prompt, return_tensors="pt")["input_ids"],
            layers,
        )
        for layer in layers:
            residual = captured.by_layer[layer]
            per_layer_residuals[layer].append(residual)
            normed = readout_service._normalize(residual)  # noqa: SLF001
            logits = normed @ readout_service.W_U.T
            per_layer_top1[layer].extend(int(i) for i in logits.argmax(dim=-1))

    profiles: List[LayerProfile] = []
    representations: Dict[int, torch.Tensor] = {}
    for layer in layers:
        stacked = torch.cat(per_layer_residuals[layer], dim=0)
        representations[layer] = stacked

        top1 = per_layer_top1[layer]
        autocorr = top1_autocorrelation(top1) if len(top1) > 1 else None
        null = (
            shuffled_null_autocorrelation(top1, seed=control_seed)
            if len(top1) > 1
            else None
        )

        profiles.append(
            LayerProfile(
                layer=layer,
                kurtosis=excess_kurtosis(stacked.flatten()),
                autocorrelation=autocorr,
                autocorrelation_null=null,
                effective_dimensionality=effective_dimensionality(stacked),
            )
        )

    return build_band_report(
        model_id=model_id,
        profiles=profiles,
        control_seed=control_seed,
        cross_layer_cka=cross_layer_cka(representations) if len(layers) > 1 else {},
    )


def _profile_dict(p: LayerProfile) -> Dict[str, Any]:
    """Serialise one layer, keeping ABSENT distinct from zero.

    A metric that could not be computed is written as null. Coercing it to 0.0
    would be averaged by any consumer and would silently understate — the same
    rule the per-layer applicability follows.
    """
    data = asdict(p)
    data["excess_autocorrelation"] = p.excess_autocorrelation
    return data


def save_band_report(directory: Path, report: BandReport) -> Path:
    path = Path(directory) / BAND_REPORT_FILENAME
    path.write_text(
        json.dumps(
            {
                "model_id": report.model_id,
                "control_seed": report.control_seed,
                # NULL means no bands for this model. There is no default and
                # no fallback: boundaries measured on another model do not
                # transfer, and the product must make porting them impossible
                # by construction (BR-002).
                "boundaries": report.boundaries,
                "derivation": report.derivation,
                "profiles": [_profile_dict(p) for p in report.profiles],
                "cross_layer_cka": {
                    str(i): {str(j): v for j, v in row.items()}
                    for i, row in report.cross_layer_cka.items()
                },
            },
            indent=2,
        )
    )
    return path


def load_band_report(directory: Path) -> Optional[Dict[str, Any]]:
    """Read a stored report, or None when this model has none.

    None is a first-class answer that the client renders as "no bands", never
    as an error and never as an empty band object.
    """
    path = Path(directory) / BAND_REPORT_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001 - reported
        logger.warning("Band report at %s is unreadable: %s", path, exc)
        return None


def save_gate(directory: Path, record: GateRecord) -> Path:
    path = Path(directory) / GATE_FILENAME
    path.write_text(
        json.dumps(
            {
                "model_id": record.model_id,
                # NO_GO stores and reads back exactly like GO. A gate whose
                # negative outcome cannot be persisted is not a gate.
                "decision": record.decision.value,
                "rationale": record.rationale,
                "blocking": record.is_blocking(),
                "has_bands": record.band_report.has_bands,
                "replication_report_id": record.replication_report_id,
            },
            indent=2,
        )
    )
    return path


def load_gate(directory: Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / GATE_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:  # noqa: BLE001
        logger.warning("Gate record at %s is unreadable: %s", path, exc)
        return None
    # Round-trip through the enum so an unrecognised decision is caught here
    # rather than rendering as a plausible string in the UI.
    GateDecision(data["decision"])
    return data
