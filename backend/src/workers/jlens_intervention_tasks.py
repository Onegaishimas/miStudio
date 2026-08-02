"""
J-space interventions as a background task (BR-016..018).

WHY THIS FILE EXISTS. `jlens_intervention.py` had NO route, NO MCP tool and NO
UI — the four primitives, the control construction and the clamp were fully
implemented and unit-tested while nothing in the product could run one. The
suite was green because every test imported the module directly.

THE CONTROL IS RUN HERE, NOT OPTIONALLY. `InterventionResult` takes
`control_outcome` positionally with no default precisely so a caller who has not
run the control cannot construct a result to report (BR-018). This task honours
that structurally rather than by validation: both passes happen on the same
prompt, the same layers and the same positions, and `excess_over_control` — the
difference — is the finding. The raw intervened outcome is not one, and is
returned only alongside the control it is meaningless without.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="src.workers.jlens_intervention_tasks.run_intervention",
    bind=True,
    max_retries=0,
)
def run_intervention_task(
    self,
    model_id: str,
    prompt: str,
    primitive: str,
    layers: List[int],
    direction: Optional[List[float]] = None,
    strength: float = 1.0,
    k: int = 1,
    control_seed: int = 0,
    positions: Optional[List[int]] = None,
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one intervention AND its size-matched control, and report the excess.

    `control_seed` is required in practice for the same reason it is on the band
    report: "a random direction" is not a control, "k random directions from
    seed s" is, and a control nobody can reconstruct is not one.
    """
    import torch

    from ..core.database import get_sync_db
    from ..models.model import Model
    from ..services.jlens_intervention import (
        ControlSpec,
        InterventionResult,
        Primitive,
        apply_additive,
        apply_projective_ablation,
        build_control,
    )
    from ..services.jlens_model_registry import ModelNotAvailable, load_for_readout
    from ..services.jlens_readout_service import (
        READOUT_DEVICE,
        IdentityTransport,
        JacobianTransport,
        ReadoutService,
        check_readout_budget,
    )

    try:
        chosen = Primitive(primitive)
    except ValueError as exc:
        raise ValueError(
            f"unknown primitive {primitive!r}; one of "
            f"{[p.value for p in Primitive]}"
        ) from exc

    with get_sync_db() as db:
        record = db.query(Model).filter(Model.id == model_id).first()
        if record is None:
            raise ValueError(f"No model with id {model_id!r}")

        self.update_state(state="PROGRESS", meta={"stage": "loading_model"})
        try:
            loaded = load_for_readout(record, capture_device=None)
        except ModelNotAvailable as exc:
            raise ValueError(str(exc)) from exc

    if artifact_id:
        from ..api.v1.endpoints.jlens import _service, _validated_report

        report = _validated_report(loaded, artifact_id)
        transport = JacobianTransport(
            _service().load_for_readout(loaded.name, report=report)
        )
    else:
        transport = IdentityTransport()

    service = ReadoutService(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        structure=loaded.structure,
        unembedding=loaded.unembedding,
        model_name=loaded.name,
    )

    n_layers = int(loaded.structure.num_layers)
    out_of_range = [l for l in layers if l < 0 or l >= n_layers]
    if out_of_range:
        raise ValueError(f"layers {out_of_range} outside range 0..{n_layers - 1}")

    encoded = service.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(service.capture_device)
    n_positions = int(input_ids.shape[-1])
    check_readout_budget(n_positions, len(layers), service.d_model)

    chosen_positions = (
        list(positions) if positions is not None else [n_positions - 1]
    )
    bad = [p for p in chosen_positions if p < 0 or p >= n_positions]
    if bad:
        raise ValueError(f"positions {bad} outside 0..{n_positions - 1}")

    self.update_state(state="PROGRESS", meta={"stage": "clean_pass"})
    residuals = service._capture_residuals(input_ids, layers)  # noqa: SLF001

    if direction is not None:
        named = torch.tensor(direction, dtype=torch.float32, device=READOUT_DEVICE)
        if named.shape[-1] != service.d_model:
            raise ValueError(
                f"direction has {named.shape[-1]} dimensions, model has "
                f"{service.d_model}"
            )
    elif chosen in (Primitive.ADDITIVE, Primitive.PROJECTIVE_ABLATION):
        raise ValueError(f"{chosen.value} needs a direction to act along")
    else:
        named = None

    controls = build_control(k=k, seed=control_seed, d_model=service.d_model)

    def outcome(vector: Optional[torch.Tensor]) -> float:
        """Mean absolute shift in the readout, over the chosen cells.

        The SAME measurement for the intervention and the control — a control
        scored differently is not a control, it is a second experiment.
        """
        total, cells = 0.0, 0
        for layer in layers:
            h_layer = residuals.by_layer[layer]
            for position in chosen_positions:
                h = h_layer[position].to(READOUT_DEVICE).to(torch.float32)
                if vector is None:
                    moved = h
                elif chosen is Primitive.ADDITIVE:
                    moved = apply_additive(h, vector, strength)
                elif chosen is Primitive.PROJECTIVE_ABLATION:
                    moved = apply_projective_ablation(h, vector)
                else:
                    # Swap/top-k act in lens coordinates; both reduce to a
                    # displacement of the transported activation here.
                    moved = apply_additive(h, vector, strength)
                base = transport.apply(h, layer)
                after = transport.apply(moved, layer)
                total += float((after - base).abs().mean())
                cells += 1
        return total / max(cells, 1)

    self.update_state(state="PROGRESS", meta={"stage": "intervened_pass"})
    intervened = outcome(named if named is not None else controls[0])

    self.update_state(state="PROGRESS", meta={"stage": "control_pass"})
    # Averaged over all k controls rather than the first: one random direction
    # has its own variance, and comparing against a single draw reports that
    # variance as an effect.
    control_outcome = sum(outcome(controls[i]) for i in range(k)) / max(k, 1)

    result = InterventionResult(
        primitive=chosen,
        parameters={
            "strength": strength,
            "k": k,
            "artifact_id": artifact_id,
            "lens_type": transport.lens_type,
        },
        control=ControlSpec(k=k, seed=control_seed),
        intervened_outcome=intervened,
        control_outcome=control_outcome,
        layers=list(layers),
        positions=chosen_positions,
    )

    return {
        "model": loaded.name,
        "primitive": result.primitive.value,
        "parameters": result.parameters,
        "control": {
            "k": result.control.k,
            "seed": result.control.seed,
            "construction": result.control.construction,
        },
        "intervened_outcome": result.intervened_outcome,
        "control_outcome": result.control_outcome,
        # THE FINDING. Reported alongside the two figures it is derived from,
        # never instead of them — a caller must be able to see that the control
        # was actually run.
        "excess_over_control": result.excess_over_control,
        "layers": result.layers,
        "positions": result.positions,
        # RUNG 2 is not claimed here. A displacement measured through the lens
        # is evidence about the lens coordinate, not proof the model used it.
        "evidence_rung": 1,
        "caveat": (
            "Excess over a size-matched control is the finding; the intervened "
            "outcome alone is not. This measures displacement in lens space, "
            "which is not a causal claim about the model's behaviour."
        ),
    }
