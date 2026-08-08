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
from .task_heartbeat import beat
from . import jlens_progress

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
    #: EXTRA PROMPTS FOR THE SAME EXPERIMENT. The paper reports a FRACTION of
    #: trials — 50 two-hop prompts, 192 swap trials — never one number from one
    #: prompt. A single trial has no interval and cannot be separated from its
    #: control; `prompt` alone is accepted and reported as n=1, which the Wilson
    #: interval will correctly render as almost no information.
    prompts: Optional[List[str]] = None,
    #: The token whose RANK is scored. Defaults to `direction_token`: steering
    #: along a direction and asking whether that token surfaces is the common
    #: case. A coordinate swap wants them different — push direction A, ask
    #: whether answer B arrives.
    target_token: Optional[str] = None,
    direction: Optional[List[float]] = None,
    direction_token: Optional[str] = None,
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
    from ..services.jlens_causal import CausalReport, Trial
    from ..services.jlens_intervention import (
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

        self.update_state(state="PROGRESS", meta=beat({"stage": "loading_model"}))
        # ON THE GPU WHEN THERE IS ONE. A readout is a single forward pass and
        # stays on CPU deliberately; this runs THREE per trial across N trials,
        # which is a different order of work. Released in the `finally` below —
        # and the release drops this frame's reference FIRST, because
        # `clear_cache` runs gc and `empty_cache` immediately, and a live
        # reference here means neither frees anything.
        device = "cuda" if torch.cuda.is_available() else None
        try:
            loaded = load_for_readout(record, capture_device=device)
        except ModelNotAvailable as exc:
            raise ValueError(str(exc)) from exc

    try:
        # RESOLVED FOR PROVENANCE AND VALIDATION, not for measurement. Building it
        # runs the artifact's publish gate, so an unvalidated lens cannot be used to
        # justify an intervention; `lens_type` is then recorded with the result. The
        # measurement itself happens inside the model, not through the transport.
        if artifact_id:
            from ..api.v1.endpoints.jlens import _jacobian_transport

            transport = _jacobian_transport(loaded, artifact_id)
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

        # NO CAPTURE PASS. The lens-space version needed the residuals to transport
        # them; this one perturbs inside the model's own forward pass and never sees
        # them, so capturing would be a full extra pass whose result is discarded.

        if direction is None and direction_token:
            # A TOKEN'S DIRECTION IS ITS UNEMBEDDING ROW. Resolved here rather than
            # in the browser: the client has neither W_U nor any way to produce a
            # d_model vector, which is why this surface had no UI.
            ids = service.tokenizer.encode(direction_token, add_special_tokens=False)
            if not ids:
                raise ValueError(
                    f"{direction_token!r} does not tokenise to anything; there is "
                    "no direction to intervene along"
                )
            if len(ids) > 1:
                # STATED, not silently truncated. A multi-token string has no single
                # direction, and taking the first piece would intervene along
                # something the caller did not name.
                raise ValueError(
                    f"{direction_token!r} is {len(ids)} tokens. A lens direction is "
                    "defined for a SINGLE token; pick one, or pass an explicit "
                    "direction vector."
                )
            named = service.W_U[ids[0]].to(READOUT_DEVICE).to(torch.float32)
        elif direction is not None:
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

        # THE TARGET IS SCORED BY ID, resolved once. A multi-token target has no
        # single rank in a next-token distribution, so it is refused rather than
        # truncated to its first piece — which would score a different token than
        # the caller named.
        wanted = target_token or direction_token
        if not wanted:
            raise ValueError(
                "a target token is required: the finding is the rank of a NAMED "
                "token in the model's output, so there is nothing to score without "
                "one. Pass target_token, or direction_token to use the same token "
                "for both."
            )
        target_ids = service.tokenizer.encode(wanted, add_special_tokens=False)
        if len(target_ids) != 1:
            raise ValueError(
                f"target {wanted!r} is {len(target_ids)} tokens; a rank in a "
                "next-token distribution is defined for a single token"
            )
        target_id = int(target_ids[0])

        trial_prompts = list(prompts) if prompts else [prompt]

        controls = build_control(k=k, seed=control_seed, d_model=service.d_model)

        # ---------------------------------------------------------------- the pass
        # PERTURB, THEN LET THE MODEL RUN. The paper applies the primitive and
        # "allow[s] the forward pass to continue", reading the effect from the
        # model's own output. This used to stop at the lens and report the mean
        # absolute displacement of the transported activation, which measured
        # `s*J(v)` — a quantity independent of the activation, the prompt and the
        # position, because the transport is linear and `apply_additive` is
        # `h + s*v`, so `h` cancels. Two unrelated prompts returned 0.01739214 to
        # seven significant figures.
        #
        # THE HOOK TARGET IS THE WHOLE DECODER LAYER. `structure.layers_module[L]`
        # is resid_post. Hooking the discovered "residual" module instead is a
        # post-attention RMSNorm on LFM2, which renormalises the added vector away —
        # steered output came back byte-identical to unsteered. Same target the
        # serving path uses, deliberately.
        hook_layers = {}
        for L in layers:
            target_module = loaded.structure.layers_module[L]
            if target_module is None:
                raise ValueError(f"No hookable layer module for layer {L} on this model")
            hook_layers[L] = target_module

        def _perturbing_hook(vector, at_positions):
            def hook(_module, _inp, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                if hidden.dim() != 3:
                    return output
                with torch.no_grad():
                    v = vector.to(dtype=hidden.dtype, device=hidden.device)
                    for pos in at_positions:
                        if pos >= hidden.shape[1]:
                            continue
                        h = hidden[0, pos]
                        if chosen is Primitive.PROJECTIVE_ABLATION:
                            hidden[0, pos] = apply_projective_ablation(h, v)
                        else:
                            hidden[0, pos] = apply_additive(h, v, strength)
                return output
            return hook

        def final_rank(text, vector, top_k=50):
            """Rank of the target token in the model's REAL next-token distribution.

            `None` when it falls outside `top_k` — distinct from a large rank, so a
            search cutoff is never reported as a measurement.
            """
            ids = service.tokenizer(text, return_tensors="pt")["input_ids"].to(
                loaded.model.device
            )
            n = int(ids.shape[-1])
            sites = [q if q >= 0 else n + q for q in chosen_positions]
            handles = []
            if vector is not None:
                hook = _perturbing_hook(vector, sites)
                for L in layers:
                    handles.append(hook_layers[L].register_forward_hook(hook))
            try:
                with torch.no_grad():
                    logits = loaded.model(input_ids=ids).logits[0, -1]
            finally:
                for handle in handles:
                    handle.remove()
            order = torch.topk(logits.float(), k=min(top_k, int(logits.shape[-1]))).indices
            hit = (order == target_id).nonzero()
            return int(hit[0, 0]) if hit.numel() else None

        # ------------------------------------------------------------- the trials
        trials = []
        for i, text in enumerate(trial_prompts):
            self.update_state(
                state="PROGRESS",
                meta=beat({"stage": "trials", "trial": i + 1, "of": len(trial_prompts)}),
            )
            jlens_progress.update_row(
                self.request.id,
                status="running",
                progress=100.0 * i / max(len(trial_prompts), 1),
            )
            # ONE CONTROL DIRECTION PER TRIAL, rotating through the seeded set. Over
            # N trials this samples N control directions rather than reusing k, and
            # keeps the cost at three forward passes per trial instead of k + 2.
            control_vector = controls[i % max(k, 1)].to(torch.float32)
            trials.append(
                Trial(
                    prompt=text,
                    baseline_rank=final_rank(text, None),
                    intervened_rank=final_rank(text, named),
                    control_rank=final_rank(text, control_vector),
                )
            )

        summary = CausalReport(
            trials=trials,
            target_token=target_token or direction_token or "<vector>",
            primitive=chosen.value,
            layers=list(layers),
            strength=strength,
        ).summary()

        jlens_progress.update_row(self.request.id, status="completed", progress=100.0)
        return {
            "model": loaded.name,
            "primitive": chosen.value,
            "parameters": {
                "strength": strength,
                "k": k,
                "artifact_id": artifact_id,
                "lens_type": transport.lens_type,
                "positions": chosen_positions,
            },
            "control": {
                "k": k,
                "seed": control_seed,
                "construction": "gaussian_unit_norm",
                "matched": "one direction per trial, rotating through the seeded set",
            },
            **summary,
            # RUNG 2. The perturbation is applied to the residual stream and the
            # model is RUN — this measures the model's behaviour, not the lens's
            # geometry. It remains one model, one direction and one prompt set: it
            # is evidence that this coordinate MOVES this model here, not that it is
            # the only direction that would.
            "evidence_rung": 2,
            "method": (
                "Perturb the residual at the named layers and positions, continue "
                "the forward pass, and score the rank of the target token in the "
                "model's own next-token distribution. Reported as top-1 and top-5 "
                "rates with Wilson 95% intervals, against a matched-norm random "
                "control run on the same prompts."
            ),
            "caveat": (
                "The FINDING is the separation between the intervened and control "
                "rates, not the intervened rate alone. Overlapping intervals mean "
                "no effect was demonstrated here — never that none exists. A "
                "baseline rate near the intervened rate means the prompts were "
                "already answering that way and the intervention moved nothing."
            ),
        }

    finally:
        if device == "cuda":
            from ..services.jlens_model_registry import clear_cache

            # DROP THIS FRAME'S REFERENCE FIRST. `clear_cache` nulls the cache
            # entry then runs gc + `empty_cache()`; with `loaded` still live
            # here, gc collects nothing and no blocks are returned. That exact
            # mistake left 2608 MiB of LFM2 weights resident after a fit.
            loaded = None  # noqa: F841 - the assignment IS the release
            clear_cache()
            logger.info("Released the intervention model from GPU memory")
