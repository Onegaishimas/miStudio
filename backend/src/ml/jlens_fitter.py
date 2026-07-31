"""
Fit a Jacobian lens for any loadable model.

WHAT IS BEING COMPUTED. `J_l` is the Jacobian of the final residual with respect
to the residual at layer `l`, taken with ATTENTION PATTERNS AND NORMALISATION
STATISTICS FROZEN. Freezing is what makes the map linear: with the patterns and
scales held fixed, `h_final = J_l @ h_l` exactly, and `J_l` is a plain
`d_model x d_model` matrix rather than a local linearisation that only holds
near the point it was taken at.

MODEL-AGNOSTIC BY CONSTRUCTION (BR-032, PADR IDL-41). Structure comes from
`discover_transformer_structure`; freezing is applied by patching the operations
themselves, not by knowing which modules an architecture happens to use. There
is deliberately no architecture name in the executable path.

THE HOOK TARGET IS `structure.layers_module[L]`, NEVER A NORM MODULE. On a
hybrid model `residual_norm_module` resolves to a post-attention RMSNorm, and a
lens fitted there is renormalised away — plausible numbers with the signal
scaled out, and no error anywhere. This project has already paid for that
confusion once in steering (PADR IDL-38).

`W_U J` IS NEVER FORMED (BR-006, PADR IDL-42). This module produces `J` alone.

CONVERGENCE IS MEASURED ON `J` ITSELF, never on a readout-quality proxy. Any
such proxy drifts toward next-token agreement, which BR-004 forbids as a quality
metric anywhere in the product — the J-lens is deliberately WORSE on that
measure than the logit lens through most of the network, so a fitter that
optimises for it is optimising for the wrong thing.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Appendix A.2. v0.1 of the BRD assumed ~10 sequences; both reference
# implementations disagree and this is the corrected figure. Fitting fewer is
# refused rather than warned about — an under-fitted lens is indistinguishable
# from a fitted one by inspection.
MIN_PROMPTS = 100

# Basis vectors pushed through the frozen network at once. Memory, not
# correctness: the full identity is d_model^2 and chunking bounds the peak.
DEFAULT_CHUNK = 128

# Convergence: relative Frobenius change in the accumulated mean below this,
# sustained for PATIENCE consecutive shards, stops the fit.
DEFAULT_CONVERGENCE_DELTA = 1e-3
PATIENCE = 2


@dataclass
class FitProgress:
    prompts_seen: int
    last_delta: Optional[float]
    converged: bool


@dataclass
class FitResult:
    """A fitted lens plus everything needed to defend it (BR-007)."""

    jacobians: Dict[int, torch.Tensor]
    d_model: int
    n_layers: int
    prompts_seen: int
    converged: bool
    convergence_delta: float
    deltas: List[float] = field(default_factory=list)

    def size_bytes(self, dtype_bytes: int = 2) -> int:
        return self.d_model * self.d_model * dtype_bytes * len(self.jacobians)


# --------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------


@contextmanager
def frozen_attention_and_norms(model: Any, freeze_qk: bool = True) -> Iterator[None]:
    """Hold attention patterns and normalisation statistics fixed.

    Applied by patching the OPERATIONS rather than the modules, so it works on
    any architecture that reaches them — which is the point of BR-032. Two
    patches:

    * `scaled_dot_product_attention` recomputed with the attention weights
      detached, so gradient flows through V but not through Q/K. This is the
      "frozen Q/K" recipe variant (Appendix A.2 choice 2).
    * every normalisation module's scale computed from detached statistics, so
      the norm behaves as a fixed diagonal rescaling.

    `freeze_qk=False` leaves attention differentiable (the "full" variant) while
    still freezing norms. Both are legitimate recipes and the artifact records
    which was used, per layer — the treatment is INAPPLICABLE on a layer that
    does not attend, which is not the same as unused.
    """
    patched: List[Callable[[], None]] = []

    if freeze_qk:
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

        def frozen_sdpa(query, key, value, *args, **kwargs):
            # Recover the pattern with Q/K detached, then apply it to V. The
            # pattern is a constant; V still carries gradient.
            with torch.no_grad():
                weights = original_sdpa(
                    query,
                    key,
                    torch.eye(
                        value.shape[-2], device=value.device, dtype=value.dtype
                    ).expand(*value.shape[:-1], value.shape[-2]),
                    *args,
                    **kwargs,
                )
            return weights @ value

        torch.nn.functional.scaled_dot_product_attention = frozen_sdpa
        patched.append(
            lambda: setattr(
                torch.nn.functional, "scaled_dot_product_attention", original_sdpa
            )
        )

    handles = [_freeze_norm(m) for m in _norm_modules(model)]

    try:
        yield
    finally:
        for undo in patched:
            undo()
        for handle in handles:
            handle()


def _norm_modules(model: Any) -> List[Any]:
    """Every normalisation module, by class name.

    A name search over the module tree, not an architecture branch: `RMSNorm`,
    `LayerNorm`, `GemmaRMSNorm` and their per-family spellings all END with
    "norm".

    ENDSWITH, not CONTAINS. A substring match also captures anything merely
    NAMED for a norm — a `NormedBlock`, a `NormalizedAttention` — and freezing a
    decoder block is not a no-op: it would replace the whole block with an
    elementwise rescaling and produce a lens with no error anywhere.
    """
    if not hasattr(model, "modules"):
        return []
    return [m for m in model.modules() if type(m).__name__.lower().endswith("norm")]


def _freeze_norm(module: Any) -> Callable[[], None]:
    """Make one norm module use detached statistics; returns the undo."""
    original_forward = module.forward

    def frozen_forward(hidden_states, *args, **kwargs):
        # Run the real norm on a detached input to obtain the scale it WOULD
        # apply, then apply that scale to the live input. The statistics become
        # constants; the linear path stays differentiable.
        with torch.no_grad():
            reference = original_forward(hidden_states.detach(), *args, **kwargs)
        # Second guard, independent of the name rule above: anything that does
        # not return a single tensor is not a norm, whatever it is called. Left
        # untouched rather than coerced — a wrong guess here is silent.
        if not isinstance(reference, torch.Tensor) or not isinstance(
            hidden_states, torch.Tensor
        ):
            return original_forward(hidden_states, *args, **kwargs)
        denom = hidden_states.detach()
        scale = torch.where(
            denom.abs() > 1e-9,
            reference / torch.where(denom.abs() > 1e-9, denom, torch.ones_like(denom)),
            torch.ones_like(denom),
        )
        return hidden_states * scale

    module.forward = frozen_forward
    return lambda: setattr(module, "forward", original_forward)


# --------------------------------------------------------------------------
# Jacobian extraction
# --------------------------------------------------------------------------


def jacobian_of(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """`d fn / d point` as a `[out_dim, in_dim]` matrix.

    Forward-mode, chunked over basis vectors. Forward mode costs one pass per
    INPUT dimension and reverse mode one per OUTPUT dimension; here they are
    both d_model, and forward mode avoids retaining a graph across the whole
    stack.

    Chunking bounds peak memory. It is not an approximation — every basis vector
    is pushed through, just not all at once.
    """
    d_in = point.numel()
    columns: List[torch.Tensor] = []

    for start in range(0, d_in, chunk):
        stop = min(start + chunk, d_in)
        basis = torch.zeros(stop - start, d_in, dtype=point.dtype, device=point.device)
        for row, col in enumerate(range(start, stop)):
            basis[row, col] = 1.0

        for row in range(basis.shape[0]):
            _, tangent = torch.autograd.functional.jvp(
                fn, point, basis[row].view_as(point), create_graph=False
            )
            columns.append(tangent.reshape(-1))

    # columns[i] is d fn / d point_i — the i-th COLUMN of J.
    return torch.stack(columns, dim=1)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def merge_shards(shards: Sequence[Dict[int, torch.Tensor]], weights: Sequence[int]) -> Dict[int, torch.Tensor]:
    """Combine per-shard means into one weighted mean.

    Fitting parallelises by splitting the CORPUS and merging, never by splitting
    the model (BRD v0.3 assumptions). Weighting by prompt count is what makes
    the merge equal to a single run over the concatenated corpus; an unweighted
    mean silently over-weights a short shard.
    """
    if not shards:
        return {}
    if len(shards) != len(weights):
        raise ValueError(f"{len(shards)} shards but {len(weights)} weights")
    total = sum(weights)
    if total <= 0:
        raise ValueError("shard weights sum to zero")

    layers = set(shards[0])
    for s in shards[1:]:
        if set(s) != layers:
            raise ValueError("shards cover different layer sets")

    merged: Dict[int, torch.Tensor] = {}
    for layer in sorted(layers):
        acc = torch.zeros_like(shards[0][layer], dtype=torch.float32)
        for shard, weight in zip(shards, weights):
            acc += shard[layer].to(torch.float32) * weight
        merged[layer] = (acc / total).to(shards[0][layer].dtype)
    return merged


def relative_change(previous: Dict[int, torch.Tensor], current: Dict[int, torch.Tensor]) -> float:
    """Relative Frobenius change across all layers.

    THE CONVERGENCE SIGNAL, and deliberately a property of `J` alone. A
    readout-quality proxy would drift toward next-token agreement, which BR-004
    forbids as a quality metric — the J-lens is meant to be worse on it.
    """
    num = 0.0
    den = 0.0
    for layer, cur in current.items():
        prev = previous.get(layer)
        if prev is None:
            return float("inf")
        num += float(torch.linalg.norm((cur - prev).to(torch.float32)) ** 2)
        den += float(torch.linalg.norm(cur.to(torch.float32)) ** 2)
    if den == 0.0:
        return 0.0
    return (num / den) ** 0.5


class JacobianFitter:
    """Fits `J` per layer over a corpus, with convergence-based stopping."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        structure: Any,
        *,
        freeze_qk: bool = True,
        convergence_delta: float = DEFAULT_CONVERGENCE_DELTA,
        min_prompts: int = MIN_PROMPTS,
        chunk: int = DEFAULT_CHUNK,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.structure = structure
        self.freeze_qk = freeze_qk
        self.convergence_delta = convergence_delta
        self.min_prompts = min_prompts
        self.chunk = chunk

    def fit(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        on_progress: Optional[Callable[[FitProgress], None]] = None,
    ) -> FitResult:
        """Accumulate a mean `J` per layer until it stops moving.

        Refuses a corpus below the floor rather than warning: an under-fitted
        lens is indistinguishable from a fitted one by inspection, and the whole
        point of the validation suite is that structure can be perfect while
        content is absent.
        """
        if len(prompts) < self.min_prompts:
            raise ValueError(
                f"{len(prompts)} prompts is below the floor of {self.min_prompts} "
                "(Appendix A.2). An under-fitted lens looks exactly like a "
                "fitted one; fitting fewer is refused rather than warned about."
            )

        selected = list(layers) if layers is not None else list(range(self.structure.num_layers))
        accumulated: Dict[int, torch.Tensor] = {}
        previous: Dict[int, torch.Tensor] = {}
        deltas: List[float] = []
        stable = 0
        seen = 0

        with frozen_attention_and_norms(self.model, freeze_qk=self.freeze_qk):
            for prompt in prompts:
                per_prompt = self._fit_one(prompt, selected)
                seen += 1
                for layer, mat in per_prompt.items():
                    if layer in accumulated:
                        # Running mean, so peak memory is one J per layer
                        # regardless of corpus size.
                        accumulated[layer] += (mat - accumulated[layer]) / seen
                    else:
                        accumulated[layer] = mat.clone()

                if seen >= self.min_prompts:
                    delta = relative_change(previous, accumulated)
                    deltas.append(delta)
                    stable = stable + 1 if delta < self.convergence_delta else 0
                    previous = {k: v.clone() for k, v in accumulated.items()}
                    if on_progress:
                        on_progress(FitProgress(seen, delta, stable >= PATIENCE))
                    if stable >= PATIENCE:
                        break
                elif on_progress:
                    on_progress(FitProgress(seen, None, False))

        return FitResult(
            jacobians={k: v.to(torch.float16) for k, v in accumulated.items()},
            d_model=int(next(iter(accumulated.values())).shape[0]) if accumulated else 0,
            n_layers=len(accumulated),
            prompts_seen=seen,
            converged=stable >= PATIENCE,
            convergence_delta=self.convergence_delta,
            deltas=deltas,
        )

    def _fit_one(self, prompt: str, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
        """One prompt's contribution: `J` per layer at the final position."""
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        out: Dict[int, torch.Tensor] = {}

        for layer in layers:
            point, forward = self._sub_network(input_ids, layer)
            out[layer] = jacobian_of(forward, point, chunk=self.chunk).detach()
        return out

    def _sub_network(self, input_ids: torch.Tensor, layer: int):
        """The map from resid_post at `layer` to the final residual.

        HOOKED AT `structure.layers_module[layer]` — the decoder layer's own
        output. Not a norm module: see the module docstring.
        """
        captured: Dict[str, torch.Tensor] = {}

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[0, -1].detach().clone()

        handle = self.structure.layers_module[layer].register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            handle.remove()

        point = captured["h"]

        def forward(h: torch.Tensor) -> torch.Tensor:
            """Run the remaining blocks from `layer` onward on a single position.

            Every block after `layer` is applied in order. With attention and
            norms frozen this composition is linear in `h`, which is what makes
            the resulting Jacobian a lens rather than a local approximation.
            """
            hidden = h.view(1, 1, -1)
            for idx in range(layer + 1, self.structure.num_layers):
                result = self.structure.layers_module[idx](hidden)
                hidden = result[0] if isinstance(result, tuple) else result
            return hidden.view(-1)

        return point, forward
