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
import threading
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

#: Serialises the process-wide SDPA patch — see `frozen_attention_and_norms`.
_FREEZE_LOCK = threading.Lock()

# How far the frozen sub-network may depart from its own linearisation before
# the fit is refused. Non-zero only to absorb floating-point error: freezing is
# meant to make the map exactly affine, so anything above this means a norm or
# an attention pattern escaped the freeze and the extracted matrix is not a lens.
MAX_AFFINE_RESIDUAL = 1e-3


#: fp16's largest finite magnitude. The contract stores fp16 (Appendix A.1), and
#: a Jacobian that exceeds this saturates to inf on the cast.
FP16_MAX = 65504.0


#: Headroom below fp16's ceiling. A matrix scaled to exactly the maximum has
#: no room for the rounding the cast itself introduces.
FP16_TARGET_PEAK = FP16_MAX / 4.0


def _to_storage_dtype(matrix: torch.Tensor, layer: int):
    """Cast to the contract's fp16, RESCALING so the cast cannot saturate.

    FOUND ON THE FIRST REAL FIT, and it is not a marginal overflow: GPT-2's
    accumulated Jacobian at layer 6 peaks at 1.7e7, roughly 256x fp16's 65504
    ceiling. The naive cast saturated 0.3% of entries to `inf`, and that
    artifact is the worst kind — it deserialises cleanly, is exactly the right
    shape and exactly the right size, passes STRUCTURAL, NAMING and ENVELOPE,
    and every readout taken through it is garbage.

    A recorded per-layer SCALE fixes it without touching the contract's dtype or
    its size arithmetic: the tensor stays fp16 and the envelope bound is
    unchanged. The scale is stored in the artifact's `config.yaml`, so the
    matrix is reconstructible rather than merely smaller.

    Ranking is invariant to a positive scalar, so a readout is unaffected either
    way — but the ARTIFACT must be faithful, because a consumer that multiplies
    by W_U for anything other than ranking would get the wrong magnitudes.
    """
    if not torch.isfinite(matrix).all():
        raise ValueError(
            f"layer {layer}: the accumulated Jacobian is not finite before "
            "casting. The fit diverged; refusing to write it."
        )

    peak = float(matrix.abs().max())
    scale = 1.0
    if peak > FP16_TARGET_PEAK:
        scale = peak / FP16_TARGET_PEAK
        matrix = matrix / scale

    cast = matrix.to(torch.float16)
    if not torch.isfinite(cast).all():
        # Belt and braces: if rescaling somehow failed to make the cast safe,
        # the artifact must not be written. An inf here is undetectable later.
        raise ValueError(
            f"layer {layer}: the fp16 cast still saturated after rescaling "
            f"(peak {peak:.1f}). Refusing to write a non-finite lens."
        )
    return cast, scale


@dataclass
class FitProgress:
    prompts_seen: int
    last_delta: Optional[float]
    converged: bool


@dataclass
class FitResult:
    """A fitted lens plus everything needed to defend it (BR-007)."""

    jacobians: Dict[int, torch.Tensor]
    #: Per-layer factor the stored matrix was divided by so the fp16 cast could
    #: not saturate. 1.0 when no rescaling was needed. Recorded in config.yaml,
    #: because a scaled matrix with an unrecorded scale is simply wrong.
    scales: Dict[int, float]
    d_model: int
    n_layers: int
    prompts_seen: int
    converged: bool
    convergence_delta: float
    deltas: List[float] = field(default_factory=list)
    #: Per-layer mean and worst local-linearisation residual over the CORPUS.
    #: The mean says how local the lens usually is; the max says how bad it
    #: gets, and that is the number a reader should judge it on.
    residual_mean: Dict[int, float] = field(default_factory=dict)
    residual_max: Dict[int, float] = field(default_factory=dict)

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

    # PROCESS-WIDE MUTATION, SERIALISED. Patching
    # `torch.nn.functional.scaled_dot_product_attention` reaches every model in
    # this process, not only the one being fitted. That reach is the point — it
    # is what makes the freeze architecture-agnostic — but it means two
    # concurrent fits would nest their patches and restore each other's
    # originals in the wrong order, leaving attention permanently frozen for
    # everything afterwards, with no error and no way to notice.
    #
    # The task queue serialises fits today, so this guards the invariant rather
    # than a symptom already seen. It is worth closing anyway: the failure is
    # silent, permanent, and would present as "readouts went strange".
    _FREEZE_LOCK.acquire()
    patched.append(_FREEZE_LOCK.release)

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
            # GROUPED-QUERY ATTENTION. Under GQA there are fewer KV heads than
            # query heads, and callers may hand SDPA the un-repeated K/V with
            # `enable_gqa=True` and let it broadcast internally. The recovered
            # pattern then has one row group per QUERY head while `value` still
            # has only the KV heads, and the matmul is a shape error — which is
            # the good case. The head counts are read off the tensors rather
            # than off a config, so this covers MHA (n_rep == 1, a no-op), GQA
            # and MQA without naming an architecture (BR-032).
            n_rep = weights.shape[-3] // value.shape[-3]
            if n_rep > 1:
                # repeat_interleave, not repeat: transformers' repeat_kv expands
                # then reshapes, which places each KV head next to its own query
                # group. `repeat` would tile the whole block and silently pair
                # every query head with the WRONG KV head — same shape, wrong
                # attention output, no error anywhere.
                value = value.repeat_interleave(n_rep, dim=-3)
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
        # Reverse order: the lock was acquired first and must be released
        # last, after every patch it protects has been undone.
        for undo in reversed(patched):
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


def jacobian_by_jvp(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
) -> torch.Tensor:
    """`d fn / d point` by one forward-mode pass per input dimension.

    THE REFERENCE IMPLEMENTATION, not the production path. It makes no
    assumption about `fn` at all, which is exactly what makes it the right thing
    to check the fast path against — and exactly what makes it unusable at
    scale: d_model jvp calls per layer per prompt is millions of forward passes
    on a real model.

    `jacobian_batched` is what actually runs. `test_jlens_fitter` asserts the
    two agree, so the assumption the fast path depends on is verified rather
    than asserted.
    """
    d_in = point.numel()
    columns: List[torch.Tensor] = []
    for i in range(d_in):
        tangent_in = torch.zeros_like(point).reshape(-1)
        tangent_in[i] = 1.0
        _, tangent = torch.autograd.functional.jvp(
            fn, point, tangent_in.view_as(point), create_graph=False
        )
        columns.append(tangent.reshape(-1))
    return torch.stack(columns, dim=1)


def jacobian_batched(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """`d fn / d point` AT THE POINT, by vectorised automatic differentiation.

    CORRECTED AFTER A HARDWARE RUN. The first version computed
    `J[:, i] = fn(e_i) - fn(0)` — secants — on the premise that freezing
    attention and normalisation makes `fn` AFFINE. It does not. Freezing removes
    the strongly input-dependent parts, but the MLP's activation (GELU on the
    model this was first run against) stays non-linear, so the residual-to-
    residual map is not affine and never was.

    A secant is not a Jacobian for a non-affine map. It answers "where does a
    unit step land", which for a curved map is a different question and gives a
    matrix that is plausible, well-shaped, and wrong.

    On the first real fit `affine_residual` measured a departure of 40.3 against
    a 1e-3 limit — the guard from review round 1 catching a premise error rather
    than a coding one, which is the only reason this was found before an
    artifact shipped.

    `vectorize=True` batches the backward passes, so this keeps the speed the
    secant version was reaching for without buying it with the wrong math.
    """
    jac = torch.autograd.functional.jacobian(
        fn, point, vectorize=True, create_graph=False
    )
    return jac.reshape(-1, point.numel())


def linearisation_residual(
    fn: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    jacobian: torch.Tensor,
    step: float = 1e-2,
) -> float:
    """How well `J` predicts `fn` in a NEIGHBOURHOOD of the fitting point.

    A DIAGNOSTIC, recorded with the fit — not a gate. This is the corrected form
    of what was `affine_residual`, and the correction matters:

    The old version compared `J h + fn(0)` against `fn(h)` — a GLOBAL affine
    prediction — on the premise that freezing makes `fn` affine. It does not
    (the MLP activation stays non-linear), so it reported a large departure for
    every real model and would have refused every genuine fit.

    A Jacobian IS a local linearisation; asking it to hold globally is asking
    the wrong question. What is worth recording is how far the linearisation
    holds LOCALLY, which is what makes a lens more or less trustworthy away from
    the exact point it was taken at. Large is informative, not disqualifying.
    """
    with torch.no_grad():
        direction = torch.randn_like(point)
        direction = direction / torch.linalg.norm(direction) * step * float(
            torch.linalg.norm(point)
        )
        predicted = fn(point).reshape(-1) + jacobian @ direction.reshape(-1)
        actual = fn(point + direction).reshape(-1)
        denom = float(torch.linalg.norm(actual.to(torch.float32)))
        if denom == 0.0:
            return 0.0
        return float(torch.linalg.norm((predicted - actual).to(torch.float32)) / denom)


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


def _batch_kwargs(kwargs: Dict[str, Any], batch: int) -> Dict[str, Any]:
    """Reshape recorded layer kwargs to the extraction batch size.

    The reference forward ran with batch 1 and the real sequence length; the
    extraction runs with batch `n` and ONE position. Tensors whose leading
    dimension is the batch are expanded and their sequence dimension truncated
    to the final position — the position the lens is taken at.

    Anything not recognisably batch-shaped is passed through untouched rather
    than reshaped on a guess: a wrong reshape here produces a running model and
    a wrong lens.
    """
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[0] == 1:
            sliced = value[:, -1:] if value.shape[1] > 1 else value
            out[key] = sliced.expand(batch, *sliced.shape[1:])
        elif isinstance(value, tuple) and value and all(
            isinstance(v, torch.Tensor) for v in value
        ):
            # Rotary embeddings arrive as a (cos, sin) tuple.
            out[key] = tuple(_batch_kwargs({"v": v}, batch)["v"] for v in value)
        else:
            out[key] = value
    return out


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
        max_affine_residual: float = MAX_AFFINE_RESIDUAL,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.structure = structure
        self.freeze_qk = freeze_qk
        self.convergence_delta = convergence_delta
        self.min_prompts = min_prompts
        self.chunk = chunk
        self.max_affine_residual = max_affine_residual
        #: Per-layer local-linearisation residual, ACCUMULATED over the corpus.
        #:
        #: This used to hold the most recent prompt's value only — overwritten
        #: on every prompt — while being written into the artifact as though it
        #: described the fit. A hundred-prompt corpus reported the hundredth
        #: prompt's number. Mean and max are both kept: the mean says how local
        #: the lens is typically, the max says how bad it gets, and a lens is
        #: trusted or not on the second.
        self._last_residuals: Dict[int, float] = {}
        self._residual_sums: Dict[int, float] = {}
        self._residual_max: Dict[int, float] = {}
        self._residual_counts: Dict[int, int] = {}

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

        cast_and_scale = {
            k: _to_storage_dtype(v, k) for k, v in accumulated.items()
        }
        return FitResult(
            jacobians={k: cast_and_scale[k][0] for k in cast_and_scale},
            scales={k: cast_and_scale[k][1] for k in cast_and_scale},
            d_model=int(next(iter(accumulated.values())).shape[0]) if accumulated else 0,
            n_layers=len(accumulated),
            prompts_seen=seen,
            converged=stable >= PATIENCE,
            convergence_delta=self.convergence_delta,
            deltas=deltas,
            residual_mean={
                l: self._residual_sums[l] / max(self._residual_counts[l], 1)
                for l in self._residual_sums
            },
            residual_max=dict(self._residual_max),
        )

    @property
    def device(self) -> torch.device:
        """The device the MODEL is on, taken from the model itself.

        Not a constructor argument and not inherited from the ambient default:
        a fitter told one device while the model sits on another produces
        `Expected all tensors to be on the same device` at the embedding, and
        only when a GPU is actually present. Every CPU test passes, because
        there the two agree by accident.
        """
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    def _fit_one(self, prompt: str, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
        """One prompt's contribution: `J` per layer at the final position."""
        encoded = self.tokenizer(prompt, return_tensors="pt")
        # MOVED TO THE MODEL'S DEVICE. The tokenizer always returns CPU
        # tensors; a model on CUDA then fails inside index_select at the
        # embedding. This is invisible on a CPU-only test stack.
        input_ids = encoded["input_ids"].to(self.device)
        out: Dict[int, torch.Tensor] = {}

        for layer in layers:
            point, forward = self._sub_network(input_ids, layer)
            jacobian = jacobian_batched(forward, point, chunk=self.chunk).detach()

            # RECORDED, not gated. A Jacobian is a local linearisation by
            # definition, so a non-zero residual is expected on any real model
            # — the MLP activation is non-linear and freezing does not change
            # that. The number says how far the lens can be trusted away from
            # the point it was taken at, which belongs in the artifact's
            # provenance rather than in a refusal.
            residual = linearisation_residual(forward, point, jacobian)
            self._last_residuals[layer] = residual
            self._residual_sums[layer] = self._residual_sums.get(layer, 0.0) + residual
            self._residual_max[layer] = max(
                self._residual_max.get(layer, 0.0), residual
            )
            self._residual_counts[layer] = self._residual_counts.get(layer, 0) + 1
            out[layer] = jacobian
        return out

    def _sub_network(self, input_ids: torch.Tensor, layer: int):
        """The map from resid_post at `layer` to the final residual.

        HOOKED AT `structure.layers_module[layer]` — the decoder layer's own
        output. Not a norm module: see the module docstring.

        The returned callable accepts EITHER a single point of shape [d_model]
        or a batch of shape [n, d_model], because `jacobian_batched` evaluates
        a whole chunk of basis vectors in one call.

        THE DOWNSTREAM LAYERS ARE REPLAYED WITH THEIR REAL KWARGS. A decoder
        layer generally needs position ids, an attention mask and rotary
        embeddings; calling it with hidden states alone either raises or —
        worse on some families — silently takes a default path and fits a lens
        for a model that was never run that way. The kwargs are recorded during
        the reference forward and replayed, so the sub-network is the same
        computation the model actually performed.
        """
        captured: Dict[str, Any] = {}
        recorded_kwargs: Dict[int, Dict[str, Any]] = {}

        def capture_output(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[0, -1].detach().clone()

        def make_kwarg_recorder(idx: int):
            def recorder(_module, _args, kwargs):
                # Positional args beyond hidden_states are not replayed: every
                # transformers decoder layer in this codebase takes them by
                # keyword, and silently dropping one would be invisible.
                recorded_kwargs[idx] = dict(kwargs)
                return None

            return recorder

        handles = [self.structure.layers_module[layer].register_forward_hook(capture_output)]
        for idx in range(layer + 1, self.structure.num_layers):
            handles.append(
                self.structure.layers_module[idx].register_forward_pre_hook(
                    make_kwarg_recorder(idx), with_kwargs=True
                )
            )
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for handle in handles:
                handle.remove()

        point = captured["h"]

        def forward(h: torch.Tensor) -> torch.Tensor:
            """Run the remaining blocks from `layer` onward, on one position.

            With attention and norms frozen this composition is AFFINE in `h`,
            which is what makes the extracted matrix a lens rather than a local
            approximation — and what lets `jacobian_batched` take the whole
            chunk in one call.
            """
            batched = h.dim() > 1
            hidden = h.reshape(-1, 1, h.shape[-1]) if batched else h.view(1, 1, -1)
            for idx in range(layer + 1, self.structure.num_layers):
                kwargs = _batch_kwargs(recorded_kwargs.get(idx, {}), hidden.shape[0])
                result = self.structure.layers_module[idx](hidden, **kwargs)
                hidden = result[0] if isinstance(result, tuple) else result
            return hidden.reshape(hidden.shape[0], -1) if batched else hidden.view(-1)

        return point, forward
