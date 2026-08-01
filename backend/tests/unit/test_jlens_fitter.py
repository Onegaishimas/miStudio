"""
Fitter correctness against a KNOWN Jacobian.

A fitted lens cannot be checked by looking at it. Every failure mode in this
area produces a well-shaped tensor of plausible magnitude: the wrong hook point,
an unfrozen norm, a transposed accumulation, an unweighted shard merge. So the
fixture here is a stack whose exact Jacobian is known analytically — a
composition of linear blocks, where `J = W_n ... W_{l+1}` — and the assertion is
equality with that product, not a smoke test.

MUTATION CONTROLS (each must turn this file red):
  * hook the norm module instead of layers_module[L]  -> "hook target" fails
  * transpose the assembled Jacobian                  -> "known Jacobian" fails
  * drop weighting from merge_shards                  -> "shard merge" fails
  * accept a corpus below the floor                   -> "corpus floor" fails
  * converge on a readout proxy instead of J          -> "convergence signal" fails
"""

from __future__ import annotations

import pytest
import torch

from src.ml.jlens_fitter import (
    MIN_PROMPTS,
    JacobianFitter,
    linearisation_residual,
    jacobian_batched,
    jacobian_by_jvp,
    merge_shards,
    relative_change,
)


class ScaleNorm(torch.nn.Module):
    """A normalisation stand-in: class name carries "norm", applies a scale.

    DELIBERATELY A DISTINCT, NON-UNIT SCALE. A block built as
    `Linear . Norm` is the shape that separates the two hook points: capturing
    at the decoder layer's output includes the NEXT block's norm in the
    downstream map, and capturing at that norm excludes it. With a unit scale
    the two coincide and a negative control on the hook target proves nothing —
    the "fixtures agree by construction" trap.
    """

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale.clone())

    def forward(self, hidden):
        return hidden * self.scale


class NormedBlock(torch.nn.Module):
    """Decoder-layer stand-in: normalise, then a linear map. Returns a tuple."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.input_norm = ScaleNorm(scale)
        self.weight = torch.nn.Parameter(weight.clone())

    def forward(self, hidden):
        return (self.input_norm(hidden) @ self.weight.T,)


class TinyStack(torch.nn.Module):
    def __init__(self, weights, scales):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [NormedBlock(w, s) for w, s in zip(weights, scales)]
        )
        self.embed = torch.nn.Embedding(16, weights[0].shape[1])

    def forward(self, input_ids=None, **_):
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)[0]
        return hidden


class Structure:
    """Minimal TransformerStructure stand-in."""

    def __init__(self, stack: TinyStack):
        self.layers_module = stack.blocks
        self.num_layers = len(stack.blocks)
        self.attention_module = None
        # Present, plausible, and the WRONG thing to hook — the trap this module
        # exists to avoid, reproduced faithfully.
        self.residual_norm_module = stack.blocks[0].input_norm


class Tokenizer:
    def __call__(self, text, return_tensors=None):
        ids = [(ord(c) % 15) + 1 for c in text[:6]] or [1]
        return {"input_ids": torch.tensor([ids])}


D = 6


def make_stack(seed: int = 0):
    torch.manual_seed(seed)
    weights = [torch.randn(D, D) * 0.3 for _ in range(4)]
    # Scales well away from 1 so the two hook points cannot coincide.
    scales = [torch.rand(D) * 2.0 + 0.5 for _ in range(4)]
    return TinyStack(weights, scales), weights, scales


def analytic_jacobian(weights, scales, layer: int) -> torch.Tensor:
    """d h_final / d h_layer = product of (W_i . diag(s_i)) for i > layer."""
    j = torch.eye(D)
    for w, s in zip(weights[layer + 1 :], scales[layer + 1 :]):
        j = (w @ torch.diag(s)) @ j
    return j


def batchable(w, bias=None):
    """A map accepting [d] or [n, d], like the real sub-network."""

    def fn(h):
        out = h @ w.T if h.dim() > 1 else w @ h
        return out if bias is None else out + bias

    return fn


def test_batched_extraction_matches_a_known_linear_map():
    torch.manual_seed(1)
    w = torch.randn(D, D)
    j = jacobian_batched(batchable(w), torch.randn(D), chunk=2)
    assert torch.allclose(j, w, atol=1e-4)


def test_batched_extraction_recovers_J_from_an_AFFINE_map():
    """Blocks have biases, so the map is affine, not linear.

    Subtracting fn(0) is what makes the extraction exact. Without it every
    column carries the bias and the lens is wrong by a constant that looks like
    signal.
    """
    torch.manual_seed(9)
    w = torch.randn(D, D)
    bias = torch.randn(D) * 3.0
    j = jacobian_batched(batchable(w, bias), torch.randn(D), chunk=3)
    assert torch.allclose(j, w, atol=1e-4)


def test_batched_extraction_agrees_with_the_jvp_reference():
    """The fast path assumes affineness; the reference assumes nothing.

    d_model jvp calls per layer per prompt is millions of forward passes on a
    real model, so the batched path is what runs. Its assumption is therefore
    verified against the general method rather than asserted.
    """
    torch.manual_seed(10)
    w = torch.randn(D, D)
    bias = torch.randn(D)
    point = torch.randn(D)
    fast = jacobian_batched(batchable(w, bias), point, chunk=4)
    reference = jacobian_by_jvp(batchable(w, bias), point)
    assert torch.allclose(fast, reference, atol=1e-4)


def test_linearisation_residual_is_zero_for_a_map_that_really_is_linear():
    """A DIAGNOSTIC, not a gate — and the distinction was a real correction.

    The earlier `affine_residual` compared a GLOBAL affine prediction against
    the map, on the premise that freezing attention and norms makes the
    residual-to-residual map affine. It does not: the MLP activation stays
    non-linear. On the first real fit that check measured 40.3 against a 1e-3
    limit and refused a perfectly good fit.

    A Jacobian IS a local linearisation. What is worth recording is how far it
    holds LOCALLY, which is what this measures.
    """
    torch.manual_seed(11)
    w = torch.randn(D, D)
    point = torch.randn(D)

    linear = batchable(w)
    assert linearisation_residual(linear, point, jacobian_batched(linear, point)) < 1e-4


def test_linearisation_residual_is_LARGER_for_a_curved_map():
    """Informative, not disqualifying: it says how local the lens is."""
    torch.manual_seed(12)
    w = torch.randn(D, D)
    point = torch.randn(D)

    def curved(h):
        base = h @ w.T if h.dim() > 1 else w @ h
        return base + torch.tanh(base * 3.0) * 5.0

    j = jacobian_batched(curved, point)
    straight = jacobian_batched(batchable(w), point)

    assert linearisation_residual(curved, point, j) > linearisation_residual(
        batchable(w), point, straight
    )


def test_the_jacobian_is_the_DERIVATIVE_not_a_secant():
    """The correction a hardware run forced.

    For a CURVED map the secant `fn(e_i) - fn(0)` and the derivative at the
    point are different matrices. The secant is what the first implementation
    computed, on a premise about freezing that was wrong — and it is
    well-shaped, plausible and not a Jacobian.
    """
    torch.manual_seed(13)
    w = torch.randn(D, D)
    point = torch.randn(D) * 2.0

    def curved(h):
        base = h @ w.T if h.dim() > 1 else w @ h
        return base + torch.tanh(base) * 3.0

    derivative = jacobian_batched(curved, point)
    reference = jacobian_by_jvp(curved, point)

    # The production path agrees with forward-mode AD at the point...
    assert torch.allclose(derivative, reference, atol=1e-4)

    # ...and does NOT agree with the secant, which is the point of the fix.
    zero = curved(torch.zeros_like(point))
    secant = torch.stack(
        [
            curved(torch.eye(D)[i]) - zero
            for i in range(D)
        ],
        dim=1,
    )
    assert not torch.allclose(derivative, secant, atol=1e-2), (
        "the fixture is not curved enough to tell a secant from a derivative"
    )


def test_extraction_is_not_the_transpose():
    """A transposed assembly is symmetric-looking and passes a norm check."""
    torch.manual_seed(2)
    w = torch.randn(D, D)
    j = jacobian_batched(batchable(w), torch.randn(D), chunk=3)
    assert not torch.allclose(j, w.T, atol=1e-3), "fixture is accidentally symmetric"
    assert torch.allclose(j, w, atol=1e-4)


def test_fit_recovers_the_known_jacobian_at_every_layer():
    stack, weights, scales = make_stack(3)
    fitter = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=2, chunk=3
    )
    result = fitter.fit([f"prompt {i}" for i in range(4)])

    assert result.d_model == D
    for layer in range(len(weights)):
        expected = analytic_jacobian(weights, scales, layer)
        got = result.jacobians[layer].to(torch.float32)
        assert torch.allclose(got, expected, atol=2e-2), f"layer {layer}"


class NormHookedFitter(JacobianFitter):
    """The WRONG fitter: sub-network starts after the next block's norm.

    This is what hooking `residual_norm_module` actually does — the norm's
    rescaling ends up OUTSIDE the fitted map instead of inside it. Reproduced
    here rather than described, because in production the difference is
    plausible numbers and no error at all.
    """

    def _sub_network(self, input_ids, layer):
        if layer + 1 >= self.structure.num_layers:
            # No next block to mis-hook; the final layer is unaffected either
            # way, which is itself worth knowing — the defect hides in depth.
            return super()._sub_network(input_ids, layer)
        captured = {}

        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["h"] = hidden[0, -1].detach().clone()

        # The norm INSIDE the next block, not the block's own output.
        target = self.structure.layers_module[layer + 1].input_norm
        handle = target.register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            handle.remove()

        def forward(h):
            batched = h.dim() > 1
            hidden = h.reshape(-1, 1, h.shape[-1]) if batched else h.view(1, 1, -1)
            # Skips the norm the correct map would include.
            hidden = hidden @ self.structure.layers_module[layer + 1].weight.T
            for idx in range(layer + 2, self.structure.num_layers):
                result = self.structure.layers_module[idx](hidden)
                hidden = result[0] if isinstance(result, tuple) else result
            return hidden.reshape(hidden.shape[0], -1) if batched else hidden.view(-1)

        return captured["h"], forward


def test_hook_target_is_the_decoder_layer_not_a_norm():
    """Negative control for the trap that cost this project an increment.

    Hooking the norm module leaves the norm's rescaling OUTSIDE the fitted map,
    so the lens no longer equals the analytic product. In production the
    symptom is not an error — it is a well-shaped artifact with the signal
    scaled out, which is why this is a control rather than a comment.
    """
    stack, weights, scales = make_stack(4)
    structure = Structure(stack)

    correct = (
        JacobianFitter(stack, Tokenizer(), structure, freeze_qk=False, min_prompts=1, chunk=3)
        .fit(["abc"])
        .jacobians[0]
        .to(torch.float32)
    )
    wrong = (
        NormHookedFitter(stack, Tokenizer(), structure, freeze_qk=False, min_prompts=1, chunk=3)
        .fit(["abc"])
        .jacobians[0]
        .to(torch.float32)
    )

    expected = analytic_jacobian(weights, scales, 0)
    assert torch.allclose(correct, expected, atol=2e-2)
    assert not torch.allclose(wrong, expected, atol=2e-2), (
        "hooking the norm produced the same lens as hooking resid_post — the "
        "control does not bite, so a wrong hook target would ship undetected"
    )


def test_corpus_floor_is_refused_not_warned():
    stack, _, _ = make_stack(5)
    fitter = JacobianFitter(stack, Tokenizer(), Structure(stack), freeze_qk=False)
    with pytest.raises(ValueError, match=str(MIN_PROMPTS)):
        fitter.fit(["one", "two"])


def test_shard_merge_is_weighted_by_prompt_count():
    """An unweighted mean over-weights a short shard, silently."""
    a = {0: torch.full((2, 2), 1.0)}
    b = {0: torch.full((2, 2), 3.0)}

    merged = merge_shards([a, b], [300, 100])
    assert torch.allclose(merged[0], torch.full((2, 2), 1.5))

    unweighted = merge_shards([a, b], [1, 1])
    assert torch.allclose(unweighted[0], torch.full((2, 2), 2.0))
    assert not torch.allclose(merged[0], unweighted[0])


def test_shard_merge_rejects_mismatched_layer_sets():
    with pytest.raises(ValueError, match="different layer sets"):
        merge_shards([{0: torch.zeros(2, 2)}, {1: torch.zeros(2, 2)}], [1, 1])


def test_convergence_signal_is_a_property_of_j_alone():
    """The stopping rule must not consult a readout.

    BR-004 forbids next-token agreement as a quality metric, and every
    readout-quality proxy drifts toward it. `relative_change` takes only the
    accumulated Jacobians, which is enforced here by its signature having
    nowhere to put a model.
    """
    prev = {0: torch.ones(3, 3)}
    same = {0: torch.ones(3, 3)}
    moved = {0: torch.ones(3, 3) * 2}

    assert relative_change(prev, same) == pytest.approx(0.0)
    assert relative_change(prev, moved) > 0.1
    assert relative_change({}, moved) == float("inf")

    import inspect

    params = set(inspect.signature(relative_change).parameters)
    assert params == {"previous", "current"}, (
        "relative_change gained a parameter; if that parameter is a model or a "
        "readout, the fitter is converging on output agreement (BR-004)"
    )


def test_fitter_module_names_no_architecture():
    """The old SUPPORTED_ARCHITECTURES whitelist is not coming back (BR-032)."""
    import ast
    import inspect

    from src.ml import jlens_fitter

    source = inspect.getsource(jlens_fitter)
    tree = ast.parse(source)
    # Docstrings may name models when explaining WHY; executable code may not.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Name):
            assert not any(
                arch in node.id.lower()
                for arch in ("lfm2", "gemma", "llama", "granite", "qwen", "mistral")
            ), f"architecture name in executable path: {node.id}"


def test_norm_discovery_does_not_capture_a_block_merely_named_for_a_norm():
    """`endswith("norm")`, not `contains`.

    A substring match captures anything NAMED for a norm — `NormedBlock` here,
    a `NormalizedAttention` elsewhere — and freezing a decoder block is not a
    no-op: it replaces the whole block with an elementwise rescaling and yields
    a lens with no error anywhere. This fixture exists because that is exactly
    what happened when the rule was `contains`.
    """
    from src.ml.jlens_fitter import _norm_modules

    stack, _, _ = make_stack(20)
    found = _norm_modules(stack)

    assert found, "no norm modules discovered at all"
    assert all(type(m).__name__ == "ScaleNorm" for m in found), (
        f"captured a non-norm module: {[type(m).__name__ for m in found]}"
    )
    assert not any(isinstance(m, NormedBlock) for m in found)


class LeakyFitter(JacobianFitter):
    """A fitter whose sub-network is NOT affine — an incomplete freeze.

    Reproduces the failure the affine guard exists for: if a norm or an
    attention pattern escapes the freeze, the extracted matrix is a local
    linearisation of nothing in particular, and it is a well-shaped tensor of
    plausible magnitude.
    """

    def _sub_network(self, input_ids, layer):
        point, forward = super()._sub_network(input_ids, layer)

        def leaky(h):
            out = forward(h)
            return out + torch.tanh(out) * 5.0

        return point, leaky


def test_a_nonlinear_sub_network_RECORDS_its_departure_rather_than_being_refused():
    """Corrected after the first real fit.

    The refusal rested on freezing making the map affine. It does not — the
    MLP activation is non-linear — so the check fired on every real model and
    would have blocked every genuine fit. The departure is a property of the
    model worth recording, not a fault worth refusing.
    """
    stack, _, _ = make_stack(21)
    fitter = LeakyFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=1, chunk=3
    )
    # No longer refused — a non-affine sub-network is the NORMAL case on any
    # real model, and refusing it refused every genuine fit. The departure is
    # RECORDED instead, so the artifact says how local its lens is.
    result = fitter.fit(["abc"])
    assert result.jacobians
    assert fitter._last_residuals, "the linearisation residual was not recorded"
    assert max(fitter._last_residuals.values()) > 0.0


def test_a_properly_frozen_fit_is_accepted():
    """Negative control for the guard: it must not refuse a good fit."""
    stack, _, _ = make_stack(22)
    fitter = JacobianFitter(
        stack, Tokenizer(), Structure(stack), freeze_qk=False, min_prompts=1, chunk=3
    )
    assert fitter.fit(["abc"]).jacobians


# ---------------------------------------------------------------------------
# Grouped-query attention under the freeze
#
# The first real fit against a GQA model died in `weights @ value` with an 8-vs-4
# head mismatch. Every test above it was green: the analytic stack has no
# attention at all, and the hardware acceptance run used GPT-2, which is plain
# multi-head attention where n_kv_heads == n_heads. The GQA branch had never
# once executed, so the fixtures agreed by construction.
#
# The shape error is the GOOD failure. The dangerous one is silent: expanding V
# with `repeat` instead of `repeat_interleave` produces a correctly shaped
# result that pairs each query head with the WRONG KV head, and nothing raises.
#
# MUTATION CONTROLS (each must turn this section red):
#   * delete the n_rep expansion entirely  -> "gqa is handled" fails (RuntimeError)
#   * repeat_interleave -> repeat          -> "kv head pairing" fails
#   * n_rep > 1 -> n_rep > 0 (or >= 1)     -> "mha is untouched" fails
# ---------------------------------------------------------------------------


def _reference_kv_repeat(t: torch.Tensor, n_rep: int) -> torch.Tensor:
    """transformers' `repeat_kv`, written out independently of the code under test."""
    b, h, s, d = t.shape
    return t[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@pytest.mark.parametrize(
    "n_heads,n_kv_heads,label",
    [(8, 4, "gqa"), (8, 1, "mqa"), (4, 4, "mha")],
)
def test_frozen_sdpa_handles_every_kv_head_arity(n_heads, n_kv_heads, label):
    """The freeze must survive GQA, MQA and MHA, and match a reference by VALUE.

    Head counts come off the tensors, never off a config or an architecture
    name — a model this repo has never seen must work (BR-032).
    """
    from src.ml.jlens_fitter import frozen_attention_and_norms

    torch.manual_seed(0)
    b, s, d = 1, 5, 6
    n_rep = n_heads // n_kv_heads
    q = torch.randn(b, n_heads, s, d, dtype=torch.float64)
    k = torch.randn(b, n_kv_heads, s, d, dtype=torch.float64)
    v = torch.randn(b, n_kv_heads, s, d, dtype=torch.float64)

    real_sdpa = torch.nn.functional.scaled_dot_product_attention

    # What the unfrozen model computes, with V repeated the way transformers does.
    expected = real_sdpa(q, _reference_kv_repeat(k, n_rep), _reference_kv_repeat(v, n_rep))

    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        # "gqa is handled" / "mha is untouched": this raised RuntimeError
        # ("size of tensor a (8) must match ... b (4)") before the fix.
        got = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, enable_gqa=(n_rep > 1)
        )

    assert got.shape == (b, n_heads, s, d), f"{label}: wrong output shape"
    # "kv head pairing": `repeat` instead of `repeat_interleave` has the right
    # shape here and fails only on these values.
    assert torch.allclose(got, expected, atol=1e-9), f"{label}: wrong attention output"


def test_the_freeze_stops_gradient_at_qk_but_not_at_v_under_gqa():
    """The POINT of the patch, asserted on the arity that broke it.

    A patch that merely stops crashing could equally have stopped freezing.
    """
    from src.ml.jlens_fitter import frozen_attention_and_norms

    torch.manual_seed(1)
    b, n_heads, n_kv, s, d = 1, 8, 4, 4, 6
    q = torch.randn(b, n_heads, s, d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, n_kv, s, d, dtype=torch.float64, requires_grad=True)
    v = torch.randn(b, n_kv, s, d, dtype=torch.float64, requires_grad=True)

    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    out.sum().backward()

    assert v.grad is not None and v.grad.abs().sum() > 0, "V must carry gradient"
    assert q.grad is None or q.grad.abs().sum() == 0, "Q must be frozen"
    assert k.grad is None or k.grad.abs().sum() == 0, "K must be frozen"


def test_the_sdpa_patch_is_removed_on_exit():
    """A leaked global patch would silently freeze attention for every later caller."""
    from src.ml.jlens_fitter import frozen_attention_and_norms

    before = torch.nn.functional.scaled_dot_product_attention
    with frozen_attention_and_norms(torch.nn.Module(), freeze_qk=True):
        assert torch.nn.functional.scaled_dot_product_attention is not before
    assert torch.nn.functional.scaled_dot_product_attention is before
