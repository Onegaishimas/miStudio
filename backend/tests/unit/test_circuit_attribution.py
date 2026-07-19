"""Attribution correctness pins (016 Task 4.x — FTID §2.6):
1. Pass-through numerical identity (the ε-dropping guard — non-negotiable).
2. Toy-model ANALYTIC gradient check: attr computable by hand.
3. Grouping: one backward per downstream, upstream grads accumulate Σ g·a.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from src.services.circuit_attribution_service import (  # noqa: E402
    PassThroughState,
    attribute_prompt,
    make_passthrough_hook,
)


class IdentitySAE(nn.Module):
    """encode = decode = identity — f IS the residual stream."""

    def encode(self, x):
        return x

    def decode(self, f):
        return f


class LossySAE(nn.Module):
    """Rank-deficient SAE — reconstruction error is NONZERO, so the
    identity pin actually exercises the ε path."""

    def __init__(self, d):
        super().__init__()
        proj = torch.randn(d, d // 2)
        self.register_buffer("P", proj)

    def encode(self, x):
        return x @ self.P

    def decode(self, f):
        return f @ self.P.t() / self.P.shape[0]


class TwoLayerToy(nn.Module):
    """layer1: identity carrier; layer2: x @ W.T — hookable submodules."""

    def __init__(self, d, W):
        super().__init__()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.layer2.weight.copy_(W)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TestPassThroughIdentity:
    def test_output_identical_with_lossy_sae(self):
        """ε-dropping guard: even with a LOSSY SAE the hooked layer output
        must equal the clean output (x̂ + ε ≡ x)."""
        torch.manual_seed(0)
        d = 8
        model = TwoLayerToy(d, torch.randn(d, d))
        x = torch.randn(2, 5, d)
        clean = model(x)
        state = PassThroughState()
        h = model.layer1.register_forward_hook(
            make_passthrough_hook(1, LossySAE(d), state))
        try:
            patched = model(x)
        finally:
            h.remove()
        assert torch.allclose(clean, patched, atol=1e-5), \
            "pass-through changed the forward — ε was dropped"

    def test_code_is_in_graph(self):
        d = 8
        model = TwoLayerToy(d, torch.randn(d, d))
        state = PassThroughState()
        h = model.layer1.register_forward_hook(
            make_passthrough_hook(1, IdentitySAE(), state))
        try:
            _ = model(torch.randn(1, 3, d))
        finally:
            h.remove()
        assert state.codes[1].requires_grad


class TestAnalyticToy:
    def test_attr_matches_hand_computation(self):
        """Identity SAEs on both layers of x→Wx: m = mean_t (Wx)[d] gives
        ∂m/∂x_u(t) = W[d,u]/T, so attr(u→d) = W[d,u] · mean_t x_u(t)."""
        torch.manual_seed(1)
        d, T = 6, 4
        W = torch.randn(d, d)
        model = TwoLayerToy(d, W)
        x = torch.randn(1, T, d)

        state = PassThroughState()
        h1 = model.layer1.register_forward_hook(
            make_passthrough_hook(1, IdentitySAE(), state))
        h2 = model.layer2.register_forward_hook(
            make_passthrough_hook(2, IdentitySAE(), state))
        try:
            _ = model(x)
            u, dn = 2, 3
            scores = attribute_prompt(state, {(2, dn): [(1, u)]})
        finally:
            h1.remove()
            h2.remove()
        expected = float(W[dn, u] * x[0, :, u].mean())
        assert scores[(1, u, 2, dn)] == pytest.approx(expected, rel=1e-4)

    def test_one_backward_covers_all_upstreams_in_group(self):
        torch.manual_seed(2)
        d, T = 6, 3
        W = torch.randn(d, d)
        model = TwoLayerToy(d, W)
        x = torch.randn(1, T, d)
        state = PassThroughState()
        h1 = model.layer1.register_forward_hook(
            make_passthrough_hook(1, IdentitySAE(), state))
        h2 = model.layer2.register_forward_hook(
            make_passthrough_hook(2, IdentitySAE(), state))
        try:
            _ = model(x)
            scores = attribute_prompt(
                state, {(2, 0): [(1, 1), (1, 4)], (2, 5): [(1, 1)]})
        finally:
            h1.remove()
            h2.remove()
        for (ul, ui, dl, di), v in scores.items():
            expected = float(W[di, ui] * x[0, :, ui].mean())
            assert v == pytest.approx(expected, rel=1e-4), (ul, ui, dl, di)
        assert len(scores) == 3
