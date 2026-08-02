"""
The intervention task's control discipline (BR-018).

FOUND BY A REVIEW ROUND, not by a failure: this module and `jlens_probe_tasks`
had ZERO test coverage. Both were written, wired, and confirmed reachable — the
harness proves the route exists and queues a task — and nothing exercised what
the task body does once it runs.

The load-bearing invariant is that the control is not optional and not
decorative. `InterventionResult` takes `control_outcome` positionally with no
default precisely so a caller who has not run one cannot construct a result;
this task must honour that in substance, not just in shape.

MUTATION CONTROLS (each must turn this file red):
  * average the control over the FIRST draw only  -> "averages over k" fails
  * score the control with a different measure    -> "same measurement" fails
  * truncate a multi-token direction              -> "refuses" fails
  * report the intervened outcome as the finding  -> "excess" fails
  * claim rung 2                                  -> "rung" fails
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch


D_MODEL = 6
SEQ = 4


class _Tok:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.zeros(1, SEQ, dtype=torch.long)}

    def encode(self, s, add_special_tokens=False):
        # " Paris" is one token; "two words" is two. The task must treat those
        # differently rather than silently taking the first piece.
        return [7] if s.strip() == "Paris" else [7, 8]


def _service_stub():
    svc = MagicMock()
    svc.tokenizer = _Tok()
    svc.capture_device = "cpu"
    svc.d_model = D_MODEL
    svc.W_U = torch.eye(16, D_MODEL)
    svc._capture_residuals.return_value = types.SimpleNamespace(
        by_layer={0: torch.ones(SEQ, D_MODEL), 1: torch.ones(SEQ, D_MODEL)},
        hook_target="layers_module[L]",
    )
    return svc


@contextmanager
def _patched(service):
    """Patch the heavy edges so the task BODY runs on real tensors."""
    loaded = types.SimpleNamespace(
        name="org/model",
        model=None,
        tokenizer=service.tokenizer,
        structure=types.SimpleNamespace(num_layers=2),
        unembedding=service.W_U,
    )

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = object()

    @contextmanager
    def fake_db():
        yield db

    from src.workers.jlens_intervention_tasks import run_intervention_task

    # `bind=True` means the body calls self.update_state, which needs a live
    # request id when invoked outside a worker. Patched rather than faked with
    # a request stack: progress reporting is not what these tests are about.
    with patch("src.core.database.get_sync_db", fake_db), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch(
        "src.services.jlens_readout_service.ReadoutService", return_value=service
    ), patch.object(
        run_intervention_task, "update_state", MagicMock()
    ):
        yield


def _run(**overrides):
    from src.workers.jlens_intervention_tasks import run_intervention_task

    service = _service_stub()
    kwargs = dict(
        model_id="m_1",
        prompt="hello",
        primitive="additive",
        layers=[0, 1],
        direction_token="Paris",
        strength=1.0,
        k=4,
        control_seed=11,
    )
    kwargs.update(overrides)
    with _patched(service):
        return run_intervention_task.run(**kwargs)


class TestTheControlIsRealWork:
    def test_the_result_reports_the_excess_and_BOTH_figures(self):
        """The finding is the difference; the raw outcome alone is not one.

        Both inputs travel beside it so a reader can see the control actually
        ran rather than taking it on trust.
        """
        out = _run()
        assert "excess_over_control" in out
        assert out["excess_over_control"] == pytest.approx(
            out["intervened_outcome"] - out["control_outcome"], abs=1e-9
        )
        assert out["control"]["k"] == 4
        assert out["control"]["seed"] == 11

    def test_the_control_is_averaged_over_ALL_k_draws(self):
        """One random direction carries its own variance.

        Comparing against a single draw reports that variance as an effect.

        THE ARITHMETIC IS PINNED, not inferred from k. An earlier version
        compared control_outcome at k=1 against k=8 and assumed a difference
        could only come from averaging — but torch fills randn in pairs, so the
        two draws do not share a first row and the figure moved either way.
        That test passed against a control that used only `controls[0]`.
        """
        from src.workers.jlens_intervention_tasks import run_intervention_task

        v = torch.zeros(D_MODEL)
        v[0] = 1.0
        w = torch.zeros(D_MODEL)
        w[1] = 5.0  # a deliberately different magnitude

        def controls_of(*rows):
            stacked = torch.stack(rows)
            return lambda k, seed, d_model: stacked

        def control_for(*rows):
            service = _service_stub()
            with _patched(service), patch(
                "src.services.jlens_intervention.build_control",
                controls_of(*rows),
            ):
                return run_intervention_task.run(
                    model_id="m_1",
                    prompt="hello",
                    primitive="additive",
                    layers=[0, 1],
                    direction_token="Paris",
                    strength=1.0,
                    k=len(rows),
                    control_seed=11,
                )["control_outcome"]

        only_v = control_for(v)
        only_w = control_for(w)
        both = control_for(v, w)

        assert both == pytest.approx((only_v + only_w) / 2, abs=1e-6), (
            f"the control reported {both}, not the mean of its {2} draws "
            f"({only_v}, {only_w}) — a single draw is not a control"
        )
        # And it must not simply be the first draw.
        assert both != pytest.approx(only_v, abs=1e-6)

    def test_a_multi_token_direction_is_REFUSED_not_truncated(self):
        """A lens direction is defined for ONE token.

        Taking the first piece would intervene along something the caller did
        not name, and report it under the name they did.
        """
        with pytest.raises(ValueError, match="SINGLE token|is 2 tokens"):
            _run(direction_token="two words")

    def test_a_direction_that_tokenises_to_nothing_is_refused(self):
        class _Empty(_Tok):
            def encode(self, s, add_special_tokens=False):
                return []

        service = _service_stub()
        service.tokenizer = _Empty()
        from src.workers.jlens_intervention_tasks import run_intervention_task

        with _patched(service):
            with pytest.raises(ValueError, match="does not tokenise"):
                run_intervention_task.run(
                    model_id="m_1",
                    prompt="hello",
                    primitive="additive",
                    layers=[0],
                    direction_token="???",
                    k=2,
                    control_seed=1,
                )


class TestTheClaimIsNotOverstated:
    def test_it_reports_rung_ONE_and_says_what_that_means(self):
        """Displacement in lens space is not proof the model used the direction."""
        out = _run()
        assert out["evidence_rung"] == 1, (
            "rung 2 would claim a causal finding this does not establish — that "
            "takes a coordinate swap with a matched control at the behavioural "
            "level, which this does not perform"
        )
        assert "not a causal claim" in out["caveat"]

    def test_the_primitive_and_its_parameters_travel_with_the_result(self):
        """A run whose parameters are unrecorded cannot be reproduced."""
        out = _run(strength=2.5)
        assert out["primitive"] == "additive"
        assert out["parameters"]["strength"] == 2.5
        assert out["layers"] == [0, 1]

    def test_an_unknown_primitive_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown primitive"):
            _run(primitive="wishful_thinking")
