"""
The intervention task, measured the way the source paper measures.

WHAT CHANGED AND WHY. The first version applied a primitive to a captured
activation, pushed it through the Jacobian transport, and reported the mean
absolute displacement in lens space. The paper instead perturbs and "allow[s]
the forward pass to continue", scoring "the fraction of trials in which the swap
places the target-appropriate answer at the top of the model's output
distribution", with "Wilson 95% CIs". No activation-space norm appears as an
effect size anywhere in it.

The deviation was not cosmetic. The transport is linear and `apply_additive` is
`h + s*v`, so `J(h + s*v) - J(h) = s*J(v)` and the activation cancels — the
reported number could not depend on the prompt, the position, or the forward
pass that produced the activation. Confirmed on hardware: "My favorite pet is a"
and "The capital of France is" both returned 0.01739214 to seven significant
figures, while the result advertised `positions: [5]` as though it mattered.

MUTATION CONTROLS:
  * read the logits without running the layers -> "the hook fires" fails
  * hook a norm module instead of the layer    -> "the hook fires" fails
  * score a fixed prompt                       -> "depends on the prompt" fails
  * drop the control arm                       -> "control is run" fails
  * claim rung 3                               -> "rung is 2" fails
"""

import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

D_MODEL = 6
VOCAB = 16
#: MUST BE < D_MODEL. `W_U` here is `torch.eye(VOCAB, D_MODEL)`, whose rows past
#: D_MODEL-1 are all ZERO — a target id of 7 gave a zero direction vector, so no
#: perturbation was possible and every arm scored identically. The first version
#: of these tests could not see that, because none of them checked the
#: intervention had any effect at all.
TARGET_ID = 3


class _Tok:
    """Different prompts tokenise DIFFERENTLY.

    A stub that ignores its input makes every prompt identical, which would let
    the prompt-dependence test pass against a prompt-independent measurement —
    the exact defect this file exists to pin.
    """

    def __call__(self, text, return_tensors=None):
        ids = [(ord(c) % VOCAB) for c in text[:4]] or [0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def encode(self, s, add_special_tokens=False):
        return [TARGET_ID] if s.strip() in ("Paris", "dog") else [7, 8]


class _Layer(torch.nn.Module):
    """A real Module: `register_forward_hook` is what is under test."""

    def forward(self, x):
        return x


class _Model:
    """Embeds ids, RUNS the layers, and unembeds. Hooks fire because the layers
    are called, which is the property the paper's method depends on."""

    device = "cpu"

    def __init__(self, layers):
        self.layers = layers
        self.calls = 0
        torch.manual_seed(0)
        self.embed = torch.randn(VOCAB, D_MODEL)
        self.w_u = torch.randn(VOCAB, D_MODEL)

    def __call__(self, input_ids=None):
        self.calls += 1
        h = self.embed[input_ids]
        for layer in self.layers:
            h = layer(h)
        return types.SimpleNamespace(logits=h @ self.w_u.T)


def _service_stub(tok):
    svc = MagicMock()
    svc.tokenizer = tok
    svc.capture_device = "cpu"
    svc.d_model = D_MODEL
    svc.W_U = torch.eye(VOCAB, D_MODEL)
    return svc


@contextmanager
def _patched(service, layers, model_box=None):
    model = _Model(layers)
    if model_box is not None:
        model_box["model"] = model
    loaded = types.SimpleNamespace(
        name="org/model",
        model=model,
        tokenizer=service.tokenizer,
        structure=types.SimpleNamespace(num_layers=2, layers_module=layers),
        unembedding=service.W_U,
    )
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = object()

    @contextmanager
    def fake_db():
        yield db

    from src.workers.jlens_intervention_tasks import run_intervention_task

    with patch("src.core.database.get_sync_db", fake_db), patch(
        "src.services.jlens_model_registry.load_for_readout", return_value=loaded
    ), patch(
        "src.services.jlens_readout_service.ReadoutService", return_value=service
    ), patch(
        "torch.cuda.is_available", lambda: False
    ), patch.object(
        run_intervention_task, "update_state", MagicMock()
    ):
        yield


def _run(**overrides):
    from src.workers.jlens_intervention_tasks import run_intervention_task

    tok = _Tok()
    service = _service_stub(tok)
    layers = [_Layer(), _Layer()]
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
    model_box = {}
    with _patched(service, layers, model_box):
        return run_intervention_task.run(**kwargs), model_box["model"]


class TestTheForwardPassIsContinued:
    def test_the_perturbation_REACHES_the_running_model(self):
        """Not "a hook was registered" — the OUTPUT has to change.

        An earlier version of this test registered its own hooks on the layers
        and asserted they fired. They fire whether or not the TASK hooks
        anything, because the model runs its layers regardless, so the test
        passed against a task that never registered a hook at all. Deleting the
        registration left it green.

        The observable that cannot be faked: with the perturbation applied, the
        intervened ranks must differ from the baseline ranks. If the hook never
        reaches the model, the two arms are the same forward pass.

        MUTATION CONTROL: remove the `register_forward_hook` loop and this
        fails.
        """
        out, _model = _run(
            prompts=["abcd", "efgh", "ijkl", "mnop"],
            strength=250.0,  # large enough that a rank MUST move
        )
        assert out["baseline_top1"]["hits"] != out["intervened_top1"]["hits"] or (
            out["baseline_top5"]["hits"] != out["intervened_top5"]["hits"]
        ), (
            "the intervened arm scored identically to the baseline: the "
            "perturbation never reached the model's forward pass"
        )

    def test_the_outcome_DEPENDS_ON_THE_PROMPT(self):
        """The defect that motivated the rewrite, pinned.

        The lens-space measurement returned 0.01739214 for both "My favorite pet
        is a" and "The capital of France is" — identical to seven significant
        figures, because `h` cancels out of `J(h + s*v) - J(h)`.

        Asserted on the BASELINE arm, which depends on nothing but the prompt,
        so the test cannot be satisfied by control randomness.

        MUTATION CONTROL: score `prompt` instead of each trial's text and this
        fails — both runs then measure the same string.
        """
        a, _ = _run(prompts=["aaaa", "aaab", "aaac", "aaad"])
        b, _ = _run(prompts=["wxyz", "zyxw", "qrst", "tsrq"])
        assert (a["baseline_top1"]["hits"], a["baseline_top5"]["hits"]) != (
            b["baseline_top1"]["hits"],
            b["baseline_top5"]["hits"],
        ), (
            "two disjoint prompt sets produced identical baselines; the "
            "measurement is not reading the prompt"
        )


class TestTheControlIsRealWork:
    def test_every_trial_runs_THREE_forward_passes(self):
        """Baseline, intervened and control, all on the SAME prompt.

        Asserting `n == n_trials` is not enough: `n` counts trials, so it stays
        correct when an arm is never actually measured. Setting `control_rank`
        to None left that assertion green. The count of forward passes cannot
        be faked the same way.

        MUTATION CONTROL: drop the control arm and this fails at 2 passes/trial.
        """
        out, model = _run(prompts=["abcd", "efgh", "ijkl"])
        assert out["n_trials"] == 3
        assert model.calls == 9, (
            f"{model.calls} forward passes for 3 trials; expected 9 "
            "(baseline + intervened + control each)"
        )
        for arm in ("baseline", "intervened", "control"):
            assert out[f"{arm}_top1"]["n"] == 3

    def test_the_control_construction_is_reconstructible(self):
        """"A random direction" is not a control; "k directions from seed s" is."""
        out, _ = _run(k=4, control_seed=11)
        assert out["control"]["k"] == 4
        assert out["control"]["seed"] == 11
        assert out["control"]["construction"] == "gaussian_unit_norm"

    def test_the_finding_is_reported_as_a_SEPARATION_not_a_bare_rate(self):
        out, _ = _run(prompts=["abcd", "efgh"])
        assert "excess_top1_over_control" in out
        assert "separated_from_control" in out
        assert out["excess_top1_over_control"] == pytest.approx(
            out["intervened_top1"]["rate"] - out["control_top1"]["rate"], abs=1e-9
        )

    def test_a_multi_token_direction_is_REFUSED_not_truncated(self):
        with pytest.raises(ValueError, match="tokens"):
            _run(direction_token="two words")

    def test_a_target_that_is_not_one_token_is_refused(self):
        """A rank in a next-token distribution is defined for a single token."""
        with pytest.raises(ValueError, match="tokens"):
            _run(direction_token="Paris", target_token="two words")


class TestTheClaimIsNotOverstated:
    def test_it_reports_rung_TWO_and_says_what_that_means(self):
        """The perturbation reaches the model and the model's output is read.

        That is a real intervention — rung 2 — where the lens-space version was
        honestly rung 1. It is still one model, one direction and one prompt
        set, and the caveat says so rather than implying generality.

        MUTATION CONTROL: claim rung 3 and this fails.
        """
        out, _ = _run()
        assert out["evidence_rung"] == 2
        assert "forward pass" in out["method"]
        assert "separation" in out["caveat"].lower()
        assert "never that none exists" in out["caveat"]

    def test_the_primitive_and_its_parameters_travel_with_the_result(self):
        """A number whose recipe is unrecorded cannot be reproduced or compared."""
        out, _ = _run(strength=2.5, layers=[0])
        assert out["primitive"] == "additive"
        assert out["parameters"]["strength"] == 2.5
        assert out["layers"] == [0]
        assert out["target_token"] == "Paris"

    def test_an_unknown_primitive_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown primitive"):
            _run(primitive="wishful_thinking")


class TestTheEvidenceIsFiledWithTheLens:
    """Writing a recorder is not calling it.

    This repo shipped 16 MCP tools that were implemented, unit-tested and never
    registered. A `record_intervention_result` with no caller is the same defect:
    the sidecar exists in tests and never appears next to a real artifact.

    MUTATION CONTROLS:
      * drop the `service.record_intervention_result(...)` call -> "it is filed" fails
      * file it when artifact_id is absent                  -> "no artifact" fails
      * omit steering_recipe from the record                -> "recipe" fails
    """

    @contextmanager
    def _recorder(self):
        recorded = []

        class _Svc:
            def __init__(self, _root):
                pass

            def record_intervention_result(self, repo_id, record):
                recorded.append((repo_id, record))

        with patch(
            "src.services.jlens_artifact_service.JLensArtifactService", _Svc
        ), patch(
            "src.api.v1.endpoints.jlens._jacobian_transport",
            return_value=types.SimpleNamespace(lens_type="JACOBIAN_LENS"),
        ):
            yield recorded

    def test_a_run_against_an_artifact_FILES_its_evidence(self):
        with self._recorder() as recorded:
            _run(artifact_id="model", prompts=["abcd", "efgh"])

        assert len(recorded) == 1, (
            f"{len(recorded)} evidence records filed; the measurement ran but "
            "nothing was written beside the lens"
        )
        repo_id, record = recorded[0]
        assert repo_id == "org/model"

        # THE RECIPE, asserted by content. A record that says an effect exists
        # without saying what to apply cannot be used by a consumer.
        recipe = record["steering_recipe"]
        assert recipe["primitive"] == "additive"
        assert recipe["direction_token"] == "Paris"
        assert recipe["layers"] == [0, 1]
        assert "resid_post" in recipe["hook_target"]
        assert record["evidence_rung"] == 2
        assert record["n_trials"] == 2
        assert "separated_from_control" in record["evidence"]

    def test_a_run_WITHOUT_an_artifact_files_nothing(self):
        """A raw unembedding direction has nothing to do with any lens.

        Crediting one for a finding it played no part in would put evidence
        beside a dictionary that was never used to produce it.
        """
        with self._recorder() as recorded:
            _run(prompts=["abcd"])
        assert recorded == []

    def test_a_failure_to_FILE_does_not_lose_the_MEASUREMENT(self):
        """The expensive half must survive the cheap half failing.

        A read-only directory should cost a warning, not the forward passes.
        """

        class _Broken:
            def __init__(self, _root):
                pass

            def record_intervention_result(self, *_a, **_kw):
                raise OSError("read-only file system")

        with patch(
            "src.services.jlens_artifact_service.JLensArtifactService", _Broken
        ), patch(
            "src.api.v1.endpoints.jlens._jacobian_transport",
            return_value=types.SimpleNamespace(lens_type="JACOBIAN_LENS"),
        ):
            out, _ = _run(artifact_id="model", prompts=["abcd"])
        assert out["evidence_rung"] == 2 and out["n_trials"] == 1
