"""
Composite (multi-tower) architectures must be described without naming them.

Reported 2026-08-25: gemma-4-12B-it offered no layers on the Training page. Its
stored architecture_config held three keys and no num_hidden_layers, because
the extractor read only top-level attributes and "gemma4_unified" keeps the
decoder nested. Five other models were fine.

The first fix matched a hand-list of sub-config names. That is the pattern this
codebase has been burned by before -- a guard whose scope is narrower than its
claim -- and it would go stale with every new modality. transformers already
maintains both halves of the answer:

  * `type(config).sub_configs` declares every tower and its class
  * `config.get_text_config()` names the language model, returning the config
    itself on a flat model

So a new composite architecture is described the day the library supports it.

THE TRAP THESE PIN. gemma-4 declares audio_config and vision_config alongside
text_config, each with its own geometry. Anything that scans for "the first
nested config with a shape" records another tower's width as the model's -- a
wrong number presented as a measurement, worse than the missing one.
"""

from types import SimpleNamespace

import pytest

from src.ml.model_loader import extract_architecture_config


class TestCompositeModels:
    """Driven through real transformers config classes, not stand-ins."""

    def _gemma3(self):
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config

        return Gemma3Config()

    def test_the_top_level_describes_the_text_tower(self):
        out = extract_architecture_config(self._gemma3())
        assert out["num_hidden_layers"] == out["towers"]["text_config"][
            "num_hidden_layers"
        ]

    def test_every_declared_tower_is_recorded(self):
        out = extract_architecture_config(self._gemma3())
        assert set(out["towers"]) == {"text_config", "vision_config"}, (
            "interpreting a non-text modality needs that tower's shape"
        )

    def test_the_vision_tower_keeps_its_own_depth(self):
        out = extract_architecture_config(self._gemma3())
        vision = out["towers"]["vision_config"]["num_hidden_layers"]
        assert vision != out["num_hidden_layers"], (
            "the towers collapsed to one set of numbers"
        )

    def test_it_says_which_tower_the_top_level_came_from(self):
        out = extract_architecture_config(self._gemma3())
        assert out["text_tower"] == "text_config"

    def test_it_does_not_take_another_tower_over_the_text_one(self):
        """The gemma-4 shape: a sibling tower with a smaller hidden size."""
        out = extract_architecture_config(self._gemma3())
        text_hidden = out["towers"]["text_config"]["hidden_size"]
        assert out["hidden_size"] == text_hidden


class TestFlatModels:
    def test_a_decoder_only_config_is_unchanged(self):
        from transformers.models.llama.configuration_llama import LlamaConfig

        out = extract_architecture_config(
            LlamaConfig(num_hidden_layers=32, hidden_size=4096)
        )
        assert out["num_hidden_layers"] == 32
        assert "towers" not in out
        assert out.get("text_tower") is None


class TestMixtureOfExperts:
    """A different axis from tower nesting: flat config, routed FFNs.

    Without these fields a reader cannot tell a dense 8B from an 8x7B, which
    changes both the memory estimate and what an SAE is being trained on.
    """

    def test_expert_counts_are_recorded(self):
        from transformers.models.mixtral.configuration_mixtral import MixtralConfig

        out = extract_architecture_config(
            MixtralConfig(num_hidden_layers=4, num_local_experts=8,
                          num_experts_per_tok=2)
        )
        assert out["num_local_experts"] == 8
        assert out["num_experts_per_tok"] == 2

    def test_a_dense_model_carries_no_expert_fields(self):
        from transformers.models.llama.configuration_llama import LlamaConfig

        out = extract_architecture_config(LlamaConfig(num_hidden_layers=4))
        assert not [k for k in out if "expert" in k]


class TestItIsDrivenByTheLibraryNotAList:
    """Behavioural proof of genericity: a tower name nothing here knows."""

    def test_an_unknown_tower_name_still_resolves(self):
        speech = SimpleNamespace(
            num_hidden_layers=7, hidden_size=111, model_type="novel_speech"
        )
        brand_new = SimpleNamespace(
            num_hidden_layers=99, hidden_size=222, model_type="novel_text"
        )

        class NovelConfig:
            # exactly the contract transformers exposes
            sub_configs = {"speech_stack": None, "brand_new_llm": None}

            model_type = "novel_omni"
            speech_stack = speech
            brand_new_llm = brand_new

            def get_text_config(self, *a, **k):
                return brand_new

        out = extract_architecture_config(NovelConfig())

        assert out["num_hidden_layers"] == 99, (
            "a tower name this module has never heard of was not resolved, so "
            "the next modality needs a code change"
        )
        assert out["text_tower"] == "brand_new_llm"
        assert set(out["towers"]) == {"speech_stack", "brand_new_llm"}
        assert out["towers"]["speech_stack"]["num_hidden_layers"] == 7

    def test_a_config_without_the_accessor_is_treated_as_flat(self):
        """Older config objects predate get_text_config; they must not crash."""
        legacy = SimpleNamespace(
            model_type="ancient", num_hidden_layers=6, hidden_size=64
        )
        out = extract_architecture_config(legacy)
        assert out["num_hidden_layers"] == 6
        assert "towers" not in out
