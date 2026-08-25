"""
A multimodal config keeps the decoder's shape nested; read it from there.

Reported 2026-08-25: selecting gemma-4-12B-it on the Training page offered no
layers to pick. The Training page derives its layer picker from
`architecture_config.num_hidden_layers`, and that model's stored config held
exactly three keys -- model_type, initializer_range, tie_word_embeddings --
because `extract_architecture_config` read only top-level attributes and
google/gemma-4-12B-it ("gemma4_unified") nests everything under `text_config`.

Five other models were unaffected, which is why this looked like a regression
and was not one.

The trap this pins: that same config carries `audio_config` and
`vision_config`, each with its own hidden_size. Scanning for the first nested
config with a shape records the AUDIO tower as the model's geometry -- a wrong
number presented as a measurement, which is worse than the missing one.
"""

from types import SimpleNamespace

import pytest

from src.ml.model_loader import extract_architecture_config


def _gemma4_like():
    """The real shape of google/gemma-4-12B-it's config.json, read off disk."""
    return SimpleNamespace(
        model_type="gemma4_unified",
        initializer_range=0.02,
        tie_word_embeddings=True,
        # decoys, both carrying their own geometry
        audio_config=SimpleNamespace(hidden_size=640, num_hidden_layers=24),
        vision_config=SimpleNamespace(hidden_size=1152, num_hidden_layers=27),
        # the language model
        text_config=SimpleNamespace(
            num_hidden_layers=48,
            hidden_size=3840,
            num_attention_heads=16,
            vocab_size=262144,
        ),
    )


class TestNestedConfigs:
    def test_it_finds_the_layer_count_in_text_config(self):
        cfg = extract_architecture_config(_gemma4_like())
        assert cfg["num_hidden_layers"] == 48, (
            "no layer count means the Training page renders no layer picker"
        )

    def test_it_takes_the_text_tower_not_the_audio_one(self):
        cfg = extract_architecture_config(_gemma4_like())
        assert cfg["hidden_size"] == 3840, (
            f"took {cfg['hidden_size']} -- that is a different tower's width"
        )
        assert cfg["num_hidden_layers"] != 24, "read the audio tower"
        assert cfg["num_hidden_layers"] != 27, "read the vision tower"

    def test_it_records_that_the_shape_is_nested(self):
        cfg = extract_architecture_config(_gemma4_like())
        assert cfg.get("shape_source") == "text_config"

    def test_top_level_still_wins(self):
        """A flat config must be unaffected, and top level beats nested."""
        flat = SimpleNamespace(
            model_type="gemma2",
            num_hidden_layers=26,
            hidden_size=2304,
            vocab_size=256000,
        )
        cfg = extract_architecture_config(flat)
        assert cfg["num_hidden_layers"] == 26
        assert "shape_source" not in cfg

    def test_a_conflicting_top_level_value_is_preferred(self):
        both = SimpleNamespace(
            model_type="weird",
            num_hidden_layers=12,
            text_config=SimpleNamespace(num_hidden_layers=48),
        )
        assert extract_architecture_config(both)["num_hidden_layers"] == 12

    @pytest.mark.parametrize("attr", ["llm_config", "language_config", "decoder"])
    def test_other_nesting_conventions_are_handled(self, attr):
        cfg_obj = SimpleNamespace(model_type="x")
        setattr(cfg_obj, attr, SimpleNamespace(num_hidden_layers=32, hidden_size=4096))
        cfg = extract_architecture_config(cfg_obj)
        assert cfg["num_hidden_layers"] == 32

    def test_a_null_layer_count_does_not_qualify(self):
        """gemma-4's audio_config really does carry num_hidden_layers: null."""
        cfg_obj = SimpleNamespace(
            model_type="x",
            text_config=SimpleNamespace(num_hidden_layers=None, hidden_size=640),
        )
        cfg = extract_architecture_config(cfg_obj)
        assert cfg.get("hidden_size") != 640
        assert "shape_source" not in cfg

    def test_an_object_that_answers_to_anything_does_not_qualify(self):
        """A Mock-like config would otherwise supply every field from nowhere."""
        from unittest.mock import Mock

        cfg_obj = Mock()
        cfg_obj.model_type = "gpt2"
        cfg_obj.num_hidden_layers = 12
        cfg = extract_architecture_config(cfg_obj)
        assert cfg["num_hidden_layers"] == 12
        assert "shape_source" not in cfg

    def test_a_nested_config_without_layers_is_not_used(self):
        """An audio-only sibling must not qualify as the text tower."""
        cfg_obj = SimpleNamespace(
            model_type="x",
            text_config=SimpleNamespace(hidden_size=99),   # no num_hidden_layers
        )
        cfg = extract_architecture_config(cfg_obj)
        assert "num_hidden_layers" not in cfg
        assert cfg.get("hidden_size") != 99
