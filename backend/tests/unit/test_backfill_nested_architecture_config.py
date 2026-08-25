"""The backfill must repair the incomplete rows and touch nothing else.

A data migration that runs at deploy time gets one attempt against real rows,
so its rebuild logic is tested here rather than discovered in production.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "alembic" / "versions"
    / "c4d8e1f60a92_backfill_nested_architecture_config.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_backfill", _MIGRATION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


#: The real google/gemma-4-12B-it config.json, read off the GPU node.
GEMMA4 = {
    "model_type": "gemma4_unified",
    "initializer_range": 0.02,
    "tie_word_embeddings": True,
    "architectures": None,
    "audio_config": {"hidden_size": 640, "num_hidden_layers": 24},
    "vision_config": {"hidden_size": 1152, "num_hidden_layers": 27},
    "text_config": {
        "num_hidden_layers": 48,
        "hidden_size": 3840,
        "num_attention_heads": 16,
        "intermediate_size": 15360,
        "vocab_size": 262144,
        "num_key_value_heads": 8,
    },
}


class TestRebuild:
    def test_it_recovers_the_layer_count(self):
        assert mod._rebuild(GEMMA4)["num_hidden_layers"] == 48

    def test_it_reads_the_text_tower_not_the_audio_or_vision_one(self):
        out = mod._rebuild(GEMMA4)
        assert out["hidden_size"] == 3840
        assert out["num_hidden_layers"] not in (24, 27)

    def test_it_marks_the_shape_as_nested(self):
        assert mod._rebuild(GEMMA4)["shape_source"] == "text_config"

    def test_top_level_wins_over_nested(self):
        raw = {"model_type": "x", "num_hidden_layers": 12,
               "text_config": {"num_hidden_layers": 48}}
        assert mod._rebuild(raw)["num_hidden_layers"] == 12

    def test_a_flat_config_is_unchanged_in_shape(self):
        raw = {"model_type": "gemma2", "num_hidden_layers": 26, "hidden_size": 2304}
        out = mod._rebuild(raw)
        assert out["num_hidden_layers"] == 26
        assert "shape_source" not in out

    def test_it_agrees_with_the_loader_it_mirrors(self):
        """Two copies of the same field list will drift; pin them together."""
        from src.ml.model_loader import _TEXT_SUBCONFIG_NAMES

        assert tuple(mod.TEXT_SUBCONFIG_NAMES) == tuple(_TEXT_SUBCONFIG_NAMES)


class TestFindConfig:
    def test_it_finds_a_huggingface_snapshot_config(self, tmp_path):
        snap = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(GEMMA4))

        found = mod._find_config(str(tmp_path))
        assert found is not None and found.name == "config.json"

    def test_it_finds_a_flat_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert mod._find_config(str(tmp_path)) is not None

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert mod._find_config(str(tmp_path / "nope")) is None

    def test_a_directory_without_a_config_is_not_an_error(self, tmp_path):
        assert mod._find_config(str(tmp_path)) is None


class TestTheQueryOnlyTargetsBrokenRows:
    def test_it_filters_on_the_missing_layer_count(self):
        import inspect

        src = inspect.getsource(mod.upgrade)
        assert "jsonb_exists" in src and "num_hidden_layers" in src, (
            "the backfill no longer restricts itself to rows missing a layer "
            "count, so it can overwrite a correct architecture_config"
        )
