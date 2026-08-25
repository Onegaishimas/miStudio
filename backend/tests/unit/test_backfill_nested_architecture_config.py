"""The backfill must repair incomplete rows, reuse the real extractor, and
touch nothing else.

A data migration gets one attempt against real rows at deploy time, so its
behaviour is pinned here rather than discovered in production.
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


class TestItUsesTheProductionExtractor:
    def test_rebuild_delegates_rather_than_reimplementing(self, tmp_path):
        """Two copies of the field list drift; there must be one."""
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
        from src.ml.model_loader import extract_architecture_config

        Gemma3Config().save_pretrained(tmp_path)

        rebuilt = mod._rebuild(tmp_path)
        from transformers import AutoConfig

        expected = extract_architecture_config(
            AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)
        )
        assert rebuilt == expected

    def test_it_recovers_a_composite_layer_count(self, tmp_path):
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config

        cfg = Gemma3Config()
        cfg.save_pretrained(tmp_path)

        out = mod._rebuild(tmp_path)
        assert out["num_hidden_layers"] == cfg.get_text_config().num_hidden_layers
        assert "vision_config" in out["towers"]


class TestFindingTheConfig:
    def test_it_finds_a_huggingface_snapshot(self, tmp_path):
        snap = tmp_path / "models--google--gemma-4-12B-it" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        assert mod._config_dir(str(tmp_path)) == snap

    def test_it_finds_a_flat_layout(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert mod._config_dir(str(tmp_path)) == tmp_path

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert mod._config_dir(str(tmp_path / "nope")) is None

    def test_a_directory_without_a_config_is_not_an_error(self, tmp_path):
        assert mod._config_dir(str(tmp_path)) is None


class TestItOnlyTargetsBrokenRows:
    def test_the_query_filters_on_the_missing_layer_count(self):
        import inspect

        src = inspect.getsource(mod.upgrade)
        assert "jsonb_exists" in src and "num_hidden_layers" in src, (
            "the backfill no longer restricts itself to rows missing a layer "
            "count, so it can overwrite a correct architecture_config"
        )

    def test_it_merges_rather_than_replaces(self):
        import inspect

        src = inspect.getsource(mod.upgrade)
        assert "merged.update(rebuilt)" in src
