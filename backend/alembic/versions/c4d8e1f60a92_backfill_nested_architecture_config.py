"""backfill architecture_config for models with a nested text_config

Models downloaded before this stored only TOP-LEVEL fields from config.json.
Multimodal and "unified" configs keep the decoder's dimensions nested, so
google/gemma-4-12B-it ("gemma4_unified") recorded three keys and no
num_hidden_layers -- and the Training page, which derives its layer picker from
that field, offered no layers at all for the model (reported 2026-08-25).

`extract_architecture_config` now reads the nested text sub-config, but that
only helps future downloads. This repairs rows already on disk by re-reading
each affected model's config.json.

Deliberately conservative:
  * only rows that are MISSING num_hidden_layers are touched, so a correct row
    can never be overwritten by this
  * the text tower is named explicitly -- gemma-4's config also carries
    audio_config and vision_config with their own geometry, and recording the
    audio tower's width as the model's would be a wrong number presented as a
    measurement
  * every model is wrapped: an unreadable or absent config.json is skipped with
    a warning. A metadata backfill must never be the reason a deploy fails to
    come up.

Revision ID: c4d8e1f60a92
Revises: e2a4c81b9d17
Create Date: 2026-08-25
"""

import json
import logging
from pathlib import Path

import sqlalchemy as sa
from alembic import op

revision = "c4d8e1f60a92"
down_revision = "e2a4c81b9d17"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

#: Same list, and same order of preference, as ml/model_loader.py.
FIELDS = [
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "num_key_value_heads",
    "hidden_act",
    "initializer_range",
    "layer_norm_eps",
    "use_cache",
    "tie_word_embeddings",
    "rope_theta",
]

TEXT_SUBCONFIG_NAMES = ("text_config", "llm_config", "language_config", "decoder")


def _find_config(file_path: str):
    """The model's config.json inside the HuggingFace cache layout."""
    root = Path(file_path)
    if not root.is_dir():
        return None
    snapshots = sorted(root.glob("**/snapshots/*/config.json"))
    if snapshots:
        return snapshots[-1]
    direct = root / "config.json"
    return direct if direct.is_file() else None


def _rebuild(raw: dict) -> dict:
    text = None
    for name in TEXT_SUBCONFIG_NAMES:
        sub = raw.get(name)
        # An integer, not just the key: gemma-4's audio_config carries
        # `num_hidden_layers: null`.
        if isinstance(sub, dict) and isinstance(sub.get("num_hidden_layers"), int):
            text = sub
            break

    out = {"model_type": raw.get("model_type")}
    for field in FIELDS:
        if field in raw:
            out[field] = raw[field]
        elif text is not None and field in text:
            out[field] = text[field]
    if text is not None:
        out["shape_source"] = "text_config"
    return out


def upgrade() -> None:
    conn = op.get_bind()

    rows = conn.execute(
        sa.text(
            "SELECT model_id, file_path, architecture_config FROM models "
            "WHERE file_path IS NOT NULL "
            "AND NOT jsonb_exists(COALESCE(architecture_config, '{}'::jsonb), "
            "                     'num_hidden_layers')"
        )
    ).fetchall()

    repaired = 0
    for model_id, file_path, existing in rows:
        try:
            config_path = _find_config(file_path)
            if config_path is None:
                logger.warning(
                    "architecture_config backfill: no config.json under %s for %s",
                    file_path, model_id,
                )
                continue

            rebuilt = _rebuild(json.loads(config_path.read_text()))
            if "num_hidden_layers" not in rebuilt:
                logger.warning(
                    "architecture_config backfill: %s has no layer count even "
                    "nested; leaving it alone", model_id,
                )
                continue

            merged = dict(existing or {})
            merged.update(rebuilt)

            conn.execute(
                sa.text(
                    "UPDATE models SET architecture_config = CAST(:cfg AS jsonb) "
                    "WHERE model_id = :mid"
                ),
                {"cfg": json.dumps(merged), "mid": model_id},
            )
            repaired += 1
            logger.info(
                "architecture_config backfill: %s -> %s layers",
                model_id, rebuilt["num_hidden_layers"],
            )
        except Exception as exc:                       # pragma: no cover
            logger.warning(
                "architecture_config backfill: skipped %s (%s)", model_id, exc
            )

    logger.info("architecture_config backfill: repaired %d model(s)", repaired)


def downgrade() -> None:
    """Not reversed: the previous values were incomplete readings, not data."""
    pass
