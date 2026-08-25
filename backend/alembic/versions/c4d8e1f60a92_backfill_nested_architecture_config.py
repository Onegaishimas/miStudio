"""backfill architecture_config for composite (multi-tower) models

Models downloaded before this stored only TOP-LEVEL fields from config.json.
Composite configs -- vision-language, audio and "omni" models -- keep the
decoder's dimensions in a sub-config, so google/gemma-4-12B-it recorded three
keys and no num_hidden_layers. The Training page derives its layer picker from
that field, so the model could be selected and offered no layers at all
(reported 2026-08-25).

`extract_architecture_config` now asks transformers which tower is the language
model and records every declared tower, but that only helps future downloads.
This repairs rows already on disk.

It re-reads each affected model's config through AutoConfig and calls the SAME
extractor rather than reimplementing it. An earlier draft duplicated the field
list here; two copies of a field list drift, and the test written to pin them
together is a worse answer than not having two.

Deliberately conservative:
  * only rows MISSING num_hidden_layers are touched, so a correct row can never
    be overwritten by this
  * existing keys are preserved and updated, not replaced wholesale
  * every model is wrapped -- an unreadable config, an architecture this
    transformers version does not know, or a model whose files have been
    deleted is skipped with a warning. A metadata backfill must never be the
    reason a deploy fails to come up.

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


def _config_dir(file_path: str):
    """The directory holding config.json, in either cache layout."""
    root = Path(file_path)
    if not root.is_dir():
        return None
    snapshots = sorted(root.glob("**/snapshots/*/config.json"))
    if snapshots:
        return snapshots[-1].parent
    return root if (root / "config.json").is_file() else None


def _rebuild(config_dir: Path) -> dict:
    """Describe the model with the production extractor."""
    from transformers import AutoConfig

    from src.ml.model_loader import extract_architecture_config

    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    return extract_architecture_config(config)


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
            config_dir = _config_dir(file_path)
            if config_dir is None:
                logger.warning(
                    "architecture_config backfill: no config.json under %s for %s",
                    file_path, model_id,
                )
                continue

            rebuilt = _rebuild(config_dir)
            if "num_hidden_layers" not in rebuilt:
                logger.warning(
                    "architecture_config backfill: %s exposes no layer count "
                    "on any tower; leaving it alone", model_id,
                )
                continue

            merged = dict(existing or {})
            merged.update(rebuilt)

            conn.execute(
                sa.text(
                    "UPDATE models SET architecture_config = CAST(:cfg AS jsonb) "
                    "WHERE model_id = :mid"
                ),
                {"cfg": json.dumps(merged, default=str), "mid": model_id},
            )
            repaired += 1
            logger.info(
                "architecture_config backfill: %s -> %s layers (towers: %s)",
                model_id,
                rebuilt["num_hidden_layers"],
                sorted(rebuilt.get("towers") or {}) or "flat",
            )
        except Exception as exc:                       # pragma: no cover
            logger.warning(
                "architecture_config backfill: skipped %s (%s)", model_id, exc
            )

    logger.info("architecture_config backfill: repaired %d model(s)", repaired)


def downgrade() -> None:
    """Not reversed: the previous values were incomplete readings, not data."""
    pass
