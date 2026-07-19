"""R2-T2: automated migration hygiene guard — the multi-head incident
(cd6c46abac48 merge) must never recur silently, and every new migration must
keep exactly one head."""

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_exactly_one_head():
    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Multiple alembic heads: {heads} — merge before shipping"
