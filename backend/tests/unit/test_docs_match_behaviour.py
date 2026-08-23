"""Task 13 — documentation that contradicts the code, where it matters.

MIS-E2E-149  the sentence that cost a real SAE was live in the SECOND manual.
             `manual/docs/core-workflow/sae-training.md` was corrected;
             `docs/miStudio_Manual.md` was not — and it is indexed in the
             repo's own knowledge graph, so an agent querying it is served the
             uncorrected text.
MIS-E2E-150  the manual said a bearer token is "always required" and its
             troubleshooting remedy was `MCP_ALLOW_ANONYMOUS=true`, which the
             guard honoured on HTTP — producing a LAN-reachable unauthenticated
             server exposing delete_circuit, GPU steering and label write-back.
             The code said "stdio only" in prose and did not enforce it.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


# ── MIS-E2E-149 · both manuals ─────────────────────────────────────────────

_MANUALS = [
    REPO / "docs" / "miStudio_Manual.md",
    REPO / "manual" / "docs" / "core-workflow" / "sae-training.md",
]


@pytest.mark.parametrize("path", _MANUALS, ids=lambda p: p.name)
def test_no_manual_claims_stop_saves_the_sae(path):
    """The identical sentence, in two places, one of them fixed.

    `train_969e90af` (granite-4.1-8b, FVU 0.065, zero dead neurons) was stopped
    at step 10,300 and its SAE forfeited, because the manual promised otherwise.
    """
    assert path.exists(), f"{path} moved — this guard would pass vacuously"
    text = path.read_text()
    assert "Gracefully end training (saves final checkpoint)" not in text, (
        f"{path.name} still promises that Stop saves the SAE. Only "
        f"Stop & Finalize writes community_format/, which is the only artifact "
        f"downstream reads."
    )


@pytest.mark.parametrize("path", _MANUALS, ids=lambda p: p.name)
def test_both_manuals_warn_that_stop_produces_no_importable_sae(path):
    """Absence of the wrong sentence is not the same as saying the right thing."""
    text = path.read_text().lower()
    assert "no importable sae" in text or "does not save an importable sae" in text, (
        f"{path.name} no longer warns that Stop leaves no importable SAE"
    )


def test_stop_and_stop_and_finalize_really_are_different_endpoints():
    """Negative control for the premise.

    If `stop` did finalize, the manuals' original sentence would have been
    right and these tests would be pinning a fiction.
    """
    import inspect

    from src.api.v1.endpoints import trainings

    src = inspect.getsource(trainings)
    assert "stop_and_finalize" in src
    assert "finalize_training_from_checkpoint_task" in src
    # The plain stop path must NOT dispatch the finalize task.
    stop_idx = src.index('"stop"')
    finalize_idx = src.index("stop_and_finalize")
    assert stop_idx < finalize_idx, "branch order changed; re-check this guard"


# ── MIS-E2E-150 · the flag is stdio-only ───────────────────────────────────

def _build(stdio: bool, **kw):
    from src.mcp_server.config import MCPSettings
    from src.mcp_server.server import build_server

    settings = MCPSettings(tool_categories="jlens", **kw)
    return build_server(settings, stdio=stdio)


def test_anonymous_over_http_is_refused(monkeypatch):
    """The hole: the flag alone satisfied the guard on the HTTP transport."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    with pytest.raises(SystemExit) as exc:
        _build(stdio=False, allow_anonymous=True)
    assert "stdio" in str(exc.value).lower()


def test_no_token_over_http_is_refused(monkeypatch):
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    with pytest.raises(SystemExit):
        _build(stdio=False)


def test_a_token_over_http_is_accepted(monkeypatch):
    """Negative control: the server must still start normally."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    _build(stdio=False, auth_token="s3cret")


def test_anonymous_over_stdio_is_accepted(monkeypatch):
    """The case the flag exists for."""
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")
    _build(stdio=True, allow_anonymous=True)


def test_the_manual_no_longer_offers_the_flag_as_the_remedy():
    """The remedy sent operators straight into the hole, on the page that told
    them a token was always required."""
    page = (REPO / "manual" / "docs" / "advanced" / "mcp-server.md").read_text()
    row = next(
        line for line in page.splitlines()
        if "MCP_AUTH_TOKEN is required" in line and line.startswith("|")
    )
    assert "will not help" in row or "stdio" in row, (
        f"the troubleshooting row still offers MCP_ALLOW_ANONYMOUS as the fix: {row}"
    )


# ── MIS-E2E-050 / -164 · the data model doc is complete ────────────────────

_DATA_MODEL = REPO / "manual" / "docs" / "reference" / "data-model.md"

#: Tables deliberately not described, with the reason. An exemption list makes
#: the omission a decision that leaves a record; silence made it an accident.
_UNDOCUMENTED_ON_PURPOSE = {
    "alembic_version": "Alembic's own migration bookkeeping, not part of the product's data model",
    "feature_activations_default": "the default partition of `feature_activations`, which IS documented",
}


def test_data_model_doc_covers_every_table():
    """The page claimed to be "verified against the ORM models" while omitting
    NINE tables — including `checkpoints`, which its own ER diagram draws.

    The claim is now enforced rather than asserted.
    """
    import re

    from src import models  # noqa: F401 — registers every table
    from src.core.database import Base

    assert _DATA_MODEL.exists(), "the data-model page moved — guard is vacuous"

    orm = set(Base.metadata.tables)
    assert len(orm) > 20, f"only {len(orm)} tables registered — imports incomplete"

    mentioned = set(re.findall(r"`([a-z_]+)`", _DATA_MODEL.read_text()))
    missing = orm - mentioned - set(_UNDOCUMENTED_ON_PURPOSE)
    assert not missing, (
        f"{len(missing)} tables are in the ORM and absent from the data-model "
        f"reference: {sorted(missing)}. Document them, or add them to "
        f"_UNDOCUMENTED_ON_PURPOSE with the reason."
    )


def test_the_exemptions_stay_confined_to_non_orm_tables():
    """A stale or growing exemption list hides a real gap by shrinking the
    required set.

    Neither exemption is in `Base.metadata` — that is precisely why they are
    exempt: `alembic_version` is created by Alembic itself, and
    `feature_activations_default` is a partition Postgres creates for a table
    that IS documented. My first version of this test asserted they were in the
    ORM, which can never hold for either. So the invariant is narrower and
    true: the list contains exactly these two, and a third has to be argued for
    here.
    """
    from src import models  # noqa: F401
    from src.core.database import Base

    orm = set(Base.metadata.tables)
    assert set(_UNDOCUMENTED_ON_PURPOSE) == {
        "alembic_version",
        "feature_activations_default",
    }, (
        "the exemption list changed; every entry must be a NON-ORM table with a "
        "recorded reason, or it is a documentation gap wearing an exemption"
    )
    overlap = set(_UNDOCUMENTED_ON_PURPOSE) & orm
    assert not overlap, (
        f"{sorted(overlap)} are mapped tables and must be documented, not exempted"
    )


def test_the_page_no_longer_claims_verification_it_did_not_do():
    text = _DATA_MODEL.read_text()
    assert "verified against the ORM models." not in text, (
        "the page asserts verification instead of being verified; the test "
        "above is what makes the claim true"
    )
