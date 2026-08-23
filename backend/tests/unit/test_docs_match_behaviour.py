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


# ── MIS-E2E-114 / -161 · the MCP contract and its counts ───────────────────

#: Tools registered CONDITIONALLY, so the AST ceiling exceeds what a default
#: server serves. Named, so the difference is a decision rather than a drift.
_CONDITIONALLY_REGISTERED = {
    "get_approval_status": "only registered when `steering_approval` is on",
}


def test_the_contract_lists_no_endpoint_that_is_really_a_dict_lookup():
    """MIS-E2E-114. The AST scraper matched `dict.get("kind")` as `GET kind`.

    The committed contract carried three such rows, and
    `test_mcp_contract_generated.py` pinned them as correct — so the contract
    defended whatever path was recorded rather than the real one.
    """
    contract = (REPO / "docs" / "mcp-contract.md").read_text()
    for bogus in ("GET kind", "GET manifests", "GET status"):
        assert f"`{bogus}`" not in contract and f"{bogus}<" not in contract, (
            f"the contract lists {bogus!r} as an endpoint; it is a dictionary "
            f"lookup the AST scraper mistook for an HTTP call"
        )


def test_every_contract_endpoint_looks_like_a_path():
    """The general rule behind the three specific rows."""
    import re

    contract = (REPO / "docs" / "mcp-contract.md").read_text()
    endpoints = re.findall(r"`(GET|POST|PUT|DELETE|PATCH) ([^`]+)`", contract)
    assert endpoints, "no endpoints found in the contract — the scan broke"
    bad = [f"{m} {p}" for m, p in endpoints if not p.startswith("/")]
    assert not bad, f"contract endpoints that are not paths: {bad}"


def test_the_server_instruction_count_is_derived_not_written():
    """MIS-E2E-161. Three places carried three different counts: the
    instructions said 92/13, the manual 97/13, the generated contract 116/14.
    Only the contract was derived."""
    import inspect

    from src.mcp_server import server

    src = inspect.getsource(server._server_instructions)
    assert "ast" in src.lower(), "the count is not derived from the registry"

    # Strip the docstring: it cites the stale numbers (92, 97, 116) to explain
    # the drift, and a bare substring check reads them as a regression.
    # Seventh occurrence of this trap in this remediation.
    doc = inspect.getdoc(server._server_instructions) or ""
    code = src
    for line in doc.splitlines():
        code = code.replace(line, "")
    assert "92" not in code and "97" not in code, "a hardcoded count is back"

    # And the template itself must carry a placeholder, not a literal.
    assert "{tool_count}" in server.SERVER_INSTRUCTIONS


def test_the_instruction_ceiling_exceeds_the_contract_by_exactly_the_conditional_tools(
    monkeypatch,
):
    """The two authorities must not drift.

    The AST ceiling counts every `@mcp.tool()`; the contract counts what a
    default server serves. The difference is exactly the conditionally-
    registered set — if it grows, one of them has changed and this says so.
    """
    monkeypatch.setenv("MILLM_API_URL", "http://millm.test")

    import ast
    import inspect as _inspect

    from src.mcp_server.contract import collect
    from src.mcp_server.tools import CATEGORY_MODULES, MILLM_CATEGORY_MODULES

    ast_names = set()
    for modules in {**CATEGORY_MODULES, **MILLM_CATEGORY_MODULES}.values():
        for module in modules:
            for node in ast.walk(ast.parse(_inspect.getsource(module))):
                if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                    continue
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    if isinstance(target, ast.Attribute) and target.attr == "tool":
                        ast_names.add(node.name)
                        break

    served = {row["name"] for rows in collect().values() for row in rows}
    assert ast_names, "no tools found by AST — the scan broke"
    assert served, "the contract collector returned nothing"

    difference = ast_names - served
    assert difference == set(_CONDITIONALLY_REGISTERED), (
        f"the AST ceiling and the served set differ by {sorted(difference)}, "
        f"expected exactly {sorted(_CONDITIONALLY_REGISTERED)}"
    )


# ── MIS-E2E-155 · CLAUDE.md's instruction references ───────────────────────

def test_every_instruct_reference_names_a_file_that_exists():
    """MIS-E2E-155. `001_generate-brd.md` was added at the front of the
    sequence and the reference list was never renumbered — so every entry
    named a real file performing a DIFFERENT action, and `008_housekeeping.md`
    did not exist at all. Following any of them by number ran the wrong step.
    """
    import re

    claude = (REPO / "CLAUDE.md").read_text()
    instruct_dir = REPO / "0xcc" / "instruct"
    assert instruct_dir.is_dir(), "0xcc/instruct moved — guard is vacuous"

    on_disk = {p.name for p in instruct_dir.glob("*.md")}
    assert on_disk, "no instruction files found"

    referenced = set(re.findall(r"0xcc/instruct/([0-9]{3}_[a-z-]+\.md)", claude))
    assert referenced, "no instruction references found in CLAUDE.md"

    missing = referenced - on_disk
    assert not missing, (
        f"CLAUDE.md references instruction files that do not exist: "
        f"{sorted(missing)}. On disk: {sorted(on_disk)}"
    )
