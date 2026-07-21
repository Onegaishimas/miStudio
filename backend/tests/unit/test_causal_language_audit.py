"""Copy-audit: Feature 016 surfaces produce rungs 0–1 ONLY, so no
user-facing 016 string may use unqualified causal language (IDL-35 /
evidence-ladder discipline). Extends the 018 grep-guard to 016.

Allowed occurrences: the word inside a NEGATION/qualifier ("NOT causal",
"not a causal", "causally validated" is a rung-2 LABEL, disclosed as such),
or in meta-text that names the discipline itself. This test pins the
discovery/attribution reports + MCP docstrings + service copy against a
regression that would let a mined association be described as a mechanism.
"""

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "src"

# 016 + 017 user-facing surfaces (report strings, MCP docstrings, service
# copy). 017's validation IS the rung-2 tier, so 'causal' is EARNED there —
# but only inside validation/faithfulness contexts, which ALLOWED_CONTEXT
# covers via 'causally validated'/'rung 2'/'017'.
SURFACES = [
    BACKEND / "services" / "circuit_discovery_service.py",
    BACKEND / "services" / "circuit_attribution_service.py",
    BACKEND / "services" / "circuit_intervention_service.py",
    BACKEND / "services" / "circuit_faithfulness_service.py",
    BACKEND / "mcp_server" / "tools" / "circuits.py",
]

# F20 task 3.4: EVERY `millm_*` MCP tool module, discovered rather than listed.
#
# `SURFACES` above is hand-maintained, and a hand-maintained list is only as
# good as the list — a new module simply is not audited, silently. The MCP
# tools are the surface an AGENT reads and relays verbatim, so they are
# globbed: adding a module opts it in automatically.
MILLM_TOOL_MODULES = sorted(
    (BACKEND / "mcp_server" / "tools").glob("millm_*.py")
)

# A causal claim is only OK when negated or naming a rung-2+ label.
CAUSAL = re.compile(r"\bcausal(?:ly)?\b", re.IGNORECASE)
# Qualifiers that make an occurrence legitimate.
ALLOWED_CONTEXT = re.compile(
    r"(not\s+(?:a\s+)?causal|NOT\s+causal|never\s+describe.*causal|"
    r"causally\s+validate|causal\s+validation|causal\s+tier|"
    r"causal\s+proof|causal\s+words|causal\s+language|causal\s+measurement|"
    # F20: naming the MECHANISM that raises a rung is not claiming one has
    # been raised. "raising a rung requires a causal intervention" is the
    # OPPOSITE of an overclaim — it says this surface cannot do it.
    r"causal\s+intervention|"
    r"causal\s+evidence|causal\s+effect|real\s+causal|rung\s*2|017)",
    re.IGNORECASE)


def _offending_lines(path: Path):
    bad = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if CAUSAL.search(line) and not ALLOWED_CONTEXT.search(line):
            bad.append((i, line.strip()))
    return bad


@pytest.mark.parametrize("path", SURFACES, ids=lambda p: p.name)
def test_no_unqualified_causal_language(path):
    offending = _offending_lines(path)
    assert not offending, (
        f"{path.name} has unqualified causal language on a rung-0/1 surface "
        f"(IDL-35): {offending}")


def test_discovery_report_lag0_disclosure_present():
    src = (BACKEND / "services" / "circuit_discovery_service.py").read_text()
    assert "lag0_disclosure" in src
    assert "lag-0" in src.lower()


# ── F20: the MCP tool surface ──────────────────────────────────────────────


def _code_only(text: str) -> str:
    """Strip comment lines: prose explaining the prohibition is not copy."""
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("#")
    )


@pytest.mark.parametrize(
    "path", MILLM_TOOL_MODULES, ids=lambda p: p.name
)
def test_no_unqualified_causal_language_in_millm_tools(path):
    """F20 task 3.4. A mutation planting "causally validated" in a tool
    description SURVIVED the entire reachability suite until this existed."""
    bad = []
    for i, line in enumerate(_code_only(path.read_text()).splitlines(), 1):
        if CAUSAL.search(line) and not ALLOWED_CONTEXT.search(line):
            bad.append((i, line.strip()))
    assert not bad, (
        f"{path.name} makes an unqualified causal claim on a surface an AGENT "
        f"reads and relays: {bad}"
    )


def test_the_millm_module_list_is_not_empty():
    """An audit over an empty set passes forever. The glob is the point — but
    a glob that matches nothing is worse than a list."""
    names = {p.name for p in MILLM_TOOL_MODULES}
    assert "millm_circuits.py" in names, (
        "the circuit tools are not being audited"
    )
    assert len(names) >= 4


def test_registered_DESCRIPTIONS_are_audited_too():
    """F20 task 3.5. A source scan cannot see a description composed at
    runtime, inherited, or rewritten by a decorator — and that is exactly what
    `list_tools()` hands an agent."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from mcp.server.fastmcp import FastMCP

    from src.mcp_server.tools import MILLM_CATEGORY_MODULES

    gate = MagicMock()
    gate.check = AsyncMock(return_value=(True, None))
    mcp = FastMCP("audit")
    for modules in MILLM_CATEGORY_MODULES.values():
        for module in modules:
            module.register(mcp, MagicMock(), gate)

    tools = asyncio.run(mcp.list_tools())
    assert any(t.name == "millm_circuit_status" for t in tools), (
        "the scan is not seeing the circuit tools, so it would pass however "
        "they were worded"
    )

    bad = []
    for tool in tools:
        for line in (tool.description or "").splitlines():
            if CAUSAL.search(line) and not ALLOWED_CONTEXT.search(line):
                bad.append(f"{tool.name}: {line.strip()}")
    assert not bad, (
        "a REGISTERED tool description asserts causal evidence:\n  "
        + "\n  ".join(bad)
    )
