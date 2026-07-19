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

# 016 user-facing surfaces (report strings, MCP docstrings, service copy).
SURFACES = [
    BACKEND / "services" / "circuit_discovery_service.py",
    BACKEND / "services" / "circuit_attribution_service.py",
    BACKEND / "mcp_server" / "tools" / "circuits.py",
]

# A causal claim is only OK when negated or naming a rung-2+ label.
CAUSAL = re.compile(r"\bcausal(?:ly)?\b", re.IGNORECASE)
# Qualifiers that make an occurrence legitimate.
ALLOWED_CONTEXT = re.compile(
    r"(not\s+(?:a\s+)?causal|NOT\s+causal|never\s+describe.*causal|"
    r"causally\s+validated|causal\s+validation|causal\s+tier|"
    r"causal\s+proof|causal\s+words|causal\s+language|rung\s*2|017)",
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
