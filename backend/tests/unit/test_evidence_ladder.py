"""
Pins for the evidence ladder (018 Task 1.1, IDL-35): single-source enum,
rung transition rules, TS mirror sync, and the no-parallel-enums grep guard.
"""

import re
from pathlib import Path

from src.schemas.evidence_ladder import (
    RUNG_LANGUAGE,
    RUNG_NEXT_STEP,
    EdgeEvidence,
    EvidenceRung,
    circuit_rung,
    rung_language,
)

REPO = Path(__file__).resolve().parents[3]


class TestRungModel:
    def test_rung_is_highest_passed(self):
        e = EdgeEvidence()
        e.record_pass(EvidenceRung.ATTRIBUTION_SUPPORTED)
        e.record_pass(EvidenceRung.MINED)  # lower pass never demotes
        assert e.rung == EvidenceRung.ATTRIBUTION_SUPPORTED

    def test_failure_records_without_demoting(self):
        e = EdgeEvidence(rung=EvidenceRung.ATTRIBUTION_SUPPORTED)
        e.record_failure(EvidenceRung.CAUSALLY_VALIDATED)
        assert e.rung == EvidenceRung.ATTRIBUTION_SUPPORTED
        assert e.tested_and_failed == [EvidenceRung.CAUSALLY_VALIDATED]

    def test_later_pass_supersedes_failure_at_same_rung(self):
        e = EdgeEvidence()
        e.record_failure(EvidenceRung.CAUSALLY_VALIDATED)
        e.record_pass(EvidenceRung.CAUSALLY_VALIDATED)
        assert e.rung == EvidenceRung.CAUSALLY_VALIDATED
        assert e.tested_and_failed == []

    def test_circuit_rung_is_min_over_edges(self):
        assert circuit_rung(
            [EvidenceRung.FAITHFULNESS_TESTED, EvidenceRung.MINED]
        ) == EvidenceRung.MINED

    def test_edgeless_circuit_is_mined_not_a_crash(self):
        assert circuit_rung([]) == EvidenceRung.MINED

    def test_language_map_total_and_causal_only_at_rung2plus(self):
        assert set(RUNG_LANGUAGE) == set(EvidenceRung) == set(RUNG_NEXT_STEP)
        for rung, phrase in RUNG_LANGUAGE.items():
            if rung < EvidenceRung.CAUSALLY_VALIDATED:
                assert "causal" not in phrase.lower()
        assert rung_language(2) == RUNG_LANGUAGE[EvidenceRung.CAUSALLY_VALIDATED]


class TestSingleSource:
    def test_ts_mirror_in_sync(self):
        ts = (REPO / "frontend" / "src" / "types" / "evidenceLadder.ts").read_text()
        for rung in EvidenceRung:
            assert re.search(rf"{rung.name}\s*=\s*{rung.value}\b", ts), (
                f"TS mirror missing/mismatched {rung.name}={rung.value}"
            )

    def test_no_parallel_rung_enums(self):
        """Grep-guard (IDL-35): the ladder must never be redefined locally."""
        offenders = []
        for py in (REPO / "backend" / "src").rglob("*.py"):
            if py.name == "evidence_ladder.py":
                continue
            text = py.read_text()
            if re.search(r"class\s+\w*(EvidenceRung|Rung)\w*\s*\((IntEnum|Enum|str,\s*Enum)", text):
                offenders.append(str(py))
        assert offenders == [], f"Parallel rung enums found: {offenders}"
