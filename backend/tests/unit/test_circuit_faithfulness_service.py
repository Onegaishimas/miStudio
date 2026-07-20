"""Faithfulness pins (017 Phase 4): member expansion, score assembly,
necessity-only mode, manifest payload records the metric identity."""

import pytest

from src.services.circuit_faithfulness_service import (
    CircuitFaithfulnessService,
    FaithfulnessConfigError,
    expand_members,
    scores_from_behaviors,
)


class TestConfig:
    def test_bad_mode(self):
        with pytest.raises(FaithfulnessConfigError):
            CircuitFaithfulnessService.create_config({"mode": "bogus"})

    def test_defaults(self):
        c = CircuitFaithfulnessService.create_config({})
        assert c["mode"] == "both"
        assert c["metric_id"] == "compare_output_shift/v1"


class TestMemberExpansion:
    def test_feature_and_cluster_members(self):
        members = [
            {"layer": 13, "feature": {"feature_idx": 5}},
            {"layer": 14, "member_kind": "cluster_ref", "cluster_profile_id": "clp"},
        ]
        by_layer = expand_members(members, lambda pid: [1, 2, 3])
        assert by_layer[13] == [5]
        assert by_layer[14] == [1, 2, 3]


class TestScores:
    def test_necessity_and_sufficiency(self):
        s = scores_from_behaviors(1.0, 0.2, 0.0, 0.9, mode="both")
        assert s["necessity"] == 0.8
        assert s["sufficiency"] == 0.9

    def test_necessity_only_marks_sufficiency_untested(self):
        s = scores_from_behaviors(1.0, 0.2, 0.0, None, mode="necessity")
        assert s["necessity"] == 0.8
        assert s["sufficiency"] is None
        assert "untested" in s["sufficiency_status"]


class TestManifest:
    def test_records_metric_and_proxy(self):
        cfg = CircuitFaithfulnessService.create_config({"ablate_all_n": 512})
        p = CircuitFaithfulnessService.build_manifest_payload(
            "crc_1", cfg, {"necessity": 0.8},
            {"b_clean": 1.0, "b_ablate_m": 0.2})
        assert p["metric_id"] == "compare_output_shift/v1"  # always recorded
        assert p["ablate_all_proxy"]["per_layer_top_n"] == 512
        assert p["circuit_id"] == "crc_1"
