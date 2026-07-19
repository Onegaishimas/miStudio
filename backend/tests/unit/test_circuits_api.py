"""API pins for /circuits (018 Task 5.1): rung_language on every response,
422 surfaces for contract violations, promote badge-not-gate, slice export."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.circuit import Circuit


def _circuit(**kw):
    from datetime import datetime
    c = Circuit(
        id="crc_test123", name="Test", granularity="feature",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192}],
        members=[{"layer": 13, "member_kind": "feature_ref",
                  "feature": {"feature_idx": 1, "strength": 0.5}}],
        edges=[], budget=None, faithfulness=None, rung=kw.pop("rung", 0),
        promoted=kw.pop("promoted", False),
        created_at=datetime(2026, 7, 19), updated_at=datetime(2026, 7, 19), **kw,
    )
    return c


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


class TestRungLanguage:
    def test_every_response_carries_server_rendered_language(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(rung=1))):
            r = client.get("/api/v1/circuits/crc_test123")
        body = r.json()
        assert body["rung"] == 1
        assert body["rung_language"] == "suggested (attribution-supported)"
        assert "causal" not in body["rung_language"]
        assert body["rung_next_step"]

    def test_rung2_language_is_causal(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(rung=2))):
            r = client.get("/api/v1/circuits/crc_test123")
        assert r.json()["rung_language"] == "causally validated (edge)"


class TestContractSurfacing:
    def test_create_violation_is_422(self, client):
        bad = {
            "name": "x",
            "saes": [{"mistudio_sae_id": "s", "layer": 13}],
            "members": [{"layer": 13, "feature": {"feature_idx": i, "strength": 0.1}}
                        for i in range(21)],
        }
        r = client.post("/api/v1/circuits", json=bad)
        assert r.status_code == 422
        assert "per-layer member cap" in r.json()["detail"]

    def test_promote_never_gated_on_rung(self, client):
        c = _circuit(rung=0)
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)), \
             patch("src.api.v1.endpoints.circuits.CircuitService.promote",
                   new=AsyncMock(return_value=_circuit(rung=0, promoted=True))):
            r = client.post("/api/v1/circuits/crc_test123/promote")
        assert r.status_code == 200 and r.json()["promoted"] is True

    def test_slices_carry_parent_rung(self, client):
        c = _circuit(rung=1)
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.post("/api/v1/circuits/crc_test123/export-slices")
        body = r.json()
        assert body["parent_rung"] == 1
        assert "causal" not in body["parent_rung_language"]
        assert body["slices"][0]["kind"] == "mistudio.cluster-definition"
        assert "partial_rendering=true" in body["slices"][0]["provenance"]["source_note"]

    def test_missing_circuit_404(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=None)):
            r = client.get("/api/v1/circuits/crc_nope")
        assert r.status_code == 404
