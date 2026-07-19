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
             patch("src.api.v1.endpoints.circuits.CircuitService.set_promoted",
                   new=AsyncMock(return_value=_circuit(rung=0, promoted=True))):
            r = client.post("/api/v1/circuits/crc_test123/promote")
        assert r.status_code == 200 and r.json()["promoted"] is True

    def test_slices_parent_rung_is_single_source(self, client):
        """R1 fix #4: response rung and slice markers BOTH come from the
        recomputed definition (edge-less circuit => rung 0), never the stored
        column — no contradictory evidence claims in one payload."""
        c = _circuit(rung=1)  # stored column deliberately lies
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.post("/api/v1/circuits/crc_test123/export-slices")
        body = r.json()
        assert body["parent_rung"] == 0  # recomputed, not the stored 1
        assert "causal" not in body["parent_rung_language"]
        assert body["slices"][0]["kind"] == "mistudio.cluster-definition"
        assert "partial_rendering=true" in body["slices"][0]["provenance"]["source_note"]
        assert "parent_rung=0" in body["slices"][0]["provenance"]["source_note"]

    def test_missing_circuit_404(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=None)):
            r = client.get("/api/v1/circuits/crc_nope")
        assert r.status_code == 404


class TestR1Fixes:
    def test_unpromote_supported(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit(promoted=True))), \
             patch("src.api.v1.endpoints.circuits.CircuitService.set_promoted",
                   new=AsyncMock(return_value=_circuit(promoted=False))) as sp:
            r = client.post("/api/v1/circuits/crc_test123/promote",
                            json={"promoted": False})
        assert r.status_code == 200 and r.json()["promoted"] is False
        assert sp.await_args.args[2] is False

    def test_import_rejects_unknown_kind(self, client):
        r = client.post("/api/v1/circuits/import",
                        json={"kind": "mistudio.cluster-definition"})
        assert r.status_code == 422
        assert "Unknown kind" in r.json()["detail"]

    def test_list_returns_slim_summaries(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.list",
                   new=AsyncMock(return_value=[_circuit()])):
            r = client.get("/api/v1/circuits?limit=10")
        body = r.json()
        row = body["circuits"][0]
        assert "members" not in row and "edges" not in row  # slim rows
        assert row["member_count"] == 1 and row["layers"] == [13]
        assert body["limit"] == 10 and body["offset"] == 0

    def test_granularity_param_validated(self, client):
        r = client.get("/api/v1/circuits?granularity=features")
        assert r.status_code == 422  # Literal — typos never silently empty

    def test_export_filename_ascii_only(self, client):
        c = _circuit()
        c.name = "Précis ↔ circuit"
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=c)):
            r = client.get("/api/v1/circuits/crc_test123/export")
        cd = r.headers["content-disposition"]
        assert cd.encode("latin-1")  # header must be encodable
        assert "pr-cis" in cd or "pr" in cd

    def test_patch_null_structural_field_rejected_cleanly(self, client):
        with patch("src.api.v1.endpoints.circuits.CircuitService.get",
                   new=AsyncMock(return_value=_circuit())):
            r = client.patch("/api/v1/circuits/crc_test123", json={"members": None})
        assert r.status_code == 422
        assert "cannot be null" in r.json()["detail"]


class TestRealWiring:
    """R1 TE-1: exercise the REAL service+DB path (no CircuitService mocks)."""

    @pytest.fixture
    def wired_client(self, async_engine):
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from src.core.database import get_db

        maker = async_sessionmaker(async_engine, class_=AsyncSession,
                                   expire_on_commit=False)

        async def _override():
            async with maker() as session:
                yield session

        app.dependency_overrides[get_db] = _override
        yield TestClient(app, raise_server_exceptions=False)
        app.dependency_overrides.pop(get_db, None)

    def test_full_lifecycle_through_real_stack(self, wired_client):
        payload = {
            "name": "Wired circuit",
            "saes": [{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192},
                     {"mistudio_sae_id": "sae_l14", "layer": 14, "n_features": 8192}],
            "members": [
                {"layer": 13, "feature": {"feature_idx": 1, "strength": 0.5}},
                {"layer": 14, "feature": {"feature_idx": 2, "strength": 0.4}},
            ],
            "edges": [{"up": {"layer": 13, "feature_idx": 1},
                       "down": {"layer": 14, "feature_idx": 2}, "rung": 1}],
            "discovery": {"mode": "seeded", "granularity": "feature"},
            "model_id": "m_lfm25",
        }
        created = wired_client.post("/api/v1/circuits", json=payload)
        assert created.status_code == 201, created.text
        cid = created.json()["id"]
        assert created.json()["discovery"]["mode"] == "seeded"

        listed = wired_client.get("/api/v1/circuits")
        assert any(row["id"] == cid for row in listed.json()["circuits"])

        patched = wired_client.patch(f"/api/v1/circuits/{cid}",
                                     json={"narrative": "updated **markdown**"})
        assert patched.status_code == 200

        promoted = wired_client.post(f"/api/v1/circuits/{cid}/promote")
        assert promoted.json()["promoted"] is True

        slices = wired_client.post(f"/api/v1/circuits/{cid}/export-slices")
        assert slices.status_code == 200
        assert len(slices.json()["slices"]) == 2

        exported = wired_client.get(f"/api/v1/circuits/{cid}/export")
        assert exported.json()["discovery"]["mode"] == "seeded"  # lossless

        # Lossless round-trip (R2 B1): model ref, granularity, created_at.
        assert exported.json()["model"]["mistudio_model_id"] == "m_lfm25"
        reimported = wired_client.post("/api/v1/circuits/import",
                                       json=exported.json())
        assert reimported.status_code == 201, reimported.text
        assert reimported.json()["edges"][0]["rung"] == 1
        assert reimported.json()["model_id"] == "m_lfm25"
        assert reimported.json()["granularity"] == "feature"
        assert reimported.json()["created_at"] == created.json()["created_at"]

        deleted = wired_client.delete(f"/api/v1/circuits/{cid}")
        assert deleted.json()["deleted"] == cid
