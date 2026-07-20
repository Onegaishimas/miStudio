"""API pins for circuit validation + manifests (017): 202/409/422, cancel,
manifest retrieval, reproduce gating."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.circuit_runs import CircuitDiscoveryRun
from src.models.validation_manifest import ValidationManifest


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def _run(**kw):
    from datetime import datetime
    return CircuitDiscoveryRun(
        id=kw.pop("id", "dsc1"), capture_run_id="cap1",
        status=kw.pop("status", "completed"), params={},
        candidates=kw.pop("candidates", [{"up": {}, "down": {}}]),
        created_at=datetime(2026, 7, 20), updated_at=datetime(2026, 7, 20), **kw)


class TestValidateAPI:
    def test_needs_completed_run_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(status="running"))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_no_candidates_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(candidates=[]))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_already_in_flight_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(validation_status="running"))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate", json={})
        assert r.status_code == 409

    def test_bad_ordering_422(self, client):
        r = client.post("/api/v1/circuit-discovery/dsc1/validate",
                        json={"ordering": "bogus"})
        assert r.status_code == 422

    def test_cancel_no_validation_409(self, client):
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=_run(validation_status=None))):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate/cancel")
        assert r.status_code == 409

    def test_cancel_running_200(self, client):
        run = _run(validation_status="running", validation_task_id="vt1")
        with patch("src.api.v1.endpoints.circuit_validation._run_or_404",
                   new=AsyncMock(return_value=run)), \
             patch("src.core.celery_app.revoke_task"):
            r = client.post("/api/v1/circuit-discovery/dsc1/validate/cancel")
        assert r.status_code == 200
        assert r.json()["validation_status"] == "cancelled"


class TestManifestAPI:
    def _m(self, **kw):
        from datetime import datetime
        return ValidationManifest(
            id=kw.pop("id", "vman_1"), kind=kw.pop("kind", "edge_batch"),
            discovery_run_id="dsc1", payload={"config": {"ordering": "coact"}},
            created_at=datetime(2026, 7, 20), **kw)

    def test_get_404(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=None)):
            r = client.get("/api/v1/validation-manifests/vman_x")
        assert r.status_code == 404

    def test_get_200(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=self._m())):
            r = client.get("/api/v1/validation-manifests/vman_1")
        assert r.status_code == 200 and r.json()["kind"] == "edge_batch"

    def test_reproduce_non_edge_batch_409(self, client):
        with patch("src.services.manifest_service.ManifestService.get",
                   new=AsyncMock(return_value=self._m(kind="faithfulness"))):
            r = client.post("/api/v1/validation-manifests/vman_1/reproduce")
        assert r.status_code == 409
