"""
Contract tests for POST /steering/cluster-allocation (Feature 013).

The endpoint is CPU-only glue: resolve the SAE, load the decoder via the shared
resolver (degrading to the approximate G=1 path on failure), delegate to the
pure math core, map ValueError → 400. These tests exercise the glue with the
service layer mocked — the math itself is covered in test_cluster_allocation.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from src.api.v1.endpoints.steering import compute_cluster_strength_allocation
from src.schemas.steering import ClusterAllocationMember, ClusterAllocationRequest
from fastapi import HTTPException


def _mk_sae(n_features=100, status="ready", local_path="saes/x"):
    sae = MagicMock()
    sae.status = status
    sae.local_path = local_path
    sae.n_features = n_features
    sae.layer = 12
    sae.d_model = 16
    sae.architecture = "jumprelu"
    return sae


def _req(idxs=(0, 1), layer=12, **kw):
    return ClusterAllocationRequest(
        sae_id="sae_x",
        members=[
            ClusterAllocationMember(feature_idx=i, layer=layer, similarity=0.8, activation_frequency=0.2)
            for i in idxs
        ],
        **kw,
    )


@pytest.mark.asyncio
async def test_404_when_sae_missing():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=AsyncMock(return_value=None)
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(), db=MagicMock())
        assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_400_when_index_out_of_bounds():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae(n_features=5)),
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(idxs=(0, 99)), db=MagicMock())
        assert e.value.status_code == 400
        assert "out of bounds" in str(e.value.detail)


@pytest.mark.asyncio
async def test_400_on_mixed_layers():
    req = ClusterAllocationRequest(
        sae_id="sae_x",
        members=[
            ClusterAllocationMember(feature_idx=0, layer=12),
            ClusterAllocationMember(feature_idx=1, layer=13),
        ],
    )
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae()),
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc:
        svc.return_value.load_sae = AsyncMock(side_effect=RuntimeError("skip decoder"))
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(req, db=MagicMock())
        assert e.value.status_code == 400
        assert "mixed-layer" in str(e.value.detail)


@pytest.mark.asyncio
async def test_decoder_failure_degrades_to_approximate():
    """Loader errors must not 500 — the allocation degrades to G=1/approximate."""
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae()),
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc:
        svc.return_value.load_sae = AsyncMock(side_effect=RuntimeError("boom"))
        resp = await compute_cluster_strength_allocation(_req(), db=MagicMock())
    assert resp.approximate is True
    assert resp.G == 1.0
    assert "approximate" in resp.flags
    assert resp.B == pytest.approx(resp.B_dir, abs=1e-6)


@pytest.mark.asyncio
async def test_happy_path_with_decoder_and_flags():
    sae = _mk_sae()
    loaded = MagicMock()
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae", new=AsyncMock(return_value=sae)
    ), patch(
        "src.api.v1.endpoints.steering.get_steering_service"
    ) as svc, patch(
        "src.api.v1.endpoints.steering.resolve_decoder_weight"
    ) as rdw, patch(
        "src.api.v1.endpoints.steering.settings"
    ) as st:
        st.resolve_data_path.return_value = MagicMock(exists=MagicMock(return_value=True))
        st.steering_cluster_constants_json = "{}"
        svc.return_value.load_sae = AsyncMock(return_value=loaded)
        dec = torch.zeros(16, 100)
        dec[0, 0] = 1.0
        dec[0, 1] = 1.0  # identical directions → G = 1
        rdw.return_value = dec
        resp = await compute_cluster_strength_allocation(
            _req(group_cohesion=0.3), db=MagicMock()
        )
    assert resp.approximate is False
    assert resp.G == pytest.approx(1.0, abs=1e-3)
    assert resp.B == pytest.approx(resp.B_dir, abs=1e-2)
    assert "low_cohesion" in resp.flags  # cohesion 0.3 < gate 0.5
    assert resp.formula_id == "freq-budget/sim-alloc@1"
    assert len(resp.strengths) == 2


@pytest.mark.asyncio
async def test_400_when_sae_not_ready():
    with patch(
        "src.api.v1.endpoints.steering.SAEManagerService.get_sae",
        new=AsyncMock(return_value=_mk_sae(status="downloading")),
    ):
        with pytest.raises(HTTPException) as e:
            await compute_cluster_strength_allocation(_req(), db=MagicMock())
        assert e.value.status_code == 400
