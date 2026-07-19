"""Pins for CircuitService (018 Task 3.2): contract-backed validation, CRUD,
promotion (badge-not-gate), rung recomputation, slices export."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.services.circuit_service import CircuitService, CircuitValidationError


@pytest.fixture
async def db(async_engine):
    maker = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


def _payload(**overrides):
    base = dict(
        name="Support voice circuit",
        saes=[{"mistudio_sae_id": "sae_l13", "layer": 13, "n_features": 8192},
              {"mistudio_sae_id": "sae_l14", "layer": 14, "n_features": 8192}],
        members=[
            {"layer": 13, "feature": {"feature_idx": 712, "strength": 0.5, "label": "condolences"}},
            {"layer": 14, "feature": {"feature_idx": 231, "strength": 0.4, "label": "reassurance"}},
        ],
        edges=[{
            "up": {"layer": 13, "feature_idx": 712},
            "down": {"layer": 14, "feature_idx": 231},
            "rung": 1,
            "coactivation": {"pmi": 2.0, "support": 30},
        }],
        budget={"layers": {"13": {"B": 1.0}, "14": {"B": 2.4}}, "intensity": 1.0},
    )
    base.update(overrides)
    return base


class TestCRUD:
    @pytest.mark.asyncio
    async def test_create_computes_rung_and_persists(self, db):
        c = await CircuitService.create(db, _payload())
        assert c.id.startswith("crc_")
        assert c.rung == 1  # min over edges
        assert c.promoted is False

    @pytest.mark.asyncio
    async def test_create_rejects_contract_violations(self, db):
        bad = _payload(members=[{"layer": 13, "feature": {"feature_idx": i, "strength": 0.1}}
                                for i in range(21)], edges=[])
        with pytest.raises(CircuitValidationError, match="per-layer member cap"):
            await CircuitService.create(db, bad)

    @pytest.mark.asyncio
    async def test_promote_is_badge_not_gate(self, db):
        c = await CircuitService.create(db, _payload())
        assert c.rung < 2  # NOT causally validated…
        c = await CircuitService.promote(db, c)  # …and promotion still succeeds
        assert c.promoted is True

    @pytest.mark.asyncio
    async def test_update_revalidates_and_recomputes_rung(self, db):
        c = await CircuitService.create(db, _payload())
        edges = list(c.edges)
        edges[0] = {**edges[0], "rung": 2, "effect_size": 0.8}
        c = await CircuitService.update(db, c, {"edges": edges})
        assert c.rung == 2
        with pytest.raises(CircuitValidationError):
            await CircuitService.update(db, c, {"edges": [
                {"up": {"layer": 14, "feature_idx": 231},
                 "down": {"layer": 13, "feature_idx": 712}}]})

    @pytest.mark.asyncio
    async def test_recompute_rung_from_stored_edges(self, db):
        c = await CircuitService.create(db, _payload())
        c.edges = [{**c.edges[0], "rung": 3}]
        c = await CircuitService.recompute_rung(db, c)
        assert c.rung == 3

    @pytest.mark.asyncio
    async def test_list_filters(self, db):
        a = await CircuitService.create(db, _payload(name="A"))
        await CircuitService.promote(db, a)
        await CircuitService.create(db, _payload(name="B"))
        promoted = await CircuitService.list(db, promoted=True)
        assert [x.name for x in promoted] == ["A"]
        rung1 = await CircuitService.list(db, min_rung=1)
        assert {x.name for x in rung1} == {"A", "B"}


class TestExport:
    @pytest.mark.asyncio
    async def test_to_definition_and_slices(self, db):
        c = await CircuitService.create(db, _payload())
        defn = CircuitService.to_definition(c)
        assert defn.kind == "mistudio.circuit-definition"
        assert defn.provenance.exported_at is not None
        slices = CircuitService.to_slices(c)
        assert [s.sae.layer for s in slices] == [13, 14]
        assert all("partial_rendering=true" in s.provenance.source_note for s in slices)
