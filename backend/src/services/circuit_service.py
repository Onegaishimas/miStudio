"""
Circuit service (Feature 018, IDL-33): CRUD, assembly, promotion, rung
recomputation, and contract projection for circuits.

Validation strategy: every write round-trips through CircuitDefinitionV1 so
the CONTRACT validators (per-layer caps, edge-endpoint integrity, layer
ascension, SAE-ref completeness) are the single source of structural truth —
the service never re-implements them.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.circuit import Circuit
from ..schemas.circuit_definition import (
    CircuitDefinitionV1,
    to_layer_slice,
)
from ..schemas.cluster_profile import ClusterDefinitionV1, DefinitionProvenance
from ..schemas.evidence_ladder import EvidenceRung, circuit_rung

logger = logging.getLogger(__name__)

# Single source: reuse the profile service constant (review R1 — version drift).
from .cluster_profile_service import APP_VERSION  # noqa: E402


class CircuitValidationError(ValueError):
    """Structural validation failure — surfaces as a 422."""


class CircuitService:
    # ── validation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate(name: str, narrative: Optional[str], saes: list, members: list,
                  edges: list, budget: Optional[dict],
                  faithfulness: Optional[dict] = None,
                  discovery: Optional[dict] = None) -> CircuitDefinitionV1:
        """Round-trip through the contract model — its validators are the law."""
        try:
            return CircuitDefinitionV1(
                name=name,
                narrative=narrative,
                saes=saes,
                members=members,
                edges=edges,
                budget=budget,
                faithfulness=faithfulness,
                discovery=discovery,
            )
        except Exception as e:  # pydantic ValidationError → domain error
            raise CircuitValidationError(str(e)) from e

    # ── CRUD ────────────────────────────────────────────────────────────

    @staticmethod
    async def create(db: AsyncSession, data: Dict[str, Any]) -> Circuit:
        defn = CircuitService._validate(
            data["name"], data.get("narrative"), data.get("saes", []),
            data.get("members", []), data.get("edges", []), data.get("budget"),
            data.get("faithfulness"), data.get("discovery"),
        )
        circuit = Circuit(
            name=defn.name,
            narrative=defn.narrative,
            granularity=data.get("granularity", "feature"),
            saes=[s.model_dump(mode="json") for s in defn.saes],
            members=[m.model_dump(mode="json") for m in defn.members],
            edges=[e.model_dump(mode="json") for e in defn.edges],
            budget=defn.budget.model_dump(mode="json") if defn.budget else None,
            faithfulness=(defn.faithfulness.model_dump(mode="json")
                          if defn.faithfulness else None),
            discovery=(defn.discovery.model_dump(mode="json")
                       if defn.discovery else None),
            rung=int(defn.displayed_rung()),
            discovery_run_id=data.get("discovery_run_id"),
            model_id=data.get("model_id"),
        )
        db.add(circuit)
        await db.commit()
        await db.refresh(circuit)
        return circuit

    @staticmethod
    async def get(db: AsyncSession, circuit_id: str) -> Optional[Circuit]:
        return (
            await db.execute(select(Circuit).where(Circuit.id == circuit_id))
        ).scalar_one_or_none()

    @staticmethod
    async def list(db: AsyncSession, *, promoted: Optional[bool] = None,
                   min_rung: Optional[int] = None,
                   granularity: Optional[str] = None) -> List[Circuit]:
        q = select(Circuit).order_by(Circuit.updated_at.desc())
        if promoted is not None:
            q = q.where(Circuit.promoted == promoted)
        if min_rung is not None:
            q = q.where(Circuit.rung >= min_rung)
        if granularity:
            q = q.where(Circuit.granularity == granularity)
        return list((await db.execute(q)).scalars().all())

    @staticmethod
    async def update(db: AsyncSession, circuit: Circuit,
                     data: Dict[str, Any]) -> Circuit:
        merged = {
            "name": data.get("name", circuit.name),
            "narrative": data.get("narrative", circuit.narrative),
            "saes": data.get("saes", circuit.saes),
            "members": data.get("members", circuit.members),
            "edges": data.get("edges", circuit.edges),
            "budget": data.get("budget", circuit.budget),
            "faithfulness": data.get("faithfulness", circuit.faithfulness),
        }
        defn = CircuitService._validate(
            merged["name"], merged["narrative"], merged["saes"],
            merged["members"], merged["edges"], merged["budget"],
            merged["faithfulness"],
        )
        circuit.name = defn.name
        circuit.narrative = defn.narrative
        circuit.saes = [s.model_dump(mode="json") for s in defn.saes]
        circuit.members = [m.model_dump(mode="json") for m in defn.members]
        circuit.edges = [e.model_dump(mode="json") for e in defn.edges]
        circuit.budget = defn.budget.model_dump(mode="json") if defn.budget else None
        circuit.faithfulness = (defn.faithfulness.model_dump(mode="json")
                                if defn.faithfulness else None)
        circuit.rung = int(defn.displayed_rung())
        await db.commit()
        await db.refresh(circuit)
        return circuit

    @staticmethod
    async def delete(db: AsyncSession, circuit: Circuit) -> None:
        await db.delete(circuit)
        await db.commit()

    @staticmethod
    async def set_promoted(db: AsyncSession, circuit: Circuit,
                           promoted: bool = True) -> Circuit:
        """Badge, not gate (BR-012) — and reversible: a badge you can pin
        is a badge you can unpin (review R1 finding #7)."""
        circuit.promoted = promoted
        await db.commit()
        await db.refresh(circuit)
        return circuit

    # Back-compat alias (tests/tools may call promote()).
    @staticmethod
    async def promote(db: AsyncSession, circuit: Circuit) -> Circuit:
        return await CircuitService.set_promoted(db, circuit, True)

    # ── rung maintenance (017 writes validation results, then calls this) ──

    @staticmethod
    async def recompute_rung(db: AsyncSession, circuit: Circuit) -> Circuit:
        # Tolerate malformed stored rungs (017 writes here): clamp into 0..3
        # instead of crashing the write path (review R1 QA-4).
        def _safe(e):
            try:
                return EvidenceRung(max(0, min(3, int(e.get("rung", 0)))))
            except (TypeError, ValueError):
                return EvidenceRung.MINED
        rungs = [_safe(e) for e in (circuit.edges or [])]
        circuit.rung = int(circuit_rung(rungs))
        await db.commit()
        await db.refresh(circuit)
        return circuit

    # ── contract projection ─────────────────────────────────────────────

    @staticmethod
    def to_definition(circuit: Circuit) -> CircuitDefinitionV1:
        defn = CircuitService._validate(
            circuit.name, circuit.narrative, circuit.saes, circuit.members,
            circuit.edges, circuit.budget, circuit.faithfulness,
            getattr(circuit, "discovery", None),
        )
        defn.provenance = DefinitionProvenance(
            created_at=circuit.created_at,
            exported_at=datetime.utcnow(),
            mistudio_version=APP_VERSION,
        )
        return defn

    @staticmethod
    def slices_of(defn: CircuitDefinitionV1) -> List[ClusterDefinitionV1]:
        """BR-014: one valid v1 slice per member-bearing layer (definition-first
        so callers compute the parent rung ONCE from the same object)."""
        layers = sorted({m.layer for m in defn.members})
        return [to_layer_slice(defn, layer) for layer in layers]

    @staticmethod
    def to_slices(circuit: Circuit) -> List[ClusterDefinitionV1]:
        return CircuitService.slices_of(CircuitService.to_definition(circuit))
