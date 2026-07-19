"""
Circuit endpoints (Feature 018, IDL-33): CRUD, promotion, rung-aware listing,
contract export, and per-layer v1 slice export.

Every response carries the rung AND the server-rendered rung_language string
(IDL-35 — the ONE language source; clients never invent causal phrasing).
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....models.circuit import Circuit
from ....schemas.evidence_ladder import EvidenceRung, RUNG_NEXT_STEP, rung_language
from ....services.circuit_service import CircuitService, CircuitValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuits", tags=["circuits"])


class CircuitCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    granularity: str = "feature"
    saes: List[Dict[str, Any]]
    members: List[Dict[str, Any]]
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Optional[Dict[str, Any]] = None
    discovery_run_id: Optional[str] = None
    model_id: Optional[str] = None


class CircuitUpdate(BaseModel):
    name: Optional[str] = None
    narrative: Optional[str] = None
    saes: Optional[List[Dict[str, Any]]] = None
    members: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None


def _out(circuit: Circuit) -> Dict[str, Any]:
    rung = EvidenceRung(circuit.rung)
    return {
        "id": circuit.id,
        "name": circuit.name,
        "narrative": circuit.narrative,
        "granularity": circuit.granularity,
        "saes": circuit.saes,
        "members": circuit.members,
        "edges": circuit.edges,
        "budget": circuit.budget,
        "faithfulness": circuit.faithfulness,
        "rung": int(rung),
        "rung_language": rung_language(rung),
        "rung_next_step": RUNG_NEXT_STEP[rung],
        "promoted": circuit.promoted,
        "discovery_run_id": circuit.discovery_run_id,
        "created_at": circuit.created_at,
        "updated_at": circuit.updated_at,
    }


async def _get_or_404(db: AsyncSession, circuit_id: str) -> Circuit:
    circuit = await CircuitService.get(db, circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit {circuit_id} not found")
    return circuit


@router.get("")
async def list_circuits(
    promoted: Optional[bool] = Query(None),
    min_rung: Optional[int] = Query(None, ge=0, le=3),
    granularity: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    circuits = await CircuitService.list(
        db, promoted=promoted, min_rung=min_rung, granularity=granularity)
    return {"circuits": [_out(c) for c in circuits], "total": len(circuits)}


@router.post("", status_code=201)
async def create_circuit(body: CircuitCreate, db: AsyncSession = Depends(get_db)):
    try:
        circuit = await CircuitService.create(db, body.model_dump())
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)


@router.get("/{circuit_id}")
async def get_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    return _out(await _get_or_404(db, circuit_id))


@router.patch("/{circuit_id}")
async def update_circuit(circuit_id: str, body: CircuitUpdate,
                         db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    try:
        circuit = await CircuitService.update(
            db, circuit, body.model_dump(exclude_unset=True))
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)


@router.delete("/{circuit_id}")
async def delete_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    await CircuitService.delete(db, circuit)
    return {"deleted": circuit_id}


@router.post("/{circuit_id}/promote")
async def promote_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """Badge, not gate (BR-012): promotion never requires a rung."""
    circuit = await _get_or_404(db, circuit_id)
    return _out(await CircuitService.promote(db, circuit))


@router.get("/{circuit_id}/export")
async def export_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    try:
        defn = CircuitService.to_definition(circuit)
    except CircuitValidationError as e:
        raise HTTPException(status_code=500, detail=f"Stored circuit invalid: {e}")
    from fastapi.responses import JSONResponse

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in circuit.name.lower())
    return JSONResponse(
        content=defn.model_dump(mode="json"),
        headers={"Content-Disposition":
                 f'attachment; filename="{safe}.circuit.json"'},
    )


@router.post("/{circuit_id}/export-slices")
async def export_circuit_slices(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """BR-014: per-layer cluster-definition/v1 slices (partial renderings)."""
    circuit = await _get_or_404(db, circuit_id)
    try:
        slices = CircuitService.to_slices(circuit)
    except (CircuitValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "parent": circuit_id,
        "parent_rung": circuit.rung,
        "parent_rung_language": rung_language(circuit.rung),
        "slices": [s.model_dump(mode="json") for s in slices],
    }
