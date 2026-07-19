"""
Circuit endpoints (Feature 018, IDL-33): CRUD, promotion, rung-aware listing,
contract import/export, and per-layer v1 slice export.

Every response carries the rung AND the server-rendered rung_language string
(IDL-35 — the ONE language source; clients never invent causal phrasing).
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.database import get_db
from ....models.circuit import Circuit
from ....schemas.circuit_definition import CircuitDefinitionV1
from ....schemas.evidence_ladder import EvidenceRung, RUNG_NEXT_STEP, rung_language
from ....services.circuit_service import CircuitService, CircuitValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuits", tags=["circuits"])

Granularity = Literal["feature", "cluster"]


class CircuitCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    granularity: Granularity = "feature"
    saes: List[Dict[str, Any]]
    members: List[Dict[str, Any]]
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None
    discovery: Optional[Dict[str, Any]] = None
    discovery_run_id: Optional[str] = Field(None, max_length=36)
    model_id: Optional[str] = Field(None, max_length=255)


class CircuitUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    narrative: Optional[str] = Field(None, max_length=10_000)
    granularity: Optional[Granularity] = None
    saes: Optional[List[Dict[str, Any]]] = None
    members: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    budget: Optional[Dict[str, Any]] = None
    faithfulness: Optional[Dict[str, Any]] = None


class PromoteBody(BaseModel):
    promoted: bool = True  # a badge you can pin is a badge you can unpin


def _summary(circuit: Circuit) -> Dict[str, Any]:
    """Slim list row — full JSONB bodies only on detail fetch (review R1)."""
    rung = EvidenceRung(circuit.rung)
    layers = sorted({m.get("layer") for m in (circuit.members or [])})
    return {
        "id": circuit.id,
        "name": circuit.name,
        "granularity": circuit.granularity,
        "layers": layers,
        "member_count": len(circuit.members or []),
        "edge_count": len(circuit.edges or []),
        "rung": int(rung),
        "rung_language": rung_language(rung),
        "rung_next_step": RUNG_NEXT_STEP[rung],
        "promoted": circuit.promoted,
        "model_id": circuit.model_id,
        "updated_at": circuit.updated_at,
    }


def _out(circuit: Circuit) -> Dict[str, Any]:
    rung = EvidenceRung(circuit.rung)
    return {
        **_summary(circuit),
        "narrative": circuit.narrative,
        "saes": circuit.saes,
        "members": circuit.members,
        "edges": circuit.edges,
        "budget": circuit.budget,
        "faithfulness": circuit.faithfulness,
        "discovery": getattr(circuit, "discovery", None),
        "discovery_run_id": circuit.discovery_run_id,
        "created_at": circuit.created_at,
    }


async def _get_or_404(db: AsyncSession, circuit_id: str) -> Circuit:
    circuit = await CircuitService.get(db, circuit_id)
    if circuit is None:
        raise HTTPException(status_code=404, detail=f"Circuit {circuit_id} not found")
    return circuit


def _ascii_filename(name: str, suffix: str) -> str:
    """Content-Disposition is latin-1 — non-ASCII names crashed the header."""
    safe = "".join(ch if (ch.isascii() and (ch.isalnum() or ch in "-_")) else "-"
                   for ch in name.lower()).strip("-") or "circuit"
    return f"{safe}{suffix}"


@router.get("")
async def list_circuits(
    promoted: Optional[bool] = Query(None),
    min_rung: Optional[int] = Query(None, ge=0, le=3),
    granularity: Optional[Granularity] = Query(None),
    edge_type: Optional[Literal["computed", "persistence", "attention_mediated"]] = Query(
        None, description="Only circuits containing at least one edge of this type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    circuits = await CircuitService.list(
        db, promoted=promoted, min_rung=min_rung, granularity=granularity)
    if edge_type is not None:
        circuits = [c for c in circuits
                    if any(e.get("type") == edge_type for e in (c.edges or []))]
    total = len(circuits)
    page = circuits[offset:offset + limit]
    return {"circuits": [_summary(c) for c in page], "total": total,
            "limit": limit, "offset": offset}


@router.post("", status_code=201)
async def create_circuit(body: CircuitCreate, db: AsyncSession = Depends(get_db)):
    try:
        circuit = await CircuitService.create(db, body.model_dump())
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)


@router.post("/import", status_code=201)
async def import_circuit(payload: Dict[str, Any], db: AsyncSession = Depends(get_db)):
    """Import a mistudio.circuit-definition/v1 file (BR-013 round-trip).

    Kind-keyed: unknown kinds/major versions are rejected with an explicit
    message — never guessed at.
    """
    kind = payload.get("kind")
    if kind != "mistudio.circuit-definition":
        raise HTTPException(
            status_code=422,
            detail=f"Unknown kind {kind!r} — expected 'mistudio.circuit-definition'")
    try:
        defn = CircuitDefinitionV1.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Invalid circuit definition: {e.error_count()} error(s): {e.errors()[0].get('msg', '')}")
    try:
        circuit = await CircuitService.create(db, {
            "name": defn.name,
            "narrative": defn.narrative,
            "saes": [s.model_dump(mode="json") for s in defn.saes],
            "members": [m.model_dump(mode="json") for m in defn.members],
            "edges": [e.model_dump(mode="json") for e in defn.edges],
            "budget": defn.budget.model_dump(mode="json") if defn.budget else None,
            "faithfulness": (defn.faithfulness.model_dump(mode="json")
                             if defn.faithfulness else None),
            "discovery": (defn.discovery.model_dump(mode="json")
                          if defn.discovery else None),
        })
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
    data = body.model_dump(exclude_unset=True)
    # Structural fields cannot be nulled — reject explicitly rather than
    # surfacing a raw pydantic error (review R1 finding #12).
    for field in ("name", "saes", "members", "edges"):
        if field in data and data[field] is None:
            raise HTTPException(status_code=422,
                                detail=f"'{field}' cannot be null")
    try:
        circuit = await CircuitService.update(db, circuit, data)
    except CircuitValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _out(circuit)


@router.delete("/{circuit_id}")
async def delete_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    await CircuitService.delete(db, circuit)
    return {"deleted": circuit_id}


@router.post("/{circuit_id}/promote")
async def promote_circuit(circuit_id: str, body: Optional[PromoteBody] = None,
                          db: AsyncSession = Depends(get_db)):
    """Badge, not gate (BR-012) — and reversible (pass {"promoted": false})."""
    circuit = await _get_or_404(db, circuit_id)
    promoted = body.promoted if body is not None else True
    return _out(await CircuitService.set_promoted(db, circuit, promoted))


@router.get("/{circuit_id}/export")
async def export_circuit(circuit_id: str, db: AsyncSession = Depends(get_db)):
    circuit = await _get_or_404(db, circuit_id)
    try:
        defn = CircuitService.to_definition(circuit)
    except CircuitValidationError:
        # Never echo internal validation text on a 500 (security hardening
        # precedent — stack-trace exposure remediation).
        logger.exception("Stored circuit %s fails contract validation", circuit_id)
        raise HTTPException(status_code=500,
                            detail="Stored circuit failed contract validation; see server logs")
    return JSONResponse(
        content=defn.model_dump(mode="json"),
        headers={"Content-Disposition":
                 f'attachment; filename="{_ascii_filename(circuit.name, ".circuit.json")}"'},
    )


@router.post("/{circuit_id}/export-slices")
async def export_circuit_slices(circuit_id: str, db: AsyncSession = Depends(get_db)):
    """BR-014: per-layer cluster-definition/v1 slices (partial renderings).

    The parent rung is computed ONCE from the definition (min over edges) so
    the response and every slice marker agree — the stored column is display
    cache, not evidence truth (review R1 finding #4).
    """
    circuit = await _get_or_404(db, circuit_id)
    try:
        defn = CircuitService.to_definition(circuit)
        slices = CircuitService.slices_of(defn)
    except (CircuitValidationError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    parent_rung = int(defn.displayed_rung())
    return {
        "parent": circuit_id,
        "parent_rung": parent_rung,
        "parent_rung_language": rung_language(parent_rung),
        "slices": [s.model_dump(mode="json") for s in slices],
    }
