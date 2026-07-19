"""
Circuit model (Feature 018, IDL-33).

A circuit is the durable record of a discovered (or hand-assembled)
cross-layer structure: members keyed to layers (feature refs or cluster
refs), typed edges carrying their full evidence trail (rung, statistics,
attribution, validation-manifest refs), per-layer budgets under one global
intensity, and optional faithfulness scores. A PROMOTED circuit IS the
loadable multi-layer steering profile — there is deliberately no dual
entity (018 FTDD §3).

Members/edges/budget/faithfulness are JSONB snapshots mirroring the
mistudio.circuit-definition/v1 contract shapes so export is a projection,
not a join. `discovery_run_id` is a soft reference — discovery runs are
prunable working data; circuits are curated artifacts.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _circuit_id() -> str:
    return f"crc_{uuid.uuid4().hex[:12]}"


class Circuit(Base):
    """A cross-layer circuit — reviewable, promotable, portable."""

    __tablename__ = "circuits"

    id = Column(String(36), primary_key=True, default=_circuit_id)

    name = Column(String(120), nullable=False)
    narrative = Column(Text, nullable=True)  # markdown
    granularity = Column(String(16), nullable=False, default="feature")  # feature | cluster

    # Contract-shaped JSONB snapshots (circuit_definition.py shapes).
    saes = Column(JSONB, nullable=False, default=list)      # [DefinitionSAERef]
    members = Column(JSONB, nullable=False, default=list)   # [CircuitMember]
    edges = Column(JSONB, nullable=False, default=list)     # [CircuitEdge]
    budget = Column(JSONB, nullable=True)                    # CircuitBudget
    faithfulness = Column(JSONB, nullable=True)              # CircuitFaithfulness
    discovery = Column(JSONB, nullable=True)                 # CircuitDiscoveryProvenance

    # Denormalized display rung (min over edges; recomputed on every write).
    rung = Column(Integer, nullable=False, default=0)

    promoted = Column(Boolean, nullable=False, default=False)
    discovery_run_id = Column(String(36), nullable=True)  # SOFT ref — runs are prunable
    model_id = Column(String(255), nullable=True)
    schema_version = Column(String(8), nullable=False, default="1")

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("idx_circuits_promoted_rung", "promoted", "rung"),
    )
