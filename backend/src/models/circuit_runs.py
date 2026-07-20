"""
Circuit capture + discovery run models (Feature 016, IDL-32/IDL-36).

`circuit_capture_runs` mirrors the on-disk store manifest (corpus refs,
layers, thresholds, split, SAE fingerprints) so listings never touch disk;
the store directory under /data/circuit_captures/{id}/ is the artifact.
Stale-flagged (not deleted) when a referenced SAE changes.

`circuit_discovery_runs` holds the mined candidates (BOTH orderings —
coactivation-only and attribution-re-ranked — so 017 can measure uplift)
plus the first-class run report (null summary, FDR discipline, held-out
replication rate, stage counts). Discovery emits raw candidates; 018's
classifier annotates types/rungs downstream.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ..core.database import Base


def _cap_id() -> str:
    return f"cap_{uuid.uuid4().hex[:12]}"


def _dsc_id() -> str:
    return f"dsc_{uuid.uuid4().hex[:12]}"


class CircuitCaptureRun(Base):
    """One capture run = one on-disk event store + its manifest mirror."""

    __tablename__ = "circuit_capture_runs"

    id = Column(String(36), primary_key=True, default=_cap_id)
    status = Column(String(16), nullable=False, default="pending")
    # pending | estimating | running | completed | failed | cancelled
    progress = Column(Float, nullable=True)  # 0-100
    error_message = Column(Text, nullable=True)

    # Manifest mirror (single JSONB — the on-disk manifest.json is identical):
    # {corpus: {dataset_id, tokenization_id, sample_cap}, layers: [{layer,
    #  sae_id, threshold_mode, epsilon}], split: {method, ratio, seed,
    #  heldout_docs[]}, counts, bytes, sae_fingerprints, attention_capture?}
    manifest = Column(JSONB, nullable=False, default=dict)

    store_path = Column(String(1000), nullable=True)
    events_total = Column(Integer, nullable=True)
    bytes_total = Column(Integer, nullable=True)
    stale = Column(Boolean, nullable=False, default=False)  # SAE changed since capture
    celery_task_id = Column(String(155), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow,
                        onupdate=datetime.utcnow)


class CircuitDiscoveryRun(Base):
    """One discovery run over a capture store (+ optional attribution pass)."""

    __tablename__ = "circuit_discovery_runs"

    id = Column(String(36), primary_key=True, default=_dsc_id)
    capture_run_id = Column(String(36), nullable=False)  # soft ref (prunable stores)
    status = Column(String(16), nullable=False, default="pending")
    progress = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # {granularity: feature|cluster, mode: seeded|open, seed_refs, s_min,
    #  null_shuffles, null_percentile, fdr_q, cohesion_floor, ...}
    params = Column(JSONB, nullable=False, default=dict)

    # First-class report (FPRD §3.6): null_summary, fdr discipline,
    # replication_rate, counts_by_stage, attribution envelope + uplift
    # placeholder (017 fills), echo-filter counts (018 feeds back).
    report = Column(JSONB, nullable=True)

    # [{up, down, granularity, stats: {pmi, lift, support, spearman,
    #   null_pct}, replicated_heldout, attribution: {...}|null,
    #   orderings: {coact_rank, attr_rank}}] — cap 2000, truncation noted.
    candidates = Column(JSONB, nullable=True)

    # Attribution is a SEPARATE lifecycle on the same run (R1 QA-P2): a failed
    # or cancelled attribution pass must not make the completed DISCOVERY
    # present as failed. null until an attribution pass is launched.
    attribution_status = Column(String(16), nullable=True)
    # pending | running | completed | failed | cancelled
    attribution_progress = Column(Float, nullable=True)
    attribution_error = Column(Text, nullable=True)
    # Attribution's OWN task id — the discovery task id must stay revocable
    # separately (R2 A3: one shared celery_task_id revoked the wrong task).
    attribution_task_id = Column(String(155), nullable=True)

    celery_task_id = Column(String(155), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow,
                        onupdate=datetime.utcnow)
