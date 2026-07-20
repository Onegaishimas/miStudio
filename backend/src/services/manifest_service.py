"""
Manifest service (Feature 017, FTDD §4): persist / retrieve / reproduce
validation manifests. Reproduction re-executes from the self-contained
payload and stores a `reproduction` manifest with per-value deltas + a
tolerance verdict — the correctness test that a manifest really is
self-contained.
"""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.validation_manifest import ValidationManifest

logger = logging.getLogger(__name__)

# Fields a manifest payload MUST carry to be self-contained (no live refs).
_REQUIRED_PAYLOAD_KEYS = {"intervention", "config", "seeds"}


class ManifestError(ValueError):
    """Malformed manifest — surfaces as a 422."""


def validate_payload(kind: str, payload: Dict[str, Any]) -> None:
    """A manifest with live refs that can drift is not reproducible."""
    if kind not in ("edge_batch", "faithfulness", "reproduction"):
        raise ManifestError(f"Unknown manifest kind {kind!r}")
    if kind in ("edge_batch", "faithfulness"):
        missing = _REQUIRED_PAYLOAD_KEYS - set(payload or {})
        if missing:
            raise ManifestError(
                f"{kind} manifest payload missing self-contained keys: "
                f"{sorted(missing)}")
    # No secrets/filesystem paths leak into a portable manifest.
    _assert_no_paths(payload)


def _assert_no_paths(obj: Any, _depth: int = 0) -> None:
    if _depth > 12:
        return
    if isinstance(obj, str):
        if obj.startswith("/data/") or obj.startswith("/home/"):
            raise ManifestError("manifest payload must not embed filesystem paths")
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_paths(v, _depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _assert_no_paths(v, _depth + 1)


class ManifestService:
    @staticmethod
    async def create(db: AsyncSession, *, kind: str, payload: Dict[str, Any],
                     discovery_run_id: Optional[str] = None,
                     circuit_id: Optional[str] = None,
                     parent_manifest_id: Optional[str] = None) -> ValidationManifest:
        validate_payload(kind, payload)
        m = ValidationManifest(
            kind=kind, payload=payload, discovery_run_id=discovery_run_id,
            circuit_id=circuit_id, parent_manifest_id=parent_manifest_id)
        db.add(m)
        await db.commit()
        await db.refresh(m)
        return m

    @staticmethod
    async def get(db: AsyncSession, manifest_id: str) -> Optional[ValidationManifest]:
        return (await db.execute(
            select(ValidationManifest).where(
                ValidationManifest.id == manifest_id))).scalar_one_or_none()

    @staticmethod
    async def list_by_parent(db: AsyncSession, *,
                             discovery_run_id: Optional[str] = None,
                             circuit_id: Optional[str] = None
                             ) -> List[ValidationManifest]:
        q = select(ValidationManifest).order_by(
            ValidationManifest.created_at.desc())
        if discovery_run_id:
            q = q.where(ValidationManifest.discovery_run_id == discovery_run_id)
        if circuit_id:
            q = q.where(ValidationManifest.circuit_id == circuit_id)
        return list((await db.execute(q)).scalars().all())

    @staticmethod
    def reproduction_verdict(original: Dict[str, Any], repro: Dict[str, Any],
                             tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare per-edge ES between an original edge_batch payload and a
        re-executed one. Deterministic greedy passes ⇒ deltas should be ~0;
        tolerance guards fp noise. Returns {deltas, max_delta, within_tolerance}."""
        orig_edges = {_edge_key(e): e for e in original.get("edges", [])}
        deltas = []
        max_delta = 0.0
        for e in repro.get("edges", []):
            k = _edge_key(e)
            o = orig_edges.get(k)
            if o is None or o.get("effect_size") is None or e.get("effect_size") is None:
                continue
            d = abs(float(e["effect_size"]) - float(o["effect_size"]))
            deltas.append({"edge": k, "delta": round(d, 5)})
            max_delta = max(max_delta, d)
        return {"deltas": deltas, "max_delta": round(max_delta, 5),
                "within_tolerance": max_delta <= tolerance,
                "tolerance": tolerance}


def _edge_key(e: Dict[str, Any]):
    up, down = e.get("up", {}), e.get("down", {})
    return (up.get("layer"), up.get("feature_idx"),
            down.get("layer"), down.get("feature_idx"))
