"""
Cluster profile schemas + the portable cluster-definition contract
(Feature 014, IDL-30).

Two families:
- Profile CRUD (`ClusterProfileCreate/Update/Out`) — the durable in-app entity.
- Interchange (`ClusterDefinitionV1` / `ClusterBundleV1`) — the versioned,
  consumer-neutral JSON that travels (export/import; future MILLM / unified-MCP
  / Open WebUI consumers). Strict validators; no secrets; no local paths.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1"
DEFINITION_KIND = "mistudio.cluster-definition"
BUNDLE_KIND = "mistudio.cluster-bundle"

MAX_MEMBERS = 20
MAX_BUNDLE = 50
MAX_NAME = 120
MAX_NARRATIVE = 10_000


# ── Shared member shape ─────────────────────────────────────────────────────

class ProfileMember(BaseModel):
    """A cluster member snapshot with its tuned strength."""

    feature_idx: int = Field(..., ge=0)
    label: Optional[str] = None
    similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    activation_frequency: Optional[float] = None
    max_activation: Optional[float] = None
    strength: float = Field(..., ge=-300.0, le=300.0)
    sign: Literal[1, -1] = 1
    pinned: bool = False


class ProfileBudget(BaseModel):
    """Allocation snapshot from Feature 013 (self-describing: formula + constants travel)."""

    B: Optional[float] = None
    B_dir: Optional[float] = None
    G: Optional[float] = None
    f_eff: Optional[float] = None
    formula_id: Optional[str] = None
    constants: Optional[Dict[str, float]] = None
    intensity: float = Field(1.0, ge=0.0, le=2.0)
    intensity_range: List[float] = Field(default_factory=lambda: [0.0, 2.0])


# ── Profile CRUD ────────────────────────────────────────────────────────────

class ClusterProfileCreate(BaseModel):
    sae_id: Optional[str] = None
    model_id: Optional[str] = None
    extraction_id: Optional[str] = None
    source_group_id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    display_token: Optional[str] = None
    members: List[ProfileMember] = Field(..., min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None


class ClusterProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    members: Optional[List[ProfileMember]] = Field(None, min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None


class ClusterProfileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    sae_id: Optional[str]
    model_id: Optional[str]
    extraction_id: Optional[str]
    source_group_id: Optional[str]
    name: str
    narrative: Optional[str]
    display_token: Optional[str]
    members: List[ProfileMember]
    budget: Optional[ProfileBudget]
    schema_version: str
    imported_from: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ClusterProfileListResponse(BaseModel):
    data: List[ClusterProfileOut]
    total: int


# ── Interchange format (versioned, consumer-neutral) ───────────────────────

class DefinitionModelRef(BaseModel):
    hf_id: Optional[str] = None
    mistudio_model_id: Optional[str] = None


class DefinitionSAERef(BaseModel):
    mistudio_sae_id: Optional[str] = None
    layer: Optional[int] = None
    hook_type: Optional[str] = None
    n_features: Optional[int] = None
    d_model: Optional[int] = None
    source_hint: Optional[str] = Field(
        None, description="e.g. 'hf:repo/path' — NEVER an absolute local path"
    )

    @field_validator("source_hint")
    @classmethod
    def no_local_paths(cls, v: Optional[str]) -> Optional[str]:
        """Reject absolute/relative filesystem paths — the format must stay portable."""
        if v and (v.startswith("/") or v.startswith("~") or v.startswith("..") or ":\\" in v):
            raise ValueError("source_hint must not be a filesystem path")
        return v


class DefinitionProvenance(BaseModel):
    created_at: Optional[datetime] = None
    exported_at: Optional[datetime] = None
    mistudio_version: Optional[str] = None
    source_note: Optional[str] = Field(None, max_length=500)


class ClusterDefinitionV1(BaseModel):
    """One portable cluster definition (the mobile artifact — IDL-30)."""

    kind: Literal["mistudio.cluster-definition"] = DEFINITION_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    name: str = Field(..., min_length=1, max_length=MAX_NAME)
    narrative: Optional[str] = Field(None, max_length=MAX_NARRATIVE)
    display_token: Optional[str] = None
    model: DefinitionModelRef = Field(default_factory=DefinitionModelRef)
    sae: DefinitionSAERef = Field(default_factory=DefinitionSAERef)
    members: List[ProfileMember] = Field(..., min_length=1, max_length=MAX_MEMBERS)
    budget: Optional[ProfileBudget] = None
    provenance: DefinitionProvenance = Field(default_factory=DefinitionProvenance)


class ClusterBundleV1(BaseModel):
    """A multi-cluster export: an array of definitions in one file."""

    kind: Literal["mistudio.cluster-bundle"] = BUNDLE_KIND
    schema_version: Literal["1"] = SCHEMA_VERSION
    definitions: List[ClusterDefinitionV1] = Field(..., min_length=1, max_length=MAX_BUNDLE)


# ── Import ──────────────────────────────────────────────────────────────────

class ImportRequest(BaseModel):
    """Import a definition or bundle (frontend reads the file client-side)."""

    payload: Dict[str, Any] = Field(..., description="Parsed JSON of a definition or bundle")
    bind_sae_id: Optional[str] = Field(
        None, description="Explicit SAE to bind to (overrides auto-binding)"
    )


class ImportItemResult(BaseModel):
    name: str
    status: Literal["imported", "imported_unbound", "blocked", "error"]
    profile_id: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ImportResponse(BaseModel):
    results: List[ImportItemResult]
    imported: int
    blocked: int
    errors: int


class ExportBundleRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1, max_length=MAX_BUNDLE)
