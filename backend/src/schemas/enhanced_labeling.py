"""Pydantic schemas for enhanced per-feature labeling."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class EnhancedLabelingJobResponse(BaseModel):
    """Response schema for an enhanced labeling job."""

    id: str
    feature_id: str
    status: str          # queued | running | completed | failed
    phase: Optional[str]  # pass1 | pass2 | None
    examples_total: int
    examples_completed: int
    workers: int
    endpoint: str
    model: str
    celery_task_id: Optional[str]
    pass1_summaries: Optional[list[dict[str, Any]]]
    raw_synthesis: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)
