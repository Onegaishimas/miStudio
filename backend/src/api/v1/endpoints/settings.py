"""
Application Settings API endpoints.

Provides CRUD operations for persistent application settings with
transparent encryption for sensitive values (API keys, tokens).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ....core.deps import get_db
from ....schemas.app_setting import (
    AppSettingUpsert,
    AppSettingResponse,
    AppSettingBulkUpsert,
    AppSettingBulkResponse,
)
from ....services.app_setting_service import AppSettingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=list[AppSettingResponse])
async def list_settings(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
):
    """List all settings, optionally filtered by category. Sensitive values are masked."""
    if category:
        return await AppSettingService.get_by_category(db, category)
    return await AppSettingService.list_all(db)


@router.get("/{key}", response_model=AppSettingResponse)
async def get_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single setting by key. Sensitive values are masked."""
    setting = await AppSettingService.get_by_key(db, key)
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return setting


@router.put("", response_model=AppSettingResponse, status_code=200)
async def upsert_setting(
    data: AppSettingUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update a setting (upsert). Sensitive values are encrypted at rest."""
    try:
        setting, is_new = await AppSettingService.upsert(db, data)
        # Return masked response for sensitive values. CRITICAL: expunge the row from
        # the session before mutating its `value` field, otherwise SQLAlchemy will
        # autoflush/commit the masked string back over the encrypted ciphertext on
        # request teardown — silently corrupting the row each save.
        if setting.is_sensitive:
            from ....core.encryption import decrypt_value, mask_value
            db.expunge(setting)
            setting.value = mask_value(decrypt_value(setting.value, setting_key=setting.key))
        return setting
    except Exception as e:
        logger.error(f"Failed to upsert setting '{data.key}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save setting: {str(e)}")


@router.put("/bulk", response_model=AppSettingBulkResponse)
async def bulk_upsert_settings(
    data: AppSettingBulkUpsert,
    db: AsyncSession = Depends(get_db),
):
    """Create or update multiple settings at once."""
    results = []
    created = 0
    updated = 0
    try:
        for item in data.settings:
            setting, is_new = await AppSettingService.upsert(db, item)
            if setting.is_sensitive:
                from ....core.encryption import decrypt_value, mask_value
                # Expunge before mutating — otherwise the masked string overwrites
                # the encrypted ciphertext on session commit (see upsert_setting).
                db.expunge(setting)
                setting.value = mask_value(decrypt_value(setting.value, setting_key=setting.key))
            results.append(setting)
            if is_new:
                created += 1
            else:
                updated += 1
        return AppSettingBulkResponse(data=results, created=created, updated=updated)
    except Exception as e:
        logger.error(f"Failed to bulk upsert settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.delete("/{key}", status_code=204)
async def delete_setting(
    key: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a setting by key."""
    deleted = await AppSettingService.delete_by_key(db, key)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
