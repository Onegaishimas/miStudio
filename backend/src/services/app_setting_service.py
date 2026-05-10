"""
AppSetting service layer for business logic.

Handles CRUD operations with transparent encryption/decryption for sensitive values.
All sensitive values are encrypted before storage and decrypted (or masked) on retrieval.
"""

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.encryption import encrypt_value, decrypt_value, mask_value
from ..models.app_setting import AppSetting
from ..schemas.app_setting import AppSettingUpsert, AppSettingResponse

logger = logging.getLogger(__name__)

# Keys that are always encrypted at rest regardless of what the client sends.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "openai_api_key",
    "hf_token",
    "neuronpedia_api_key",
})


class AppSettingService:
    """Service class for application settings operations."""

    @staticmethod
    async def upsert(db: AsyncSession, data: AppSettingUpsert) -> tuple[AppSetting, bool]:
        """Create or update a setting. Returns (setting, is_new).

        Sensitive values are encrypted before storage.
        """
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == data.key)
        )
        existing = result.scalar_one_or_none()

        # Server-side sensitivity: known secrets are always encrypted regardless
        # of the client-supplied is_sensitive flag, preventing plaintext downgrade.
        is_sensitive = data.key in _SENSITIVE_KEYS or data.is_sensitive
        store_value = encrypt_value(data.value) if is_sensitive else data.value

        if existing:
            existing.value = store_value
            existing.is_sensitive = is_sensitive
            existing.category = data.category
            await db.flush()
            await db.refresh(existing)
            return existing, False
        else:
            setting = AppSetting(
                key=data.key,
                value=store_value,
                is_sensitive=is_sensitive,
                category=data.category,
            )
            db.add(setting)
            await db.flush()
            await db.refresh(setting)
            return setting, True

    @staticmethod
    async def get_by_key(db: AsyncSession, key: str, unmask: bool = False) -> Optional[AppSetting]:
        """Get a setting by key. Decrypts sensitive values if unmask=True, otherwise masks them."""
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if setting and setting.is_sensitive:
            # Expunge to prevent in-place mutation from dirtying the session
            db.expunge(setting)
            if unmask:
                setting.value = decrypt_value(setting.value, setting_key=setting.key)
            else:
                decrypted = decrypt_value(setting.value, setting_key=setting.key)
                setting.value = mask_value(decrypted)
        return setting

    @staticmethod
    async def get_by_category(
        db: AsyncSession, category: str
    ) -> list[AppSetting]:
        """Get all settings in a category. Sensitive values are masked."""
        result = await db.execute(
            select(AppSetting)
            .where(AppSetting.category == category)
            .order_by(AppSetting.key)
        )
        settings = list(result.scalars().all())
        for s in settings:
            if s.is_sensitive:
                db.expunge(s)
                try:
                    decrypted = decrypt_value(s.value, setting_key=s.key)
                    s.value = mask_value(decrypted)
                except Exception:
                    s.value = "***"
        return settings

    @staticmethod
    async def list_all(db: AsyncSession) -> list[AppSetting]:
        """List all settings. Sensitive values are masked."""
        result = await db.execute(
            select(AppSetting).order_by(AppSetting.category, AppSetting.key)
        )
        settings = list(result.scalars().all())
        for s in settings:
            if s.is_sensitive:
                db.expunge(s)
                try:
                    decrypted = decrypt_value(s.value, setting_key=s.key)
                    s.value = mask_value(decrypted)
                except Exception:
                    s.value = "***"
        return settings

    @staticmethod
    async def delete_by_key(db: AsyncSession, key: str) -> bool:
        """Delete a setting by key. Returns True if deleted."""
        result = await db.execute(
            delete(AppSetting).where(AppSetting.key == key)
        )
        return result.rowcount > 0

    @staticmethod
    async def get_decrypted_value(db: AsyncSession, key: str) -> Optional[str]:
        """Get the plaintext value of a setting (for internal backend use only).

        Returns None if the key doesn't exist.
        """
        result = await db.execute(
            select(AppSetting).where(AppSetting.key == key)
        )
        setting = result.scalar_one_or_none()
        if not setting:
            return None
        if setting.is_sensitive:
            return decrypt_value(setting.value, setting_key=setting.key)
        return setting.value
