"""
Tests for /api/v1/settings — specifically the round-trip safety of sensitive values.

Regression coverage for a bug where the upsert endpoint mutated the SQLAlchemy
session-bound `setting.value` to the masked form for the response, causing the
masked string to overwrite the encrypted ciphertext on session commit.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models.app_setting import AppSetting

pytestmark = pytest.mark.asyncio


REAL_OPENAI_KEY = "sk-proj-" + "x" * 40 + "TMKA"  # 52 chars, like a real key


class TestUpsertPreservesEncryptedValue:
    """Saving a sensitive value must not corrupt the at-rest ciphertext."""

    async def test_single_upsert_keeps_ciphertext_in_db(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        response = await client.put(
            "/api/v1/settings",
            json={
                "key": "openai_api_key",
                "value": REAL_OPENAI_KEY,
                "is_sensitive": True,
                "category": "api_keys",
            },
        )
        assert response.status_code == 200
        body = response.json()

        # Response should be masked (sk-...TMKA — first 3 + ... + last 4)
        assert body["value"] != REAL_OPENAI_KEY
        assert body["value"].startswith("sk-")
        assert body["value"].endswith("TMKA")
        assert "..." in body["value"]

        # DB must contain the actual encrypted ciphertext, not the mask.
        # AES-GCM envelope = 12-byte nonce + ciphertext + 16-byte tag, base64-encoded.
        # For a 52-char plaintext, the stored value will be ~110+ chars.
        result = await async_session.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        row = result.scalar_one()
        assert row.is_sensitive is True
        assert len(row.value) > len(REAL_OPENAI_KEY), (
            f"DB value len={len(row.value)} is not longer than plaintext "
            f"len={len(REAL_OPENAI_KEY)} — masked string was committed over ciphertext"
        )
        assert "..." not in row.value, "DB value contains '...' — the mask leaked into storage"

    async def test_re_save_does_not_progressively_truncate(
        self, client: AsyncClient, async_session: AsyncSession
    ):
        """Multiple saves of the same key must each round-trip cleanly."""
        for _ in range(3):
            response = await client.put(
                "/api/v1/settings",
                json={
                    "key": "openai_api_key",
                    "value": REAL_OPENAI_KEY,
                    "is_sensitive": True,
                    "category": "api_keys",
                },
            )
            assert response.status_code == 200

        result = await async_session.execute(
            select(AppSetting).where(AppSetting.key == "openai_api_key")
        )
        row = result.scalar_one()
        assert len(row.value) > len(REAL_OPENAI_KEY)
        assert "..." not in row.value
