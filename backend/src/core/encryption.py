"""
AES-256-GCM envelope encryption for sensitive application settings.

Provides encrypt/decrypt functions and masked display for API keys and secrets.
Uses HKDF key derivation from SETTINGS_ENCRYPTION_KEY or falls back to SECRET_KEY.

Security properties:
- AES-256-GCM: Authenticated encryption (confidentiality + integrity)
- 96-bit random nonce per encryption (never reused)
- HKDF-SHA256 key derivation ensures proper key material
- Post-quantum resistant for at-rest encryption (256-bit symmetric)
"""

import base64
import os
import logging

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)

# Module-level derived key (initialized on first use)
_derived_key: bytes | None = None


def _get_encryption_key() -> bytes:
    """Derive a 256-bit AES key from the configured key material using HKDF."""
    global _derived_key
    if _derived_key is not None:
        return _derived_key

    from .config import settings

    # Prefer dedicated encryption key, fall back to secret_key
    key_material = os.environ.get("SETTINGS_ENCRYPTION_KEY") or settings.secret_key

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,  # 256 bits
        salt=b"mistudio-settings-v1",
        info=b"settings-encryption",
    )
    _derived_key = hkdf.derive(key_material.encode("utf-8"))
    return _derived_key


def encrypt_value(plaintext: str) -> str:
    """Encrypt a plaintext string using AES-256-GCM.

    Returns a base64-encoded string containing: nonce (12 bytes) + ciphertext + tag (16 bytes).
    """
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # Concatenate nonce + ciphertext+tag, then base64 encode
    return base64.b64encode(nonce + ciphertext).decode("ascii")


def decrypt_value(encrypted: str) -> str:
    """Decrypt a base64-encoded AES-256-GCM ciphertext.

    Expects the format produced by encrypt_value(): base64(nonce + ciphertext + tag).
    """
    key = _get_encryption_key()
    raw = base64.b64decode(encrypted)
    nonce = raw[:12]
    ciphertext = raw[12:]
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


def mask_value(plaintext: str, visible_prefix: int = 3, visible_suffix: int = 4) -> str:
    """Create a masked display string for a sensitive value.

    Examples:
        "sk-proj-abc123xyz789" -> "sk-...789"
        "hf_abcdefghijk" -> "hf_...hijk"
        "short" -> "sho...ort"
    """
    if len(plaintext) <= visible_prefix + visible_suffix:
        return plaintext[:visible_prefix] + "..." + plaintext[-visible_suffix:] if len(plaintext) > 3 else "***"
    return plaintext[:visible_prefix] + "..." + plaintext[-visible_suffix:]


def reset_key_cache() -> None:
    """Reset the cached derived key. Useful for testing."""
    global _derived_key
    _derived_key = None
