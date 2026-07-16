"""MCP server configuration (env prefix ``MCP_``)."""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_CATEGORIES = {"read", "groups", "steering", "labeling", "experiments", "profiles", "jobs", "admin"}
DEFAULT_CATEGORIES = "read,groups,steering,labeling,experiments,profiles,jobs"


class MCPSettings(BaseSettings):
    """Runtime configuration for the miStudio MCP server."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="ignore")

    auth_token: str = Field(default="", description="Bearer token required on the HTTP transport")
    allow_anonymous: bool = Field(
        default=False, description="Permit startup without a token (stdio/localhost dev only)"
    )
    host: str = Field(default="0.0.0.0", description="Bind host (LAN-reachable by default)")
    port: int = Field(default=8765)
    tool_categories: str = Field(default=DEFAULT_CATEGORIES)
    steering_max_concurrent: int = Field(default=2, ge=1)
    steering_max_new_tokens: int = Field(default=512, ge=1, le=2048)
    steering_approval: bool = Field(
        default=False, description="Route agent steering through operator approval"
    )

    # Backend base URL — not MCP_-prefixed; matches worker convention.
    @property
    def api_url(self) -> str:
        return os.environ.get("MISTUDIO_API_URL", "http://localhost:8000").rstrip("/")

    def enabled_categories(self) -> set[str]:
        requested = {c.strip() for c in self.tool_categories.split(",") if c.strip()}
        unknown = requested - VALID_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown MCP tool categories: {sorted(unknown)} (valid: {sorted(VALID_CATEGORIES)})"
            )
        return requested
