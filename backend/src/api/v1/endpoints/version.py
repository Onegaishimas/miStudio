"""Version endpoint."""

from pathlib import Path
from fastapi import APIRouter

router = APIRouter(prefix="/version", tags=["version"])

def _read_version() -> str:
    """Read version from VERSION file at repo root."""
    for candidate in [
        Path(__file__).parents[5] / "VERSION",   # repo root when installed normally
        Path(__file__).parents[4] / "VERSION",
        Path(__file__).parents[3] / "VERSION",
    ]:
        if candidate.exists():
            return candidate.read_text().strip()
    return "unknown"


@router.get("", summary="Get application version")
def get_version():
    return {"version": _read_version(), "app": "miStudio"}
