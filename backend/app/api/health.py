"""Endpoints de salud y diagnóstico del sistema."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.config import settings
from app.core.deps import check_ffmpeg

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    bins = check_ffmpeg()
    return {
        "app": settings.app_name,
        "version": __version__,
        "status": "ok",
        "provider": settings.llm_provider,
        "binaries": {
            name: {"available": s.available, "version": s.version, "path": s.path}
            for name, s in bins.items()
        },
    }
