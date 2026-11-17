"""Endpoints de sistema (read-only)."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.media_capabilities import get_media_capabilities

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/media-capabilities")
def media_capabilities(
    refresh: bool = Query(default=False, description="Fuerza un re-diagnóstico"),
) -> dict:
    """Diagnóstico multimedia (Capability Doctor). Read-only, sin rutas locales."""
    return get_media_capabilities(refresh=refresh)
