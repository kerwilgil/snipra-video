"""Punto de entrada de la API FastAPI de MediaShorts."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import (
    channels,
    clips,
    compositions,
    editor,
    extension_pairing,
    health,
    jobs,
    storage,
    system,
)
from app.api import settings as settings_api
from app.core.config import settings
from app.core.db import init_db
from app.core.frontend_static import mount_frontend
from app.core.runtime_settings import load_runtime_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    load_runtime_settings()
    init_db()
    yield


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 sin traceback ni secretos: bug real, pero la respuesta sigue siendo util.

    Ni la respuesta ni el log pueden exponer secretos: el traceback se formatea
    y sanitiza ANTES de registrarse (nada de logger.exception() crudo).
    """
    import logging

    from app.core.sanitize import sanitize_error_text, sanitize_traceback

    logging.getLogger("snipra.api").error(
        "error no controlado en %s %s\n%s",
        request.method,
        request.url.path,
        sanitize_traceback(exc),
    )
    detail = sanitize_error_text(str(exc) or type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "code": "INTERNAL_ERROR",
                "message": f"{type(exc).__name__}: {detail[:300]}",
            }
        },
    )


_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_EXTENSION_TOKEN_HEADER = "x-snipra-extension-token"
_EXTENSION_ORIGIN_PREFIXES = ("chrome-extension://", "moz-extension://")
# Unico endpoint que la extension puede llamar SIN token: solo crea un
# `request_id` pendiente (ver app/api/extension_pairing.py), no otorga nada por
# si mismo. El token real solo se genera cuando un humano aprueba desde /pair.
_PAIRING_BOOTSTRAP_PATH = "/api/extension/pair/request"


@app.middleware("http")
async def reject_cross_origin_mutations(request: Request, call_next):
    """Bloquea escrituras desde sitios web ajenos sin impedir clientes locales directos.

    La extensión propia de Snipra (popup MV3) llega con Origin: chrome-extension://<id>.
    Una web maliciosa no puede spoofear ese scheme, pero CUALQUIER otra extensión
    instalada también correría con ese mismo scheme, así que el origin por sí solo no
    autoriza nada: se exige además el token compartido (X-Snipra-Extension-Token) que
    el usuario configura una sola vez en Ajustes -> Extensión de navegador. Sin token
    configurado (vacío por defecto), estas peticiones se rechazan siempre (fail-closed).
    """
    if request.url.path.startswith("/api/") and request.method in _MUTATING_METHODS:
        origin = request.headers.get("origin")
        if origin is not None and origin.startswith(_EXTENSION_ORIGIN_PREFIXES):
            if request.url.path == _PAIRING_BOOTSTRAP_PATH:
                return await call_next(request)
            configured_token = settings.extension_token
            provided_token = request.headers.get(_EXTENSION_TOKEN_HEADER, "")
            if not configured_token or not hmac.compare_digest(provided_token, configured_token):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Token de extensión inválido o no configurado."},
                )
            return await call_next(request)

        allowed_origins = {item.rstrip("/") for item in settings.cors_origins}
        cross_site = request.headers.get("sec-fetch-site", "").lower() == "cross-site"
        if cross_site or (origin is not None and origin.rstrip("/") not in allowed_origins):
            return JSONResponse(
                status_code=403,
                content={"detail": "Origen no permitido para modificar Snipra."},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")
app.include_router(clips.router, prefix="/api")
app.include_router(settings_api.router, prefix="/api")
app.include_router(storage.router, prefix="/api")
app.include_router(channels.router, prefix="/api")
app.include_router(compositions.router, prefix="/api")
app.include_router(editor.router, prefix="/api")
app.include_router(extension_pairing.router, prefix="/api")
app.include_router(system.router, prefix="/api")


if settings.frontend_dist_dir is None:

    @app.get("/")
    def root() -> dict:
        return {"app": settings.app_name, "version": __version__, "docs": "/docs"}
else:
    # Debe quedar al final: las rutas /api ya están registradas y tienen prioridad.
    mount_frontend(app, settings.frontend_dist_dir)
