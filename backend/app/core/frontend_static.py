"""Sirve el frontend compilado desde el mismo origen que la API.

Solo se activa cuando `FRONTEND_DIST_DIR` apunta a un build de Vite (la app macOS
empaquetada). En desarrollo la UI sigue viniendo del servidor de Vite.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
    """Devuelve index.html para rutas del cliente (p. ej. /jobs/123).

    Las rutas /api/* nunca caen al index: un endpoint inexistente sigue siendo 404.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            normalized = path.replace("\\", "/").lstrip("/")
            if exc.status_code != 404 or normalized == "api" or normalized.startswith("api/"):
                raise
            return await super().get_response("index.html", scope)


def mount_frontend(app: FastAPI, dist_dir: Path) -> None:
    """Monta `dist_dir` en la raíz. Debe llamarse DESPUÉS de registrar las rutas /api."""
    if not (dist_dir / "index.html").is_file():
        raise RuntimeError(
            f"FRONTEND_DIST_DIR no contiene index.html: {dist_dir}. "
            "Ejecuta el build del frontend antes de empaquetar."
        )
    app.mount("/", SPAStaticFiles(directory=dist_dir, html=True), name="frontend")
