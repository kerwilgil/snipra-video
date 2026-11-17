"""Emparejamiento de un clic para el token de la extension de navegador.

Reemplaza el copiar/pegar manual del token por un flujo de 3 pasos, sin debilitar
el modelo de seguridad de main.py::reject_cross_origin_mutations:

1. La extension llama POST /pair/request (sin token -- ver excepcion explicita en
   main.py) y recibe un `request_id` de un solo uso. Esto NO otorga nada por si
   solo: solo registra "alguien pidio emparejar".
2. La extension abre en una pestaña normal el Studio local en /pair?request_id=...
   El usuario, ya en su propia sesion de navegador autenticada localmente, hace
   clic en "Aprobar" -- esa es la unica prueba de que quien aprueba es el dueño
   de la maquina, igual que antes lo era copiar el token desde Ajustes. Esa accion
   llama POST /pair/approve, que recien ahi genera/reutiliza el token real.
3. La extension hace polling de GET /pair/claim: cuando el request fue aprobado,
   se entrega el token UNA sola vez y el request se borra (no se puede reclamar
   dos veces). Si nadie aprueba, expira solo a los 5 minutos.

Una extension maliciosa podria llamar a /pair/request tambien, pero no puede
forzar el clic humano de aprobacion ni sabe a que pestaña corresponde su
request_id salvo que ya tenga permisos para leer otras pestañas -- en cuyo caso
el copy/paste manual anterior tampoco la habria detenido.
"""

from __future__ import annotations

import secrets
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.runtime_settings import apply_runtime_settings

router = APIRouter(prefix="/extension", tags=["extension"])

_PENDING_TTL_SECONDS = 5 * 60


class _PendingRequest:
    __slots__ = ("created_at", "approved", "token")

    def __init__(self) -> None:
        self.created_at = time.monotonic()
        self.approved = False
        self.token: str | None = None


_pending: dict[str, _PendingRequest] = {}


def _prune_expired() -> None:
    now = time.monotonic()
    expired = [rid for rid, req in _pending.items() if now - req.created_at > _PENDING_TTL_SECONDS]
    for rid in expired:
        del _pending[rid]


@router.post("/pair/request")
def create_pairing_request() -> dict:
    _prune_expired()
    request_id = secrets.token_urlsafe(24)
    _pending[request_id] = _PendingRequest()
    return {"request_id": request_id, "expires_in_seconds": _PENDING_TTL_SECONDS}


class ApproveBody(BaseModel):
    request_id: str


@router.post("/pair/approve")
def approve_pairing_request(body: ApproveBody) -> dict:
    _prune_expired()
    req = _pending.get(body.request_id)
    if req is None:
        raise HTTPException(404, "Esta solicitud de emparejamiento expiro o no existe.")

    token = secrets.token_urlsafe(32)
    apply_runtime_settings({"extension_token": token})
    req.approved = True
    req.token = token
    return {"ok": True}


@router.get("/pair/claim")
def claim_pairing_token(request_id: str) -> dict:
    _prune_expired()
    req = _pending.get(request_id)
    if req is None:
        raise HTTPException(404, "Esta solicitud de emparejamiento expiro o no existe.")
    if not req.approved:
        return {"pending": True}

    # Entrega unica: una vez reclamado, el request se borra aunque el token ya
    # quedo persistido en app_settings.json por approve_pairing_request.
    del _pending[request_id]
    return {"pending": False, "token": req.token}
