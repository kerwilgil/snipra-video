"""Tests del emparejamiento de un clic para el token de la extension."""

from fastapi.testclient import TestClient

from app.api import extension_pairing
from app.core.config import settings
from app.main import app


def _reset_pending():
    extension_pairing._pending.clear()


def test_request_creates_pending_entry_without_granting_token():
    _reset_pending()
    with TestClient(app) as c:
        r = c.post("/api/extension/pair/request")
        assert r.status_code == 200
        request_id = r.json()["request_id"]

        claim = c.get(f"/api/extension/pair/claim?request_id={request_id}")
        assert claim.status_code == 200
        assert claim.json() == {"pending": True}


def test_approve_then_claim_delivers_token_once(monkeypatch):
    _reset_pending()
    monkeypatch.setattr(settings, "extension_token", "")
    with TestClient(app) as c:
        request_id = c.post("/api/extension/pair/request").json()["request_id"]

        approve = c.post("/api/extension/pair/approve", json={"request_id": request_id})
        assert approve.status_code == 200

        first_claim = c.get(f"/api/extension/pair/claim?request_id={request_id}")
        assert first_claim.status_code == 200
        body = first_claim.json()
        assert body["pending"] is False
        assert body["token"]
        assert body["token"] == settings.extension_token

        # Un solo uso: el segundo intento de reclamar el mismo request_id falla.
        second_claim = c.get(f"/api/extension/pair/claim?request_id={request_id}")
        assert second_claim.status_code == 404


def test_claim_unknown_request_id_returns_404():
    _reset_pending()
    with TestClient(app) as c:
        r = c.get("/api/extension/pair/claim?request_id=no-existe")
        assert r.status_code == 404


def test_approve_unknown_request_id_returns_404():
    _reset_pending()
    with TestClient(app) as c:
        r = c.post("/api/extension/pair/approve", json={"request_id": "no-existe"})
        assert r.status_code == 404


def test_extension_origin_can_call_request_without_token():
    """El bootstrap /pair/request es el UNICO endpoint mutante exento del token
    compartido (ver excepcion explicita en main.py) -- sin el, la extension
    nunca podria arrancar el emparejamiento la primera vez."""
    _reset_pending()
    with TestClient(app) as c:
        r = c.post(
            "/api/extension/pair/request",
            headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
        )
        assert r.status_code == 200


def test_extension_origin_still_blocked_from_other_mutations_without_token(monkeypatch):
    """La excepcion es puntual: cualquier OTRA mutacion desde origin de extension
    sigue exigiendo el token compartido como antes."""
    monkeypatch.setattr(settings, "extension_token", "")
    with TestClient(app) as c:
        r = c.put(
            "/api/settings",
            json={"default_num_clips": 3},
            headers={"origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop"},
        )
        assert r.status_code == 403
