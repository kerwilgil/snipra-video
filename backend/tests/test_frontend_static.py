"""Frontend compilado servido desde el mismo origen que la API (app macOS empaquetada)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.frontend_static import mount_frontend


@pytest.fixture()
def dist_dir(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Snipra</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
    return dist


@pytest.fixture()
def client(dist_dir):
    app = FastAPI()

    @app.get("/api/ping")
    def ping() -> dict:
        return {"ok": True}

    mount_frontend(app, dist_dir)
    return TestClient(app)


def test_root_serves_index_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Snipra</title>" in response.text


def test_static_assets_are_served(client):
    response = client.get("/assets/app.js")
    assert response.status_code == 200
    assert "console.log" in response.text


def test_client_route_falls_back_to_index(client):
    response = client.get("/jobs/123")
    assert response.status_code == 200
    assert "<title>Snipra</title>" in response.text


def test_api_routes_keep_priority(client):
    assert client.get("/api/ping").json() == {"ok": True}


def test_unknown_api_route_is_not_replaced_by_index(client):
    response = client.get("/api/no-existe")
    assert response.status_code == 404
    assert "<title>Snipra</title>" not in response.text


@pytest.mark.parametrize("path", ["/api", "/api%5Cno-existe"])
def test_api_root_and_windows_separator_do_not_fall_back_to_html(client, path):
    response = client.get(path)
    assert response.status_code == 404
    assert "<title>Snipra</title>" not in response.text


def test_missing_index_is_rejected(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="index.html"):
        mount_frontend(FastAPI(), empty)
