"""Tests de la API del compositor (validaciones de seleccion y ciclo de vida)."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.db import engine, init_db
from app.main import app
from app.models import Clip, ClipAsset, Compilation, CompilationStatus, Job, JobStatus, SourceType


@pytest.fixture
def library(tmp_path):
    """Crea un job con 3 clips renderizados: dos en 9x16 y uno en 16x9."""
    init_db()
    with Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/video.mp4",
            title="Podcast de prueba",
            status=JobStatus.done,
        )
        session.add(job)
        session.flush()

        asset_ids: dict[str, str] = {}
        for index, fmt in enumerate(["9x16", "9x16", "16x9"]):
            clip = Clip(job_id=job.id, index=index, start=index * 10, end=index * 10 + 8)
            session.add(clip)
            session.flush()
            # El endpoint filtra assets cuyo archivo ya no existe, asi que hay
            # que crear un archivo real en disco.
            video = tmp_path / f"clip{index}.mp4"
            video.write_bytes(b"fake-mp4")
            asset = ClipAsset(
                clip_id=clip.id,
                format=fmt,
                path=str(video),
                width=1080,
                height=1920,
                duration=8.0,
                size_bytes=video.stat().st_size,
            )
            session.add(asset)
            session.flush()
            asset_ids[f"{fmt}_{index}"] = asset.id
        session.commit()
        return asset_ids


def test_composable_clips_lists_assets_with_job_context(library):
    with TestClient(app) as c:
        res = c.get("/api/composable-clips")
        assert res.status_code == 200
        items = res.json()
        listed = {item["asset_id"] for item in items}
        assert set(library.values()) <= listed
        sample = next(item for item in items if item["asset_id"] == library["9x16_0"])
        assert sample["job_title"] == "Podcast de prueba"
        assert sample["format"] == "9x16"
        assert sample["video_url"].endswith("/video")


def test_composable_clips_hides_assets_whose_file_disappeared(library, tmp_path):
    (tmp_path / "clip0.mp4").unlink()
    with TestClient(app) as c:
        items = c.get("/api/composable-clips").json()
        listed = {item["asset_id"] for item in items}
        assert library["9x16_0"] not in listed
        assert library["9x16_1"] in listed


def test_create_requires_at_least_two_clips(library):
    with TestClient(app) as c:
        res = c.post("/api/compositions", json={"asset_ids": [library["9x16_0"]]})
        assert res.status_code == 400
        assert "al menos 2" in res.json()["detail"]


def test_create_rejects_mixed_formats(library):
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["16x9_2"]]},
        )
        assert res.status_code == 400
        assert "mismo formato" in res.json()["detail"]


def test_create_rejects_duplicate_clips(library):
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_0"]]},
        )
        assert res.status_code == 400
        assert "repetidos" in res.json()["detail"]


def test_create_rejects_unknown_asset(library):
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], "no-existe"]},
        )
        assert res.status_code == 404


def test_create_rejects_bgm_path_that_does_not_exist(library):
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={
                "asset_ids": [library["9x16_0"], library["9x16_1"]],
                "bgm_path": "C:/no/existe/musica.mp3",
            },
        )
        assert res.status_code == 400
        assert "audio de fondo" in res.json()["detail"]


@pytest.mark.parametrize("unc", ["\\\\servidor\\share\\musica.mp3", "//servidor/share/musica.mp3"])
def test_create_rejects_network_paths_as_bgm(library, unc):
    """Una UNC pasa Path.exists() y FFmpeg la abriria por SMB: en Windows eso
    filtra credenciales NTLM al host remoto. Snipra es local por diseño."""
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]], "bgm_path": unc},
        )
        assert res.status_code == 400
        assert "ruta de red" in res.json()["detail"]


def test_create_rejects_directory_as_bgm(library, tmp_path):
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={
                "asset_ids": [library["9x16_0"], library["9x16_1"]],
                "bgm_path": str(tmp_path),
            },
        )
        assert res.status_code == 400
        assert "no es un archivo" in res.json()["detail"]


def test_valid_bgm_path_is_stored_normalized(library, tmp_path):
    music = tmp_path / "musica.mp3"
    music.write_bytes(b"fake-audio")
    # ruta con un ".." de por medio: debe guardarse ya resuelta
    noisy = str(tmp_path / "sub" / ".." / "musica.mp3")
    (tmp_path / "sub").mkdir()
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={
                "asset_ids": [library["9x16_0"], library["9x16_1"]],
                "bgm_path": noisy,
            },
        )
        assert res.status_code == 200
        assert res.json()["bgm_path"] == str(music.resolve())


def test_create_queues_compilation_preserving_order_and_clamping_values(library):
    ordered = [library["9x16_1"], library["9x16_0"]]
    with TestClient(app) as c:
        res = c.post(
            "/api/compositions",
            json={
                "asset_ids": ordered,
                "title": "Mejores momentos",
                "transition": "fade",
                "transition_seconds": 99.0,
                "bgm_volume": 5.0,
                "bgm_auto_level": True,
                "bgm_duck_lu": 99.0,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "queued"
        assert body["asset_ids"] == ordered  # el orden elegido se respeta
        assert body["clip_count"] == 2
        assert body["format"] == "9x16"
        assert body["transition"] == "fade"
        assert body["transition_seconds"] == 3.0  # acotado al maximo
        assert body["bgm_volume"] == 1.0  # acotado al maximo
        assert body["bgm_auto_level"] is True
        assert body["bgm_duck_lu"] == 24.0
        assert body["audio_report"] == {}

        listed = c.get("/api/compositions").json()
        assert body["id"] in {item["id"] for item in listed}


def test_video_download_is_409_until_render_finishes(library):
    with TestClient(app) as c:
        created = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]]},
        ).json()
        res = c.get(f"/api/compositions/{created['id']}/video")
        assert res.status_code == 409


def test_delete_removes_compilation_but_keeps_source_clips(library):
    with TestClient(app) as c:
        created = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]]},
        ).json()
        assert c.delete(f"/api/compositions/{created['id']}").status_code == 200
        assert c.get(f"/api/compositions/{created['id']}").status_code == 404
        # los clips originales siguen disponibles en la biblioteca
        listed = {item["asset_id"] for item in c.get("/api/composable-clips").json()}
        assert library["9x16_0"] in listed
        assert library["9x16_1"] in listed


def test_storage_reports_compositions_separately_from_jobs(library, monkeypatch, tmp_path):
    """Sin esto las compilaciones crecen invisibles: el panel solo medía work/."""
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    comp_dir = tmp_path / "compositions" / "abc123"
    comp_dir.mkdir(parents=True)
    (comp_dir / "output.mp4").write_bytes(b"x" * 2048)

    with TestClient(app) as c:
        status = c.get("/api/storage").json()
        assert status["compositions_bytes"] == 2048
        assert "compositions_dir" in status
        # 'abc123' no existe como fila => cuenta como huerfana
        assert status["orphan_composition_count"] == 1


def test_cleanup_removes_orphan_composition_dirs_but_not_live_ones(library, monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    with TestClient(app) as c:
        live = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]]},
        ).json()

        comp_root = tmp_path / "compositions"
        live_dir = comp_root / live["id"]
        live_dir.mkdir(parents=True)
        (live_dir / "output.mp4").write_bytes(b"vivo")
        orphan_dir = comp_root / "ya-no-existe"
        orphan_dir.mkdir(parents=True)
        (orphan_dir / "output.mp4").write_bytes(b"huerfano")

        c.post("/api/storage/cleanup", json={"remove_orphans": True})

        assert not orphan_dir.exists(), "la huerfana debia borrarse"
        assert live_dir.exists(), "una compilacion viva NUNCA se borra en la limpieza"


def test_cannot_delete_or_retry_while_rendering(library):
    with TestClient(app) as c:
        created = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]]},
        ).json()
        with Session(engine) as session:
            compilation = session.get(Compilation, created["id"])
            compilation.status = CompilationStatus.rendering
            session.add(compilation)
            session.commit()

        assert c.delete(f"/api/compositions/{created['id']}").status_code == 409
        assert c.post(f"/api/compositions/{created['id']}/retry").status_code == 409


def test_retry_requeues_failed_compilation(library):
    with TestClient(app) as c:
        created = c.post(
            "/api/compositions",
            json={"asset_ids": [library["9x16_0"], library["9x16_1"]]},
        ).json()
        with Session(engine) as session:
            compilation = session.get(Compilation, created["id"])
            compilation.status = CompilationStatus.failed
            compilation.error = "fallo controlado"
            session.add(compilation)
            session.commit()

        res = c.post(f"/api/compositions/{created['id']}/retry")
        assert res.status_code == 200
        assert res.json()["status"] == "queued"
        assert res.json()["error"] is None
