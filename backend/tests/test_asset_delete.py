"""Borrado de un formato renderizado desde la biblioteca del compositor."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core import config
from app.core.db import engine, init_db
from app.main import app
from app.models import (
    Clip,
    ClipAsset,
    Compilation,
    CompilationStatus,
    Job,
    JobStatus,
    SourceType,
)


@pytest.fixture
def clip_with_two_formats(tmp_path, monkeypatch):
    """Un clip renderizado en 9x16 y 16x9, con la estructura real en disco:
    work/<job>/clips/<slug>/ con el corte base compartido por ambos formatos."""
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    with Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/v.mp4",
            status=JobStatus.done,
        )
        session.add(job)
        session.flush()
        clip = Clip(job_id=job.id, index=0, start=0, end=10, title="Momento")
        session.add(clip)
        session.flush()

        clip_dir = config.settings.work_dir / job.id / "clips" / "00-momento"
        clip_dir.mkdir(parents=True)
        (clip_dir / "clip_base.mp4").write_bytes(b"base compartida")

        ids = {}
        for fmt in ("9x16", "16x9"):
            video = clip_dir / f"{fmt}.mp4"
            thumb = clip_dir / f"{fmt}.jpg"
            video.write_bytes(b"video")
            thumb.write_bytes(b"thumb")
            asset = ClipAsset(clip_id=clip.id, format=fmt, path=str(video), thumb_path=str(thumb))
            session.add(asset)
            session.flush()
            ids[fmt] = asset.id
        session.commit()
        return {"ids": ids, "clip_id": clip.id, "clip_dir": clip_dir}


def test_deleting_one_format_keeps_the_other_and_the_clip(clip_with_two_formats):
    ids = clip_with_two_formats["ids"]
    clip_dir = clip_with_two_formats["clip_dir"]
    with TestClient(app) as c:
        res = c.delete(f"/api/assets/{ids['9x16']}")
        assert res.status_code == 200
        assert res.json() == {"ok": True, "clip_removed": False, "format": "9x16"}

    assert not (clip_dir / "9x16.mp4").exists()
    assert not (clip_dir / "9x16.jpg").exists()
    # el otro formato y el corte base compartido siguen intactos
    assert (clip_dir / "16x9.mp4").exists()
    assert (clip_dir / "clip_base.mp4").exists()
    with Session(engine) as session:
        assert session.get(ClipAsset, ids["9x16"]) is None
        assert session.get(ClipAsset, ids["16x9"]) is not None
        assert session.get(Clip, clip_with_two_formats["clip_id"]) is not None


def test_deleting_last_format_removes_clip_and_its_directory(clip_with_two_formats):
    ids = clip_with_two_formats["ids"]
    clip_dir = clip_with_two_formats["clip_dir"]
    with TestClient(app) as c:
        c.delete(f"/api/assets/{ids['9x16']}")
        res = c.delete(f"/api/assets/{ids['16x9']}")
        assert res.status_code == 200
        assert res.json()["clip_removed"] is True

    # se va la carpeta entera, incluido el corte base y los frames de Timeline QA
    assert not clip_dir.exists()
    with Session(engine) as session:
        assert session.get(Clip, clip_with_two_formats["clip_id"]) is None
        remaining = session.exec(
            select(ClipAsset).where(ClipAsset.clip_id == clip_with_two_formats["clip_id"])
        ).all()
        assert remaining == []


def test_delete_is_blocked_while_a_compilation_using_it_is_pending(clip_with_two_formats):
    ids = clip_with_two_formats["ids"]
    with Session(engine) as session:
        session.add(
            Compilation(
                title="Recopilado",
                asset_ids=[ids["9x16"], ids["16x9"]],
                status=CompilationStatus.queued,
            )
        )
        session.commit()

    with TestClient(app) as c:
        res = c.delete(f"/api/assets/{ids['9x16']}")
        assert res.status_code == 409
        assert "Recopilado" in res.json()["detail"]

    # nada se toco
    with Session(engine) as session:
        assert session.get(ClipAsset, ids["9x16"]) is not None


def test_delete_is_allowed_when_the_compilation_already_finished(clip_with_two_formats):
    """Una compilacion terminada ya tiene su mp4 propio: no depende del asset."""
    ids = clip_with_two_formats["ids"]
    with Session(engine) as session:
        session.add(
            Compilation(
                title="Vieja",
                asset_ids=[ids["9x16"]],
                status=CompilationStatus.done,
            )
        )
        session.commit()

    with TestClient(app) as c:
        assert c.delete(f"/api/assets/{ids['9x16']}").status_code == 200


def test_delete_unknown_asset_is_404(clip_with_two_formats):
    with TestClient(app) as c:
        assert c.delete("/api/assets/no-existe").status_code == 404


def test_deleted_asset_disappears_from_the_composer_library(clip_with_two_formats):
    ids = clip_with_two_formats["ids"]
    with TestClient(app) as c:
        before = {i["asset_id"] for i in c.get("/api/composable-clips").json()}
        assert ids["9x16"] in before
        c.delete(f"/api/assets/{ids['9x16']}")
        after = {i["asset_id"] for i in c.get("/api/composable-clips").json()}
        assert ids["9x16"] not in after
        assert ids["16x9"] in after


def test_rmtree_guard_ignores_paths_outside_work_dir(tmp_path, monkeypatch):
    """Si una ruta de la BD quedara malformada, el borrado no debe salirse
    del storage."""
    from app.api import clips as clips_api

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    outsider = tmp_path.parent / "no-tocar"
    outsider.mkdir(exist_ok=True)
    (outsider / "importante.txt").write_text("no borrar", encoding="utf-8")

    clips_api._remove_dir_inside_work(outsider)
    assert outsider.exists()
    assert (outsider / "importante.txt").exists()

    # y tampoco el propio work_dir completo
    config.settings.work_dir.mkdir(parents=True, exist_ok=True)
    clips_api._remove_dir_inside_work(config.settings.work_dir)
    assert config.settings.work_dir.exists()
