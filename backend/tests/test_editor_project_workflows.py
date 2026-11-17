
import pytest
from fastapi import HTTPException
from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import service


def test_asset_removal_preserves_source_and_rejects_timeline_references(tmp_path):
    init_db()
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    with Session(engine) as session:
        project = service.create_project(session, "Media")
        asset = service.add_asset(session, project, source)
        service.remove_asset(session, project.id, asset.id)
        assert source.exists()
        assert service.project_assets(session, project.id) == []

        asset = service.add_asset(session, project, source, duration_us=1_000_000)
        track = service.current_version(session, project).timeline["tracks"][0]
        service.apply_operation(
            session,
            project.id,
            {
                "type": "insert",
                "parameters": {
                    "asset_id": asset.id,
                    "track_id": track["id"],
                    "source_in_us": 0,
                    "source_out_us": 1_000_000,
                    "timeline_start_us": 0,
                },
                "base_version": project.current_version,
                "idempotency_key": "insert",
            },
        )
        with pytest.raises(
            HTTPException, match="Actualmente está siendo utilizado por 1 clip"
        ) as exc:
            service.remove_asset(session, project.id, asset.id)
        assert exc.value.status_code == 409
        assert source.exists()


def test_projects_rename_and_delete_only_project_records(tmp_path):
    init_db()
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    with Session(engine) as session:
        project = service.create_project(session, "Before")
        service.add_asset(session, project, source)
        renamed = service.rename_project(session, project.id, "After")
        assert renamed.id == project.id and renamed.name == "After"
        with pytest.raises(HTTPException, match="no puede estar vacío"):
            service.rename_project(session, project.id, "  ")
        service.delete_project(session, project.id)
        assert source.exists()
        with pytest.raises(HTTPException):
            service.project_or_404(session, project.id)
