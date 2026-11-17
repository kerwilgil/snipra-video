"""A damaged draft must not make the editor's entire project list unavailable."""

from fastapi import HTTPException
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.api.editor import list_projects
from app.editor import service
from app.editor.models import EditorProject


def test_list_keeps_valid_projects_when_one_version_is_missing() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        healthy = service.create_project(session, "Healthy")
        damaged = EditorProject(name="Damaged")
        session.add(damaged)
        session.commit()

        items = list_projects(session)
        by_id = {item["id"]: item for item in items}
        assert len(items) == 2
        assert by_id[healthy.id]["timeline"] is not None
        assert by_id[damaged.id]["timeline"] is None
        assert by_id[damaged.id]["unavailable_reason"]

        try:
            service.serialize_project(session, damaged)
        except HTTPException as exc:
            assert exc.status_code == 409
        else:
            raise AssertionError("The damaged project must stay unavailable when opened")
