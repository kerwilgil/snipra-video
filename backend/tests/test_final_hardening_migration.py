import pytest
from sqlalchemy import inspect, text
from sqlmodel import create_engine

from app.core import db


@pytest.mark.parametrize("already_migrated", [False, True])
def test_legacy_asset_task_first_and_second_boot(monkeypatch, tmp_path, already_migrated):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    monkeypatch.setattr(db, "engine", engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE job (id VARCHAR PRIMARY KEY, updated_at DATETIME)"))
        conn.execute(text("CREATE TABLE assettask (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("INSERT INTO assettask VALUES ('preserved')"))
        if already_migrated:
            conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_id VARCHAR"))
            conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_state VARCHAR"))
            conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_metadata JSON"))
            conn.execute(text("UPDATE assettask SET external_task_id = 'remote-id'"))
    db.init_db()
    db.init_db()
    columns = {c["name"] for c in inspect(engine).get_columns("assettask")}
    assert {"external_task_id", "external_task_state", "external_task_metadata",
            "next_poll_at", "submit_count", "cancel_requested"} <= columns
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, external_task_id FROM assettask")).one()
        assert row == ("preserved", "remote-id" if already_migrated else None)
    engine.dispose()
