from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.main import app
from app.models import Job, JobStatus
from app.pipeline import ingest
from app.worker import runner


def test_upload_is_not_claimable_until_source_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    original = ingest.save_upload
    seen = []

    def save(job_id, name, stream):
        with Session(engine) as session:
            assert session.get(Job, job_id) is None
        assert not runner._claim_for_processing(job_id)
        seen.append(job_id)
        return original(job_id, name, stream)

    monkeypatch.setattr(ingest, "save_upload", save)
    with TestClient(app) as client:
        response = client.post("/api/jobs", files={"file": ("fixture.mp4", b"complete bytes")})
        assert response.status_code == 201
        job_id = response.json()["id"]
        assert seen == [job_id]
        assert ingest.find_source(job_id).read_bytes() == b"complete bytes"
        assert runner._claim_for_processing(job_id)
        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.done
            session.add(job)
            session.commit()


def test_failed_upload_does_not_publish_or_leave_partial_source(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    seen = []

    def save(job_id, name, stream):
        seen.append(job_id)
        (ingest.job_dir(job_id) / "source.mp4").write_bytes(b"partial")
        raise OSError("fixture upload interrupted")

    monkeypatch.setattr(ingest, "save_upload", save)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/api/jobs", files={"file": ("fixture.mp4", b"bytes")}).status_code == 500
        with Session(engine) as session:
            assert not session.exec(select(Job).where(Job.id == seen[0])).all()
        assert not runner._claim_for_processing(seen[0])
        assert not (settings.work_dir / seen[0]).exists()
