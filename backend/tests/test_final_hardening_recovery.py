from types import SimpleNamespace

from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine, init_db
from app.models import Clip, ClipAsset, Job, JobStatus, SourceType
from app.pipeline import ingest, render
from app.providers.base import ClipCandidate
from app.worker import runner


def test_partial_render_recovery_replays_without_duplicate_clips(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    init_db()
    with Session(engine) as session:
        job = Job(source_type=SourceType.upload, source_name="fixture.mp4",
                  status=JobStatus.rendering, formats="16x9")
        session.add(job)
        session.commit()
        job_id = job.id
        directory = ingest.job_dir(job_id) / "clips" / "partial"
        directory.mkdir(parents=True)
        partial = directory / "partial.mp4"
        partial.write_bytes(b"partial")
        clip = Clip(job_id=job_id, index=0, start=0, end=10)
        session.add(clip)
        session.flush()
        old_clip_id = clip.id
        asset = ClipAsset(clip_id=clip.id, format="16x9", path=str(partial))
        session.add(asset)
        session.commit()
        old_asset_id = asset.id
    runner.recover_claimed_jobs()
    assert runner.recover_claimed_jobs() == 0
    assert not partial.exists()
    with Session(engine) as session:
        assert session.get(Clip, old_clip_id) is None
        assert session.get(ClipAsset, old_asset_id) is None
    def render_clip(*args, **kwargs):
        path = ingest.job_dir(job_id) / "clips" / "complete.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"complete")
        return [SimpleNamespace(format="16x9", path=str(path), thumb_path=None,
                                width=640, height=360, duration=10, size_bytes=8)]
    monkeypatch.setattr(render, "render_clip", render_clip)
    assert runner._claim_for_processing(job_id)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        runner._render_candidates(session, job, "source", [ClipCandidate(start=0, end=10)], {})
        job.status = JobStatus.done
        session.add(job)
        session.commit()
        clips = session.exec(select(Clip).where(Clip.job_id == job_id)).all()
        assert [c.index for c in clips] == [0]
        assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clips[0].id)).all()
        assert len(assets) == 1
        assert assets[0].path.endswith("complete.mp4")
    assert runner.recover_claimed_jobs() == 0
