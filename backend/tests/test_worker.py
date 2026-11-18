import os
import subprocess
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, select

from app.core import job_control, media
from app.core.db import engine, init_db
from app.core.media import VideoInfo
from app.core.storage_maintenance import cleanup_expired_job_files
from app.models import (
    Clip,
    ClipOperation,
    ClipOperationStatus,
    Job,
    JobLog,
    JobStatus,
    SourceType,
)
from app.pipeline import ingest
from app.worker import clip_operations, runner


def test_stage_timer_resets_only_when_main_status_changes():
    class FakeSession:
        def refresh(self, job):
            return None

        def add(self, job):
            return None

        def commit(self):
            return None

    started = datetime(2026, 1, 1, tzinfo=UTC)
    job = Job(
        id="stage-timer-test",
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        status=JobStatus.detecting,
        stage_started_at=started,
    )
    session = FakeSession()

    runner._update(
        session,
        job,
        status=JobStatus.detecting,
        stage="Visión IA: candidato 2/6",
        log_stage=False,
    )
    assert job.stage_started_at == started

    runner._update(
        session,
        job,
        status=JobStatus.rendering,
        stage="Generando clips",
        log_stage=False,
    )
    assert job.stage_started_at > started


def test_cancel_marker_round_trip(monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)

    job_control.request_cancel("job-cancel")
    assert job_control.is_cancel_requested("job-cancel")
    job_control.clear_cancel_request("job-cancel")
    assert not job_control.is_cancel_requested("job-cancel")


def test_worker_lock_prevents_duplicate_instance(monkeypatch, tmp_path):
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)

    assert runner.acquire_worker_lock()
    assert not runner.acquire_worker_lock()
    runner.release_worker_lock()
    assert runner.acquire_worker_lock()
    runner.release_worker_lock()


def test_lock_with_recycled_pid_is_treated_as_stale(monkeypatch, tmp_path):
    """Un PID vivo pero sin latido reciente no debe bloquear a un worker nuevo.

    Es el caso real: se mata el worker a la fuerza, el SO le da ese PID a otro
    proceso cualquiera y el lock quedaba envenenado para siempre.
    """
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)
    # PID vivo (el del propio test) pero con el latido vencido.
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(runner, "_heartbeat_age_seconds", lambda: 9999.0)

    assert runner.worker_lock_is_stale()
    assert runner.acquire_worker_lock()
    runner.release_worker_lock()


def test_lock_with_live_pid_and_fresh_heartbeat_blocks(monkeypatch, tmp_path):
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    monkeypatch.setattr(runner, "_heartbeat_age_seconds", lambda: 1.0)

    assert not runner.worker_lock_is_stale()
    assert not runner.acquire_worker_lock()


def test_lock_of_dead_pid_is_stale_even_with_fresh_mtime(monkeypatch, tmp_path):
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)
    lock_path.write_text("999999999", encoding="utf-8")
    monkeypatch.setattr(runner, "_heartbeat_age_seconds", lambda: 0.0)
    monkeypatch.setattr(runner, "_pid_is_alive", lambda pid: False)

    assert runner.worker_lock_is_stale()


def test_corrupt_lock_file_is_stale(monkeypatch, tmp_path):
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)
    lock_path.write_text("no-es-un-pid", encoding="utf-8")

    assert runner.worker_lock_is_stale()
    assert runner.acquire_worker_lock()
    runner.release_worker_lock()


def test_touch_refreshes_heartbeat(monkeypatch, tmp_path):
    lock_path = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock_path)
    assert runner.acquire_worker_lock()
    # Envejecemos el lock a mano y comprobamos que el latido lo rejuvenece.
    old = time.time() - 5000
    os.utime(lock_path, (old, old))
    assert runner._heartbeat_age_seconds() > 4000
    runner.touch_worker_lock()
    assert runner._heartbeat_age_seconds() < 60
    runner.release_worker_lock()


def test_media_run_stops_process_when_cancel_is_requested(monkeypatch):
    class FakeProcess:
        returncode = None

        def __init__(self):
            self.terminated = False

        def communicate(self, timeout=None):
            if not self.terminated:
                raise subprocess.TimeoutExpired(["ffmpeg"], timeout)
            self.returncode = -1
            return "", ""

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(media.subprocess, "Popen", lambda *args, **kwargs: process)
    checks = iter([False, True])
    token = media.set_cancel_check(lambda: next(checks))
    try:
        with pytest.raises(media.FFmpegCanceledError):
            media.run(["ffmpeg", "-version"])
    finally:
        media.reset_cancel_check(token)

    assert process.terminated


def test_recovery_preserves_pending_cancellation(monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        status=JobStatus.rendering,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    job_control.request_cancel(job_id)
    runner.recover_claimed_jobs()

    with Session(engine) as session:
        recovered = session.get(Job, job_id)
        assert recovered is not None
        assert recovered.status == JobStatus.canceled
        assert recovered.stage == "Cancelado"


def test_download_mode_warns_when_source_falls_short_of_requested_quality(monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        output_mode="download",
        download_quality="1080",
        status=JobStatus.ingesting,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    monkeypatch.setattr(ingest, "download_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingest,
        "probe_source",
        lambda _job_id: VideoInfo(duration=60.0, width=640, height=360, has_audio=True),
    )

    runner._process_claimed_job(job_id)

    with Session(engine) as session:
        logs = session.exec(select(JobLog).where(JobLog.job_id == job_id)).all()
        warning = next((log for log in logs if log.level == "warning"), None)
        assert warning is not None
        assert warning.message.startswith("Calidad pedida: hasta 1080p")
        assert "360p" in warning.message

        finished = session.get(Job, job_id)
        assert finished is not None
        assert finished.status == JobStatus.done
        # la resolucion real sondeada queda persistida en el job para mostrarla en la UI
        assert finished.source_width == 640
        assert finished.source_height == 360


def test_no_quality_warning_when_source_matches_request(monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        output_mode="download",
        download_quality="1080",
        status=JobStatus.ingesting,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    monkeypatch.setattr(ingest, "download_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingest,
        "probe_source",
        lambda _job_id: VideoInfo(duration=60.0, width=1920, height=1080, has_audio=True),
    )

    runner._process_claimed_job(job_id)

    with Session(engine) as session:
        logs = session.exec(select(JobLog).where(JobLog.job_id == job_id)).all()
        assert not any(log.level == "warning" for log in logs)


def test_audio_mode_downloads_audio_and_skips_quality_warning(monkeypatch, tmp_path):
    """output_mode=audio usa download_audio_url/probe_audio_source (no video) y no
    dispara el aviso de calidad (ese solo aplica a download_quality/resolucion de video)."""
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        output_mode="audio",
        audio_format="wav",
        status=JobStatus.ingesting,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        job_id = job.id

    audio_calls: list[str] = []
    monkeypatch.setattr(
        ingest,
        "download_audio_url",
        lambda job_id, url, **kwargs: audio_calls.append(kwargs.get("audio_format")),
    )
    monkeypatch.setattr(ingest, "download_url", lambda *a, **k: pytest.fail("no debe bajar video"))
    monkeypatch.setattr(
        ingest,
        "probe_audio_source",
        lambda _job_id: VideoInfo(duration=180.0, width=0, height=0, has_audio=True),
    )
    monkeypatch.setattr(
        ingest, "probe_source", lambda _job_id: pytest.fail("no debe sondear video")
    )

    runner._process_claimed_job(job_id)

    assert audio_calls == ["wav"]
    with Session(engine) as session:
        finished = session.get(Job, job_id)
        assert finished is not None
        assert finished.status == JobStatus.done
        assert finished.duration == 180.0
        assert finished.source_width == 0 and finished.source_height == 0

        logs = session.exec(select(JobLog).where(JobLog.job_id == job_id)).all()
        assert not any(log.level == "warning" for log in logs)
        assert any("Audio extraido" in log.message for log in logs)


def test_clip_operation_processes_every_clip(monkeypatch):
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        status=JobStatus.done,
    )
    with Session(engine) as session:
        session.add(job)
        session.flush()
        clips = [
            Clip(job_id=job.id, index=index, start=index * 10, end=index * 10 + 5)
            for index in range(2)
        ]
        session.add_all(clips)
        session.flush()
        operation = ClipOperation(
            job_id=job.id,
            clip_id=clips[0].id,
            kind="style",
            payload={"style_config": {"style": "clean"}},
            status=ClipOperationStatus.running,
        )
        session.add(operation)
        session.commit()
        operation_id = operation.id

    processed: list[str] = []

    def fake_rerender(session, job, clip, style_config=None, subtitles=None):
        processed.append(clip.id)
        assert style_config == {"style": "clean"}
        return {"id": clip.id}

    monkeypatch.setattr(clip_operations, "_rerender_from_artifacts", fake_rerender)
    clip_operations.process_operation(operation_id)

    with Session(engine) as session:
        completed = session.get(ClipOperation, operation_id)
        assert completed is not None
        assert completed.status == ClipOperationStatus.done
        assert completed.progress == 1.0
    assert len(processed) == 2


def test_clip_operation_recovery_and_failure():
    init_db()
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        status=JobStatus.done,
    )
    with Session(engine) as session:
        session.add(job)
        session.flush()
        clip = Clip(job_id=job.id, index=0, start=0, end=5)
        session.add(clip)
        session.flush()
        operation = ClipOperation(
            job_id=job.id,
            clip_id=clip.id,
            kind="style",
            status=ClipOperationStatus.running,
        )
        session.add(operation)
        session.commit()
        operation_id = operation.id

    assert clip_operations.recover_operations() >= 1
    with Session(engine) as session:
        recovered = session.get(ClipOperation, operation_id)
        assert recovered is not None
        assert recovered.status == ClipOperationStatus.queued

    clip_operations.fail_operation(operation_id, "fallo controlado")
    with Session(engine) as session:
        failed = session.get(ClipOperation, operation_id)
        assert failed is not None
        assert failed.status == ClipOperationStatus.failed
        assert failed.error == "fallo controlado"


def test_storage_retention_removes_only_unprotected_terminal_jobs(monkeypatch, tmp_path):
    from app.core import config

    init_db()
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    old = datetime.now(UTC) - timedelta(days=40)
    with Session(engine) as session:
        removable = Job(
            source_type=SourceType.url,
            source_name="https://example.com/removable.mp4",
            status=JobStatus.done,
            updated_at=old,
        )
        active = Job(
            source_type=SourceType.url,
            source_name="https://example.com/active.mp4",
            status=JobStatus.rendering,
            updated_at=old,
        )
        protected = Job(
            source_type=SourceType.url,
            source_name="https://example.com/protected.mp4",
            status=JobStatus.done,
            updated_at=old,
        )
        session.add_all([removable, active, protected])
        session.flush()
        protected_clip = Clip(job_id=protected.id, index=0, start=0, end=5)
        session.add(protected_clip)
        session.flush()
        session.add(
            ClipOperation(
                job_id=protected.id,
                clip_id=protected_clip.id,
                kind="style",
                status=ClipOperationStatus.queued,
            )
        )
        session.commit()

        for job in (removable, active, protected):
            path = config.settings.work_dir / job.id
            path.mkdir(parents=True)
            (path / "artifact.bin").write_bytes(b"1234")

        result = cleanup_expired_job_files(session, 30)

        assert result == {"removed_paths": 1, "removed_bytes": 4}
        assert not (config.settings.work_dir / removable.id).exists()
        assert (config.settings.work_dir / active.id).exists()
        assert (config.settings.work_dir / protected.id).exists()


def test_retention_is_disabled_by_default_and_never_deletes_existing_jobs(monkeypatch, tmp_path):
    """Regresion: la retencion borro carpetas antiguas al iniciar sin que el usuario la activara."""
    from app.core import config

    default_days = config.Settings.model_fields["storage_retention_days"].default
    assert default_days == 0
    init_db()
    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    old = datetime.now(UTC) - timedelta(days=400)
    with Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/old.mp4",
            status=JobStatus.done,
            updated_at=old,
        )
        session.add(job)
        session.commit()
        path = config.settings.work_dir / job.id
        path.mkdir(parents=True)
        (path / "artifact.bin").write_bytes(b"1234")

        result = cleanup_expired_job_files(session, default_days)

        assert result == {"removed_paths": 0, "removed_bytes": 0}
        assert (path / "artifact.bin").exists()


def test_worker_loop_does_not_run_maintenance_at_startup(monkeypatch):
    from unittest.mock import Mock

    class StopLoop(Exception):
        pass

    for name in (
        "init_db", "load_runtime_settings", "recover_claimed_jobs", "recover_operations",
        "recover_compilations", "recover_asset_tasks",
    ):
        monkeypatch.setattr(runner, name, Mock(return_value=0))
    # Start beyond one maintenance interval: an initial timestamp of zero would clean immediately.
    start = runner.MAINTENANCE_INTERVAL_SECONDS + 100.0
    monkeypatch.setattr(runner.time, "monotonic", Mock(side_effect=[
        start, start, start + runner.MAINTENANCE_INTERVAL_SECONDS,
    ]))
    cleanup = Mock(return_value={"removed_paths": 0, "removed_bytes": 0})
    monkeypatch.setattr(runner, "cleanup_expired_job_files", cleanup)
    monkeypatch.setattr(runner.channel_watch, "poll_due_channels", Mock(return_value=0))

    def work_round(index, types):
        if cleanup.called:
            raise StopLoop
        return False, None, None, index

    monkeypatch.setattr(runner, "_try_one_work_round", work_round)
    monkeypatch.setattr(runner, "_sleep_if_idle", lambda done: cleanup.assert_not_called())
    with pytest.raises(StopLoop):
        runner._run_loop()
    cleanup.assert_called_once()
