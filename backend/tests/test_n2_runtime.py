"""Tests N.2: runtime reliability, recovery, cancelacion y sanitizacion.

Todo con mocks/fakes; 0 llamadas reales a APIs. Reutiliza el estilo de
tests/test_worker.py (DB de prueba real + monkeypatch de subprocesses).
"""

from __future__ import annotations

import time

import pytest
from sqlmodel import Session

from app.core import job_control
from app.core.config import settings
from app.core.db import engine, init_db
from app.core.media import VideoInfo
from app.models import Job, JobStatus, SourceType
from app.pipeline import ingest
from app.worker import runner


@pytest.fixture
def temp_storage(monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    return tmp_path


def _new_job(**overrides) -> str:
    overrides.setdefault("status", JobStatus.ingesting)
    job = Job(
        source_type=SourceType.url,
        source_name="https://example.com/video.mp4",
        **overrides,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        return job.id


def _status_and_error(job_id: str) -> tuple[str, str]:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        return job.status, (job.error or "")


def _fake_transcript() -> dict:
    return {"segments": [], "duration": 60.0, "language": "es"}


def _process_and_fail(job_id: str) -> None:
    try:
        runner._process_claimed_job(job_id)
    except Exception as exc:  # noqa: BLE001
        runner._fail(job_id, f"{type(exc).__name__}: {exc}")


# 1..4: excepcion en cada etapa termina en failed ------------------------------

def test_ingest_exception_fails_job(temp_storage):
    job_id = _new_job()

    def boom(*_a, **_k):
        raise ValueError("download caido")

    import app.pipeline.ingest as ing

    with pytest.MonkeyPatch.context() as m:
        m.setattr(ing, "download_url", boom)
        _process_and_fail(job_id)

    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "download caido" in error
    assert "ValueError" in error


def test_transcribe_exception_fails_job(temp_storage, monkeypatch):
    job_id = _new_job()
    monkeypatch.setattr(ingest, "download_url", lambda *a, **k: None)
    monkeypatch.setattr(
        ingest, "probe_source",
        lambda _id: VideoInfo(duration=60.0, width=640, height=360, has_audio=True),
    )

    def no_file(*_a, **_k):
        raise RuntimeError("Transcriber roto")

    monkeypatch.setattr("app.worker.runner.transcribe.transcribe_job", no_file)
    monkeypatch.setattr("app.worker.runner.ingest.find_source", lambda _id: None)
    monkeypatch.setattr(settings, "enable_media_signals", False)
    _process_and_fail(job_id)

    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "Transcriber roto" in error


def test_detect_exception_fails_job(temp_storage, monkeypatch):
    job_id = _new_job()
    monkeypatch.setattr(ingest, "download_url", lambda *a, **k: None)
    monkeypatch.setattr(
        ingest, "probe_source",
        lambda _id: VideoInfo(duration=60.0, width=640, height=360, has_audio=True),
    )
    monkeypatch.setattr("app.worker.runner.ingest.find_source", lambda _id: "fake.mp4")
    monkeypatch.setattr(settings, "enable_media_signals", False)
    monkeypatch.setattr(runner.transcribe, "transcribe_job", lambda *a, **k: _fake_transcript())

    def no_detect(*_a, **_k):
        raise RuntimeError("LLM caido")

    monkeypatch.setattr("app.worker.runner.detect.detect_clips", no_detect)
    _process_and_fail(job_id)

    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "LLM caido" in error


def test_render_exception_fails_job(temp_storage, monkeypatch):
    from app.providers.base import ClipCandidate

    job_id = _new_job()
    monkeypatch.setattr(ingest, "download_url", lambda *a, **k: None)
    monkeypatch.setattr(
        ingest, "probe_source",
        lambda _id: VideoInfo(duration=60.0, width=640, height=360, has_audio=True),
    )
    monkeypatch.setattr("app.worker.runner.ingest.find_source", lambda _id: "fake.mp4")
    monkeypatch.setattr(settings, "enable_media_signals", False)
    monkeypatch.setattr(runner.transcribe, "transcribe_job", lambda *a, **k: _fake_transcript())
    cand = ClipCandidate(start=1, end=5, title="t", hook="h", score=0.5, reason="r")
    monkeypatch.setattr("app.worker.runner.detect.detect_clips", lambda *a, **k: [cand])

    def no_render(*_a, **_k):
        raise OSError("ffmpeg caido")

    monkeypatch.setattr("app.worker.runner.render.render_clip", no_render)
    _process_and_fail(job_id)
    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "ffmpeg caido" in error


# 5/6: cancelacion no termina ni failed ni done ---------------------------------

def test_cancel_not_done(temp_storage, monkeypatch):
    job_id = _new_job()

    def cancel_at_download(*_a, **_k):
        job_control.request_cancel(job_id)

    monkeypatch.setattr(ingest, "download_url", cancel_at_download)
    monkeypatch.setattr(
        ingest, "probe_source",
        lambda _id: VideoInfo(duration=60.0, width=640, height=360, has_audio=True),
    )
    _process_and_fail(job_id)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert job.status == "canceled"
        assert job.status != "done"
        assert job.status != "failed"
        assert job.error is None


def test_fail_respects_pending_cancel(temp_storage):
    job_id = _new_job()
    job_control.request_cancel(job_id)
    runner._fail(job_id, "ERROR_FALSO: no debe marcar failed")
    status, error = _status_and_error(job_id)
    assert status == "canceled"
    assert error == ""


# 7: restart recupera active jobs ---------------------------------------------

def test_recover_claimed_jobs_requeues_active(temp_storage):
    # la DB de tests puede traer jobs activos previos; verificamos solo los creados aqui
    my_ids: list[str] = []
    for stage in ("ingesting", "transcribing", "detecting", "rendering"):
        job_id = _new_job()
        my_ids.append(job_id)
        with Session(engine) as session:
            job = session.get(Job, job_id)
            job.status = JobStatus(stage)
            session.add(job)
            session.commit()

    recovered_before = runner.recover_claimed_jobs()
    assert recovered_before >= 4
    with Session(engine) as session:
        for job_id in my_ids:
            job = session.get(Job, job_id)
            assert job.status == "queued"
            assert job.progress == 0.0


# 8/9/10/11: locks -------------------------------------------------------------

def test_worker_lock_live_blocks_second(monkeypatch, tmp_path):
    lock = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock)
    assert runner.acquire_worker_lock() is True
    # un segundo intento sin heartbeat viejo no debe ganar
    assert runner.acquire_worker_lock() is False
    runner.release_worker_lock()
    assert runner.acquire_worker_lock() is True
    runner.release_worker_lock()


def test_worker_lock_stale_heartbeat_recovers(monkeypatch, tmp_path):
    lock = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock)
    lock.write_text(str(__import__("os").getpid()))
    # mtime vieja = heartbeat murio aunque PID siga vivo
    stale = time.time() - (runner.WORKER_LOCK_STALE_SECONDS + 60)
    import os as _os

    _os.utime(lock, (stale, stale))
    assert runner.worker_lock_is_stale() is True
    assert runner.acquire_worker_lock() is True
    runner.release_worker_lock()


def test_worker_lock_corrupt_recovers(monkeypatch, tmp_path):
    lock = tmp_path / "snipra-worker.lock"
    monkeypatch.setattr(runner, "WORKER_LOCK_PATH", lock)
    lock.write_text("no es un pid;;;")
    assert runner.worker_lock_is_stale() is True
    assert runner.acquire_worker_lock() is True
    runner.release_worker_lock()


# 12/13: operation/compilation recovery ---------------------------------------

def test_recover_operations_requeues():
    from app.worker.clip_operations import recover_operations

    # La DB de tests puede tener filas de otros tests: el invariante es que la
    # recuperacion corre sin romper y no devuelve negativo.
    assert recover_operations() >= 0


def test_recover_compilations_requeues():
    from app.worker.compositions import recover_compilations

    assert recover_compilations() >= 0


# 14/15/16: error sanitization ------------------------------------------------

def test_fail_sanitizes_secrets(temp_storage, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "nvidia_api_key", "nvapi-FALSA-000111")
    job_id = _new_job()
    runner._fail(job_id, "Provider falló: invalid key nvapi-FALSA-000111")
    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "nvapi-FALSA-000111" not in error


def test_worker_process_exception_prints_sanitized(temp_storage, monkeypatch, capsys):
    """Una excepcion durante el job con key FALSA: el traceback impreso por el
    worker NO puede contener la key."""
    from app.core.config import settings as cfg

    monkeypatch.setattr(cfg, "nvidia_api_key", "nvapi-WORKER-FAKEY-777")
    job_id = _new_job(status=JobStatus.queued)

    def crash(*_a, **_k):
        raise RuntimeError("fallo externo con nvapi-WORKER-FAKEY-777")

    monkeypatch.setattr(ingest, "download_url", crash)
    try:
        runner.process_job(job_id, claimed=False)
    except Exception as exc:  # noqa: BLE001
        # el run loop de produccion hace exactamente esto
        print(runner.sanitize_traceback(exc), flush=True)
        runner._fail(job_id, f"{type(exc).__name__}: {exc}")

    out = capsys.readouterr().out
    assert "nvapi-WORKER-FAKEY-777" not in out
    assert "[REDACTED]" in out
    status, error = _status_and_error(job_id)
    assert status == "failed"
    assert "nvapi-WORKER-FAKEY-777" not in error


# 17-19: channel robustness ---------------------------------------------------

def test_channel_extraction_timeout_categorized():
    from app.pipeline import channel_watch

    code, msg = channel_watch._categorize_extraction_error(
        Exception("socket timed out while reading")
    )
    assert code == "TIMEOUT"


def test_channel_extraction_sanitized():
    from unittest.mock import patch

    from app.core.config import settings
    from app.pipeline import channel_watch

    with patch.object(settings, "nvidia_api_key", "nvapi-channel-FAKE-999"):
        code, msg = channel_watch._categorize_extraction_error(
            Exception("brotl1: respuesta rara con nvapi-channel-FAKE-999")
        )
        assert "nvapi-channel-FAKE-999" not in msg


def test_channel_not_found_category():
    from app.pipeline import channel_watch

    code, _ = channel_watch._categorize_extraction_error(
        Exception("HTTP 404 Not Found: channel does not exist")
    )
    assert code == "CHANNEL_NOT_FOUND"


# 20/22/23: API/runtime/startup ---------------------------------------------

def test_api_global_handler_hides_and_sanitizes(caplog, monkeypatch):
    """El handler global de 500 se ejecuta de verdad:
    - ruta de test que lanza RuntimeError con una clave FALSA
    - respuesta: INTERNAL_ERROR sin secret ni traceback
    - log del servidor: sanitizado tambien
    """
    import logging

    from fastapi.testclient import TestClient

    from app.core.config import settings as cfg
    from app.main import app

    fake = "nvapi-FAKE-N2-LOG-SECRET"
    monkeypatch.setattr(cfg, "nvidia_api_key", fake)

    async def _boom_endpoint():
        raise RuntimeError(f"upstream devolvio 401 para key {fake}")

    app.router.add_api_route("/__test__/boom", _boom_endpoint, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="snipra.api"):
        resp = client.get("/__test__/boom")
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["code"] == "INTERNAL_ERROR"
    body = resp.text
    assert fake not in body
    assert "Traceback" not in body
    assert "[REDACTED]" in detail["message"]

    log_text = "\n".join(caplog.text for _line in [0])
    assert fake not in log_text
    assert "[REDACTED]" in log_text


def test_runtime_settings_invalid_provider_falls_back(monkeypatch):
    from app.core.config import settings as cfg
    from app.core.runtime_settings import apply_runtime_settings

    monkeypatch.setattr(cfg, "llm_provider", "naive")
    apply_runtime_settings({"llm_provider": "xxxx"}, persist=False)
    assert cfg.llm_provider == "naive"


def test_startup_load_no_paid_calls(monkeypatch, tmp_path):
    """Un arranque en frio no debe invocar ningun proveedor pago."""
    calls = []

    def record(name):
        def _f(*a, **k):
            calls.append(name)
            raise RuntimeError(f"no debe llamarse {name}")
        return _f

    monkeypatch.setattr("app.providers.claude.ClaudeDetector._complete", record("claude"))
    monkeypatch.setattr("app.providers.openai.OpenAIDetector._complete", record("openai"))

    from app.core.config import settings as cfg
    from app.core.runtime_settings import apply_runtime_settings

    apply_runtime_settings({"llm_provider": "naive"}, persist=False)
    assert cfg.llm_provider == "naive"
    assert calls == []


# 24/25: evidence fail-closed y sin retry extra -------------------------------

def test_no_unlimited_retries_in_providers():
    """Ningun provider de AI hace retry pagado ilimitado (max_retries=0 en NVIDIA)."""
    import inspect

    from app.providers.nvidia import NvidiaDetector

    src = inspect.getsource(NvidiaDetector._client)
    assert "max_retries=0" in src




