"""Worker: consume jobs en cola (SQLite) y ejecuta el pipeline completo.

Ejecutar en proceso aparte:  uv run python -m app.worker.runner
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import text
from sqlmodel import Session, select

from app.core import job_control, media
from app.core.config import settings
from app.core.db import engine, init_db
from app.core.editorial_options import (
    normalize_editorial_options,
    whisper_model_for_quality,
)
from app.core.job_state import ACTIVE_JOB_STATUSES, add_job_log
from app.core.processing_modes import is_music_mode
from app.core.runtime_settings import load_runtime_settings
from app.core.sanitize import sanitize_error_text, sanitize_traceback
from app.core.storage_maintenance import cleanup_expired_job_files
from app.decision.models import DecisionResult, DecisionTask
from app.decision.service import evaluate_candidates
from app.models import Clip, ClipAsset, Job, JobStatus, SourceType
from app.pipeline import (
    channel_watch,
    detect,
    editorial,
    ingest,
    render,
    signals,
    sponsorblock,
    transcribe,
)
from app.worker.asset_tasks import (
    recover_asset_tasks,
)
from app.worker.clip_operations import (
    recover_operations,
)
from app.worker.compositions import (
    recover_compilations,
)

POLL_SECONDS = 2.0
MAINTENANCE_INTERVAL_SECONDS = 6 * 60 * 60
CHANNEL_POLL_INTERVAL_SECONDS = 5 * 60
CLAIMED_BUT_NOT_STARTED_PROGRESS = 0.02
WORKER_LOCK_PATH = settings.storage_dir / "snipra-worker.lock"
# El worker vivo refresca la mtime del lock desde un hilo aparte. Un lock cuya
# mtime quedo vieja significa "el dueño ya no esta trabajando" aunque su PID
# siga existiendo, que es justo lo que pasa cuando el SO recicla el PID de un
# worker matado a la fuerza: sin esto el lock quedaba envenenado para siempre y
# ningun worker nuevo podia arrancar (los jobs se quedaban en cola en silencio).
WORKER_HEARTBEAT_SECONDS = 15.0
WORKER_LOCK_STALE_SECONDS = 90.0

# Round-robin work types - exported for testing
WORK_TYPES = ["job", "operation", "compilation", "asset_task"]


# yt-dlp descarga fragmentos en paralelo (concurrent_fragment_downloads) y llama
# el progress_hook desde varios hilos a la vez. La Session de SQLAlchemy no es
# thread-safe: dos commits concurrentes sobre la misma sesion revientan con
# "Method 'commit()' can't be called here; method 'commit()' is already in
# progress". Este lock serializa cualquier escritura de progreso a la DB.
_session_lock = threading.Lock()


def _touch(job: Job) -> None:
    job.updated_at = datetime.now(UTC)


def _log(session: Session, job: Job, message: str, level: str = "info") -> None:
    add_job_log(session, job.id, sanitize_error_text(message), level)


def _clip_detection_source(reason: str) -> str:
    if reason.startswith("Fallback local:"):
        return "fallback"
    if reason.startswith("modo música:"):
        return "music"
    # Candidatos de la Etapa 1: nacen de picos de audio y cambios de escena, no
    # del transcript. Antes caian en el "return llm" de abajo y el job decia que
    # los habia elegido el LLM, que es justo lo contrario de lo que paso.
    if reason.startswith("señal:"):
        return "signal"
    return "llm"


def _job_detection_source(sources: list[str]) -> str:
    unique = {source for source in sources if source}
    if not unique:
        return ""
    for single in ("fallback", "music", "llm", "signal"):
        if unique == {single}:
            return single
    return "mixed"


def _update(
    session: Session,
    job: Job,
    *,
    status=None,
    stage=None,
    progress=None,
    log_stage: bool = True,
) -> None:
    with _session_lock:
        session.refresh(job)
        if job.status == JobStatus.canceled or job_control.is_cancel_requested(job.id):
            raise RuntimeError("Job cancelado manualmente")
        previous_status = job.status
        previous_stage = job.stage
        if status is not None:
            job.status = status
        if job.stage_started_at is None or (status is not None and job.status != previous_status):
            job.stage_started_at = datetime.now(UTC)
        if stage is not None:
            job.stage = stage
        if progress is not None:
            job.progress = round(progress, 3)
        if log_stage and stage is not None and stage != previous_stage:
            _log(session, job, stage)
        _touch(job)
        session.add(job)
        session.commit()


def _stage_progress(
    session: Session,
    job: Job,
    *,
    status: JobStatus,
    label: str,
    start: float,
    end: float,
) -> Callable[[float, str], None]:
    last_pct = -1
    last_message = ""

    def update(local_progress: float, message: str = "") -> None:
        nonlocal last_message, last_pct
        local_progress = max(0.0, min(1.0, local_progress))
        pct = round(local_progress * 100)
        if pct == last_pct and message == last_message and pct not in {0, 100}:
            return
        last_pct = pct
        last_message = message
        stage = message or f"{label} {pct}%"
        overall = start + (end - start) * local_progress
        log_stage = pct in {0, 100} or pct % 10 == 0
        _update(session, job, status=status, stage=stage, progress=overall, log_stage=log_stage)

    return update


def _claim_for_processing(job_id: str) -> bool:
    """Reclamo atómico del job (queued → ingesting). False si otro proceso ya lo tomó."""
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE job SET status = 'ingesting', stage = 'Preparando trabajo', "
                "stage_started_at = CURRENT_TIMESTAMP "
                "WHERE id = :id AND status = 'queued'"
            ),
            {"id": job_id},
        )
        return result.rowcount == 1


def process_job(job_id: str, *, claimed: bool = False) -> None:
    if not claimed and not _claim_for_processing(job_id):
        return
    token = media.set_cancel_check(lambda: job_control.is_cancel_requested(job_id))
    try:
        _process_claimed_job(job_id)
    finally:
        media.reset_cancel_check(token)


def _ingest_and_probe(session: Session, job: Job) -> media.VideoInfo:
    """Descarga la fuente (si es una URL) y la sondea con ffprobe.

    Persiste duration/width/height en el job y deja un aviso si la resolucion
    real (modo video, no audio) quedo por debajo de lo pedido.
    """
    _update(session, job, status=JobStatus.ingesting, stage="Analizando fuente", progress=0.05)
    is_audio_only = job.output_mode == "audio"
    if job.source_type == SourceType.url:
        download_progress = _stage_progress(
            session,
            job,
            status=JobStatus.ingesting,
            label="Descargando",
            start=0.05,
            end=0.20,
        )
        download_progress(0.0, "Descargando 0%")
        if is_audio_only:
            ingest.download_audio_url(
                job.id,
                job.source_name,
                progress_callback=download_progress,
                audio_format=job.audio_format,
                cancel_check=lambda: job_control.is_cancel_requested(job.id),
            )
        else:
            ingest.download_url(
                job.id,
                job.source_name,
                progress_callback=download_progress,
                quality=job.download_quality,
                cancel_check=lambda: job_control.is_cancel_requested(job.id),
            )
    info = ingest.probe_audio_source(job.id) if is_audio_only else ingest.probe_source(job.id)
    job.duration = info.duration
    job.source_width = info.width
    job.source_height = info.height
    session.add(job)
    session.commit()

    if job.source_type == SourceType.url and not is_audio_only:
        requested_height = ingest.requested_quality_height(job.download_quality)
        # margen del 15%: evita ruido por redondeos (1072 vs 1080) sin dejar pasar
        # una diferencia real como pedir 1080p y recibir 360p.
        if requested_height and info.height and info.height < requested_height * 0.85:
            _log(
                session,
                job,
                f"Calidad pedida: hasta {requested_height}p. La fuente solo ofrecio "
                f"{info.height}p ahora mismo (posible restriccion del sitio de origen, "
                "no es un limite de Snipra) -- se uso lo mejor disponible por debajo del "
                "techo pedido.",
                "warning",
            )
    return info


def _apply_sponsorblock(session: Session, job: Job) -> None:
    """Solo aplica a fuentes URL de YouTube; nunca a uploads directos.

    Fail-open: cualquier error de red/parseo deja sponsorblock_segments en [].
    """
    if job.sponsorblock_enabled and job.source_type == SourceType.url:
        sponsor_segments = sponsorblock.fetch_segments_for_url(
            job.source_name, settings.sponsorblock_categories
        )
        if sponsor_segments:
            job.sponsorblock_segments = sponsor_segments
            _log(
                session,
                job,
                f"SponsorBlock: {len(sponsor_segments)} tramo(s) excluidos del pool de clips",
            )
            session.add(job)
            session.commit()


def _detect_candidates(
    session: Session,
    job: Job,
    info: media.VideoInfo,
    doc: dict,
    signals_doc: dict,
    src,
    music_mode: bool,
) -> list:
    editorial_options = normalize_editorial_options(job.editorial_options)
    detection_progress = _stage_progress(
        session,
        job,
        status=JobStatus.detecting,
        label="Detectando momentos",
        start=0.55,
        end=0.65,
    )
    detection_progress(0.0, "Detectando momentos 0%")
    sponsor_segments = job.sponsorblock_segments if job.sponsorblock_enabled else []
    if music_mode:
        candidates = detect.detect_music_clips(
            info.duration,
            signals_doc,
            num_clips=job.num_clips,
            min_seconds=settings.min_clip_seconds,
            max_seconds=settings.max_clip_seconds,
            sponsorblock_segments=sponsor_segments,
            event_callback=lambda level, message: _log(session, job, message, level),
        )
        detection_progress(1.0, "Detección completada")
    else:
        if job.provider == "nvidia":
            from app.providers.nvidia_catalog import resolve_model

            requested_model = job.model or "auto"
            job.model = resolve_model(requested_model)
            session.add(job)
            session.commit()
            _log(session, job, f"NVIDIA: {requested_model} → {job.model}")
        candidates = detect.detect_clips(
            doc,
            provider=job.provider,
            model=job.model,
            num_clips=job.num_clips,
            formats=job.format_list,
            processing_mode=job.processing_mode,
            signals_doc=signals_doc,
            video_path=str(src),
            event_callback=lambda level, message: _log(session, job, message, level),
            progress_callback=detection_progress,
            sponsorblock_segments=sponsor_segments,
            semantic_cards=bool(editorial_options["semantic_cards"]),
        )
    if editorial_options["semantic_cards"]:
        candidates = [
            candidate.model_copy(
                update={
                    "editorial_cards": editorial.sanitize_cards(candidate, doc.get("segments", []))
                }
            )
            for candidate in candidates
        ]
    if editorial_options["compact_editing"] and job.processing_mode in {
        "auto_smart",
        "podcast_viral",
        "vlog",
    }:
        candidates = [
            editorial.compact_candidate(
                candidate,
                doc.get("segments", []),
                signals_doc.get("silence_intervals", []),
            )
            for candidate in candidates
        ]
    detection_sources = [_clip_detection_source(c.reason) for c in candidates]
    job.detection_source = _job_detection_source(detection_sources)
    if job.detection_source:
        _log(
            session,
            job,
            f"Detección: {job.detection_source} ({job.provider}/{job.model or 'default'})",
        )
        session.add(job)
        session.commit()
    found = len(candidates)
    target = 1 if found == 1 and getattr(candidates[0], "source_ranges", []) else job.num_clips
    if found >= target:
        _log(session, job, f"Detectados {found} clips · listo para render")
    elif found == 0:
        _log(session, job, "No se encontraron momentos virales en el video", "warning")
    else:
        _log(
            session,
            job,
            f"Se encontraron {found} de {target} clips solicitados "
            f"(video corto o sin suficientes momentos distintos)",
            "warning",
        )
    _update(
        session,
        job,
        status=JobStatus.detecting,
        stage=f"Detectados {found}/{target} clips",
        progress=0.65,
    )
    return candidates


def _render_candidates(
    session: Session, job: Job, src, candidates: list, doc: dict,
    decision_results: list[list[dict]] | None = None,
) -> None:
    _update(session, job, status=JobStatus.rendering, stage="Generando clips 0%", progress=0.65)
    total = max(1, len(candidates))
    for i, cand in enumerate(candidates):
        clip_base_progress = i / total
        _update(
            session,
            job,
            stage=f"Generando clip {i + 1}/{total} · {round(clip_base_progress * 100)}%",
            progress=0.65 + 0.35 * clip_base_progress,
        )
        clip = Clip(
            job_id=job.id,
            index=i,
            start=cand.start,
            end=cand.end,
            title=cand.title,
            hook=cand.hook,
            score=cand.score,
            reason=cand.reason,
            detection_source=_clip_detection_source(cand.reason),
            hashtags=cand.hashtags,
            description=cand.description,
            source_ranges=[item.model_dump() for item in cand.source_ranges],
            editorial_cards=[item.model_dump() for item in cand.editorial_cards],
            score_breakdown=[item.model_dump() for item in cand.score_breakdown],
            decision_results=decision_results[i] if decision_results else [],
        )
        session.add(clip)
        session.flush()  # asigna clip.id

        initial_style_path = ingest.job_dir(job.id) / "initial_style.json"
        initial_style: dict | None = None
        if initial_style_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                initial_style = json.loads(initial_style_path.read_text(encoding="utf-8"))
        rendered = render.render_clip(
            job.id,
            src,
            i,
            cand,
            job.format_list,
            segments=doc.get("segments", []),
            processing_mode=job.processing_mode,
            caption_style=job.caption_style,
            captions_enabled=job.captions_enabled,
            style_config=initial_style,
            event_callback=lambda level, message: _log(session, job, message, level),
            editorial_options=normalize_editorial_options(job.editorial_options),
            media_options=job.media_options or {},
        )
        for a in rendered:
            session.add(
                ClipAsset(
                    clip_id=clip.id,
                    format=a.format,
                    path=a.path,
                    thumb_path=a.thumb_path,
                    width=a.width,
                    height=a.height,
                    duration=a.duration,
                    size_bytes=a.size_bytes,
                )
            )
        clip.privacy_suggestions = render.load_clip_privacy_suggestions(job.id, i, cand.title)
        clip.protected_zones = render.load_clip_protected_zones(job.id, i, cand.title)
        clip.editorial_qa = render.load_clip_editorial_qa(job.id, i, cand.title)
        session.add(clip)
        render_pct = (i + 1) / total
        _update(
            session,
            job,
            stage=f"Generando clips {round(render_pct * 100)}%",
            progress=0.65 + 0.35 * render_pct,
        )


def _process_claimed_job(job_id: str) -> None:
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return

        # 1) Ingesta / sondeo
        info = _ingest_and_probe(session, job)

        if job.output_mode == "audio":
            _log(session, job, "Audio extraido; listo para guardar")
            _update(
                session,
                job,
                status=JobStatus.done,
                stage="Listo para descargar",
                progress=1.0,
            )
            return

        if job.output_mode == "download":
            _log(session, job, "Video original descargado; listo para guardar")
            _update(
                session,
                job,
                status=JobStatus.done,
                stage="Listo para descargar",
                progress=1.0,
            )
            return

        _apply_sponsorblock(session, job)

        signals_doc = {}
        src = ingest.find_source(job_id)
        if settings.enable_media_signals:
            _update(
                session,
                job,
                status=JobStatus.ingesting,
                stage="Analizando escenas/audio",
                progress=0.20,
            )
            signals_doc = signals.analyze_media(job_id, src, info.duration)

        music_mode = is_music_mode(job.processing_mode)

        # 2) Transcripción (se omite en modo música: no hay voz que transcribir)
        if music_mode:
            doc = {"segments": [], "duration": info.duration, "language": ""}
        else:
            editorial_options = normalize_editorial_options(job.editorial_options)
            whisper_model = whisper_model_for_quality(
                str(editorial_options["transcription_quality"]),
                settings.whisper_model,
            )
            if editorial_options["transcription_quality"] != "default":
                _log(
                    session,
                    job,
                    "Transcripción "
                    f"{editorial_options['transcription_quality']}: modelo {whisper_model}",
                )
            transcribe_progress = _stage_progress(
                session,
                job,
                status=JobStatus.transcribing,
                label="Transcribiendo",
                start=0.25 if settings.enable_media_signals else 0.20,
                end=0.55,
            )
            transcribe_progress(0.0, "Transcribiendo 0%")
            doc = transcribe.transcribe_job(
                job_id,
                progress_callback=transcribe_progress,
                model_name=whisper_model,
            )

        # 3) Detección
        candidates = _detect_candidates(session, job, info, doc, signals_doc, src, music_mode)

        # Advisory only, after final selection/scoring. Boundary also isolates
        # unexpected adapter/service defects before invoking the unchanged renderer.
        decisions = []
        if settings.decision_engine_enabled:
            try:
                decisions = evaluate_candidates(
                    candidates, doc, signals_doc, video_id=job.id, config=settings,
                    job_provider=job.provider, job_model=job.model,
                )
            except Exception:
                decisions = [[DecisionResult(
                    engine=settings.decision_engine_provider, task=task, status="error",
                ).model_dump(mode="json") for task in DecisionTask] for _ in candidates]

        # 4) Render
        if decisions:
            _render_candidates(session, job, src, candidates, doc, decision_results=decisions)
        else:
            _render_candidates(session, job, src, candidates, doc)

        _update(
            session,
            job,
            status=JobStatus.done,
            stage=f"Listo · {len(candidates)} clips",
            progress=1.0,
        )


def _fail(job_id: str, message: str) -> None:
    message = sanitize_error_text(message)

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        if job.status == JobStatus.canceled or job_control.is_cancel_requested(job_id):
            if job.status != JobStatus.canceled:
                job.status = JobStatus.canceled
                job.stage = "Cancelado"
                job.error = None
                job.stage_started_at = datetime.now(UTC)
                _log(session, job, "Job cancelado durante el procesamiento")
                _touch(job)
                session.add(job)
                session.commit()
            return
        job.status = JobStatus.failed
        job.error = message[:2000]
        job.stage = "Error"
        job.stage_started_at = datetime.now(UTC)
        _log(session, job, message, level="error")
        _touch(job)
        session.add(job)
        session.commit()


def claim_next() -> str | None:
    with Session(engine) as session:
        job = session.exec(
            select(Job).where(Job.status == JobStatus.queued).order_by(Job.created_at)
        ).first()
        if job is None:
            return None
        job.status = JobStatus.ingesting
        job.stage = "Preparando trabajo"
        job.progress = max(job.progress, CLAIMED_BUT_NOT_STARTED_PROGRESS)
        job.stage_started_at = datetime.now(UTC)
        _log(session, job, "Trabajo tomado por el worker")
        _touch(job)
        session.add(job)
        session.commit()
        return job.id


def recover_claimed_jobs() -> int:
    """Re-encola jobs activos de una ejecución anterior."""
    recovered = 0
    with Session(engine) as session:
        recoverable = ACTIVE_JOB_STATUSES - {JobStatus.queued}
        jobs = session.exec(select(Job).where(Job.status.in_(recoverable))).all()
        for job in jobs:
            if job_control.is_cancel_requested(job.id):
                job.status = JobStatus.canceled
                job.stage = "Cancelado"
                job.error = None
                job.stage_started_at = datetime.now(UTC)
                _log(session, job, "Cancelación recuperada tras reiniciar el worker")
                _touch(job)
                session.add(job)
                continue
            # Replay starts from source. Partial render rows and files cannot be
            # reused: detection may produce a different candidate list on replay.
            for clip in session.exec(select(Clip).where(Clip.job_id == job.id)).all():
                session.delete(clip)  # assets cascade via Clip.assets
            clips_dir = (settings.work_dir / job.id / "clips").resolve()
            if settings.work_dir.resolve() not in clips_dir.parents:
                raise ValueError("Partial clips directory escapes work storage")
            if clips_dir.exists():
                shutil.rmtree(clips_dir)
            job.status = JobStatus.queued
            job.stage = "En cola"
            job.progress = 0.0
            job.stage_started_at = datetime.now(UTC)
            _log(session, job, "Job recuperado tras reiniciar el worker")
            _touch(job)
            session.add(job)
            recovered += 1
        if jobs:
            session.commit()
    return recovered


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and (
                exit_code.value == still_active
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _heartbeat_age_seconds() -> float:
    """Segundos desde el ultimo latido del dueño del lock (inf si no se puede leer)."""
    try:
        return max(0.0, time.time() - WORKER_LOCK_PATH.stat().st_mtime)
    except OSError:
        return float("inf")


def worker_lock_is_stale() -> bool:
    """True si el lock existe pero su dueño ya no esta trabajando.

    Se exige que el PID siga vivo Y que el latido sea reciente. Con solo el PID
    no alcanza: tras matar un worker a la fuerza el SO puede asignar ese mismo
    PID a un proceso cualquiera, y el lock quedaba bloqueando para siempre.
    """
    try:
        content = WORKER_LOCK_PATH.read_text(encoding="utf-8").strip()
        # O_EXCL publishes the file before its owner can write the PID. An
        # empty, fresh lock belongs to an acquisition in progress, on POSIX too.
        if not content:
            return _heartbeat_age_seconds() > WORKER_LOCK_STALE_SECONDS
        pid = int(content)
    except (OSError, ValueError):
        return True
    if not _pid_is_alive(pid):
        return True
    return _heartbeat_age_seconds() > WORKER_LOCK_STALE_SECONDS


def touch_worker_lock() -> None:
    """Latido: marca que este worker sigue vivo y trabajando."""
    with contextlib.suppress(OSError):
        WORKER_LOCK_PATH.touch()


def _heartbeat_loop(stop: threading.Event) -> None:
    # Vive en su propio hilo a proposito: el loop principal puede quedarse horas
    # dentro de un render y no volveria a latir a tiempo.
    while not stop.wait(WORKER_HEARTBEAT_SECONDS):
        touch_worker_lock()


def acquire_worker_lock() -> bool:
    """Evita que dos workers recuperen o rendericen la misma cola local."""
    settings.ensure_dirs()
    for _ in range(2):
        try:
            fd = os.open(WORKER_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not worker_lock_is_stale():
                return False
            WORKER_LOCK_PATH.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as lock:
            lock.write(str(os.getpid()))
        return True
    return False


def release_worker_lock() -> None:
    try:
        owner = int(WORKER_LOCK_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    if owner == os.getpid():
        WORKER_LOCK_PATH.unlink(missing_ok=True)


# Test seam: extract one round of the round-robin scheduler for behavioral testing.
# Returns (work_done: bool, work_type: str | None, work_id: str | None, next_index: int)
def _try_one_work_round(
    work_type_index: int,
    WORK_TYPES: list[str],
    process_job_fn=None,
    process_operation_fn=None,
    process_compilation_fn=None,
    process_asset_task_fn=None,
) -> tuple[bool, str | None, str | None, int]:
    """Execute one round of the worker's round-robin scheduler.
    
    This is the single implementation of the round-robin logic, used by both
    the production _run_loop() and the test seam.
    
    Args:
        work_type_index: Current index in WORK_TYPES rotation
        WORK_TYPES: List of work type names in order
        process_job_fn: Function to process a job (default: process_job)
        process_operation_fn: Function to process an operation (default: process_operation)
        process_compilation_fn: Function to process a compilation (default: process_compilation)
        process_asset_task_fn: Function to process an asset task (default: process_asset_task)
    
    Returns:
        (work_done, work_type, work_id, next_work_type_index)
    """
    from app.worker import asset_tasks as worker_asset_tasks
    from app.worker.clip_operations import claim_next_operation, fail_operation, process_operation
    from app.worker.compositions import (
        claim_next_compilation,
        fail_compilation,
        process_compilation,
    )
    from app.worker.runner import claim_next, process_job

    # API and worker are separate processes. Refresh once at the work boundary,
    # preserving provider/model and editorial overrides already stored on Jobs.
    load_runtime_settings()
    
    # Use injected functions or defaults
    process_job_fn = process_job_fn or process_job
    process_operation_fn = process_operation_fn or process_operation
    process_compilation_fn = process_compilation_fn or process_compilation
    process_asset_task_fn = process_asset_task_fn or worker_asset_tasks.process_asset_task
    
    for _ in range(len(WORK_TYPES)):
        work_type = WORK_TYPES[work_type_index]
        work_type_index = (work_type_index + 1) % len(WORK_TYPES)
        if work_type == "job":
            job_id = None
            try:
                job_id = claim_next()
            except Exception as exc:  # noqa: BLE001
                print(sanitize_traceback(exc), flush=True)
            if job_id is not None:
                try:
                    process_job_fn(job_id, claimed=True)
                except Exception as exc:  # noqa: BLE001
                    print(sanitize_traceback(exc), flush=True)
                    _fail(job_id, f"{type(exc).__name__}: {exc}")
                return True, "job", job_id, work_type_index
        elif work_type == "operation":
            operation_id = None
            try:
                operation_id = claim_next_operation()
            except Exception as exc:  # noqa: BLE001
                print(sanitize_traceback(exc), flush=True)
            if operation_id is not None:
                try:
                    process_operation_fn(operation_id)
                except Exception as exc:  # noqa: BLE001
                    print(sanitize_traceback(exc), flush=True)
                    fail_operation(operation_id, f"{type(exc).__name__}: {exc}")
                return True, "operation", operation_id, work_type_index
        elif work_type == "compilation":
            compilation_id = None
            try:
                compilation_id = claim_next_compilation()
            except Exception as exc:  # noqa: BLE001
                print(sanitize_traceback(exc), flush=True)
            if compilation_id is not None:
                try:
                    process_compilation_fn(compilation_id)
                except Exception as exc:  # noqa: BLE001
                    print(sanitize_traceback(exc), flush=True)
                    fail_compilation(compilation_id, f"{type(exc).__name__}: {exc}")
                return True, "compilation", compilation_id, work_type_index
        elif work_type == "asset_task":
            asset_task_id = None
            try:
                asset_task_id = worker_asset_tasks.claim_next_asset_task()
            except Exception as exc:  # noqa: BLE001
                print(sanitize_traceback(exc), flush=True)
            if asset_task_id is not None:
                try:
                    process_asset_task_fn(asset_task_id)
                except Exception as exc:  # noqa: BLE001
                    print(sanitize_traceback(exc), flush=True)
                return True, "asset_task", asset_task_id, work_type_index
    return False, None, None, work_type_index


def _run_loop() -> None:
    init_db()
    load_runtime_settings()
    recovered = recover_claimed_jobs()
    recovered_operations = recover_operations()
    recovered_compilations = recover_compilations()
    recovered_asset_tasks = recover_asset_tasks()
    if recovered:
        print(f"[worker] {recovered} job(s) recuperados y reencolados", flush=True)
    if recovered_operations:
        print(
            f"[worker] {recovered_operations} operacion(es) masivas recuperadas",
            flush=True,
        )
    if recovered_compilations:
        print(
            f"[worker] {recovered_compilations} compilacion(es) recuperadas",
            flush=True,
        )
    if recovered_asset_tasks:
        print(
            f"[worker] {recovered_asset_tasks} tarea(s) de mejora recuperadas",
            flush=True,
        )
    print("[worker] Snipra worker iniciado. Esperando jobs…", flush=True)
    last_maintenance = 0.0
    last_channel_poll = 0.0
    # Round-robin fairness: rotate through work types each iteration
    work_type_index = 0
    while True:
        now = time.monotonic()
        if now - last_maintenance >= MAINTENANCE_INTERVAL_SECONDS:
            with Session(engine) as session:
                cleanup = cleanup_expired_job_files(
                    session,
                    settings.storage_retention_days,
                )
            if cleanup["removed_paths"]:
                print(
                    f"[worker] retencion elimino {cleanup['removed_paths']} carpeta(s)",
                    flush=True,
                )
            last_maintenance = now
        if now - last_channel_poll >= CHANNEL_POLL_INTERVAL_SECONDS:
            try:
                checked = channel_watch.poll_due_channels()
                if checked:
                    print(f"[worker] {checked} canal(es) trackeado(s) revisado(s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - nunca debe tumbar el loop principal
                print(sanitize_traceback(exc), flush=True)
            last_channel_poll = now
        # Round-robin claim: try each work type in rotation, starting from last position
        work_done, work_type, work_id, work_type_index = _try_one_work_round(
            work_type_index,
            WORK_TYPES,
        )
        if work_done:
            print(f"[worker] procesando {work_type} {work_id}", flush=True)
            print(f"[worker] {work_type} {work_id} OK", flush=True)
        if not work_done:
            _sleep_if_idle(work_done)


def _sleep_if_idle(work_done: bool) -> None:
    """Sleep if no work was done in this iteration.
    
    This is a testable seam for the idle path.
    """
    if not work_done:
        time.sleep(POLL_SECONDS)


def main() -> None:
    if not acquire_worker_lock():
        print("[worker] Ya existe un worker activo; no se inicia otro.", flush=True)
        return
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat_loop, args=(stop_heartbeat,), name="snipra-worker-heartbeat", daemon=True
    )
    heartbeat.start()
    try:
        _run_loop()
    finally:
        stop_heartbeat.set()
        release_worker_lock()


if __name__ == "__main__":
    main()
