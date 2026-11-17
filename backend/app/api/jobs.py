"""Endpoints de jobs: crear (upload/URL), listar, detalle."""

from __future__ import annotations

import contextlib
import json
import mimetypes
import shutil
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, delete, select

from app.core import job_control
from app.core.config import settings
from app.core.db import get_session
from app.core.editorial_options import normalize_editorial_options
from app.core.job_state import ACTIVE_JOB_STATUSES, add_job_log
from app.core.media_options import normalize_media_options
from app.core.processing_modes import normalize_processing_mode
from app.core.runtime_settings import AUDIO_FORMATS, DOWNLOAD_QUALITIES, effective_caption_style
from app.models import (
    Clip,
    ClipAsset,
    ClipOperation,
    ClipOperationStatus,
    Job,
    JobLog,
    JobStatus,
    SourceType,
    TrackedChannel,
    TrackedChannelVideo,
)
from app.pipeline import ingest
from app.providers.registry import DEFAULT_MODELS

router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobCreateURL(BaseModel):
    url: str
    num_clips: int | None = None
    formats: list[str] | None = None
    processing_mode: str | None = None
    provider: str | None = None
    model: str | None = None
    caption_style: str | None = None
    style_config: dict | None = None
    output_mode: str | None = None
    captions_enabled: bool | None = None
    download_quality: str | None = None
    audio_format: str | None = None
    sponsorblock_enabled: bool | None = None
    editorial_options: dict | None = None
    media_options: dict | None = None


def _save_style_config(job_id: str, style_config: dict | None) -> None:
    if not style_config:
        return
    job_path = ingest.job_dir(job_id)
    job_path.mkdir(parents=True, exist_ok=True)
    (job_path / "initial_style.json").write_text(
        json.dumps(style_config, ensure_ascii=False), encoding="utf-8"
    )


PROVIDER_ALIASES = {
    "anthropic": "claude",
    "claude": "claude",
    "openai": "openai",
    "nvidia": "nvidia",
    "naive": "naive",
    "claude_code": "claude_code",
    "codex": "codex",
}


def _provider_from_model(model: str | None) -> str | None:
    value = (model or "").strip().lower()
    if value.startswith("claude"):
        return "claude"
    if value.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def _normalize_provider_model(provider: str | None, model: str | None) -> tuple[str, str]:
    def _valid(name: str) -> bool:
        return name in DEFAULT_MODELS or name == "naive"

    requested = (provider or "").strip().lower()
    requested = PROVIDER_ALIASES.get(requested, requested)

    from_settings = (settings.llm_provider or "").strip().lower()
    from_settings = PROVIDER_ALIASES.get(from_settings, from_settings)

    # Un proveedor ELEGIDO manda, venga del request o de Ajustes: no se re-deduce.
    # Tratar solo el request como "elegido" no alcanzaba: la UI normalmente NO
    # manda proveedor y cae en el de Ajustes, asi que la deduccion por nombre de
    # modelo seguia reescribiendo claude_code -> claude, es decir la suscripcion
    # del usuario terminaba facturandose a su API key. La API solo debe usarse si
    # el usuario la eligio, nunca por deduccion.
    if _valid(requested):
        selected_provider = requested
        explicit = True
    elif _valid(from_settings):
        selected_provider = from_settings
        explicit = True
    else:
        # Fail-safe: NUNCA caer en una API pagada por omision. Sin provider
        # valido explicito ni en Ajustes, el default es el provider local.
        selected_provider = from_settings or "naive"
        explicit = False

    selected_model = (model or "").strip()

    if not selected_model:
        # settings.llm_model SOLO aplica si el proveedor elegido es el mismo que
        # el guardado en Ajustes. Si el usuario elige un proveedor DISTINTO al de
        # Ajustes (p.ej. Claude Code al subir, con NVIDIA/deepseek guardado en
        # Ajustes), ese modelo guardado pertenece al otro proveedor y no tiene
        # sentido aca -- job.model terminaba en "deepseek-ai/deepseek-v4-pro" con
        # provider="claude_code", y ese nombre se pasaba tal cual como --model al
        # CLI de Claude Code, que no lo reconoce. Con proveedor distinto, usa el
        # default de ESE proveedor en vez del modelo de otro.
        if selected_provider == from_settings and settings.llm_model:
            selected_model = settings.llm_model
        else:
            selected_model = DEFAULT_MODELS.get(selected_provider, "")

    # La deduccion por nombre de modelo es el ULTIMO recurso: solo si nadie eligio
    # un proveedor valido.
    if not explicit:
        provider_from_model = _provider_from_model(selected_model)
        if provider_from_model:
            selected_provider = provider_from_model

    if selected_provider not in DEFAULT_MODELS and selected_provider != "naive":
        # Fallback final: SOLO si Ajustes tiene un provider valido. Nunca Claude
        # por omision.
        settings_provider = PROVIDER_ALIASES.get(settings.llm_provider, settings.llm_provider)
        selected_provider = settings_provider if settings_provider in DEFAULT_MODELS else "naive"

    if not selected_model:
        # Mismo criterio que arriba: solo reusar settings.llm_model si de verdad
        # es el modelo de ESTE proveedor (selected_provider pudo cambiar arriba
        # por deduccion o por ser invalido).
        if selected_provider == from_settings and settings.llm_model:
            selected_model = settings.llm_model
        else:
            selected_model = DEFAULT_MODELS.get(selected_provider, "")

    return selected_provider, selected_model


OUTPUT_MODES = {"clips", "download", "audio"}


def _normalize_output_mode(value: str | None) -> str:
    mode = (value or "clips").strip().lower()
    if mode not in OUTPUT_MODES:
        raise HTTPException(400, "Modo de salida no soportado: usa clips, download o audio")
    return mode


def _normalize_download_quality(value: str | None) -> str:
    quality = (value or settings.download_quality or "auto").strip().lower()
    quality = {"2k": "1440", "4k": "2160", "best": "auto", "max": "auto"}.get(quality, quality)
    if quality not in DOWNLOAD_QUALITIES:
        raise HTTPException(400, "Calidad no soportada: usa auto, 720, 1080, 1440 o 2160")
    return quality


def _normalize_audio_format(value: str | None) -> str:
    fmt = (value or settings.audio_format or "mp3").strip().lower()
    if fmt not in AUDIO_FORMATS:
        raise HTTPException(400, "Formato de audio no soportado: usa mp3, wav, m4a, opus o flac")
    return fmt


def _defaults(
    num_clips,
    formats,
    provider,
    model,
    processing_mode,
    caption_style=None,
    output_mode=None,
    captions_enabled=None,
    download_quality=None,
    sponsorblock_enabled=None,
    audio_format=None,
    editorial_options=None,
    media_options=None,
) -> dict:
    try:
        mode = normalize_processing_mode(processing_mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    format_list = formats or settings.default_formats
    default_caption_style = "viral_bold" if "9x16" in format_list else effective_caption_style()
    normalized_provider, normalized_model = _normalize_provider_model(provider, model)
    normalized_output_mode = _normalize_output_mode(output_mode)

    requested_clips = num_clips or settings.default_num_clips
    if mode in {"gaming_hook", "gaming_highlights"}:
        requested_clips = 1

    return {
        "num_clips": requested_clips,
        "formats": ",".join(format_list),
        "processing_mode": mode,
        "caption_style": caption_style or default_caption_style,
        "output_mode": normalized_output_mode,
        "captions_enabled": True if captions_enabled is None else bool(captions_enabled),
        "download_quality": _normalize_download_quality(download_quality),
        "audio_format": _normalize_audio_format(audio_format),
        "provider": normalized_provider,
        "model": normalized_model,
        "sponsorblock_enabled": (
            True if sponsorblock_enabled is None else bool(sponsorblock_enabled)
        ),
        "editorial_options": normalize_editorial_options(editorial_options),
        "media_options": normalize_media_options(media_options),
    }


@router.post("", status_code=201)
async def create_job_upload(
    file: UploadFile = File(...),
    num_clips: int | None = Form(None),
    formats: str | None = Form(None),
    processing_mode: str | None = Form(None),
    caption_style: str | None = Form(None),
    style_config: str | None = Form(None),
    provider: str | None = Form(None),
    model: str | None = Form(None),
    output_mode: str | None = Form(None),
    captions_enabled: bool | None = Form(None),
    download_quality: str | None = Form(None),
    audio_format: str | None = Form(None),
    sponsorblock_enabled: bool | None = Form(None),
    editorial_options: str | None = Form(None),
    media_options: str | None = Form(None),
    session: Session = Depends(get_session),
) -> dict:
    fmt_list = [f.strip() for f in formats.split(",")] if formats else None
    editorial_cfg: dict | None = None
    if editorial_options:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            editorial_cfg = json.loads(editorial_options)
    media_cfg: dict | None = None
    if media_options:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            media_cfg = json.loads(media_options)
    defaults = _defaults(
        num_clips,
        fmt_list,
        provider,
        model,
        processing_mode,
        caption_style,
        output_mode,
        captions_enabled,
        download_quality,
        sponsorblock_enabled,
        audio_format,
        editorial_cfg,
        media_cfg,
    )
    if defaults["output_mode"] in {"download", "audio"}:
        raise HTTPException(
            400, "Los modos de descarga directa y extraer audio requieren una URL publica."
        )
    job = Job(
        source_type=SourceType.upload,
        source_name=file.filename or "upload.mp4",
        title=file.filename or "Video",
        status=JobStatus.queued,
        **defaults,
    )
    style_cfg_parsed: dict | None = None
    if style_config:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            style_cfg_parsed = json.loads(style_config)
    # Do not publish a claimable row until all source bytes and initial style
    # exist. The UUID is assigned locally; no DB transaction is held during IO.
    try:
        ingest.save_upload(job.id, file.filename or "upload.mp4", file.file)
        _save_style_config(job.id, style_cfg_parsed)
        session.add(job)
        session.flush()
        _log(session, job.id, "Job creado y enviado al procesador")
        session.commit()
    except Exception:
        session.rollback()
        shutil.rmtree(ingest.job_dir(job.id), ignore_errors=True)
        raise

    session.refresh(job)

    return {"id": job.id, "status": job.status}


@router.post("/url", status_code=201)
def create_job_url(
    body: JobCreateURL,
    session: Session = Depends(get_session),
) -> dict:
    if not body.url.strip():
        raise HTTPException(400, "URL vacía")
    try:
        ingest.validate_download_url(body.url)
    except ingest.VideoDownloadError as exc:
        raise HTTPException(400, str(exc)) from exc
    job = Job(
        source_type=SourceType.url,
        source_name=body.url,
        title=body.url,
        status=JobStatus.queued,
        **_defaults(
            body.num_clips,
            body.formats,
            body.provider,
            body.model,
            body.processing_mode,
            body.caption_style,
            body.output_mode,
            body.captions_enabled,
            body.download_quality,
            body.sponsorblock_enabled,
            body.audio_format,
            body.editorial_options,
            body.media_options,
        ),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    _save_style_config(job.id, body.style_config)
    _log(session, job.id, "Job creado y enviado al procesador")
    session.commit()

    return {"id": job.id, "status": job.status}


def _channel_labels_by_job(session: Session, job_ids: list[str] | None = None) -> dict[str, str]:
    """Mapa job_id -> label del canal trackeado que lo creó (una sola query, sin N+1)."""
    query = (
        select(TrackedChannelVideo.job_id, TrackedChannel.label)
        .join(TrackedChannel, TrackedChannel.id == TrackedChannelVideo.channel_id)
        .where(TrackedChannelVideo.job_id.is_not(None))
    )
    if job_ids is not None:
        if not job_ids:
            return {}
        query = query.where(TrackedChannelVideo.job_id.in_(job_ids))
    rows = session.exec(query).all()
    return {job_id: label for job_id, label in rows if job_id}


def _iso_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _serialize(job: Job, channel_label: str | None = None) -> dict:
    return {
        "id": job.id,
        "created_at": _iso_utc(job.created_at),
        "updated_at": _iso_utc(job.updated_at),
        "stage_started_at": _iso_utc(job.stage_started_at),
        "source_type": job.source_type,
        "source_name": job.source_name,
        "title": job.title,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error": job.error,
        "duration": job.duration,
        "source_width": job.source_width,
        "source_height": job.source_height,
        "num_clips": job.num_clips,
        "formats": job.format_list,
        "processing_mode": job.processing_mode,
        "caption_style": job.caption_style,
        "output_mode": job.output_mode,
        "captions_enabled": job.captions_enabled,
        "download_quality": job.download_quality or settings.download_quality,
        "audio_format": job.audio_format or settings.audio_format,
        "provider": job.provider,
        "model": job.model,
        "detection_source": job.detection_source,
        "sponsorblock_enabled": job.sponsorblock_enabled,
        "sponsorblock_segments": job.sponsorblock_segments,
        "editorial_options": normalize_editorial_options(job.editorial_options),
        "media_options": normalize_media_options(job.media_options),
        "channel_label": channel_label,
    }


def _delete_clip_outputs(session: Session, job_id: str) -> None:
    clips = session.exec(select(Clip).where(Clip.job_id == job_id)).all()
    for clip in clips:
        assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip.id)).all()
        for asset in assets:
            session.delete(asset)
        session.delete(clip)
    work = settings.work_dir / job_id
    shutil.rmtree(work / "clips", ignore_errors=True)
    for name in ("segments.json", "audio.wav"):
        path = work / name
        if path.exists():
            path.unlink(missing_ok=True)


def _log(session: Session, job_id: str, message: str, level: str = "info") -> None:
    add_job_log(session, job_id, message, level)


@router.get("")
def list_jobs(session: Session = Depends(get_session)) -> list[dict]:
    jobs = session.exec(select(Job).order_by(Job.created_at.desc())).all()
    labels = _channel_labels_by_job(session, [j.id for j in jobs])
    return [_serialize(j, labels.get(j.id)) for j in jobs]


@router.get("/{job_id}")
def get_job(job_id: str, session: Session = Depends(get_session)) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    labels = _channel_labels_by_job(session, [job_id])
    return _serialize(job, labels.get(job_id))


@router.get("/{job_id}/download")
def download_original(job_id: str, session: Session = Depends(get_session)):
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.output_mode not in {"download", "audio"}:
        raise HTTPException(400, "Este job no es una descarga directa ni de audio")
    if job.status != JobStatus.done:
        raise HTTPException(409, "La descarga aun no esta lista")
    try:
        source = (
            ingest.find_audio_source(job.id)
            if job.output_mode == "audio"
            else ingest.find_source(job.id)
        )
    except FileNotFoundError as exc:
        raise HTTPException(404, "Archivo descargado no encontrado") from exc
    media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    prefix = "snipra-audio" if job.output_mode == "audio" else "snipra-original"
    filename = f"{prefix}-{job.id[:8]}{source.suffix.lower() or '.mp4'}"
    return FileResponse(source, media_type=media_type, filename=filename)


@router.get("/{job_id}/logs")
def get_job_logs(job_id: str, session: Session = Depends(get_session)) -> list[dict]:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    logs = session.exec(
        select(JobLog).where(JobLog.job_id == job_id).order_by(JobLog.created_at)
    ).all()
    return [
        {
            "id": item.id,
            "created_at": item.created_at.isoformat(),
            "level": item.level,
            "message": item.message,
        }
        for item in logs
    ]


@router.get("/{job_id}/export-manifest")
def export_manifest(job_id: str, session: Session = Depends(get_session)) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    clips = session.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.index)).all()
    payload = _serialize(job)
    payload["clips"] = []
    for clip in clips:
        assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip.id)).all()
        payload["clips"].append(
            {
                "index": clip.index,
                "title": clip.title,
                "hook": clip.hook,
                "score": clip.score,
                **({"decision_results": clip.decision_results} if clip.decision_results else {}),
                "reason": clip.reason,
                "detection_source": clip.detection_source,
                "start": clip.start,
                "end": clip.end,
                "duration": clip.duration,
                "source_ranges": clip.source_ranges or [],
                "assets": [
                    {
                        "format": asset.format,
                        "path": asset.path,
                        "thumb_path": asset.thumb_path,
                        "width": asset.width,
                        "height": asset.height,
                        "duration": asset.duration,
                        "size_bytes": asset.size_bytes,
                    }
                    for asset in assets
                ],
                "suggested_caption": clip.description or clip.hook or clip.title,
                "description": clip.description,
                "hashtags": clip.hashtags or _hashtags_for(job.processing_mode),
            }
        )
    return payload


def _hashtags_for(processing_mode: str) -> list[str]:
    tags = {
        "gaming": ["#gaming", "#gameplay", "#clips"],
        "gaming_auto": ["#gaming", "#gameplay", "#clips"],
        "gaming_hook": ["#gaming", "#gameplay", "#gamingclips"],
        "gaming_highlights": ["#gaming", "#highlights", "#gamingclips"],
        "podcast_viral": ["#podcast", "#clips", "#shorts"],
        "tutorial_screen": ["#tutorial", "#aprende", "#tips"],
    }
    return tags.get(processing_mode, ["#shorts", "#clips"])


@router.post("/{job_id}/retry")
def retry_job(
    job_id: str,
    session: Session = Depends(get_session),
) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.status not in {JobStatus.failed, JobStatus.done, JobStatus.canceled}:
        raise HTTPException(400, "Solo se puede reintentar un job fallido o terminado")

    _delete_clip_outputs(session, job.id)
    job.status = JobStatus.queued
    job.stage = "En cola"
    job.progress = 0.0
    job.error = None
    job_control.clear_cancel_request(job.id)
    now = datetime.now(UTC)
    job.updated_at = now
    job.stage_started_at = now
    _log(session, job.id, "Job reencolado manualmente")
    session.add(job)
    session.commit()
    session.refresh(job)

    return _serialize(job)


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, session: Session = Depends(get_session)) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.status not in ACTIVE_JOB_STATUSES:
        raise HTTPException(400, "Solo se puede cancelar un job activo")

    job.status = JobStatus.canceled
    job.stage = "Cancelado"
    job.error = None
    now = datetime.now(UTC)
    job.updated_at = now
    job.stage_started_at = now
    job_control.request_cancel(job.id)
    _log(session, job.id, "Job cancelado manualmente")
    session.add(job)
    session.commit()
    session.refresh(job)
    return _serialize(job)


@router.delete("/{job_id}", status_code=204)
def delete_job(job_id: str, session: Session = Depends(get_session)) -> None:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if job.status in ACTIVE_JOB_STATUSES - {JobStatus.queued}:
        raise HTTPException(400, "El job está en proceso; cancélalo antes de borrarlo")
    active_operation = session.exec(
        select(ClipOperation).where(
            ClipOperation.job_id == job_id,
            ClipOperation.status.in_({ClipOperationStatus.queued, ClipOperationStatus.running}),
        )
    ).first()
    if active_operation is not None:
        raise HTTPException(400, "Hay una operacion masiva activa para este job")
    # borra clips/assets (cascade ORM), logs y el registro
    session.exec(delete(JobLog).where(JobLog.job_id == job_id))
    session.exec(delete(ClipOperation).where(ClipOperation.job_id == job_id))
    session.delete(job)
    session.commit()
    # borra los archivos del job en disco
    work = settings.work_dir / job_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
