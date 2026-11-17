"""Polling de canales de YouTube trackeados: detecta videos nuevos y encola jobs.

Todo el listado remoto (yt-dlp extract_flat) es best-effort: canal privado, URL
inválida o timeout registran diagnósticos estructurados y actualizan la salud del
canal, sin tumbar el worker ni otros canales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlmodel import Session, select

from app.core.db import engine
from app.core.job_state import add_job_log
from app.core.sanitize import sanitize_error_text
from app.models import Job, JobStatus, SourceType, TrackedChannel, TrackedChannelVideo
from app.pipeline import ingest

logger = logging.getLogger(__name__)

MAX_VIDEOS_PER_CHECK = 15


class ExtractionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    ERROR = "ERROR"


@dataclass
class ExtractionResult:
    status: ExtractionStatus
    entries: list[dict] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    channel_title: str | None = None
    channel_id: str | None = None
    canonical_url: str | None = None


def _categorize_extraction_error(exc: Exception) -> tuple[str, str]:
    text = sanitize_error_text(str(exc))
    lower = text.lower()
    if "404" in lower or "not found" in lower or "does not exist" in lower:
        return "CHANNEL_NOT_FOUND", "El canal o playlist no fue encontrado en YouTube (HTTP 404)."
    if "timed out" in lower or "timeout" in lower or "socket_timeout" in lower:
        return "TIMEOUT", "Tiempo de espera agotado al consultar el canal."
    if (
        "private" in lower
        or "members-only" in lower
        or "sign in" in lower
        or "login required" in lower
        or "bot" in lower
    ):
        return "AUTH_REQUIRED", "El canal o contenido requiere autenticación o sesión."
    if "unsupported url" in lower:
        return "UNSUPPORTED_URL", "La URL no es compatible con el extractor de canales."
    
    clean_snippet = text.split("\n")[0].strip()
    if len(clean_snippet) > 120:
        clean_snippet = clean_snippet[:117] + "..."
    if "traceback" in clean_snippet.lower() or "error" not in clean_snippet.lower():
        clean_snippet = "No se pudo consultar el canal de YouTube."
    return "EXTRACTION_FAILED", f"Error al consultar el canal: {clean_snippet}"


def _extract_flat_entries(channel_url: str) -> ExtractionResult:
    """Lista hasta MAX_VIDEOS_PER_CHECK videos recientes de un canal sin descargarlos."""
    import yt_dlp

    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "playlistend": MAX_VIDEOS_PER_CHECK,
        "socket_timeout": 20,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
    except Exception as exc:  # noqa: BLE001 - best-effort, nunca rompe el worker
        from app.core.sanitize import sanitize_error_text

        # El texto del extractor puede contener URLs con credenciales del proxy
        # o del entorno yt-dlp; nunca loguear la excepcion cruda.
        logger.warning(
            "channel_watch: extract_flat falló para %s: %s",
            channel_url,
            sanitize_error_text(str(exc)),
        )
        code, msg = _categorize_extraction_error(exc)
        return ExtractionResult(status=ExtractionStatus.ERROR, error_code=code, error_message=msg)

    if not isinstance(info, dict):
        return ExtractionResult(
            status=ExtractionStatus.ERROR,
            error_code="EXTRACTION_FAILED",
            error_message="Respuesta no estructurada del extractor.",
        )

    channel_id = (
        info.get("channel_id")
        or (
            info.get("id")
            if isinstance(info.get("id"), str) and info.get("id", "").startswith("UC")
            else None
        )
    )
    channel_title = info.get("title") or info.get("uploader") or info.get("channel")
    canonical_url = info.get("channel_url") or info.get("uploader_url") or info.get("webpage_url")

    raw_entries = info.get("entries")
    if raw_entries is None:
        # direct video URL or single item
        vid = info.get("id")
        if vid and isinstance(vid, str) and not (vid.startswith("UC") and len(vid) > 15):
            url = (
                info.get("url")
                or info.get("webpage_url")
                or f"https://www.youtube.com/watch?v={vid}"
            )
            entry = {"id": vid, "url": url, "title": info.get("title") or url}
            return ExtractionResult(
                status=ExtractionStatus.SUCCESS,
                entries=[entry],
                channel_title=channel_title,
                channel_id=channel_id,
                canonical_url=canonical_url,
            )
        return ExtractionResult(
            status=ExtractionStatus.EMPTY,
            channel_title=channel_title,
            channel_id=channel_id,
            canonical_url=canonical_url,
        )

    flat: list[dict] = []
    seen_ids: set[str] = set()

    def _process_item(item: dict) -> None:
        if not isinstance(item, dict) or len(flat) >= MAX_VIDEOS_PER_CHECK:
            return

        # Desempaquetar tabs / sub-playlists anidadas (ej. YoutubeTab: Videos, Shorts, Live)
        sub = item.get("entries")
        if sub is not None:
            for child in sub:
                _process_item(child)
                if len(flat) >= MAX_VIDEOS_PER_CHECK:
                    return
            return

        video_id = item.get("id")
        if not video_id or not isinstance(video_id, str):
            return

        # Ignorar entradas de playlist/tab intermedias que usan el channel ID como id
        if item.get("_type") == "playlist" or (video_id.startswith("UC") and len(video_id) > 15):
            return

        if video_id in seen_ids:
            return
        seen_ids.add(video_id)

        url = item.get("url") or item.get("webpage_url")
        if not url or not url.startswith("http"):
            url = f"https://www.youtube.com/watch?v={video_id}"

        title = item.get("title") or url
        flat.append({"id": video_id, "url": url, "title": title})

    for entry in raw_entries:
        if not entry:
            continue
        _process_item(entry)
        if len(flat) >= MAX_VIDEOS_PER_CHECK:
            break

    status = ExtractionStatus.SUCCESS if flat else ExtractionStatus.EMPTY
    return ExtractionResult(
        status=status,
        entries=flat,
        channel_title=channel_title,
        channel_id=channel_id,
        canonical_url=canonical_url,
    )


def _create_job_for_channel_video(session: Session, channel: TrackedChannel, video: dict) -> str:
    from app.api.jobs import _defaults

    defaults = _defaults(
        channel.num_clips,
        channel.format_list,
        channel.provider,
        channel.model,
        channel.processing_mode,
        channel.caption_style,
        channel.output_mode,
        channel.captions_enabled,
        channel.download_quality,
        channel.sponsorblock_enabled,
    )
    job = Job(
        source_type=SourceType.url,
        source_name=video["url"],
        title=video.get("title") or video["url"],
        status=JobStatus.queued,
        **defaults,
    )
    session.add(job)
    session.flush()  # asigna job.id
    add_job_log(
        session,
        job.id,
        f"Job creado automáticamente por el canal trackeado '{channel.label or channel.url}'",
    )
    return job.id


def check_channel(channel_id: str) -> dict:
    """Revisa un canal: registra videos nuevos y encola jobs.

    En el primer chequeo de un canal (sin ningún TrackedChannelVideo previo) solo
    se registran los videos existentes como "ya vistos" — no se crean jobs, para
    evitar inundar la cola al agregar un canal con historial largo.

    Revalida la URL (guard SSRF de `ingest.validate_download_url`) en CADA chequeo,
    no solo al crear el canal: la validación de alta es un check en un punto en el
    tiempo (TOCTOU) — el DNS del dominio guardado puede cambiar entre chequeos
    periódicos indefinidos, así que confiar solo en la validación de creación deja
    una ventana de SSRF repetible.
    """
    with Session(engine) as session:
        channel = session.get(TrackedChannel, channel_id)
        if channel is None:
            return {
                "checked": False,
                "success": False,
                "first_check": False,
                "videos_seen": 0,
                "new_videos": 0,
                "jobs_created": 0,
                "error_code": "NOT_FOUND",
                "error_message": "canal no encontrado",
            }

        is_first_check = (
            session.exec(
                select(TrackedChannelVideo.id)
                .where(TrackedChannelVideo.channel_id == channel_id)
                .limit(1)
            ).first()
            is None
        )

        try:
            ingest.validate_download_url(channel.url)
        except ingest.VideoDownloadError as exc:
            logger.warning(
                "channel_watch: URL del canal %s ya no pasa la validacion SSRF: %s",
                channel_id,
                exc,
            )
            now = datetime.now(UTC)
            channel.last_checked_at = now
            channel.last_check_status = "error"
            channel.last_check_error = f"URL invalida: {exc}"
            channel.consecutive_failures += 1
            session.add(channel)
            session.commit()
            return {
                "checked": True,
                "success": False,
                "first_check": is_first_check,
                "videos_seen": 0,
                "new_videos": 0,
                "jobs_created": 0,
                "error_code": "SSRF_REJECTED",
                "error_message": f"URL invalida: {exc}",
            }

        result = _extract_flat_entries(channel.url)

        if result.status == ExtractionStatus.ERROR:
            now = datetime.now(UTC)
            channel.last_checked_at = now
            channel.last_check_status = "error"
            channel.last_check_error = result.error_message
            channel.consecutive_failures += 1
            session.add(channel)
            session.commit()
            return {
                "checked": True,
                "success": False,
                "first_check": is_first_check,
                "videos_seen": 0,
                "new_videos": 0,
                "jobs_created": 0,
                "error_code": result.error_code,
                "error_message": result.error_message,
            }

        # Actualizar identidad canónica si se resolvió
        if result.canonical_url and not channel.canonical_url:
            channel.canonical_url = result.canonical_url
        if result.channel_title and not channel.channel_title:
            channel.channel_title = result.channel_title

        created_jobs: list[str] = []
        new_videos = 0
        for entry in result.entries:
            video_id = entry["id"]
            existing = session.exec(
                select(TrackedChannelVideo).where(
                    TrackedChannelVideo.channel_id == channel_id,
                    TrackedChannelVideo.video_id == video_id,
                )
            ).first()
            if existing is not None:
                continue

            new_videos += 1
            job_id: str | None = None
            if not is_first_check:
                try:
                    job_id = _create_job_for_channel_video(session, channel, entry)
                    created_jobs.append(job_id)
                except Exception as exc:  # noqa: BLE001 - un video roto no debe tumbar el chequeo
                    logger.warning(
                        "channel_watch: no se pudo crear job para video %s del canal %s: %s",
                        video_id,
                        channel_id,
                        sanitize_error_text(str(exc)),
                    )
                    job_id = None
            session.add(
                TrackedChannelVideo(channel_id=channel_id, video_id=video_id, job_id=job_id)
            )

        now = datetime.now(UTC)
        channel.last_checked_at = now
        channel.last_success_at = now
        channel.last_check_status = "healthy"
        channel.last_check_error = None
        channel.consecutive_failures = 0
        channel.videos_seen_last_check = len(result.entries)
        channel.new_videos_last_check = new_videos
        session.add(channel)
        session.commit()

        return {
            "checked": True,
            "success": True,
            "first_check": is_first_check,
            "videos_seen": len(result.entries),
            "new_videos": new_videos,
            "jobs_created": len(created_jobs),
            "error_code": None,
            "error_message": None,
        }


def poll_due_channels() -> int:
    """Revisa los canales activos cuyo intervalo de chequeo ya venció.

    Secuencial y simple a propósito (MVP): no bloquea el procesamiento normal de
    jobs porque el caller solo lo invoca cada CHANNEL_POLL_INTERVAL_SECONDS.
    """
    with Session(engine) as session:
        channels = session.exec(
            select(TrackedChannel).where(TrackedChannel.enabled.is_(True))
        ).all()
        now = datetime.now(UTC)
        due_ids = []
        for channel in channels:
            last_checked = channel.last_checked_at
            if last_checked is None:
                due_ids.append(channel.id)
                continue
            if last_checked.tzinfo is None:
                last_checked = last_checked.replace(tzinfo=UTC)
            if now - last_checked >= timedelta(minutes=channel.poll_interval_minutes):
                due_ids.append(channel.id)

    checked = 0
    for channel_id in due_ids:
        try:
            check_channel(channel_id)
        except Exception as exc:  # noqa: BLE001 - un canal roto no debe tumbar el resto
            logger.warning(
                "channel_watch: fallo al chequear canal %s: %s",
                channel_id,
                sanitize_error_text(str(exc)),
            )
        checked += 1
    return checked
