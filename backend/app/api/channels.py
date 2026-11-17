"""Endpoints de canales trackeados: alta, listado, edición, borrado y chequeo manual."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, delete, func, select

from app.core.db import get_session
from app.models import TrackedChannel, TrackedChannelVideo
from app.pipeline import ingest

router = APIRouter(prefix="/channels", tags=["channels"])


class ChannelCreate(BaseModel):
    url: str
    label: str | None = None
    poll_interval_minutes: int | None = None
    num_clips: int | None = None
    formats: list[str] | None = None
    processing_mode: str | None = None
    provider: str | None = None
    model: str | None = None
    caption_style: str | None = None
    output_mode: str | None = None
    captions_enabled: bool | None = None
    download_quality: str | None = None
    sponsorblock_enabled: bool | None = None


class ChannelUpdate(BaseModel):
    label: str | None = None
    enabled: bool | None = None
    poll_interval_minutes: int | None = None
    num_clips: int | None = None
    formats: list[str] | None = None
    processing_mode: str | None = None
    provider: str | None = None
    model: str | None = None
    caption_style: str | None = None
    output_mode: str | None = None
    captions_enabled: bool | None = None
    download_quality: str | None = None
    sponsorblock_enabled: bool | None = None


def _jobs_created_count(session: Session, channel_id: str) -> int:
    return session.exec(
        select(func.count())
        .select_from(TrackedChannelVideo)
        .where(
            TrackedChannelVideo.channel_id == channel_id,
            TrackedChannelVideo.job_id.is_not(None),
        )
    ).one()


def _serialize(channel: TrackedChannel, session: Session) -> dict:
    return {
        "id": channel.id,
        "created_at": channel.created_at.isoformat(),
        "url": channel.url,
        "label": channel.label,
        "enabled": channel.enabled,
        "poll_interval_minutes": channel.poll_interval_minutes,
        "last_checked_at": (
            channel.last_checked_at.isoformat() if channel.last_checked_at else None
        ),
        "last_check_status": channel.last_check_status or ("healthy" if channel.last_checked_at else ""),
        "last_check_error": channel.last_check_error,
        "last_success_at": (
            channel.last_success_at.isoformat() if channel.last_success_at else None
        ),
        "consecutive_failures": channel.consecutive_failures,
        "videos_seen_last_check": channel.videos_seen_last_check,
        "new_videos_last_check": channel.new_videos_last_check,
        "canonical_url": channel.canonical_url,
        "channel_title": channel.channel_title,
        "num_clips": channel.num_clips,
        "formats": channel.format_list,
        "processing_mode": channel.processing_mode,
        "caption_style": channel.caption_style,
        "output_mode": channel.output_mode,
        "captions_enabled": channel.captions_enabled,
        "download_quality": channel.download_quality,
        "provider": channel.provider,
        "model": channel.model,
        "sponsorblock_enabled": channel.sponsorblock_enabled,
        "jobs_created": _jobs_created_count(session, channel.id),
    }


@router.post("", status_code=201)
def create_channel(body: ChannelCreate, session: Session = Depends(get_session)) -> dict:
    if not body.url.strip():
        raise HTTPException(400, "URL vacía")
    try:
        ingest.validate_download_url(body.url)
    except ingest.VideoDownloadError as exc:
        raise HTTPException(400, str(exc)) from exc

    from app.api.jobs import _defaults  # import diferido: evita import circular en carga del módulo

    defaults = _defaults(
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
    )
    channel = TrackedChannel(
        url=body.url.strip(),
        label=(body.label or "").strip(),
        poll_interval_minutes=max(5, body.poll_interval_minutes or 60),
        **defaults,
    )
    session.add(channel)
    session.commit()
    session.refresh(channel)
    return _serialize(channel, session)


@router.get("")
def list_channels(session: Session = Depends(get_session)) -> list[dict]:
    channels = session.exec(select(TrackedChannel).order_by(TrackedChannel.created_at.desc())).all()
    return [_serialize(c, session) for c in channels]


@router.patch("/{channel_id}")
def update_channel(
    channel_id: str,
    body: ChannelUpdate,
    session: Session = Depends(get_session),
) -> dict:
    channel = session.get(TrackedChannel, channel_id)
    if channel is None:
        raise HTTPException(404, "Canal no encontrado")

    data = body.model_dump(exclude_unset=True)
    if "formats" in data and data["formats"] is not None:
        data["formats"] = ",".join(f.strip() for f in data["formats"] if f.strip()) or "9x16"
    if "poll_interval_minutes" in data and data["poll_interval_minutes"] is not None:
        data["poll_interval_minutes"] = max(5, data["poll_interval_minutes"])
    if "label" in data and data["label"] is not None:
        data["label"] = data["label"].strip()

    for key, value in data.items():
        setattr(channel, key, value)
    session.add(channel)
    session.commit()
    session.refresh(channel)
    return _serialize(channel, session)


@router.delete("/{channel_id}", status_code=204)
def delete_channel(channel_id: str, session: Session = Depends(get_session)) -> None:
    channel = session.get(TrackedChannel, channel_id)
    if channel is None:
        raise HTTPException(404, "Canal no encontrado")
    session.exec(delete(TrackedChannelVideo).where(TrackedChannelVideo.channel_id == channel_id))
    session.delete(channel)
    session.commit()


@router.post("/{channel_id}/check")
def check_channel_now(channel_id: str, session: Session = Depends(get_session)) -> dict:
    channel = session.get(TrackedChannel, channel_id)
    if channel is None:
        raise HTTPException(404, "Canal no encontrado")

    from app.pipeline import channel_watch  # import diferido: evita import circular

    result = channel_watch.check_channel(channel_id)
    session.refresh(channel)
    payload = _serialize(channel, session)
    payload["check_result"] = result
    return payload
