"""Endpoints de clips: listar por job, descargar/stream de assets y thumbnails."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import get_session
from app.core.editorial_options import normalize_editorial_options
from app.core.job_state import add_job_log
from app.models import (
    Clip,
    ClipAsset,
    ClipOperation,
    ClipOperationStatus,
    Compilation,
    CompilationStatus,
    Job,
)
from app.pipeline import ingest, reedit, render, transcribe
from app.pipeline.subtitles import normalize_style_config
from app.providers.base import ClipCandidate, ScoreContribution
from app.providers.captions import (
    TranslationError,
    rewrite_subtitle_blocks,
    translate_subtitle_blocks,
)

router = APIRouter(tags=["clips"])


class ClipUpdate(BaseModel):
    start: float | None = None
    end: float | None = None
    title: str | None = None
    hook: str | None = None
    description: str | None = None
    hashtags: list[str] | None = None


class ReeditRequest(BaseModel):
    remove: list[list[float]] = []  # rangos [start, end] relativos al clip (segundos) a eliminar
    remove_fillers: bool = False
    remove_repeated_takes: bool = False
    remove_long_pauses: bool = False


class StyleEditRequest(BaseModel):
    style: str | None = None
    style_config: dict | None = None
    apply_to_all: bool = False


class SubtitleEditRequest(BaseModel):
    subtitles: list[dict]
    style_config: dict | None = None
    apply_to_all: bool = False


class CaptionRewriteRequest(BaseModel):
    apply_to_all: bool = False
    provider: str | None = None
    model: str | None = None


class CaptionTranslateRequest(BaseModel):
    target_language: str
    apply_to_all: bool = False
    provider: str | None = None
    model: str | None = None


class PrivacyApplyRequest(BaseModel):
    suggestion_indices: list[int] | None = None
    masks: list[dict] | None = None


def _serialize_operation(operation: ClipOperation) -> dict:
    return {
        "id": operation.id,
        "job_id": operation.job_id,
        "clip_id": operation.clip_id,
        "kind": operation.kind,
        "status": operation.status,
        "progress": operation.progress,
        "error": operation.error,
        "created_at": operation.created_at,
        "updated_at": operation.updated_at,
    }


def _enqueue_operation(
    session: Session,
    job: Job,
    clip: Clip,
    kind: str,
    payload: dict,
) -> dict:
    active = session.exec(
        select(ClipOperation).where(
            ClipOperation.job_id == job.id,
            ClipOperation.status.in_({ClipOperationStatus.queued, ClipOperationStatus.running}),
        )
    ).first()
    if active is not None:
        raise HTTPException(409, "Ya hay una operacion masiva activa para este job")
    operation = ClipOperation(
        job_id=job.id,
        clip_id=clip.id,
        kind=kind,
        payload=payload,
    )
    session.add(operation)
    _log(session, job.id, f"Operacion masiva '{kind}' encolada")
    session.commit()
    session.refresh(operation)
    return _serialize_operation(operation)


@router.get("/clip-operations/{operation_id}")
def get_clip_operation(
    operation_id: str,
    session: Session = Depends(get_session),
) -> dict:
    operation = session.get(ClipOperation, operation_id)
    if operation is None:
        raise HTTPException(404, "Operacion no encontrada")
    return _serialize_operation(operation)


def _serialize_clip(clip: Clip, assets: list[ClipAsset]) -> dict:
    return {
        "id": clip.id,
        "index": clip.index,
        "start": clip.start,
        "end": clip.end,
        "duration": clip.duration,
        "source_ranges": clip.source_ranges or [],
        "title": clip.title,
        "hook": clip.hook,
        "score": clip.score,
        "reason": clip.reason,
        "detection_source": clip.detection_source,
        "hashtags": clip.hashtags or [],
        "description": clip.description,
        "editorial_cards": clip.editorial_cards or [],
        "privacy_suggestions": clip.privacy_suggestions or [],
        "privacy_masks": clip.privacy_masks or [],
        "protected_zones": clip.protected_zones or [],
        "editorial_qa": clip.editorial_qa or {},
        "score_breakdown": clip.score_breakdown or [],
        **({"decision_results": clip.decision_results} if clip.decision_results else {}),
        "assets": [
            {
                "id": a.id,
                "format": a.format,
                "width": a.width,
                "height": a.height,
                "duration": a.duration,
                "size_bytes": a.size_bytes,
                "video_url": f"/api/assets/{a.id}/video",
                "thumb_url": f"/api/assets/{a.id}/thumb" if a.thumb_path else None,
            }
            for a in assets
        ],
    }


@router.get("/jobs/{job_id}/clips")
def list_clips(job_id: str, session: Session = Depends(get_session)) -> list[dict]:
    clips = session.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.index)).all()
    out = []
    for clip in clips:
        assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip.id)).all()
        out.append(_serialize_clip(clip, assets))
    return out


def _clip_or_404(clip_id: str, session: Session) -> Clip:
    clip = session.get(Clip, clip_id)
    if clip is None:
        raise HTTPException(404, "Clip no encontrado")
    return clip


@router.get("/clips/{clip_id}/transcript")
def clip_transcript(clip_id: str, session: Session = Depends(get_session)) -> dict:
    """Palabras del clip (tiempo relativo al clip) para el editor por transcript."""
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    try:
        doc = transcribe.load_segments(job.id)
    except FileNotFoundError:
        return {"words": [], "has_words": False}

    ranges = [(r["start"], r["end"]) for r in clip.source_ranges] or [(clip.start, clip.end)]
    segments, _ = reedit.remap_segments(doc.get("segments", []), ranges)
    words: list[dict] = []
    for seg in segments:
        for w in seg.get("words") or []:
            ws, we = w.get("start"), w.get("end")
            if ws is None or we is None:
                continue
            words.append(
                {
                    "word": str(w.get("word", "")).strip(),
                    "start": round(float(ws), 3),
                    "end": round(float(we), 3),
                }
            )
    return {"words": words, "has_words": bool(words)}


@router.get("/clips/{clip_id}/timeline")
def clip_timeline(clip_id: str, session: Session = Depends(get_session)) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    timeline_path = _clip_dir(job, clip) / "timeline.json"
    timeline = _read_json(timeline_path, None)
    if timeline is None:
        raise HTTPException(404, "Timeline no encontrado")

    for fmt in timeline.get("formats", []):
        for frame in fmt.get("frames", []):
            filename = str(frame.get("filename", ""))
            frame["url"] = f"/api/clips/{clip.id}/timeline/frames/{filename}"
    return timeline


@router.get("/clips/{clip_id}/timeline/frames/{filename}")
def clip_timeline_frame(
    clip_id: str,
    filename: str,
    session: Session = Depends(get_session),
):
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if not filename.startswith("timeline-") or not filename.endswith(".jpg"):
        raise HTTPException(400, "Frame invalido")
    path = _clip_dir(job, clip) / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Frame no encontrado")
    return FileResponse(path, media_type="image/jpeg")


def _assets_for_clip(session: Session, clip_id: str) -> list[ClipAsset]:
    return session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip_id)).all()


def _delete_assets(session: Session, clip: Clip, keep_paths: frozenset[str] = frozenset()) -> None:
    for asset in _assets_for_clip(session, clip.id):
        for path in (asset.path, asset.thumb_path):
            if path and path not in keep_paths:
                Path(path).unlink(missing_ok=True)
        session.delete(asset)


def _replace_assets(session: Session, clip: Clip, rendered: list) -> None:
    """Sustituye los assets del clip por los recién renderizados.

    Se llama DESPUÉS del render: si FFmpeg falla, los assets previos quedan intactos.
    """
    keep = frozenset(p for a in rendered for p in (a.path, a.thumb_path) if p)
    _delete_assets(session, clip, keep_paths=keep)
    _persist_assets(session, clip.id, rendered)


def _sync_editorial_artifacts(job: Job, clip: Clip) -> None:
    clip.privacy_suggestions = render.load_clip_privacy_suggestions(job.id, clip.index, clip.title)
    clip.protected_zones = render.load_clip_protected_zones(job.id, clip.index, clip.title)
    clip.editorial_qa = render.load_clip_editorial_qa(job.id, clip.index, clip.title)


def _log(session: Session, job_id: str, message: str, level: str = "info") -> None:
    add_job_log(session, job_id, message, level)


def _clip_dir(job: Job, clip: Clip) -> Path:
    clips_dir = ingest.job_dir(job.id) / "clips"
    portable = clips_dir / render._slug(clip.index, clip.title)
    if portable.exists():
        return portable
    legacy = clips_dir / render._legacy_slug(clip.index, clip.title)
    if legacy.exists():
        return legacy
    prefix = f"{clip.index + 1:02d}-"
    candidates = [path for path in clips_dir.glob(f"{prefix}*") if path.is_dir()]
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return portable


def _read_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _rerender_from_artifacts(
    session: Session,
    job: Job,
    clip: Clip,
    style_config: dict | None = None,
    subtitles: list[dict] | None = None,
    privacy_masks: list[dict] | None = None,
) -> dict:
    clip_dir = _clip_dir(job, clip)
    base = clip_dir / "clip_base.mp4"
    if not base.exists():
        raise HTTPException(400, "No existe clip_base.mp4 para re-renderizar sin reprocesar")

    current_style = _read_json(clip_dir / "style_config.json", {})
    style_name = style_config.get("style") if style_config else job.caption_style
    next_style = normalize_style_config(
        {**current_style, **(style_config or {})},
        style_name,
    )
    current_subtitles = _read_json(clip_dir / "subtitles.json", [])
    next_subtitles = subtitles if subtitles is not None else current_subtitles
    transcript = _read_json(clip_dir / "transcript.json", {"segments": []})

    _write_json(clip_dir / "style_config.json", next_style)
    _write_json(clip_dir / "subtitles.json", next_subtitles)

    candidate = _candidate_from_clip(clip, start=0.0, end=max(0.1, clip.duration))
    rendered = render.render_from_base(
        job.id,
        base,
        clip.index,
        candidate,
        job.format_list,
        segments=transcript.get("segments", []),
        processing_mode=job.processing_mode,
        caption_style=next_style["style"],
        captions_enabled=job.captions_enabled,
        style_config=next_style,
        subtitle_blocks=next_subtitles,
        editorial_options=normalize_editorial_options(job.editorial_options),
        privacy_masks=clip.privacy_masks if privacy_masks is None else privacy_masks,
        media_options=job.media_options or {},
    )
    _replace_assets(session, clip, rendered)
    _sync_editorial_artifacts(job, clip)
    session.add(clip)
    session.flush()
    return _serialize_clip(clip, _assets_for_clip(session, clip.id))


@router.patch("/clips/{clip_id}")
def update_clip(clip_id: str, body: ClipUpdate, session: Session = Depends(get_session)) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    if body.start is not None:
        clip.start = max(0.0, body.start)
        clip.source_ranges = []
    if body.end is not None:
        clip.end = max(0.0, body.end)
        clip.source_ranges = []
    if clip.end <= clip.start:
        raise HTTPException(400, "El final debe ser mayor que el inicio")
    if job.duration is not None and clip.end > job.duration + 0.001:
        raise HTTPException(400, "El final no puede superar la duracion del video")
    if body.title is not None:
        clip.title = body.title.strip()
    if body.hook is not None:
        clip.hook = body.hook.strip()
    if body.description is not None:
        clip.description = body.description.strip()
    if body.hashtags is not None:
        clip.hashtags = [t.strip() for t in body.hashtags if t.strip()]

    _log(session, job.id, f"Clip {clip.index + 1} editado manualmente")
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return _serialize_clip(clip, _assets_for_clip(session, clip.id))


@router.post("/clips/{clip_id}/render")
def rerender_clip(clip_id: str, session: Session = Depends(get_session)) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    try:
        doc = transcribe.load_segments(job.id)
        src = ingest.find_source(job.id)
    except FileNotFoundError as exc:
        raise HTTPException(400, "No hay source/transcripcion para regenerar este clip") from exc

    candidate = _candidate_from_clip(clip)
    rendered = render.render_clip(
        job.id,
        src,
        clip.index,
        candidate,
        job.format_list,
        segments=doc.get("segments", []),
        processing_mode=job.processing_mode,
        caption_style=job.caption_style,
        captions_enabled=job.captions_enabled,
        editorial_options=normalize_editorial_options(job.editorial_options),
        privacy_masks=clip.privacy_masks or [],
        media_options=job.media_options or {},
    )
    _replace_assets(session, clip, rendered)
    _sync_editorial_artifacts(job, clip)
    _log(session, job.id, f"Clip {clip.index + 1} regenerado")
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return _serialize_clip(clip, _assets_for_clip(session, clip.id))


def _sanitize_privacy_masks(masks: list[dict], clip: Clip, formats: list[str]) -> list[dict]:
    from app.pipeline.reframe import target_size

    allowed_formats = formats or ["9x16"]
    clean: list[dict] = []
    for value in masks[:40]:
        fmt = str(value.get("format") or allowed_formats[0])
        if fmt not in allowed_formats:
            continue
        width, height = target_size(fmt)
        x = max(0, min(width - 2, int(value.get("x", 0))))
        y = max(0, min(height - 2, int(value.get("y", 0))))
        box_width = max(2, min(width - x, int(value.get("width", 0))))
        box_height = max(2, min(height - y, int(value.get("height", 0))))
        start = max(0.0, min(clip.duration, float(value.get("start", 0.0))))
        end = max(start + 0.05, min(clip.duration, float(value.get("end", start + 2.0))))
        clean.append(
            {
                "format": fmt,
                "x": x,
                "y": y,
                "width": box_width,
                "height": box_height,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(value.get("text", ""))[:120],
                "confirmed": True,
            }
        )
    return clean


@router.post("/clips/{clip_id}/privacy")
def apply_clip_privacy(
    clip_id: str,
    body: PrivacyApplyRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Aplica desenfoques únicamente después de la confirmación explícita del usuario."""
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if body.masks is not None:
        selected = body.masks
    else:
        suggestions = clip.privacy_suggestions or []
        indices = body.suggestion_indices
        selected = (
            suggestions
            if indices is None
            else [suggestions[index] for index in indices if 0 <= index < len(suggestions)]
        )
    masks = _sanitize_privacy_masks(selected, clip, job.format_list)
    clip.privacy_masks = masks
    _rerender_from_artifacts(session, job, clip, privacy_masks=masks)
    _log(session, job.id, f"Privacidad del clip {clip.index + 1}: {len(masks)} zona(s) confirmadas")
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return _serialize_clip(clip, _assets_for_clip(session, clip.id))


@router.post("/clips/{clip_id}/style")
def edit_clip_style(
    clip_id: str,
    body: StyleEditRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    style_payload = dict(body.style_config or {})
    if body.style:
        style_payload["style"] = body.style

    if body.apply_to_all:
        response.status_code = 202
        return _enqueue_operation(session, job, clip, "style", {"style_config": style_payload})

    result = _rerender_from_artifacts(session, job, clip, style_config=style_payload)
    _log(session, job.id, f"Estilo del clip {clip.index + 1} actualizado sin reanalizar")
    session.commit()
    return result


@router.post("/clips/{clip_id}/subtitles")
def edit_clip_subtitles(
    clip_id: str,
    body: SubtitleEditRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    if body.apply_to_all:
        response.status_code = 202
        return _enqueue_operation(
            session,
            job,
            clip,
            "subtitles",
            {"style_config": body.style_config, "subtitles": body.subtitles},
        )

    result = _rerender_from_artifacts(
        session,
        job,
        clip,
        style_config=body.style_config,
        subtitles=body.subtitles,
    )
    _log(session, job.id, f"Subtitulos del clip {clip.index + 1} actualizados sin reanalizar")
    session.commit()
    return result


@router.post("/clips/{clip_id}/captions/rewrite")
def rewrite_clip_captions(
    clip_id: str,
    body: CaptionRewriteRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    provider = body.provider or job.provider
    model = body.model if body.model is not None else job.model

    if body.apply_to_all:
        response.status_code = 202
        return _enqueue_operation(
            session,
            job,
            clip,
            "rewrite_captions",
            {"provider": provider, "model": model},
        )

    result = _rewrite_one(session, job, clip, provider=provider, model=model)
    _log(session, job.id, f"Captions del clip {clip.index + 1} reescritos")
    session.commit()
    return result


def _rewrite_one(
    session: Session,
    job: Job,
    clip: Clip,
    *,
    provider: str,
    model: str,
) -> dict:
    item_dir = _clip_dir(job, clip)
    subtitles = _read_json(item_dir / "subtitles.json", [])
    if not subtitles:
        raise HTTPException(400, "No hay subtitulos para reescribir")
    rewritten = rewrite_subtitle_blocks(subtitles, provider=provider, model=model)
    return _rerender_from_artifacts(session, job, clip, subtitles=rewritten)


@router.post("/clips/{clip_id}/captions/translate")
def translate_clip_captions(
    clip_id: str,
    body: CaptionTranslateRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")

    target_language = body.target_language.strip()
    if not target_language:
        raise HTTPException(400, "Debes indicar el idioma destino")

    provider = body.provider or job.provider
    model = body.model if body.model is not None else job.model

    if body.apply_to_all:
        response.status_code = 202
        return _enqueue_operation(
            session,
            job,
            clip,
            "translate_captions",
            {"target_language": target_language, "provider": provider, "model": model},
        )

    result = _translate_one(
        session, job, clip, target_language=target_language, provider=provider, model=model
    )
    _log(session, job.id, f"Captions del clip {clip.index + 1} traducidos a {target_language}")
    session.commit()
    return result


def _translate_one(
    session: Session,
    job: Job,
    clip: Clip,
    *,
    target_language: str,
    provider: str,
    model: str,
) -> dict:
    item_dir = _clip_dir(job, clip)
    subtitles = _read_json(item_dir / "subtitles.json", [])
    if not subtitles:
        raise HTTPException(400, "No hay subtitulos para traducir")
    try:
        translated = translate_subtitle_blocks(
            subtitles, target_language=target_language, provider=provider, model=model
        )
    except TranslationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _rerender_from_artifacts(session, job, clip, subtitles=translated)


@router.post("/clips/{clip_id}/reedit")
def reedit_clip(clip_id: str, body: ReeditRequest, session: Session = Depends(get_session)) -> dict:
    clip = _clip_or_404(clip_id, session)
    job = session.get(Job, clip.job_id)
    if job is None:
        raise HTTPException(404, "Job no encontrado")
    if clip.source_ranges and any(r.get("kind") != "reedit" for r in clip.source_ranges):
        raise HTTPException(
            409,
            "Este clip ya es un montaje de varios tramos; edita subtitulos o estilo sin "
            "recalcular sus cortes.",
        )

    try:
        doc = transcribe.load_segments(job.id)
        src = ingest.find_source(job.id)
    except FileNotFoundError as exc:
        raise HTTPException(400, "No hay source/transcripcion para reeditar este clip") from exc

    source_ranges = [(r["start"], r["end"]) for r in clip.source_ranges] or [
        (clip.start, clip.end)
    ]
    segments, current_duration = reedit.remap_segments(doc.get("segments", []), source_ranges)
    remove_abs = [(r[0], r[1]) for r in body.remove if len(r) == 2]
    if body.remove_fillers or body.remove_repeated_takes or body.remove_long_pauses:
        words_in = sorted(
            (
                w
                for seg in segments
                for w in (seg.get("words") or [])
                if w.get("start") is not None and 0 <= w["start"] <= current_duration
            ),
            key=lambda w: w["start"],
        )
        if body.remove_fillers:
            remove_abs += reedit.detect_filler_ranges(words_in)
        if body.remove_repeated_takes:
            remove_abs += reedit.detect_repeated_take_ranges(words_in)
        if body.remove_long_pauses:
            remove_abs += reedit.detect_long_pause_ranges(words_in)

    kept_relative = reedit.compute_kept_ranges(0, current_duration, remove_abs)
    kept = reedit.source_ranges_for_edit(source_ranges, kept_relative)
    if not kept:
        raise HTTPException(400, "No queda contenido tras la edicion")

    clips_dir = ingest.job_dir(job.id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    base = reedit.build_concat_base(src, kept, clips_dir / f"reedit-{clip.index}-base.mp4")
    new_segments, total = reedit.remap_segments(doc.get("segments", []), kept)

    candidate = _candidate_from_clip(clip, start=0.0, end=max(0.1, total))
    rendered = render.render_from_base(
        job.id,
        base,
        clip.index,
        candidate,
        job.format_list,
        segments=new_segments,
        processing_mode=job.processing_mode,
        caption_style=job.caption_style,
        captions_enabled=job.captions_enabled,
        # Reedit owns this timeline: replace old subtitle artifacts and prevent
        # render_from_base's fresh-clip intro trimming from changing kept ranges.
        subtitle_blocks=(
            render._segments_to_subtitle_blocks(new_segments, 0.0, total)
            if job.captions_enabled else []
        ),
        media_options=job.media_options or {},
    )
    _replace_assets(session, clip, rendered)
    clip.start, clip.end = kept[0][0], kept[-1][1]
    clip.source_ranges = [{"start": s, "end": e, "kind": "reedit", "score": clip.score}
                          for s, e in kept]
    _sync_editorial_artifacts(job, clip)
    _log(session, job.id, f"Clip {clip.index + 1} reeditado ({len(kept)} tramos, {total:.1f}s)")
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return _serialize_clip(clip, _assets_for_clip(session, clip.id))


def _candidate_from_clip(clip: Clip, start: float | None = None, end: float | None = None):
    return ClipCandidate(
        start=clip.start if start is None else start,
        end=clip.end if end is None else end,
        title=clip.title,
        hook=clip.hook,
        score=clip.score,
        reason=clip.reason,
        hashtags=clip.hashtags or [],
        description=clip.description,
        source_ranges=(clip.source_ranges or []) if start is None and end is None else [],
        editorial_cards=clip.editorial_cards or [],
        score_breakdown=[ScoreContribution(**item) for item in (clip.score_breakdown or [])],
    )


def _persist_assets(session: Session, clip_id: str, rendered: list) -> None:
    for asset in rendered:
        session.add(
            ClipAsset(
                clip_id=clip_id,
                format=asset.format,
                path=asset.path,
                thumb_path=asset.thumb_path,
                width=asset.width,
                height=asset.height,
                duration=asset.duration,
                size_bytes=asset.size_bytes,
            )
        )


def _asset_or_404(asset_id: str, session: Session) -> ClipAsset:
    asset = session.get(ClipAsset, asset_id)
    if asset is None:
        raise HTTPException(404, "Asset no encontrado")
    return asset


@router.get("/assets/{asset_id}/video")
def get_asset_video(asset_id: str, session: Session = Depends(get_session)):
    asset = _asset_or_404(asset_id, session)
    p = Path(asset.path)
    if not p.exists():
        raise HTTPException(404, "Archivo no encontrado")
    return FileResponse(p, media_type="video/mp4", filename=p.name)


@router.get("/assets/{asset_id}/thumb")
def get_asset_thumb(asset_id: str, session: Session = Depends(get_session)):
    asset = _asset_or_404(asset_id, session)
    if not asset.thumb_path or not Path(asset.thumb_path).exists():
        raise HTTPException(404, "Thumbnail no encontrado")
    return FileResponse(asset.thumb_path, media_type="image/jpeg")


def _remove_dir_inside_work(path: Path) -> None:
    """Borra un directorio solo si esta realmente dentro de work_dir.

    Las rutas vienen de la BD; si alguna quedara malformada, un rmtree a ciegas
    podria salirse del storage. Se valida contra work_dir antes de tocar nada.
    """
    try:
        resolved = path.resolve()
        root = settings.work_dir.resolve()
    except OSError:
        return
    if resolved == root or root not in resolved.parents:
        return
    shutil.rmtree(resolved, ignore_errors=True)


@router.delete("/assets/{asset_id}")
def delete_asset(asset_id: str, session: Session = Depends(get_session)) -> dict:
    """Borra un formato renderizado. Si era el ultimo del clip, borra el clip.

    La biblioteca del compositor lista assets (un clip en 9x16 y 16x9 son dos
    tarjetas), asi que se borra exactamente lo que el usuario ve.
    """
    asset = _asset_or_404(asset_id, session)
    clip = session.get(Clip, asset.clip_id)

    # Una compilacion ya renderizada no depende del asset (su mp4 ya existe),
    # pero una en cola fallaria al no encontrar el archivo. Se bloquea con un
    # mensaje util en vez de dejar que reviente despues en el worker.
    pending = session.exec(
        select(Compilation).where(
            Compilation.status.in_([CompilationStatus.queued, CompilationStatus.rendering])
        )
    ).all()
    blocking = [item for item in pending if asset_id in (item.asset_ids or [])]
    if blocking:
        raise HTTPException(
            409,
            f"Este clip lo esta usando una compilacion en curso "
            f'("{blocking[0].title or "sin titulo"}"). Espera a que termine o borrala.',
        )

    siblings = session.exec(select(ClipAsset).where(ClipAsset.clip_id == asset.clip_id)).all()
    clip_removed = len(siblings) <= 1

    if clip_removed and clip is not None:
        # Ultimo formato: se va la carpeta entera del clip, que ademas guarda el
        # corte base y los frames del Timeline QA.
        _remove_dir_inside_work(Path(asset.path).parent)
        session.delete(clip)  # cascade borra el asset
    else:
        Path(asset.path).unlink(missing_ok=True)
        if asset.thumb_path:
            Path(asset.thumb_path).unlink(missing_ok=True)
        session.delete(asset)

    session.commit()
    return {"ok": True, "clip_removed": clip_removed, "format": asset.format}
