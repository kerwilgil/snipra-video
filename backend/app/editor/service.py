from __future__ import annotations

import copy
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.config import settings
from app.core.media import probe
from app.editor import timeline as tl
from app.editor.asset_provenance import AssetOrigin, read_asset_provenance
from app.editor.models import (
    EditorAgentRun,
    EditorAsset,
    EditorOperation,
    EditorProject,
    EditorVersion,
    new_id,
)


def error(status: int, code: str, detail: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": detail})


#: Source/cache identity fingerprint = SHA-256 of at most the first 1 MiB.
#: Distinct from the enhanced artifact's full-file ``output_hash``
#: (``asset_provenance.compute_full_file_sha256``).
_FINGERPRINT_BYTES = 1024 * 1024


def first_mib_fingerprint(path: Path) -> str:
    """SHA-256 of the first ``_FINGERPRINT_BYTES`` bytes, read in a single
    bounded ``read()`` - never loads the whole file into memory. A file shorter
    than 1 MiB is hashed whole; bytes past the first MiB never affect it."""
    with path.open("rb") as fh:
        return hashlib.sha256(fh.read(_FINGERPRINT_BYTES)).hexdigest()


def project_or_404(session: Session, project_id: str) -> EditorProject:
    project = session.get(EditorProject, project_id)
    if not project:
        raise error(404, "PROJECT_NOT_FOUND", "Proyecto no encontrado")
    return project


def project_assets(session: Session, project_id: str) -> list[EditorAsset]:
    return session.exec(select(EditorAsset).where(EditorAsset.project_id == project_id)).all()


def asset_or_404(session: Session, project_id: str, asset_id: str) -> EditorAsset:
    asset = session.get(EditorAsset, asset_id)
    if not asset or asset.project_id != project_id:
        raise error(404, "MEDIA_MISSING", "Asset no encontrado")
    return asset


def current_version(session: Session, project: EditorProject) -> EditorVersion:
    version = session.exec(
        select(EditorVersion).where(
            EditorVersion.project_id == project.id, EditorVersion.number == project.current_version
        )
    ).first()
    if not version:
        raise error(409, "PROJECT_CONFLICT", "El proyecto no tiene una versión válida")
    return version


def create_project(
    session: Session,
    name: str,
    source_job_id: str | None = None,
    source_clip_id: str | None = None,
) -> EditorProject:
    project = EditorProject(
        name=name.strip() or "Untitled project",
        source_job_id=source_job_id,
        source_clip_id=source_clip_id,
    )
    session.add(project)
    session.flush()
    timeline = tl.empty_timeline()
    timeline["tracks"] = [
        tl.new_track("video", 0),
        tl.new_track("audio", 1),
        tl.new_track("text", 2),
        tl.new_track("overlay", 3),
    ]
    session.add(
        EditorVersion(project_id=project.id, number=0, timeline=timeline, summary="Project created")
    )
    session.commit()
    session.refresh(project)
    return project


def add_asset(
    session: Session,
    project: EditorProject,
    path: Path,
    *,
    kind: str = "video",
    duration_us: int = 0,
    width: int = 0,
    height: int = 0,
    has_audio: bool = False,
    asset_metadata: dict | None = None,
    commit: bool = True,
) -> EditorAsset:
    """Register a media file as an EditorAsset.

    ``commit=True`` (default) keeps the historical behaviour: the row is
    committed and refreshed before returning. ``commit=False`` only stages and
    flushes the row (its primary key is populated) so the caller can fold the
    insert into a larger transaction and commit once - used by the enhancement
    finaliser so the ENHANCED asset and its AssetTask land in a single commit.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        raise error(400, "MEDIA_MISSING", "El archivo multimedia no existe")
    digest = first_mib_fingerprint(resolved)
    asset = EditorAsset(
        project_id=project.id,
        original_path=str(resolved),
        filename=resolved.name,
        fingerprint=digest,
        kind=kind,
        duration_us=duration_us,
        width=width,
        height=height,
        has_audio=has_audio,
        asset_metadata=asset_metadata or {},
    )
    session.add(asset)
    if commit:
        session.commit()
        session.refresh(asset)
    else:
        session.flush()
    return asset


def import_upload(
    session: Session, project: EditorProject, source: Path, filename: str
) -> EditorAsset:
    """Copy an untrusted upload into the project's controlled media directory."""
    safe_name = Path(filename).name or "media"
    target_dir = settings.editor_dir / project.id / "media"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{new_id()}-{safe_name}"
    shutil.copyfile(source, target)
    info = probe(target)
    return add_asset(
        session,
        project,
        target,
        duration_us=round(info.duration * 1_000_000),
        width=info.width,
        height=info.height,
        has_audio=info.has_audio,
    )


def serialize_project(session: Session, project: EditorProject) -> dict:
    version = current_version(session, project)
    assets = session.exec(
        select(EditorAsset)
        .where(EditorAsset.project_id == project.id)
        .order_by(EditorAsset.created_at)
    ).all()
    return {
        "id": project.id,
        "name": project.name,
        "source_job_id": project.source_job_id,
        "source_clip_id": project.source_clip_id,
        "current_version": project.current_version,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
        "timeline": version.timeline,
        "assets": [serialize_asset(a) for a in assets],
    }


def serialize_project_listing(session: Session, project: EditorProject) -> dict:
    """Keep a damaged project visible without blocking every healthy project."""
    try:
        return serialize_project(session, project)
    except HTTPException as exc:
        if exc.status_code != 409 or not isinstance(exc.detail, dict) or exc.detail.get("code") != "PROJECT_CONFLICT":
            raise
        return {
            "id": project.id,
            "name": project.name,
            "source_job_id": project.source_job_id,
            "source_clip_id": project.source_clip_id,
            "current_version": project.current_version,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
            "timeline": None,
            "assets": [],
            "unavailable_reason": exc.detail["message"],
        }


def serialize_asset(asset: EditorAsset) -> dict:
    return {
        "id": asset.id,
        "filename": asset.filename,
        "kind": asset.kind,
        "duration_us": asset.duration_us,
        "width": asset.width,
        "height": asset.height,
        "has_audio": asset.has_audio,
        "fingerprint": asset.fingerprint,
        "metadata": asset.asset_metadata,
    }


def rename_project(session: Session, project_id: str, name: str) -> EditorProject:
    project = project_or_404(session, project_id)
    clean_name = name.strip()
    if not clean_name:
        raise error(400, "INVALID_PROJECT_NAME", "El nombre del proyecto no puede estar vacío")
    project.name = clean_name
    project.updated_at = datetime.now(UTC)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def _asset_reference_count(session: Session, project: EditorProject, asset_id: str) -> int:
    timeline = current_version(session, project).timeline
    return sum(
        1
        for track in timeline.get("tracks", [])
        for clip in track.get("clips", [])
        if clip.get("asset_id") == asset_id
    )


def remove_asset(session: Session, project_id: str, asset_id: str) -> None:
    project = project_or_404(session, project_id)
    asset = asset_or_404(session, project_id, asset_id)
    references = _asset_reference_count(session, project, asset.id)
    if references:
        noun = "clip" if references == 1 else "clips"
        raise error(
            409,
            "MEDIA_IN_USE",
            f"No se puede quitar este medio. Actualmente está siendo utilizado por {references} "
            f"{noun} de la timeline. Elimina primero los clips que lo utilizan.",
        )
    # Only the project record is removed.  Neither an external source nor the
    # project's uploaded media copy is deleted here.
    session.delete(asset)
    project.updated_at = datetime.now(UTC)
    session.add(project)
    session.commit()


def delete_project(session: Session, project_id: str) -> None:
    project = project_or_404(session, project_id)
    # These are project-owned database records. Media files are intentionally
    # retained: deleting a project must never delete a user's source video.
    for model in (EditorAgentRun, EditorOperation, EditorVersion, EditorAsset):
        for record in session.exec(select(model).where(model.project_id == project.id)).all():
            session.delete(record)
    session.delete(project)
    session.commit()


def apply_operation(session: Session, project_id: str, body: dict) -> dict:
    project = project_or_404(session, project_id)
    key = str(body.get("idempotency_key") or "")
    if not key:
        raise error(400, "INVALID_OPERATION", "idempotency_key es obligatorio")
    previous = session.exec(
        select(EditorOperation).where(
            EditorOperation.project_id == project.id, EditorOperation.idempotency_key == key
        )
    ).first()
    if previous:
        return {
            "operation_id": previous.id,
            "version": previous.version_number,
            "idempotent": True,
            "timeline": current_version(session, project).timeline,
        }
    base = body.get("base_version")
    if base != project.current_version:
        raise error(409, "STALE_VERSION", "La timeline cambió; inspecciona y planifica de nuevo")
    version = current_version(session, project)
    kind, params = str(body.get("type") or ""), dict(body.get("parameters") or {})

    # Pre-dispatch validation for replace_asset
    if kind == "replace_asset":
        clip_id = str(params.get("clip_id") or "")
        new_asset_id = str(params.get("asset_id") or "")
        if not clip_id or not new_asset_id:
            raise error(400, "INVALID_OPERATION", "replace_asset requires clip_id and asset_id")
        # Find the clip in current timeline
        target_clip = None
        target_track = None
        for track in version.timeline.get("tracks", []):
            for clip in track.get("clips", []):
                if clip["id"] == clip_id:
                    target_clip = clip
                    target_track = track
                    break
            if target_clip:
                break
        if not target_clip:
            raise error(404, "CLIP_NOT_FOUND", "Clip no encontrado en la timeline actual")
        if target_track.get("locked"):
            raise error(409, "TRACK_LOCKED", "La pista del clip está bloqueada")
        if target_clip.get("kind") in {"caption", "text"}:
            raise error(400, "INVALID_OPERATION", "No se puede reemplazar asset en clip de texto/caption")
        # Validate target asset
        target_asset = session.get(EditorAsset, new_asset_id)
        if not target_asset or target_asset.project_id != project.id:
            raise error(404, "ASSET_NOT_FOUND", "Asset de destino no encontrado en este proyecto")
        prov = read_asset_provenance(target_asset.asset_metadata)
        if prov is None:
            raise error(400, "INVALID_ASSET", "Asset de destino no tiene provenance válida")
        if prov.origin != AssetOrigin.ENHANCED:
            raise error(400, "INVALID_ASSET", "Solo assets ENHANCED pueden usarse para replace_asset")
        if prov.parent_asset_id != target_clip["asset_id"]:
            raise error(409, "LINEAGE_MISMATCH", "El asset ENHANCED no deriva del asset actual del clip")
        if (target_asset.duration_us or 0) < target_clip["source_out_us"]:
            raise error(400, "DURATION_INSUFFICIENT", "El asset ENHANCED es más corto que el rango del clip")

    try:
        result = _dispatch(version.timeline, kind, params)
    except tl.TimelineError as exc:
        raise error(400, exc.code, str(exc)) from exc
    next_number = project.current_version + 1
    created_by = str(body.get("generated_by") or "user")
    op = EditorOperation(
        project_id=project.id,
        version_number=next_number,
        idempotency_key=key,
        kind=kind,
        generated_by=created_by,
        payload=params,
    )
    session.add(
        EditorVersion(
            project_id=project.id,
            number=next_number,
                parent_number=project.current_version,
            timeline=result,
            created_by=created_by,
            summary=kind,
        )
    )
    session.add(op)
    project.current_version, project.updated_at = next_number, datetime.now(UTC)
    session.add(project)
    session.commit()
    session.refresh(op)
    return {"operation_id": op.id, "version": next_number, "idempotent": False, "timeline": result}


def _dispatch(timeline: dict, kind: str, p: dict) -> dict:
    aliases = {
        "set_volume": ("audio", "volume"),
        "set_mute": ("audio", "muted"),
        "set_fade": ("audio", None),
        "set_crop": ("transform", "crop"),
        "set_position": ("transform", None),
        "set_scale": ("transform", "scale"),
        "change_speed": ("rate", "playback_rate"),
    }
    if kind in aliases:
        group, field = aliases[kind]
        if group == "rate":
            return tl.set_clip_properties(
                timeline, p["clip_id"], playback_rate=float(p["playback_rate"])
            )
        values = dict(p.get(group) or {})
        if field and field in p:
            values[field] = p[field]
        return tl.set_clip_properties(timeline, p["clip_id"], **{group: values})
    if kind == "split":
        return tl.split_clip(timeline, p["clip_id"], int(p["at_us"]))
    if kind == "trim":
        return tl.trim_clip(
            timeline,
            p["clip_id"],
            source_in_us=p.get("source_in_us"),
            source_out_us=p.get("source_out_us"),
        )
    if kind == "move":
        return tl.move_clip(timeline, p["clip_id"], p["track_id"], int(p["timeline_start_us"]))
    if kind == "delete":
        return tl.delete_clip(timeline, p["clip_id"], ripple=bool(p.get("ripple")))
    if kind == "duplicate":
        return tl.duplicate_clip(timeline, p["clip_id"], p["track_id"], int(p["timeline_start_us"]))
    if kind == "insert":
        return tl.add_clip(
            timeline,
            p["track_id"],
            tl.new_clip(
                p["asset_id"],
                int(p["source_in_us"]),
                int(p["source_out_us"]),
                int(p["timeline_start_us"]),
                linked_group=p.get("linked_group"),
            ),
        )
    if kind == "auto_import":
        clips = list(p.get("clips") or [])
        if not clips:
            raise tl.InvalidRange("Auto import requires at least one source range")
        video_track = next(
            (track for track in timeline.get("tracks", []) if track["type"] == "video"), None
        )
        if video_track is None:
            raise tl.TimelineError("Video track not found")
        result = timeline
        cursor = 0
        for item in clips:
            source_in_us, source_out_us = int(item["source_in_us"]), int(item["source_out_us"])
            clip = tl.new_clip(
                str(p["asset_id"]),
                source_in_us,
                source_out_us,
                cursor,
                provenance=dict(item.get("provenance") or {}),
            )
            result = tl.add_clip(result, video_track["id"], clip)
            cursor += tl.clip_duration_us(clip)
        return result
    if kind == "track_lock":
        return tl.set_track_properties(timeline, p["track_id"], locked=bool(p["locked"]))
    if kind == "track_properties":
        return tl.set_track_properties(timeline, p["track_id"], **p.get("properties", {}))
    if kind == "clip_properties":
        return tl.set_clip_properties(timeline, p["clip_id"], **p.get("properties", {}))
    if kind == "replace_asset":
        clip_id = str(p.get("clip_id") or "")
        new_asset_id = str(p.get("asset_id") or "")
        if not clip_id or not new_asset_id:
            raise tl.TimelineError("replace_asset requires clip_id and asset_id")
        # Validation will happen in apply_operation before dispatch
        return tl.replace_clip_asset(timeline, clip_id, new_asset_id)
    if kind in {"add_caption", "add_text"}:
        content_track = p.get("track_id") or next(
            (t["id"] for t in timeline.get("tracks", []) if t["type"] == "text"), None
        )
        if not content_track:
            raise tl.TimelineError("Text track not found")
        factory = tl.new_caption if kind == "add_caption" else tl.new_text
        return tl.add_timed_content(
            timeline,
            content_track,
            factory(
                p["text"],
                int(p["start_us"]),
                int(p["end_us"]),
                position=p.get("position"),
                style=p.get("style"),
                provenance=p.get("provenance", "USER"),
            ),
        )
    if kind in {"edit_caption", "edit_text"}:
        return tl.edit_timed_content(
            timeline,
            p["item_id"],
            text=p.get("text"),
            start_us=p.get("start_us"),
            end_us=p.get("end_us"),
            position=p.get("position"),
            style=p.get("style"),
        )
    raise tl.TimelineError("CAPABILITY_UNAVAILABLE")


def restore_version(
    session: Session, project_id: str, number: int, *, generated_by: str = "user"
) -> dict:
    project = project_or_404(session, project_id)
    target = session.exec(
        select(EditorVersion).where(
            EditorVersion.project_id == project.id, EditorVersion.number == number
        )
    ).first()
    if not target:
        raise error(404, "VERSION_NOT_FOUND", "Versión no encontrada")
    next_number = project.current_version + 1
    snapshot = copy.deepcopy(target.timeline)
    session.add(
        EditorVersion(
            project_id=project.id,
            number=next_number,
            # Restores are themselves versions, but their undo parent is the
            # restored snapshot's parent. This keeps repeated Undo moving back.
            parent_number=target.parent_number,
            timeline=snapshot,
            created_by=generated_by,
            summary=f"restore version {number}",
        )
    )
    session.add(
        EditorOperation(
            project_id=project.id,
            version_number=next_number,
            idempotency_key=f"restore:{number}:{project.current_version}",
            kind="restore",
            generated_by=generated_by,
            payload={"number": number},
        )
    )
    project.current_version = next_number
    project.updated_at = datetime.now(UTC)
    session.add(project)
    session.commit()
    return {"version": next_number, "timeline": snapshot, "restored_from": number}


def undo(session: Session, project_id: str, *, generated_by: str = "user") -> dict:
    project = project_or_404(session, project_id)
    current = current_version(session, project)
    if current.parent_number is None:
        raise error(409, "HISTORY_EMPTY", "No hay una versión anterior para restaurar")
    return restore_version(session, project_id, current.parent_number, generated_by=generated_by)
