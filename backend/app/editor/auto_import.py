"""Deterministic bridge from an existing Auto result into Editor IA.

Auto remains read-only here: we consume its persisted Clip/ClipAsset evidence and
create a normal versioned Editor project.  Source mappings are never guessed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select

from app.editor import service
from app.editor.models import EditorProject
from app.models import Clip, ClipAsset, Job
from app.pipeline import ingest


def _us(seconds: object) -> int:
    return round(float(seconds) * 1_000_000)


def _ranges(clip: Clip) -> list[tuple[int, int]]:
    raw = clip.source_ranges or []
    if raw:
        parsed: list[tuple[int, int]] = []
        for item in raw:
            try:
                start, end = _us(item["start"]), _us(item["end"])
            except (KeyError, TypeError, ValueError):
                return []
            if start < 0 or end <= start:
                return []
            parsed.append((start, end))
        return sorted(parsed)
    try:
        start, end = _us(clip.start), _us(clip.end)
    except (TypeError, ValueError):
        return []
    return [(start, end)] if start >= 0 and end > start else []


def _source(job: Job) -> Path | None:
    try:
        source = ingest.find_source(job.id).resolve()
    except FileNotFoundError:
        return None
    return source if source.is_file() else None


def _rendered_asset(session: Session, clip: Clip) -> ClipAsset | None:
    assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip.id)).all()
    for asset in sorted(assets, key=lambda item: (item.format != "9x16", item.id)):
        if Path(asset.path).is_file() and asset.duration > 0:
            return asset
    return None


def open_auto_result(session: Session, clip_id: str) -> tuple[EditorProject, bool]:
    """Return the canonical editable project for an Auto Clip and whether it was created."""
    existing = session.exec(
        select(EditorProject).where(EditorProject.source_clip_id == clip_id)
    ).first()
    if existing:
        return existing, False

    clip = session.get(Clip, clip_id)
    if clip is None:
        raise service.error(404, "AUTO_RESULT_NOT_FOUND", "Resultado Auto no encontrado")
    job = session.get(Job, clip.job_id)
    if job is None:
        raise service.error(404, "AUTO_SOURCE_NOT_FOUND", "Fuente del resultado Auto no encontrada")

    ranges, source = _ranges(clip), _source(job)
    use_source = bool(ranges and source)
    rendered = None if use_source else _rendered_asset(session, clip)
    if not use_source and rendered is None:
        raise service.error(
            409,
            "AUTO_IMPORT_UNAVAILABLE",
            "El resultado no conserva un mapping fuente válido ni un render disponible",
        )

    project = service.create_project(
        session,
        clip.title or f"Clip {clip.index + 1}",
        source_job_id=job.id,
        source_clip_id=clip.id,
    )
    imported_at = datetime.now(UTC).isoformat()
    mode = "SOURCE_RANGES" if use_source else "RENDERED_FALLBACK"
    if use_source:
        assert source is not None
        asset = service.add_asset(
            session,
            project,
            source,
            duration_us=_us(job.duration or 0),
            width=job.source_width,
            height=job.source_height,
            has_audio=True,
            asset_metadata={
                "origin": "AUTO_IMPORT",
                "source_job_id": job.id,
                "source_kind": "ORIGINAL",
            },
        )
        imported_ranges = ranges
    else:
        assert rendered is not None
        asset = service.add_asset(
            session,
            project,
            Path(rendered.path),
            duration_us=_us(rendered.duration),
            width=rendered.width,
            height=rendered.height,
            has_audio=True,
            asset_metadata={
                "origin": "AUTO_IMPORT",
                "source_job_id": job.id,
                "source_kind": "RENDERED_FALLBACK",
                "flattened": True,
            },
        )
        imported_ranges = [(0, _us(rendered.duration))]

    clips = []
    for source_in_us, source_out_us in imported_ranges:
        provenance = {
            "origin": "AUTO_IMPORT",
            "source_job_id": job.id,
            "source_asset_id": asset.id,
            "output_type": rendered.format if rendered else job.output_mode,
            "mode": job.processing_mode,
            "candidate_id": clip.id,
            "event_id": clip.id,
            "original_source_range": {"start_us": source_in_us, "end_us": source_out_us},
            "imported_at": imported_at,
            "auto_metadata_version": 1,
            "reconstruction": mode,
            "flattened": not use_source,
        }
        clips.append(
            {
                "source_in_us": source_in_us,
                "source_out_us": source_out_us,
                "provenance": provenance,
            }
        )

    service.apply_operation(
        session,
        project.id,
        {
            "type": "auto_import",
            "parameters": {"asset_id": asset.id, "clips": clips},
            "base_version": 0,
            "idempotency_key": f"auto-import:{clip.id}",
            "generated_by": "auto_import",
        },
    )
    project.agent_metadata = {
        **project.agent_metadata,
        "auto_import": {"mode": mode, "flattened": not use_source, "source_clip_id": clip.id},
    }
    session.add(project)
    session.commit()
    session.refresh(project)
    return project, True
