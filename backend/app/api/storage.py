"""Endpoints de storage: diagnostico de espacio y limpieza segura."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import get_session
from app.models import Compilation, Job, JobStatus

router = APIRouter(prefix="/storage", tags=["storage"])


class CleanupRequest(BaseModel):
    remove_orphans: bool = True
    remove_canceled_failed_days: int | None = None


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _job_dirs() -> list[Path]:
    if not settings.work_dir.exists():
        return []
    return [item for item in settings.work_dir.iterdir() if item.is_dir()]


def _composition_dirs() -> list[Path]:
    if not settings.compositions_dir.exists():
        return []
    return [item for item in settings.compositions_dir.iterdir() if item.is_dir()]


def _orphan_composition_dirs(session: Session) -> list[Path]:
    known = {row.id for row in session.exec(select(Compilation)).all()}
    return [path for path in _composition_dirs() if path.name not in known]


@router.get("")
def storage_status(session: Session = Depends(get_session)) -> dict:
    jobs = session.exec(select(Job)).all()
    known_ids = {job.id for job in jobs}
    orphan_dirs = [path for path in _job_dirs() if path.name not in known_ids]
    compilations = session.exec(select(Compilation)).all()
    return {
        "work_bytes": _dir_size(settings.work_dir),
        "job_count": len(jobs),
        "orphan_count": len(orphan_dirs),
        "work_dir": str(settings.work_dir),
        # Las compilaciones se reportan aparte: quedan FUERA de la retencion
        # automatica a proposito (son la salida final del usuario, no artefactos
        # intermedios de un job), asi que si no se muestran aqui crecen invisibles.
        "compositions_bytes": _dir_size(settings.compositions_dir),
        "compilation_count": len(compilations),
        "orphan_composition_count": len(_orphan_composition_dirs(session)),
        "compositions_dir": str(settings.compositions_dir),
    }


@router.post("/cleanup")
def cleanup_storage(
    body: CleanupRequest,
    session: Session = Depends(get_session),
) -> dict:
    jobs = session.exec(select(Job)).all()
    known_ids = {job.id for job in jobs}
    removed_paths = 0
    removed_bytes = 0

    def remove_path(path: Path) -> None:
        nonlocal removed_paths, removed_bytes
        if not path.exists():
            return
        removed_bytes += _dir_size(path)
        shutil.rmtree(path, ignore_errors=True)
        removed_paths += 1

    if body.remove_orphans:
        for path in _job_dirs():
            if path.name not in known_ids:
                remove_path(path)
        # Carpetas de compilaciones cuya fila ya no existe (p.ej. si el proceso
        # murio entre borrar la fila y borrar el directorio). NO se tocan las
        # compilaciones vivas: esas solo se borran a mano desde Componer.
        for path in _orphan_composition_dirs(session):
            remove_path(path)

    if body.remove_canceled_failed_days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=max(0, body.remove_canceled_failed_days))
        for job in jobs:
            if job.status not in {JobStatus.canceled, JobStatus.failed}:
                continue
            updated_at = job.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
            if updated_at <= cutoff:
                remove_path(settings.work_dir / job.id)

    return {
        "removed_paths": removed_paths,
        "removed_bytes": removed_bytes,
        "status": storage_status(session),
    }
