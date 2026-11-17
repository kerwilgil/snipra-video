"""Retencion segura de artefactos locales de jobs terminados."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta

from sqlmodel import Session, select

from app.core.config import settings
from app.models import ClipOperation, ClipOperationStatus, Job, JobStatus

TERMINAL_JOB_STATUSES = {JobStatus.done, JobStatus.failed, JobStatus.canceled}
ACTIVE_OPERATION_STATUSES = {ClipOperationStatus.queued, ClipOperationStatus.running}


def cleanup_expired_job_files(session: Session, retention_days: int) -> dict[str, int]:
    if retention_days <= 0:
        return {"removed_paths": 0, "removed_bytes": 0}

    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    protected_job_ids = set(
        session.exec(
            select(ClipOperation.job_id).where(ClipOperation.status.in_(ACTIVE_OPERATION_STATUSES))
        ).all()
    )
    jobs = session.exec(select(Job).where(Job.status.in_(TERMINAL_JOB_STATUSES))).all()
    removed_paths = 0
    removed_bytes = 0
    for job in jobs:
        updated_at = job.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)
        if updated_at > cutoff or job.id in protected_job_ids:
            continue
        path = settings.work_dir / job.id
        if not path.exists():
            continue
        removed_bytes += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
        shutil.rmtree(path, ignore_errors=True)
        removed_paths += 1
    return {"removed_paths": removed_paths, "removed_bytes": removed_bytes}
