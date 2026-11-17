"""Coordinacion ligera de cancelacion entre API y worker local."""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings


def cancel_marker(job_id: str) -> Path:
    return settings.work_dir / job_id / ".cancel-request"


def request_cancel(job_id: str) -> None:
    marker = cancel_marker(job_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch(exist_ok=True)


def clear_cancel_request(job_id: str) -> None:
    cancel_marker(job_id).unlink(missing_ok=True)


def is_cancel_requested(job_id: str) -> bool:
    return cancel_marker(job_id).exists()
