"""Cola persistente de ediciones masivas de clips."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlmodel import Session, select

from app.api.clips import _rerender_from_artifacts, _rewrite_one, _translate_one
from app.core.db import engine
from app.core.job_state import add_job_log
from app.models import Clip, ClipOperation, ClipOperationStatus, Job


def claim_next_operation() -> str | None:
    with Session(engine) as session:
        operation = session.exec(
            select(ClipOperation)
            .where(ClipOperation.status == ClipOperationStatus.queued)
            .order_by(ClipOperation.created_at)
        ).first()
        if operation is None:
            return None
        operation_id = operation.id

    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE clipoperation SET status = 'running', updated_at = :updated "
                "WHERE id = :id AND status = 'queued'"
            ),
            {"id": operation_id, "updated": datetime.now(UTC).isoformat()},
        )
    return operation_id if result.rowcount == 1 else None


def process_operation(operation_id: str) -> None:
    with Session(engine) as session:
        operation = session.get(ClipOperation, operation_id)
        if operation is None or operation.status != ClipOperationStatus.running:
            return
        job = session.get(Job, operation.job_id)
        if job is None:
            raise RuntimeError("Job de la operacion no encontrado")
        clips = session.exec(select(Clip).where(Clip.job_id == job.id).order_by(Clip.index)).all()
        if not clips:
            raise RuntimeError("El job no tiene clips para editar")

        payload = operation.payload or {}
        total = len(clips)
        for index, clip in enumerate(clips, start=1):
            if operation.kind == "style":
                _rerender_from_artifacts(
                    session,
                    job,
                    clip,
                    style_config=payload.get("style_config"),
                )
            elif operation.kind == "subtitles":
                _rerender_from_artifacts(
                    session,
                    job,
                    clip,
                    style_config=payload.get("style_config"),
                    subtitles=payload.get("subtitles"),
                )
            elif operation.kind == "rewrite_captions":
                _rewrite_one(
                    session,
                    job,
                    clip,
                    provider=str(payload.get("provider") or job.provider),
                    model=str(payload.get("model") or job.model),
                )
            elif operation.kind == "translate_captions":
                _translate_one(
                    session,
                    job,
                    clip,
                    target_language=str(payload.get("target_language") or ""),
                    provider=str(payload.get("provider") or job.provider),
                    model=str(payload.get("model") or job.model),
                )
            else:
                raise RuntimeError(f"Operacion de clips no soportada: {operation.kind}")

            operation.progress = round(index / total, 3)
            operation.updated_at = datetime.now(UTC)
            session.add(operation)
            session.commit()

        operation.status = ClipOperationStatus.done
        operation.progress = 1.0
        operation.error = None
        operation.updated_at = datetime.now(UTC)
        add_job_log(session, job.id, f"Operacion masiva '{operation.kind}' completada")
        session.add(operation)
        session.commit()


def fail_operation(operation_id: str, message: str) -> None:
    from app.core.sanitize import sanitize_error_text

    message = sanitize_error_text(message)
    with Session(engine) as session:
        operation = session.get(ClipOperation, operation_id)
        if operation is None:
            return
        operation.status = ClipOperationStatus.failed
        operation.error = message[:2000]
        operation.updated_at = datetime.now(UTC)
        add_job_log(session, operation.job_id, message, level="error")
        session.add(operation)
        session.commit()


def recover_operations() -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE clipoperation SET status = 'queued', progress = 0, "
                "updated_at = :updated WHERE status = 'running'"
            ),
            {"updated": datetime.now(UTC).isoformat()},
        )
    return result.rowcount
