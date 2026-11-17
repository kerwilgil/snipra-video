"""Cola persistente de compilaciones (varios clips unidos en un video nuevo).

Mismo patron que clip_operations: claim atomico via UPDATE condicional para que
dos workers no tomen la misma compilacion, y recover al arrancar para devolver a
la cola lo que quedo a medias si el proceso murio.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text
from sqlmodel import Session, select

from app.core.db import engine
from app.models import ClipAsset, Compilation, CompilationStatus
from app.pipeline import compose


def claim_next_compilation() -> str | None:
    with Session(engine) as session:
        compilation = session.exec(
            select(Compilation)
            .where(Compilation.status == CompilationStatus.queued)
            .order_by(Compilation.created_at)
        ).first()
        if compilation is None:
            return None
        compilation_id = compilation.id

    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE compilation SET status = 'rendering', updated_at = :updated "
                "WHERE id = :id AND status = 'queued'"
            ),
            {"id": compilation_id, "updated": datetime.now(UTC).isoformat()},
        )
    return compilation_id if result.rowcount == 1 else None


def _resolve_asset_paths(session: Session, asset_ids: list[str]) -> list[str]:
    """Resuelve los ids a rutas conservando el orden elegido por el usuario."""
    assets = session.exec(select(ClipAsset).where(ClipAsset.id.in_(asset_ids))).all()
    by_id = {asset.id: asset for asset in assets}
    missing = [asset_id for asset_id in asset_ids if asset_id not in by_id]
    if missing:
        raise compose.CompileError(
            f"{len(missing)} clip(s) de la compilacion ya no existen en la biblioteca"
        )
    return [by_id[asset_id].path for asset_id in asset_ids]


def process_compilation(compilation_id: str) -> None:
    with Session(engine) as session:
        compilation = session.get(Compilation, compilation_id)
        if compilation is None or compilation.status != CompilationStatus.rendering:
            return

        asset_ids = list(compilation.asset_ids or [])
        if not asset_ids:
            raise compose.CompileError("La compilacion no tiene clips seleccionados")

        paths = _resolve_asset_paths(session, asset_ids)
        compilation.progress = 0.1
        compilation.updated_at = datetime.now(UTC)
        session.add(compilation)
        session.commit()

        result = compose.render_compilation(
            compilation_id,
            paths,
            transition=compilation.transition,
            transition_seconds=compilation.transition_seconds,
            bgm_path=compilation.bgm_path or None,
            bgm_volume=compilation.bgm_volume,
            bgm_ducking=compilation.bgm_ducking,
            bgm_auto_level=compilation.bgm_auto_level,
            bgm_duck_lu=compilation.bgm_duck_lu,
        )

        compilation.output_path = str(result.path)
        compilation.duration = result.duration
        compilation.width = result.width
        compilation.height = result.height
        compilation.size_bytes = result.size_bytes
        compilation.audio_report = result.audio_report
        compilation.status = CompilationStatus.done
        compilation.progress = 1.0
        compilation.error = None
        compilation.updated_at = datetime.now(UTC)
        session.add(compilation)
        session.commit()


def fail_compilation(compilation_id: str, message: str) -> None:
    from app.core.sanitize import sanitize_error_text

    message = sanitize_error_text(message)
    with Session(engine) as session:
        compilation = session.get(Compilation, compilation_id)
        if compilation is None:
            return
        compilation.status = CompilationStatus.failed
        compilation.error = message[:2000]
        compilation.updated_at = datetime.now(UTC)
        session.add(compilation)
        session.commit()


def recover_compilations() -> int:
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE compilation SET status = 'queued', progress = 0, "
                "updated_at = :updated WHERE status = 'rendering'"
            ),
            {"updated": datetime.now(UTC).isoformat()},
        )
    return result.rowcount
