"""Endpoints del compositor: biblioteca de clips disponibles y compilaciones."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.core.db import get_session
from app.models import Clip, ClipAsset, Compilation, CompilationStatus, Job
from app.pipeline.compose import TRANSITIONS, compilation_dir, normalize_transition

router = APIRouter(tags=["compositions"])

MAX_CLIPS_PER_COMPILATION = 30


class CompilationCreate(BaseModel):
    asset_ids: list[str]
    title: str = ""
    transition: str = "none"
    transition_seconds: float = 0.4
    bgm_path: str = ""
    bgm_volume: float = 0.18
    bgm_ducking: bool = True
    bgm_auto_level: bool = True
    bgm_duck_lu: float = 12.0


def _validated_bgm_path(raw: str) -> str:
    """Valida la ruta de la musica de fondo y la devuelve normalizada.

    Rechaza rutas de red (UNC `\\\\servidor\\share` o `//servidor/share`):
    `Path.exists()` las da por buenas y FFmpeg las abriria por SMB, lo que en
    Windows genera una conexion saliente que filtra credenciales NTLM al host
    remoto. Snipra es local por diseño: la musica sale de un archivo del disco,
    nunca de la red.
    """
    candidate = raw.strip()
    if not candidate:
        return ""
    if candidate.startswith(("\\\\", "//")) or Path(candidate).drive.startswith("\\\\"):
        raise HTTPException(400, "La musica debe ser un archivo local, no una ruta de red.")
    try:
        resolved = Path(candidate).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(400, "La ruta del audio de fondo no existe") from exc
    if not resolved.is_file():
        raise HTTPException(400, "La ruta del audio de fondo no es un archivo")
    return str(resolved)


def _serialize(compilation: Compilation) -> dict:
    return {
        "id": compilation.id,
        "created_at": compilation.created_at.isoformat(),
        "updated_at": compilation.updated_at.isoformat(),
        "title": compilation.title,
        "format": compilation.format,
        "asset_ids": list(compilation.asset_ids or []),
        "clip_count": len(compilation.asset_ids or []),
        "transition": compilation.transition,
        "transition_seconds": compilation.transition_seconds,
        "bgm_path": compilation.bgm_path,
        "bgm_volume": compilation.bgm_volume,
        "bgm_ducking": compilation.bgm_ducking,
        "bgm_auto_level": compilation.bgm_auto_level,
        "bgm_duck_lu": compilation.bgm_duck_lu,
        "audio_report": compilation.audio_report or {},
        "status": compilation.status.value,
        "progress": compilation.progress,
        "error": compilation.error,
        "duration": compilation.duration,
        "width": compilation.width,
        "height": compilation.height,
        "size_bytes": compilation.size_bytes,
    }


@router.get("/composable-clips")
def list_composable_clips(session: Session = Depends(get_session)) -> list[dict]:
    """Clips renderizados de toda la biblioteca, agrupados por asset.

    Un asset (no un clip) es la unidad componible: un mismo clip puede existir en
    9x16 y 16x9, y solo se pueden concatenar assets del mismo formato.
    """
    rows = session.exec(
        select(ClipAsset, Clip, Job)
        .join(Clip, Clip.id == ClipAsset.clip_id)
        .join(Job, Job.id == Clip.job_id)
        .order_by(Job.created_at.desc(), Clip.index)
    ).all()

    items: list[dict] = []
    for asset, clip, job in rows:
        if not Path(asset.path).exists():
            continue  # el job pudo borrarse o la retencion pudo limpiarlo
        items.append(
            {
                "asset_id": asset.id,
                "clip_id": clip.id,
                "job_id": job.id,
                "job_title": job.title or job.source_name,
                "clip_index": clip.index,
                "title": clip.title,
                "hook": clip.hook,
                "score": clip.score,
                "format": asset.format,
                "duration": asset.duration,
                "width": asset.width,
                "height": asset.height,
                "size_bytes": asset.size_bytes,
                "video_url": f"/api/assets/{asset.id}/video",
                "thumb_url": f"/api/assets/{asset.id}/thumb" if asset.thumb_path else None,
            }
        )
    return items


@router.get("/compositions")
def list_compositions(session: Session = Depends(get_session)) -> list[dict]:
    compilations = session.exec(select(Compilation).order_by(Compilation.created_at.desc())).all()
    return [_serialize(item) for item in compilations]


@router.get("/compositions/{compilation_id}")
def get_composition(compilation_id: str, session: Session = Depends(get_session)) -> dict:
    compilation = session.get(Compilation, compilation_id)
    if compilation is None:
        raise HTTPException(404, "Compilacion no encontrada")
    return _serialize(compilation)


@router.post("/compositions")
def create_composition(body: CompilationCreate, session: Session = Depends(get_session)) -> dict:
    asset_ids = [item.strip() for item in body.asset_ids if item and item.strip()]
    if len(asset_ids) < 2:
        raise HTTPException(400, "Elige al menos 2 clips para componer")
    if len(asset_ids) > MAX_CLIPS_PER_COMPILATION:
        raise HTTPException(400, f"Maximo {MAX_CLIPS_PER_COMPILATION} clips por compilacion")
    if len(set(asset_ids)) != len(asset_ids):
        raise HTTPException(400, "Hay clips repetidos en la seleccion")

    assets = session.exec(select(ClipAsset).where(ClipAsset.id.in_(asset_ids))).all()
    by_id = {asset.id: asset for asset in assets}
    missing = [asset_id for asset_id in asset_ids if asset_id not in by_id]
    if missing:
        raise HTTPException(404, f"{len(missing)} clip(s) no existen")

    # Mezclar formatos obligaria a recortar o deformar: se rechaza explicitamente.
    formats = {by_id[asset_id].format for asset_id in asset_ids}
    if len(formats) > 1:
        raise HTTPException(
            400,
            "Todos los clips deben ser del mismo formato. Seleccionados: "
            + ", ".join(sorted(formats)),
        )

    if body.transition not in TRANSITIONS:
        raise HTTPException(400, f"Transicion invalida. Usa: {', '.join(sorted(TRANSITIONS))}")

    bgm_path = _validated_bgm_path(body.bgm_path)

    compilation = Compilation(
        title=body.title.strip(),
        format=next(iter(formats)),
        asset_ids=asset_ids,
        transition=normalize_transition(body.transition),
        transition_seconds=max(0.1, min(3.0, body.transition_seconds)),
        bgm_path=bgm_path,
        bgm_volume=max(0.0, min(1.0, body.bgm_volume)),
        bgm_ducking=body.bgm_ducking,
        bgm_auto_level=body.bgm_auto_level,
        bgm_duck_lu=max(6.0, min(24.0, body.bgm_duck_lu)),
        status=CompilationStatus.queued,
    )
    session.add(compilation)
    session.commit()
    session.refresh(compilation)
    return _serialize(compilation)


@router.post("/compositions/{compilation_id}/retry")
def retry_composition(compilation_id: str, session: Session = Depends(get_session)) -> dict:
    compilation = session.get(Compilation, compilation_id)
    if compilation is None:
        raise HTTPException(404, "Compilacion no encontrada")
    if compilation.status == CompilationStatus.rendering:
        raise HTTPException(409, "La compilacion se esta renderizando ahora mismo")
    compilation.status = CompilationStatus.queued
    compilation.progress = 0.0
    compilation.error = None
    session.add(compilation)
    session.commit()
    session.refresh(compilation)
    return _serialize(compilation)


@router.get("/compositions/{compilation_id}/video")
def composition_video(compilation_id: str, session: Session = Depends(get_session)):
    compilation = session.get(Compilation, compilation_id)
    if compilation is None:
        raise HTTPException(404, "Compilacion no encontrada")
    if compilation.status != CompilationStatus.done or not compilation.output_path:
        raise HTTPException(409, "La compilacion aun no esta lista")
    path = Path(compilation.output_path)
    if not path.exists():
        raise HTTPException(404, "El archivo de la compilacion no existe")
    safe_title = "".join(
        char for char in compilation.title if char.isalnum() or char in " -_"
    ).strip()
    filename = f"snipra-{safe_title or 'compilacion'}-{compilation.id[:8]}.mp4"
    return FileResponse(path, media_type="video/mp4", filename=filename)


@router.get("/compositions/{compilation_id}/thumb")
def composition_thumb(compilation_id: str, session: Session = Depends(get_session)):
    compilation = session.get(Compilation, compilation_id)
    if compilation is None:
        raise HTTPException(404, "Compilacion no encontrada")
    thumb = compilation_dir(compilation_id) / "thumb.jpg"
    if not thumb.exists():
        raise HTTPException(404, "Sin thumbnail")
    return FileResponse(thumb, media_type="image/jpeg")


@router.delete("/compositions/{compilation_id}")
def delete_composition(compilation_id: str, session: Session = Depends(get_session)) -> dict:
    compilation = session.get(Compilation, compilation_id)
    if compilation is None:
        raise HTTPException(404, "Compilacion no encontrada")
    if compilation.status == CompilationStatus.rendering:
        raise HTTPException(409, "No se puede borrar mientras se renderiza")

    # Solo se borra la salida de la compilacion; los clips originales quedan
    # intactos en su job (esta tabla nunca es dueña de ellos). rmtree y no
    # iterdir+unlink+rmdir: si algo dejara un subdirectorio ahi, unlink sobre
    # una carpeta reventaria el endpoint con un 500.
    shutil.rmtree(compilation_dir(compilation_id), ignore_errors=True)

    session.delete(compilation)
    session.commit()
    return {"ok": True}
