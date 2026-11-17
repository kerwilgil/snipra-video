from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import get_session
from app.editor import (
    agent,
    asset_task_service,
    auto_import,
    capabilities,
    director,
    events,
    generation_task_service,
    media_artifacts,
    service,
    timeline,
    transcript,
    verification,
)
from app.editor.agent import EditorialPlan
from app.editor.asset_tasks import AssetTaskStatus, EnhancementValidationError
from app.editor.director_bridge import BridgeValidationError, validate_director_evidence
from app.editor.enhancement import EnhancementProfile, EnhancementRequest
from app.editor.generation import (
    GenerationAspectRatio,
    GenerationMode,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResolution,
    LocalGenerationUnavailable,
    UnsupportedGenerationProvider,
    get_generation_provider_capabilities,
)
from app.editor.metadata_generator import (
    MetadataGenerationError,
    generate_metadata_for_montages,
    generate_metadata_for_shorts,
)
from app.editor.models import (
    AssetTask,
    EditorAgentRun,
    EditorAsset,
    EditorProject,
    EditorVersion,
    new_id,
)
from app.editor.render import RenderError, render_timeline
from app.editor.short_models import MetadataGenerationRequest
from app.editor.short_orchestrator import MontageBatchRequest, ShortBatchRequest
from app.editor.short_orchestrator import create_shorts as orchestrate_shorts

router = APIRouter(prefix="/editor/projects", tags=["editor"])


class ProjectCreate(BaseModel):
    name: str = "Untitled project"
    source_job_id: str | None = None


class ProjectUpdate(BaseModel):
    name: str


class OperationCreate(BaseModel):
    base_version: int
    idempotency_key: str
    type: str
    parameters: dict[str, Any] = {}
    generated_by: str = "user"


class AgentRequest(BaseModel):
    request: str
    intent: str | None = None
    execute: bool = True
    idempotency_key: str = ""


class EnhancementPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: EnhancementProfile = EnhancementProfile.DEFAULT
    target_width: int | None = Field(default=None, ge=16, le=15360)
    target_height: int | None = Field(default=None, ge=16, le=15360)
    upscale_factor: float | None = Field(default=None, ge=1.0, le=8.0)
    sharpen_strength: float | None = Field(default=None, ge=0.0, le=3.0)
    contrast: float | None = Field(default=None, ge=0.0, le=3.0)
    brightness: float | None = Field(default=None, ge=-1.0, le=1.0)
    saturation: float | None = Field(default=None, ge=0.0, le=3.0)
    denoise_strength: float | None = Field(default=None, ge=0.0, le=3.0)
    preserve_audio: bool = True


class EnhancementApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_hash: str


# --- Generation API Models (SNIPRA 1.6D Block 04) ---


class GenerationPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: GenerationMode = GenerationMode.TEXT_TO_VIDEO
    provider: GenerationProvider = GenerationProvider.LOCAL_LTX
    model: GenerationModel = GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8

    prompt_intent: str = Field(min_length=1, max_length=4000)
    visual_direction: str | None = None
    camera_direction: str | None = None
    avoid_constraints: str | None = None

    duration_seconds: int = Field(ge=1, le=30)
    aspect_ratio: GenerationAspectRatio = GenerationAspectRatio.RATIO_16_9
    resolution: GenerationResolution = GenerationResolution.SD

    reference_asset_ids: list[str] = Field(default_factory=list)
    grounding_asset_ids: list[str] = Field(default_factory=list)

    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    generation_parameters: dict = Field(default_factory=dict)


class GenerationApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_hash: str


def _serialize_asset_task(task: AssetTask, session: Session) -> dict:
    result = {
        "id": task.id,
        "project_id": task.project_id,
        "kind": task.kind,
        "status": task.status,
        "provider": task.provider,
        "model": task.model,
        "request": task.request,
        "request_hash": task.request_hash,
        "cost_estimate": task.cost_estimate,
        "approval_state": task.approval_state,
        "approved_request_hash": task.approved_request_hash,
        "progress": task.progress,
        "output_asset_id": task.output_asset_id,
        "error": task.error,
        "cancel_requested": task.cancel_requested,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
    }
    if task.output_asset_id:
        output_asset = session.get(EditorAsset, task.output_asset_id)
        if output_asset:
            result["output_asset"] = service.serialize_asset(output_asset)
    return result


@router.get("/runtime/capabilities")
def runtime_capabilities() -> list[dict]:
    return capabilities.public_capabilities()


@router.get("/runtime/director-context")
def director_context(intent: str | None = None) -> dict:
    context = director.load_director_context(intent)
    return {
        "skill_name": context["skill_name"],
        "skill_fingerprint": context["skill_fingerprint"],
        "loaded_references": list(context["references"]),
    }


@router.post("")
def create(body: ProjectCreate, session: Session = Depends(get_session)) -> dict:
    return service.serialize_project(
        session, service.create_project(session, body.name, body.source_job_id)
    )


@router.post("/from-auto/{clip_id}")
def open_from_auto(clip_id: str, session: Session = Depends(get_session)) -> dict:
    project, created = auto_import.open_auto_result(session, clip_id)
    return {**service.serialize_project(session, project), "created": created}


@router.get("")
def list_projects(session: Session = Depends(get_session)) -> list[dict]:
    return [
        service.serialize_project_listing(session, p)
        for p in session.exec(select(EditorProject).order_by(EditorProject.updated_at.desc())).all()
    ]


@router.get("/{project_id}")
def get(project_id: str, session: Session = Depends(get_session)) -> dict:
    return service.serialize_project(session, service.project_or_404(session, project_id))


@router.patch("/{project_id}")
def rename(project_id: str, body: ProjectUpdate, session: Session = Depends(get_session)) -> dict:
    renamed = service.rename_project(session, project_id, body.name)
    return service.serialize_project(session, renamed)


@router.delete("/{project_id}")
def delete(project_id: str, session: Session = Depends(get_session)) -> dict:
    service.delete_project(session, project_id)
    return {"ok": True}


@router.get("/{project_id}/events")
def project_events(
    project_id: str, intent: str | None = None, session: Session = Depends(get_session)
) -> dict:
    project = service.project_or_404(session, project_id)
    discovered = events.events_for_job(session, project.source_job_id)
    return {"events": discovered, "best_event_pool": events.best_event_pool(discovered, intent)}


@router.get("/{project_id}/transcript")
def project_transcript(
    project_id: str,
    start_us: int = 0,
    end_us: int = 0,
    session: Session = Depends(get_session),
) -> dict:
    project = service.project_or_404(session, project_id)
    if end_us <= start_us:
        end_us = service.current_version(session, project).timeline.get("duration_us", 0)
    return {"segments": transcript.get_range(project.source_job_id, start_us, end_us)}


@router.get("/{project_id}/verify")
def verify_project(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = service.project_or_404(session, project_id)
    version = service.current_version(session, project)
    asset_ids = {asset.id for asset in service.project_assets(session, project.id)}
    return verification.verify(version.timeline, asset_ids)


@router.post("/{project_id}/assets/upload")
def upload_asset(
    project_id: str,
    file: UploadFile,
    session: Session = Depends(get_session),
) -> dict:
    project = service.project_or_404(session, project_id)
    upload_dir = settings.editor_dir / project.id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary = upload_dir / f"{new_id()}.upload"
    with temporary.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            handle.write(chunk)
    try:
        asset = service.import_upload(session, project, temporary, file.filename or "media")
    finally:
        temporary.unlink(missing_ok=True)
    return service.serialize_asset(asset)


@router.delete("/{project_id}/assets/{asset_id}")
def remove_asset(project_id: str, asset_id: str, session: Session = Depends(get_session)) -> dict:
    service.remove_asset(session, project_id, asset_id)
    return {"ok": True, "asset_removed": True}


# ---- Enhancement endpoints (SNIPRA 1.6C Block 04) ----

@router.post("/{project_id}/assets/{asset_id}/enhancements")
def prepare_enhancement(
    project_id: str,
    asset_id: str,
    body: EnhancementPrepareRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Create an enhancement task in ESTIMATING -> AWAITING_APPROVAL.
    Does NOT execute FFmpeg. Returns task with cost estimate for explicit approval."""
    service.project_or_404(session, project_id)
    source_asset = service.asset_or_404(session, project_id, asset_id)

    try:
        request = EnhancementRequest(
            source_asset_id=source_asset.id,
            profile=body.profile,
            target_width=body.target_width,
            target_height=body.target_height,
            upscale_factor=body.upscale_factor,
            sharpen_strength=body.sharpen_strength,
            contrast=body.contrast,
            brightness=body.brightness,
            saturation=body.saturation,
            denoise_strength=body.denoise_strength,
            preserve_audio=body.preserve_audio,
        )
    except ValidationError as exc:
        from app.core.sanitize import sanitize_error_text
        errors = exc.errors()
        detail = "; ".join(
            f"{'.'.join(str(e) for e in err['loc'])}: {err['msg']}" for err in errors
        )
        raise service.error(422, "VALIDATION_ERROR", sanitize_error_text(detail)) from exc

    task = asset_task_service.create_enhancement_task(
        session, project_id=project_id, request=request
    )
    try:
        task = asset_task_service.attach_estimate(session, task.id)
    except EnhancementValidationError as exc:
        from app.core.sanitize import sanitize_error_text
        raise service.error(422, "VALIDATION_ERROR", sanitize_error_text(str(exc))) from exc

    return _serialize_asset_task(task, session)


@router.get("/{project_id}/enhancements/{task_id}")
def get_enhancement_task(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de mejora no encontrada")
    return _serialize_asset_task(task, session)


@router.get("/{project_id}/enhancements")
def list_enhancement_tasks(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    service.project_or_404(session, project_id)
    tasks = session.exec(
        select(AssetTask).where(
            AssetTask.project_id == project_id,
            AssetTask.kind == "ENHANCEMENT",
        ).order_by(AssetTask.created_at.desc())
    ).all()
    return [_serialize_asset_task(t, session) for t in tasks]


@router.post("/{project_id}/enhancements/{task_id}/approve")
def approve_enhancement(
    project_id: str,
    task_id: str,
    body: EnhancementApproveRequest,
    session: Session = Depends(get_session),
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de mejora no encontrada")
    if task.status != AssetTaskStatus.AWAITING_APPROVAL.value:
        raise service.error(
            409,
            "INVALID_STATE",
            f"Tarea no está en AWAITING_APPROVAL (estado: {task.status})",
        )
    try:
        task = asset_task_service.approve_task(session, task_id, request_hash=body.request_hash)
    except ValueError as exc:
        raise service.error(400, "HASH_MISMATCH", str(exc)) from exc
    task = asset_task_service.queue_task(session, task_id)
    return _serialize_asset_task(task, session)


@router.post("/{project_id}/enhancements/{task_id}/reject")
def reject_enhancement(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de mejora no encontrada")
    invalid_states = (
        AssetTaskStatus.ESTIMATING.value,
        AssetTaskStatus.AWAITING_APPROVAL.value,
    )
    if task.status not in invalid_states:
        raise service.error(409, "INVALID_STATE", f"No se puede rechazar en estado {task.status}")
    task = asset_task_service.reject_task(session, task_id)
    return _serialize_asset_task(task, session)


@router.post("/{project_id}/enhancements/{task_id}/cancel")
def cancel_enhancement(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de mejora no encontrada")
    terminal_states = (
        AssetTaskStatus.COMPLETED.value,
        AssetTaskStatus.FAILED.value,
        AssetTaskStatus.CANCELLED.value,
    )
    if task.status in terminal_states:
        raise service.error(409, "INVALID_STATE", f"No se puede cancelar en estado {task.status}")
    if task.status == AssetTaskStatus.RUNNING.value:
        raise service.error(
            409,
            "TASK_ALREADY_RUNNING",
            "La tarea ya está en ejecución; no se puede cancelar de forma segura",
        )
    from app.editor.asset_tasks import assert_transition
    assert_transition(task.status, AssetTaskStatus.CANCELLED)
    task.status = AssetTaskStatus.CANCELLED.value
    task.updated_at = datetime.now(UTC)
    session.add(task)
    session.commit()
    session.refresh(task)
    return _serialize_asset_task(task, session)


# ---- Generation endpoints (SNIPRA 1.6D Block 04) ----


@router.get("/{project_id}/generations/capability")
def generation_capability(
    project_id: str, session: Session = Depends(get_session)
) -> dict:
    """Get generation capability state for the project.
    Exposes local LTX capability and cloud provider status without executing generation."""
    service.project_or_404(session, project_id)

    # Local LTX capability - hardware check
    hw_supported = True
    hw_message = "Hardware compatible"
    hw_reason = ""

    try:
        from app.editor.generation import probe_local_generation_hardware
        hw_info = probe_local_generation_hardware()
        if not hw_info.runtime_supported:
            hw_supported = False
            hw_message = f"Generación local no disponible en este hardware: {hw_info.reason_message}"
            hw_reason = hw_info.reason_message
        elif not hw_info.fp8_supported:
            hw_supported = False
            hw_message = "Hardware no soporta FP8 nativo (requiere arquitectura Ada/Hopper+)"
            hw_reason = "FP8_UNSUPPORTED"
    except Exception:
        hw_supported = False
        hw_message = "No se pudo determinar capacidad de hardware local"
        hw_reason = "ERROR"

    # Local runtime readiness - physical backend is DEFERRED per Block 03
    runtime_ready = False
    runtime_message = "Hardware compatible, pero la ejecución física LTX aún no está habilitada"
    physical_validation = "DEFERRED"

    if not hw_supported:
        runtime_ready = False
        runtime_message = hw_message
        physical_validation = "DEFERRED"

    # Cloud providers - no real configuration mechanism exists yet
    # Return as available in contract but not ready for use
    cloud_providers = []
    for provider in (GenerationProvider.RUNWAY, GenerationProvider.LUMA):
        try:
            model = (
                GenerationModel.RUNWAY_GEN45
                if provider == GenerationProvider.RUNWAY
                else GenerationModel.LUMA_RAY32
            )
            caps = get_generation_provider_capabilities(provider, model)
            # Provider exists in contract but cloud backend not configured/habilitado en esta versión
            cloud_providers.append({
                "provider": provider.value,
                "model": caps.model.value,
                "supported": True,
                "available": False,
                "ready": False,
                "physical_validation": "N/A",
                "message": "Proveedor disponible en el contrato, pero backend cloud no configurado/habilitado en esta versión",
                "paid": True,
                "billable": True,
                "requires_api_key": True,
                "enabled_by_default": False,
            })
        except UnsupportedGenerationProvider:
            pass  # Provider not available in capabilities

    return {
        "local_ltx": {
            "provider": "local_ltx",
            "model": "ltx-video-2b-distilled-fp8",
            "hardware_supported": hw_supported,
            "hardware_message": hw_message,
            "hardware_reason": hw_reason,
            "runtime_ready": runtime_ready,
            "runtime_message": runtime_message,
            "physical_validation": physical_validation,
            "paid": False,
            "billable": False,
            "requires_api_key": False,
            "enabled_by_default": True,
        },
        "cloud_providers": cloud_providers,
    }


@router.post("/{project_id}/generations")
def prepare_generation(
    project_id: str,
    body: GenerationPrepareRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Create a generation task in ESTIMATING -> AWAITING_APPROVAL.
    Does NOT execute generation. Returns task with cost estimate for explicit approval."""
    service.project_or_404(session, project_id)

    # Validate reference assets exist and belong to project
    for ref_id in body.reference_asset_ids:
        asset = session.get(EditorAsset, ref_id)
        if asset is None or asset.project_id != project_id:
            raise service.error(
                422,
                "VALIDATION_ERROR",
                f"reference asset {ref_id!r} not found in project {project_id}",
            )

    # Validate grounding assets exist and belong to project
    for ground_id in body.grounding_asset_ids:
        asset = session.get(EditorAsset, ground_id)
        if asset is None or asset.project_id != project_id:
            raise service.error(
                422,
                "VALIDATION_ERROR",
                f"grounding asset {ground_id!r} not found in project {project_id}",
            )

    try:
        request = GenerationRequest(
            project_id=project_id,
            mode=body.mode,
            provider=body.provider,
            model=body.model,
            prompt_intent=body.prompt_intent,
            visual_direction=body.visual_direction,
            camera_direction=body.camera_direction,
            avoid_constraints=body.avoid_constraints,
            duration_seconds=body.duration_seconds,
            aspect_ratio=body.aspect_ratio,
            resolution=body.resolution,
            reference_asset_ids=body.reference_asset_ids,
            grounding_asset_ids=body.grounding_asset_ids,
            seed=body.seed,
            generation_parameters=body.generation_parameters,
        )
    except ValidationError as exc:
        from app.core.sanitize import sanitize_error_text
        errors = exc.errors()
        detail = "; ".join(
            f"{'.'.join(str(e) for e in err['loc'])}: {err['msg']}" for err in errors
        )
        raise service.error(422, "VALIDATION_ERROR", sanitize_error_text(detail)) from exc

    task = generation_task_service.create_generation_task(
        session, project_id=project_id, request=request
    )
    try:
        task = generation_task_service.attach_generation_estimate(session, task.id)
    except LocalGenerationUnavailable as exc:
        from app.core.sanitize import sanitize_error_text
        raise service.error(
                422,
                "LOCAL_GENERATION_UNAVAILABLE",
                sanitize_error_text(str(exc)),
            ) from exc
    except ValueError as exc:
        from app.core.sanitize import sanitize_error_text
        raise service.error(
            422,
            "VALIDATION_ERROR",
            sanitize_error_text(str(exc)),
        ) from exc

    return _serialize_asset_task(task, session)


@router.get("/{project_id}/generations/{task_id}")
def get_generation_task(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de generación no encontrada")
    if task.kind != "GENERATION":
        raise service.error(404, "TASK_NOT_FOUND", "Tarea no es de generación")
    return _serialize_asset_task(task, session)


@router.get("/{project_id}/generations")
def list_generation_tasks(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    service.project_or_404(session, project_id)
    tasks = session.exec(
        select(AssetTask).where(
            AssetTask.project_id == project_id,
            AssetTask.kind == "GENERATION",
        ).order_by(AssetTask.created_at.desc())
    ).all()
    return [_serialize_asset_task(t, session) for t in tasks]


@router.post("/{project_id}/generations/{task_id}/approve")
def approve_generation(
    project_id: str,
    task_id: str,
    body: GenerationApproveRequest,
    session: Session = Depends(get_session),
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de generación no encontrada")
    if task.kind != "GENERATION":
        raise service.error(404, "TASK_NOT_FOUND", "Tarea no es de generación")
    if task.status != AssetTaskStatus.AWAITING_APPROVAL.value:
        raise service.error(
            409,
            "INVALID_STATE",
            f"Tarea no está en AWAITING_APPROVAL (estado: {task.status})",
        )
    try:
        task = generation_task_service.approve_generation_task(
                session, task_id, request_hash=body.request_hash
            )
    except ValueError as exc:
        raise service.error(400, "HASH_MISMATCH", str(exc)) from exc
    task = generation_task_service.queue_generation_task(session, task_id)
    return _serialize_asset_task(task, session)


@router.post("/{project_id}/generations/{task_id}/reject")
def reject_generation(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de generación no encontrada")
    if task.kind != "GENERATION":
        raise service.error(404, "TASK_NOT_FOUND", "Tarea no es de generación")
    invalid_states = (
        AssetTaskStatus.ESTIMATING.value,
        AssetTaskStatus.AWAITING_APPROVAL.value,
    )
    if task.status not in invalid_states:
        raise service.error(409, "INVALID_STATE", f"No se puede rechazar en estado {task.status}")
    task = generation_task_service.reject_generation_task(session, task_id)
    return _serialize_asset_task(task, session)


@router.post("/{project_id}/generations/{task_id}/cancel")
def cancel_generation(
    project_id: str, task_id: str, session: Session = Depends(get_session)
) -> dict:
    service.project_or_404(session, project_id)
    task = session.get(AssetTask, task_id)
    if not task or task.project_id != project_id:
        raise service.error(404, "TASK_NOT_FOUND", "Tarea de generación no encontrada")
    if task.kind != "GENERATION":
        raise service.error(404, "TASK_NOT_FOUND", "Tarea no es de generación")
    task = generation_task_service.cancel_generation_task(session, task_id)
    return _serialize_asset_task(task, session)


@router.post("/{project_id}/operations")
def operate(
    project_id: str, body: OperationCreate, session: Session = Depends(get_session)
) -> dict:
    return service.apply_operation(session, project_id, body.model_dump())


@router.get("/{project_id}/preview-at/{at_us}")
def preview_at(project_id: str, at_us: int, session: Session = Depends(get_session)) -> dict:
    project = service.project_or_404(session, project_id)
    version = service.current_version(session, project)
    return {
        "version": project.current_version,
        "at_us": at_us,
        "active": timeline.resolve_at(version.timeline, at_us),
    }


@router.get("/{project_id}/preview-plan")
def preview_plan(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = service.project_or_404(session, project_id)
    version = service.current_version(session, project)
    assets = {
        asset.id: service.serialize_asset(asset)
        for asset in service.project_assets(session, project.id)
    }
    return {
        "version": project.current_version,
        "duration_us": version.timeline.get("duration_us", 0),
        "segments": [
            {
                **segment,
                "video_url": (
                    f"/api/editor/projects/{project.id}/assets/{segment['asset_id']}/video"
                ),
                "asset": assets.get(segment["asset_id"]),
            }
            for segment in timeline.preview_segments(version.timeline)
        ],
        "content": timeline.preview_content(version.timeline),
        "output_canvas": version.timeline.get("output_canvas", {"width": 1280, "height": 720}),
    }


@router.get("/{project_id}/history")
def history(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    service.project_or_404(session, project_id)
    return [
        {
            "number": v.number,
            "parent_number": v.parent_number,
            "created_by": v.created_by,
            "summary": v.summary,
            "created_at": v.created_at.isoformat(),
        }
        for v in session.exec(
            select(EditorVersion)
            .where(EditorVersion.project_id == project_id)
            .order_by(EditorVersion.number.desc())
        ).all()
    ]


@router.get("/{project_id}/assets/{asset_id}/video")
def asset_video(project_id: str, asset_id: str, session: Session = Depends(get_session)):
    asset = service.asset_or_404(session, project_id, asset_id)
    return FileResponse(asset.original_path, media_type="video/mp4", filename=asset.filename)


@router.get("/{project_id}/assets/{asset_id}/thumbnail")
def asset_thumbnail(
    project_id: str, asset_id: str, at_us: int = 0, session: Session = Depends(get_session)
):
    asset = service.asset_or_404(session, project_id, asset_id)
    try:
        path = media_artifacts.thumbnail(asset, at_us)
    except Exception as exc:
        raise service.error(409, "PREVIEW_FAILED", str(exc)) from exc
    return FileResponse(path, media_type="image/jpeg", filename=path.name)


@router.get("/{project_id}/assets/{asset_id}/waveform")
def asset_waveform(project_id: str, asset_id: str, session: Session = Depends(get_session)):
    asset = service.asset_or_404(session, project_id, asset_id)
    try:
        path = media_artifacts.waveform(asset)
    except Exception as exc:
        raise service.error(409, "PREVIEW_FAILED", str(exc)) from exc
    if path is None:
        raise service.error(404, "WAVEFORM_MISSING", "El asset no tiene audio")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.post("/{project_id}/assets/{asset_id}/proxy")
def create_proxy(project_id: str, asset_id: str, session: Session = Depends(get_session)) -> dict:
    asset = service.asset_or_404(session, project_id, asset_id)
    try:
        path = media_artifacts.proxy(asset)
    except Exception as exc:
        raise service.error(409, "PREVIEW_FAILED", str(exc)) from exc
    return {
        "available": path is not None,
        "url": f"/api/editor/projects/{project_id}/assets/{asset_id}/proxy/video" if path else None,
    }


@router.get("/{project_id}/assets/{asset_id}/proxy/video")
def asset_proxy(project_id: str, asset_id: str, session: Session = Depends(get_session)):
    asset = service.asset_or_404(session, project_id, asset_id)
    path = media_artifacts.proxy(asset)
    if path is None:
        raise service.error(404, "PROXY_UNAVAILABLE", "Este asset no requiere proxy")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.post("/{project_id}/render")
def render(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = service.project_or_404(session, project_id)
    version = service.current_version(session, project)
    assets = {
        asset.id: asset.original_path for asset in service.project_assets(session, project.id)
    }
    output = settings.editor_dir / project.id / "renders" / f"v{project.current_version}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = render_timeline(version.timeline, assets, output)
    except RenderError as exc:
        raise service.error(409, "RENDER_FAILED", str(exc)) from exc
    return {
        "version": project.current_version,
        "output": result.name,
        "url": f"/api/editor/projects/{project.id}/renders/{result.name}",
    }


@router.get("/{project_id}/renders/{filename}")
def rendered(project_id: str, filename: str):
    path = (settings.editor_dir / project_id / "renders" / Path(filename).name).resolve()
    root = (settings.editor_dir / project_id / "renders").resolve()
    if root not in path.parents or not path.is_file():
        raise service.error(404, "RENDER_MISSING", "Render no encontrado")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@router.post("/{project_id}/restore/{number}")
def restore(project_id: str, number: int, session: Session = Depends(get_session)) -> dict:
    return service.restore_version(session, project_id, number)


@router.post("/{project_id}/undo")
def undo(project_id: str, session: Session = Depends(get_session)) -> dict:
    project = service.project_or_404(session, project_id)
    current = service.current_version(session, project)
    if current.parent_number is None:
        raise service.error(409, "HISTORY_EMPTY", "No hay una versión anterior para restaurar")
    return service.restore_version(session, project_id, current.parent_number)


@router.post("/{project_id}/shorts")
async def create_short_batch(
    project_id: str,
    body: ShortBatchRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Create N Shorts or Montages from a source job/asset via C-G pipeline + Director.

    This endpoint:
    1. Runs C-G pipeline (1.5C-1.5G) exactly once per batch
    2. Builds enhanced evidence via 1.5H bridge
    3. Builds smart collections (1.5J)
    4. Invokes Director for editorial decisions
    5. Returns a ShortBatchPlan (highlight) or MontageBatchPlan (montage) with full grounding

    Does NOT render or publish. Returns a structured plan for N shorts/montages.
    """
    try:
        plan = await orchestrate_shorts(
            session=session,
            project_id=project_id,
            request=body,
            intent=body.intent,
        )
    except ValueError as exc:
        raise service.error(422, "SHORTS_PLAN_INVALID", str(exc)) from exc
    except RuntimeError as exc:
        # Fallo del provider (CLI/API) al invocar Director: no dejar un 500 vacío.
        raise service.error(503, "SHORTS_PROVIDER_UNAVAILABLE", str(exc)) from exc
    return plan.to_dict()


@router.post("/{project_id}/montages")
async def create_montage_batch(
    project_id: str,
    body: MontageBatchRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Create N Montages from a source job/asset via C-G pipeline + Director + Smart Collections.

    This endpoint:
    1. Runs C-G pipeline (1.5C-1.5G) exactly once per batch
    2. Builds enhanced evidence via 1.5H bridge
    3. Builds smart collections (1.5J) - structural groupings of related events
    4. Invokes Director for editorial montage decisions
    5. Returns a MontageBatchPlan with selected montages and full grounding

    Does NOT render or publish. Returns a structured plan for N montages.
    """
    try:
        plan = await orchestrate_shorts(
            session=session,
            project_id=project_id,
            request=body,
            intent="montage",
        )
    except ValueError as exc:
        raise service.error(422, "SHORTS_PLAN_INVALID", str(exc)) from exc
    except RuntimeError as exc:
        # Fallo del provider (CLI/API) al invocar Director: no dejar un 500 vacío.
        raise service.error(503, "SHORTS_PROVIDER_UNAVAILABLE", str(exc)) from exc
    return plan.to_dict()


@router.post("/{project_id}/metadata")
def create_metadata(
    project_id: str,
    body: MetadataGenerationRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Generate metadata from an existing I/J plan without changing it or the project."""
    service.project_or_404(session, project_id)
    if body.plan.get("project_id") != project_id:
        raise service.error(422, "METADATA_PROJECT_MISMATCH", "El plan no pertenece al proyecto")
    try:
        from app.providers.registry import get_detector

        detector = get_detector(settings.llm_provider, settings.llm_model)
        complete_fn = getattr(detector, "_complete", None)
        if not callable(complete_fn):
            raise RuntimeError("Metadata provider unavailable")
        generator = (
            generate_metadata_for_shorts
            if body.plan_type == "shorts"
            else generate_metadata_for_montages
        )
        # Director's retry helper parses the *editorial* output schema and is
        # intentionally incompatible with metadata's independent JSON schema.
        return generator(
            body.plan,
            complete_fn,
            platform=body.platform,
            language=body.language,
        ).to_dict()
    except MetadataGenerationError as exc:
        raise service.error(422, "METADATA_INVALID", str(exc)) from exc
    except RuntimeError as exc:
        raise service.error(503, "METADATA_PROVIDER_UNAVAILABLE", str(exc)) from exc


@router.post("/{project_id}/agent")
def run_agent(project_id: str, body: AgentRequest, session: Session = Depends(get_session)) -> dict:
    try:
        return agent.run_request(
            session,
            project_id,
            body.request,
            intent=body.intent,
            execute=body.execute,
            key_prefix=body.idempotency_key or new_id(),
        )
    except ValueError as exc:
        raise service.error(400, "AGENT_PLAN_INVALID", str(exc)) from exc
    except RuntimeError as exc:
        raise service.error(503, "AGENT_TOOL_FAILED", str(exc)) from exc
    except FileNotFoundError as exc:
        # Runtime/packaging failure (canonical Skill missing from this install),
        # not a user input error: fail closed with a safe, actionable message
        # instead of leaking an unhandled 500 traceback to the client.
        raise service.error(503, "AGENT_SKILL_UNAVAILABLE", str(exc)) from exc


@router.post("/{project_id}/agent/{run_id}/confirm")
def confirm_agent_plan(
    project_id: str, run_id: str, session: Session = Depends(get_session)
) -> dict:
    """Confirm a pending agent plan (NEEDS_USER) by executing the server-side plan."""
    # Use populate_existing to ensure we see latest DB state (raw SQL may have updated pending_plan)
    run = session.exec(
        select(EditorAgentRun)
        .where(
            EditorAgentRun.project_id == project_id,
            EditorAgentRun.id == run_id,
        )
        .execution_options(populate_existing=True)
    ).first()
    if run is None:
        raise service.error(404, "AGENT_RUN_NOT_FOUND", "Ejecución de agente no encontrada")
    
    # Idempotent response for already-confirmed runs
    if run.pending_plan is None and run.resulting_version is not None and run.operation_ids:
        project = service.project_or_404(session, project_id)
        timeline_data = service.current_version(session, project).timeline
        return {
            "status": run.status if run.status in ("PASS", "NEEDS_USER") else "PASS",
            "state": "VERIFYING",
            "version": run.resulting_version,
            "operations": [
                {"operation_id": op_id, "version": run.resulting_version, "idempotent": True}
                for op_id in run.operation_ids
            ],
            "plan": {},
            "timeline": timeline_data,
            "verification": run.verification or {},
            "report": {
                "changes": len(run.operation_ids),
                "repair_rounds": run.repair_rounds,
                "duration_us": timeline_data.get("duration_us", 0),
                "verification": (run.verification or {}).get("status", "PASS"),
                "undo_available": bool(run.operation_ids),
            },
            "run_id": run.id,
        }

    if run.status not in ("NEEDS_USER",):
        msg = f"Run no está en estado confirmable: {run.status}"
        raise service.error(409, "INVALID_RUN_STATE", msg)
    if run.pending_plan is None:
        raise service.error(409, "NO_PENDING_PLAN", "No hay plan pendiente para confirmar")

    project = service.project_or_404(session, project_id)
    if project.current_version != run.base_version:
        raise service.error(
            409,
            "STALE_VERSION",
            "La timeline cambió desde que se generó este plan. "
            "Genera un plan nuevo antes de confirmar.",
        )

    plan_data = run.pending_plan["plan"]
    plan = EditorialPlan.model_validate(plan_data)

    # Rehydrate evidence if needed
    intent = run.pending_plan.get("intent", "edit")
    evidence = run.pending_plan.get("evidence", {})
    if evidence.get("needs_transcript"):
        evidence["transcript"] = transcript.get_range(
            project.source_job_id,
            0,
            min(int(project.timeline.get("duration_us", 0)), 30_000_000),
        )
    else:
        evidence["transcript"] = []
    if intent in {"highlight", "montage"}:
        evidence["best_event_pool"] = events.best_event_pool(
            events.events_for_job(session, project.source_job_id), intent
        )
    else:
        evidence["best_event_pool"] = []
    via_evidence = agent.rehydrate_via_evidence(evidence.get("via_evidence"))
    if via_evidence:
        evidence["via_evidence"] = via_evidence
    else:
        evidence.pop("via_evidence", None)

    # Enhanced Event Pool (1.5H) - use persisted copy, do NOT reconstruct
    enhanced_pool = evidence.get("enhanced_event_pool")
    if enhanced_pool is not None:
        try:
            evidence["enhanced_event_pool"] = validate_director_evidence(enhanced_pool)
        except BridgeValidationError as exc:
            raise service.error(409, "ENHANCED_EVIDENCE_CORRUPT", str(exc)) from exc

    # Execute with stable idempotency key based on run.id
    result = agent.execute_plan(session, project_id, plan, key_prefix=f"confirm:{run.id}")
    checked = agent._check(session, project_id, plan, evidence)

    # Get updated timeline after execution
    project = service.project_or_404(session, project_id)
    timeline_data = service.current_version(session, project).timeline

    run.status = "PASS" if checked["status"] == "PASS" else "NEEDS_USER"
    run.resulting_version = result["version"]
    run.operation_ids = [item["operation_id"] for item in result["operations"]]
    model_usage = (run.verification or {}).get("model_usage")
    run.verification = {**checked, "model_usage": model_usage} if model_usage else checked
    run.pending_plan = None  # Clear after execution
    session.add(run)
    session.commit()

    return {
        "status": "PASS" if checked["status"] == "PASS" else "NEEDS_USER",
        "state": "VERIFYING",
        **result,
        "timeline": timeline_data,
        "verification": checked,
        "report": {
            "changes": len(result["operations"]),
            "repair_rounds": 0,
            "duration_us": timeline_data.get("duration_us", 0),
            "verification": checked["status"],
            "undo_available": bool(result["operations"]),
        },
        "run_id": run.id,
    }
