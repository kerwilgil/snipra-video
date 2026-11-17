"""AssetTask ENHANCEMENT lifecycle orchestration (SNIPRA 1.6C Block 02).

Drives a single local FFmpeg enhancement through the Block 01 state machine:

    ESTIMATING -> AWAITING_APPROVAL -> QUEUED -> RUNNING -> COMPLETED

with approval bound to the exact request hash, fail-closed cost policy, and
partial-output cleanup on failure. It never mutates a timeline, never runs VIA,
and never touches the source asset.

Finalisation is atomic **at the database layer**: on success the ENHANCED
`EditorAsset` row and the `AssetTask` completion (status, output_asset_id,
model, progress) are written in a single commit - never two. There is no
filesystem/DB two-phase commit: a process crash in the narrow window between
`os.replace()` of the output file (inside the provider) and this commit can
still leave an orphan file on disk. That is a recoverable filesystem residue,
not a DB inconsistency - the task stays durably RUNNING and no asset row exists.
"""

from __future__ import annotations

from pathlib import Path

from sqlmodel import Session

from app.core.sanitize import sanitize_error_text
from app.editor import service as editor_service
from app.editor.asset_provenance import (
    AssetOrigin,
    enhanced_provenance,
    infer_legacy_origin,
    read_asset_provenance,
    source_provenance,
    write_asset_provenance,
)
from app.editor.asset_tasks import (
    AssetTaskApprovalState,
    AssetTaskKind,
    AssetTaskStatus,
    InvalidTaskTransition,
    approval_is_valid,
    assert_transition,
    estimate_permits_execution,
    guard_ready_to_queue,
)
from app.editor.enhancement import EnhancementRequest
from app.editor.ffmpeg_enhancement import (
    FFmpegEnhancementProvider,
    default_enhanced_output_path,
)
from app.editor.models import AssetTask, EditorAsset, now


def _checkpoint_commit(session: Session) -> None:
    """Commit an intermediate lifecycle transition (e.g. the durable RUNNING
    checkpoint before the long FFmpeg run)."""
    session.commit()


def _finalize_commit(session: Session) -> None:
    """The single atomic finalisation commit - success (asset + task together)
    or the failure record. A dedicated seam so fault-injection tests can make
    exactly this commit fail."""
    session.commit()


def _pre_commit_barrier() -> None:
    """No-op in production. Test seam: an injected fault here simulates a crash
    after the ENHANCED asset has been flushed but before the finalising commit."""


class AmbiguousSourceProvenance(RuntimeError):
    """The source asset has no explicit provenance and cannot be safely
    interpreted as a SOURCE. Refuse rather than guess a lineage."""


class PostCommitReadError(RuntimeError):
    """The finalisation commit SUCCEEDED but reading the task row back
    afterwards failed. The task is durably COMPLETED (or FAILED) - this is a
    read-back problem, never a reason to mark the task FAILED."""


def _post_commit_return(session: Session, task: AssetTask, task_id: str) -> AssetTask:
    """Hand back the durable task after a successful commit. A read-back failure
    here must never change the outcome that was just persisted, and must never
    be caught by a pre-commit failure handler."""
    try:
        session.refresh(task)
        return task
    except Exception as exc:  # noqa: BLE001 - the commit already succeeded
        try:
            reloaded = session.get(AssetTask, task_id)
        except Exception:  # noqa: BLE001
            reloaded = None
        if reloaded is not None:
            return reloaded
        raise PostCommitReadError(
            f"AssetTask {task_id} was committed but could not be re-read: "
            f"{sanitize_error_text(str(exc))}"
        ) from exc


def _task_or_error(session: Session, task_id: str) -> AssetTask:
    task = session.get(AssetTask, task_id)
    if task is None:
        raise InvalidTaskTransition(f"AssetTask {task_id!r} not found")
    return task


def _rehydrate_request(task: AssetTask) -> EnhancementRequest:
    return EnhancementRequest.model_validate(task.request)


def create_enhancement_task(
    session: Session,
    *,
    project_id: str,
    request: EnhancementRequest,
) -> AssetTask:
    """New task in ESTIMATING. The request hash is fixed here and every later
    approval must bind to it exactly."""
    editor_service.project_or_404(session, project_id)
    editor_service.asset_or_404(session, project_id, request.source_asset_id)
    task = AssetTask(
        project_id=project_id,
        kind=AssetTaskKind.ENHANCEMENT.value,
        status=AssetTaskStatus.ESTIMATING.value,
        provider="ffmpeg",
        request=request.model_dump(mode="json"),
        request_hash=request.request_hash(),
        approval_state=AssetTaskApprovalState.PENDING.value,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def attach_estimate(
    session: Session,
    task_id: str,
    *,
    provider: FFmpegEnhancementProvider | None = None,
) -> AssetTask:
    """ESTIMATING -> AWAITING_APPROVAL, storing a LOCAL $0 estimate."""
    task = _task_or_error(session, task_id)
    provider = provider or _build_provider(session)
    request = _rehydrate_request(task)
    provider.validate_request(request)
    estimate = provider.estimate(request)
    assert_transition(task.status, AssetTaskStatus.AWAITING_APPROVAL)
    task.cost_estimate = estimate.model_dump(mode="json")
    task.status = AssetTaskStatus.AWAITING_APPROVAL.value
    _touch(session, task)
    return task


def approve_task(session: Session, task_id: str, *, request_hash: str) -> AssetTask:
    """Bind an approval to the exact request that will run. A mismatching hash
    is rejected outright."""
    task = _task_or_error(session, task_id)
    if request_hash != task.request_hash:
        raise ValueError("approval hash does not match the current request")
    task.approval_state = AssetTaskApprovalState.APPROVED.value
    task.approved_request_hash = request_hash
    _touch(session, task)
    return task


def reject_task(session: Session, task_id: str) -> AssetTask:
    task = _task_or_error(session, task_id)
    task.approval_state = AssetTaskApprovalState.REJECTED.value
    task.approved_request_hash = None
    _touch(session, task)
    return task


def update_request(
    session: Session, task_id: str, *, request: EnhancementRequest
) -> AssetTask:
    """Change the pending request. Any existing approval is voided the moment
    the hash it was bound to no longer matches."""
    task = _task_or_error(session, task_id)
    if AssetTaskStatus(task.status) not in (
        AssetTaskStatus.ESTIMATING,
        AssetTaskStatus.AWAITING_APPROVAL,
    ):
        raise InvalidTaskTransition("request can only change before the task is queued")
    task.request = request.model_dump(mode="json")
    task.request_hash = request.request_hash()
    if task.approved_request_hash != task.request_hash:
        task.approval_state = AssetTaskApprovalState.PENDING.value
        task.approved_request_hash = None
    task.cost_estimate = {}
    task.status = AssetTaskStatus.ESTIMATING.value
    _touch(session, task)
    return task


def queue_task(session: Session, task_id: str) -> AssetTask:
    """AWAITING_APPROVAL -> QUEUED. Enforces the full Block 01 guard: a valid
    approval bound to the current hash AND an estimate that permits execution."""
    task = _task_or_error(session, task_id)
    guard_ready_to_queue(
        approval_state=task.approval_state,
        approved_request_hash=task.approved_request_hash,
        current_request_hash=task.request_hash,
        cost_estimate=task.cost_estimate or None,
    )
    assert_transition(task.status, AssetTaskStatus.QUEUED)
    task.status = AssetTaskStatus.QUEUED.value
    _touch(session, task)
    return task


def execute_enhancement_task(
    session: Session,
    task_id: str,
    *,
    provider: FFmpegEnhancementProvider | None = None,
) -> AssetTask:
    """QUEUED -> RUNNING -> COMPLETED (or FAILED). On success a NEW EditorAsset
    (origin=ENHANCED) is registered; the source asset is never modified."""
    task = _task_or_error(session, task_id)
    if AssetTaskStatus(task.status) != AssetTaskStatus.QUEUED:
        # Fail closed: a COMPLETED/FAILED/other task never re-runs.
        raise InvalidTaskTransition(
            f"execute requires QUEUED, task is {task.status}"
        )

    # The request that runs must be exactly the one that was approved.
    if not approval_is_valid(
        approval_state=task.approval_state,
        approved_request_hash=task.approved_request_hash,
        current_request_hash=task.request_hash,
    ):
        raise InvalidTaskTransition("approval no longer bound to the current request")

    # Durable RUNNING checkpoint before the long FFmpeg run. This is the state
    # a crash before finalisation rolls back to.
    assert_transition(task.status, AssetTaskStatus.RUNNING)
    task.status = AssetTaskStatus.RUNNING.value
    task.updated_at = now()
    session.add(task)
    _checkpoint_commit(session)
    session.refresh(task)

    # Delegate to shared core for the actual enhancement execution
    return execute_claimed_enhancement_task(session, task.id, provider=provider)


def _cleanup_task_output(out_path: Path, task: AssetTask) -> None:
    """Delete only the output this task generated - never the source. Safe to
    call after a rollback: the task carries no durable output_asset_id then, so
    the orphan file is removed."""
    if task.output_asset_id is None and out_path.exists():
        out_path.unlink(missing_ok=True)
    partial = out_path.with_suffix(".partial" + out_path.suffix)
    partial.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _build_provider(session: Session) -> FFmpegEnhancementProvider:
    def _resolve(asset_id: str) -> Path:
        asset = session.get(EditorAsset, asset_id)
        if asset is None:
            raise AmbiguousSourceProvenance(f"source asset {asset_id!r} not found")
        return Path(asset.original_path)

    return FFmpegEnhancementProvider(resolve_source_path=_resolve)


def _source_provenance_for(session: Session, source_asset_id: str):
    src = session.get(EditorAsset, source_asset_id)
    if src is None:
        raise AmbiguousSourceProvenance(f"source asset {source_asset_id!r} not found")
    existing = read_asset_provenance(src.asset_metadata)
    if existing is not None:
        return existing
    legacy = infer_legacy_origin(src.asset_metadata)
    if legacy == AssetOrigin.SOURCE:
        return source_provenance(src.id)
    raise AmbiguousSourceProvenance(
        f"source asset {source_asset_id!r} has no provenance and cannot be "
        "safely treated as SOURCE; normalise it explicitly first"
    )


def _register_enhanced_asset(
    session: Session,
    task: AssetTask,
    request: EnhancementRequest,
    parent_prov,
    result: dict,
    out_path: Path,
) -> EditorAsset:
    """Atomic: only called after the provider has verified the output file."""
    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise RuntimeError("enhanced output vanished before registration")
    enh_prov = enhanced_provenance(
        parent=parent_prov,
        parent_asset_id=request.source_asset_id,
        provider=result["provider"],
        model_or_engine=result["model_or_engine"],
        request_hash=result["request_hash"],
        output_hash=result["output_hash"],
        output_size_bytes=result["output_size_bytes"],
        operation_version=result["operation_version"],
        parameters=result["parameters"],
        duration_us=result["duration_us"],
        width=result["width"],
        height=result["height"],
        fps=result["fps"],
    )
    project = editor_service.project_or_404(session, task.project_id)
    return editor_service.add_asset(
        session,
        project,
        out_path,
        kind="video",
        duration_us=result["duration_us"],
        width=result["width"],
        height=result["height"],
        has_audio=result["has_audio"],
        asset_metadata=write_asset_provenance({}, enh_prov),
        commit=False,  # folded into the single finalisation commit
    )


def execute_claimed_enhancement_task(
    session: Session,
    task_id: str,
    *,
    provider: FFmpegEnhancementProvider | None = None,
) -> AssetTask:
    """Core enhancement execution for a task already in RUNNING status.
    
    This is the single shared implementation used by both the synchronous
    execute_enhancement_task() and the worker's process_asset_task().
    
    Precondition: task.status == RUNNING (already durably committed).
    All validation and execution are inside the try block for fail-closed behavior.
    """
    task = _task_or_error(session, task_id)
    if AssetTaskStatus(task.status) != AssetTaskStatus.RUNNING:
        raise InvalidTaskTransition(
            f"execute_claimed requires RUNNING, task is {task.status}"
        )

    # All validation and execution inside try block for proper error handling
    out_path = None
    try:
        # Validate kind and provider (fail-closed: invalid RUNNING -> FAILED)
        if task.kind != AssetTaskKind.ENHANCEMENT.value:
            raise InvalidTaskTransition(f"task kind must be ENHANCEMENT, got {task.kind}")
        if task.provider != "ffmpeg":
            raise InvalidTaskTransition(f"provider must be ffmpeg, got {task.provider}")

        # Re-validate approval and hash before executing
        if not approval_is_valid(
            approval_state=task.approval_state,
            approved_request_hash=task.approved_request_hash,
            current_request_hash=task.request_hash,
        ):
            raise InvalidTaskTransition("approval no longer bound to the current request")

        if not estimate_permits_execution(task.cost_estimate):
            raise InvalidTaskTransition("cost estimate does not permit execution")

        request = _rehydrate_request(task)
        provider = provider or _build_provider(session)

        out_path = default_enhanced_output_path(task.project_id, task.id, request.profile.value)
        parent_prov = _source_provenance_for(session, request.source_asset_id)
        result = provider.enhance(request, output_path=str(out_path))

        # Stage the ENHANCED asset (flush only - no commit yet)...
        enhanced_asset = _register_enhanced_asset(
            session, task, request, parent_prov, result, out_path
        )

        # Task completion - commit BOTH in one transaction
        assert_transition(task.status, AssetTaskStatus.COMPLETED)
        task.status = AssetTaskStatus.COMPLETED.value
        task.output_asset_id = enhanced_asset.id
        task.model = result["model_or_engine"]
        task.progress = 1.0
        task.error = None
        task.updated_at = now()
        session.add(task)
        _pre_commit_barrier()
        _finalize_commit(session)
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        task = _task_or_error(session, task_id)
        if out_path is not None:
            _cleanup_task_output(out_path, task)
        assert_transition(task.status, AssetTaskStatus.FAILED)
        task.status = AssetTaskStatus.FAILED.value
        task.error = sanitize_error_text(f"{type(exc).__name__}: {exc}")[:2000]
        task.progress = 0.0
        task.updated_at = now()
        session.add(task)
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)

    return _post_commit_return(session, task, task_id)


def _touch(session: Session, task: AssetTask) -> None:
    task.updated_at = now()
    session.add(task)
    session.commit()
    session.refresh(task)
