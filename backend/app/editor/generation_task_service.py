"""AssetTask GENERATION lifecycle orchestration (SNIPRA 1.6D Block 03).

Drives a local LTX generation through the Block 01/02 state machine:

    ESTIMATING -> AWAITING_APPROVAL -> QUEUED -> RUNNING -> COMPLETED

with approval bound to the exact request hash, fail-closed cost policy, and
partial-output cleanup on failure. It never mutates a timeline, never runs VIA,
and never touches the source asset.

Finalisation is atomic **at the database layer**: on success the GENERATED
`EditorAsset` row and the `AssetTask` completion (status, output_asset_id,
model, progress) are written in a single commit - never two. There is no
filesystem/DB two-phase commit: a process crash in the narrow window between
`os.replace()` of the output file (inside the provider) and this commit can
still leave an orphan file on disk. That is a recoverable filesystem residue,
not a DB inconsistency - the task stays durably RUNNING and no asset row exists.

Cloud providers (Runway/Luma) are NOT executed in this Block.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session

from app.core.sanitize import sanitize_error_text
from app.editor import service as editor_service
from app.editor.asset_provenance import (
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

# Import LocalGenerativeRuntime for type hints
from app.editor.generation import (
    ExternalTaskState,
    GenerationAspectRatio,
    GenerationExternalTask,
    GenerationMode,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResolution,
    LocalGenerationUnavailable,
    LocalGenerativeRuntime,
    PromptCompiler,
    estimate_local_generation,
    generated_provenance,
    resolve_generation_provider,
)
from app.editor.models import AssetTask, EditorAsset, now


def _checkpoint_commit(session: Session) -> None:
    """Commit an intermediate lifecycle transition (e.g. the durable RUNNING
    checkpoint before the long generation run)."""
    session.commit()


def _finalize_commit(session: Session) -> None:
    """The single atomic finalisation commit - success (asset + task together)
    or the failure record. A dedicated seam so fault-injection tests can make
    exactly this commit fail."""
    session.commit()


def _pre_commit_barrier() -> None:
    """No-op in production. Test seam: an injected fault here simulates a crash
    after the GENERATED asset has been flushed but before the finalising commit."""


class GenerationExecutionError(RuntimeError):
    """Raised when generation execution fails. Wrapped for cleanup."""


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


def _rehydrate_request(task: AssetTask) -> GenerationRequest:
    return GenerationRequest.model_validate(task.request)


def _rebuild_external_task(task: AssetTask) -> GenerationExternalTask:
    """Reconstruct GenerationExternalTask from persisted external_task_* fields."""
    if not task.external_task_id:
        raise ValueError("external_task_id is not set")
    return GenerationExternalTask(
        external_task_id=task.external_task_id,
        provider=GenerationProvider(task.provider),
        model=GenerationModel(task.model),
        state=ExternalTaskState(task.external_task_state or ExternalTaskState.PENDING.value),
        completed_at=None,
        failure_code=task.error,
        failure_message=task.error,
        output_asset_id=task.output_asset_id,
        metadata=task.external_task_metadata or {},
    )


def _persist_external_task_state(session: Session, task: AssetTask, ext_task: GenerationExternalTask) -> None:
    """Persist external task state to AssetTask fields."""
    task.external_task_id = ext_task.external_task_id
    task.external_task_state = ext_task.state.value
    task.external_task_metadata = ext_task.metadata
    task.updated_at = now()
    session.add(task)


# --- Public API ---

def create_generation_task(
    session: Session,
    *,
    project_id: str,
    request: GenerationRequest,
) -> AssetTask:
    """New task in ESTIMATING. The request hash is fixed here and every later
    approval must bind to it exactly."""
    editor_service.project_or_404(session, project_id)

    # Validate reference assets exist and belong to project
    for ref_id in request.reference_asset_ids:
        asset = session.get(EditorAsset, ref_id)
        if asset is None or asset.project_id != project_id:
            raise ValueError(f"reference asset {ref_id!r} not found in project {project_id}")

    # Validate grounding assets exist and belong to project
    for ground_id in request.grounding_asset_ids:
        asset = session.get(EditorAsset, ground_id)
        if asset is None or asset.project_id != project_id:
            raise ValueError(f"grounding asset {ground_id!r} not found in project {project_id}")

    task = AssetTask(
        project_id=project_id,
        kind=AssetTaskKind.GENERATION.value,
        status=AssetTaskStatus.ESTIMATING.value,
        provider=request.provider.value,
        model=request.model.value,
        request=request.model_dump(mode="json"),
        request_hash=request.request_hash(),
        approval_state=AssetTaskApprovalState.PENDING.value,
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    return task


def attach_generation_estimate(
    session: Session,
    task_id: str,
) -> AssetTask:
    """ESTIMATING -> AWAITING_APPROVAL, storing a LOCAL $0 estimate."""
    task = _task_or_error(session, task_id)
    assert_transition(task.status, AssetTaskStatus.AWAITING_APPROVAL)

    request = _rehydrate_request(task)
    estimate = estimate_local_generation(request)

    task.cost_estimate = estimate.model_dump(mode="json")
    task.status = AssetTaskStatus.AWAITING_APPROVAL.value
    _touch(session, task)
    return task


def approve_generation_task(session: Session, task_id: str, *, request_hash: str) -> AssetTask:
    """Bind an approval to the exact request that will run. A mismatching hash
    is rejected outright."""
    task = _task_or_error(session, task_id)
    if request_hash != task.request_hash:
        raise ValueError("approval hash does not match the current request")
    task.approval_state = AssetTaskApprovalState.APPROVED.value
    task.approved_request_hash = request_hash
    _touch(session, task)
    return task


def reject_generation_task(session: Session, task_id: str) -> AssetTask:
    task = _task_or_error(session, task_id)
    task.approval_state = AssetTaskApprovalState.REJECTED.value
    task.approved_request_hash = None
    _touch(session, task)
    return task


def update_generation_request(
    session: Session, task_id: str, *, request: GenerationRequest
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


def queue_generation_task(session: Session, task_id: str) -> AssetTask:
    """AWAITING_APPROVAL -> QUEUED. Enforces the full Block 01 guard: a valid
    approval bound to the current hash AND an estimate that permits execution.
    Only LOCAL_LTX is allowed in this Block."""
    task = _task_or_error(session, task_id)

    # Only LOCAL_LTX is supported for execution in this Block
    if task.provider != GenerationProvider.LOCAL_LTX.value:
        raise InvalidTaskTransition("only local_ltx provider is supported for execution in this Block")

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


def cancel_generation_task(session: Session, task_id: str) -> AssetTask:
    """Cancel a generation task. If RUNNING with external_task_id, delegates to
    runtime cancel. Otherwise just marks CANCELLED."""
    task = _task_or_error(session, task_id)
    current_status = AssetTaskStatus(task.status)

    if current_status == AssetTaskStatus.CANCELLED:
        return task

    if current_status == AssetTaskStatus.COMPLETED:
        raise InvalidTaskTransition("cannot cancel a completed task")

    if current_status == AssetTaskStatus.QUEUED:
        # Never submitted - just cancel
        assert_transition(current_status, AssetTaskStatus.CANCELLED)
        task.status = AssetTaskStatus.CANCELLED.value
        task.error = "cancelled by user"
        task.updated_at = now()
        _touch(session, task)
        return task

    if current_status == AssetTaskStatus.RUNNING:
        if task.external_task_id:
            # Has been submitted - request cancellation via cancel_requested
            # The reconciliation loop will call runtime.cancel()
            task.cancel_requested = True
            task.error = "cancel requested by user"
            task.updated_at = now()
            _touch(session, task)
            return task
        else:
            # Never submitted
            assert_transition(current_status, AssetTaskStatus.CANCELLED)
            task.status = AssetTaskStatus.CANCELLED.value
            task.error = "cancelled by user"
            task.updated_at = now()
            _touch(session, task)
            return task

    # For other states (ESTIMATING, AWAITING_APPROVAL)
    assert_transition(current_status, AssetTaskStatus.CANCELLED)
    task.status = AssetTaskStatus.CANCELLED.value
    task.error = "cancelled by user"
    task.updated_at = now()
    _touch(session, task)
    return task


# --- Execution Core ---

def execute_generation_task(
    session: Session,
    task_id: str,
    *,
    local_runtime: LocalGenerativeRuntime,
) -> AssetTask:
    """QUEUED -> RUNNING -> COMPLETED (or FAILED). On success a NEW EditorAsset
    (origin=GENERATED, semantic_role=ILLUSTRATIVE) is registered."""
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

    # Only LOCAL_LTX supported for execution in this Block
    if task.provider != GenerationProvider.LOCAL_LTX.value:
        raise InvalidTaskTransition("only local_ltx provider is supported for execution in this Block")

    # Durable RUNNING checkpoint before the long generation run.
    assert_transition(task.status, AssetTaskStatus.RUNNING)
    task.status = AssetTaskStatus.RUNNING.value
    task.updated_at = now()
    session.add(task)
    _checkpoint_commit(session)
    session.refresh(task)

    # Delegate to shared core for the actual generation execution
    return execute_claimed_generation_task(session, task.id, local_runtime=local_runtime)


def execute_claimed_generation_task(
    session: Session,
    task_id: str,
    *,
    local_runtime: LocalGenerativeRuntime,
) -> AssetTask:
    """Core generation execution for a task already in RUNNING status.

    Performs EXACTLY ONE step per call: submit, poll, or finalize.
    Returns the task after the step. The caller should call this repeatedly
    until the task reaches a terminal state (COMPLETED/FAILED/CANCELLED).

    Precondition: task.status == RUNNING (already durably committed).
    All validation and execution are inside the try block for fail-closed behavior.
    """
    task = _task_or_error(session, task_id)
    if AssetTaskStatus(task.status) != AssetTaskStatus.RUNNING:
        raise InvalidTaskTransition(
            f"execute_claimed requires RUNNING, task is {task.status}"
        )

    # Validate kind and provider (fail-closed: invalid RUNNING -> FAILED)
    if task.kind != AssetTaskKind.GENERATION.value:
        raise InvalidTaskTransition(f"task kind must be GENERATION, got {task.kind}")
    if task.provider != GenerationProvider.LOCAL_LTX.value:
        raise InvalidTaskTransition(f"provider must be local_ltx, got {task.provider}")

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

    try:
        # Resolve provider (fail closed for unsupported hardware)
        provider, runtime = resolve_generation_provider(
            request=request,
            local_runtime=local_runtime,
            cloud_registry={},  # No cloud providers in this Block
        )
    except LocalGenerationUnavailable as exc:
        # Hardware unavailable - mark task as FAILED
        assert_transition(task.status, AssetTaskStatus.FAILED)
        task.status = AssetTaskStatus.FAILED.value
        task.error = sanitize_error_text(f"{type(exc).__name__}: {exc}")[:2000]
        task.progress = 0.0
        task.updated_at = now()
        session.add(task)
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)
    except InvalidTaskTransition as exc:
        # Invalid configuration - mark task as FAILED
        assert_transition(task.status, AssetTaskStatus.FAILED)
        task.status = AssetTaskStatus.FAILED.value
        task.error = sanitize_error_text(f"{type(exc).__name__}: {exc}")[:2000]
        task.progress = 0.0
        task.updated_at = now()
        session.add(task)
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)

    # Initialize submit_count if not present
    if task.submit_count is None:
        task.submit_count = 0

    # --- SUBMIT PHASE (exactly once) ---
    if task.external_task_id is None:
        # Compile and submit EXACTLY ONCE
        compiled_request = runtime.compile_request(request)
        # Handle both sync and async submit methods
        submit_result = runtime.submit(compiled_request)
        if hasattr(submit_result, '__await__'):
            import asyncio
            ext_task = asyncio.run(submit_result)
        else:
            ext_task = submit_result

        task.submit_count = 1

        # Persist external task state BEFORE polling
        task.external_task_id = ext_task.external_task_id
        task.external_task_state = ext_task.state.value
        task.external_task_metadata = ext_task.metadata
        task.updated_at = now()
        session.add(task)
        _checkpoint_commit(session)
        session.refresh(task)
        return task

    # Already submitted - reconstruct external task for polling
    ext_task = _rebuild_external_task(task)

    # Check for cancellation
    session.refresh(task)
    if AssetTaskStatus(task.status) == AssetTaskStatus.CANCELLED:
        # Try to cancel the external task
        try:
            if ext_task and ext_task.state in (ExternalTaskState.PENDING, ExternalTaskState.RUNNING):
                ext_task = runtime.cancel(ext_task)
                _persist_external_task_state(session, task, ext_task)
        except Exception:
            pass  # Best effort
        raise GenerationExecutionError("cancelled by user")

    # Check for cancel_requested flag (post-submit cancellation)
    if task.cancel_requested:
        try:
            if ext_task and ext_task.state in (ExternalTaskState.PENDING, ExternalTaskState.RUNNING):
                ext_task = runtime.cancel(ext_task)
                _persist_external_task_state(session, task, ext_task)
        except Exception:
            pass  # Best effort
        # Wait for runtime to confirm cancellation
        ext_task = runtime.get_status(ext_task)
        _persist_external_task_state(session, task, ext_task)
        _checkpoint_commit(session)
        if ext_task.state == ExternalTaskState.CANCELLED:
            assert_transition(task.status, AssetTaskStatus.CANCELLED)
            task.status = AssetTaskStatus.CANCELLED.value
            task.error = "cancelled by user"
            task.updated_at = now()
            session.add(task)
            _finalize_commit(session)
            return _post_commit_return(session, task, task_id)
        # Not yet cancelled by runtime, schedule next poll
        task.next_poll_at = datetime.now(UTC) + timedelta(seconds=2)
        session.add(task)
        _checkpoint_commit(session)
        return task

    # Poll external task
    ext_task = runtime.get_status(ext_task)
    _persist_external_task_state(session, task, ext_task)
    _checkpoint_commit(session)

    if ext_task.state == ExternalTaskState.SUCCEEDED:
        # --- FETCH RESULT PHASE ---
        if not ext_task.output_asset_id and not ext_task.output_url:
            raise GenerationExecutionError("no output asset from generation")

        # For fake runtime, output is written to path
        output_path = ext_task.output_url
        if not output_path:
            # Use a temporary path for fetching
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_path = tmp.name
            runtime.fetch_result_to_path(ext_task, output_path)

        output_path = Path(output_path)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise GenerationExecutionError("generated output vanished before registration")

        # Validate video output
        width, height, fps, duration_us, has_audio = _probe_video(output_path)
        if width <= 0 or height <= 0 or duration_us <= 0 or fps <= 0:
            raise GenerationExecutionError(f"invalid video output: {width}x{height}@{fps}fps duration={duration_us}us")

        # --- REGISTER GENERATED ASSET ---
        generated_asset = _register_generated_asset(
            session, task, request, ext_task, output_path
        )

        # --- TASK COMPLETION - ATOMIC COMMIT ---
        assert_transition(task.status, AssetTaskStatus.COMPLETED)
        task.status = AssetTaskStatus.COMPLETED.value
        task.output_asset_id = generated_asset.id
        task.model = task.model
        task.progress = 1.0
        task.error = None
        task.updated_at = now()
        session.add(task)
        _pre_commit_barrier()
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)

    elif ext_task.state == ExternalTaskState.FAILED:
        error_msg = ext_task.failure_message or ext_task.failure_code or "generation failed"
        # Mark task as FAILED
        assert_transition(task.status, AssetTaskStatus.FAILED)
        task.status = AssetTaskStatus.FAILED.value
        task.error = sanitize_error_text(f"generation failed: {error_msg}")[:2000]
        task.progress = 0.0
        task.updated_at = now()
        session.add(task)
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)

    elif ext_task.state == ExternalTaskState.CANCELLED:
        # Mark task as CANCELLED
        assert_transition(task.status, AssetTaskStatus.CANCELLED)
        task.status = AssetTaskStatus.CANCELLED.value
        task.error = "generation cancelled externally"
        task.updated_at = now()
        session.add(task)
        _finalize_commit(session)
        return _post_commit_return(session, task, task_id)

    # For PENDING/RUNNING, schedule next poll and return
    task.next_poll_at = datetime.now(UTC) + timedelta(seconds=2)
    session.add(task)
    _checkpoint_commit(session)
    return task


def _register_generated_asset(
    session: Session,
    task: AssetTask,
    request: GenerationRequest,
    ext_task: GenerationExternalTask,
    output_path: Path,
) -> EditorAsset:
    """Atomic: only called after the provider has verified the output file."""
    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError("generated output vanished before registration")

    # Compute output hash
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    output_size_bytes = output_path.stat().st_size

    # Use ffprobe to get video metadata
    width, height, fps, duration_us, has_audio = _probe_video(output_path)

    # Validate video output
    if width <= 0 or height <= 0 or duration_us <= 0 or fps <= 0:
        raise GenerationExecutionError(f"invalid video output: {width}x{height}@{fps}fps duration={duration_us}us")

    # Build compiled prompt hash
    compiler = PromptCompiler()
    compiled = compiler.compile(request)
    compiled_prompt_hash = compiled.compiled_hash

    # Build provenance
    # For text_to_video: no parent, no root (generated from scratch)
    # For image_to_video: parent = first reference asset, root = ONLY factual root from reference asset's provenance
    if request.mode == GenerationMode.IMAGE_TO_VIDEO and request.reference_asset_ids:
        parent_asset_id = request.reference_asset_ids[0]
        # Get root from reference asset's provenance - ONLY factual root
        ref_asset = session.get(EditorAsset, parent_asset_id)
        root_source_asset_id = None
        if ref_asset:
            from app.editor.asset_provenance import read_asset_provenance
            ref_prov = read_asset_provenance(ref_asset.asset_metadata)
            if ref_prov:
                # ONLY use factual root from reference asset's provenance
                root_source_asset_id = ref_prov.root_source_asset_id
    else:
        # text_to_video: no parent, no root
        parent_asset_id = None
        root_source_asset_id = None

    gen_prov = generated_provenance(
        parent_asset_id=parent_asset_id,
        root_source_asset_id=root_source_asset_id,
        provider=GenerationProvider(task.provider),
        model=GenerationModel(task.model),
        mode=request.mode,
        duration_seconds=request.duration_seconds,
        aspect_ratio=GenerationAspectRatio(request.aspect_ratio),
        resolution=GenerationResolution(request.resolution),
        request_hash=task.request_hash,
        compiled_prompt_hash=compiled_prompt_hash,
        external_task_id=ext_task.external_task_id,
        output_hash=output_hash,
        output_size_bytes=output_size_bytes,
        operation_version="gen_v1",
        parameters={
            "seed": request.seed,
            "generation_parameters": request.generation_parameters,
        },
        duration_us=duration_us,
        width=width,
        height=height,
        fps=fps,
        reference_asset_ids=request.reference_asset_ids,
        grounding_asset_ids=request.grounding_asset_ids,
    )

    project = editor_service.project_or_404(session, task.project_id)
    return editor_service.add_asset(
        session,
        project,
        output_path,
        kind="video",
        duration_us=duration_us,
        width=width,
        height=height,
        has_audio=has_audio,
        asset_metadata=write_asset_provenance({}, gen_prov),
        commit=False,  # folded into the single finalisation commit
    )


# --- Helpers ---

def _probe_video(path: Path) -> tuple[int, int, float, int, bool]:
    """Extract video metadata using ffprobe."""
    import json
    import subprocess

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return 0, 0, 0.0, 0, False

    data = json.loads(result.stdout)
    width = height = 0
    fps = 0.0
    duration_us = 0
    has_audio = False

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            if "/" in r_frame_rate:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 0.0
        elif stream.get("codec_type") == "audio":
            has_audio = True

    fmt = data.get("format", {})
    duration_s = float(fmt.get("duration", 0))
    duration_us = int(duration_s * 1_000_000)

    return width, height, fps, duration_us, has_audio


def _touch(session: Session, task: AssetTask) -> None:
    task.updated_at = now()
    session.add(task)
    session.commit()
    session.refresh(task)