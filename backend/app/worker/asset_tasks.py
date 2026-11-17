"""Worker module for AssetTask ENHANCEMENT and GENERATION processing.

SNIPRA 1.6C Block 04 / 1.6D Block 03. Reuses the existing worker pattern:
atomic claim via UPDATE, dedicated Session, recovery at startup.
Processes ENHANCEMENT (ffmpeg) and GENERATION (local_ltx).

GENERATION tasks use durable polling: each worker iteration performs exactly
one step (submit/poll/finalize) and schedules the next poll via next_poll_at.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from sqlmodel import Session, select

from app.core.db import engine
from app.editor import asset_task_service, generation_task_service
from app.editor.asset_tasks import (
    AssetTaskKind,
    AssetTaskStatus,
    InvalidTaskTransition,
    approval_is_valid,
    assert_transition,
    estimate_permits_execution,
)
from app.editor.generation import GenerationProvider
from app.editor.models import AssetTask, now

# Re-export validation functions for worker use
__all__ = [
    "claim_next_asset_task",
    "process_asset_task",
    "recover_asset_tasks",
    "approval_is_valid",
    "estimate_permits_execution",
]

# Default poll interval for GENERATION tasks
GENERATION_POLL_INTERVAL = timedelta(seconds=2)


def _is_task_eligible(task: AssetTask) -> bool:
    """Check if a task is eligible for claiming (for final safety check)."""
    if task.kind == AssetTaskKind.GENERATION.value:
        # Only claim GENERATION tasks with local_ltx provider
        return task.provider == GenerationProvider.LOCAL_LTX.value
    return True  # ENHANCEMENT tasks are always eligible


def claim_next_asset_task() -> str | None:
    """Atomically claim the next eligible task (QUEUED or RUNNING submitted GENERATION).

    Fair scheduling: oldest eligible task first, regardless of kind.

    Eligible tasks:
    - QUEUED: ENHANCEMENT (ffmpeg) or GENERATION (local_ltx)
    - RUNNING: GENERATION (local_ltx) with external_task_id and poll due
    """
    with Session(engine) as session:
        # Build eligibility filter for QUEUED tasks
        from sqlalchemy import or_
        
        # Get the oldest eligible QUEUED task
        queued_task = session.exec(
            select(AssetTask).where(
                AssetTask.status == AssetTaskStatus.QUEUED.value,
                or_(
                    # ENHANCEMENT with ffmpeg
                    (AssetTask.kind == AssetTaskKind.ENHANCEMENT.value) & 
                    (AssetTask.provider == "ffmpeg"),
                    # GENERATION with local_ltx
                    (AssetTask.kind == AssetTaskKind.GENERATION.value) & 
                    (AssetTask.provider == GenerationProvider.LOCAL_LTX.value),
                ),
            ).order_by(AssetTask.created_at)
        ).first()

        # Also check for RUNNING GENERATION tasks that need reconciliation
        running_task = session.exec(
            select(AssetTask).where(
                AssetTask.status == AssetTaskStatus.RUNNING.value,
                AssetTask.kind == AssetTaskKind.GENERATION.value,
                AssetTask.provider == GenerationProvider.LOCAL_LTX.value,
                AssetTask.external_task_id.is_not(None),
                (AssetTask.next_poll_at.is_(None)) | (AssetTask.next_poll_at <= datetime.now(UTC)),
            ).order_by(AssetTask.created_at)
        ).first()

        # Choose the oldest eligible task
        tasks = []
        if queued_task:
            tasks.append(queued_task)
        if running_task:
            tasks.append(running_task)

        if not tasks:
            return None

        task = min(tasks, key=lambda t: t.created_at)

        if not _is_task_eligible(task):
            return None

        task_id = task.id

    with engine.begin() as conn:
        # Build dynamic query based on task kind/provider
        kind = task.kind
        provider = task.provider
        if task.status == AssetTaskStatus.QUEUED.value:
            result = conn.execute(
                text(
                    "UPDATE assettask SET status = 'RUNNING', updated_at = :updated "
                    "WHERE id = :id AND status = 'QUEUED' AND kind = :kind "
                    "AND provider = :provider"
                ),
                {
                    "id": task_id,
                    "updated": datetime.now(UTC).isoformat(),
                    "kind": kind,
                    "provider": provider,
                },
            )
        else:
            # RUNNING task - use lease-based claim with next_poll_at as lease
            result = conn.execute(
                text(
                    "UPDATE assettask SET updated_at = :updated, next_poll_at = :lease_until "
                    "WHERE id = :id AND status = 'RUNNING' AND kind = :kind "
                    "AND provider = :provider "
                    "AND external_task_id IS NOT NULL "
                    "AND (next_poll_at IS NULL OR next_poll_at <= :now)"
                ),
                {
                    "id": task_id,
                    "updated": datetime.now(UTC).isoformat(),
                    "kind": kind,
                    "provider": provider,
                    "lease_until": datetime.now(UTC) + timedelta(seconds=30),
                    "now": datetime.now(UTC),
                },
            )
    return task_id if result.rowcount == 1 else None


def process_asset_task(task_id: str) -> None:
    """Process a claimed asset task (status must be RUNNING).

    Performs exactly one step: submit, poll, or finalize.
    """
    from app.core.sanitize import sanitize_error_text
    from app.editor.asset_tasks import AssetTaskStatus
    from app.editor.generation import (
        GenerationProvider,
        LocalGenerationUnavailable,
        LocalRuntimeNotAvailable,
        LocalRuntimeNotReady,
        get_local_generation_runtime,
    )
    from app.editor.generation_task_service import _finalize_commit

    with Session(engine) as session:
        try:
            task = session.get(AssetTask, task_id)
            if task is None:
                print(f"[worker] Task {task_id} not found")
                return

            if task.kind == AssetTaskKind.ENHANCEMENT.value:
                asset_task_service.execute_claimed_enhancement_task(session, task_id)
            elif task.kind == AssetTaskKind.GENERATION.value:
                # Validate provider: only local_ltx is supported for execution
                if task.provider != GenerationProvider.LOCAL_LTX.value:
                    raise InvalidTaskTransition(
                        f"provider must be local_ltx, got {task.provider}"
                    )
                # Get real runtime from factory
                try:
                    runtime = get_local_generation_runtime()
                except (LocalRuntimeNotAvailable, LocalRuntimeNotReady) as exc:
                    # Convert factory exceptions to LOCAL_GENERATION_UNAVAILABLE for the task
                    raise LocalGenerationUnavailable(str(exc)) from exc
                generation_task_service.execute_claimed_generation_task(
                    session, task_id, local_runtime=runtime
                )
            else:
                print(f"[worker] Unknown task kind: {task.kind}")
        except Exception as exc:  # noqa: BLE001
            # Mark GENERATION task as FAILED if it errored
            try:
                with Session(engine) as error_session:
                    error_task = error_session.get(AssetTask, task_id)
                    if error_task and error_task.kind == AssetTaskKind.GENERATION.value:
                        print(f"[worker] Marking task {task_id} as FAILED due to: {exc}")
                        assert_transition(error_task.status, AssetTaskStatus.FAILED)
                        error_task.status = AssetTaskStatus.FAILED.value
                        error_msg = f"{type(exc).__name__}: {exc}"
                        error_task.error = sanitize_error_text(error_msg)[:2000]
                        error_task.progress = 0.0
                        error_task.updated_at = now()
                        error_session.add(error_task)
                        _finalize_commit(error_session)
                        print(f"[worker] Task {task_id} marked as FAILED")
            except Exception as e:
                print(f"[worker] Failed to mark task as FAILED: {e}")
                pass  # Best effort
            # The shared cores handle FAILED transition and commit internally
            # If it raises, it's an unexpected error - log and move on
            print(f"[worker] Unexpected error processing asset task {task_id}: {exc}")


def _schedule_next_poll(session: Session, task: AssetTask, interval_seconds: int = 2) -> None:
    """Schedule the next poll by setting next_poll_at."""
    task.next_poll_at = datetime.now(UTC) + timedelta(seconds=interval_seconds)
    session.add(task)
    session.commit()
    session.refresh(task)


def _probe_video(path):
    """Extract video metadata using ffprobe."""
    import json
    import subprocess
    from pathlib import Path

    path = Path(path)
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


def recover_asset_tasks() -> int:
    """Recover tasks that were RUNNING at worker crash.

    Strategy:
    - ENHANCEMENT (ffmpeg): RUNNING + output_asset_id is None -> QUEUED (retry)
      RUNNING + output_asset_id != None -> fail closed (incoherent state)
    - GENERATION (local_ltx): RUNNING + external_task_id is None -> QUEUED (retry, never submitted)
      RUNNING + external_task_id != None -> leave RUNNING for reconciliation polling
    - COMPLETED/FAILED/CANCELLED -> untouched
    """
    recovered = 0
    with engine.begin() as conn:
        # ENHANCEMENT tasks: no output_asset_id -> requeue
        result = conn.execute(
            text(
                "SELECT id FROM assettask "
                "WHERE kind = 'ENHANCEMENT' AND provider = 'ffmpeg' "
                "AND status = 'RUNNING' AND output_asset_id IS NULL"
            )
        )
        task_ids = [row[0] for row in result.fetchall()]
        if task_ids:
            placeholders = ",".join([f":id{i}" for i in range(len(task_ids))])
            params = {f"id{i}": tid for i, tid in enumerate(task_ids)}
            params["updated"] = datetime.now(UTC).isoformat()
            conn.execute(
                text(
                    f"UPDATE assettask SET status = 'QUEUED', progress = 0, "
                    f"updated_at = :updated WHERE id IN ({placeholders})"
                ),
                params,
            )
            recovered += len(task_ids)

        # Check for incoherent ENHANCEMENT RUNNING + output_asset_id != NULL
        incoherent = conn.execute(
            text(
                "SELECT id FROM assettask "
                "WHERE kind = 'ENHANCEMENT' AND provider = 'ffmpeg' "
                "AND status = 'RUNNING' AND output_asset_id IS NOT NULL"
            )
        ).fetchall()
        if incoherent:
            print(
                f"[worker] WARNING: {len(incoherent)} ENHANCEMENT task(s) in incoherent state "
                "(RUNNING with output_asset_id). Manual intervention required.",
                flush=True,
            )

        # GENERATION tasks: no external_task_id -> requeue (never submitted)
        # With external_task_id -> leave RUNNING for reconciliation polling
        gen_result = conn.execute(
            text(
                "SELECT id FROM assettask "
                "WHERE kind = 'GENERATION' AND provider = 'local_ltx' "
                "AND status = 'RUNNING' AND external_task_id IS NULL"
            )
        )
        gen_task_ids = [row[0] for row in gen_result.fetchall()]
        if gen_task_ids:
            placeholders = ",".join([f":id{i}" for i in range(len(gen_task_ids))])
            params = {f"id{i}": tid for i, tid in enumerate(gen_task_ids)}
            params["updated"] = datetime.now(UTC).isoformat()
            conn.execute(
                text(
                    f"UPDATE assettask SET status = 'QUEUED', progress = 0, "
                    f"updated_at = :updated WHERE id IN ({placeholders})"
                ),
                params,
            )
            recovered += len(gen_task_ids)

        # Check for GENERATION with external_task_id -> leave RUNNING for reconciliation
        gen_incoherent = conn.execute(
            text(
                "SELECT id FROM assettask "
                "WHERE kind = 'GENERATION' AND provider = 'local_ltx' "
                "AND status = 'RUNNING' AND external_task_id IS NOT NULL"
            )
        ).fetchall()
        if gen_incoherent:
            print(
                f"[worker] INFO: {len(gen_incoherent)} GENERATION task(s) in RUNNING "
                "with external_task_id. Will reconcile on next poll.",
                flush=True,
            )

    return recovered