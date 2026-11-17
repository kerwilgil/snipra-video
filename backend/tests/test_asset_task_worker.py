"""Tests for AssetTask worker processing (SNIPRA 1.6C Block 04)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, delete, select

from app.core.db import engine, init_db
from app.editor import asset_task_service
from app.editor.asset_provenance import (
    AssetOrigin,
    read_asset_provenance,
    source_provenance,
    write_asset_provenance,
)
from app.editor.asset_tasks import AssetTaskKind, AssetTaskStatus
from app.editor.enhancement import EnhancementProfile, EnhancementRequest
from app.editor.models import AssetTask, EditorAsset, EditorProject
from app.models import (
    ClipOperation,
    ClipOperationStatus,
    Compilation,
    CompilationStatus,
    Job,
    JobStatus,
)
from app.worker import asset_tasks as worker_asset_tasks
from app.worker.runner import claim_next as claim_next_job


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    init_db()


@pytest.fixture
def session():
    """Provide a clean session for each test with automatic rollback."""
    with Session(engine) as s:
        # Clean up any existing test data
        s.exec(delete(AssetTask))
        s.exec(delete(EditorAsset))
        s.exec(delete(EditorProject))
        s.exec(delete(Job))
        s.exec(delete(ClipOperation))
        s.exec(delete(Compilation))
        s.commit()
        yield s
        s.rollback()


@pytest.fixture
def project_with_source_asset(session: Session):
    """Create a project with a SOURCE asset."""
    project = EditorProject(name="Test Project")
    session.add(project)
    session.flush()

    # Create a SOURCE asset directly with proper provenance
    asset = EditorAsset(
        project_id=project.id,
        original_path="/fake/source.mp4",
        filename="source.mp4",
        fingerprint="src123",
        kind="video",
        duration_us=5_000_000,
        width=1280,
        height=720,
        has_audio=True,
        asset_metadata={},
    )
    session.add(asset)
    session.flush()
    
    # Set provenance with actual asset id
    asset.asset_metadata = write_asset_provenance({}, source_provenance(asset_id=asset.id))
    session.add(asset)
    session.commit()
    session.refresh(project)
    session.refresh(asset)
    return project, asset


class FakeProvider:
    """Fake enhancement provider for deterministic testing."""
    
    def __init__(self, should_fail=False, error_message="fake failure"):
        self.should_fail = should_fail
        self.error_message = error_message
        self.enhance_called = False
        self.last_request = None
        self.last_output_path = None
        
    def enhance(self, request, output_path):
        self.enhance_called = True
        self.last_request = request
        self.last_output_path = output_path
        
        if self.should_fail:
            raise ValueError(self.error_message)
        
        # Create a fake output file
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake enhanced content")
        
        return {
            "output_path": output_path,
            "output_hash": "fake_hash",
            "output_size_bytes": 100,
            "duration_us": 5_000_000,
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "has_audio": True,
            "provider": "ffmpeg",
            "model_or_engine": "fake provider",
            "request_hash": "fake_request_hash",
            "operation_version": "enh_v1",
            "parameters": {"profile": "DEFAULT"},
        }
    
    def capabilities(self):
        return MagicMock(
            supports_upscale=True,
            supports_sharpen=True,
            supports_color_adjustment=True,
            supports_denoise=True,
            supports_gpu=False,
            requires_external_binary=False,
        )


class TestAtomicClaim:
    """Tests for atomic claim of asset tasks."""

    def test_claim_returns_task_id_when_queued(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        # Verify status changed to RUNNING
        session.refresh(task)
        assert task.status == AssetTaskStatus.RUNNING.value

    def test_double_claim_returns_none_for_second(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # First claim succeeds
        first = worker_asset_tasks.claim_next_asset_task()
        assert first == task.id

        # Second claim returns None (already RUNNING)
        second = worker_asset_tasks.claim_next_asset_task()
        assert second is None

    def test_only_queued_can_be_claimed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        # NOT queued

        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed is None

    def test_only_local_ltx_claimed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        # Create a GENERATION task with cloud provider (should not be claimed)
        cloud_task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.GENERATION.value,
            status=AssetTaskStatus.QUEUED.value,
            provider="runway",
            request={"prompt": "test"},
            request_hash="gen_hash",
            approval_state="APPROVED",
            approved_request_hash="gen_hash",
        )
        session.add(cloud_task)
        session.commit()

        claimed = worker_asset_tasks.claim_next_asset_task()
        # Should not claim GENERATION task with cloud provider
        assert claimed is None

    def test_local_ltx_generation_claimed(self, session: Session, project_with_source_asset):
        """GENERATION tasks with local_ltx provider should be claimed."""
        from app.editor.generation import GenerationModel, GenerationProvider
        project, asset = project_with_source_asset
        gen_task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.GENERATION.value,
            status=AssetTaskStatus.QUEUED.value,
            provider=GenerationProvider.LOCAL_LTX.value,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8.value,
            request={"prompt": "test"},
            request_hash="gen_hash",
            approval_state="APPROVED",
            approved_request_hash="gen_hash",
        )
        session.add(gen_task)
        session.commit()

        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed == gen_task.id

    def test_other_kind_untouched(self, session: Session, project_with_source_asset):
        """Verify GENERATION tasks with cloud provider remain in QUEUED and are not processed."""
        project, asset = project_with_source_asset
        gen_task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.GENERATION.value,
            status=AssetTaskStatus.QUEUED.value,
            provider="runway",
            request={"prompt": "test"},
            request_hash="gen_hash",
            approval_state="APPROVED",
            approved_request_hash="gen_hash",
        )
        session.add(gen_task)
        session.commit()

        # Run claim loop - should only claim ENHANCEMENT and local_ltx GENERATION
        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed is None

        # Verify GENERATION task with cloud provider remains QUEUED
        session.refresh(gen_task)
        assert gen_task.status == AssetTaskStatus.QUEUED.value


class TestWorkerRevalidation:
    """Tests for worker revalidation before execution."""

    def test_approval_revalidated_before_execute(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # Change request after queuing by directly modifying the DB
        # This simulates a scenario where the request was changed externally
        from sqlmodel import select
        task_db = session.exec(select(AssetTask).where(AssetTask.id == task.id)).first()
        new_request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.UPSCALE,
            upscale_factor=2.0,
        )
        task_db.request = new_request.model_dump(mode="json")
        task_db.request_hash = new_request.request_hash()
        session.add(task_db)
        session.commit()

        # Claim and process - should fail due to invalid approval
        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        worker_asset_tasks.process_asset_task(claimed_id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert "approval" in task.error.lower() or "hash" in task.error.lower()

    def test_hash_revalidated_before_execute(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # Manually corrupt the approved hash
        from sqlmodel import select
        task_db = session.exec(select(AssetTask).where(AssetTask.id == task.id)).first()
        task_db.approved_request_hash = "corrupted_hash"
        session.add(task_db)
        session.commit()

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        worker_asset_tasks.process_asset_task(claimed_id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value

    def test_cost_estimate_revalidated(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # Clear cost estimate
        task.cost_estimate = {}
        session.add(task)
        session.commit()

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        worker_asset_tasks.process_asset_task(claimed_id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value


class TestWorkerFailClosed:
    """Tests for fail-closed behavior when RUNNING task has invalid kind/provider."""

    def test_running_generation_wrong_provider_fails(self, session: Session, project_with_source_asset):
        """A RUNNING GENERATION task with ffmpeg provider must transition to FAILED."""
        project, asset = project_with_source_asset
        # Create a RUNNING task with GENERATION kind and ffmpeg provider (invalid)
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.GENERATION.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="wrong_provider_hash",
            approval_state="APPROVED",
            approved_request_hash="wrong_provider_hash",
        )
        session.add(task)
        session.commit()

        # Process should fail and transition to FAILED
        worker_asset_tasks.process_asset_task(task.id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert "provider must be local_ltx" in task.error.lower()

    def test_running_enhancement_wrong_provider_fails(self, session: Session, project_with_source_asset):
        """A RUNNING ENHANCEMENT task with wrong provider must transition to FAILED."""
        project, asset = project_with_source_asset
        # Create a RUNNING task with ENHANCEMENT kind and wrong provider
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="runway",  # Wrong provider
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="wrong_provider_hash",
            approval_state="APPROVED",
            approved_request_hash="wrong_provider_hash",
        )
        session.add(task)
        session.commit()

        # Process should fail and transition to FAILED
        worker_asset_tasks.process_asset_task(task.id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert "provider" in task.error.lower() or "ffmpeg" in task.error.lower()

    def test_approval_invalid_fails(self, session: Session, project_with_source_asset):
        """Invalid approval must cause FAILED transition."""
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="approval_hash",
            approval_state="PENDING",  # Not APPROVED
            approved_request_hash="different_hash",  # Doesn't match
        )
        session.add(task)
        session.commit()

        worker_asset_tasks.process_asset_task(task.id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert "approval" in task.error.lower() or "hash" in task.error.lower()

    def test_cost_invalid_fails(self, session: Session, project_with_source_asset):
        """Invalid cost estimate must cause FAILED transition."""
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="cost_hash",
            approval_state="APPROVED",
            approved_request_hash="cost_hash",
            cost_estimate={},  # Empty - invalid
        )
        session.add(task)
        session.commit()

        worker_asset_tasks.process_asset_task(task.id)

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert "cost" in task.error.lower() or "estimate" in task.error.lower()


class TestWorkerProcessing:
    """Tests for successful and failed processing."""

    def test_successful_processing_completes_task(
        self, session: Session, project_with_source_asset
    ):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        # Use fake provider for deterministic success
        fake_provider = FakeProvider(should_fail=False)
        
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            worker_asset_tasks.process_asset_task(claimed_id)
        
        session.refresh(task)
        assert task.status == AssetTaskStatus.COMPLETED.value
        assert task.output_asset_id is not None
        assert task.model == "fake provider"
        assert task.progress == 1.0
        assert task.error is None

        # Verify exactly one new ENHANCED asset was created
        enhanced_assets = session.exec(
            select(EditorAsset).where(EditorAsset.project_id == project.id)
        ).all()
        # Should have original SOURCE + 1 new ENHANCED = 2 total
        assert len(enhanced_assets) == 2
        
        # Find the enhanced asset
        enhanced = next(a for a in enhanced_assets if a.id == task.output_asset_id)
        # Verify provenance using canonical helper
        prov = read_asset_provenance(enhanced.asset_metadata)
        assert prov is not None
        assert prov.origin == AssetOrigin.ENHANCED
        assert prov.parent_asset_id == asset.id
        assert prov.provider == "ffmpeg"
        assert prov.model_or_engine == "fake provider"
        assert prov.request_hash == "fake_request_hash"
        assert prov.output_hash == "fake_hash"
        assert prov.output_size_bytes == 100
        assert prov.operation_version == "enh_v1"
        assert prov.parameters == {"profile": "DEFAULT"}
        assert prov.duration_us == 5_000_000
        assert prov.width == 1920
        assert prov.height == 1080
        assert prov.fps == 30.0
        
        # Source asset should still exist and be unchanged
        session.refresh(asset)
        source_prov = read_asset_provenance(asset.asset_metadata)
        assert source_prov is not None
        assert source_prov.origin == AssetOrigin.SOURCE
        assert source_prov.root_source_asset_id == asset.id
        # Verify fingerprint/path/metadata unchanged
        assert asset.fingerprint == "src123"
        assert asset.original_path == "/fake/source.mp4"

    def test_failure_marks_task_failed_without_phantom_asset(
        self, session: Session, project_with_source_asset
    ):
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        # Use fake provider that fails
        fake_provider = FakeProvider(should_fail=True, error_message="simulated failure")
        
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            worker_asset_tasks.process_asset_task(claimed_id)
        
        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value
        assert task.output_asset_id is None
        assert task.error is not None
        assert "simulated failure" in task.error

        # Verify no new ENHANCED assets were created
        enhanced_assets = session.exec(
            select(EditorAsset).where(EditorAsset.project_id == project.id)
        ).all()
        # Should only have the original SOURCE asset
        assert len(enhanced_assets) == 1
        assert enhanced_assets[0].id == asset.id

        # Source asset should be intact
        session.refresh(asset)
        assert asset.fingerprint == "src123"


class TestRecovery:
    """Tests for worker recovery at startup."""

    def test_recovery_running_no_output_requeues(
        self, session: Session, project_with_source_asset
    ):
        project, asset = project_with_source_asset
        # Create a RUNNING task with no output asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="recovery_hash",
            approval_state="APPROVED",
            approved_request_hash="recovery_hash",
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        recovered = worker_asset_tasks.recover_asset_tasks()
        assert recovered == 1

        session.refresh(task)
        assert task.status == AssetTaskStatus.QUEUED.value

    def test_recovery_running_with_output_fails_closed(
        self, session: Session, project_with_source_asset
    ):
        project, asset = project_with_source_asset
        # Create a RUNNING task with output asset (incoherent state)
        output_asset = EditorAsset(
            project_id=project.id,
            original_path="/fake/enhanced.mp4",
            filename="enhanced.mp4",
            fingerprint="enh123",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
        )
        session.add(output_asset)
        session.flush()

        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="recovery_hash",
            approval_state="APPROVED",
            approved_request_hash="recovery_hash",
            output_asset_id=output_asset.id,
        )
        session.add(task)
        session.commit()

        recovered = worker_asset_tasks.recover_asset_tasks()
        # Should NOT requeue - incoherent state
        assert recovered == 0

        session.refresh(task)
        assert task.status == AssetTaskStatus.RUNNING.value

    def test_recovery_does_not_touch_completed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        output_asset = EditorAsset(
            project_id=project.id,
            original_path="/fake/enhanced.mp4",
            filename="enhanced.mp4",
            fingerprint="enh123",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
        )
        session.add(output_asset)
        session.flush()

        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.COMPLETED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="completed_hash",
            approval_state="APPROVED",
            approved_request_hash="completed_hash",
            output_asset_id=output_asset.id,
        )
        session.add(task)
        session.commit()

        recovered = worker_asset_tasks.recover_asset_tasks()
        assert recovered == 0

        session.refresh(task)
        assert task.status == AssetTaskStatus.COMPLETED.value

    def test_recovery_does_not_touch_failed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.FAILED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="failed_hash",
            approval_state="APPROVED",
            approved_request_hash="failed_hash",
            error="Previous failure",
        )
        session.add(task)
        session.commit()

        recovered = worker_asset_tasks.recover_asset_tasks()
        assert recovered == 0

        session.refresh(task)
        assert task.status == AssetTaskStatus.FAILED.value

    def test_recovery_does_not_touch_cancelled(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.CANCELLED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="cancelled_hash",
            approval_state="APPROVED",
            approved_request_hash="cancelled_hash",
        )
        session.add(task)
        session.commit()

        recovered = worker_asset_tasks.recover_asset_tasks()
        assert recovered == 0

        session.refresh(task)
        assert task.status == AssetTaskStatus.CANCELLED.value


class TestTerminalStates:
    """Tests that terminal states are not reprocessed."""

    def test_completed_task_cannot_be_rerun(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        output_asset = EditorAsset(
            project_id=project.id,
            original_path="/fake/enhanced.mp4",
            filename="enhanced.mp4",
            fingerprint="enh123",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
        )
        session.add(output_asset)
        session.flush()

        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.COMPLETED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="completed_hash",
            approval_state="APPROVED",
            approved_request_hash="completed_hash",
            output_asset_id=output_asset.id,
        )
        session.add(task)
        session.commit()

        # Try to claim - should not work
        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed is None

    def test_failed_task_not_claimed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.FAILED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="failed_hash",
            approval_state="APPROVED",
            approved_request_hash="failed_hash",
            error="Previous error",
        )
        session.add(task)
        session.commit()

        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed is None

    def test_cancelled_task_not_claimed(self, session: Session, project_with_source_asset):
        project, asset = project_with_source_asset
        task = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.ENHANCEMENT.value,
            status=AssetTaskStatus.CANCELLED.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="cancelled_hash",
            approval_state="APPROVED",
            approved_request_hash="cancelled_hash",
        )
        session.add(task)
        session.commit()

        claimed = worker_asset_tasks.claim_next_asset_task()
        assert claimed is None


class TestWorkerSessionOwnership:
    """Tests that worker uses its own Session."""

    def test_process_uses_own_session(self, session: Session, project_with_source_asset):
        """Verify process_asset_task opens its own Session."""
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        claimed_id = worker_asset_tasks.claim_next_asset_task()
        assert claimed_id == task.id

        # This should not fail due to session issues
        fake_provider = FakeProvider(should_fail=False)
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            worker_asset_tasks.process_asset_task(claimed_id)
        # If we get here without session errors, test passes


class TestWorkerFairness:
    """Tests for worker round-robin fairness - testing the REAL runner scheduler."""

    def test_fairness_job_progresses(self, session: Session, project_with_source_asset):
        """Job category progresses when it has work."""
        project, asset = project_with_source_asset
        
        job = Job(
            id="test-job-fairness",
            project_id=project.id,
            status=JobStatus.queued,
            stage="En cola",
            progress=0.0,
            source_type="upload",
            source_name="test.mp4",
            title="Test Job",
            output_mode="clips",
        )
        session.add(job)
        session.commit()

        # Call claim_next directly from runner (which uses the real claim_next)
        job_id = claim_next_job()
        assert job_id == "test-job-fairness"

    def test_fairness_operation_progresses(self, session: Session, project_with_source_asset):
        """Operation category progresses when it has work."""
        project, asset = project_with_source_asset
        
        job = Job(
            id="test-job-op",
            project_id=project.id,
            status=JobStatus.queued,
            stage="En cola",
            progress=0.0,
            source_type="upload",
            source_name="test.mp4",
            title="Test Job",
            output_mode="clips",
        )
        session.add(job)
        session.commit()
        
        clip_op = ClipOperation(
            id="test-clip-op-fair",
            job_id=job.id,
            clip_id="clip1",
            kind="style",
            status=ClipOperationStatus.queued,
            progress=0.0,
        )
        session.add(clip_op)
        session.commit()

        # Import the real claim_next_operation from clip_operations
        from app.worker.clip_operations import claim_next_operation
        op_id = claim_next_operation()
        assert op_id == "test-clip-op-fair"

    def test_fairness_compilation_progresses(self, session: Session, project_with_source_asset):
        """Compilation category progresses when it has work."""
        project, asset = project_with_source_asset
        
        compilation = Compilation(
            id="test-comp-fair",
            project_id=project.id,
            asset_ids=[asset.id],
            status=CompilationStatus.queued,
            progress=0.0,
        )
        session.add(compilation)
        session.commit()

        from app.worker.compositions import claim_next_compilation
        comp_id = claim_next_compilation()
        assert comp_id == "test-comp-fair"

    def test_fairness_asset_task_progresses(self, session: Session, project_with_source_asset):
        """Asset task category progresses when it has work."""
        project, asset = project_with_source_asset
        
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)
        session.commit()

        asset_task_id = worker_asset_tasks.claim_next_asset_task()
        assert asset_task_id == task.id

    def test_fairness_no_starvation_when_all_queues_busy(self, session: Session, project_with_source_asset):
        """All four categories progress when all have continuous work."""
        project, asset = project_with_source_asset
        
        # Create multiple items in each category
        from app.models import (
            ClipOperation,
            ClipOperationStatus,
            Compilation,
            CompilationStatus,
            Job,
            JobStatus,
        )
        
        # Create 3 jobs
        for i in range(3):
            job = Job(
                id=f"test-job-{i}",
                project_id=project.id,
                status=JobStatus.queued,
                stage="En cola",
                progress=0.0,
                source_type="upload",
                source_name=f"test{i}.mp4",
                title=f"Test Job {i}",
                output_mode="clips",
            )
            session.add(job)
        
        # Create 3 clip operations
        for i in range(3):
            clip_op = ClipOperation(
                id=f"test-clip-op-{i}",
                job_id=f"test-job-{i % 3}",
                clip_id=f"clip{i}",
                kind="style",
                status=ClipOperationStatus.queued,
                progress=0.0,
            )
            session.add(clip_op)
        
        # Create 3 compilations
        for i in range(3):
            compilation = Compilation(
                id=f"test-comp-{i}",
                project_id=project.id,
                asset_ids=[asset.id],
                status=CompilationStatus.queued,
                progress=0.0,
            )
            session.add(compilation)
        
        # Create 3 asset tasks
        for i in range(3):
            request = EnhancementRequest(
                source_asset_id=asset.id,
                profile=EnhancementProfile.DEFAULT,
            )
            task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
            task = asset_task_service.attach_estimate(session, task.id)
            task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
            task = asset_task_service.queue_task(session, task.id)
        session.commit()
        
        # Use a fake provider that succeeds quickly
        fake_provider = FakeProvider(should_fail=False)
        
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            # Import real claimers
            from app.worker.clip_operations import claim_next_operation
            from app.worker.compositions import claim_next_compilation
            from app.worker.runner import claim_next as claim_next_job
            
            # Simulate the runner's round-robin: each iteration tries one work type
            processed = {"job": 0, "operation": 0, "compilation": 0, "asset_task": 0}
            max_iterations = 50
            
            for _ in range(max_iterations):
                # Try each category in rotation (simulating runner's WORK_TYPES)
                # job
                job_id = claim_next_job()
                if job_id:
                    processed["job"] += 1
                    continue
                
                # operation
                op_id = claim_next_operation()
                if op_id:
                    processed["operation"] += 1
                    continue
                
                # compilation
                comp_id = claim_next_compilation()
                if comp_id:
                    processed["compilation"] += 1
                    continue
                
                # asset_task
                asset_task_id = worker_asset_tasks.claim_next_asset_task()
                if asset_task_id:
                    processed["asset_task"] += 1
                    # Process with fake provider
                    with patch("app.editor.ffmpeg_enhancement.FFmpegEnhancementProvider", return_value=fake_provider):
                        worker_asset_tasks.process_asset_task(asset_task_id)
                    continue
                
                # If nothing was claimed in this round, break
                if sum(processed.values()) == 0:
                    break
            
            # All categories should have made progress
            assert processed["job"] >= 1, "Job should have been processed"
            assert processed["operation"] >= 1, "Operation should have been processed"
            assert processed["compilation"] >= 1, "Compilation should have been processed"
            assert processed["asset_task"] >= 1, "Asset task should have been processed"

    def test_fairness_empty_category_skipped(self, session: Session, project_with_source_asset):
        """Verify empty categories are skipped in round-robin."""
        project, asset = project_with_source_asset
        
        # Only create asset task
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)
        session.commit()
        
        # Only asset tasks should be processed
        fake_provider = FakeProvider(should_fail=False)
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            # Simulate multiple rounds - only asset tasks should be claimed
            for _ in range(5):
                from app.worker.clip_operations import claim_next_operation
                from app.worker.compositions import claim_next_compilation
                from app.worker.runner import claim_next as claim_next_job
                
                assert claim_next_job() is None
                assert claim_next_operation() is None
                assert claim_next_compilation() is None
                
                claimed = worker_asset_tasks.claim_next_asset_task()
                if claimed:
                    with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=FakeProvider(should_fail=False)):
                        worker_asset_tasks.process_asset_task(claimed)
                    break

    def test_fairness_claimer_exception_does_not_break_rotation(self, session: Session, project_with_source_asset):
        """Exception in one claimer should not break round-robin."""
        project, asset = project_with_source_asset
        
        # Create an asset task
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)
        session.commit()
        
        # Create a job as well to verify rotation continues past the failing category
        job = Job(
            id="test-job-exception",
            project_id=project.id,
            status=JobStatus.queued,
            stage="En cola",
            progress=0.0,
            source_type="upload",
            source_name="test.mp4",
            title="Test Job",
            output_mode="clips",
        )
        session.add(job)
        session.commit()
        
        # Mock claim_next_asset_task to raise exception once, then succeed
        call_count = [0]
        original_claim = worker_asset_tasks.claim_next_asset_task
        
        def flaky_claim():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated claim error")
            return original_claim()
        
        from app.worker.runner import WORK_TYPES, _try_one_work_round
        
        # Mock the processor functions to avoid actual processing
        mock_process_job = MagicMock()
        mock_process_asset_task = MagicMock()
        
        with patch("app.worker.asset_tasks.claim_next_asset_task", side_effect=flaky_claim):
            # First call: start at asset_task (index 3), claimer raises, should continue to job (index 0)
            work_done, work_type, work_id, next_index = _try_one_work_round(
                3, WORK_TYPES,
                process_job_fn=mock_process_job,
                process_asset_task_fn=mock_process_asset_task,
            )
            assert work_done is True
            assert work_type == "job"
            assert work_id == "test-job-exception"
            mock_process_job.assert_called_once_with("test-job-exception", claimed=True)
            
            # Second call: asset_task claimer succeeds, should get asset task
            work_done, work_type, work_id, next_index = _try_one_work_round(
                next_index, WORK_TYPES,
                process_job_fn=mock_process_job,
                process_asset_task_fn=mock_process_asset_task,
            )
            assert work_done is True
            assert work_type == "asset_task"
            assert work_id == task.id
            mock_process_asset_task.assert_called_once_with(task.id)

    def test_fairness_no_double_claim(self, session: Session, project_with_source_asset):
        """Same task should not be claimed twice."""
        project, asset = project_with_source_asset
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)
        session.commit()
        
        fake_provider = FakeProvider(should_fail=False)
        with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
            # First claim
            claimed1 = worker_asset_tasks.claim_next_asset_task()
            assert claimed1 == task.id
            
            # Second claim should return None (already RUNNING)
            claimed2 = worker_asset_tasks.claim_next_asset_task()
            assert claimed2 is None
            
            # Process the task
            with patch("app.editor.asset_task_service.FFmpegEnhancementProvider", return_value=fake_provider):
                worker_asset_tasks.process_asset_task(claimed1)
            
            # Should not be able to claim again
            claimed3 = worker_asset_tasks.claim_next_asset_task()
            assert claimed3 is None

    def test_fairness_when_no_work_sleeps(self, session: Session):
        """When no category has work, worker takes the idle path (sleeps)."""
        from app.worker.runner import POLL_SECONDS, _sleep_if_idle
        
        # Test idle path: work_done=False should call time.sleep(POLL_SECONDS)
        with patch("app.worker.runner.time.sleep") as mock_sleep:
            _sleep_if_idle(False)
            mock_sleep.assert_called_once_with(POLL_SECONDS)
        
        # Test busy path: work_done=True should NOT call time.sleep
        with patch("app.worker.runner.time.sleep") as mock_sleep:
            _sleep_if_idle(True)
            mock_sleep.assert_not_called()


class TestAssetTaskFairness:
    """Asset task specific fairness tests for GENERATION/ENHANCEMENT."""

    def test_fairness_generation_runway_never_blocks_local(
        self, session: Session, project_with_source_asset
    ):
        """Cloud GENERATION (runway) never blocks local_ltx or ENHANCEMENT (ffmpeg)."""
        project, asset = project_with_source_asset
        
        from app.editor.generation import GenerationProvider, GenerationRequest, GenerationModel
        from app.editor.generation import GenerationMode, GenerationResolution, GenerationAspectRatio
        from app.editor import generation_task_service
        from app.editor.enhancement import EnhancementRequest, EnhancementProfile
        from app.editor import asset_task_service
        from app.editor.asset_tasks import AssetTaskStatus, AssetTaskKind, AssetTaskApprovalState
        from app.editor.models import AssetTask, now
        from datetime import UTC, datetime
        
        # 1. Create GENERATION runway QUEUED (oldest) - bypass service to avoid local estimate check
        gen_request_runway = GenerationRequest(
            project_id=project.id,
            provider=GenerationProvider.RUNWAY,
            model=GenerationModel.RUNWAY_GEN45,
            prompt_intent="Test runway",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task_runway = AssetTask(
            project_id=project.id,
            kind=AssetTaskKind.GENERATION.value,
            status=AssetTaskStatus.QUEUED.value,
            provider=GenerationProvider.RUNWAY.value,
            request_hash=gen_request_runway.request_hash(),
            payload=gen_request_runway.canonical_payload(),
            approval_state=AssetTaskApprovalState.APPROVED.value,
            approved_request_hash=gen_request_runway.request_hash(),
            created_at=now(),
            updated_at=now(),
        )
        session.add(task_runway)
        session.flush()
        
        # 2. Create ENHANCEMENT ffmpeg QUEUED (later)
        enh_request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task_enh = asset_task_service.create_enhancement_task(session, project_id=project.id, request=enh_request)
        task_enh = asset_task_service.attach_estimate(session, task_enh.id)
        task_enh = asset_task_service.approve_task(session, task_enh.id, request_hash=task_enh.request_hash)
        task_enh = asset_task_service.queue_task(session, task_enh.id)
        
        # 3. Create GENERATION local_ltx QUEUED (later)
        gen_request_local = GenerationRequest(
            project_id=project.id,
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            prompt_intent="Test local",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task_local = generation_task_service.create_generation_task(
            session, project_id=project.id, request=gen_request_local
        )
        task_local = generation_task_service.attach_generation_estimate(session, task_local.id)
        task_local = generation_task_service.approve_generation_task(
            session, task_local.id, request_hash=gen_request_local.request_hash()
        )
        task_local = generation_task_service.queue_generation_task(session, task_local.id)
        
        session.commit()
        
        # Verify all tasks are QUEUED
        assert task_runway.status == "QUEUED"
        assert task_enh.status == "QUEUED"
        assert task_local.status == "QUEUED"
        
        # First claim should be ENHANCEMENT (ffmpeg) - it's the first ELIGIBLE
        claimed1 = worker_asset_tasks.claim_next_asset_task()
        assert claimed1 == task_enh.id, f"Expected enhancement task, got {claimed1}"
        
        # Second claim should be GENERATION local_ltx - next eligible
        claimed2 = worker_asset_tasks.claim_next_asset_task()
        assert claimed2 == task_local.id, f"Expected local generation task, got {claimed2}"
        
        # Cloud RUNWAY should remain QUEUED (never claimed by asset task worker)
        claimed3 = worker_asset_tasks.claim_next_asset_task()
        assert claimed3 is None, f"Cloud task should not be claimed, got {claimed3}"
        
        # Verify runway task still QUEUED
        session.refresh(task_runway)
        assert task_runway.status == "QUEUED"


class TestRunningLease:
    """Tests for RUNNING lease reconciliation with next_poll_at."""

    def test_running_lease_reconciliation(
        self, session: Session, project_with_source_asset
    ):
        """RUNNING task with expired next_poll_at can be reclaimed with new lease."""
        project, asset = project_with_source_asset
        
        from app.editor.generation import GenerationProvider, GenerationRequest
        from app.editor.generation import GenerationMode, GenerationResolution, GenerationAspectRatio
        from app.editor import generation_task_service
        from app.editor.asset_tasks import AssetTaskStatus
        from datetime import UTC, datetime, timedelta
        
        # Create GENERATION local_ltx task and advance to RUNNING
        gen_request = GenerationRequest(
            project_id=project.id,
            provider=GenerationProvider.LOCAL_LTX,
            prompt_intent="Test local",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=gen_request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=gen_request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)
        
        # Manually set to RUNNING with expired next_poll_at
        task.status = AssetTaskStatus.RUNNING.value
        task.external_task_id = "ext-task-123"
        task.external_task_state = "RUNNING"
        task.next_poll_at = datetime.now(UTC) - timedelta(seconds=10)  # EXPIRED
        session.add(task)
        session.commit()
        session.refresh(task)
        
        # First claim should succeed and update next_poll_at to future lease
        claimed1 = worker_asset_tasks.claim_next_asset_task()
        assert claimed1 == task.id, f"Expected task {task.id}, got {claimed1}"
        
        session.refresh(task)
        assert task.next_poll_at is not None
        assert task.next_poll_at > datetime.now(UTC), "next_poll_at should be future lease"
        
        # Second immediate claim should NOT return the same task (lease not expired)
        claimed2 = worker_asset_tasks.claim_next_asset_task()
        assert claimed2 is None, f"Task should not be re-claimed immediately, got {claimed2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])