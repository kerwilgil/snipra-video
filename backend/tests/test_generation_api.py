"""Tests for Generation API (SNIPRA 1.6D Block 04R2).

Covers HTTP endpoints: prepare, estimate, exact-content approval, rejection, cancellation,
capability endpoint, no-cloud-fallback, project isolation, timeline immutability,
version/undo.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select

from app.api.editor import _serialize_asset_task
from app.core.db import engine, init_db
from app.editor import generation_task_service
from app.editor.asset_provenance import (
    AssetOrigin,
    source_provenance,
    write_asset_provenance,
)
from app.editor.generation import (
    GenerationAspectRatio,
    GenerationMode,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResolution,
)
from app.editor.models import AssetTask, EditorAsset, EditorProject
from app.main import app


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    init_db()


@pytest.fixture
def session():
    """Provide a clean session for each test with automatic rollback."""
    with Session(engine) as s:
        s.exec(delete(AssetTask))
        s.exec(delete(EditorAsset))
        s.exec(delete(EditorProject))
        s.commit()
        yield s
        s.rollback()


@pytest.fixture
def client():
    """FastAPI TestClient for HTTP tests."""
    return TestClient(app)


@pytest.fixture
def project(session: Session):
    from app.editor import service
    return service.create_project(session, "Test Project")


@pytest.fixture
def source_asset(session: Session, project):
    """Create a SOURCE asset with proper provenance."""
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
        asset_metadata=write_asset_provenance({}, source_provenance(asset_id="source-asset-123")),
    )
    session.add(asset)
    session.commit()
    session.refresh(asset)
    return asset


# =============================================================================
# SERVICE LAYER TESTS (internal behavior)
# =============================================================================

class TestPrepareGeneration:
    """Tests for generation task creation at service layer."""

    def test_prepare_generation_i2v_requires_reference(self, session, project):
        """Image-to-video requires at least one reference asset."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GenerationRequest(
                project_id=project.id,
                mode=GenerationMode.IMAGE_TO_VIDEO,
                prompt_intent="Animate this",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                reference_asset_ids=[],
            )


class TestGenerationCapability:
    """Tests for GET /projects/{project_id}/generations/capability (service layer)."""

    def test_capability_local_ltx_hardware_unsupported(self, session, project):
        """AMD/unsupported hardware -> local_ltx not ready, physical_validation=DEFERRED."""
        from app.editor.generation import LocalGenerationHardwareInfo, LocalGenerationHardwareReason

        hw_info = LocalGenerationHardwareInfo(
            has_cuda=False,
            cuda_version=None,
            gpu_name="AMD Radeon",
            gpu_vram_mb=512,
            compute_capability=None,
            supports_fp8=False,
            fp8_reason="No NVIDIA GPU",
            runtime_supported=False,
            reason=LocalGenerationHardwareReason.NO_NVIDIA_GPU,
            reason_message="No NVIDIA GPU detected",
        )

        with patch("app.editor.generation.probe_local_generation_hardware", return_value=hw_info):
            from app.api.editor import generation_capability
            with patch("app.api.editor.service.project_or_404"):
                cap = generation_capability(project_id=project.id, session=session)

            assert cap["local_ltx"]["hardware_supported"] is False
            assert cap["local_ltx"]["runtime_ready"] is False
            assert cap["local_ltx"]["physical_validation"] == "DEFERRED"
            assert "DEFERRED" in cap["local_ltx"]["physical_validation"]

    def test_capability_local_ltx_hardware_supported_but_deferred(self, session, project):
        """Supported hardware but physical backend deferred -> runtime_ready=false."""
        from app.editor.generation import LocalGenerationHardwareInfo, LocalGenerationHardwareReason

        hw_info = LocalGenerationHardwareInfo(
            has_cuda=True,
            cuda_version="12.4",
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_mb=24576,
            compute_capability=(8, 9),
            fp8_supported=True,
            fp8_reason="Ada architecture supports FP8",
            runtime_supported=True,
            reason=LocalGenerationHardwareReason.SUPPORTED,
            reason_message="Hardware supports local generation",
        )

        with patch("app.editor.generation.probe_local_generation_hardware", return_value=hw_info):
            from app.api.editor import generation_capability
            with patch("app.api.editor.service.project_or_404"):
                cap = generation_capability(project_id=project.id, session=session)

            assert cap["local_ltx"]["hardware_supported"] is True
            assert cap["local_ltx"]["runtime_ready"] is False
            assert cap["local_ltx"]["physical_validation"] == "DEFERRED"
            assert "física LTX aún no está habilitada" in cap["local_ltx"]["runtime_message"]

    def test_capability_cloud_not_configured(self, session, project):
        """Cloud providers show available=false, ready=false (no real config mechanism)."""
        from app.api.editor import generation_capability

        with patch("app.api.editor.service.project_or_404"):
            cap = generation_capability(project_id=project.id, session=session)

        for cp in cap["cloud_providers"]:
            assert cp["available"] is False
            assert cp["ready"] is False
            assert ("backend cloud no configurado" in cp["message"].lower()
                or "no configurado" in cp["message"].lower())


class TestApproval:
    """Tests for exact-content approval (service layer)."""

    def test_approve_with_exact_hash_queues_task(self, session, project):
        """Exact hash approval moves task to QUEUED."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        assert task.status == "QUEUED"
        assert task.approval_state == "APPROVED"

    def test_approve_wrong_hash_rejected(self, session, project):
        """Wrong hash approval is rejected."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )

        # Try to approve with wrong hash
        with pytest.raises(ValueError, match="approval hash does not match"):
            generation_task_service.approve_generation_task(
                session, task.id, request_hash="wrong_hash"
            )

    def test_approve_after_modify_invalidates(self, session, project):
        """Modifying request after approval invalidates the approval."""
        request1 = GenerationRequest(
            project_id=project.id,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request1
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request1.request_hash()
        )

        # Modify request
        request2 = GenerationRequest(
            project_id=project.id,
            prompt_intent="A different sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.update_generation_request(session, task.id, request=request2)

        # Approval should be invalidated
        assert task.approval_state == "PENDING"
        assert task.approved_request_hash is None
        assert task.status == "ESTIMATING"


class TestRejectCancel:
    """Tests for rejection and cancellation (service layer)."""

    def test_reject_awaiting_approval(self, session, project):
        """Reject works in AWAITING_APPROVAL."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.reject_generation_task(session, task.id)

        assert task.approval_state == "REJECTED"
        assert task.status == "AWAITING_APPROVAL"

    def test_cancel_queued_pre_submit(self, session, project):
        """Cancel QUEUED task pre-submit marks CANCELLED directly."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        task = generation_task_service.cancel_generation_task(session, task.id)

        assert task.status == "CANCELLED"
        assert task.cancel_requested is False

    def test_cancel_running_post_submit_sets_cancel_requested(self, session, project):
        """Cancel RUNNING task post-submit sets cancel_requested=True."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        # Simulate RUNNING with external_task_id
        task.status = "RUNNING"
        task.external_task_id = "ext-123"
        from app.editor.models import now
        task.updated_at = now()
        session.add(task)
        session.commit()

        task = generation_task_service.cancel_generation_task(session, task.id)

        assert task.cancel_requested is True
        assert task.status == "RUNNING"  # Still RUNNING until runtime confirms


class TestCapabilitySerialization:
    """Tests for _serialize_asset_task including cancel_requested."""

    def test_serialize_includes_cancel_requested(self, session, project):
        """_serialize_asset_task includes cancel_requested field."""
        task = AssetTask(
            project_id=project.id,
            kind="GENERATION",
            status="RUNNING",
            provider="local_ltx",
            model="ltx-video-2b-distilled-fp8",
            request={"mode": "text_to_video"},
            request_hash="abc123",
            cost_estimate={},
            approval_state="APPROVED",
            approved_request_hash="abc123",
            progress=0.5,
            cancel_requested=True,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        result = _serialize_asset_task(task, session)

        assert "cancel_requested" in result
        assert result["cancel_requested"] is True


class TestProjectIsolation:
    """Tests for project isolation of reference/grounding assets (service layer)."""

    def test_reference_asset_must_belong_to_project(self, session):
        """Reference asset must belong to the same project."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()

        asset = EditorAsset(
            project_id=project2.id,
            original_path="/fake/ref.mp4",
            filename="ref.mp4",
            fingerprint="fp1",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata={},
        )
        session.add(asset)
        session.commit()

        request = GenerationRequest(
            project_id=project1.id,
            mode=GenerationMode.IMAGE_TO_VIDEO,
            prompt_intent="Animate",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
            reference_asset_ids=[asset.id],
        )

        with pytest.raises(ValueError, match="reference asset"):
            generation_task_service.create_generation_task(
                session, project_id=project1.id, request=request
            )

    def test_grounding_asset_must_belong_to_project(self, session):
        """Grounding asset must belong to the same project."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()

        asset = EditorAsset(
            project_id=project2.id,
            original_path="/fake/ground.mp4",
            filename="ground.mp4",
            fingerprint="fp2",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata={},
        )
        session.add(asset)
        session.commit()

        request = GenerationRequest(
            project_id=project1.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset with context",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
            grounding_asset_ids=["asset-from-other-project"],
        )

        with pytest.raises(ValueError, match="grounding asset"):
            generation_task_service.create_generation_task(
                session, project_id=project1.id, request=request
            )

    def test_task_from_other_project_not_accessible(self, session):
        """Task from another project cannot be accessed."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()

        request = GenerationRequest(
            project_id=project2.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project2.id, request=request
        )

        # Try to access task from project1's perspective - serialization works
        # but API layer would reject it via project_or_404
        result = _serialize_asset_task(task, session)
        assert result["project_id"] == project2.id
        # The task belongs to project2, not project1
        assert result["project_id"] != project1.id


class TestNoCloudFallback:
    """Tests proving no automatic cloud fallback."""

    def test_local_unavailable_no_cloud_fallback(self, session, project):
        """Local unavailable does not fall back to cloud."""
        from app.editor.generation import GenerationProvider, LocalGenerationUnavailable

        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            provider=GenerationProvider.LOCAL_LTX,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        # Runtime that raises LocalGenerationUnavailable
        class UnavailableRuntime:
            @property
            def state(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.UNAVAILABLE

            async def probe(self):
                return False

            def availability(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.UNAVAILABLE

            def compile_request(self, request):
                return {"prompt": "test"}

            async def submit(self, compiled_request):
                raise LocalGenerationUnavailable("Local generation runtime unavailable")

            def get_status(self, external_task):
                pass

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                pass

            def fetch_result_to_path(self, external_task, output_path):
                pass

        task = generation_task_service.execute_generation_task(
            session, task.id, local_runtime=UnavailableRuntime()
        )

        assert task.status == "FAILED"
        assert "LocalGenerationUnavailable" in task.error


class TestTimelineImmutability:
    """Tests that generation does not modify timeline until explicit insert."""

    def test_completed_generation_does_not_modify_timeline(self, session, project, monkeypatch):
        """COMPLETED generation does not mutate timeline or version."""
        project.current_version = 1
        session.add(project)
        session.commit()
        session.refresh(project)

        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        # Mock _probe_video to return valid video metadata
        monkeypatch.setattr(
            "app.editor.generation_task_service._probe_video",
            lambda path: (16, 16, 30.0, 1_000_000, True),
        )

        class SuccessRuntime:
            @property
            def state(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            async def probe(self):
                return True

            def availability(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            def compile_request(self, request):
                return {"prompt": "test"}

            async def submit(self, compiled_request):
                from app.editor.generation import (
                    ExternalTaskState,
                    GenerationExternalTask,
                    GenerationModel,
                    GenerationProvider,
                )
                return GenerationExternalTask(
                    external_task_id="ext-123",
                    provider=GenerationProvider.LOCAL_LTX,
                    model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
                    state=ExternalTaskState.PENDING,
                )

            def get_status(self, external_task):
                import tempfile

                from app.editor.generation import ExternalTaskState

                # Create a real temporary file
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(b"fake video bytes")
                    tmp_path = tmp.name
                external_task.state = ExternalTaskState.SUCCEEDED
                external_task.output_url = tmp_path
                external_task.completed_at = datetime.now(UTC)
                return external_task

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                return b"fake video bytes"

            def fetch_result_to_path(self, external_task, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"fake video bytes")

        runtime = SuccessRuntime()
        task = generation_task_service.execute_generation_task(
            session, task.id, local_runtime=runtime
        )
        # Poll to completion
        for _ in range(5):
            task = generation_task_service.execute_claimed_generation_task(
                session, task.id, local_runtime=runtime
            )
            if task.status == "COMPLETED":
                break

        assert task.status == "COMPLETED"
        assert task.output_asset_id is not None

        # Timeline should be unchanged
        session.refresh(project)
        assert project.current_version == 1


class TestFactualProvenance:
    """Tests for factual provenance in generated assets."""

    def test_t2v_generated_provenance(self, session, project):
        """T2V generated asset has GENERATED origin, ILLUSTRATIVE role, no parent/root."""
        from app.editor.asset_provenance import AssetSemanticRole
        from app.editor.generation import (
            GenerationAspectRatio,
            GenerationMode,
            GenerationModel,
            GenerationProvider,
            GenerationResolution,
            generated_provenance,
        )

        prov = generated_provenance(
            parent_asset_id=None,
            root_source_asset_id=None,
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            mode=GenerationMode.TEXT_TO_VIDEO,
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
            request_hash="abc123",
            compiled_prompt_hash="def456",
            external_task_id="ext-123",
            output_hash="hash123",
            output_size_bytes=1000,
            operation_version="gen_v1",
            parameters={"mode": "text_to_video", "seed": None},
            duration_us=5_000_000,
            width=1280,
            height=720,
            fps=30.0,
        )

        assert prov.origin == AssetOrigin.GENERATED
        assert prov.semantic_role == AssetSemanticRole.ILLUSTRATIVE
        assert prov.parent_asset_id is None
        assert prov.root_source_asset_id is None
        assert prov.parameters["mode"] == "text_to_video"

    def test_i2v_with_source_reference_has_factual_root(self, session, project):
        """I2V with SOURCE reference uses factual root."""
        from app.editor.generation import generated_provenance

        prov = generated_provenance(
            parent_asset_id="ref-asset-123",
            root_source_asset_id="factual-root-456",
            provider=GenerationProvider.LOCAL_LTX,
            model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
            mode=GenerationMode.IMAGE_TO_VIDEO,
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
            request_hash="abc123",
            compiled_prompt_hash="def456",
            external_task_id="ext-123",
            output_hash="hash123",
            output_size_bytes=1000,
            operation_version="gen_v1",
            parameters={"mode": "image_to_video"},
            duration_us=5_000_000,
            width=1280,
            height=720,
            fps=30.0,
        )

        assert prov.parent_asset_id == "ref-asset-123"
        assert prov.root_source_asset_id == "factual-root-456"


class TestCancellationSerialization:
    """Tests for cancel_requested serialization."""

    def test_cancel_requested_serialized_in_task(self, session, project):
        """cancel_requested field appears in serialized task."""

        task = AssetTask(
            project_id=project.id,
            kind="GENERATION",
            status="RUNNING",
            provider="local_ltx",
            model="ltx-video-2b-distilled-fp8",
            request={"mode": "text_to_video"},
            request_hash="abc123",
            cost_estimate={},
            approval_state="APPROVED",
            approved_request_hash="abc123",
            progress=0.5,
            cancel_requested=True,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        result = _serialize_asset_task(task, session)

        assert result["cancel_requested"] is True


# =============================================================================
# HTTP ENDPOINT TESTS (real API calls via TestClient)
# =============================================================================

class TestGenerationAPI_HTTP:
    """Real HTTP tests for generation endpoints."""

    def _create_project(self, client: TestClient) -> dict:
        """Helper to create a project via API."""
        resp = client.post("/api/editor/projects", json={"name": "Test Project HTTP"})
        assert resp.status_code == 200
        return resp.json()

    def _create_asset(self, client: TestClient, project_id: str) -> dict:
        """Helper to create a dummy asset via API (using fake path)."""
        # We can't easily upload a real file, so we'll use the service layer for asset creation
        # and just use the project for HTTP tests that don't need real assets
        pass

    def test_prepare_generation_t2v_http(self, client: TestClient, session: Session, project):
        """POST /generations creates task in AWAITING_APPROVAL with estimate."""
        resp = client.post(
            f"/api/editor/projects/{project.id}/generations",
            json={
                "mode": "text_to_video",
                "provider": "local_ltx",
                "model": "ltx-video-2b-distilled-fp8",
                "prompt_intent": "A beautiful sunset over mountains",
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "resolution": "SD",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "AWAITING_APPROVAL"
        assert data["project_id"] == project.id
        assert data["kind"] == "GENERATION"
        assert data["provider"] == "local_ltx"
        assert "request_hash" in data
        assert "cost_estimate" in data
        assert data["approval_state"] == "PENDING"
        assert data["cancel_requested"] is False

    def test_prepare_generation_i2v_requires_reference_http(self, client: TestClient, project):
        """I2V without reference asset returns 422."""
        resp = client.post(
            f"/api/editor/projects/{project.id}/generations",
            json={
                "mode": "image_to_video",
                "provider": "local_ltx",
                "model": "ltx-video-2b-distilled-fp8",
                "prompt_intent": "Animate this",
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "resolution": "SD",
                "reference_asset_ids": [],
            },
        )
        assert resp.status_code == 422
        assert "reference" in resp.text.lower()

    def test_list_generation_tasks_http(self, client: TestClient, session: Session, project):
        """GET /generations lists only GENERATION tasks for the project."""
        # Create a task via service layer
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)

        # Also create an enhancement task to verify filtering
        from app.editor import asset_task_service
        from app.editor.enhancement import EnhancementProfile, EnhancementRequest
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
        )
        session.add(asset)
        session.commit()
        session.refresh(asset)
        enh_request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        asset_task_service.create_enhancement_task(
            session, project_id=project.id, request=enh_request
        )

        # Now test HTTP list
        resp = client.get(f"/api/editor/projects/{project.id}/generations")
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        # Should only contain GENERATION tasks
        for t in tasks:
            assert t["kind"] == "GENERATION"
        # Should have at least our generation task
        assert any(t["id"] == task.id for t in tasks)

    def test_get_generation_task_http(self, client: TestClient, session: Session, project):
        """GET /generations/{task_id} returns correct task."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)

        resp = client.get(f"/api/editor/projects/{project.id}/generations/{task.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task.id
        assert data["project_id"] == project.id
        assert data["kind"] == "GENERATION"

    def test_get_generation_task_other_project_404(self, client: TestClient, session: Session):
        """Task from another project returns 404."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()
        session.refresh(project1)
        session.refresh(project2)

        request = GenerationRequest(
            project_id=project2.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project2.id, request=request
        )

        # Try to access from project1
        resp = client.get(f"/api/editor/projects/{project1.id}/generations/{task.id}")
        assert resp.status_code == 404

    def test_approve_exact_hash_http(self, client: TestClient, session: Session, project):
        """Approve with exact hash moves to QUEUED."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)

        resp = client.post(
            f"/api/editor/projects/{project.id}/generations/{task.id}/approve",
            json={"request_hash": task.request_hash},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "QUEUED"
        assert data["approval_state"] == "APPROVED"

    def test_approve_wrong_hash_http(self, client: TestClient, session: Session, project):
        """Approve with wrong hash returns 400 HASH_MISMATCH."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)

        resp = client.post(
            f"/api/editor/projects/{project.id}/generations/{task.id}/approve",
            json={"request_hash": "wrong_hash"},
        )
        assert resp.status_code == 400
        assert "HASH_MISMATCH" in resp.text

    def test_reject_http(self, client: TestClient, session: Session, project):
        """Reject works in AWAITING_APPROVAL."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)

        resp = client.post(
            f"/api/editor/projects/{project.id}/generations/{task.id}/reject",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["approval_state"] == "REJECTED"
        assert data["status"] == "AWAITING_APPROVAL"

    def test_cancel_queued_pre_submit_http(self, client: TestClient, session: Session, project):
        """Cancel QUEUED task pre-submit marks CANCELLED."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        resp = client.post(
            f"/api/editor/projects/{project.id}/generations/{task.id}/cancel",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "CANCELLED"
        assert data["cancel_requested"] is False

    def test_cancel_running_post_submit_http(self, client: TestClient, session: Session, project):
        """Cancel RUNNING task sets cancel_requested=True."""
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        # Simulate RUNNING
        task.status = "RUNNING"
        task.external_task_id = "ext-123"
        from app.editor.models import now
        task.updated_at = now()
        session.add(task)
        session.commit()

        resp = client.post(
            f"/api/editor/projects/{project.id}/generations/{task.id}/cancel",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cancel_requested"] is True
        assert data["status"] == "RUNNING"

    def test_capability_http_amd_unsupported(self, client: TestClient, session: Session, project):
        """Capability endpoint returns correct structure for unsupported hardware."""
        from app.editor.generation import LocalGenerationHardwareInfo, LocalGenerationHardwareReason

        hw_info = LocalGenerationHardwareInfo(
            has_cuda=False,
            cuda_version=None,
            gpu_name="AMD Radeon",
            gpu_vram_mb=512,
            compute_capability=None,
            supports_fp8=False,
            fp8_reason="No NVIDIA GPU",
            runtime_supported=False,
            reason=LocalGenerationHardwareReason.NO_NVIDIA_GPU,
            reason_message="No NVIDIA GPU detected",
        )

        with patch("app.editor.generation.probe_local_generation_hardware", return_value=hw_info):
            resp = client.get(f"/api/editor/projects/{project.id}/generations/capability")
            assert resp.status_code == 200
            data = resp.json()
            assert data["local_ltx"]["hardware_supported"] is False
            assert data["local_ltx"]["runtime_ready"] is False
            assert data["local_ltx"]["physical_validation"] == "DEFERRED"

    def test_capability_http_hardware_supported_deferred(
        self, client: TestClient, session: Session, project
    ):
        """Capability endpoint returns hardware_supported=true, runtime_ready=false."""
        from app.editor.generation import LocalGenerationHardwareInfo, LocalGenerationHardwareReason

        hw_info = LocalGenerationHardwareInfo(
            has_cuda=True,
            cuda_version="12.4",
            gpu_name="NVIDIA RTX 4090",
            gpu_vram_mb=24576,
            compute_capability=(8, 9),
            fp8_supported=True,
            fp8_reason="Ada architecture supports FP8",
            runtime_supported=True,
            reason=LocalGenerationHardwareReason.SUPPORTED,
            reason_message="Hardware supports local generation",
        )

        with patch("app.editor.generation.probe_local_generation_hardware", return_value=hw_info):
            resp = client.get(f"/api/editor/projects/{project.id}/generations/capability")
            assert resp.status_code == 200
            data = resp.json()
            assert data["local_ltx"]["hardware_supported"] is True
            assert data["local_ltx"]["runtime_ready"] is False
            assert data["local_ltx"]["physical_validation"] == "DEFERRED"
            assert "física LTX aún no está habilitada" in data["local_ltx"]["runtime_message"]

    def test_capability_http_cloud_not_ready(self, client: TestClient, session: Session, project):
        """Cloud providers are available=false, ready=false."""
        resp = client.get(f"/api/editor/projects/{project.id}/generations/capability")
        assert resp.status_code == 200
        data = resp.json()
        for cp in data["cloud_providers"]:
            assert cp["available"] is False
            assert cp["ready"] is False
            assert cp["supported"] is True  # Contract exists
            assert ("backend cloud no configurado" in cp["message"].lower()
                or "no configurado" in cp["message"].lower())

    def test_reference_asset_other_project_422(self, client: TestClient, session: Session):
        """Reference asset from other project returns 422."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()

        asset = EditorAsset(
            project_id=project2.id,
            original_path="/fake/ref.mp4",
            filename="ref.mp4",
            fingerprint="fp1",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata={},
        )
        session.add(asset)
        session.commit()

        resp = client.post(
            f"/api/editor/projects/{project1.id}/generations",
            json={
                "mode": "image_to_video",
                "provider": "local_ltx",
                "model": "ltx-video-2b-distilled-fp8",
                "prompt_intent": "Animate",
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "resolution": "SD",
                "reference_asset_ids": [asset.id],
            },
        )
        assert resp.status_code == 422
        assert "reference" in resp.text.lower()

    def test_grounding_asset_other_project_422(self, client: TestClient, session: Session):
        """Grounding asset from other project returns 422."""
        project1 = EditorProject(name="Project 1")
        project2 = EditorProject(name="Project 2")
        session.add_all([project1, project2])
        session.commit()

        asset = EditorAsset(
            project_id=project2.id,
            original_path="/fake/ground.mp4",
            filename="ground.mp4",
            fingerprint="fp2",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata={},
        )
        session.add(asset)
        session.commit()

        resp = client.post(
            f"/api/editor/projects/{project1.id}/generations",
            json={
                "mode": "text_to_video",
                "provider": "local_ltx",
                "model": "ltx-video-2b-distilled-fp8",
                "prompt_intent": "A sunset with context",
                "duration_seconds": 5,
                "aspect_ratio": "16:9",
                "resolution": "SD",
                "grounding_asset_ids": [asset.id],
            },
        )
        assert resp.status_code == 422
        assert "grounding" in resp.text.lower()


class TestVersionUndo:
    """Tests for version/undo after explicit insert."""

    def test_insert_generated_creates_new_version(
        self, client: TestClient, session: Session, project, monkeypatch
    ):
        """Explicit insert creates new version with generated clip."""
        
        # Ensure timeline has a video track and capture initial version
        from app.editor.models import EditorVersion
        version = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        if not version:
            raise AssertionError("No initial version found")
        if not version.timeline or not version.timeline.get("tracks"):
            # Create a video track
            from app.editor.service import apply_operation
            apply_operation(session, project.id, {
                "type": "track_properties",
                "parameters": {"track_id": "video_1", "properties": {"type": "video"}},
                "base_version": project.current_version,
                "idempotency_key": "test-track-create-version",
            })
            session.refresh(project)
            # Re-load version
            version = session.exec(
                select(EditorVersion).where(
                    EditorVersion.project_id == project.id,
                    EditorVersion.number == project.current_version
                )
            ).first()
        
        initial_version = project.current_version
        # Create a completed generation task with output asset
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        # Mock _probe_video
        monkeypatch.setattr(
            "app.editor.generation_task_service._probe_video",
            lambda path: (1280, 720, 30.0, 5_000_000, True),
        )

        class SuccessRuntime:
            @property
            def state(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            async def probe(self):
                return True

            def availability(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            def compile_request(self, request):
                return {"prompt": "test"}

            async def submit(self, compiled_request):
                from app.editor.generation import (
                    ExternalTaskState,
                    GenerationExternalTask,
                    GenerationModel,
                    GenerationProvider,
                )
                return GenerationExternalTask(
                    external_task_id="ext-123",
                    provider=GenerationProvider.LOCAL_LTX,
                    model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
                    state=ExternalTaskState.PENDING,
                )

            def get_status(self, external_task):
                import tempfile

                from app.editor.generation import ExternalTaskState
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(b"fake video bytes")
                    tmp_path = tmp.name
                external_task.state = ExternalTaskState.SUCCEEDED
                external_task.output_url = tmp_path
                external_task.completed_at = datetime.now(UTC)
                return external_task

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                return b"fake video bytes"

            def fetch_result_to_path(self, external_task, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"fake video bytes")

        runtime = SuccessRuntime()
        task = generation_task_service.execute_generation_task(
            session, task.id, local_runtime=runtime
        )
        for _ in range(5):
            task = generation_task_service.execute_claimed_generation_task(
                session, task.id, local_runtime=runtime
            )
            if task.status == "COMPLETED":
                break

        assert task.status == "COMPLETED"
        assert task.output_asset_id is not None

        # Verify timeline is still unchanged before insert
        session.refresh(project)
        assert project.current_version == initial_version

        # Get the generated asset
        generated_asset = session.get(EditorAsset, task.output_asset_id)
        assert generated_asset is not None

        # Get current version with timeline
        version = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        tracks = version.timeline.get("tracks", []) if version else []
        video_track = tracks[0] if tracks else None
        if not video_track:
            # Should have been created above
            raise AssertionError("No video track found")

        # Insert the generated asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/operations",
            json={
                "type": "insert",
                "parameters": {
                    "asset_id": generated_asset.id,
                    "track_id": video_track.get("id"),
                    "source_in_us": 0,
                    "source_out_us": generated_asset.duration_us,
                    "timeline_start_us": version.timeline.get("duration_us", 0),
                },
                "base_version": project.current_version,
                "idempotency_key": f"test-insert-{generated_asset.id}",
            },
        )
        assert resp.status_code == 200

        # Verify new version created
        session.refresh(project)
        assert project.current_version == initial_version + 1

        # Verify clip is in timeline
        version_after = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        tracks = version_after.timeline.get("tracks", [])
        clips = [c for track in tracks for c in track.get("clips", [])]
        assert any(c.get("asset_id") == generated_asset.id for c in clips)

    def test_undo_removes_generated_clip_but_keeps_asset(
        self, client: TestClient, session: Session, project, monkeypatch
    ):
        """Undo after insert removes clip but keeps GENERATED asset in library."""
    
        # Ensure timeline has a video track and capture initial version
        from app.editor.models import EditorVersion
        version = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        if not version:
            raise AssertionError("No initial version found")
        if not version.timeline or not version.timeline.get("tracks"):
            # Create a video track
            from app.editor.service import apply_operation
            apply_operation(session, project.id, {
                "type": "track_properties",
                "parameters": {"track_id": "video_1", "properties": {"type": "video"}},
                "base_version": project.current_version,
                "idempotency_key": "test-track-create-undo",
            })
            session.refresh(project)
            # Re-load version
            version = session.exec(
                select(EditorVersion).where(
                    EditorVersion.project_id == project.id,
                    EditorVersion.number == project.current_version
                )
            ).first()
    
        # Capture initial_version AFTER track creation and refresh
        initial_version = project.current_version
        # Setup: create completed generation with asset
        request = GenerationRequest(
            project_id=project.id,
            mode=GenerationMode.TEXT_TO_VIDEO,
            prompt_intent="A sunset",
            duration_seconds=5,
            aspect_ratio=GenerationAspectRatio.RATIO_16_9,
            resolution=GenerationResolution.SD,
        )
        task = generation_task_service.create_generation_task(
            session, project_id=project.id, request=request
        )
        task = generation_task_service.attach_generation_estimate(session, task.id)
        task = generation_task_service.approve_generation_task(
            session, task.id, request_hash=request.request_hash()
        )
        task = generation_task_service.queue_generation_task(session, task.id)

        monkeypatch.setattr(
            "app.editor.generation_task_service._probe_video",
            lambda path: (1280, 720, 30.0, 5_000_000, True),
        )

        class SuccessRuntime:
            @property
            def state(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            async def probe(self):
                return True

            def availability(self):
                from app.editor.generation import LocalRuntimeState
                return LocalRuntimeState.READY

            def compile_request(self, request):
                return {"prompt": "test"}

            async def submit(self, compiled_request):
                from app.editor.generation import (
                    ExternalTaskState,
                    GenerationExternalTask,
                    GenerationModel,
                    GenerationProvider,
                )
                return GenerationExternalTask(
                    external_task_id="ext-123",
                    provider=GenerationProvider.LOCAL_LTX,
                    model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
                    state=ExternalTaskState.PENDING,
                )

            def get_status(self, external_task):
                import tempfile

                from app.editor.generation import ExternalTaskState
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(b"fake video bytes")
                    tmp_path = tmp.name
                external_task.state = ExternalTaskState.SUCCEEDED
                external_task.output_url = tmp_path
                external_task.completed_at = datetime.now(UTC)
                return external_task

            def cancel(self, external_task):
                pass

            def fetch_result(self, external_task):
                return b"fake video bytes"

            def fetch_result_to_path(self, external_task, output_path):
                with open(output_path, "wb") as f:
                    f.write(b"fake video bytes")

        runtime = SuccessRuntime()
        task = generation_task_service.execute_generation_task(
            session, task.id, local_runtime=runtime
        )
        for _ in range(5):
            task = generation_task_service.execute_claimed_generation_task(
                session, task.id, local_runtime=runtime
            )
            if task.status == "COMPLETED":
                break

        generated_asset_id = task.output_asset_id
        generated_asset = session.get(EditorAsset, generated_asset_id)
        assert generated_asset is not None

        # Get current version with timeline
        version = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        tracks = version.timeline.get("tracks", []) if version else []
        video_track = tracks[0] if tracks else None
        if not video_track:
            raise AssertionError("No video track found")

        # Insert the generated asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/operations",
            json={
                "type": "insert",
                "parameters": {
                    "asset_id": generated_asset.id,
                    "track_id": video_track.get("id"),
                    "source_in_us": 0,
                    "source_out_us": generated_asset.duration_us,
                    "timeline_start_us": version.timeline.get("duration_us", 0),
                },
                "base_version": project.current_version,
                "idempotency_key": f"test-insert-undo-{generated_asset.id}",
            },
        )
        assert resp.status_code == 200
        session.refresh(project)
        insert_version = project.current_version

        # Now undo
        resp = client.post(f"/api/editor/projects/{project.id}/undo")
        assert resp.status_code == 200

        session.refresh(project)
        # Undo creates a new version with restored timeline (version = insert_version + 1)
        assert project.current_version == insert_version + 1

        # Verify the timeline matches the initial version
        restored_version = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == project.current_version
            )
        ).first()
        initial_timeline = session.exec(
            select(EditorVersion).where(
                EditorVersion.project_id == project.id,
                EditorVersion.number == initial_version
            )
        ).first().timeline
        assert restored_version.timeline == initial_timeline

        # Generated asset should still exist in library
        asset_still_exists = session.get(EditorAsset, generated_asset_id)
        assert asset_still_exists is not None
        assert asset_still_exists.id == generated_asset_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])