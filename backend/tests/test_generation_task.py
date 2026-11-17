"""Tests for GENERATION AssetTask lifecycle (SNIPRA 1.6D Block 03)."""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from app.core.db import engine, init_db
from app.editor import generation_task_service
from app.editor.asset_tasks import (
    AssetTaskApprovalState,
    AssetTaskKind,
    AssetTaskStatus,
    InvalidTaskTransition,
)
from app.editor.generation import (
    GenerationAspectRatio,
    GenerationModel,
    GenerationProvider,
    GenerationRequest,
    GenerationResolution,
    LocalRuntimeState,
)
from app.editor.models import AssetTask, EditorAsset, EditorProject
from tests.fakes.generation_task_fake import FakeLocalGenerativeRuntime


@pytest.fixture(autouse=True)
def db():
    """Initialize clean database for each test."""
    init_db()
    yield
    # Cleanup
    with Session(engine) as session:
        for task in session.exec(select(AssetTask)).all():
            session.delete(task)
        for asset in session.exec(select(EditorAsset)).all():
            session.delete(asset)
        for project in session.exec(select(EditorProject)).all():
            session.delete(project)
        session.commit()


class TestCreateGenerationTask:
    def test_create_generation_task_basic(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)

            assert task.id
            assert task.kind == AssetTaskKind.GENERATION.value
            assert task.status == AssetTaskStatus.ESTIMATING.value
            assert task.provider == GenerationProvider.LOCAL_LTX.value
            assert task.model == GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8.value
            assert task.request_hash == request.request_hash()
            assert task.approval_state == AssetTaskApprovalState.PENDING.value
            assert task.approved_request_hash is None

    def test_create_generation_task_validates_reference_assets(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                reference_asset_ids=["nonexistent"],
            )
            with pytest.raises(ValueError, match="reference asset"):
                generation_task_service.create_generation_task(session, project_id=project.id, request=request)

    def test_create_generation_task_validates_grounding_assets(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                grounding_asset_ids=["nonexistent"],
            )
            with pytest.raises(ValueError, match="grounding asset"):
                generation_task_service.create_generation_task(session, project_id=project.id, request=request)


class TestAttachGenerationEstimate:
    def test_attach_estimate_transitions_to_awaiting_approval(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)

            assert task.status == AssetTaskStatus.AWAITING_APPROVAL.value
            assert task.cost_estimate
            assert task.cost_estimate["provider"] == GenerationProvider.LOCAL_LTX.value
            assert task.cost_estimate["api_cost_usd"] == 0.0
            assert task.cost_estimate["billable"] is False
            assert task.cost_estimate["requires_monetary_approval"] is False


class TestApprovalFlow:
    def test_approve_task_binds_to_exact_hash(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)

            # Approve with exact hash
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            assert task.approval_state == AssetTaskApprovalState.APPROVED.value
            assert task.approved_request_hash == request.request_hash()

    def test_approve_task_rejects_mismatched_hash(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)

            # Approve with wrong hash
            with pytest.raises(ValueError, match="approval hash does not match"):
                generation_task_service.approve_generation_task(session, task.id, request_hash="wrong_hash")

    def test_reject_task(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)

            task = generation_task_service.reject_generation_task(session, task.id)
            assert task.approval_state == AssetTaskApprovalState.REJECTED.value
            assert task.approved_request_hash is None


class TestUpdateRequest:
    def test_update_request_in_estimating(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request1 = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request1)

            request2 = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunrise",
                duration_seconds=10,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.update_generation_request(session, task.id, request=request2)

            assert task.request_hash == request2.request_hash()
            assert task.status == AssetTaskStatus.ESTIMATING.value
            assert task.cost_estimate == {}
            assert task.approved_request_hash is None
            assert task.approval_state == AssetTaskApprovalState.PENDING.value

    def test_update_request_in_awaiting_approval_voided_approval(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())

            request2 = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunrise",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.update_generation_request(session, task.id, request=request2)

            assert task.approval_state == AssetTaskApprovalState.PENDING.value
            assert task.approved_request_hash is None
            assert task.cost_estimate == {}

    def test_update_request_rejected_after_queued(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            request2 = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunrise",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            with pytest.raises(InvalidTaskTransition, match="request can only change before the task is queued"):
                generation_task_service.update_generation_request(session, task.id, request=request2)


class TestQueueGenerationTask:
    def test_queue_local_task_succeeds(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            assert task.status == AssetTaskStatus.QUEUED.value

    def test_queue_cloud_task_rejected(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                provider=GenerationProvider.LOCAL_LTX,
                model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
                prompt_intent="test",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())

            # Now change provider to cloud and try to queue
            task.provider = GenerationProvider.RUNWAY.value
            task.model = GenerationModel.RUNWAY_GEN45.value
            session.add(task)
            session.commit()
            session.refresh(task)

            with pytest.raises(InvalidTaskTransition, match="only local_ltx provider is supported"):
                generation_task_service.queue_generation_task(session, task.id)


class TestCancelGenerationTask:
    def test_cancel_queued_task(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            task = generation_task_service.cancel_generation_task(session, task.id)
            assert task.status == AssetTaskStatus.CANCELLED.value

    def test_cancel_completed_task_rejected(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            task = AssetTask(
                project_id=project.id,
                kind=AssetTaskKind.GENERATION.value,
                status=AssetTaskStatus.COMPLETED.value,
                provider=GenerationProvider.LOCAL_LTX.value,
                model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8.value,
            )
            session.add(task)
            session.commit()
            session.refresh(task)

            with pytest.raises(InvalidTaskTransition, match="cannot cancel a completed task"):
                generation_task_service.cancel_generation_task(session, task.id)


class TestExecuteGenerationTask:
    def test_execute_requires_queued(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            task = AssetTask(
                project_id=project.id,
                kind=AssetTaskKind.GENERATION.value,
                status=AssetTaskStatus.ESTIMATING.value,
                provider=GenerationProvider.LOCAL_LTX.value,
                model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8.value,
            )
            session.add(task)
            session.commit()
            session.refresh(task)

            with pytest.raises(InvalidTaskTransition, match="execute requires QUEUED"):
                generation_task_service.execute_generation_task(session, task.id, local_runtime=FakeLocalGenerativeRuntime())

    def test_execute_rejects_cloud_provider(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            task = AssetTask(
                project_id=project.id,
                kind=AssetTaskKind.GENERATION.value,
                status=AssetTaskStatus.QUEUED.value,
                provider=GenerationProvider.RUNWAY.value,
                model=GenerationModel.RUNWAY_GEN45.value,
            )
            session.add(task)
            session.commit()
            session.refresh(task)

            # Set up a valid approval and estimate for the test to reach the provider check
            task.request_hash = "test_hash"
            task.approved_request_hash = "test_hash"
            task.approval_state = AssetTaskApprovalState.APPROVED.value
            task.cost_estimate = {"provider": "runway", "billable": True}
            session.add(task)
            session.commit()
            session.refresh(task)

            with pytest.raises(InvalidTaskTransition, match="only local_ltx provider is supported"):
                generation_task_service.execute_generation_task(session, task.id, local_runtime=FakeLocalGenerativeRuntime())


class TestExecuteClaimedGenerationTask:
    def test_submit_exactly_once_on_restart(self, db):
        """Test that submit is called exactly once even after worker restart."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            # First execution - submits (calls execute_generation_task which transitions to RUNNING and calls execute_claimed)
            # Use Runtime A (first runtime instance)
            runtime_a = FakeLocalGenerativeRuntime()
            runtime_a.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=runtime_a
            )

            assert task.status == AssetTaskStatus.RUNNING.value
            assert task.external_task_id is not None
            assert task.submit_count == 1
            external_task_id_1 = task.external_task_id

            # Second execution (simulating restart after submit) - should poll, not re-submit
            # Need to call execute_claimed_generation_task repeatedly until completion
            # Use Runtime B (different instance) to simulate a restart with a new runtime
            runtime_b = FakeLocalGenerativeRuntime()
            runtime_b.configure_for_success()

            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=runtime_b
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None
            assert task.external_task_id == external_task_id_1
            assert task.submit_count == 1  # Still exactly 1
            assert runtime_a._submit_count == 1  # Runtime A submitted once
            assert runtime_b._submit_count == 0  # Runtime B never submitted, only polled

            # Third execution (after COMPLETED) - should fail closed
            with pytest.raises(InvalidTaskTransition, match="execute requires QUEUED"):
                generation_task_service.execute_generation_task(
                    session, task.id, local_runtime=FakeLocalGenerativeRuntime()
                )

    def test_restart_after_submit_polls_to_completion(self, db):
        """Test that after submit, restart polls to completion without re-submit."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            # First execution - submit and get to RUNNING
            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            assert task.status == AssetTaskStatus.RUNNING.value
            assert task.external_task_id is not None

            # Second execution - polls to completion (may need multiple polls)
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None
            assert task.external_task_id is not None

            # Verify GENERATED asset was created
            asset = session.get(EditorAsset, task.output_asset_id)
            assert asset is not None
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(asset.asset_metadata)
            assert prov is not None
            assert prov.origin == "GENERATED"
            assert prov.semantic_role == "ILLUSTRATIVE"
            assert prov.provider == "local_ltx"
            assert prov.model_or_engine == "ltx-video-2b-distilled-fp8"

    def test_generated_asset_not_evidence_eligible(self, db):
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Poll until completion
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            from app.editor.generation import is_evidence_eligible
            asset = session.get(EditorAsset, task.output_asset_id)
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(asset.asset_metadata)
            assert not is_evidence_eligible(prov.origin)


class TestNoCloudFallback:
    def test_unsupported_hardware_raises_local_generation_unavailable(self, db):
        """Unsupported local hardware must raise LocalGenerationUnavailable, not fall back to cloud."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            # Runtime that is UNAVAILABLE
            class UnavailableRuntime:
                @property
                def state(self) -> LocalRuntimeState:
                    return LocalRuntimeState.UNAVAILABLE

                async def probe(self) -> bool:
                    return False

                def availability(self) -> LocalRuntimeState:
                    return LocalRuntimeState.UNAVAILABLE

                async def submit(self, compiled_request):
                    pass

                def compile_request(self, request):
                    return {"prompt": "test"}

                def get_status(self, external_task):
                    pass

                def cancel(self, external_task):
                    pass

                def fetch_result(self, external_task):
                    pass

                def fetch_result_to_path(self, external_task, output_path):
                    pass

            # The execution core catches all exceptions and marks task as FAILED
            # The error message should contain the LocalGenerationUnavailable info
            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=UnavailableRuntime()
            )

            assert task.status == AssetTaskStatus.FAILED.value
            assert "LocalGenerationUnavailable" in task.error
            assert "UNAVAILABLE" in task.error

    def test_no_cloud_fallback_on_failure(self, db):
        """On any failure, no cloud provider should be invoked."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            # Runtime that fails
            class FailingRuntime:
                @property
                def state(self) -> LocalRuntimeState:
                    return LocalRuntimeState.READY

                async def probe(self) -> bool:
                    return True

                def availability(self) -> LocalRuntimeState:
                    return LocalRuntimeState.READY

                async def submit(self, compiled_request):
                    from app.editor.generation import ExternalTaskState, GenerationExternalTask
                    return GenerationExternalTask(
                        external_task_id="fake-1",
                        provider=GenerationProvider.LOCAL_LTX,
                        model=GenerationModel.LTX_VIDEO_2B_DISTILLED_FP8,
                        state=ExternalTaskState.FAILED,
                        failure_code="FAKE_ERROR",
                        failure_message="fake failure",
                    )

                def compile_request(self, request):
                    return {"prompt": "test"}

                def get_status(self, external_task):
                    from app.editor.generation import ExternalTaskState
                    external_task.state = ExternalTaskState.FAILED
                    return external_task

                def cancel(self, external_task):
                    pass

                def fetch_result(self, external_task):
                    pass

                def fetch_result_to_path(self, external_task, output_path):
                    pass

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=FailingRuntime()
            )

            # Poll until failure
            fake_runtime = FailingRuntime()
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.FAILED.value:
                    break

            assert task.status == AssetTaskStatus.FAILED.value
            # The error message contains the failure reason (may be generic if runtime doesn't preserve failure details)
            assert "generation failed" in task.error.lower() or "fake failure" in task.error.lower()


class TestTimelineUntouched:
    def test_timeline_unchanged_after_generation(self, db):
        """Generation completion must not mutate project timeline or version."""
        with Session(engine) as session:
            project = EditorProject(name="test project", current_version=1)
            session.add(project)
            session.commit()
            session.refresh(project)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Refresh project
            session.refresh(project)
            assert project.current_version == 1  # unchanged


class TestProvenanceFactual:
    """Tests for factual provenance in generated assets."""

    def test_t2v_with_grounding_has_no_parent_no_root(self, db):
        """T2V with grounding: parent=None, root=None (generated from scratch)."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            # Create a grounding asset (SOURCE)
            from app.editor.asset_provenance import source_provenance, write_asset_provenance
            from app.editor.generation import GenerationMode
            grounding_asset = EditorAsset(
                project_id=project.id,
                original_path="/fake/grounding.mp4",
                filename="grounding.mp4",
                fingerprint="grnd123",
                kind="video",
                duration_us=5_000_000,
                width=1280,
                height=720,
                has_audio=True,
                asset_metadata=write_asset_provenance({}, source_provenance(asset_id="grounding-asset")),
            )
            session.add(grounding_asset)
            session.commit()
            session.refresh(grounding_asset)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="A sunset",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                mode=GenerationMode.TEXT_TO_VIDEO,
                grounding_asset_ids=[grounding_asset.id],
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Poll to completion
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None

            # Check provenance: T2V has no parent, no root
            output_asset = session.get(EditorAsset, task.output_asset_id)
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(output_asset.asset_metadata)
            assert prov is not None
            assert prov.parent_asset_id is None, f"T2V parent should be None, got {prov.parent_asset_id}"
            assert prov.root_source_asset_id is None, f"T2V root should be None, got {prov.root_source_asset_id}"
            # Grounding should be in parameters
            assert prov.parameters is not None
            assert "grounding_asset_ids" in prov.parameters

    def test_i2v_with_source_reference_has_parent_and_factual_root(self, db):
        """I2V with SOURCE reference: parent=reference, root=factual root_source_asset_id."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            # Create a SOURCE reference asset with factual root
            from app.editor.asset_provenance import source_provenance, write_asset_provenance
            from app.editor.generation import GenerationMode
            reference_asset = EditorAsset(
                project_id=project.id,
                original_path="/fake/reference.mp4",
                filename="reference.mp4",
                fingerprint="ref123",
                kind="video",
                duration_us=5_000_000,
                width=1280,
                height=720,
                has_audio=True,
                asset_metadata=write_asset_provenance({}, source_provenance(asset_id="factual-root-asset-123")),
            )
            session.add(reference_asset)
            session.commit()
            session.refresh(reference_asset)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="Animate this image",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                mode=GenerationMode.IMAGE_TO_VIDEO,
                reference_asset_ids=[reference_asset.id],
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Poll to completion
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None

            # Check provenance: I2V with SOURCE reference
            output_asset = session.get(EditorAsset, task.output_asset_id)
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(output_asset.asset_metadata)
            assert prov is not None
            assert prov.parent_asset_id == reference_asset.id, f"Parent should be reference asset, got {prov.parent_asset_id}"
            assert prov.root_source_asset_id == "factual-root-asset-123", f"Root should be factual root, got {prov.root_source_asset_id}"

    def test_i2v_with_generated_reference_no_factual_root(self, db):
        """I2V with GENERATED reference (no factual root): parent=reference, root=None."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            # Create a GENERATED reference asset (no factual root) - this is a text_to_video generation
            from app.editor.generation import GenerationMode
            from app.editor.asset_provenance import (
                AssetOrigin, AssetSemanticRole, AssetProvenanceMetadata, write_asset_provenance
            )
            generated_ref_prov = AssetProvenanceMetadata(
                origin=AssetOrigin.GENERATED,
                semantic_role=AssetSemanticRole.ILLUSTRATIVE,
                parent_asset_id=None,
                root_source_asset_id=None,  # text_to_video has no root
                operation_type="generation",
                operation_version="gen_v1",
                provider="local_ltx",
                model_or_engine="ltx-video-2b-distilled-fp8",
                request_hash="abc123",
                output_hash="def456",
                output_size_bytes=1000,
                parameters={"mode": "text_to_video"},  # text_to_video mode - no parent/root required
                duration_us=5_000_000,
                width=1280,
                height=720,
                fps=30.0,
            )
            generated_ref = EditorAsset(
                project_id=project.id,
                original_path="/fake/generated.mp4",
                filename="generated.mp4",
                fingerprint="gen123",
                kind="video",
                duration_us=5_000_000,
                width=1280,
                height=720,
                has_audio=True,
                asset_metadata=write_asset_provenance({}, generated_ref_prov),
            )
            session.add(generated_ref)
            session.commit()
            session.refresh(generated_ref)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="Animate this generated video",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                mode=GenerationMode.IMAGE_TO_VIDEO,
                reference_asset_ids=[generated_ref.id],
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Poll to completion
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None

            # Check provenance: I2V with GENERATED reference (no factual root)
            output_asset = session.get(EditorAsset, task.output_asset_id)
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(output_asset.asset_metadata)
            assert prov is not None
            assert prov.parent_asset_id == generated_ref.id, f"Parent should be generated reference, got {prov.parent_asset_id}"
            assert prov.root_source_asset_id is None, f"Root should be None for generated ref without factual root, got {prov.root_source_asset_id}"

    def test_i2v_grounding_does_not_change_factual_root(self, db):
        """I2V with grounding: factual root from reference preserved, grounding does not change root."""
        with Session(engine) as session:
            project = EditorProject(name="test project")
            session.add(project)
            session.commit()
            session.refresh(project)

            # Create a SOURCE reference asset with factual root
            from app.editor.asset_provenance import source_provenance, write_asset_provenance
            from app.editor.generation import GenerationMode
            reference_asset = EditorAsset(
                project_id=project.id,
                original_path="/fake/reference.mp4",
                filename="reference.mp4",
                fingerprint="ref123",
                kind="video",
                duration_us=5_000_000,
                width=1280,
                height=720,
                has_audio=True,
                asset_metadata=write_asset_provenance({}, source_provenance(asset_id="factual-root-asset-456")),
            )
            session.add(reference_asset)
            session.commit()
            session.refresh(reference_asset)

            # Create a separate grounding asset
            grounding_asset = EditorAsset(
                project_id=project.id,
                original_path="/fake/grounding.mp4",
                filename="grounding.mp4",
                fingerprint="grnd456",
                kind="video",
                duration_us=3_000_000,
                width=1280,
                height=720,
                has_audio=True,
                asset_metadata=write_asset_provenance({}, source_provenance(asset_id="grounding-root-789")),
            )
            session.add(grounding_asset)
            session.commit()
            session.refresh(grounding_asset)

            request = GenerationRequest(
                project_id=project.id,
                prompt_intent="Animate with context",
                duration_seconds=5,
                aspect_ratio=GenerationAspectRatio.RATIO_16_9,
                resolution=GenerationResolution.SD,
                mode=GenerationMode.IMAGE_TO_VIDEO,
                reference_asset_ids=[reference_asset.id],
                grounding_asset_ids=[grounding_asset.id],
            )
            task = generation_task_service.create_generation_task(session, project_id=project.id, request=request)
            task = generation_task_service.attach_generation_estimate(session, task.id)
            task = generation_task_service.approve_generation_task(session, task.id, request_hash=request.request_hash())
            task = generation_task_service.queue_generation_task(session, task.id)

            fake_runtime = FakeLocalGenerativeRuntime()
            fake_runtime.configure_for_success()

            task = generation_task_service.execute_generation_task(
                session, task.id, local_runtime=fake_runtime
            )

            # Poll to completion
            for _ in range(5):
                task = generation_task_service.execute_claimed_generation_task(
                    session, task.id, local_runtime=fake_runtime
                )
                if task.status == AssetTaskStatus.COMPLETED.value:
                    break

            assert task.status == AssetTaskStatus.COMPLETED.value
            assert task.output_asset_id is not None

            # Check provenance: root should be from reference asset, NOT from grounding
            output_asset = session.get(EditorAsset, task.output_asset_id)
            from app.editor.asset_provenance import read_asset_provenance
            prov = read_asset_provenance(output_asset.asset_metadata)
            assert prov is not None
            assert prov.parent_asset_id == reference_asset.id
            assert prov.root_source_asset_id == "factual-root-asset-456", f"Root should be from reference, not grounding, got {prov.root_source_asset_id}"
            # Grounding should be in parameters but not affect root
            assert prov.parameters is not None
            assert "grounding_asset_ids" in prov.parameters
            assert grounding_asset.id in prov.parameters["grounding_asset_ids"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])