"""Tests for enhancement API endpoints (SNIPRA 1.6C Block 04)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select, delete
from unittest.mock import patch

from app.core.config import settings
from app.core.db import engine, init_db
from app.editor import asset_task_service
from app.editor.asset_tasks import AssetTaskStatus
from app.editor.enhancement import EnhancementProfile
from app.editor.models import AssetTask, EditorAsset, EditorProject
from app.main import app
from app.worker import asset_tasks as worker_asset_tasks


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    init_db()


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def project_with_asset(session: Session):
    """Create a project with a video asset."""
    project = EditorProject(name="Test Project")
    session.add(project)
    session.flush()

    # Create a dummy video asset (we'll use a fake path, the tests mock FFmpeg)
    asset = EditorAsset(
        project_id=project.id,
        original_path="/fake/source.mp4",
        filename="source.mp4",
        fingerprint="abc123",
        kind="video",
        duration_us=5_000_000,
        width=1280,
        height=720,
        has_audio=True,
    )
    session.add(asset)
    session.commit()
    session.refresh(project)
    session.refresh(asset)
    return project, asset


class TestEnhancementAPI:
    """Tests for the enhancement API endpoints."""

    def test_prepare_enhancement_returns_awaiting_approval(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == AssetTaskStatus.AWAITING_APPROVAL.value
        assert data["project_id"] == project.id
        assert data["kind"] == "ENHANCEMENT"
        assert data["provider"] == "ffmpeg"
        assert "request_hash" in data
        assert "cost_estimate" in data
        assert data["approval_state"] == "PENDING"

    def test_prepare_enhancement_does_not_execute_ffmpeg(
        self, client: TestClient, project_with_asset, session: Session
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        assert resp.status_code == 200
        task_id = resp.json()["id"]

        # Verify task is in AWAITING_APPROVAL, not RUNNING or COMPLETED
        task = session.get(AssetTask, task_id)
        assert task.status == AssetTaskStatus.AWAITING_APPROVAL.value

    def test_prepare_rejects_asset_from_different_project(
        self, client: TestClient, session: Session
    ):
        # Create two projects with assets
        proj1 = EditorProject(name="Project 1")
        proj2 = EditorProject(name="Project 2")
        session.add(proj1)
        session.add(proj2)
        session.flush()

        asset1 = EditorAsset(
            project_id=proj1.id,
            original_path="/fake/a.mp4",
            filename="a.mp4",
            fingerprint="a",
            kind="video",
            duration_us=1_000_000,
            width=640,
            height=480,
            has_audio=False,
        )
        session.add(asset1)
        session.commit()
        session.refresh(asset1)

        # Try to use asset from proj1 in proj2's enhancement
        resp = client.post(
            f"/api/editor/projects/{proj2.id}/assets/{asset1.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        assert resp.status_code == 404  # asset not found in project

    def test_prepare_rejects_unknown_fields(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={
                "profile": "DEFAULT",
                "source_asset_id": "hack",  # should be forbidden
                "provider": "openai",  # should be forbidden
                "raw_ffmpeg_args": "-vf scale=100:100",  # should be forbidden
            },
        )
        assert resp.status_code == 422

    def test_approve_correct_hash_queues_task(
        self, client: TestClient, project_with_asset, session: Session
    ):
        project, asset = project_with_asset
        # Prepare
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        assert resp.status_code == 200
        task = resp.json()
        task_id = task["id"]
        request_hash = task["request_hash"]

        # Approve with correct hash
        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/approve",
            json={"request_hash": request_hash},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == AssetTaskStatus.QUEUED.value
        assert data["approval_state"] == "APPROVED"

    def test_approve_rejects_wrong_hash(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        task_id = resp.json()["id"]

        # Approve with wrong hash
        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/approve",
            json={"request_hash": "wrong_hash"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"]["code"] == "HASH_MISMATCH"

    def test_double_approval_fails(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        task_id = resp.json()["id"]
        request_hash = resp.json()["request_hash"]

        # First approval
        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/approve",
            json={"request_hash": request_hash},
        )
        assert resp.status_code == 200

        # Second approval should fail
        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/approve",
            json={"request_hash": request_hash},
        )
        assert resp.status_code == 409

    def test_reject_pending_task(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        task_id = resp.json()["id"]

        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/reject",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["approval_state"] == "REJECTED"

    def test_cancel_pending_task(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        task_id = resp.json()["id"]

        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task_id}/cancel",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "CANCELLED"

    def test_cancel_running_returns_409(
        self, client: TestClient, project_with_asset, session: Session
    ):
        project, asset = project_with_asset
        # Create a task directly in RUNNING state
        task = AssetTask(
            project_id=project.id,
            kind="ENHANCEMENT",
            status=AssetTaskStatus.RUNNING.value,
            provider="ffmpeg",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="test_hash",
            approval_state="APPROVED",
            approved_request_hash="test_hash",
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        resp = client.post(
            f"/api/editor/projects/{project.id}/enhancements/{task.id}/cancel",
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["code"] == "TASK_ALREADY_RUNNING"

    def test_get_enhancement_task(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        task_id = resp.json()["id"]

        resp = client.get(f"/api/editor/projects/{project.id}/enhancements/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == task_id

    def test_list_enhancement_tasks(
        self, client: TestClient, project_with_asset
    ):
        project, asset = project_with_asset
        # Create multiple tasks
        for _ in range(3):
            client.post(
                f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
                json={"profile": "DEFAULT"},
            )

        resp = client.get(f"/api/editor/projects/{project.id}/enhancements")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 3

    def test_task_serializer_includes_output_asset_on_completed(
        self, client: TestClient, project_with_asset, session: Session
    ):
        project, asset = project_with_asset
        # Create a completed task with output asset
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
            kind="ENHANCEMENT",
            status=AssetTaskStatus.COMPLETED.value,
            provider="ffmpeg",
            model="ffmpeg 7.1 / DEFAULT",
            request={"source_asset_id": asset.id, "profile": "DEFAULT"},
            request_hash="hash123",
            approval_state="APPROVED",
            approved_request_hash="hash123",
            progress=1.0,
            output_asset_id=output_asset.id,
        )
        session.add(task)
        session.commit()
        session.refresh(task)

        resp = client.get(f"/api/editor/projects/{project.id}/enhancements/{task.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert "output_asset" in data
        assert data["output_asset"]["id"] == output_asset.id


class TestEnhancementProfileValidation:
    """Tests for profile-specific validation."""

    def test_upscale_requires_target_or_factor(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "UPSCALE"},
        )
        assert resp.status_code == 422  # validation error

    def test_upscale_with_factor_accepted(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "UPSCALE", "upscale_factor": 2.0},
        )
        assert resp.status_code == 200

    def test_upscale_with_dimensions_accepted(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "UPSCALE", "target_width": 2560, "target_height": 1440},
        )
        assert resp.status_code == 200

    def test_custom_requires_at_least_one_knob(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "CUSTOM"},
        )
        assert resp.status_code == 422

    def test_custom_with_knob_accepted(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "CUSTOM", "sharpen_strength": 1.0},
        )
        assert resp.status_code == 200


class TestEnhancementEstimate:
    """Tests for estimate behavior."""

    def test_estimate_is_local_zero_cost(self, client: TestClient, project_with_asset):
        project, asset = project_with_asset
        resp = client.post(
            f"/api/editor/projects/{project.id}/assets/{asset.id}/enhancements",
            json={"profile": "DEFAULT"},
        )
        data = resp.json()
        estimate = data["cost_estimate"]
        assert estimate["cost_class"] == "local"
        assert estimate["monetary_cost_usd"] == 0.0
        assert estimate["monetary_cost_known"] is True


class TestErrorHandling:
    """Tests for error handling behavior."""

    def test_generic_value_error_not_converted_to_422(
        self, client: TestClient, project_with_asset, session: Session
    ):
        """A generic ValueError in execute_claimed_enhancement_task should NOT be converted to 422."""
        from app.editor.enhancement import EnhancementRequest
        from app.editor.asset_tasks import AssetTaskStatus
        from app.editor.models import AssetTask

        project, asset = project_with_asset

        # Clean up any existing tasks to ensure we claim the right one
        session.exec(delete(AssetTask))
        session.commit()

        # Create a task in QUEUED state (ready for worker to pick up)
        request = EnhancementRequest(
            source_asset_id=asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(
            session, project_id=project.id, request=request
        )
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # Verify task is in QUEUED state
        assert task.status == AssetTaskStatus.QUEUED.value

        # Mock the provider's enhance method to raise a generic ValueError
        # This simulates an internal error during FFmpeg execution
        def mock_enhance(self, request, output_path):
            raise ValueError("internal must never become client validation")

        with patch("app.editor.ffmpeg_enhancement.FFmpegEnhancementProvider.enhance", mock_enhance):
            # Claim and process the task - this will call the mocked function
            claimed_id = worker_asset_tasks.claim_next_asset_task()
            assert claimed_id == task.id
            worker_asset_tasks.process_asset_task(claimed_id)

        # Check task status - should be FAILED with the internal error
        session.refresh(task)
        assert task.status == "FAILED"
        assert "internal must never become client validation" in task.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])