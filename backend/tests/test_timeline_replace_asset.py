"""Tests for timeline replace_asset operation (SNIPRA 1.6C Block 04)."""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from app.core.db import engine, init_db
from app.editor import service
from app.editor.asset_provenance import AssetOrigin, enhanced_provenance, source_provenance, write_asset_provenance
from app.editor.models import EditorAsset, EditorProject, EditorVersion
from app.editor.timeline import replace_clip_asset, new_track, new_clip, empty_timeline, recalculate_duration


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    init_db()


@pytest.fixture
def session():
    with Session(engine) as s:
        yield s


@pytest.fixture
def project_with_timeline(session: Session):
    """Create a project with a timeline containing a clip."""
    project = EditorProject(name="Test Project")
    session.add(project)
    session.flush()

    # Create source asset
    source_asset = EditorAsset(
        project_id=project.id,
        original_path="/fake/source.mp4",
        filename="source.mp4",
        fingerprint="src123",
        kind="video",
        duration_us=10_000_000,
        width=1280,
        height=720,
        has_audio=True,
        asset_metadata=write_asset_provenance({}, source_provenance(asset_id="temp")),
    )
    session.add(source_asset)
    session.flush()
    # Update provenance with actual asset id
    source_asset.asset_metadata = write_asset_provenance({}, source_provenance(asset_id=source_asset.id))
    session.add(source_asset)

    # Create enhanced asset
    enhanced_asset = EditorAsset(
        project_id=project.id,
        original_path="/fake/enhanced.mp4",
        filename="enhanced.mp4",
        fingerprint="enh123",
        kind="video",
        duration_us=10_000_000,
        width=1920,
        height=1080,
        has_audio=True,
        asset_metadata=write_asset_provenance(
            {},
            enhanced_provenance(
                parent=source_provenance(source_asset.id),
                parent_asset_id=source_asset.id,
                provider="ffmpeg",
                model_or_engine="ffmpeg 7.1 / DEFAULT",
                request_hash="req123",
                output_hash="outhash",
                output_size_bytes=1000000,
                operation_version="enh_v1",
                parameters={"profile": "DEFAULT"},
                duration_us=10_000_000,
                width=1920,
                height=1080,
                fps=30.0,
            ),
        ),
    )
    session.add(enhanced_asset)

    # Create timeline with a clip using source asset
    timeline = empty_timeline()
    video_track = new_track("video", 0)
    timeline["tracks"] = [video_track]  # Add track to timeline FIRST
    clip = new_clip(
        asset_id=source_asset.id,
        source_in_us=0,
        source_out_us=5_000_000,
        timeline_start_us=0,
    )
    timeline = recalculate_duration(
        service.tl.add_clip(timeline, video_track["id"], clip)
    )

    version = EditorVersion(
        project_id=project.id,
        number=0,
        parent_number=None,
        timeline=timeline,
        created_by="user",
        summary="Initial",
    )
    session.add(version)
    session.commit()
    session.refresh(project)
    session.refresh(source_asset)
    session.refresh(enhanced_asset)

    return project, source_asset, enhanced_asset, clip, timeline


class TestReplaceClipAsset:
    """Tests for replace_clip_asset timeline helper."""

    def test_replace_changes_only_asset_id(self, project_with_timeline):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        new_timeline = replace_clip_asset(timeline, clip['id'], enhanced_asset.id)

        # Find the clip in new timeline
        new_clip = None
        for track in new_timeline["tracks"]:
            for c in track["clips"]:
                if c["id"] == clip['id']:
                    new_clip = c
                    break

        assert new_clip is not None
        assert new_clip["asset_id"] == enhanced_asset.id

        # All other properties preserved
        assert new_clip["source_in_us"] == clip['source_in_us']
        assert new_clip["source_out_us"] == clip['source_out_us']
        assert new_clip["timeline_start_us"] == clip['timeline_start_us']
        assert new_clip["playback_rate"] == clip['playback_rate']
        assert new_clip["transform"] == clip['transform']
        assert new_clip["audio"] == clip['audio']
        assert new_clip["linked_group"] == clip['linked_group']

    def test_input_timeline_immutable(self, project_with_timeline):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        replace_clip_asset(timeline, clip['id'], enhanced_asset.id)

        # Original timeline unchanged
        original_clip = None
        for track in timeline["tracks"]:
            for c in track["clips"]:
                if c["id"] == clip['id']:
                    original_clip = c
                    break

        assert original_clip is not None
        assert original_clip["asset_id"] == source_asset.id

    def test_rejects_locked_track(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Lock the track
        locked_timeline = timeline.copy()
        for track in locked_timeline["tracks"]:
            if any(c["id"] == clip['id'] for c in track["clips"]):
                track["locked"] = True
                break

        with pytest.raises(Exception) as exc_info:
            replace_clip_asset(locked_timeline, clip['id'], enhanced_asset.id)
        # Check for the error code TRACK_LOCKED
        assert "TRACK_LOCKED" in str(exc_info.value) or "locked" in str(exc_info.value).lower()

    def test_rejects_caption_text_clip(self, project_with_timeline):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Add a text track for captions
        text_track = service.tl.new_track("text", 2)
        timeline_with_text = timeline.copy()
        timeline_with_text["tracks"] = timeline_with_text["tracks"] + [text_track]
        
        # Add a caption clip
        caption = service.tl.new_caption("Test", 0, 5_000_000)
        new_timeline = service.tl.add_timed_content(timeline_with_text, text_track["id"], caption)

        with pytest.raises(Exception) as exc_info:
            replace_clip_asset(new_timeline, caption['id'], enhanced_asset.id)
        assert "text/caption" in str(exc_info.value).lower() or "not" in str(exc_info.value).lower()


class TestReplaceAssetOperation:
    """Tests for replace_asset via apply_operation."""

    def test_replace_asset_requires_clip_and_asset(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Missing clip_id
        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test1",
                "type": "replace_asset",
                "parameters": {"asset_id": enhanced_asset.id},
            })
        assert "requires clip_id and asset_id" in str(exc_info.value).lower()

        # Missing asset_id
        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test2",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id']},
            })
        assert "requires clip_id and asset_id" in str(exc_info.value).lower()

    def test_replace_asset_clip_not_found(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test3",
                "type": "replace_asset",
                "parameters": {"clip_id": "nonexistent", "asset_id": enhanced_asset.id},
            })
        assert "CLIP_NOT_FOUND" in str(exc_info.value)

    def test_replace_asset_locked_track(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Lock the track
        service.apply_operation(session, project.id, {
            "base_version": 0,
            "idempotency_key": "lock1",
            "type": "track_lock",
            "parameters": {"track_id": timeline["tracks"][0]["id"], "locked": True},
        })
        session.refresh(project)

        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 1,
                "idempotency_key": "test4",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id'], "asset_id": enhanced_asset.id},
            })
        assert "TRACK_LOCKED" in str(exc_info.value)

    def test_replace_asset_rejects_caption(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Add text track for caption by creating a new version with the track
        text_track = service.tl.new_track("text", 2)
        new_timeline = timeline.copy()
        new_timeline["tracks"] = new_timeline["tracks"] + [text_track]
        
        # Create new version with text track
        from app.editor.models import EditorVersion
        new_version = EditorVersion(
            project_id=project.id,
            number=project.current_version + 1,
            parent_number=project.current_version,
            timeline=new_timeline,
            created_by="test",
            summary="Add text track",
        )
        session.add(new_version)
        project.current_version = new_version.number
        session.add(project)
        session.commit()
        session.refresh(project)

        # Now add caption to the text track
        service.apply_operation(session, project.id, {
            "base_version": project.current_version,
            "idempotency_key": "cap1",
            "type": "add_caption",
            "parameters": {"text": "Test", "start_us": 0, "end_us": 5_000_000, "track_id": text_track["id"]},
        })
        session.refresh(project)

        caption_id = None
        current_version = service.current_version(session, project)
        for track in current_version.timeline["tracks"]:
            for c in track["clips"]:
                if c.get("kind") == "caption":
                    caption_id = c["id"]
                    break
        
        # Try to use caption as replace_asset source - should fail
        current_ver = service.current_version(session, project)
        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": current_ver.number,
                "idempotency_key": "test5",
                "type": "replace_asset",
                "parameters": {"clip_id": caption_id, "asset_id": enhanced_asset.id},
            })
        assert "caption" in str(exc_info.value).lower() or "text" in str(exc_info.value).lower()

    def test_replace_asset_target_must_belong_to_project(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Create another project with an asset
        other_project = EditorProject(name="Other Project")
        session.add(other_project)
        session.flush()

        other_asset = EditorAsset(
            project_id=other_project.id,
            original_path="/fake/other.mp4",
            filename="other.mp4",
            fingerprint="other123",
            kind="video",
            duration_us=5_000_000,
            width=640,
            height=480,
            has_audio=False,
        )
        session.add(other_asset)
        session.commit()

        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test6",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id'], "asset_id": other_asset.id},
            })
        assert "ASSET_NOT_FOUND" in str(exc_info.value)

    def test_replace_asset_requires_enhanced_origin(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Try to replace with another SOURCE asset
        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test7",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id'], "asset_id": source_asset.id},
            })
        assert "INVALID_ASSET" in str(exc_info.value) or "ENHANCED" in str(exc_info.value)

    def test_replace_asset_requires_lineage_match(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Create a different enhanced asset from a different source
        other_source = EditorAsset(
            project_id=project.id,
            original_path="/fake/other_source.mp4",
            filename="other_source.mp4",
            fingerprint="other_src",
            kind="video",
            duration_us=5_000_000,
            width=640,
            height=480,
            has_audio=False,
        )
        session.add(other_source)
        session.flush()

        other_enhanced = EditorAsset(
            project_id=project.id,
            original_path="/fake/other_enhanced.mp4",
            filename="other_enhanced.mp4",
            fingerprint="other_enh",
            kind="video",
            duration_us=5_000_000,
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata=write_asset_provenance(
                {},
                enhanced_provenance(
                    parent=source_provenance(other_source.id),
                    parent_asset_id=other_source.id,
                    provider="ffmpeg",
                    model_or_engine="ffmpeg 7.1 / DEFAULT",
                    request_hash="req456",
                    output_hash="outhash456",
                    output_size_bytes=2000000,
                    operation_version="enh_v1",
                    parameters={"profile": "DEFAULT"},
                    duration_us=5_000_000,
                    width=1280,
                    height=720,
                    fps=30.0,
                ),
            ),
        )
        session.add(other_enhanced)
        session.commit()

        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test8",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id'], "asset_id": other_enhanced.id},
            })
        assert "LINEAGE_MISMATCH" in str(exc_info.value)

    def test_replace_asset_requires_sufficient_duration(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Create enhanced asset shorter than clip's source range
        short_enhanced = EditorAsset(
            project_id=project.id,
            original_path="/fake/short_enhanced.mp4",
            filename="short_enhanced.mp4",
            fingerprint="short_enh",
            kind="video",
            duration_us=2_000_000,  # Less than clip's source_out_us (5_000_000)
            width=1280,
            height=720,
            has_audio=True,
            asset_metadata=write_asset_provenance(
                {},
                enhanced_provenance(
                    parent=source_provenance(source_asset.id),
                    parent_asset_id=source_asset.id,
                    provider="ffmpeg",
                    model_or_engine="ffmpeg 7.1 / DEFAULT",
                    request_hash="req789",
                    output_hash="shortouthash",
                    output_size_bytes=500000,
                    operation_version="enh_v1",
                    parameters={"profile": "DEFAULT"},
                    duration_us=2_000_000,
                    width=1280,
                    height=720,
                    fps=30.0,
                ),
            ),
        )
        session.add(short_enhanced)
        session.commit()

        with pytest.raises(Exception) as exc_info:
            service.apply_operation(session, project.id, {
                "base_version": 0,
                "idempotency_key": "test9",
                "type": "replace_asset",
                "parameters": {"clip_id": clip['id'], "asset_id": short_enhanced.id},
            })
        assert "DURATION_INSUFFICIENT" in str(exc_info.value)

    def test_replace_asset_preserves_source_timing(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        result = service.apply_operation(session, project.id, {
            "base_version": 0,
            "idempotency_key": "test10",
            "type": "replace_asset",
            "parameters": {"clip_id": clip['id'], "asset_id": enhanced_asset.id},
        })

        # Check new timeline
        new_timeline = result["timeline"]
        new_clip = None
        for track in new_timeline["tracks"]:
            for c in track["clips"]:
                if c["id"] == clip['id']:
                    new_clip = c
                    break

        assert new_clip is not None
        assert new_clip["asset_id"] == enhanced_asset.id
        assert new_clip["source_in_us"] == clip['source_in_us']
        assert new_clip["source_out_us"] == clip['source_out_us']
        assert new_clip["timeline_start_us"] == clip['timeline_start_us']
        assert new_clip["playback_rate"] == clip['playback_rate']
        assert new_clip["transform"] == clip['transform']
        assert new_clip["audio"] == clip['audio']
        assert new_clip["linked_group"] == clip['linked_group']

    def test_replace_asset_increments_version(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        assert project.current_version == 0

        result = service.apply_operation(session, project.id, {
            "base_version": 0,
            "idempotency_key": "test11",
            "type": "replace_asset",
            "parameters": {"clip_id": clip['id'], "asset_id": enhanced_asset.id},
        })

        assert result["version"] == 1
        session.refresh(project)
        assert project.current_version == 1

    def test_replace_asset_creates_editor_operation(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        result = service.apply_operation(session, project.id, {
            "base_version": 0,
            "idempotency_key": "test12",
            "type": "replace_asset",
            "parameters": {"clip_id": clip['id'], "asset_id": enhanced_asset.id},
        })

        assert "operation_id" in result
        ops = session.exec(
            select(service.EditorOperation).where(
                service.EditorOperation.project_id == project.id
            )
        ).all()
        assert len(ops) == 1
        assert ops[0].kind == "replace_asset"
        assert ops[0].payload["clip_id"] == clip['id']
        assert ops[0].payload["asset_id"] == enhanced_asset.id

    def test_replace_asset_undo_restores_original(self, project_with_timeline, session: Session):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Replace
        service.apply_operation(session, project.id, {
            "base_version": 0,
            "idempotency_key": "test13",
            "type": "replace_asset",
            "parameters": {"clip_id": clip['id'], "asset_id": enhanced_asset.id},
        })
        session.refresh(project)

        # Undo
        service.undo(session, project.id)
        session.refresh(project)

        # Verify original asset restored
        timeline = service.current_version(session, project).timeline
        restored_clip = None
        for track in timeline["tracks"]:
            for c in track["clips"]:
                if c["id"] == clip['id']:
                    restored_clip = c
                    break

        assert restored_clip is not None
        assert restored_clip["asset_id"] == source_asset.id


class TestNoAutoReplacement:
    """Tests that enhancement completion does NOT auto-replace in timeline."""

    def test_completed_enhancement_does_not_mutate_timeline(
        self, project_with_timeline, session: Session
    ):
        project, source_asset, enhanced_asset, clip, timeline = project_with_timeline

        # Manually create a completed enhancement task
        from app.editor import asset_task_service
        from app.editor.enhancement import EnhancementRequest, EnhancementProfile
        from app.editor.asset_tasks import AssetTaskStatus

        request = EnhancementRequest(
            source_asset_id=source_asset.id,
            profile=EnhancementProfile.DEFAULT,
        )
        task = asset_task_service.create_enhancement_task(session, project_id=project.id, request=request)
        task = asset_task_service.attach_estimate(session, task.id)
        task = asset_task_service.approve_task(session, task.id, request_hash=task.request_hash)
        task = asset_task_service.queue_task(session, task.id)

        # Simulate completion by directly setting task to COMPLETED with output asset
        task.status = AssetTaskStatus.COMPLETED.value
        task.output_asset_id = enhanced_asset.id
        task.model = "ffmpeg 7.1 / DEFAULT"
        task.progress = 1.0
        session.add(task)
        session.commit()

        # Timeline should still have source asset
        current_version = service.current_version(session, project)
        timeline = current_version.timeline
        clip_in_timeline = None
        for track in timeline["tracks"]:
            for c in track["clips"]:
                if c["id"] == clip['id']:
                    clip_in_timeline = c
                    break

        assert clip_in_timeline is not None
        assert clip_in_timeline["asset_id"] == source_asset.id