from __future__ import annotations

import uuid
from pathlib import Path

from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import agent, auto_import, service
from app.models import Clip, ClipAsset, Job, SourceType


def _seed(session: Session, ranges: list[dict] | None = None) -> tuple[Job, Clip]:
    suffix = uuid.uuid4().hex
    job = Job(
        id=f"auto-job-{suffix}",
        source_type=SourceType.upload,
        source_name="original.mp4",
        title="Auto source",
        duration=1_500.0,
        source_width=1920,
        source_height=1080,
        processing_mode="gaming_highlights",
    )
    clip = Clip(
        id=f"auto-clip-{suffix}",
        job_id=job.id,
        index=2,
        start=1109.0,
        end=1129.5,
        title="Auto highlight",
        source_ranges=ranges or [],
    )
    session.add(job)
    session.add(clip)
    session.commit()
    return job, clip


def _video_clips(session: Session, project_id: str) -> list[dict]:
    project = service.project_or_404(session, project_id)
    timeline = service.current_version(session, project).timeline
    return next(track["clips"] for track in timeline["tracks"] if track["type"] == "video")


def test_auto_import_single_range_is_editable_idempotent_and_agent_visible(monkeypatch, tmp_path):
    init_db()
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(auto_import.ingest, "find_source", lambda _job_id: source)
    with Session(engine) as session:
        job, clip = _seed(session)
        project, created = auto_import.open_auto_result(session, clip.id)
        assert created is True
        imported = _video_clips(session, project.id)
        assert len(imported) == 1
        assert (imported[0]["source_in_us"], imported[0]["source_out_us"]) == (
            1_109_000_000,
            1_129_500_000,
        )
        assert imported[0]["provenance"]["origin"] == "AUTO_IMPORT"
        assert imported[0]["provenance"]["source_job_id"] == job.id
        imported_asset = service.project_assets(session, project.id)[0]
        assert Path(imported_asset.original_path) == source.resolve()

        same, created_again = auto_import.open_auto_result(session, clip.id)
        assert same.id == project.id
        assert created_again is False
        assert len(_video_clips(session, project.id)) == 1

        before_trim = service.current_version(session, project).timeline
        service.apply_operation(
            session,
            project.id,
            {
                "type": "trim",
                "parameters": {"clip_id": imported[0]["id"], "source_out_us": 1_120_000_000},
                "base_version": project.current_version,
                "idempotency_key": f"trim-{clip.id}",
            },
        )
        restored = service.restore_version(session, project.id, 1)
        assert restored["timeline"] == before_trim
        inspected = agent.inspect_project(session, project.id)
        provenance = inspected["timeline"]["tracks"][0]["clips"][0]["provenance"]
        assert provenance["candidate_id"] == clip.id


def test_auto_import_multirange_preserves_order_and_exact_source_mapping(monkeypatch, tmp_path):
    init_db()
    source = tmp_path / "original.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(auto_import.ingest, "find_source", lambda _job_id: source)
    ranges = [
        {"start": 30.0, "end": 34.25},
        {"start": 10.0, "end": 12.0},
        {"start": 70.5, "end": 73.0},
    ]
    with Session(engine) as session:
        _, clip = _seed(session, ranges)
        project, _ = auto_import.open_auto_result(session, clip.id)
        imported = _video_clips(session, project.id)
        assert [(item["source_in_us"], item["source_out_us"]) for item in imported] == [
            (10_000_000, 12_000_000),
            (30_000_000, 34_250_000),
            (70_500_000, 73_000_000),
        ]
        assert [item["timeline_start_us"] for item in imported] == [0, 2_000_000, 6_250_000]
        assert all(item["provenance"]["flattened"] is False for item in imported)


def test_auto_import_falls_back_to_a_flattened_render_when_mapping_is_unavailable(
    monkeypatch, tmp_path
):
    init_db()
    missing = tmp_path / "missing.mp4"
    rendered = tmp_path / "rendered.mp4"
    rendered.write_bytes(b"rendered")
    monkeypatch.setattr(auto_import.ingest, "find_source", lambda _job_id: missing)
    with Session(engine) as session:
        _, clip = _seed(session, [{"start": "bad", "end": 2}])
        session.add(
            ClipAsset(
                clip_id=clip.id,
                format="9x16",
                path=str(rendered),
                duration=8.5,
                width=1080,
                height=1920,
            )
        )
        session.commit()
        project, _ = auto_import.open_auto_result(session, clip.id)
        imported = _video_clips(session, project.id)
        assert [(item["source_in_us"], item["source_out_us"]) for item in imported] == [
            (0, 8_500_000)
        ]
        assert imported[0]["provenance"]["flattened"] is True
        assert imported[0]["provenance"]["reconstruction"] == "RENDERED_FALLBACK"
