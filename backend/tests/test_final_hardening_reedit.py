from types import SimpleNamespace

from sqlmodel import Session

from app.api import clips, jobs
from app.core.db import engine, init_db
from app.models import Clip, Job, JobStatus, SourceType
from app.pipeline import ingest, reedit, render, transcribe


def test_reedit_chain_persists_source_timeline_and_export(monkeypatch, tmp_path):
    init_db()
    captured = []
    monkeypatch.setattr(ingest, "find_source", lambda _: tmp_path / "source.mp4")
    words = [{"word": name, "start": s, "end": e} for name, s, e in [
        ("kept", 0, 1), ("cut-first", 2, 3), ("cut-second", 4, 5), ("kept-last", 5, 6)
    ]]
    monkeypatch.setattr(transcribe, "load_segments", lambda _: {
        "segments": [{"start": 0, "end": 6, "words": words}]
    })
    def concat(src, kept, output):
        captured.append(kept)
        return output
    monkeypatch.setattr(reedit, "build_concat_base", concat)
    def rendered(*args, **kwargs):
        candidate = args[3]
        assert not candidate.source_ranges
        assert kwargs["subtitle_blocks"] is not None
        assert "cut-first" not in str(kwargs["subtitle_blocks"])
        if len(captured) == 2:
            assert "cut-second" not in str(kwargs["subtitle_blocks"])
        return [SimpleNamespace(format="16x9", path=str(tmp_path / "output.mp4"),
                                thumb_path=None, width=640, height=360,
                                duration=candidate.duration, size_bytes=10)]
    monkeypatch.setattr(render, "render_from_base", rendered)
    with Session(engine) as session:
        job = Job(source_type=SourceType.upload, source_name="source.mp4", status=JobStatus.done)
        session.add(job)
        session.flush()
        clip = Clip(job_id=job.id, index=0, start=0, end=10, score=0.73,
                    decision_results=[{"original": True}])
        session.add(clip)
        session.commit()
        first = clips.reedit_clip(clip.id, clips.ReeditRequest(remove=[[2, 4]]), session)
        assert first["duration"] == first["assets"][0]["duration"] == 8
        assert captured[-1] == [(0, 2), (4, 10)]
        second = clips.reedit_clip(clip.id, clips.ReeditRequest(remove=[[1, 3]]), session)
        assert captured[-1] == [(0, 1), (5, 10)]
        assert second["duration"] == second["assets"][0]["duration"] == 6
        assert second["score"] == 0.73
        assert second["decision_results"] == [{"original": True}]
        candidate = clips._candidate_from_clip(clip)
        assert [(r.start, r.end) for r in candidate.source_ranges] == [(0, 1), (5, 10)]
        assert candidate.duration == 6
        manifest = jobs.export_manifest(job.id, session)["clips"][0]
        assert manifest["duration"] == 6
        assert manifest["source_ranges"] == second["source_ranges"]
        transcript = clips.clip_transcript(clip.id, session)
        assert [w["word"] for w in transcript["words"]] == ["kept", "kept-last"]
        assert transcript["words"][1]["start"] == 1
