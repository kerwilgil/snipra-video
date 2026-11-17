"""Decision foundation validation and real worker/media safety regressions."""

import asyncio
import math

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select

from app.core.config import Settings, settings
from app.decision.engines import RulesDecisionEngine
from app.decision.evidence import normalize_candidate
from app.decision.models import (
    DEFAULT_OPTIONS,
    ClipEvidence,
    DecisionRequest,
    DecisionResult,
    DecisionTask,
)
from app.decision.registry import (
    get_decision_engine,
    list_decision_engines,
    register_decision_engine,
)
from app.decision.service import classify_batch, evaluate_candidates
from app.providers.base import ClipCandidate, SourceRange


def request(task=DecisionTask.clip_quality, **overrides):
    data = dict(
        task=task,
        video_id="video",
        segment_id="0",
        evidence=ClipEvidence(
            start_ms=0,
            end_ms=10000,
            duration_ms=10000,
            existing_score=0.87,
            evidence_refs=["candidate:video:0"],
        ),
        options=list(DEFAULT_OPTIONS.get(task, ("excellent", "review"))),
    )
    return DecisionRequest(**(data | overrides))


def result(**overrides):
    data = dict(
        engine="rules",
        task="clip_quality",
        status="success",
        classification="excellent",
        probabilities={"excellent": 1, "review": 0},
    )
    return DecisionResult(**(data | overrides))


def config(**overrides):
    return Settings(_env_file=None, **(dict(decision_engine_enabled=True) | overrides))


def test_valid_contracts_and_score_separation():
    assert request().evidence.existing_score == 0.87
    assert result().probabilities["excellent"] == 1
    assert DecisionRequest.model_validate_json(request().model_dump_json()) == request()
    assert DecisionResult.model_validate_json(result().model_dump_json()) == result()


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "unknown"},
        {"options": []},
        {"options": [""]},
        {"options": ["a", "a"]},
        {"video_id": ""},
    ],
)
def test_invalid_request(overrides):
    with pytest.raises(ValidationError):
        request(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_ms": -1},
        {"end_ms": 0},
        {"end_ms": 1},
        {"duration_ms": 1},
        {"start_ms": math.nan},
        {"existing_score": 87},
        {"signals": {"nested": [math.inf]}},
        {"audio": {"silence_ratio": -1}},
        {"source_ranges": [SourceRange(start=0, end=11)]},
        {"source_ranges": [SourceRange(start=0, end=math.inf)]},
    ],
)
def test_invalid_evidence(overrides):
    with pytest.raises(ValidationError):
        ClipEvidence(**(dict(start_ms=0, end_ms=10000, duration_ms=10000) | overrides))


@pytest.mark.parametrize("p", [0, 1, 0.37])
def test_valid_probability(p):
    assert result(probabilities={"excellent": p, "review": 1 - p})


@pytest.mark.parametrize("p", [-0.1, 1.01, math.nan, math.inf, -math.inf])
def test_invalid_probability(p):
    with pytest.raises(ValidationError):
        result(probabilities={"excellent": p, "review": 0})


def test_sum_tolerance_and_nonexclusive():
    assert result(probabilities={"excellent": 0.5, "review": 0.5000005})
    with pytest.raises(ValidationError):
        result(probabilities={"excellent": 0.5, "review": 0.50001})
    assert result(probabilities={"excellent": 0.5, "review": 0.8}, mutually_exclusive=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"classification": "unknown"},
        {"status": "error"},
        {"probabilities": {}},
        {"confidence": math.nan},
        {"recommended_action": "discard"},
    ],
)
def test_invalid_result(overrides):
    with pytest.raises(ValidationError):
        result(**overrides)


@pytest.mark.parametrize("task", list(DecisionTask))
async def test_rules_deterministic_and_provenance(task):
    req = request(task)
    if task in {DecisionTask.content_type, DecisionTask.hook_detection}:
        req.evidence.signals[task.value] = "hook"
    engine = RulesDecisionEngine()
    first = await engine.classify(req)
    assert first == await engine.classify(req)
    assert first.status == "success"
    assert first.classification in req.options
    assert first.evidence_refs == req.evidence.evidence_refs
    assert first.probabilities[first.classification] == 1


async def test_semantic_abstention_and_no_score_abstention():
    engine = RulesDecisionEngine()
    assert (await engine.classify(request(DecisionTask.content_type))).status == "unavailable"
    assert (await engine.classify(request(DecisionTask.hook_detection))).status == "unavailable"
    req = request()
    req.evidence.existing_score = None
    assert (await engine.classify(req)).status == "unavailable"


def test_registry(monkeypatch):
    from app.decision import registry

    monkeypatch.setattr(registry, "_ENGINES", {})
    register_decision_engine(RulesDecisionEngine())
    assert list_decision_engines() == ["rules"]
    assert get_decision_engine("rules").id() == "rules"
    assert get_decision_engine("clef-flash") is None
    with pytest.raises(ValueError):
        register_decision_engine(RulesDecisionEngine())


def test_default_configuration(monkeypatch):
    for name in ("ENABLED", "PROVIDER", "MIN_CONFIDENCE", "TIMEOUT_MS"):
        monkeypatch.delenv(f"DECISION_ENGINE_{name}", raising=False)
    cfg = Settings(_env_file=None)
    assert cfg.decision_engine_enabled is False
    assert cfg.decision_engine_provider == "rules"
    assert cfg.decision_engine_min_confidence == 0.65
    assert cfg.decision_engine_timeout_ms == 2000


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision_engine_min_confidence", math.nan),
        ("decision_engine_min_confidence", 1.1),
        ("decision_engine_timeout_ms", 0),
        ("decision_engine_provider", ""),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValidationError):
        config(**{field: value})


class FaultEngine(RulesDecisionEngine):
    def __init__(self, mode):
        self.mode = mode

    async def available(self):
        if self.mode == "unavailable":
            return False
        if self.mode == "availability_error":
            raise RuntimeError("availability failed")
        return True

    async def batch_classify(self, requests):
        if self.mode == "timeout":
            await asyncio.sleep(1)
        if self.mode == "error":
            raise RuntimeError("provider failed")
        if self.mode == "short_batch":
            return []
        results = await super().batch_classify(requests)
        if self.mode == "bad_task":
            results[0].task = DecisionTask.content_type
        if self.mode == "bad_probability":
            results[0].probabilities["excellent"] = math.nan
        if self.mode == "fake_provenance":
            results[0].evidence_refs = ["invented"]
        if self.mode == "low_confidence":
            results[0].confidence = 0.1
        return results


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("disabled", "disabled"),
        ("unknown", "unavailable"),
        ("unavailable", "unavailable"),
        ("timeout", "timeout"),
        ("error", "error"),
        ("availability_error", "error"),
        ("short_batch", "error"),
        ("bad_task", "error"),
        ("bad_probability", "error"),
        ("fake_provenance", "error"),
        ("low_confidence", "unavailable"),
    ],
)
async def test_fallback(monkeypatch, mode, expected):
    from app.decision import registry

    monkeypatch.setitem(registry._ENGINES, "rules", FaultEngine(mode))
    cfg = config(
        decision_engine_enabled=mode != "disabled",
        decision_engine_provider="unknown" if mode == "unknown" else "rules",
        decision_engine_timeout_ms=10,
    )
    results = await classify_batch([request()], cfg)
    assert results[0].status == expected
    assert results[0].evidence_refs == request().evidence.evidence_refs
    assert results[0].duration_ms >= 0
    assert not results[0].probabilities


def test_adapter_existing_features_montage_and_missing_data():
    candidate = ClipCandidate(
        start=0,
        end=10,
        score=0.87,
        source_ranges=[SourceRange(start=0, end=2), SourceRange(start=8, end=10)],
    )
    before = candidate.model_dump()
    doc = {
        "segments": [
            {
                "start": 0,
                "end": 2,
                "text": "Hola",
                "words": [{"start": 0, "end": 1, "word": "Hola"}],
            }
        ]
    }
    evidence = normalize_candidate(
        candidate,
        doc,
        {
            "loudness": [[1, -12], [5, -1], [9, -8]],
            "scene_times": [1, 5, 9],
            "silence_intervals": [[0, 1], [0.5, 1.5], [4, 6]],
        },
        video_id="v",
    )
    assert evidence.duration_ms == 4000
    assert evidence.audio.energy == -10
    assert evidence.audio.silence_ratio == 0.375
    assert evidence.audio.speech_ratio == 0.5
    assert evidence.speech.words_per_second == 0.25
    assert evidence.visual.scene_changes == 2
    assert evidence.visual.motion_intensity is None
    assert evidence.speech.sentiment is None
    assert candidate.model_dump() == before
    empty = normalize_candidate(candidate, {}, {}, video_id="v")
    assert empty.audio.energy is None
    assert empty.speech.transcript is None
    assert len(empty.evidence_refs) == 1


def test_disabled_skips_adapter(monkeypatch):
    from app.decision import evidence

    monkeypatch.setattr(evidence, "normalize_candidate", lambda *a, **k: pytest.fail("called"))
    assert (
        evaluate_candidates(
            [object()], {}, {}, video_id="v", config=config(decision_engine_enabled=False)
        )
        == []
    )


@pytest.mark.parametrize(
    "mode", ["disabled", "unknown", "unavailable", "timeout", "error", "service_error", "success"]
)
def test_worker_detection_score_generation_export_isolation(
    monkeypatch, sample_video, tmp_path, mode
):
    from app.api.clips import _serialize_clip
    from app.api.jobs import export_manifest
    from app.core.db import engine, init_db
    from app.core.media import probe
    from app.models import Clip, ClipAsset, Job, JobStatus, SourceType
    from app.pipeline import detect, render
    from app.worker import runner
    from app.decision import registry

    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "decision_engine_enabled", mode != "disabled")
    monkeypatch.setattr(
        settings, "decision_engine_provider", "unknown" if mode == "unknown" else "rules"
    )
    monkeypatch.setattr(settings, "decision_engine_timeout_ms", 10)
    monkeypatch.setattr(settings, "enable_visual_rerank", False)
    monkeypatch.setattr(settings, "enable_media_signals", False)
    monkeypatch.setattr(settings, "min_clip_seconds", 3)
    monkeypatch.setattr(settings, "max_clip_seconds", 10)
    monkeypatch.setitem(registry._ENGINES, "rules", FaultEngine(mode))
    if mode == "service_error":

        def broken(*args, **kwargs):
            raise RuntimeError("service defect")

        monkeypatch.setattr(runner, "evaluate_candidates", broken)
    init_db()
    job = Job(
        source_type=SourceType.upload,
        source_name="sample.mp4",
        processing_mode="music",
        num_clips=1,
        formats="16x9",
        captions_enabled=False,
        sponsorblock_enabled=False,
    )
    with Session(engine) as session:
        session.add(job)
        session.commit()
        job_id = job.id
    info = probe(sample_video)
    baseline = detect.detect_music_clips(
        info.duration, {}, num_clips=1, min_seconds=3, max_seconds=10
    )
    # Real probe, detector, persistence, FFmpeg renderer and export; no Whisper/network required.
    monkeypatch.setattr(runner.ingest, "find_source", lambda _: sample_video)
    monkeypatch.setattr(runner.ingest, "probe_source", lambda _: info)
    runner.process_job(job_id)
    with Session(engine) as session:
        assert session.get(Job, job_id).status == JobStatus.done
        clip = session.exec(select(Clip).where(Clip.job_id == job_id)).one()
        assets = session.exec(select(ClipAsset).where(ClipAsset.clip_id == clip.id)).all()
        assert clip.score == baseline[0].score
        assert clip.start == baseline[0].start and clip.end == baseline[0].end
        assert clip.score_breakdown == [x.model_dump() for x in baseline[0].score_breakdown]
        assert assets and probe(assets[0].path).duration > 0
        manifest = export_manifest(job_id, session)
        assert manifest["clips"][0]["score"] == baseline[0].score
        assert manifest["clips"][0]["assets"][0]["path"] == assets[0].path
        api_clip = _serialize_clip(clip, assets)
        if mode == "disabled":
            assert not clip.decision_results
            assert "decision_results" not in api_clip
            assert "decision_results" not in manifest["clips"][0]
        elif mode == "success":
            assert len(clip.decision_results) == 5
            assert clip.decision_results[0]["classification"] == "adequate"
            assert api_clip["decision_results"] == manifest["clips"][0]["decision_results"]
        else:
            expected = (
                "error"
                if mode in {"error", "service_error"}
                else ("unavailable" if mode == "unknown" else mode)
            )
            assert {x["status"] for x in clip.decision_results} == {expected}
            assert api_clip["decision_results"] == manifest["clips"][0]["decision_results"]


async def test_non_batch_adapter_and_batch_order(monkeypatch):
    from app.decision import registry

    class SingleEngine:
        def id(self):
            return "local-model"

        async def available(self):
            return True

        async def classify(self, req):
            result = await RulesDecisionEngine().classify(req)
            return result.model_copy(update={"engine": self.id()})

    monkeypatch.setattr(registry, "_ENGINES", {})
    register_decision_engine(SingleEngine())
    requests = [request(), request(DecisionTask.recommended_action)] * 10
    results = await classify_batch(requests, config(decision_engine_provider="local-model"))
    assert len(results) == 20
    assert [x.task for x in results] == [x.task for x in requests]
    assert all(x.status == "success" for x in results)


def test_invalid_adapter_evidence_isolated():
    candidates = [ClipCandidate(start=1, end=0.5), ClipCandidate(start=0, end=10)]
    results = evaluate_candidates(candidates, {}, {}, video_id="v", config=config())
    assert {x["status"] for x in results[0]} == {"error"}
    assert results[1][0]["status"] == "success"


def test_environment_config(monkeypatch):
    monkeypatch.setenv("DECISION_ENGINE_ENABLED", "true")
    monkeypatch.setenv("DECISION_ENGINE_PROVIDER", "future-adapter")
    assert Settings(_env_file=None).decision_engine_enabled is True
    assert Settings(_env_file=None).decision_engine_provider == "future-adapter"


def test_existing_database_additive_migration(monkeypatch, tmp_path):
    from sqlalchemy import inspect, text
    from sqlmodel import create_engine
    from app.core import db

    test_engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with test_engine.begin() as conn:
        conn.execute(text("CREATE TABLE job (id VARCHAR PRIMARY KEY, updated_at DATETIME)"))
        conn.execute(text("CREATE TABLE clip (id VARCHAR PRIMARY KEY, score FLOAT)"))
        conn.execute(text("INSERT INTO clip VALUES ('old', 0.87)"))
    monkeypatch.setattr(db, "engine", test_engine)
    db._migrate_existing_sqlite()
    db._migrate_existing_sqlite()
    assert "decision_results" in {c["name"] for c in inspect(test_engine).get_columns("clip")}
    with test_engine.connect() as conn:
        assert conn.execute(text("SELECT score, decision_results FROM clip")).one() == (0.87, "[]")
