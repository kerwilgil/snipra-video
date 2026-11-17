"""Tests de score_breakdown (P0.1): observabilidad aditiva, sin cambiar score/reason.

Cada punto de población debe demostrar: (a) el score y el reason finales son
EXACTAMENTE los mismos de antes del cambio, (b) score_breakdown gana una
entrada nueva con score_before/contribution/score_after consistentes.
"""

from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.db import engine, init_db
from app.models import Clip, Job, JobStatus, SourceType
from app.pipeline.editorial import compact_candidate
from app.pipeline.gaming_edits import annotate_visual_rewards, sanitize_gaming_outcomes
from app.pipeline.signals import apply_signal_scores, score_candidate
from app.providers import vision
from app.providers.base import ClipCandidate, ScoreContribution, SourceRange


def _candidate(
    start: float, end: float, score: float, reason: str = "señal gaming"
) -> ClipCandidate:
    return ClipCandidate(start=start, end=end, score=score, title=f"Jugada {start}", reason=reason)


# ---------------------------------------------------------------------------
# signals.py::apply_signal_scores
# ---------------------------------------------------------------------------


def test_apply_signal_scores_appends_breakdown_without_changing_score_or_reason():
    signals_doc = {
        "scene_times": [10.0, 14.0],
        "active_intervals": [(8.0, 20.0)],
        "silence_intervals": [(6.5, 8.0), (20.0, 22.0)],
    }
    cand = ClipCandidate(start=8.0, end=20.0, score=0.5, reason="base")

    bonus, labels = score_candidate(cand, signals_doc, processing_mode="gaming")
    adjusted = apply_signal_scores([cand], signals_doc, processing_mode="gaming")[0]

    expected_score = min(1.0, cand.score + bonus)
    expected_reason = f"{cand.reason}; señales: " + ", ".join(labels[:3])
    assert adjusted.score == expected_score
    assert adjusted.reason == expected_reason

    assert len(adjusted.score_breakdown) == 1
    entry = adjusted.score_breakdown[0]
    assert entry.signal == "señales_audio_escena"
    assert entry.score_before == cand.score
    assert entry.score_after == expected_score
    assert entry.contribution == pytest.approx(expected_score - cand.score)


def test_apply_signal_scores_no_bonus_leaves_breakdown_empty():
    cand = ClipCandidate(start=0.0, end=5.0, score=0.5)
    adjusted = apply_signal_scores([cand], {}, processing_mode="auto")[0]
    assert adjusted.score == 0.5
    assert adjusted.score_breakdown == []


# ---------------------------------------------------------------------------
# gaming_edits.py::annotate_visual_rewards
# ---------------------------------------------------------------------------


def test_annotate_visual_rewards_defeat_records_negative_breakdown(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        Path(cmd[-2]).write_bytes(b"frame")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", lambda path: ["eliminacion", "teelimino:"])
    candidate = _candidate(400, 430, 0.70)

    result = annotate_visual_rewards([candidate], "fake.mp4")[0]

    assert result.score == pytest.approx(0.48)  # comportamiento preexistente sin cambios
    assert len(result.score_breakdown) == 1
    entry = result.score_breakdown[0]
    assert entry.signal == "recompensa_ocr"
    assert entry.score_before == 0.70
    assert entry.score_after == pytest.approx(0.48)
    assert entry.contribution == pytest.approx(-0.22)


def test_annotate_visual_rewards_reward_records_positive_breakdown(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        Path(cmd[-2]).write_bytes(b"frame")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", lambda path: ["victoria"])
    candidate = _candidate(400, 430, 0.60)

    result = annotate_visual_rewards([candidate], "fake.mp4")[0]

    assert result.score == pytest.approx(0.76)  # 0.60 + 0.16, comportamiento preexistente
    entry = result.score_breakdown[0]
    assert entry.signal == "recompensa_ocr"
    assert entry.contribution == pytest.approx(0.16)


# ---------------------------------------------------------------------------
# providers/vision.py::rerank_by_vision
# ---------------------------------------------------------------------------


def test_vision_rerank_records_breakdown_entry_with_delta(monkeypatch):
    cand = ClipCandidate(start=0, end=30, score=0.80, title="a")
    monkeypatch.setattr(vision, "_extract_frames", lambda *a, **k: [b"jpgbytes"])
    monkeypatch.setattr(vision, "_score_claude", lambda model, frames, title: (0.10, "estático"))

    out = vision.rerank_by_vision([cand], "fake.mp4", provider="claude", model="claude-x")[0]

    expected = round(0.55 * 0.80 + 0.45 * 0.10, 3)
    assert out.score == expected  # comportamiento preexistente sin cambios
    assert len(out.score_breakdown) == 1
    entry = out.score_breakdown[0]
    assert entry.signal == "vision_rerank"
    assert entry.score_before == 0.80
    assert entry.score_after == expected
    assert entry.contribution == pytest.approx(expected - 0.80)
    assert entry.note == "estático"


# ---------------------------------------------------------------------------
# gaming_edits.py::sanitize_gaming_outcomes
# ---------------------------------------------------------------------------


def test_sanitize_gaming_outcomes_records_breakdown_entry(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        Path(cmd[-2]).write_bytes(b"frame")

    def fake_texts(path):
        return ["scoreboard"] if "qa-0-0-2" in path else []

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", fake_texts)
    edit = _candidate(10, 80, 0.8).model_copy(
        update={
            "source_ranges": [
                SourceRange(start=10, end=20, kind="setup", score=0.75),
                SourceRange(start=70, end=80, kind="payoff", score=0.85),
            ]
        }
    )

    result = sanitize_gaming_outcomes([edit], "fake.mp4")[0]

    assert len(result.source_ranges) == 2  # comportamiento preexistente sin cambios
    assert len(result.score_breakdown) == 1
    entry = result.score_breakdown[0]
    assert entry.signal == "qa_ocr_final"
    assert entry.score_before == 0.8
    assert entry.score_after == result.score


# ---------------------------------------------------------------------------
# editorial.py::compact_candidate
# ---------------------------------------------------------------------------


def test_compact_candidate_records_zero_contribution_breakdown_entry():
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Primera parte larga de la explicacion completa."},
        {"start": 7.0, "end": 15.0, "text": "Segunda parte distinta que continua el tema."},
    ]
    silence_intervals = [(5.0, 7.0)]
    candidate = _candidate(0.0, 15.0, 0.6, reason="base")

    result = compact_candidate(candidate, segments, silence_intervals)

    assert result.score == candidate.score  # compact_candidate nunca toca el score
    if result.source_ranges:  # solo si el recorte fue suficiente para aplicarse
        assert len(result.score_breakdown) == 1
        entry = result.score_breakdown[0]
        assert entry.signal == "edicion_compacta"
        assert entry.contribution == 0.0
        assert entry.score_before == entry.score_after == candidate.score


# ---------------------------------------------------------------------------
# Persistencia / round-trip completo
# ---------------------------------------------------------------------------


def test_score_breakdown_round_trips_through_clip_persistence(tmp_path, monkeypatch):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    init_db()
    breakdown = [
        ScoreContribution(
            signal="vision_rerank",
            stage="semantic_validation",
            score_before=0.71,
            contribution=0.16,
            score_after=0.87,
            at=38.4,
            note="verified payoff",
        ).model_dump()
    ]
    with Session(engine) as session:
        job = Job(
            source_type=SourceType.url,
            source_name="https://example.com/v.mp4",
            status=JobStatus.done,
        )
        session.add(job)
        session.flush()
        clip = Clip(
            job_id=job.id,
            index=0,
            start=0,
            end=10,
            title="Momento",
            score=0.87,
            score_breakdown=breakdown,
        )
        session.add(clip)
        session.commit()
        clip_id = clip.id

    with Session(engine) as session:
        reloaded = session.get(Clip, clip_id)
        assert reloaded is not None
        assert reloaded.score_breakdown == breakdown
        assert reloaded.score_breakdown[0]["signal"] == "vision_rerank"
        assert reloaded.score_breakdown[0]["score_after"] == 0.87
