"""Tests del harness de benchmark (Paso 7 del plan P0): métricas puras, sin FFmpeg."""

import json

import pytest

from app.pipeline.benchmark import GroundTruth, GroundTruthEvent, evaluate, load_ground_truth
from app.providers.base import ClipCandidate


def _cand(start: float, end: float, score: float) -> ClipCandidate:
    return ClipCandidate(start=start, end=end, score=score)


def test_load_ground_truth_requires_annotation_status(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps({"video": "x", "ground_truth": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="annotation_status"):
        load_ground_truth(path)


def test_load_ground_truth_parses_events(tmp_path):
    path = tmp_path / "gt.json"
    path.write_text(
        json.dumps(
            {
                "video": "x.mp4",
                "mode": "gaming",
                "output_type": "hook",
                "ground_truth": [{"start": 37.0, "peak": 41.2, "end": 46.0, "type": "hook"}],
                "annotation_status": "human_confirmed",
            }
        ),
        encoding="utf-8",
    )
    gt = load_ground_truth(path)
    assert gt.events == [GroundTruthEvent(start=37.0, end=46.0, peak=41.2, type="hook")]
    assert gt.annotation_status == "human_confirmed"


def _gt(*events: GroundTruthEvent) -> GroundTruth:
    return GroundTruth(
        video="x",
        mode="gaming",
        output_type="hook",
        events=list(events),
        annotation_status="human_confirmed",
    )


def test_candidate_recall_is_zero_when_event_never_enters_pool():
    """Si el evento correcto nunca entra al pool, ningún ranker puede recuperarlo."""
    candidates = [_cand(0, 10, 0.9), _cand(100, 110, 0.8)]
    ground_truth = _gt(GroundTruthEvent(start=500, end=510))  # muy lejos de cualquier candidato

    report = evaluate(candidates, ground_truth)

    assert report.candidate_recall == 0.0
    assert report.top_k_recall(5) == 0.0


def test_candidate_recall_and_rank_when_event_present_but_low_ranked():
    """El evento correcto está en el pool, pero en la posición #3 por score."""
    correct = _cand(40.0, 46.0, 0.50)  # score bajo -> queda 3ro
    candidates = [_cand(0, 10, 0.95), _cand(100, 110, 0.90), correct]
    ground_truth = _gt(GroundTruthEvent(start=41.0, end=46.0, peak=43.0))

    report = evaluate(candidates, ground_truth)

    assert report.candidate_recall == 1.0
    assert report.top_k_recall(1) == 0.0  # no es el top-1
    assert report.top_k_recall(3) == 1.0  # sí entra en top-3
    match = report.matches[0]
    assert match.rank == 3
    assert match.overlap_ratio > 0.0


def test_top1_recall_when_event_is_best_ranked():
    correct = _cand(40.0, 46.0, 0.95)
    candidates = [correct, _cand(100, 110, 0.50)]
    ground_truth = _gt(GroundTruthEvent(start=41.0, end=46.0))

    report = evaluate(candidates, ground_truth)

    assert report.top_k_recall(1) == 1.0
    assert report.matches[0].rank == 1


def test_boundary_error_measures_start_end_offset():
    candidate = _cand(35.0, 50.0, 0.9)  # ground truth real: 37-46
    ground_truth = _gt(GroundTruthEvent(start=37.0, end=46.0, peak=41.2))

    report = evaluate([candidate], ground_truth)

    match = report.matches[0]
    assert match.start_error == pytest.approx(2.0)
    assert match.end_error == pytest.approx(4.0)


def test_false_positive_count_excludes_matched_candidates():
    matched = _cand(40.0, 46.0, 0.9)
    unrelated = _cand(200.0, 210.0, 0.3)
    ground_truth = _gt(GroundTruthEvent(start=41.0, end=46.0))

    report = evaluate([matched, unrelated], ground_truth)

    assert report.false_positive_count == 1
    assert report.precision == pytest.approx(0.5)


def test_multiple_ground_truth_events_evaluated_independently():
    """Highlights puede tener varios ground truth (uno por beat)."""
    hit_a = _cand(10.0, 20.0, 0.9)
    hit_b = _cand(100.0, 110.0, 0.8)
    candidates = [hit_a, hit_b, _cand(500, 510, 0.1)]
    ground_truth = _gt(
        GroundTruthEvent(start=11.0, end=19.0, type="highlight"),
        GroundTruthEvent(start=101.0, end=109.0, type="highlight"),
        GroundTruthEvent(start=999.0, end=1005.0, type="highlight"),  # sin candidato
    )

    report = evaluate(candidates, ground_truth)

    assert report.candidate_recall == pytest.approx(2 / 3)
    assert report.summary()["candidate_count"] == 3
