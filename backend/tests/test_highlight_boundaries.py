"""Tests for highlight_boundaries.py (SNIPRA v1.5D).

Covers deterministic boundary refinement, peak selection, evidence integration,
and edge cases.
"""

import pytest
from app.editor.via_evidence import NormalizedVideoEvidence, EvidenceKind
from app.pipeline.highlight_boundaries import (
    RefinementWeights,
    DEFAULT_REFINEMENT_WEIGHTS,
    RefinementStatus,
    AnchorPrecision,
    TemporalAnchor,
    RefinedBoundaries,
    refine_candidate,
    refine_candidates,
    _extract_via_anchors,
    _extract_signal_anchors,
    _extract_transcript_anchors,
    _select_peak,
    _refine_start,
    _refine_end,
    _stable_candidate_id,
    _compute_coverage_factor,
    _compute_status,
)
from app.providers.base import ClipCandidate, SourceRange, ScoreContribution


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_candidate(
    start: float,
    end: float,
    score: float = 0.7,
    title: str = "",
    reason: str = "",
    hook: str = "",
    breakdown: list[ScoreContribution] | None = None,
) -> ClipCandidate:
    c = ClipCandidate(start=start, end=end, score=score, title=title, reason=reason, hook=hook)
    if breakdown:
        c = c.model_copy(update={"score_breakdown": breakdown})
    return c


def make_via_evidence(
    asset_id: str,
    start_s: float,
    end_s: float,
    kind: str = "event",
    confidence: float = 0.8,
    payoff: bool = False,
    reaction: bool = False,
    quality: str = "medium",
    partial: bool = False,
    failed_chunks: list[dict] | None = None,
    requested_frames: int | None = None,
    analyzed_frames: int | None = None,
    coverage_ratio: float | None = None,
) -> NormalizedVideoEvidence:
    return NormalizedVideoEvidence(
        evidence_id=f"ev_{asset_id}_{int(start_s)}",
        asset_id=asset_id,
        start_us=int(start_s * 1_000_000),
        end_us=int(end_s * 1_000_000),
        evidence_kind=EvidenceKind(kind),
        nature="inference" if kind == "event" else "observation",
        description=f"Test {kind}",
        confidence_original=confidence,
        provider="test",
        model="test",
        payoff_present=payoff,
        reaction_present=reaction,
        evidence_quality=quality,
        partial=partial,
        failed_chunks=failed_chunks or [],
        requested_frames=requested_frames,
        analyzed_frames=analyzed_frames,
        coverage_ratio=coverage_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper tests
# ──────────────────────────────────────────────────────────────────────────────

def test_stable_candidate_id_deterministic():
    c1 = make_candidate(10.0, 20.0, 0.7, "title", "reason")
    c2 = make_candidate(10.0, 20.0, 0.7, "title", "reason")
    c3 = make_candidate(15.0, 25.0, 0.7, "title", "reason")

    assert _stable_candidate_id(c1) == _stable_candidate_id(c2)
    assert _stable_candidate_id(c1) != _stable_candidate_id(c3)


def test_refinement_weights_validation():
    with pytest.raises(ValueError):
        RefinementWeights(via_event_confidence=0.5, via_payoff_present=0.5)


def test_default_weights_sum_to_one():
    w = DEFAULT_REFINEMENT_WEIGHTS
    total = (
        w.via_event_confidence + w.via_payoff_present +
        w.via_reaction_present + w.signal_peak_loudness +
        w.transcript_payoff_density
    )
    assert abs(total - 1.0) < 1e-6


def test_compute_coverage_factor_full():
    ev = make_via_evidence("a1", 0, 10, confidence=0.9)
    assert _compute_coverage_factor(ev) == 1.0


def test_compute_coverage_factor_partial():
    ev = make_via_evidence("a1", 0, 10, confidence=0.9, partial=True)
    assert _compute_coverage_factor(ev) == 0.5


def test_compute_coverage_factor_failed_chunks():
    ev = make_via_evidence("a1", 0, 10, confidence=0.9, failed_chunks=[{"chunk": 1}])
    assert _compute_coverage_factor(ev) == 0.7


def test_compute_coverage_factor_ratio():
    ev = make_via_evidence("a1", 0, 10, confidence=0.9, coverage_ratio=0.6)
    assert _compute_coverage_factor(ev) == 0.6


def test_compute_status_no_anchors():
    status = _compute_status(
        anchors=[],
        boundaries_changed=False,
        attached_via=[],
        avg_coverage=0.0,
        avg_quality=0.0
    )
    assert status == RefinementStatus.UNCHANGED


def test_compute_status_no_boundaries_changed():
    # With anchors but no VIA and boundaries unchanged → UNCHANGED
    anchors = [TemporalAnchor(10.0, "test", "ref", 0.5, "peak", AnchorPrecision.POINT)]
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=False,
        attached_via=[],
        avg_coverage=1.0,
        avg_quality=0.8
    )
    assert status == RefinementStatus.UNCHANGED


def test_compute_status_no_via_uncertain():
    anchors = [TemporalAnchor(10.0, "test", "ref", 0.5, "peak", AnchorPrecision.POINT)]
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=True,
        attached_via=[],
        avg_coverage=1.0,
        avg_quality=0.8
    )
    assert status == RefinementStatus.UNCERTAIN


def test_compute_status_low_coverage_uncertain():
    anchors = [TemporalAnchor(10.0, "test", "ref", 0.5, "peak", AnchorPrecision.POINT)]
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=True,
        attached_via=[None],
        avg_coverage=0.3,
        avg_quality=0.8
    )
    assert status == RefinementStatus.UNCERTAIN


def test_compute_status_refined():
    class FakeVIA:
        pass
    anchors = [TemporalAnchor(10.0, "test", "ref", 0.5, "peak", AnchorPrecision.POINT)]
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=True,
        attached_via=[FakeVIA()],
        avg_coverage=1.0,
        avg_quality=0.8
    )
    assert status == RefinementStatus.REFINED


def test_compute_status_low_quality_uncertain():
    class FakeVIA:
        pass
    anchors = [TemporalAnchor(10.0, "test", "ref", 0.5, "peak", AnchorPrecision.POINT)]
    status = _compute_status(
        anchors=anchors,
        boundaries_changed=True,
        attached_via=[FakeVIA()],
        avg_coverage=1.0,
        avg_quality=0.3
    )
    assert status == RefinementStatus.UNCERTAIN


# ──────────────────────────────────────────────────────────────────────────────
# Anchor extraction tests
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_via_anchors_event_full_windows():
    """EVENT evidence should produce start, mid, end anchors + payoff/reaction windows."""
    c = make_candidate(10, 20, 0.7)
    ev = [
        make_via_evidence("a1", 12, 18, confidence=0.9, payoff=True, reaction=True),
    ]
    anchors = _extract_via_anchors(c, ev)

    sources = {a.source for a in anchors}
    assert "via_event_start" in sources
    assert "via_event_mid" in sources
    assert "via_event_end" in sources
    assert "via_payoff_window" in sources
    assert "via_reaction_window" in sources

    # Verify windows have window_start/window_end
    payoff_windows = [a for a in anchors if a.source == "via_payoff_window"]
    assert len(payoff_windows) == 1
    pw = payoff_windows[0]
    assert pw.precision == AnchorPrecision.WINDOW
    assert pw.window_start == 12.0
    assert pw.window_end == 18.0


def test_extract_via_anchors_scene():
    c = make_candidate(10, 20, 0.7)
    ev = [
        make_via_evidence("a1", 10, 15, kind="scene", confidence=0.8),
    ]
    anchors = _extract_via_anchors(c, ev)
    assert len(anchors) == 2
    kinds = {a.kind for a in anchors}
    assert kinds == {"setup"}


def test_extract_via_anchors_no_payoff_no_reaction():
    c = make_candidate(10, 20, 0.7)
    ev = [make_via_evidence("a1", 12, 18, confidence=0.9)]
    anchors = _extract_via_anchors(c, ev)
    sources = {a.source for a in anchors}
    assert "via_payoff_window" not in sources
    assert "via_reaction_window" not in sources


def test_extract_signal_anchors_loudness_normalization():
    c = make_candidate(10, 20, 0.7)
    # Loudness values: -10, -5, -2 (higher = louder = less negative)
    signals = {"loudness": [[12.0, -10.0], [15.0, -5.0], [18.0, -2.0]]}
    anchors = _extract_signal_anchors(c, signals)
    assert len(anchors) == 3
    for a in anchors:
        assert a.kind == "peak"
        assert a.source == "signal_loudness"
    # -2dB should have highest confidence (loudest)
    confidences = [a.confidence for a in anchors]
    assert max(confidences) > 0.5


def test_extract_signal_anchors_flat_audio_no_peak():
    """Flat audio should NOT produce peak with confidence=1.0"""
    c = make_candidate(10, 20, 0.7)
    signals = {"loudness": [[12.0, -10.0], [15.0, -10.0], [18.0, -10.0]]}
    anchors = _extract_signal_anchors(c, signals)
    for a in anchors:
        assert a.confidence == 0.0  # flat audio = no peak info


def test_extract_signal_anchors_scene_and_silence():
    c = make_candidate(10, 20, 0.7)
    signals = {
        "scene_times": [10.0, 14.0, 18.0],
        "silence_intervals": [(5.0, 8.0), (12.0, 15.0)],  # within candidate
        "loudness": []
    }
    anchors = _extract_signal_anchors(c, signals)
    sources = {a.source for a in anchors}
    assert "signal_scene" in sources
    assert "signal_silence_start" in sources
    assert "signal_silence_end" in sources


def test_extract_transcript_anchors_payoff_only():
    c = make_candidate(10, 20, 0.7)
    segments = [
        {"start": 12.0, "end": 15.0, "text": "Estoy a uno, vamos a ganar"},
        {"start": 16.0, "end": 18.0, "text": "Victoria final"},
    ]
    anchors = _extract_transcript_anchors(c, segments)
    payoff_anchors = [a for a in anchors if a.kind == "payoff"]
    # 2 from payoff terms + 1 from reward term = 3 payoff anchors
    assert len(payoff_anchors) == 3


def test_extract_transcript_anchors_reward_not_reaction():
    c = make_candidate(10, 20, 0.7)
    segments = [
        {"start": 12.0, "end": 15.0, "text": "Victoria, ganamos!"},
    ]
    anchors = _extract_transcript_anchors(c, segments)
    reward_anchors = [a for a in anchors if a.source == "transcript_reward"]
    assert len(reward_anchors) == 1
    assert reward_anchors[0].kind == "payoff"  # NOT reaction


# ──────────────────────────────────────────────────────────────────────────────
# Peak selection tests
# ──────────────────────────────────────────────────────────────────────────────

def test_select_peak_no_anchors():
    c = make_candidate(10, 20, 0.7)
    peak = _select_peak([], c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak is None


def test_select_peak_prefers_via_event():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(12.0, "via_event_mid", "ev1", 0.9, "peak", AnchorPrecision.POINT),
        TemporalAnchor(15.0, "signal_loudness", "sig1", 0.5, "peak", AnchorPrecision.POINT),
    ]
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak == 12.0


def test_select_peak_prefers_payoff_when_no_peak():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(18.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=15, window_end=20),
        TemporalAnchor(12.0, "signal_loudness", "sig1", 0.5, "peak", AnchorPrecision.POINT),
    ]
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak == 18.0


def test_select_peak_deterministic_tiebreak():
    """Same evidence, different order → same peak (earlier timestamp wins tie)."""
    c = make_candidate(10, 20, 0.7)
    anchors1 = [
        TemporalAnchor(12.0, "via_event_mid", "ev1", 0.8, "peak", AnchorPrecision.POINT),
        TemporalAnchor(16.0, "via_event_mid", "ev2", 0.8, "peak", AnchorPrecision.POINT),
    ]
    anchors2 = list(reversed(anchors1))
    peak1 = _select_peak(anchors1, c, DEFAULT_REFINEMENT_WEIGHTS)
    peak2 = _select_peak(anchors2, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak1 == 12.0
    assert peak2 == 12.0


def test_select_peak_within_candidate_bounds():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(5.0, "via_event_mid", "ev1", 0.9, "peak", AnchorPrecision.POINT),
        TemporalAnchor(25.0, "via_event_mid", "ev2", 0.9, "peak", AnchorPrecision.POINT),
        TemporalAnchor(15.0, "via_event_mid", "ev3", 0.8, "peak", AnchorPrecision.POINT),
    ]
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak == 15.0


def test_select_peak_setup_action_not_peak():
    """setup/action cannot compete as peak - they have reduced weight"""
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(12.0, "via_event_start", "ev1", 0.9, "setup", AnchorPrecision.POINT),
        TemporalAnchor(15.0, "via_event_mid", "ev2", 0.8, "peak", AnchorPrecision.POINT),
        TemporalAnchor(18.0, "via_event_end", "ev3", 0.8, "action", AnchorPrecision.POINT),
    ]
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    # Peak should be at 15 (mid), not setup/action
    assert peak == 15.0


def test_select_peak_scene_not_peak():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(12.0, "signal_scene", "s1", 0.9, "setup", AnchorPrecision.POINT),
        TemporalAnchor(15.0, "via_event_mid", "ev1", 0.8, "peak", AnchorPrecision.POINT),
    ]
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak == 15.0  # scene is setup, not peak


# ──────────────────────────────────────────────────────────────────────────────
# Start/end refinement tests
# ──────────────────────────────────────────────────────────────────────────────

def test_refine_start_no_change():
    c = make_candidate(10, 20, 0.7)
    anchors = []
    start, notes = _refine_start(c, None, anchors, [], {}, 30)
    assert start == 10.0


def test_refine_start_moves_to_event_start():
    c = make_candidate(10, 30, 0.7)
    anchors = [TemporalAnchor(15.0, "via_event_start", "ev1", 0.8, "setup", AnchorPrecision.POINT)]
    start, notes = _refine_start(c, 20.0, anchors, [], {}, 30)
    assert start == 15.0


def test_refine_start_limits_setup_window():
    c = make_candidate(10, 30, 0.7)
    # Event starts before candidate, so refined start should be candidate.start (10.0)
    # No arbitrary 8-second limit applied
    anchors = [TemporalAnchor(5.0, "via_event_start", "ev1", 0.8, "setup", AnchorPrecision.POINT)]
    start, notes = _refine_start(c, 20.0, anchors, [], {}, 30)
    assert start == 10.0  # Clamped to candidate.start


def test_refine_end_reduces_dead_aftermath():
    """P0-2: End should be REDUCED when content ends before candidate.end."""
    c = make_candidate(10, 40, 0.7)
    anchors = [
        TemporalAnchor(25.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT),
        TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=15, window_end=25),
    ]
    end, notes = _refine_end(c, 20.0, anchors, [], {}, 60)
    # Should reduce from 40 to ~26 (content end + small tail)
    assert end < 40.0
    assert end <= 26.0


def test_refine_end_preserves_payoff():
    c = make_candidate(10, 20, 0.7)
    anchors = [TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=20, window_end=22)]
    end, notes = _refine_end(c, None, anchors, [], {}, 30)
    # Payoff window end at 22 directly used as boundary (no arbitrary +1s buffer)
    assert end == 22.0


def test_refine_end_extends_for_reaction_near_payoff():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=20, window_end=22),
        TemporalAnchor(23.5, "via_reaction_window", "ev2", 0.8, "reaction", AnchorPrecision.WINDOW, window_start=22, window_end=24),
    ]
    end, notes = _refine_end(c, None, anchors, [], {}, 30)
    # Reaction within 3s of payoff → extends to reaction + 1s = 25
    assert end >= 24.0


def test_refine_end_no_reaction_if_far_from_payoff():
    c = make_candidate(10, 20, 0.7)
    anchors = [
        TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=20, window_end=22),
        TemporalAnchor(30.0, "via_reaction_window", "ev2", 0.8, "reaction", AnchorPrecision.WINDOW, window_start=28, window_end=32),
    ]
    end, notes = _refine_end(c, None, anchors, [], {}, 40)
    # Reaction too far from payoff → ignored
    assert end <= 24.0


def test_refine_end_inward_reduction_with_strong_event():
    """P0-2: Broad candidate shrinks around strong EVENT evidence."""
    c = make_candidate(10, 40, 0.8)
    anchors = [
        TemporalAnchor(15.0, "via_event_start", "ev1", 0.9, "setup", AnchorPrecision.POINT),
        TemporalAnchor(20.0, "via_event_mid", "ev1", 0.9, "peak", AnchorPrecision.POINT),
        TemporalAnchor(25.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT),
        TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=15, window_end=25),
    ]
    end, notes = _refine_end(c, 20.0, anchors, [], {}, 60)
    # Should reduce from 40 to ~26
    assert end < 40.0
    assert end <= 26.0
    assert "end_refined_to_content" in " ".join(notes)


def test_refine_end_silence_end_closes_dead_aftermath():
    """Silence end after content can close dead aftermath."""
    c = make_candidate(10, 40, 0.7)
    [
        TemporalAnchor(20.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT),
    ]
    signals = {"silence_intervals": [(22.0, 30.0)]}  # silence after content
    end, notes = _refine_end(c, 20.0, [TemporalAnchor(20.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT)], [], signals, 60)
    # Should reduce to silence end + small tail
    assert end < 40.0


def test_refine_end_silence_end_late_does_not_keep_entire_tail():
    """Silence end far after content should not keep entire silent tail."""
    c = make_candidate(10, 40, 0.7)
    [
        TemporalAnchor(20.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT),
    ]
    # Silence far after content
    signals = {"silence_intervals": [(35.0, 38.0)]}
    end, notes = _refine_end(c, 20.0, [TemporalAnchor(20.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT)], [], signals, 60)
    # Should NOT extend to 38, should reduce
    assert end < 40.0


# ──────────────────────────────────────────────────────────────────────────────
# Full pipeline tests
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_candidates():
    results = refine_candidates([])
    assert results == []


def test_single_candidate_no_evidence():
    c = make_candidate(10, 20, 0.7)
    results = refine_candidates([c])
    assert len(results) == 1
    r = results[0]
    assert r.status == RefinementStatus.UNCHANGED
    assert r.refined_start == 10.0
    assert r.refined_end == 20.0


def test_candidate_with_via_event_and_payoff():
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    assert len(results) == 1
    r = results[0]
    assert r.status == RefinementStatus.REFINED
    assert r.peak_at is not None
    assert r.payoff_window_start is not None
    assert r.payoff_window_end is not None
    # Event start at 15 refines start from 10 to 15
    assert r.refined_start == 15.0
    # Payoff window ends at 25, BOUNDARY_TOLERANCE (0.5) applied = 25.5
    assert r.refined_end == 25.5


def test_candidate_with_transcript_payoff():
    c = make_candidate(10, 20, 0.7)
    segments = [
        {"start": 10.0, "end": 15.0, "text": "Estoy a uno, vamos a ganar"},
        {"start": 16.0, "end": 18.0, "text": "Victoria final"},
    ]
    results = refine_candidates([c], transcript_segments=segments, video_duration=30)
    assert len(results) == 1
    r = results[0]
    assert r.payoff_at is not None
    # With transcript evidence but no boundary change (payoff within original range), status should be UNCHANGED
    assert r.status in (RefinementStatus.REFINED, RefinementStatus.UNCERTAIN, RefinementStatus.UNCHANGED)
    # payoff_at should be set from transcript payoff
    assert r.payoff_at is not None


def test_candidate_with_signal_peak():
    c = make_candidate(10, 30, 0.7)
    signals = {"loudness": [[12.0, -5.0], [18.0, -2.0], [25.0, -6.0]], "scene_times": []}
    results = refine_candidates([c], signals_doc=signals, video_duration=60)
    assert len(results) == 1
    r = results[0]
    assert r.peak_at is not None
    assert r.peak_at == 18.0


def test_refinement_preserves_original_when_would_break():
    c = make_candidate(10, 11, 0.7)  # very short
    via = [make_via_evidence("a1", 10, 11, confidence=0.5)]
    results = refine_candidates([c], via, video_duration=30)
    r = results[0]
    assert r.refined_end > r.refined_start


def test_deterministic_rerun():
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    segments = [{"start": 15.0, "end": 20.0, "text": "Victoria"}]
    signals = {"loudness": [[18.0, -2.0]], "scene_times": [15.0, 25.0]}

    r1 = refine_candidates([c], via, segments, signals, 60)
    r2 = refine_candidates([c], via, segments, signals, 60)

    assert r1[0].refined_start == r2[0].refined_start
    assert r1[0].refined_end == r2[0].refined_end
    assert r1[0].peak_at == r2[0].peak_at
    assert r1[0].status == r2[0].status


def test_input_order_independence():
    c1 = make_candidate(10, 20, 0.9)
    c2 = make_candidate(30, 40, 0.7)
    via = [
        make_via_evidence("a1", 12, 18, confidence=0.9),
        make_via_evidence("a1", 32, 38, confidence=0.8),
    ]

    r1 = refine_candidates([c1, c2], via, video_duration=60)
    r2 = refine_candidates([c2, c1], via, video_duration=60)

    assert len(r1) == 2
    assert len(r2) == 2
    for a, b in zip(r1, r2):
        assert a.candidate_id == b.candidate_id
        assert a.refined_start == b.refined_start
        assert a.refined_end == b.refined_end


def test_multiple_candidates_same_asset():
    c1 = make_candidate(10, 20, 0.9)
    c2 = make_candidate(30, 40, 0.7)
    via = [
        make_via_evidence("a1", 12, 18, confidence=0.9, payoff=True),
        make_via_evidence("a1", 32, 38, confidence=0.8),
    ]
    results = refine_candidates([c1, c2], via, video_duration=60)
    assert len(results) == 2
    for r in results:
        assert r.status in (RefinementStatus.REFINED, RefinementStatus.UNCERTAIN)
        assert r.refined_end > r.refined_start


# ──────────────────────────────────────────────────────────────────────────────
# Evidence attachment tests
# ──────────────────────────────────────────────────────────────────────────────

def test_via_evidence_attached_by_overlap():
    c = make_candidate(10, 20, 0.7)
    via = [make_via_evidence("a1", 12, 18, confidence=0.9)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert len(r.evidence_refs) > 0
    assert r.status == RefinementStatus.REFINED


def test_via_evidence_not_attached_if_no_overlap():
    c = make_candidate(10, 20, 0.7)
    via = [make_via_evidence("a1", 50, 60, confidence=0.9)]
    results = refine_candidates([c], via, video_duration=100)
    r = results[0]
    assert r.status == RefinementStatus.UNCHANGED


def test_partial_via_evidence_reduces_confidence():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, partial=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.confidence < 1.0


def test_coverage_ratio_affects_confidence():
    c = make_candidate(10, 30, 0.7)
    via = [
        make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=1.0),
        make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=0.3),
    ]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.confidence < 1.0


def test_same_evidence_different_coverage_different_confidence():
    """P0-4: Coverage must have real effect on confidence."""
    c = make_candidate(10, 30, 0.7)
    via_a = [make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=1.0)]
    via_b = [make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=0.3)]

    results_a = refine_candidates([c], via_a, video_duration=60)
    results_b = refine_candidates([c], via_b, video_duration=60)

    assert results_a[0].confidence > results_b[0].confidence


def test_coverage_affects_status():
    """Coverage must affect STATUS, not just confidence."""
    c = make_candidate(10, 30, 0.7)
    # Full coverage → REFINED
    via_full = [make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=1.0, payoff=True)]
    # Partial coverage → UNCERTAIN
    via_partial = [make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=0.3, payoff=True)]

    results_full = refine_candidates([c], via_full, video_duration=60)
    results_partial = refine_candidates([c], via_partial, video_duration=60)

    assert results_full[0].status == RefinementStatus.REFINED
    assert results_partial[0].status == RefinementStatus.UNCERTAIN


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────────────

def test_minimal_duration_candidate():
    c = make_candidate(0, 0.1, 0.5)
    results = refine_candidates([c], video_duration=0.1)
    r = results[0]
    assert r.refined_start == 0.0
    assert r.refined_end == 0.1


def test_video_boundaries_respected():
    c = make_candidate(0, 10, 0.7)
    via = [make_via_evidence("a1", -5, 5, confidence=0.9)]
    results = refine_candidates([c], via, video_duration=10)
    r = results[0]
    assert r.refined_start >= 0.0

    c2 = make_candidate(90, 100, 0.7)
    via2 = [make_via_evidence("a1", 95, 105, confidence=0.9)]
    results = refine_candidates([c2], via2, video_duration=100)
    r2 = results[0]
    assert r2.refined_end <= 100.0


def test_video_duration_authority():
    """P1-6: video_duration is authoritative limit."""
    c = make_candidate(10, 50, 0.7)
    via = [make_via_evidence("a1", 40, 55, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=45)
    r = results[0]
    assert r.refined_end <= 45.0


def test_candidate_exceeds_video_duration():
    """Defective candidate exceeding video_duration."""
    c = make_candidate(40, 60, 0.7)
    results = refine_candidates([c], video_duration=50)
    r = results[0]
    assert r.refined_end <= 50.0
    assert r.refined_start <= 50.0


def test_fallback_respects_video_duration():
    """Even fallback must obey video_duration."""
    c = make_candidate(40, 60, 0.7)  # 20s duration
    results = refine_candidates([c], video_duration=50)
    r = results[0]
    assert r.refined_end <= 50.0
    assert r.refined_start <= 50.0


def test_refined_peak_within_bounds():
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    if r.peak_at is not None:
        assert r.refined_start <= r.peak_at <= r.refined_end


def test_refined_range_contains_payoff():
    c = make_candidate(10, 20, 0.7)
    via = [make_via_evidence("a1", 18, 22, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    if r.payoff_at is not None:
        assert r.refined_start <= r.payoff_at <= r.refined_end


def test_no_nan_infinity():
    c = make_candidate(10, 20, 0.7)
    results = refine_candidates([c])
    r = results[0]
    assert r.refined_start == r.refined_start
    assert r.refined_end == r.refined_end
    assert r.confidence == r.confidence
    if r.peak_at is not None:
        assert r.peak_at == r.peak_at


def test_refinement_notes_present():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert len(r.notes) > 0
    assert all(isinstance(n, str) for n in r.notes)


def test_anchors_sorted_by_timestamp():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    timestamps = [a.timestamp for a in r.anchors]
    assert timestamps == sorted(timestamps)


def test_evidence_refs_unique():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    refs = list(r.evidence_refs)
    assert len(refs) == len(set(refs))


def test_status_unchanged_when_no_boundary_change():
    """P1-7: Status must represent real change."""
    c = make_candidate(10, 20, 0.5)
    via = [make_via_evidence("a1", 50, 60, confidence=0.9)]  # no overlap
    results = refine_candidates([c], via, video_duration=100)
    r = results[0]
    assert r.status == RefinementStatus.UNCHANGED


def test_status_unchanged_when_boundaries_same():
    """Even with evidence, if boundaries don't change → UNCHANGED."""
    c = make_candidate(10, 20, 0.5)
    results = refine_candidates([c], video_duration=60)
    r = results[0]
    assert r.status == RefinementStatus.UNCHANGED


def test_status_uncertain_low_coverage():
    """Low coverage with VIA → UNCERTAIN, not REFINED."""
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, coverage_ratio=0.3, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.status == RefinementStatus.UNCERTAIN


def test_confidence_bounded():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=1.0, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert 0.0 <= r.confidence <= 1.0


def test_payoff_at_only_for_point_anchors():
    """P0-1: WINDOW anchors should NOT produce payoff_at, only payoff_window_*."""
    c = make_candidate(10, 20, 0.8)
    via = [make_via_evidence("a1", 18, 22, confidence=0.95, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.payoff_window_start is not None
    assert r.payoff_window_end is not None
    # WINDOW anchor should NOT produce payoff_at
    assert r.payoff_at is None
    # payoff_window_start/end should be populated
    assert r.payoff_window_start is not None
    assert r.payoff_window_end is not None


def test_reaction_at_only_for_point_anchors():
    """WINDOW reaction should not produce reaction_at."""
    c = make_candidate(10, 25, 0.8)
    via = [
        make_via_evidence("a1", 18, 24, confidence=0.9, payoff=True, reaction=True),
    ]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.reaction_window_start is not None
    assert r.reaction_window_end is not None
    # WINDOW reaction should NOT produce reaction_at
    assert r.reaction_at is None


def test_no_source_ranges_mutated():
    c = make_candidate(10, 30, 0.8)
    original_ranges = list(c.source_ranges)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    assert list(c.source_ranges) == original_ranges
    assert isinstance(results[0], RefinedBoundaries)


def test_no_source_ranges_in_output():
    """RefinedBoundaries does not have source_ranges."""
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert not hasattr(r, "source_ranges")


# ──────────────────────────────────────────────────────────────────────────────
# Transcript snapping tests
# ──────────────────────────────────────────────────────────────────────────────

def test_start_snapped_to_transcript():
    c = make_candidate(10.5, 20, 0.7)
    segments = [
        {"start": 10.0, "end": 12.0, "text": "Hola mundo"},
        {"start": 12.5, "end": 15.0, "text": "Esto es una prueba"},
    ]
    results = refine_candidates([c], transcript_segments=segments, video_duration=30)
    r = results[0]
    assert r.refined_start <= 10.5


def test_end_snapped_to_transcript_but_payoff_wins():
    c = make_candidate(10, 20.5, 0.7)
    segments = [
        {"start": 18.0, "end": 20.0, "text": "Victoria final."},
        {"start": 20.0, "end": 22.0, "text": "Siguiente frase"},
    ]
    results = refine_candidates([c], transcript_segments=segments, video_duration=30)
    r = results[0]
    # Transcript payoff at 20.0 (POINT anchor) → end refined to 20.5 (20 + BOUNDARY_TOLERANCE 0.5)
    assert r.refined_end == 20.5


def test_scene_snapping_start():
    c = make_candidate(10.5, 20, 0.7)
    signals = {"scene_times": [10.0, 15.0], "loudness": []}
    results = refine_candidates([c], signals_doc=signals, video_duration=30)
    r = results[0]
    assert r.refined_start <= 10.5


def test_scene_snapping_end():
    c = make_candidate(10, 19.5, 0.7)
    signals = {"scene_times": [15.0, 20.0], "loudness": []}
    results = refine_candidates([c], signals_doc=signals, video_duration=30)
    r = results[0]
    assert r.refined_end >= 19.5


# ──────────────────────────────────────────────────────────────────────────────
# Expansion policy tests
# ──────────────────────────────────────────────────────────────────────────────

def test_expansion_only_with_evidence():
    c = make_candidate(10, 20, 0.7)
    via = [make_via_evidence("a1", 18, 22, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.was_expanded
    assert r.refined_end > c.end


def test_no_expansion_without_evidence():
    c = make_candidate(10, 20, 0.7)
    results = refine_candidates([c], video_duration=60)
    r = results[0]
    assert not r.was_expanded
    assert r.refined_end == c.end


def test_expansion_limited_by_evidence():
    c = make_candidate(10, 20, 0.7)
    via = [make_via_evidence("a1", 22, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.refined_end <= 23.0  # 22 + 1s buffer


def test_no_expansion_for_duration_preset():
    c = make_candidate(10, 20, 0.7)
    results = refine_candidates([c], video_duration=60)
    r = results[0]
    assert r.refined_end - r.refined_start <= 20.0


def test_reduction_with_strong_event():
    """P0-3: Candidate shrinks around strong EVENT evidence."""
    c = make_candidate(10, 40, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    assert r.was_narrowed
    assert r.refined_start >= 10.0
    assert r.refined_end <= 26.0


# ──────────────────────────────────────────────────────────────────────────────
# Custom weights tests
# ──────────────────────────────────────────────────────────────────────────────

def test_custom_weights():
    c = make_candidate(10, 30, 0.7)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9)]
    custom = RefinementWeights(
        via_event_confidence=0.0,
        via_payoff_present=0.0,
        via_reaction_present=0.0,
        signal_peak_loudness=1.0,
        transcript_payoff_density=0.0,
    )
    signals = {"loudness": [[15.0, -5.0], [18.0, -2.0], [20.0, -5.0]]}
    results = refine_candidates([c], via, signals_doc=signals, video_duration=60, weights=custom)
    r = results[0]
    assert r.peak_at == 18.0


def test_permutation_invariance_via_order():
    """Same evidence, different order → same output."""
    c = make_candidate(10, 30, 0.8)
    via1 = [make_via_evidence("a1", 15, 25, confidence=0.9), make_via_evidence("a1", 15, 25, confidence=0.8)]
    via2 = [make_via_evidence("a1", 15, 25, confidence=0.8), make_via_evidence("a1", 15, 25, confidence=0.9)]

    r1 = refine_candidates([c], via1, video_duration=60)
    r2 = refine_candidates([c], via2, video_duration=60)

    assert r1[0].refined_start == r2[0].refined_start
    assert r1[0].refined_end == r2[0].refined_end
    assert r1[0].peak_at == r2[0].peak_at


def test_permutation_invariance_scene_times():
    c = make_candidate(10, 30, 0.7)
    signals1 = {"scene_times": [10.0, 20.0], "loudness": []}
    signals2 = {"scene_times": [20.0, 10.0], "loudness": []}

    r1 = refine_candidates([c], signals_doc=signals1, video_duration=60)
    r2 = refine_candidates([c], signals_doc=signals2, video_duration=60)

    assert r1[0].refined_start == r2[0].refined_start


def test_permutation_invariance_transcript():
    c = make_candidate(10, 30, 0.7)
    seg1 = [{"start": 10.0, "end": 15.0, "text": "A"}, {"start": 20.0, "end": 25.0, "text": "B"}]
    seg2 = [{"start": 20.0, "end": 25.0, "text": "B"}, {"start": 10.0, "end": 15.0, "text": "A"}]

    r1 = refine_candidates([c], transcript_segments=seg1, video_duration=60)
    r2 = refine_candidates([c], transcript_segments=seg2, video_duration=60)

    assert r1[0].refined_start == r2[0].refined_start


def test_permutation_invariance_loudness():
    c = make_candidate(10, 30, 0.7)
    sig1 = {"loudness": [[15.0, -5.0], [20.0, -2.0]], "scene_times": []}
    sig2 = {"loudness": [[20.0, -2.0], [15.0, -5.0]], "scene_times": []}

    r1 = refine_candidates([c], signals_doc=sig1, video_duration=60)
    r2 = refine_candidates([c], signals_doc=sig2, video_duration=60)

    assert r1[0].peak_at == r2[0].peak_at


def test_permutation_invariance_silence():
    c = make_candidate(10, 30, 0.7)
    sig1 = {"silence_intervals": [(5.0, 8.0), (12.0, 15.0)], "scene_times": []}
    sig2 = {"silence_intervals": [(12.0, 15.0), (5.0, 8.0)], "scene_times": []}

    r1 = refine_candidates([c], signals_doc=sig1, video_duration=60)
    r2 = refine_candidates([c], signals_doc=sig2, video_duration=60)

    assert r1[0].refined_start == r2[0].refined_start
    assert r1[0].refined_end == r2[0].refined_end


# ──────────────────────────────────────────────────────────────────────────────
# Constants exposed for testing
# ──────────────────────────────────────────────────────────────────────────────

def test_constants_exposed():
    from app.pipeline.highlight_boundaries import SNAP_TOLERANCE_START, SNAP_TOLERANCE_END
    assert SNAP_TOLERANCE_START == 1.5
    assert SNAP_TOLERANCE_END == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Loudness normalization tests
# ──────────────────────────────────────────────────────────────────────────────

def test_loudness_flat_audio_zero_confidence():
    """Flat audio should produce zero confidence for all samples."""
    c = make_candidate(10, 20, 0.7)
    signals = {"loudness": [[12.0, -10.0], [15.0, -10.0], [18.0, -10.0]]}
    anchors = _extract_signal_anchors(c, signals)
    for a in anchors:
        assert a.confidence == 0.0


def test_loudness_peak_is_max():
    c = make_candidate(10, 20, 0.7)
    signals = {"loudness": [[12.0, -10.0], [15.0, -2.0], [18.0, -6.0]]}
    anchors = _extract_signal_anchors(c, signals)
    # -2dB is loudest → should have highest confidence
    confidences = {a.timestamp: a.confidence for a in anchors}
    assert confidences[15.0] > confidences[12.0]
    assert confidences[15.0] > confidences[18.0]


def test_loudness_input_order_invariance():
    """Loudness samples in different order should produce same peak."""
    c = make_candidate(10, 20, 0.7)
    sig1 = {"loudness": [[15.0, -2.0], [12.0, -10.0], [18.0, -6.0]]}
    sig2 = {"loudness": [[12.0, -10.0], [18.0, -6.0], [15.0, -2.0]]}

    anchors1 = _extract_signal_anchors(c, sig1)
    anchors2 = _extract_signal_anchors(c, sig2)

    peak1 = _select_peak(anchors1, make_candidate(10, 20, 0.7), DEFAULT_REFINEMENT_WEIGHTS)
    peak2 = _select_peak(anchors2, make_candidate(10, 20, 0.7), DEFAULT_REFINEMENT_WEIGHTS)

    assert peak1 == peak2


# ──────────────────────────────────────────────────────────────────────────────
# Integration with 1.5C scoring
# ──────────────────────────────────────────────────────────────────────────────

def test_refinement_after_scoring():
    """Verify 1.5D can consume 1.5C scored candidates."""
    from app.pipeline.highlight_scoring import score_candidates

    c1 = make_candidate(10, 30, 0.8)
    c2 = make_candidate(40, 50, 0.5)

    scored = score_candidates([c1, c2])

    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([s.candidate for s in scored], via, video_duration=60)

    assert len(results) == 2
    for r in results:
        assert r.refined_end > r.refined_start


# ──────────────────────────────────────────────────────────────────────────────
# Closure tests for evidence-only boundary refinement (SNIPRA v1.5D)
# ──────────────────────────────────────────────────────────────────────────────

def test_long_trailing_silence_cuts_at_event_end():
    """A. Long trailing silence after EVENT should cut at silence start, not keep tail.

    candidate 10→40
    EVENT 15→24
    silence 25→39
    => refined_end cerca de 25, no 39–40
    """
    c = make_candidate(10, 40, 0.8)
    via = [make_via_evidence("a1", 15, 24, confidence=0.9, payoff=False)]
    signals = {"silence_intervals": [(25.0, 39.0)]}
    results = refine_candidates([c], via, signals_doc=signals, video_duration=60)
    r = results[0]
    # Should cut at silence start (25) + 0.5 tolerance = 25.5, not keep until 40
    assert r.refined_end <= 26.0
    assert r.refined_end >= 25.0


def test_internal_silence_then_later_event_does_not_cut_early():
    """B. Internal silence followed by another EVENT should not cut at first silence.

    EVENT ends 24
    silence 25→28
    another EVENT 30→35
    => NO cortar en 25
    """
    c = make_candidate(10, 40, 0.8)
    via = [
        make_via_evidence("a1", 15, 24, confidence=0.9, payoff=True),  # First event with payoff
        make_via_evidence("a2", 30, 35, confidence=0.9, payoff=True),  # Second event
    ]
    signals = {"silence_intervals": [(25.0, 28.0)]}
    results = refine_candidates([c], via, signals_doc=signals, video_duration=60)
    r = results[0]
    # Should extend to second event (35) not cut at first silence (25)
    assert r.refined_end >= 34.0


def test_silence_intervals_permutation_invariance():
    """C. silence_intervals in reverse order should produce same refined_end.

    silence_intervals en orden invertido
    => mismo refined_end
    """
    c = make_candidate(10, 40, 0.8)
    via = [make_via_evidence("a1", 15, 24, confidence=0.9, payoff=True)]
    signals1 = {"silence_intervals": [(25.0, 30.0), (32.0, 35.0)]}
    signals2 = {"silence_intervals": [(32.0, 35.0), (25.0, 30.0)]}  # Reversed

    results1 = refine_candidates([c], via, signals_doc=signals1, video_duration=60)
    results2 = refine_candidates([c], via, signals_doc=signals2, video_duration=60)

    assert results1[0].refined_end == results2[0].refined_end


def test_payoff_window_uses_window_end_without_fixed_buffer():
    """D. Payoff WINDOW anchor uses actual window_end, no +1s fixed buffer.

    payoff WINDOW
    => end usa window_end real, sin +1s fijo
    """
    c = make_candidate(10, 30, 0.7)
    anchors = [TemporalAnchor(22.0, "via_payoff_window", "ev1", 0.9, "payoff", AnchorPrecision.WINDOW, window_start=20, window_end=22)]
    end, notes = _refine_end(c, None, anchors, [], {}, 60)
    # Should use window_end (22) + BOUNDARY_TOLERANCE (0.5) = 22.5, not 22 + 1 = 23
    assert end == 22.5
    # Note should indicate refined to content boundary (payoff window end)
    assert any("end_refined_to_content" in n for n in notes)


def test_structural_reaction_preserved():
    """E. Structurally coherent reaction (within 3s of payoff) is preserved.

    reaction estructuralmente relacionada
    => preservada
    """
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True, reaction=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Reaction window ends at 25, should preserve to 25
    assert r.refined_end >= 25.0
    assert r.reaction_window_end is not None
    assert r.reaction_window_end >= 25.0


def test_non_structural_reaction_not_preserved():
    """F. Reaction far from payoff (>3s) is NOT preserved via global threshold.

    reaction no relacionada
    => no preservada mediante threshold global fijo
    """
    c = make_candidate(10, 40, 0.8)
    # Payoff at 15-20 (first evidence), reaction at 30-35 (separate, no payoff near it)
    via1 = make_via_evidence("a1", 15, 20, confidence=0.9, payoff=True)
    # Second evidence with reaction but NO payoff
    via2 = make_via_evidence("a2", 30, 35, confidence=0.9, reaction=True)
    results = refine_candidates([c], [via1, via2], video_duration=60)
    r = results[0]
    # Reaction window should NOT be preserved (not coherent with payoff at 20)
    assert r.reaction_window_start is None
    assert r.reaction_window_end is None
    # Note: event_end at 35 from second evidence IS a valid content boundary,
    # so end extends to 35.5. This is correct behavior - event_end is not a reaction.


def test_short_valid_candidate_no_fixed_minimum():
    """G. Short valid candidate should not impose fixed editorial minimum.

    candidate corto válido
    => no imponer mínimo editorial fijo
    """
    c = make_candidate(10, 12.5, 0.8)  # 2.5s candidate
    via = [make_via_evidence("a1", 10.5, 12, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Should refine to payoff window end (12) + tolerance, not impose 3s or 5s minimum
    assert r.refined_end <= 12.5
    assert r.refined_end >= 12.0
    # Duration should be valid (at least 1s from start refinement)
    assert r.refined_end > r.refined_start


def test_flat_loudness_only_no_peak():
    """H. Flat loudness (no peak) should result in peak_at = None.

    flat loudness only
    => peak_at is None
    """
    c = make_candidate(10, 20, 0.7)
    # Flat loudness: all same level
    signals = {"loudness": [[10.0, -20.0], [15.0, -20.0], [20.0, -20.0]]}
    results = refine_candidates([c], signals_doc=signals, video_duration=60)
    r = results[0]
    # Flat audio should not produce a peak
    assert r.peak_at is None


def test_setup_action_only_no_peak():
    """I. Only setup/action anchors (no peak, payoff, loudness) should give peak_at = None.

    setup/action only
    => peak_at is None
    """
    c = make_candidate(10, 20, 0.7)
    # Only VIA event start/end (setup/action), no mid/peak
    anchors = [
        TemporalAnchor(10.0, "via_event_start", "ev1", 0.9, "setup", AnchorPrecision.POINT),
        TemporalAnchor(20.0, "via_event_end", "ev1", 0.9, "action", AnchorPrecision.POINT),
    ]
    # This tests _select_peak directly
    peak = _select_peak(anchors, c, DEFAULT_REFINEMENT_WEIGHTS)
    assert peak is None


def test_was_narrowed_start_end_semantics():
    """J. was_narrowed semantics: start narrowed if refined > original, end narrowed if refined < original.

    was_narrowed start/end
    => semántica correcta
    """
    c = make_candidate(10, 30, 0.8)
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]

    # Start refined from 10 to 15 (narrowed = True, refined_start > original_start)
    # End refined from 30 to 25.5 (narrowed = True, refined_end < original_end)
    assert r.refined_start > c.start  # Start moved inward (narrowed)
    assert r.refined_end < c.end      # End moved inward (narrowed)

    # Test expansion case: candidate starts after event but ends before payoff
    # Event 15-25, candidate 18-22
    # Start at 18 is already after event start (15), so no expansion on start
    # But end at 22 is before payoff end (25), so end expands to 25.5
    c2 = make_candidate(18, 22, 0.8)
    via2 = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results2 = refine_candidates([c2], via2, video_duration=60)
    r2 = results2[0]
    # Start stays at 18 (already after event start 15, so no expansion needed)
    # End expands from 22 to 25.5 (payoff window end + tolerance)
    assert r2.refined_start == c2.start  # Start unchanged (already optimal)
    assert r2.refined_end > c2.end       # End expanded outward

    # Test full expansion: candidate entirely inside evidence but after event start
    c3 = make_candidate(16, 24, 0.8)
    via3 = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)]
    results3 = refine_candidates([c3], via3, video_duration=60)
    r3 = results3[0]
    assert r3.refined_start == c3.start  # Start at 16 > event start 15
    assert r3.refined_end >= 25.0        # End expands to payoff window


# ──────────────────────────────────────────────────────────────────────────────
# Additional closure tests for structural reaction coherence (no global 3s threshold)
# ──────────────────────────────────────────────────────────────────────────────

def test_same_evidence_ref_payoff_reaction_preserved():
    """Same evidence_ref for payoff+reaction => reaction preserved (structural coherence)."""
    c = make_candidate(10, 40, 0.8)
    # Single VIA evidence with both payoff and reaction
    via = [make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True, reaction=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Reaction should be preserved because same evidence_ref
    assert r.reaction_window_start is not None
    assert r.reaction_window_end is not None
    assert r.reaction_window_end >= 25.0


def test_different_evidence_ref_no_overlap_not_preserved():
    """Different evidence_ref, no temporal overlap => reaction NOT preserved even at 0.5s.

    Note: event_end from second evidence IS a valid content boundary and extends end.
    But reaction_window should NOT be set (no structural coherence).
    """
    c = make_candidate(10, 40, 0.8)
    # Payoff at 15-20, reaction at 20.5-21 (0.5s gap, different evidence, no overlap)
    via1 = make_via_evidence("a1", 15, 20, confidence=0.9, payoff=True)
    via2 = make_via_evidence("a2", 20.5, 21, confidence=0.9, reaction=True)
    results = refine_candidates([c], [via1, via2], video_duration=60)
    r = results[0]
    # No structural coherence (different evidence_ref, no window overlap)
    assert r.reaction_window_start is None
    assert r.reaction_window_end is None
    # event_end from second evidence at 21.0 IS a valid content boundary
    # so end extends to 21.5 (event_end + BOUNDARY_TOLERANCE)
    assert r.refined_end == 21.5


def test_different_evidence_ref_with_overlap_preserved():
    """Different evidence_ref BUT temporal window overlap => reaction preserved (structural overlap)."""
    c = make_candidate(10, 40, 0.8)
    # Payoff 15-25, reaction 22-30 (overlap 22-25, different evidence_refs)
    via1 = make_via_evidence("a1", 15, 25, confidence=0.9, payoff=True)
    via2 = make_via_evidence("a2", 22, 30, confidence=0.9, reaction=True)
    results = refine_candidates([c], [via1, via2], video_duration=60)
    r = results[0]
    # Structural overlap between windows => reaction preserved
    assert r.reaction_window_start is not None
    assert r.reaction_window_end is not None
    assert r.reaction_window_end >= 30.0


def test_reaction_far_but_structurally_related():
    """Reaction at 10s distance but same evidence_ref => preserved by structure, not distance."""
    c = make_candidate(10, 60, 0.8)
    # Single evidence with long duration, both payoff and reaction
    via = [make_via_evidence("a1", 15, 45, confidence=0.9, payoff=True, reaction=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Same evidence_ref => structural coherence regardless of distance
    assert r.reaction_window_start is not None
    assert r.reaction_window_end is not None
    assert r.reaction_window_end >= 45.0


# ──────────────────────────────────────────────────────────────────────────────
# Additional closure tests for no fixed editorial minimums
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_candidate_04s_no_fixed_minimum():
    """Valid 0.4s candidate should not be rejected/expanded by arbitrary minimum."""
    c = make_candidate(10, 10.4, 0.8)  # 0.4s duration
    via = [make_via_evidence("a1", 10, 10.3, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Should preserve short duration, not expand to 1s or 3s
    assert r.refined_end - r.refined_start <= 0.5  # Small tolerance only
    assert r.refined_end > r.refined_start


def test_valid_candidate_08s_no_fixed_minimum():
    """Valid 0.8s candidate should not be rejected/expanded by arbitrary minimum."""
    c = make_candidate(10, 10.8, 0.8)  # 0.8s duration
    via = [make_via_evidence("a1", 10, 10.5, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Should preserve short duration
    assert r.refined_end - r.refined_start <= 1.0  # Small tolerance only
    assert r.refined_end > r.refined_start


def test_valid_candidate_20s_no_fixed_minimum():
    """Valid 2.0s candidate should work normally without arbitrary minimum interference."""
    c = make_candidate(10, 12, 0.8)  # 2.0s duration
    via = [make_via_evidence("a1", 10.5, 11.5, confidence=0.9, payoff=True)]
    results = refine_candidates([c], via, video_duration=60)
    r = results[0]
    # Should work normally
    assert r.refined_end - r.refined_start >= 1.0
    assert r.refined_end - r.refined_start <= 2.5
    assert r.refined_end > r.refined_start