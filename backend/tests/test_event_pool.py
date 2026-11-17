"""Tests for Event Pool (SNIPRA v1.5F)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.pipeline.event_pool import (
    EventPool,
    EventTopicContext,
    _associate_topics,
    _select_primary_topic,
    _stable_candidate_id_from_candidate,
    _union_evidence_refs,
    build_event_pool,
    event_by_id,
    events_for_range,
    events_for_topic,
)
from app.pipeline.highlight_boundaries import (
    RefinedBoundaries,
    RefinementStatus,
)
from app.pipeline.highlight_scoring import ComponentScore, ScoredCandidate
from app.pipeline.topic_timeline import TopicSpan, TopicTimeline, TopicTimelineConfig
from app.providers.base import ClipCandidate

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_candidate(
    start: float,
    end: float,
    title: str = "Test",
    hook: str = "",
    reason: str = "",
    score: float = 0.5,
) -> ClipCandidate:
    return ClipCandidate(
        start=start,
        end=end,
        title=title,
        hook=hook,
        reason=reason,
        score=score,
    )


def _make_scored_candidate(
    candidate: ClipCandidate,
    overall_score: float = 0.7,
    rank: int = 1,
    component_scores: list[ComponentScore] | None = None,
) -> ScoredCandidate:
    if component_scores is None:
        component_scores = [
            ComponentScore(
                name="base_score",
                raw_value=candidate.score,
                weight=0.5,
                contribution=candidate.score * 0.5,
                evidence_refs=[],
                note="base",
            ),
            ComponentScore(
                name="event_strength",
                raw_value=0.8,
                weight=0.2,
                contribution=0.16,
                evidence_refs=["ev1"],
                note="event",
            ),
        ]
    return ScoredCandidate(
        candidate=candidate,
        overall_score=overall_score,
        rank=rank,
        component_scores=tuple(component_scores),
    )


def _make_refinement(
    candidate: ClipCandidate,
    refined_start: float | None = None,
    refined_end: float | None = None,
    status: RefinementStatus = RefinementStatus.REFINED,
    confidence: float = 0.8,
    peak_at: float | None = 15.0,
    payoff_at: float | None = None,
    payoff_window_start: float | None = None,
    payoff_window_end: float | None = None,
    reaction_at: float | None = None,
    reaction_window_start: float | None = None,
    reaction_window_end: float | None = None,
    evidence_refs: tuple[str, ...] = ("refine-1", "shared"),
) -> RefinedBoundaries:
    cand_id = _stable_candidate_id_from_candidate(candidate)
    return RefinedBoundaries(
        candidate_id=cand_id,
        original_start=candidate.start,
        original_end=candidate.end,
        refined_start=refined_start if refined_start is not None else candidate.start,
        refined_end=refined_end if refined_end is not None else candidate.end,
        peak_at=peak_at,
        setup_at=None,
        action_at=None,
        payoff_at=payoff_at,
        payoff_window_start=payoff_window_start,
        payoff_window_end=payoff_window_end,
        reaction_at=reaction_at,
        reaction_window_start=reaction_window_start,
        reaction_window_end=reaction_window_end,
        status=status,
        confidence=confidence,
        anchors=(),
        evidence_refs=evidence_refs,
        notes=("note1",),
    )


def _make_topic_span(
    topic_id: str,
    start: float,
    end: float,
    keywords: tuple[str, ...] = ("topic",),
    confidence: float = 0.9,
    evidence_refs: tuple[str, ...] = ("topic-ev",),
) -> TopicSpan:
    return TopicSpan(
        topic_id=topic_id,
        start=start,
        end=end,
        keywords=keywords,
        representative_text="representative text",
        confidence=confidence,
        basis=("lexical",),
        transcript_refs=(f"{start:.3f}-{end:.3f}",),
        evidence_refs=evidence_refs,
    )


def _make_timeline(
    spans: list[TopicSpan] | None = None,
    asset_id: str | None = "asset-1",
) -> TopicTimeline:
    if spans is None:
        spans = []
    return TopicTimeline(
        asset_id=asset_id,
        spans=tuple(spans),
        total_duration=100.0,
        config=TopicTimelineConfig(),
    )


def _basic_setup() -> tuple[list[ScoredCandidate], list[RefinedBoundaries], TopicTimeline]:
    """Basic valid setup: 1 candidate, 1 refinement, 1 topic."""
    cand = _make_candidate(0.0, 10.0, "Event A", score=0.8)
    scored = [_make_scored_candidate(cand, overall_score=0.85, rank=1)]
    refinement = _make_refinement(cand, refined_start=1.0, refined_end=9.0)
    topic = _make_topic_span("topic-1", 0.0, 20.0, evidence_refs=("topic-ev",))
    timeline = _make_timeline([topic])
    return scored, [refinement], timeline


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_input_returns_empty_pool(self):
        pool = build_event_pool([], [], _make_timeline(asset_id=None))
        assert pool.items == ()
        assert pool.asset_id is None

    def test_single_candidate_single_refinement_one_item(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        assert len(pool.items) == 1
        assert pool.items[0].event_id == refinements[0].candidate_id

    def test_event_id_equals_refinement_candidate_id(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        assert pool.items[0].event_id == refinements[0].candidate_id

    def test_compatibility_helper_matches_15d(self):
        cand = _make_candidate(0.0, 10.0, "Title", "Hook", "Reason")
        # This is the exact formula from 1.5C/1.5D
        import hashlib
        parts = [str(cand.start), str(cand.end), cand.title or "", cand.hook or "", cand.reason or ""]
        expected = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        assert _stable_candidate_id_from_candidate(cand) == expected

    def test_missing_refinement_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="missing refinement"):
            build_event_pool(scored, [], timeline)

    def test_orphan_refinement_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        timeline = _make_timeline()
        # Create a different candidate for refinement
        other_cand = _make_candidate(20.0, 30.0)
        orphan_ref = _make_refinement(other_cand)
        with pytest.raises(ValueError, match="orphan refinement"):
            build_event_pool(scored, [orphan_ref], timeline)

    def test_duplicate_scored_candidate_id_raises(self):
        cand1 = _make_candidate(0.0, 10.0, "Same")
        cand2 = _make_candidate(0.0, 10.0, "Same")  # Same fields = same ID
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=2),
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Duplicate candidate IDs"):
            build_event_pool(scored, refinements, timeline)

    def test_duplicate_refinement_candidate_id_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        # Two refinements with same candidate_id
        ref1 = _make_refinement(cand)
        ref2 = replace(ref1, refined_start=2.0)  # Same candidate_id
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Duplicate candidate_ids"):
            build_event_pool(scored, [ref1, ref2], timeline)

    def test_rank_none_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [ScoredCandidate(candidate=cand, overall_score=0.5, rank=None, component_scores=())]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Missing rank"):
            build_event_pool(scored, [refinement], timeline)

    def test_rank_zero_or_negative_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=0)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Invalid rank"):
            build_event_pool(scored, [refinement], timeline)

    def test_duplicate_rank_raises(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=1),  # Duplicate rank
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Duplicate ranks"):
            build_event_pool(scored, refinements, timeline)

    def test_ranks_not_contiguous_raises(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=3),  # Gap
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="contiguous"):
            build_event_pool(scored, refinements, timeline)

    def test_overall_score_negative_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=-0.1)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="out of bounds"):
            build_event_pool(scored, [refinement], timeline)

    def test_overall_score_gt_one_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=1.1)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="out of bounds"):
            build_event_pool(scored, [refinement], timeline)

    def test_nan_score_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=float('nan'))]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Invalid score"):
            build_event_pool(scored, [refinement], timeline)

    def test_inf_score_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=float('inf'))]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Invalid score"):
            build_event_pool(scored, [refinement], timeline)

    def test_invalid_refined_range_end_le_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=10.0, refined_end=5.0)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Invalid refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_negative_refined_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=-1.0)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Negative refined_start"):
            build_event_pool(scored, [refinement], timeline)

    def test_refinement_original_start_mismatch_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=1.0)
        # Manually corrupt original_start
        bad_ref = replace(refinement, original_start=100.0)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Original start mismatch"):
            build_event_pool(scored, [bad_ref], timeline)

    def test_refinement_original_end_mismatch_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand)
        bad_ref = replace(refinement, original_end=100.0)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Original end mismatch"):
            build_event_pool(scored, [bad_ref], timeline)

    # ──────────────────────────────────────────────────────────────────────────────
    # Refined range validation tests (strict finite checks)
    # ──────────────────────────────────────────────────────────────────────────────

    def test_negative_refined_start_tiny_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        # Very small negative: -1e-9
        refinement = _make_refinement(cand, refined_start=-1e-9)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Negative refined_start"):
            build_event_pool(scored, [refinement], timeline)

    def test_refined_end_equals_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=1.0, refined_end=1.0)
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Invalid refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_nan_refined_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=float('nan'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_nan_refined_end_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_end=float('nan'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_inf_refined_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=float('inf'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_inf_refined_end_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_end=float('inf'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_neg_inf_refined_start_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_start=float('-inf'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)

    def test_neg_inf_refined_end_raises(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand)]
        refinement = _make_refinement(cand, refined_end=float('-inf'))
        timeline = _make_timeline()
        with pytest.raises(ValueError, match="Non-finite refined range"):
            build_event_pool(scored, [refinement], timeline)


# ──────────────────────────────────────────────────────────────────────────────
# Ordering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOrdering:
    def test_scored_input_unordered_pool_ordered_by_rank(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        # Provide in wrong order (rank 2 first)
        scored = [
            _make_scored_candidate(cand2, rank=2),
            _make_scored_candidate(cand1, rank=1),
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        pool = build_event_pool(scored, refinements, timeline)
        # Output must be rank 1 then rank 2
        assert pool.items[0].rank == 1
        assert pool.items[1].rank == 2

    def test_refinements_unordered_output_identical(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=2),
        ]
        # Provide refinements in reverse order
        refinements = [_make_refinement(cand2), _make_refinement(cand1)]
        timeline = _make_timeline()
        pool1 = build_event_pool(scored, refinements, timeline)
        pool2 = build_event_pool(scored, list(reversed(refinements)), timeline)
        assert pool1 == pool2

    def test_deterministic_rerun_identical(self):
        scored, refinements, timeline = _basic_setup()
        pool1 = build_event_pool(scored, refinements, timeline)
        pool2 = build_event_pool(scored, refinements, timeline)
        assert pool1 == pool2

    def test_event_id_stable_across_topic_timeline_changes(self):
        scored, refinements, timeline = _basic_setup()
        pool1 = build_event_pool(scored, refinements, timeline)
        # Different timeline, same scored/refinements
        other_topic = _make_topic_span("topic-99", 0.0, 20.0)
        timeline2 = _make_timeline([other_topic])
        pool2 = build_event_pool(scored, refinements, timeline2)
        assert pool1.items[0].event_id == pool2.items[0].event_id


# ──────────────────────────────────────────────────────────────────────────────
# Topic association tests (using REFINED range)
# ──────────────────────────────────────────────────────────────────────────────

class TestTopicAssociation:
    def test_topic_overlap_uses_refined_range_not_original(self):
        cand = _make_candidate(0.0, 30.0, "Event")
        scored = [_make_scored_candidate(cand, rank=1)]
        # Refined: 10-20 (inside Topic B only)
        refinement = _make_refinement(cand, refined_start=10.0, refined_end=20.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 10.0, 20.0)
        topic_c = _make_topic_span("topic-c", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b, topic_c])
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert item.primary_topic_id == "topic-b"
        assert len(item.topic_contexts) == 1
        assert item.topic_contexts[0].topic_id == "topic-b"

    def test_event_inside_one_topic_one_context(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert len(item.topic_contexts) == 1
        assert item.topic_contexts[0].topic_id == "topic-1"
        assert item.topic_contexts[0].overlap_seconds == 6.0

    def test_event_crosses_two_topics_two_contexts(self):
        cand = _make_candidate(0.0, 30.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=8.0, refined_end=22.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 10.0, 20.0)
        topic_c = _make_topic_span("topic-c", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b, topic_c])
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        # Event 8-22 overlaps with all three: a(8-10), b(10-20), c(20-22)
        assert len(item.topic_contexts) == 3
        topic_ids = {c.topic_id for c in item.topic_contexts}
        assert topic_ids == {"topic-a", "topic-b", "topic-c"}
        # Check overlap seconds
        ctx_a = next(c for c in item.topic_contexts if c.topic_id == "topic-a")
        ctx_b = next(c for c in item.topic_contexts if c.topic_id == "topic-b")
        ctx_c = next(c for c in item.topic_contexts if c.topic_id == "topic-c")
        assert ctx_a.overlap_seconds == 2.0  # 8-10
        assert ctx_b.overlap_seconds == 10.0  # 10-20
        assert ctx_c.overlap_seconds == 2.0  # 20-22

    def test_event_touches_boundary_exactly_no_positive_overlap(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=0.0, refined_end=10.0)
        # Topic starts exactly at event end
        topic = _make_topic_span("topic-1", 10.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert item.topic_contexts == ()  # No positive overlap

    def test_event_inside_gap_no_topic_contexts(self):
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b])  # Gap 10-20
        cand = _make_candidate(12.0, 18.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=12.0, refined_end=18.0)
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert item.topic_contexts == ()
        assert item.primary_topic_id is None

    def test_event_inside_gap_remains_in_pool(self):
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b])
        cand = _make_candidate(12.0, 18.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=12.0, refined_end=18.0)
        pool = build_event_pool(scored, [refinement], timeline)
        assert len(pool.items) == 1
        assert pool.items[0].event_id == refinement.candidate_id


# ──────────────────────────────────────────────────────────────────────────────
# Primary topic selection tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPrimaryTopicSelection:
    def test_primary_topic_highest_overlap(self):
        cand = _make_candidate(0.0, 30.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=8.0, refined_end=22.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)  # 2s overlap
        topic_b = _make_topic_span("topic-b", 10.0, 20.0, confidence=0.5)  # 10s overlap
        topic_c = _make_topic_span("topic-c", 20.0, 30.0)  # 2s overlap
        timeline = _make_timeline([topic_a, topic_b, topic_c])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].primary_topic_id == "topic-b"

    def test_tie_overlap_higher_confidence_wins(self):
        cand = _make_candidate(0.0, 20.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=0.0, refined_end=20.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0, confidence=0.9)  # 10s
        topic_b = _make_topic_span("topic-b", 10.0, 20.0, confidence=0.5)  # 10s
        timeline = _make_timeline([topic_a, topic_b])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].primary_topic_id == "topic-a"

    def test_tie_overlap_confidence_earlier_start_wins(self):
        cand = _make_candidate(0.0, 20.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=0.0, refined_end=20.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0, confidence=0.8)  # 10s
        topic_b = _make_topic_span("topic-b", 10.0, 20.0, confidence=0.8)  # 10s
        timeline = _make_timeline([topic_a, topic_b])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].primary_topic_id == "topic-a"

    def test_total_tie_topic_id_lexicographic_wins(self):
        cand = _make_candidate(0.0, 20.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=0.0, refined_end=20.0)
        topic_a = _make_topic_span("aaa", 0.0, 10.0, confidence=0.8)
        topic_b = _make_topic_span("bbb", 10.0, 20.0, confidence=0.8)
        timeline = _make_timeline([topic_a, topic_b])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].primary_topic_id == "aaa"

    def test_primary_topic_four_way_tie_break_topic_id_asc(self):
        """Test 4th tie-break criterion: topic_id lexicographically ASC."""
        # Two contexts with identical overlap_seconds, confidence, topic_start
        # Only topic_id differs: "bbb" vs "aaa" -> "aaa" should win
        ctx_b = EventTopicContext(
            topic_id="bbb", topic_start=10.0, topic_end=20.0,
            overlap_seconds=5.0, overlap_ratio=0.5, confidence=0.8,
            keywords=(), representative_text="", evidence_refs=(),
        )
        ctx_a = EventTopicContext(
            topic_id="aaa", topic_start=10.0, topic_end=20.0,
            overlap_seconds=5.0, overlap_ratio=0.5, confidence=0.8,
            keywords=(), representative_text="", evidence_refs=(),
        )
        # Input order: bbb first, then aaa
        assert _select_primary_topic((ctx_b, ctx_a)) == "aaa"
        # Input order: aaa first, then bbb
        assert _select_primary_topic((ctx_a, ctx_b)) == "aaa"

    def test_primary_topic_input_order_independence(self):
        """Primary topic selection must be independent of input order."""
        ctx1 = EventTopicContext(
            topic_id="topic-2", topic_start=10.0, topic_end=20.0,
            overlap_seconds=8.0, overlap_ratio=0.6, confidence=0.9,
            keywords=(), representative_text="", evidence_refs=(),
        )
        ctx2 = EventTopicContext(
            topic_id="topic-1", topic_start=0.0, topic_end=10.0,
            overlap_seconds=8.0, overlap_ratio=0.6, confidence=0.9,
            keywords=(), representative_text="", evidence_refs=(),
        )
        # topic-1 has earlier topic_start, so it wins regardless of input order
        assert _select_primary_topic((ctx1, ctx2)) == "topic-1"
        assert _select_primary_topic((ctx2, ctx1)) == "topic-1"

    def test_topic_contexts_chronological_order(self):
        cand = _make_candidate(0.0, 30.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=8.0, refined_end=22.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 10.0, 20.0)
        topic_c = _make_topic_span("topic-c", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b, topic_c])
        pool = build_event_pool(scored, [refinement], timeline)
        ctx_ids = [c.topic_id for c in pool.items[0].topic_contexts]
        assert ctx_ids == ["topic-a", "topic-b", "topic-c"]  # chronological

    def test_overlap_seconds_correct(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        ctx = pool.items[0].topic_contexts[0]
        assert ctx.overlap_seconds == 6.0  # 2-8 within 0-20

    def test_overlap_ratio_correct(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        ctx = pool.items[0].topic_contexts[0]
        # overlap_ratio = overlap_seconds / event_duration = 6.0 / 6.0 = 1.0
        assert ctx.overlap_ratio == 1.0

    def test_overlap_ratio_bounded_0_1(self):
        # Event larger than topic
        cand = _make_candidate(0.0, 100.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=0.0, refined_end=100.0)
        topic = _make_topic_span("topic-1", 10.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        ctx = pool.items[0].topic_contexts[0]
        assert 0.0 <= ctx.overlap_ratio <= 1.0

    def test_topic_confidence_does_not_alter_overall_score(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0, confidence=0.1)  # Low confidence
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].overall_score == 0.7

    def test_refinement_confidence_does_not_alter_overall_score(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0, confidence=0.1)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].overall_score == 0.7


# ──────────────────────────────────────────────────────────────────────────────
# Score and rank preservation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreRankPreservation:
    def test_rank_preserved_exactly(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].rank == 1

    def test_overall_score_preserved_exactly(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7345, rank=1)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].overall_score == 0.7345

    def test_component_scores_copied_exactly(self):
        comp = ComponentScore(
            name="test_comp",
            raw_value=0.42,
            weight=0.3,
            contribution=0.126,
            evidence_refs=["ev1", "ev2"],
            note="test note",
        )
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1, component_scores=[comp])]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        trace = pool.items[0].score_components[0]
        assert trace.name == "test_comp"
        assert trace.raw_value == 0.42
        assert trace.weight == 0.3
        assert trace.contribution == 0.126
        assert trace.evidence_refs == ("ev1", "ev2")
        assert trace.note == "test note"

    def test_component_evidence_refs_converted_to_tuple(self):
        cand = _make_candidate(0.0, 10.0)
        comp = ComponentScore(
            name="test",
            raw_value=0.5,
            weight=0.5,
            contribution=0.25,
            evidence_refs=["b", "a", "c"],  # Unsorted
            note="",
        )
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1, component_scores=[comp])]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        trace = pool.items[0].score_components[0]
        assert trace.evidence_refs == ("a", "b", "c")  # Sorted tuple

    def test_refinement_evidence_refs_in_union(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, evidence_refs=("refine-only",))
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert "refine-only" in pool.items[0].evidence_refs

    def test_topic_evidence_refs_in_union(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand)
        topic = _make_topic_span("topic-1", 0.0, 20.0, evidence_refs=("topic-only",))
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert "topic-only" in pool.items[0].evidence_refs

    def test_duplicate_evidence_refs_deduplicated(self):
        cand = _make_candidate(0.0, 10.0)
        comp = ComponentScore(
            name="test", raw_value=0.5, weight=0.5, contribution=0.25,
            evidence_refs=["shared", "score-a"], note=""
        )
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1, component_scores=[comp])]
        refinement = _make_refinement(cand, evidence_refs=("shared", "refine-a"))
        topic = _make_topic_span("topic-1", 0.0, 20.0, evidence_refs=("shared", "topic-a"))
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        refs = pool.items[0].evidence_refs
        assert refs.count("shared") == 1
        assert set(refs) == {"refine-a", "score-a", "shared", "topic-a"}

    def test_evidence_refs_sorted_deterministic(self):
        cand = _make_candidate(0.0, 10.0)
        comp = ComponentScore(
            name="test", raw_value=0.5, weight=0.5, contribution=0.25,
            evidence_refs=["z", "a", "m"], note=""
        )
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1, component_scores=[comp])]
        refinement = _make_refinement(cand, evidence_refs=("b", "y"))
        topic = _make_topic_span("topic-1", 0.0, 20.0, evidence_refs=("c", "x"))
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].evidence_refs == tuple(sorted(["z", "a", "m", "b", "y", "c", "x"]))


# ──────────────────────────────────────────────────────────────────────────────
# Refinement status preservation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRefinementStatusPreservation:
    def test_uncertain_refinement_remains_in_pool(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, status=RefinementStatus.UNCERTAIN)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert len(pool.items) == 1
        assert pool.items[0].refinement_status == "uncertain"

    def test_unchanged_refinement_remains_in_pool(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, status=RefinementStatus.UNCHANGED)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert len(pool.items) == 1
        assert pool.items[0].refinement_status == "unchanged"

    def test_refined_status_preserved(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, status=RefinementStatus.REFINED)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].refinement_status == "refined"


# ──────────────────────────────────────────────────────────────────────────────
# Boundary field preservation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBoundaryFields:
    def test_peak_at_preserved(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, peak_at=15.5)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].peak_at == 15.5

    def test_payoff_point_preserved(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, payoff_at=12.0)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].payoff_at == 12.0

    def test_payoff_window_preserved(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, payoff_window_start=10.0, payoff_window_end=14.0)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].payoff_window_start == 10.0
        assert pool.items[0].payoff_window_end == 14.0

    def test_reaction_window_preserved(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, reaction_window_start=14.0, reaction_window_end=18.0)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].reaction_window_start == 14.0
        assert pool.items[0].reaction_window_end == 18.0


# ──────────────────────────────────────────────────────────────────────────────
# Free-text fields copied literally
# ──────────────────────────────────────────────────────────────────────────────

class TestFreeTextFields:
    def test_title_hook_reason_copied_literally(self):
        cand = _make_candidate(0.0, 10.0, title="My Title", hook="My Hook", reason="My Reason")
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand)
        timeline = _make_timeline()
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert item.title == "My Title"
        assert item.hook == "My Hook"
        assert item.reason == "My Reason"

    def test_free_text_does_not_affect_topic_assignment(self):
        # Even with misleading title, topic assignment uses refined range only
        cand = _make_candidate(0.0, 30.0, title="Topic A content", hook="", reason="")
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=10.0, refined_end=20.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)  # NOT in refined range
        topic_b = _make_topic_span("topic-b", 10.0, 20.0)  # In refined range
        timeline = _make_timeline([topic_a, topic_b])
        pool = build_event_pool(scored, [refinement], timeline)
        assert pool.items[0].primary_topic_id == "topic-b"  # Based on refined range, not title


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryHelpers:
    def test_event_by_id_existing(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        found = event_by_id(pool, refinements[0].candidate_id)
        assert found is not None
        assert found.event_id == refinements[0].candidate_id

    def test_event_by_id_nonexistent(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        found = event_by_id(pool, "nonexistent-id")
        assert found is None

    def test_events_for_topic_correct(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        result = events_for_topic(pool, "topic-1")
        assert len(result) == 1
        assert result[0].event_id == refinement.candidate_id

    def test_events_for_topic_unknown_empty(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        result = events_for_topic(pool, "unknown-topic")
        assert result == ()

    def test_events_for_range_correct(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        # Query range overlaps with refined range 2-8
        result = events_for_range(pool, 5.0, 15.0)
        assert len(result) == 1

    def test_events_for_range_outside_empty(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        # Event at 0-10, query 50-60
        result = events_for_range(pool, 50.0, 60.0)
        assert result == ()

    def test_query_helpers_preserve_pool_order(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=2),
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        topic = _make_topic_span("topic-1", 0.0, 100.0)
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, refinements, timeline)

        # events_for_topic should preserve order
        result = events_for_topic(pool, "topic-1")
        assert [r.rank for r in result] == [1, 2]

        # events_for_range should preserve order
        result = events_for_range(pool, 0.0, 100.0)
        assert [r.rank for r in result] == [1, 2]


# ──────────────────────────────────────────────────────────────────────────────
# Asset ID tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAssetId:
    def test_asset_id_from_topic_timeline(self):
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        assert pool.asset_id == "asset-1"

    def test_asset_id_none_allowed(self):
        scored, refinements, _ = _basic_setup()
        timeline = _make_timeline(asset_id=None)
        pool = build_event_pool(scored, refinements, timeline)
        assert pool.asset_id is None


# ──────────────────────────────────────────────────────────────────────────────
# No filtering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNoFiltering:
    def test_no_filtering_by_low_score(self):
        cand1 = _make_candidate(0.0, 10.0)
        cand2 = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand1, overall_score=0.9, rank=1),
            _make_scored_candidate(cand2, overall_score=0.1, rank=2),  # Very low score
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        pool = build_event_pool(scored, refinements, timeline)
        assert len(pool.items) == 2
        assert pool.items[0].overall_score == 0.9
        assert pool.items[1].overall_score == 0.1

    def test_no_filtering_by_low_topic_confidence(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=2.0, refined_end=8.0)
        topic = _make_topic_span("topic-1", 0.0, 20.0, confidence=0.01)  # Very low
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        assert len(pool.items) == 1

    def test_no_dedupe_overlapping_events(self):
        cand1 = _make_candidate(0.0, 20.0)
        cand2 = _make_candidate(10.0, 30.0)  # Overlaps with cand1
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=2),
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        pool = build_event_pool(scored, refinements, timeline)
        assert len(pool.items) == 2

    def test_two_overlapping_candidates_remain_both(self):
        cand1 = _make_candidate(0.0, 20.0)
        cand2 = _make_candidate(15.0, 35.0)  # Overlaps
        scored = [
            _make_scored_candidate(cand1, rank=1),
            _make_scored_candidate(cand2, rank=2),
        ]
        refinements = [_make_refinement(cand1), _make_refinement(cand2)]
        timeline = _make_timeline()
        pool = build_event_pool(scored, refinements, timeline)
        assert len(pool.items) == 2
        assert pool.items[0].event_id != pool.items[1].event_id


# ──────────────────────────────────────────────────────────────────────────────
# No mutation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNoMutation:
    def test_input_objects_not_mutated(self):
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand)
        topic = _make_topic_span("topic-1", 0.0, 20.0)
        timeline = _make_timeline([topic])

        original_cand_start = cand.start
        original_ref_start = refinement.refined_start
        original_topic_start = topic.start

        build_event_pool(scored, [refinement], timeline)

        assert cand.start == original_cand_start
        assert refinement.refined_start == original_ref_start
        assert topic.start == original_topic_start


# ──────────────────────────────────────────────────────────────────────────────
# No LLM/Network/Media tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNoExternalCalls:
    def test_no_llm_network_media_calls(self):
        # Just verify it runs without external dependencies
        scored, refinements, timeline = _basic_setup()
        pool = build_event_pool(scored, refinements, timeline)
        assert isinstance(pool, EventPool)


# ──────────────────────────────────────────────────────────────────────────────
# No scoring / no re-ranking tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNoRescoring:
    def test_no_re_scoring(self):
        # Event A: high score, low topic confidence
        # Event B: low score, high topic confidence
        cand_a = _make_candidate(0.0, 10.0)
        cand_b = _make_candidate(20.0, 30.0)
        scored = [
            _make_scored_candidate(cand_a, overall_score=0.90, rank=1),
            _make_scored_candidate(cand_b, overall_score=0.60, rank=2),
        ]
        refinements = [
            _make_refinement(cand_a, refined_start=2.0, refined_end=8.0),
            _make_refinement(cand_b, refined_start=22.0, refined_end=28.0),
        ]
        topic_a = _make_topic_span("topic-a", 0.0, 20.0, confidence=0.1)
        topic_b = _make_topic_span("topic-b", 20.0, 40.0, confidence=0.9)
        timeline = _make_timeline([topic_a, topic_b])
        pool = build_event_pool(scored, refinements, timeline)
        # A stays rank 1 with score 0.90
        assert pool.items[0].rank == 1
        assert pool.items[0].overall_score == 0.90
        assert pool.items[0].primary_topic_id == "topic-a"
        # B stays rank 2 with score 0.60
        assert pool.items[1].rank == 2
        assert pool.items[1].overall_score == 0.60
        assert pool.items[1].primary_topic_id == "topic-b"


# ──────────────────────────────────────────────────────────────────────────────
# Integration tests (C + D + E → F)
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_integration_c_d_e_to_f(self):
        """Integration test using real 1.5C/1.5D/1.5E functions."""
        # VIA v2 contract migration: Block 02 renamed VideoObservationNature ->
        # EvidenceNature (same members/values OBSERVATION/INFERENCE/UNKNOWN).
        from app.editor.via_evidence import EvidenceKind, EvidenceNature, NormalizedVideoEvidence
        from app.pipeline.highlight_boundaries import refine_candidates
        from app.pipeline.highlight_scoring import score_candidates
        from app.pipeline.topic_timeline import build_topic_timeline

        # Create synthetic candidates
        cand1 = ClipCandidate(start=0.0, end=15.0, title="Event A", score=0.8, reason="First event")
        cand2 = ClipCandidate(start=20.0, end=35.0, title="Event B", score=0.7, reason="Second event")
        cand3 = ClipCandidate(start=40.0, end=50.0, title="Event C", score=0.6, reason="Third event")

        candidates = [cand1, cand2, cand3]

        # Synthetic VIA evidence
        via_evidence = [
            NormalizedVideoEvidence(
                evidence_id="via-1",
                asset_id="asset-1",
                start_us=1_000_000,
                end_us=14_000_000,
                evidence_kind=EvidenceKind.EVENT,
                nature=EvidenceNature.OBSERVATION,
                description="Action event",
                confidence_original=0.9,
                source="via",
                provider="test",
                model="test",
            ),
            NormalizedVideoEvidence(
                evidence_id="via-2",
                asset_id="asset-1",
                start_us=21_000_000,
                end_us=34_000_000,
                evidence_kind=EvidenceKind.EVENT,
                nature=EvidenceNature.OBSERVATION,
                description="Another action",
                confidence_original=0.8,
                source="via",
                provider="test",
                model="test",
            ),
        ]

        # Transcript segments
        transcript = [
            {"start": 0.0, "end": 5.0, "text": "Hello world"},
            {"start": 5.0, "end": 10.0, "text": "I won the game victory"},
            {"start": 20.0, "end": 25.0, "text": "Another segment"},
            {"start": 25.0, "end": 30.0, "text": "Clutch play right there"},
            {"start": 40.0, "end": 45.0, "text": "Final segment"},
        ]

        # 1.5C: Score
        scored = score_candidates(candidates, via_evidence, transcript)
        assert len(scored) == 3
        assert all(s.rank is not None for s in scored)

        # 1.5D: Refine
        refinements = refine_candidates(
            candidates,
            via_evidence,
            transcript,
            signals_doc={"loudness": [], "scene_times": [], "silence_intervals": []},
            video_duration=100.0,
        )
        assert len(refinements) == 3

        # 1.5E: Topic Timeline
        topic_timeline = build_topic_timeline(transcript, via_evidence, video_duration=100.0, asset_id="asset-1")
        assert len(topic_timeline.spans) >= 1

        # 1.5F: Event Pool
        pool = build_event_pool(scored, refinements, topic_timeline)
        assert len(pool.items) == 3
        assert pool.asset_id == "asset-1"

        # Verify all requirements
        for i, item in enumerate(pool.items):
            # Rank preserved
            assert item.rank == i + 1
            # Score preserved
            assert item.overall_score == scored[i].overall_score
            # Refined range from D
            assert item.refined_start >= 0
            assert item.refined_end > item.refined_start
            # Topic from E
            if item.topic_contexts:
                assert all(isinstance(ctx, EventTopicContext) for ctx in item.topic_contexts)

        # Evidence refs traceable
        all_refs = set()
        for item in pool.items:
            all_refs.update(item.evidence_refs)
        # Should have refs from scoring, refinement, and topics
        assert len(all_refs) > 0

        # Deterministic
        pool2 = build_event_pool(scored, refinements, topic_timeline)
        assert pool == pool2

    def test_permutation_invariance(self):
        """Same inputs permuted produce identical pool."""
        # VIA v2 contract migration: Block 02 renamed VideoObservationNature ->
        # EvidenceNature (same members/values OBSERVATION/INFERENCE/UNKNOWN).
        from app.editor.via_evidence import EvidenceKind, EvidenceNature, NormalizedVideoEvidence
        from app.pipeline.highlight_boundaries import refine_candidates
        from app.pipeline.highlight_scoring import score_candidates
        from app.pipeline.topic_timeline import build_topic_timeline

        cand1 = ClipCandidate(start=0.0, end=15.0, title="A", score=0.8)
        cand2 = ClipCandidate(start=20.0, end=35.0, title="B", score=0.7)
        candidates = [cand1, cand2]

        via_evidence = [
            NormalizedVideoEvidence(
                evidence_id="via-1",
                asset_id="asset-1",
                start_us=1_000_000,
                end_us=14_000_000,
                evidence_kind=EvidenceKind.EVENT,
                nature=EvidenceNature.OBSERVATION,
                description="Event 1",
                confidence_original=0.9,
                source="via",
                provider="test",
                model="test",
            ),
            NormalizedVideoEvidence(
                evidence_id="via-2",
                asset_id="asset-1",
                start_us=21_000_000,
                end_us=34_000_000,
                evidence_kind=EvidenceKind.EVENT,
                nature=EvidenceNature.OBSERVATION,
                description="Event 2",
                confidence_original=0.8,
                source="via",
                provider="test",
                model="test",
            ),
        ]
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "First segment"},
            {"start": 20.0, "end": 30.0, "text": "Second segment"},
        ]

        scored = score_candidates(candidates, via_evidence, transcript)
        refinements = refine_candidates(candidates, via_evidence, transcript, {}, 100.0)
        topic_timeline = build_topic_timeline(transcript, via_evidence, video_duration=100.0, asset_id="asset-1")

        pool1 = build_event_pool(scored, refinements, topic_timeline)

        # Permute inputs
        import random
        random.seed(42)
        scored_perm = scored.copy()
        refinements_perm = refinements.copy()
        random.shuffle(scored_perm)
        random.shuffle(refinements_perm)

        pool2 = build_event_pool(scored_perm, refinements_perm, topic_timeline)

        assert pool1 == pool2


# ──────────────────────────────────────────────────────────────────────────────
# Legacy pool guard test
# ──────────────────────────────────────────────────────────────────────────────

class TestLegacyPoolGuard:
    def test_event_pool_does_not_import_editor_events(self):
        import inspect

        import app.pipeline.event_pool as ep_module
        source = inspect.getsource(ep_module)
        assert "app.editor.events" not in source
        assert "best_event_pool" not in source


# ──────────────────────────────────────────────────────────────────────────────
# Multi-topic event test
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiTopicEvent:
    def test_event_crossing_multiple_topics_remains_single_item(self):
        cand = _make_candidate(0.0, 30.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=8.0, refined_end=22.0)
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 10.0, 20.0)
        topic_c = _make_topic_span("topic-c", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b, topic_c])
        pool = build_event_pool(scored, [refinement], timeline)
        assert len(pool.items) == 1
        assert len(pool.items[0].topic_contexts) == 3
        # Primary is B (10s overlap vs 2s for A and C)
        assert pool.items[0].primary_topic_id == "topic-b"


# ──────────────────────────────────────────────────────────────────────────────
# Gap behavior test
# ──────────────────────────────────────────────────────────────────────────────

class TestGapBehavior:
    def test_event_in_gap_no_topic_contexts_present_in_pool(self):
        topic_a = _make_topic_span("topic-a", 0.0, 10.0)
        topic_b = _make_topic_span("topic-b", 20.0, 30.0)
        timeline = _make_timeline([topic_a, topic_b])
        cand = _make_candidate(12.0, 18.0)
        scored = [_make_scored_candidate(cand, rank=1)]
        refinement = _make_refinement(cand, refined_start=12.0, refined_end=18.0)
        pool = build_event_pool(scored, [refinement], timeline)
        item = pool.items[0]
        assert item.topic_contexts == ()
        assert item.primary_topic_id is None
        assert item.event_id == refinement.candidate_id


# ──────────────────────────────────────────────────────────────────────────────
# Evidence union test
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidenceUnion:
    def test_evidence_union_sorted_lexicographic(self):
        comp = ComponentScore(
            name="test", raw_value=0.5, weight=0.5, contribution=0.25,
            evidence_refs=["score-A", "shared"], note=""
        )
        cand = _make_candidate(0.0, 10.0)
        scored = [_make_scored_candidate(cand, overall_score=0.7, rank=1, component_scores=[comp])]
        refinement = _make_refinement(cand, evidence_refs=("refine-A", "shared"))
        topic = _make_topic_span("topic-1", 0.0, 20.0, evidence_refs=("topic-A", "shared"))
        timeline = _make_timeline([topic])
        pool = build_event_pool(scored, [refinement], timeline)
        expected = ("refine-A", "score-A", "shared", "topic-A")
        assert pool.items[0].evidence_refs == expected


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper tests
# ──────────────────────────────────────────────────────────────────────────────

class TestInternalHelpers:
    def test_select_primary_topic_empty(self):
        assert _select_primary_topic(()) is None

    def test_select_primary_topic_single(self):
        ctx = EventTopicContext(
            topic_id="t1", topic_start=0.0, topic_end=10.0,
            overlap_seconds=5.0, overlap_ratio=0.5, confidence=0.8,
            keywords=(), representative_text="", evidence_refs=(),
        )
        assert _select_primary_topic((ctx,)) == "t1"

    def test_associate_topics_empty_timeline(self):
        result = _associate_topics(0.0, 10.0, _make_timeline([]))
        assert result == ()

    def test_associate_topics_zero_duration_event(self):
        topic = _make_topic_span("t1", 0.0, 10.0)
        timeline = _make_timeline([topic])
        result = _associate_topics(5.0, 5.0, timeline)
        assert result == ()

    def test_union_evidence_refs_deterministic(self):
        class DummyScored:
            component_scores = (
                ComponentScore(name="a", raw_value=0, weight=0, contribution=0, evidence_refs=["b", "a"], note=""),
            )
        class DummyRef:
            evidence_refs = ("c", "a")
        class DummyCtx:
            evidence_refs = ("d", "a")
        scored = DummyScored()
        ref = DummyRef()
        contexts = (DummyCtx(),)
        result = _union_evidence_refs(scored, ref, contexts)
        assert result == tuple(sorted(["b", "a", "c", "d"]))


# ──────────────────────────────────────────────────────────────────────────────
# Refinement status propagation test
# ──────────────────────────────────────────────────────────────────────────────

class TestRefinementStatusPropagation:
    def test_all_three_statuses_preserved(self):
        for status in [RefinementStatus.REFINED, RefinementStatus.UNCHANGED, RefinementStatus.UNCERTAIN]:
            cand = _make_candidate(0.0, 10.0)
            scored = [_make_scored_candidate(cand, rank=1)]
            refinement = _make_refinement(cand, status=status)
            timeline = _make_timeline()
            pool = build_event_pool(scored, [refinement], timeline)
            assert pool.items[0].refinement_status == status.value.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])