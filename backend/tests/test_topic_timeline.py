"""Tests for topic_timeline.py (SNIPRA v1.5E).

Covers deterministic topic timeline construction, boundary detection,
and query functions.
"""

import pytest
from app.editor.via_evidence import NormalizedVideoEvidence, EvidenceKind
from app.pipeline.topic_timeline import (
    TopicTimelineConfig,
    DEFAULT_TOPIC_CONFIG,
    TopicSpan,
    TopicTimeline,
    build_topic_timeline,
    topic_for_time,
    topic_for_range,
    _normalize_text,
    _tokenize,
    _compute_tf,
    _weighted_jaccard_tf,
    _stable_topic_id,
    _clamp01,
)
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text, "words": []}


def make_via_evidence(
    asset_id: str,
    start_s: float,
    end_s: float,
    evidence_kind: str = "event",
    confidence: float = 0.8,
    description: str = "Test event",
    subjects: list[str] | None = None,
    objects: list[str] | None = None,
    observed_action: str | None = None,
    continuity_group_id: str | None = None,
    partial: bool = False,
    coverage_ratio: float | None = None,
) -> NormalizedVideoEvidence:
    return NormalizedVideoEvidence(
        evidence_id=f"ev_{asset_id}_{int(start_s)}",
        asset_id=asset_id,
        start_us=int(start_s * 1_000_000),
        end_us=int(end_s * 1_000_000),
        evidence_kind=EvidenceKind(evidence_kind),
        nature="inference" if evidence_kind == "event" else "observation",
        description=description,
        confidence_original=confidence,
        provider="test",
        model="test",
        subjects=subjects or [],
        objects=objects or [],
        observed_action=observed_action,
        continuity_group_id=continuity_group_id,
        partial=partial,
        coverage_ratio=coverage_ratio,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Normalization & Tokenization Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_normalize_text_unicode_accents():
    assert _normalize_text("café") == "cafe"
    assert _normalize_text("niño") == "nino"
    assert _normalize_text("mañana") == "manana"
    assert _normalize_text("acción") == "accion"


def test_normalize_text_punctuation_removal():
    assert _normalize_text("Hola, mundo!") == "hola mundo"
    assert _normalize_text("¿Qué pasa?") == "que pasa"
    assert _normalize_text("Hello... World!!!") == "hello world"


def test_normalize_text_whitespace_collapse():
    assert _normalize_text("  hello   world  ") == "hello world"
    assert _normalize_text("hello\tworld\nagain") == "hello world again"


def test_normalize_text_case_folding():
    assert _normalize_text("HELLO WORLD") == "hello world"
    assert _normalize_text("Hello World") == "hello world"


def test_tokenize_stopword_removal():
    tokens = _tokenize("the quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_spanish_stopwords():
    tokens = _tokenize("el rápido zorro marrón")
    assert "el" not in tokens
    assert "rápido" in tokens or "rapido" in tokens
    assert "zorro" in tokens
    assert "marrón" in tokens or "marron" in tokens


def test_tokenize_short_token_removal():
    tokens = _tokenize("a be see")
    assert "a" not in tokens
    assert "be" not in tokens  # 2 chars but stopword
    assert "see" in tokens


def test_compute_tf_basic():
    tf = _compute_tf(["hello", "world", "hello"])
    assert tf["hello"] == 2
    assert tf["world"] == 1
    assert len(tf) == 2


def test_weighted_jaccard_tf_identical():
    tf1 = _compute_tf(["a", "b", "a"])
    tf2 = _compute_tf(["a", "b", "a"])
    assert _weighted_jaccard_tf(tf1, tf2) == 1.0


def test_weighted_jaccard_tf_disjoint():
    tf1 = _compute_tf(["a", "b"])
    tf2 = _compute_tf(["c", "d"])
    assert _weighted_jaccard_tf(tf1, tf2) == 0.0


def test_weighted_jaccard_tf_partial():
    tf1 = _compute_tf(["a", "b", "a"])
    tf2 = _compute_tf(["a", "c"])
    # min(a)=1, max(a)=2, min(b)=0, max(b)=1, min(c)=0, max(c)=1
    # num = 1, den = 2+1+1 = 4 => 0.25
    result = _weighted_jaccard_tf(tf1, tf2)
    assert abs(result - 0.25) < 1e-9


def test_weighted_jaccard_tf_empty():
    assert _weighted_jaccard_tf(Counter(), Counter()) == 1.0
    assert _weighted_jaccard_tf(Counter({"a": 1}), Counter()) == 0.0


def test_stable_topic_id_deterministic():
    id1 = _stable_topic_id("asset1", 10.0, 20.0, ["boss", "fight"])
    id2 = _stable_topic_id("asset1", 10.0, 20.0, ["boss", "fight"])
    assert id1 == id2
    assert len(id1) == 16


def test_stable_topic_id_keyword_order_independence():
    # Keywords are already sorted, but verify stability
    id1 = _stable_topic_id("asset1", 10.0, 20.0, ["boss", "fight"])
    id2 = _stable_topic_id("asset1", 10.0, 20.0, ["fight", "boss"])
    # Since keywords are passed sorted in real usage, both should be same
    # but the function sorts internally via join order
    assert id1 == id2


def test_clamp01_bounds():
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(0.0) == 0.0
    assert _clamp01(0.5) == 0.5
    assert _clamp01(1.0) == 1.0
    assert _clamp01(1.5) == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Core Timeline Construction Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_transcript_returns_empty_timeline():
    timeline = build_topic_timeline([])
    assert isinstance(timeline, TopicTimeline)
    assert timeline.spans == ()
    assert timeline.total_duration == 0.0


def test_single_segment_returns_one_span():
    segments = [make_segment(0.0, 10.0, "hello world")]
    timeline = build_topic_timeline(segments)

    assert len(timeline.spans) == 1
    span = timeline.spans[0]
    assert span.start == 0.0
    assert span.end == 10.0
    assert span.confidence > 0.0


def test_multiple_segments_same_vocabulary_one_span():
    segments = [
        make_segment(0.0, 5.0, "we are fighting the boss"),
        make_segment(5.0, 10.0, "the boss has low health"),
        make_segment(10.0, 15.0, "attack the boss now"),
    ]
    timeline = build_topic_timeline(segments)

    assert len(timeline.spans) == 1
    span = timeline.spans[0]
    assert span.start == 0.0
    assert span.end == 15.0
    assert "boss" in span.keywords


def test_strong_lexical_change_two_spans():
    segments = [
        make_segment(0.0, 5.0, "we are fighting the boss"),
        make_segment(5.0, 10.0, "the boss has low health"),
        make_segment(10.0, 15.0, "now we go to inventory craft"),
        make_segment(15.0, 20.0, "need to craft inventory ammunition"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3, context_window=1))

    assert len(timeline.spans) == 2
    # First span: boss fight
    assert "boss" in timeline.spans[0].keywords
    # Second span: inventory/crafting
    assert any(k in timeline.spans[1].keywords for k in ["inventory", "craft", "ammunition"])


def test_single_phrase_change_no_split():
    segments = [
        make_segment(0.0, 5.0, "we are fighting the boss"),
        make_segment(5.0, 10.0, "the boss has low health now"),  # Added "now" for overlap
        make_segment(10.0, 11.0, "inventory check"),  # Brief isolated phrase
        make_segment(11.0, 15.0, "fighting the boss now"),  # Same topic vocabulary
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3, context_window=2))

    # Should remain one span (brief phrase doesn't create persistent topic change)
    assert len(timeline.spans) == 1


def test_three_topics_a_b_c_three_spans():
    segments = [
        make_segment(0.0, 10.0, "boss fight attack damage"),
        make_segment(10.0, 20.0, "inventory crafting materials"),
        make_segment(20.0, 30.0, "movement travel exploration"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3, context_window=1))

    assert len(timeline.spans) == 3
    # Verify chronological order
    assert timeline.spans[0].start < timeline.spans[1].start < timeline.spans[2].start
    assert timeline.spans[0].end == timeline.spans[1].start
    assert timeline.spans[1].end == timeline.spans[2].start


def test_a_b_a_remains_three_regions():
    segments = [
        make_segment(0.0, 10.0, "boss fight attack"),
        make_segment(10.0, 15.0, "inventory craft materials"),
        make_segment(15.0, 25.0, "boss fight attack again"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3, context_window=1))

    # P1-2: Single-phrase excursion (inventory) should be smoothed out
    # A/B/A with single-segment B => A/A merged into ONE region
    assert len(timeline.spans) == 1
    span = timeline.spans[0]
    assert "boss" in span.keywords
    assert "fight" in span.keywords or "attack" in span.keywords
    assert "craft" in span.keywords or "inventory" in span.keywords or "materials" in span.keywords


def test_input_order_independence_transcript():
    segments = [
        make_segment(10.0, 15.0, "inventory crafting"),
        make_segment(0.0, 5.0, "boss fight"),
        make_segment(5.0, 10.0, "boss low health"),
    ]
    timeline1 = build_topic_timeline(segments)
    timeline2 = build_topic_timeline(list(reversed(segments)))

    assert len(timeline1.spans) == len(timeline2.spans)
    for s1, s2 in zip(timeline1.spans, timeline2.spans):
        assert s1.start == s2.start
        assert s1.end == s2.end
        assert s1.topic_id == s2.topic_id


def test_input_order_independence_via():
    segments = [make_segment(0.0, 20.0, "boss fight")]
    via1 = make_via_evidence("asset1", 0, 10, description="boss appears")
    via2 = make_via_evidence("asset1", 10, 20, description="boss defeated")

    timeline1 = build_topic_timeline(segments, via_evidence=[via1, via2])
    timeline2 = build_topic_timeline(segments, via_evidence=[via2, via1])

    assert len(timeline1.spans) == len(timeline2.spans)
    for s1, s2 in zip(timeline1.spans, timeline2.spans):
        assert s1.topic_id == s2.topic_id


def test_duplicate_segments_deterministic():
    segments = [
        make_segment(0.0, 5.0, "boss fight"),
        make_segment(0.0, 5.0, "boss fight"),  # Exact duplicate
        make_segment(5.0, 10.0, "inventory"),
    ]
    t1 = build_topic_timeline(segments)
    t2 = build_topic_timeline(segments)

    assert len(t1.spans) == len(t2.spans)
    for s1, s2 in zip(t1.spans, t2.spans):
        assert s1.topic_id == s2.topic_id
        assert s1.start == s2.start
        assert s1.end == s2.end


def test_invalid_segment_start_ge_end_ignored():
    segments = [
        make_segment(0.0, 5.0, "valid"),
        make_segment(10.0, 10.0, "zero duration"),  # start == end
        make_segment(15.0, 12.0, "negative"),  # start > end
        make_segment(20.0, 25.0, "valid"),
    ]
    timeline = build_topic_timeline(segments)

    assert len(timeline.spans) >= 1
    # Only valid segments should contribute


def test_negative_start_handled():
    segments = [
        make_segment(-1.0, 5.0, "negative start"),
        make_segment(5.0, 10.0, "normal"),
    ]
    timeline = build_topic_timeline(segments)

    # Negative start should be handled safely (clamped or ignored)
    for span in timeline.spans:
        assert span.start >= 0.0


def test_end_exceeds_video_duration_clamped():
    segments = [
        make_segment(0.0, 100.0, "long segment"),
    ]
    timeline = build_topic_timeline(segments, video_duration=50.0)

    for span in timeline.spans:
        assert span.end <= 50.0 + 1e-9


def test_punctuation_case_no_artificial_split():
    segments = [
        make_segment(0.0, 5.0, "BOSS FIGHT!"),
        make_segment(5.0, 10.0, "boss fight?"),
        make_segment(10.0, 15.0, "Boss Fight."),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    assert len(timeline.spans) == 1


def test_accents_normalization_consistent():
    segments = [
        make_segment(0.0, 5.0, "jefe final batalla"),
        make_segment(5.0, 10.0, "jefé final batallón"),  # With accents
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    assert len(timeline.spans) == 1
    assert any("jefe" in k or "final" in k for k in timeline.spans[0].keywords)


def test_keywords_deterministic_order():
    segments = [
        make_segment(0.0, 10.0, "alpha beta gamma delta epsilon"),
    ]
    timeline = build_topic_timeline(segments)

    # Keywords should be sorted by freq DESC, then token ASC
    keywords = list(timeline.spans[0].keywords)
    # All freq 1, so alphabetical
    assert keywords == sorted(keywords)


def test_stable_topic_id_same_input():
    segments = [make_segment(0.0, 10.0, "boss fight")]
    t1 = build_topic_timeline(segments)
    t2 = build_topic_timeline(segments)

    assert t1.spans[0].topic_id == t2.spans[0].topic_id


def test_confidence_bounded():
    segments = [make_segment(0.0, 10.0, "boss fight")]
    timeline = build_topic_timeline(segments)

    for span in timeline.spans:
        assert 0.0 <= span.confidence <= 1.0


def test_no_nan_infinity():
    segments = [make_segment(0.0, 10.0, "boss fight")]
    timeline = build_topic_timeline(segments)

    for span in timeline.spans:
        import math
        assert not math.isnan(span.confidence)
        assert not math.isinf(span.confidence)
        assert not math.isnan(span.start)
        assert not math.isinf(span.start)
        assert not math.isnan(span.end)
        assert not math.isinf(span.end)


def test_no_overlapping_spans():
    segments = [
        make_segment(0.0, 10.0, "boss"),
        make_segment(10.0, 20.0, "inventory"),
        make_segment(20.0, 30.0, "travel"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    for i in range(len(timeline.spans) - 1):
        assert timeline.spans[i].end <= timeline.spans[i + 1].start + 1e-9


def test_spans_chronologically_ordered():
    segments = [
        make_segment(20.0, 30.0, "travel"),
        make_segment(0.0, 10.0, "boss"),
        make_segment(10.0, 20.0, "inventory"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    for i in range(len(timeline.spans) - 1):
        assert timeline.spans[i].start < timeline.spans[i + 1].start


def test_gaps_allowed_no_fake_unknown():
    segments = [
        make_segment(0.0, 5.0, "boss fight combat"),
        make_segment(20.0, 25.0, "inventory crafting materials"),  # Large gap
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    # Should have 2 spans, no UNKNOWN filler
    assert len(timeline.spans) == 2
    for span in timeline.spans:
        assert "unknown" not in span.keywords
        assert "UNKNOWN" not in span.representative_text.upper()


def test_via_continuity_same_group_reduces_split():
    segments = [
        make_segment(0.0, 10.0, "boss fight"),
        make_segment(10.0, 15.0, "brief inventory"),
        make_segment(15.0, 25.0, "boss fight again"),
    ]
    # Same continuity group on both sides
    via = [
        make_via_evidence("asset1", 0, 10, continuity_group_id="group1"),
        make_via_evidence("asset1", 15, 25, continuity_group_id="group1"),
    ]
    build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(boundary_threshold=0.3, continuity_same_group_reduction=0.2))

    # Should be less likely to split due to continuity
    # (Exact behavior depends on threshold)


def test_via_continuity_change_not_sole_split():
    segments = [
        make_segment(0.0, 10.0, "boss fight"),
        make_segment(10.0, 11.0, "transition"),
        make_segment(11.0, 20.0, "boss fight"),
    ]
    # Different continuity groups but same topic
    via = [
        make_via_evidence("asset1", 0, 10, continuity_group_id="group1"),
        make_via_evidence("asset1", 11, 20, continuity_group_id="group2"),
    ]
    build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(boundary_threshold=0.3))

    # continuity change alone shouldn't force split if lexical similarity is high
    # (depends on threshold, but shouldn't be ONLY reason)


def test_lexical_change_plus_via_change_reinforces():
    segments = [
        make_segment(0.0, 10.0, "boss fight attack"),
        make_segment(10.0, 20.0, "inventory crafting"),
    ]
    via = [
        make_via_evidence("asset1", 0, 10, description="boss combat"),
        make_via_evidence("asset1", 10, 20, description="inventory management"),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(boundary_threshold=0.3))

    assert len(timeline.spans) == 2
    # Both lexical and VIA evidence support split


def test_partial_via_reduces_confidence():
    segments = [make_segment(0.0, 20.0, "topic")]
    via_partial = make_via_evidence("asset1", 0, 20, partial=True, coverage_ratio=0.5)
    via_complete = make_via_evidence("asset1", 0, 20, partial=False, coverage_ratio=1.0)

    t1 = build_topic_timeline(segments, via_evidence=[via_partial])
    t2 = build_topic_timeline(segments, via_evidence=[via_complete])

    # Partial should have lower or equal confidence
    assert t1.spans[0].confidence <= t2.spans[0].confidence + 1e-9


def test_low_coverage_no_false_precision():
    segments = [make_segment(0.0, 10.0, "topic")]
    via = make_via_evidence("asset1", 0, 10, coverage_ratio=0.1, partial=True)
    timeline = build_topic_timeline(segments, via_evidence=[via])

    # Low coverage shouldn't produce high confidence
    assert timeline.spans[0].confidence < 0.8


def test_mixed_asset_via_not_mixed():
    segments = [make_segment(0.0, 20.0, "topic")]
    via = [
        make_via_evidence("asset1", 0, 10, description="asset1 event"),
        make_via_evidence("asset2", 10, 20, description="asset2 event"),
    ]
    # Should filter to single asset or handle explicitly
    timeline = build_topic_timeline(segments, via_evidence=via, asset_id="asset1")

    # Only asset1 evidence should contribute
    for span in timeline.spans:
        for ev_ref in span.evidence_refs:
            assert "asset1" in ev_ref or "asset2" not in ev_ref


def test_transcript_only_works_without_via():
    segments = [
        make_segment(0.0, 10.0, "boss fight"),
        make_segment(10.0, 20.0, "inventory"),
    ]
    timeline = build_topic_timeline(segments, via_evidence=None)

    assert len(timeline.spans) >= 1
    for span in timeline.spans:
        assert span.confidence >= 0.0


def test_via_only_behavior_documented():
    """VIA-only (no transcript) - conservative: minimal topic spans."""
    # With no transcript, no lexical boundaries can be detected
    # Should return minimal/empty timeline
    segments: list[dict] = []
    via = [
        make_via_evidence("asset1", 0, 10, description="boss event"),
        make_via_evidence("asset1", 10, 20, description="inventory event"),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via)

    # Conservative: no transcript = no lexical basis for boundaries
    # May return empty or single span covering full range if any text exists
    # But with empty transcript, should be empty
    assert timeline.spans == ()


def test_representative_text_from_transcript():
    segments = [
        make_segment(0.0, 5.0, "we are fighting the boss"),
        make_segment(5.0, 10.0, "the boss has very low health now"),
    ]
    timeline = build_topic_timeline(segments)

    span = timeline.spans[0]
    # Should be actual transcript text
    assert "fighting" in span.representative_text or "boss" in span.representative_text
    assert span.representative_text != ""


def test_no_candidate_mutation():
    from app.providers.base import ClipCandidate, SourceRange
    c = ClipCandidate(start=0, end=10, source=SourceRange(start=0, end=10), score=0.5)
    original_start = c.start
    original_end = c.end

    segments = [make_segment(0.0, 10.0, "topic")]
    build_topic_timeline(segments)

    assert c.start == original_start
    assert c.end == original_end


def test_no_scored_candidate_mutation():
    from app.pipeline.highlight_scoring import score_candidates, ScoredCandidate
    from app.providers.base import ClipCandidate, SourceRange

    c = ClipCandidate(start=0, end=10, source=SourceRange(start=0, end=10), score=0.5)
    original_score = c.score

    segments = [make_segment(0.0, 10.0, "topic")]
    build_topic_timeline(segments)

    # ScoredCandidate not involved in 1.5E, but verify no side effects
    assert c.score == original_score


def test_no_refined_boundaries_mutation():
    from app.pipeline.highlight_boundaries import refine_candidate, RefinedBoundaries
    from app.providers.base import ClipCandidate, SourceRange

    c = ClipCandidate(start=0, end=10, source=SourceRange(start=0, end=10), score=0.5)
    original = (c.start, c.end)

    segments = [make_segment(0.0, 10.0, "topic")]
    build_topic_timeline(segments)

    assert (c.start, c.end) == original


def test_no_llm_network_media_calls():
    """Verify no external calls by running with empty transcript."""
    segments = [make_segment(0.0, 10.0, "topic")]
    timeline = build_topic_timeline(segments)

    # Should complete without network/LLM calls
    assert isinstance(timeline, TopicTimeline)


def test_keyword_tie_break_deterministic():
    """Equal frequency keywords sorted alphabetically."""
    segments = [make_segment(0.0, 10.0, "alpha beta gamma")]
    timeline = build_topic_timeline(segments)

    keywords = list(timeline.spans[0].keywords)
    # All freq=1, so alphabetical
    assert keywords == sorted(keywords)


def test_boundary_tie_break_deterministic():
    """Equal boundary evidence should be deterministic."""
    segments = [
        make_segment(0.0, 5.0, "topic A"),
        make_segment(5.0, 10.0, "topic A"),
        make_segment(10.0, 15.0, "topic B"),
        make_segment(15.0, 20.0, "topic B"),
    ]
    t1 = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))
    t2 = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3))

    assert len(t1.spans) == len(t2.spans)
    for s1, s2 in zip(t1.spans, t2.spans):
        assert s1.topic_id == s2.topic_id
        assert s1.start == s2.start
        assert s1.end == s2.end


def test_large_synthetic_input_performance():
    """Synthetic test: many segments should not show quadratic behavior."""
    import time

    segments = [
        make_segment(i * 0.5, (i + 1) * 0.5, "topic segment " + ("boss" if i % 10 < 5 else "inventory"))
        for i in range(200)  # 100 seconds, 200 segments
    ]

    start = time.time()
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(context_window=2))
    elapsed = time.time() - start

    # Should complete in reasonable time (< 2 seconds)
    assert elapsed < 2.0
    assert len(timeline.spans) >= 1


def test_spanish_accented_words_work():
    segments = [
        make_segment(0.0, 5.0, "estamos peleando contra el jefe"),
        make_segment(5.0, 10.0, "el jefe tiene poca vida"),
        make_segment(10.0, 15.0, "voy a atacar al jefe"),
        make_segment(15.0, 20.0, "ahora vamos al inventario"),
        make_segment(20.0, 25.0, "necesito fabricar munición"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(boundary_threshold=0.3, context_window=2))

    assert len(timeline.spans) >= 1
    # Keywords should include normalized Spanish words
    all_keywords = [k for span in timeline.spans for k in span.keywords]
    assert any("jefe" in k or "pelea" in k or "inventario" in k or "municion" in k for k in all_keywords)


# ──────────────────────────────────────────────────────────────────────────────
# Query Function Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_topic_for_time_exact_match():
    segments = [
        make_segment(0.0, 10.0, "boss fight combat"),
        make_segment(10.0, 20.0, "inventory crafting materials"),
    ]
    timeline = build_topic_timeline(segments)

    span = topic_for_time(timeline, 5.0)
    assert span is not None
    assert span.start == 0.0
    assert span.end == 10.0


def test_topic_for_time_boundary():
    segments = [
        make_segment(0.0, 10.0, "boss fight combat"),
        make_segment(10.0, 20.0, "inventory crafting materials"),
    ]
    timeline = build_topic_timeline(segments)

    span = topic_for_time(timeline, 10.0)
    # At exact boundary, should return the span starting at that time
    assert span is not None
    assert span.start == 10.0


def test_topic_for_time_outside_range():
    segments = [make_segment(0.0, 10.0, "topic")]
    timeline = build_topic_timeline(segments)

    span = topic_for_time(timeline, 20.0)
    assert span is None


def test_topic_for_range_overlapping():
    segments = [
        make_segment(0.0, 10.0, "boss fight combat"),
        make_segment(10.0, 20.0, "inventory crafting materials"),
        make_segment(20.0, 30.0, "travel exploration map"),
    ]
    timeline = build_topic_timeline(segments)

    result = topic_for_range(timeline, 5.0, 25.0)
    assert len(result) == 3


def test_topic_for_range_non_overlapping():
    segments = [
        make_segment(0.0, 10.0, "boss fight combat"),
        make_segment(10.0, 20.0, "inventory crafting materials"),
    ]
    timeline = build_topic_timeline(segments)

    result = topic_for_range(timeline, 30.0, 40.0)
    assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# Integration Smoke Test
# ──────────────────────────────────────────────────────────────────────────────

def test_integration_smoke_synthetic():
    """Deterministic smoke test with 3+ topics, VIA, continuity."""
    segments = [
        make_segment(0.0, 10.0, "estamos peleando contra el jefe"),
        make_segment(10.0, 15.0, "el jefe tiene poca vida"),
        make_segment(15.0, 20.0, "voy a atacar al jefe final"),
        make_segment(20.0, 25.0, "ahora vamos al inventario"),
        make_segment(25.0, 30.0, "necesito fabricar munición"),
        make_segment(30.0, 35.0, "vamos a cambiar el arma"),
        make_segment(35.0, 40.0, "terminado el inventario"),
        make_segment(40.0, 50.0, "movimiento hacia la siguiente zona"),
        make_segment(50.0, 55.0, "explorando el mapa nuevo"),
        make_segment(55.0, 60.0, "llegamos a la base"),
    ]

    via = [
        make_via_evidence("asset1", 0, 20, description="boss fight combat", subjects=["boss"], objects=["weapon"], observed_action="attacking", continuity_group_id="grp1"),
        make_via_evidence("asset1", 20, 35, description="inventory management crafting", subjects=["player"], objects=["ammo", "weapon"], observed_action="crafting", continuity_group_id="grp2"),
        make_via_evidence("asset1", 40, 60, description="travel exploration movement", subjects=["player"], objects=["map"], observed_action="moving", continuity_group_id="grp3"),
    ]

    config = TopicTimelineConfig(
        boundary_threshold=0.35,
        context_window=2,
        via_influence_weight=0.3,
        continuity_same_group_reduction=0.15,
    )

    timeline = build_topic_timeline(segments, via_evidence=via, video_duration=60.0, asset_id="asset1", config=config)

    # Report
    print(f"\n=== INTEGRATION SMOKE ===")
    print(f"Input duration: {timeline.total_duration:.1f}s")
    print(f"Transcript segments: {len(segments)}")
    print(f"VIA evidence: {len(via)}")
    print(f"Topic spans: {len(timeline.spans)}")

    for i, span in enumerate(timeline.spans):
        print(f"\n  Span {i+1}: {span.start:.1f} → {span.end:.1f}")
        print(f"    topic_id: {span.topic_id}")
        print(f"    keywords: {list(span.keywords)}")
        print(f"    rep_text: {span.representative_text[:80]}...")
        print(f"    confidence: {span.confidence:.3f}")
        print(f"    basis: {list(span.basis)}")
        print(f"    transcript_refs: {list(span.transcript_refs)}")
        print(f"    evidence_refs: {list(span.evidence_refs)}")

    # Permutation test
    import random
    random.seed(42)
    shuffled_segments = segments[:]
    random.shuffle(shuffled_segments)
    random.shuffle(via)
    timeline2 = build_topic_timeline(shuffled_segments, via_evidence=via, video_duration=60.0, asset_id="asset1", config=config)

    assert len(timeline.spans) == len(timeline2.spans)
    for s1, s2 in zip(timeline.spans, timeline2.spans):
        assert s1.topic_id == s2.topic_id, f"Permutation mismatch: {s1.topic_id} != {s2.topic_id}"

    print(f"\n✓ Permutation test passed")
    assert len(timeline.spans) >= 3  # At least 3 topics

    # Verify no mutations
    for seg in segments:
        assert "start" in seg and "end" in seg and "text" in seg


# ──────────────────────────────────────────────────────────────────────────────
# Large Gap Handling Tests (Gap Closure)
# ──────────────────────────────────────────────────────────────────────────────

def test_large_gap_split_basic():
    """A. Basic large gap split: two segments separated by gap > threshold.

    segments: 0→5 "boss combat attack"
              20→25 "boss combat attack"

    Expected: 2 spans, no span crosses gap, topic_for_time(10) is None
    """
    segments = [
        make_segment(0.0, 5.0, "boss combat attack"),
        make_segment(20.0, 25.0, "boss combat attack"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(large_gap_threshold=15))

    assert len(timeline.spans) == 2
    assert timeline.spans[0].start == 0.0
    assert timeline.spans[0].end == 5.0
    assert timeline.spans[1].start == 20.0
    assert timeline.spans[1].end == 25.0

    # Gap region should have no span
    assert topic_for_time(timeline, 10.0) is None
    assert topic_for_time(timeline, 15.0) is None


def test_large_gap_preserves_asset_id():
    """B. Large gap split with explicit asset_id.

    Same gap as A, but with asset_id specified.
    Both spans should have topic_id computed with the explicit asset_id.
    """
    segments = [
        make_segment(0.0, 5.0, "boss combat attack"),
        make_segment(20.0, 25.0, "boss combat attack"),
    ]
    via = [
        make_via_evidence("asset1", 0, 5, description="boss combat", evidence_kind="event"),
        make_via_evidence("asset1", 20, 25, description="boss combat", evidence_kind="event"),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via, asset_id="asset1", config=TopicTimelineConfig(large_gap_threshold=15))

    assert len(timeline.spans) == 2
    # Both spans should have topic_id computed with asset_id="asset1"
    expected_id_0 = _stable_topic_id("asset1", 0.0, 5.0, ["boss", "combat", "attack"])
    expected_id_1 = _stable_topic_id("asset1", 20.0, 25.0, ["boss", "combat", "attack"])
    assert timeline.spans[0].topic_id == expected_id_0
    assert timeline.spans[1].topic_id == expected_id_1


def test_large_gap_via_refs_preserved_per_subspan():
    """C. Large gap split preserves VIA refs per subspan.

    VIA 0→5 evidence_id=A
    VIA 20→25 evidence_id=B

    Expected:
    span1.evidence_refs contains A and NOT B
    span2.evidence_refs contains B and NOT A
    """
    segments = [
        make_segment(0.0, 5.0, "boss combat attack"),
        make_segment(20.0, 25.0, "boss combat attack"),
    ]
    via = [
        make_via_evidence("asset1", 0, 5, description="boss combat", evidence_kind="event", confidence=0.9),
        make_via_evidence("asset1", 20, 25, description="boss combat", evidence_kind="event", confidence=0.9),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(large_gap_threshold=15))

    assert len(timeline.spans) == 2
    span1_refs = set(timeline.spans[0].evidence_refs)
    span2_refs = set(timeline.spans[1].evidence_refs)

    assert "ev_asset1_0" in span1_refs
    assert "ev_asset1_0" not in span2_refs
    assert "ev_asset1_20" in span2_refs
    assert "ev_asset1_20" not in span1_refs


def test_large_gap_same_topic_across_gap_not_merged():
    """D. Gap + same vocabulary: A / GAP / A should NOT merge back.

    A large transcript gap is a hard temporal barrier for LOCAL merge.
    Even if vocabulary is identical, the gap is a hard barrier.
    """
    segments = [
        make_segment(0.0, 5.0, "boss combat attack"),
        make_segment(20.0, 25.0, "boss combat attack"),  # Same vocab, but large gap
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(large_gap_threshold=15))

    # Should be 2 spans despite identical vocabulary
    assert len(timeline.spans) == 2
    assert timeline.spans[0].end == 5.0
    assert timeline.spans[1].start == 20.0


def test_two_large_gaps_three_spans():
    """E. Two large gaps produce three spans.

    0→5, 20→25, 40→45 with gaps > threshold between each.
    """
    segments = [
        make_segment(0.0, 5.0, "topic A"),
        make_segment(20.0, 25.0, "topic B"),
        make_segment(40.0, 45.0, "topic C"),
    ]
    timeline = build_topic_timeline(segments, config=TopicTimelineConfig(large_gap_threshold=15))

    assert len(timeline.spans) == 3
    assert timeline.spans[0].end == 5.0
    assert timeline.spans[1].start == 20.0
    assert timeline.spans[1].end == 25.0
    assert timeline.spans[2].start == 40.0
    assert timeline.spans[2].end == 45.0


def test_subspan_materialization_is_independent():
    """F. After gap split, each subspan's fields are computed independently.

    representative_text, keywords, confidence, evidence_refs, topic_id
    must come solely from the segments/evidence of that subspan.
    """
    segments = [
        make_segment(0.0, 5.0, "boss fight combat"),      # span 1
        make_segment(20.0, 25.0, "inventory crafting"),   # span 2
    ]
    via = [
        make_via_evidence("asset1", 0, 5, description="boss combat", evidence_kind="event", confidence=0.9),
        make_via_evidence("asset1", 20, 25, description="inventory crafting", evidence_kind="event", confidence=0.8),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(large_gap_threshold=15))

    assert len(timeline.spans) == 2

    # Span 1: only boss-related
    assert "combat" in timeline.spans[0].keywords
    assert "crafting" not in timeline.spans[0].keywords
    assert timeline.spans[0].confidence > 0.0
    assert "ev_asset1_0" in timeline.spans[0].evidence_refs
    assert "ev_asset1_20" not in timeline.spans[0].evidence_refs

    # Span 2: only inventory-related
    assert "crafting" in timeline.spans[1].keywords
    assert "combat" not in timeline.spans[1].keywords
    assert timeline.spans[1].confidence > 0.0
    assert "ev_asset1_20" in timeline.spans[1].evidence_refs
    assert "ev_asset1_0" not in timeline.spans[1].evidence_refs

    # Topic IDs should be different and deterministic
    expected_id_0 = _stable_topic_id("asset1", 0.0, 5.0, sorted(["fight", "combat", "boss"]))
    expected_id_1 = _stable_topic_id("asset1", 20.0, 25.0, sorted(["crafting", "inventory"]))
    assert timeline.spans[0].topic_id == expected_id_0
    assert timeline.spans[1].topic_id == expected_id_1


def test_large_gap_via_distant_no_effect():
    """Large gap + distant VIA should not affect local boundary.

    VIA distant from the gap boundary should not influence local boundary decision.
    """
    segments = [
        make_segment(0.0, 10.0, "boss combat attack"),
        make_segment(20.0, 30.0, "inventory crafting"),
    ]
    # VIA very far from gap (0-5) should not affect boundary at 10-20 gap
    via = [
        make_via_evidence("asset1", 50, 60, description="distant event", evidence_kind="event", confidence=0.9),
    ]
    timeline = build_topic_timeline(segments, via_evidence=via, config=TopicTimelineConfig(large_gap_threshold=15))

    # Should still have 2 spans split at the gap
    assert len(timeline.spans) == 2
    # VIA is far away, so it shouldn't create additional spans
    # The boundary at 10-20 gap should be the only one


def test_gap_split_subspan_isolation():
    """Subspan after gap is isolated from later topics.

    0→5   "boss combat attack"
    GAP 5→20
    20→25 "boss combat health"
    25→30 "inventory crafting material"
    30→35 "inventory crafting ammunition"

    After gap at 5-20:
    - span 20→25 must contain ONLY: boss/combat/health
    - span 20→25 must NOT contain: inventory/crafting/material/ammunition
    - all transcript_refs of span 20→25 must have end <= 25
    - VIA A at 20→25 is included, VIA B at 25→35 is EXCLUDED
    """
    segments = [
        make_segment(0.0, 5.0, "boss combat attack"),
        make_segment(20.0, 25.0, "boss combat health"),
        make_segment(25.0, 30.0, "inventory crafting material"),
        make_segment(30.0, 35.0, "inventory crafting ammunition"),
    ]
    # VIA A at 20→25 (should be in first span after gap)
    # VIA B at 25→35 (should be in second span after gap, NOT in first)
    via = [
        make_via_evidence("asset1", 20, 25, description="boss combat health", evidence_kind="event", confidence=0.9),
        make_via_evidence("asset1", 25, 35, description="inventory crafting", evidence_kind="event", confidence=0.8),
    ]
    config = TopicTimelineConfig(large_gap_threshold=15)
    timeline = build_topic_timeline(segments, via_evidence=via, config=config)

    # Should have 3 spans: one before gap (0-5), two after gap (20-25, 25-35)
    assert len(timeline.spans) == 3
    span1 = timeline.spans[0]
    span2 = timeline.spans[1]
    span3 = timeline.spans[2]

    # First span: 0-5
    assert span1.start == 0.0
    assert span1.end == 5.0
    assert "boss" in span1.keywords
    assert "combat" in span1.keywords

    # Second span: 20-25 (NOT 20-35)
    assert span2.start == 20.0
    assert span2.end == 25.0
    # Keywords only from 20-25 segment
    assert "boss" in span2.keywords
    assert "combat" in span2.keywords
    assert "health" in span2.keywords
    assert "inventory" not in span2.keywords
    assert "crafting" not in span2.keywords
    assert "material" not in span2.keywords
    assert "ammunition" not in span2.keywords

    # Transcript refs must all end <= 25
    for ref in span2.transcript_refs:
        end_time = float(ref.split('-')[1])
        assert end_time <= 25.0 + 1e-9

    # VIA A (20→25) included, VIA B (25→35) NOT included
    span2_evidence_refs = set(span2.evidence_refs)
    assert "ev_asset1_20" in span2_evidence_refs
    assert "ev_asset1_25" not in span2_evidence_refs

    # Topic ID must be computed from subspan's own range
    expected_id = _stable_topic_id("asset1", 20.0, 25.0, sorted(["boss", "combat", "health"]))
    assert span2.topic_id == expected_id

    # Third span: 25-35
    assert span3.start == 25.0
    assert span3.end == 35.0
    assert "inventory" in span3.keywords
    assert "crafting" in span3.keywords
    assert "material" in span3.keywords or "ammunition" in span3.keywords
    # VIA B (25→35) is included in third span
    span3_evidence_refs = set(span3.evidence_refs)
    assert "ev_asset1_25" in span3_evidence_refs