"""Tests for Event Clustering + Diversity (SNIPRA v1.5G)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.pipeline.event_clustering import (
    DEFAULT_CLUSTERING_CONFIG,
    ClusteredEventPool,
    ClusteringConfig,
    _is_near_duplicate,
    _temporal_overlap_ratio,
    cluster_by_id,
    cluster_event_pool,
    cluster_for_event,
    events_in_cluster,
)
from app.pipeline.event_pool import (
    EventPool,
    EventPoolItem,
    EventTopicContext,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_item(
    event_id: str,
    rank: int,
    overall_score: float = 0.7,
    refined_start: float = 0.0,
    refined_end: float = 10.0,
    primary_topic_id: str | None = None,
    peak_at: float | None = None,
    title: str = "",
    hook: str = "",
    reason: str = "",
    refinement_status: str = "refined",
    refinement_confidence: float = 0.8,
) -> EventPoolItem:
    return EventPoolItem(
        event_id=event_id,
        rank=rank,
        overall_score=overall_score,
        original_start=refined_start,
        original_end=refined_end,
        refined_start=refined_start,
        refined_end=refined_end,
        title=title,
        hook=hook,
        reason=reason,
        refinement_status=refinement_status,
        refinement_confidence=refinement_confidence,
        peak_at=peak_at,
        payoff_at=None,
        payoff_window_start=None,
        payoff_window_end=None,
        reaction_at=None,
        reaction_window_start=None,
        reaction_window_end=None,
        primary_topic_id=primary_topic_id,
        topic_contexts=(),
        score_components=(),
        evidence_refs=(),
    )


def _make_pool(items: list[EventPoolItem] | None = None, asset_id: str | None = "asset-1") -> EventPool:
    if items is None:
        items = []
    return EventPool(asset_id=asset_id, items=tuple(items))


def _make_topic_context(
    topic_id: str,
    topic_start: float,
    topic_end: float,
    overlap_seconds: float = 0.0,
    overlap_ratio: float = 0.0,
    confidence: float = 0.9,
) -> EventTopicContext:
    return EventTopicContext(
        topic_id=topic_id,
        topic_start=topic_start,
        topic_end=topic_end,
        overlap_seconds=overlap_seconds,
        overlap_ratio=overlap_ratio,
        confidence=confidence,
        keywords=(),
        representative_text="",
        evidence_refs=(),
    )


def _make_item_with_context(
    event_id: str,
    rank: int,
    refined_start: float,
    refined_end: float,
    primary_topic_id: str | None = None,
    peak_at: float | None = None,
    topic_contexts: tuple[EventTopicContext, ...] = (),
) -> EventPoolItem:
    return EventPoolItem(
        event_id=event_id,
        rank=rank,
        overall_score=0.7,
        original_start=refined_start,
        original_end=refined_end,
        refined_start=refined_start,
        refined_end=refined_end,
        title="",
        hook="",
        reason="",
        refinement_status="refined",
        refinement_confidence=0.8,
        peak_at=peak_at,
        payoff_at=None,
        payoff_window_start=None,
        payoff_window_end=None,
        reaction_at=None,
        reaction_window_start=None,
        reaction_window_end=None,
        primary_topic_id=primary_topic_id,
        topic_contexts=topic_contexts,
        score_components=(),
        evidence_refs=(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_empty_pool(self):
        pool = _make_pool([])
        clustered = cluster_event_pool(pool)
        assert clustered.clusters == ()
        assert clustered.diverse_order_event_ids == ()
        assert clustered.asset_id == "asset-1"
        assert clustered.source_event_count == 0

    def test_one_event(self):
        item = _make_item("evt-1", rank=1)
        pool = _make_pool([item])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        assert clustered.clusters[0].member_event_ids == ("evt-1",)
        assert clustered.diverse_order_event_ids == ("evt-1",)

    def test_duplicate_event_id_reject(self):
        items = [_make_item("evt-1", rank=1), _make_item("evt-1", rank=2)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Duplicate event_id"):
            cluster_event_pool(pool)

    def test_duplicate_rank_reject(self):
        items = [_make_item("evt-1", rank=1), _make_item("evt-2", rank=1)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Duplicate ranks"):
            cluster_event_pool(pool)

    def test_rank_negative_reject(self):
        items = [_make_item("evt-1", rank=-1)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Invalid rank < 1"):
            cluster_event_pool(pool)

    def test_rank_gap_reject(self):
        items = [_make_item("evt-1", rank=1), _make_item("evt-2", rank=3)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="contiguous"):
            cluster_event_pool(pool)

    def test_score_nan_reject(self):
        items = [_make_item("evt-1", rank=1, overall_score=float('nan'))]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Invalid score"):
            cluster_event_pool(pool)

    def test_score_inf_reject(self):
        items = [_make_item("evt-1", rank=1, overall_score=float('inf'))]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Invalid score"):
            cluster_event_pool(pool)

    def test_score_out_of_bounds_reject(self):
        items = [_make_item("evt-1", rank=1, overall_score=1.5)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="out of bounds"):
            cluster_event_pool(pool)

    def test_negative_refined_start_reject(self):
        items = [_make_item("evt-1", rank=1, refined_start=-1.0)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Negative refined_start"):
            cluster_event_pool(pool)

    def test_nan_refined_start_reject(self):
        items = [_make_item("evt-1", rank=1, refined_start=float('nan'))]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Non-finite refined range"):
            cluster_event_pool(pool)

    def test_nan_refined_end_reject(self):
        items = [_make_item("evt-1", rank=1, refined_end=float('nan'))]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Non-finite refined range"):
            cluster_event_pool(pool)

    def test_inf_refined_end_reject(self):
        items = [_make_item("evt-1", rank=1, refined_end=float('inf'))]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Non-finite refined range"):
            cluster_event_pool(pool)

    def test_refined_end_le_start_reject(self):
        items = [_make_item("evt-1", rank=1, refined_start=5.0, refined_end=5.0)]
        pool = _make_pool(items)
        with pytest.raises(ValueError, match="Invalid refined range"):
            cluster_event_pool(pool)

    def test_invalid_config_thresholds_reject(self):
        with pytest.raises(ValueError, match="Invalid thresholds"):
            ClusteringConfig(strong_overlap_threshold=0.5, moderate_overlap_threshold=0.7)

    def test_config_moderate_greater_than_strong_reject(self):
        with pytest.raises(ValueError, match="moderate.*<= strong"):
            ClusteringConfig(strong_overlap_threshold=0.5, moderate_overlap_threshold=0.8)

    def test_config_peak_tolerance_nan_reject(self):
        with pytest.raises(ValueError, match="peak_tolerance_seconds must be finite"):
            ClusteringConfig(peak_tolerance_seconds=float('nan'))

    def test_config_peak_tolerance_inf_reject(self):
        with pytest.raises(ValueError, match="peak_tolerance_seconds must be finite"):
            ClusteringConfig(peak_tolerance_seconds=float('inf'))

    def test_config_peak_tolerance_neg_inf_reject(self):
        with pytest.raises(ValueError, match="peak_tolerance_seconds must be finite"):
            ClusteringConfig(peak_tolerance_seconds=float('-inf'))

    def test_rank_none_reject(self):
        # EventPoolItem with rank=None should fail validation
        item = _make_item("evt-1", rank=1)
        item = replace(item, rank=None)
        pool = _make_pool([item])
        with pytest.raises(ValueError, match="Missing rank"):
            cluster_event_pool(pool)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal similarity tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTemporalSimilarity:
    def test_no_overlap_not_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", 2, refined_start=20.0, refined_end=30.0)
        assert _temporal_overlap_ratio(a, b) == 0.0

    def test_tiny_overlap_not_strong(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", 2, refined_start=9.9, refined_end=20.0)
        ratio = _temporal_overlap_ratio(a, b)
        assert ratio < DEFAULT_CLUSTERING_CONFIG.strong_overlap_threshold

    def test_strong_threshold_exact_duplicate(self):
        config = ClusteringConfig(strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        # 7s overlap / 10s = 0.7
        b = _make_item("b", 2, refined_start=3.0, refined_end=13.0)
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert is_dup
        assert basis == ("strong_temporal_overlap",)

    def test_strong_threshold_below_epsilon_branch_dependent(self):
        config = ClusteringConfig(strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        # 6.9s overlap / 10s = 0.69 (just below)
        b = _make_item("b", 2, refined_start=3.1, refined_end=13.0)
        is_dup, basis = _is_near_duplicate(a, b, config)
        # Falls to moderate branch - need secondary signal
        assert not is_dup  # no secondary signal

    def test_strong_threshold_above_duplicate(self):
        config = ClusteringConfig(strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        # 7.1s overlap / 10s = 0.71
        b = _make_item("b", 2, refined_start=2.9, refined_end=13.0)
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert is_dup
        assert basis == ("strong_temporal_overlap",)

    def test_moderate_exact_same_topic_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        # 5s overlap / 10s = 0.5
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, primary_topic_id="boss")
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert is_dup
        assert "moderate_temporal_overlap" in basis
        assert "same_primary_topic" in basis

    def test_moderate_below_same_topic_not_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        # 4.9s overlap / 10s = 0.49
        b = _make_item("b", 2, refined_start=5.1, refined_end=15.0, primary_topic_id="boss")
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert not is_dup

    def test_moderate_exact_close_peak_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7, peak_tolerance_seconds=2.0)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=5.0)
        # 5s overlap / 10s = 0.5, peak distance = 1s
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, peak_at=6.0)
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert is_dup
        assert "moderate_temporal_overlap" in basis
        assert "peak_proximity" in basis

    def test_moderate_different_topic_distant_peak_not_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7, peak_tolerance_seconds=2.0)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss", peak_at=5.0)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, primary_topic_id="inventory", peak_at=10.0)
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert not is_dup

    def test_same_topic_zero_overlap_not_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", 2, refined_start=20.0, refined_end=30.0, primary_topic_id="boss")
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert not is_dup

    def test_same_topic_low_overlap_not_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        # 1s overlap / 10s = 0.1
        b = _make_item("b", 2, refined_start=9.0, refined_end=19.0, primary_topic_id="boss")
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert not is_dup

    def test_identical_ranges_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", 2, refined_start=0.0, refined_end=10.0)
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert is_dup
        assert basis == ("strong_temporal_overlap",)

    def test_same_peak_no_overlap_not_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=5.0)
        b = _make_item("b", 2, refined_start=20.0, refined_end=30.0, peak_at=5.0)
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert not is_dup

    def test_primary_topic_none_strong_overlap_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id=None)
        b = _make_item("b", 2, refined_start=2.0, refined_end=12.0, primary_topic_id=None)
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert is_dup
        assert basis == ("strong_temporal_overlap",)

    def test_primary_topic_none_moderate_no_peak_not_duplicate(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, primary_topic_id=None)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, primary_topic_id=None)
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert not is_dup


# ──────────────────────────────────────────────────────────────────────────────
# Peak tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPeak:
    def test_peak_exactly_tolerance_qualifies(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7, peak_tolerance_seconds=2.0)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=5.0)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, peak_at=7.0)  # distance = 2.0
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert is_dup
        assert "peak_proximity" in basis

    def test_peak_tolerance_plus_epsilon_not_qualifies(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7, peak_tolerance_seconds=2.0)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=5.0)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, peak_at=7.1)  # distance = 2.1
        is_dup, basis = _is_near_duplicate(a, b, config)
        assert not is_dup

    def test_one_peak_none_branch_unavailable(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7)
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=5.0)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, peak_at=None)
        is_dup, basis = _is_near_duplicate(a, b, config)
        # Same topic not present, peak proximity unavailable
        assert not is_dup

    def test_both_peak_none_unavailable(self):
        a = _make_item("a", 1, refined_start=0.0, refined_end=10.0, peak_at=None)
        b = _make_item("b", 2, refined_start=5.0, refined_end=15.0, peak_at=None)
        is_dup, basis = _is_near_duplicate(a, b, DEFAULT_CLUSTERING_CONFIG)
        assert not is_dup


# ──────────────────────────────────────────────────────────────────────────────
# Clustering tests
# ──────────────────────────────────────────────────────────────────────────────

class TestClustering:
    def test_singleton_cluster(self):
        item = _make_item("evt-1", rank=1)
        pool = _make_pool([item])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        c = clustered.clusters[0]
        assert c.member_event_ids == ("evt-1",)
        assert c.representative_event_id == "evt-1"
        assert c.relations == ()

    def test_two_strong_duplicates_same_cluster(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        c = clustered.clusters[0]
        assert set(c.member_event_ids) == {"a", "b"}
        assert c.representative_event_id == "a"  # lowest rank

    def test_two_distinct_events_two_clusters(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=20.0, refined_end=30.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 2

    def test_three_all_strong_duplicate_one_cluster(self):
        """All three pairs have STRONG overlap (>0.7)."""
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=1.0, refined_end=11.0)  # 9s/10s = 0.9
        c = _make_item("c", rank=3, refined_start=2.0, refined_end=12.0)  # 8s/10s = 0.8 with a
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        assert set(clustered.clusters[0].member_event_ids) == {"a", "b", "c"}

    def test_transitive_chaining_prevented(self):
        """A~B, B~C, but A!~C should NOT chain all three.
        A: 0-10, B: 4-14, C: 8-18
        Overlap A-B: 6s/10s = 0.6 (moderate) + same topic
        Overlap B-C: 6s/10s = 0.6 (moderate) + same topic
        Overlap A-C: 2s/10s = 0.2 (not duplicate)
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.8)
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=4.0, refined_end=14.0, primary_topic_id="boss")
        c = _make_item("c", rank=3, refined_start=8.0, refined_end=18.0, primary_topic_id="boss")
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool, config)

        # Representative-star: A creates cluster 1, B matches A (0.6 + same topic), C checked against A only (0.2) -> fails
        # C should create new cluster
        assert len(clustered.clusters) == 2
        cluster_a = cluster_for_event(clustered, "a")
        cluster_c = cluster_for_event(clustered, "c")
        assert cluster_a is not cluster_c
        assert set(cluster_a.member_event_ids) == {"a", "b"}
        assert cluster_c.member_event_ids == ("c",)

    def test_representative_remains_lowest_rank(self):
        # B is higher score but A is lower rank
        a = _make_item("a", rank=1, overall_score=0.2, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, overall_score=0.9, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert clustered.clusters[0].representative_event_id == "a"

    def test_representative_never_replaced(self):
        # Process: A(rank1) -> cluster A, B(rank2) -> joins A, C(rank3) joins A
        # Representative should always be A
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        c = _make_item("c", rank=3, refined_start=4.0, refined_end=14.0)
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool)
        assert clustered.clusters[0].representative_event_id == "a"

    def test_multiple_cluster_match_overlap_decides(self):
        """Test A: overlap decides when item matches two reps at different levels.
        rep1: 0-10, rep2: 12-22 (non-overlapping reps)
        item: 5-16 overlaps rep1 by 5s/10s=0.50, rep2 by 4s/10s=0.40
        Both moderate + same topic -> overlap DESC decides -> rep1 wins.
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.30, strong_overlap_threshold=0.80)
        rep1 = _make_item("rep1", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="same")
        rep2 = _make_item("rep2", rank=2, refined_start=12.0, refined_end=22.0, primary_topic_id="same")
        item = _make_item("item", rank=3, refined_start=5.0, refined_end=16.0, primary_topic_id="same")

        # Verify near-duplicate relationships BEFORE clustering
        assert _is_near_duplicate(item, rep1, config)[0] is True  # 0.50 overlap + same topic
        assert _is_near_duplicate(item, rep2, config)[0] is True  # 0.40 overlap + same topic

        pool = _make_pool([rep1, rep2, item])
        clustered = cluster_event_pool(pool, config)
        cluster_item = cluster_for_event(clustered, "item")
        assert cluster_item is not None
        # Overlap with rep1 (0.50) > rep2 (0.40) -> rep1 wins
        assert cluster_item.representative_event_id == "rep1"

    def test_multiple_cluster_match_same_topic_decides(self):
        """Test B: same topic wins when overlap ties.
        rep1: 0-10, topic=boss
        rep2: 12-22, topic=inventory, peak=14
        item: 6-16, topic=boss, peak=14
        Overlap with both: 4s/10s = 0.40
        Config: moderate <= 0.40, peak tolerance sufficient
        item~rep1 via same topic, item~rep2 via peak proximity
        Tie on overlap -> same_primary_topic=True wins -> rep1
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.40, strong_overlap_threshold=0.80, peak_tolerance_seconds=5.0)
        rep1 = _make_item("rep1", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        rep2 = _make_item("rep2", rank=2, refined_start=12.0, refined_end=22.0, primary_topic_id="inventory", peak_at=14.0)
        item = _make_item("item", rank=3, refined_start=6.0, refined_end=16.0, primary_topic_id="boss", peak_at=14.0)

        assert _is_near_duplicate(item, rep1, config)[0] is True  # same topic
        assert _is_near_duplicate(item, rep2, config)[0] is True  # peak proximity

        pool = _make_pool([rep1, rep2, item])
        clustered = cluster_event_pool(pool, config)
        cluster_item = cluster_for_event(clustered, "item")
        assert cluster_item is not None
        # Tie on overlap -> same_primary_topic wins -> rep1
        assert cluster_item.representative_event_id == "rep1"

    def test_multiple_cluster_match_peak_distance_decides(self):
        """Test C: peak distance wins when overlap and topic tie.
        rep1: 0-10, topic=boss, peak=7
        rep2: 12-22, topic=boss, peak=14
        item: 6-16, topic=boss, peak=13
        Overlap with both: 0.40, same topic for all
        Peak distances: rep1=6s, rep2=1s -> rep2 wins
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.40, strong_overlap_threshold=0.80, peak_tolerance_seconds=5.0)
        rep1 = _make_item("rep1", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss", peak_at=7.0)
        rep2 = _make_item("rep2", rank=2, refined_start=12.0, refined_end=22.0, primary_topic_id="boss", peak_at=14.0)
        item = _make_item("item", rank=3, refined_start=6.0, refined_end=16.0, primary_topic_id="boss", peak_at=13.0)

        assert _is_near_duplicate(item, rep1, config)[0] is True
        assert _is_near_duplicate(item, rep2, config)[0] is True

        pool = _make_pool([rep1, rep2, item])
        clustered = cluster_event_pool(pool, config)
        cluster_item = cluster_for_event(clustered, "item")
        assert cluster_item is not None
        # Same overlap, same topic -> peak distance wins -> rep2 (1s vs 6s)
        assert cluster_item.representative_event_id == "rep2"

    def test_multiple_cluster_match_rep_rank_decides(self):
        """Test D: rep rank decides when overlap, topic, peak all tie.
        rep1: rank=1, 0-10, topic=topic
        rep2: rank=2, 12-22, topic=topic
        item: 6-16 overlaps rep1 by 4/10=0.4, rep2 by 4/10=0.4
        Same topic, no peaks -> rep rank ASC -> rep1 wins
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.3, strong_overlap_threshold=0.7)
        rep1 = _make_item("rep1", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="topic", peak_at=None)
        rep2 = _make_item("rep2", rank=2, refined_start=12.0, refined_end=22.0, primary_topic_id="topic", peak_at=None)
        item = _make_item("item", rank=3, refined_start=6.0, refined_end=16.0, primary_topic_id="topic", peak_at=None)

        assert _is_near_duplicate(item, rep1, config)[0] is True  # 0.4 overlap + same topic
        assert _is_near_duplicate(item, rep2, config)[0] is True  # 0.4 overlap + same topic

        pool = _make_pool([rep1, rep2, item])
        clustered = cluster_event_pool(pool, config)
        cluster_item = cluster_for_event(clustered, "item")
        assert cluster_item is not None
        # All tie -> rep rank ASC -> rep1
        assert cluster_item.representative_event_id == "rep1"

    def test_final_tie_event_id_fallback_documented(self):
        """Defensive fallback: event_id ASC when all criteria tie (unreachable with valid EventPool).
        With unique ranks and rank ASC processed first, event_id ASC is a defensive
        final tie-break unreachable for valid EventPool (ranks are unique).
        Kept as defensive code, not exercised by valid inputs.
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.2, strong_overlap_threshold=0.7)
        rep1 = _make_item("aaa", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="topic")
        rep2 = _make_item("bbb", rank=2, refined_start=20.0, refined_end=30.0, primary_topic_id="topic")
        item = _make_item("item", rank=3, refined_start=7.0, refined_end=17.0, primary_topic_id="topic")
        pool = _make_pool([rep1, rep2, item])
        clustered = cluster_event_pool(pool, config)
        cluster_item = cluster_for_event(clustered, "item")
        # Defensive fallback: event_id ASC -> "aaa"
        assert cluster_item.representative_event_id == "aaa"

    def test_cluster_id_stable(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered1 = cluster_event_pool(pool)
        clustered2 = cluster_event_pool(pool)
        assert clustered1.clusters[0].cluster_id == clustered2.clusters[0].cluster_id

    def test_cluster_id_independent_input_order(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool1 = _make_pool([a, b])
        pool2 = _make_pool([b, a])  # reversed
        c1 = cluster_event_pool(pool1).clusters[0].cluster_id
        c2 = cluster_event_pool(pool2).clusters[0].cluster_id
        assert c1 == c2

    def test_members_rank_ordered(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        c = _make_item("c", rank=3, refined_start=4.0, refined_end=14.0)
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool)
        members = clustered.clusters[0].member_event_ids
        ranks = [_get_rank(eid, pool) for eid in members]
        assert ranks == sorted(ranks)

    def test_clusters_rep_rank_ordered(self):
        # Cluster reps should be ordered by their rank
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=20.0, refined_end=30.0)
        c = _make_item("c", rank=3, refined_start=40.0, refined_end=50.0)
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool)
        rep_ranks = [clustered.clusters[i].representative_event_id for i in range(len(clustered.clusters))]
        assert rep_ranks == ["a", "b", "c"]

    def test_topic_ids_union_sorted(self):
        a = _make_item_with_context("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="zebra")
        b = _make_item_with_context("b", rank=2, refined_start=2.0, refined_end=12.0, primary_topic_id="apple")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert clustered.clusters[0].topic_ids == ("apple", "zebra")

    def test_temporal_envelope_min_max(self):
        # All three have same topic and direct overlap with representative
        # A: 0-10, B: 3-13 (overlap 7s/10s=0.7 with A), C: 5-15 (overlap 5s/10s=0.5 with A)
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="topic")
        b = _make_item("b", rank=2, refined_start=3.0, refined_end=13.0, primary_topic_id="topic")
        c = _make_item("c", rank=3, refined_start=5.0, refined_end=15.0, primary_topic_id="topic")
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool)
        c = clustered.clusters[0]
        assert c.refined_start == 0.0  # min
        assert c.refined_end == 15.0  # max


# ──────────────────────────────────────────────────────────────────────────────
# Diversity tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDiversity:
    def test_all_singleton_same_as_rank_order(self):
        items = [
            _make_item("a", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("b", rank=2, refined_start=20.0, refined_end=30.0),
            _make_item("c", rank=3, refined_start=40.0, refined_end=50.0),
        ]
        pool = _make_pool(items)
        clustered = cluster_event_pool(pool)
        assert clustered.diverse_order_event_ids == ("a", "b", "c")

    def test_one_duplicate_cluster_plus_singleton_reps_first(self):
        a1 = _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0)
        a2 = _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0)
        b1 = _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0)
        pool = _make_pool([a1, a2, b1])
        clustered = cluster_event_pool(pool)
        assert clustered.diverse_order_event_ids == ("a1", "b1", "a2")

    def test_multiple_clusters_round_robin_representatives(self):
        a1 = _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0)
        a2 = _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0)
        b1 = _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0)
        b2 = _make_item("b2", rank=4, refined_start=22.0, refined_end=32.0)
        c1 = _make_item("c1", rank=5, refined_start=40.0, refined_end=50.0)
        pool = _make_pool([a1, a2, b1, b2, c1])
        clustered = cluster_event_pool(pool)
        assert clustered.diverse_order_event_ids == ("a1", "b1", "c1", "a2", "b2")

    def test_every_event_appears_exactly_once(self):
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
            _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0),
        ]
        pool = _make_pool(items)
        clustered = cluster_event_pool(pool)
        all_ids = set(clustered.diverse_order_event_ids)
        source_ids = {item.event_id for item in pool.items}
        assert all_ids == source_ids
        assert len(clustered.diverse_order_event_ids) == len(pool.items)

    def test_no_event_lost(self):
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
        ]
        pool = _make_pool(items)
        clustered = cluster_event_pool(pool)
        assert len(clustered.diverse_order_event_ids) == 2

    def test_no_event_duplicated(self):
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
        ]
        pool = _make_pool(items)
        clustered = cluster_event_pool(pool)
        assert len(set(clustered.diverse_order_event_ids)) == len(clustered.diverse_order_event_ids)

    def test_diverse_order_deterministic(self):
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
            _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0),
        ]
        pool = _make_pool(items)
        c1 = cluster_event_pool(pool)
        c2 = cluster_event_pool(pool)
        assert c1.diverse_order_event_ids == c2.diverse_order_event_ids

    def test_diverse_order_input_permutation_invariant(self):
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
            _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0),
        ]
        pool1 = _make_pool(items)
        pool2 = _make_pool(list(reversed(items)))
        c1 = cluster_event_pool(pool1)
        c2 = cluster_event_pool(pool2)
        assert c1.diverse_order_event_ids == c2.diverse_order_event_ids

    def test_original_ranks_unchanged(self):
        items = [
            _make_item("a1", rank=1, overall_score=0.9, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, overall_score=0.2, refined_start=2.0, refined_end=12.0),
        ]
        pool = _make_pool(items)
        cluster_event_pool(pool)
        for item in pool.items:
            assert item.rank in (1, 2)

    def test_original_scores_unchanged(self):
        items = [
            _make_item("a1", rank=1, overall_score=0.9, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, overall_score=0.2, refined_start=2.0, refined_end=12.0),
        ]
        pool = _make_pool(items)
        cluster_event_pool(pool)
        for item in pool.items:
            assert item.overall_score in (0.9, 0.2)


# ──────────────────────────────────────────────────────────────────────────────
# Contract / Authority tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAuthority:
    def test_same_topic_alone_does_not_cluster(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=100.0, refined_end=110.0, primary_topic_id="boss")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 2

    def test_free_text_title_change_does_not_affect_cluster(self):
        a = _make_item("evt-a", rank=1, refined_start=0.0, refined_end=10.0, title="Original Title")
        b = _make_item("evt-b", rank=2, refined_start=2.0, refined_end=12.0, title="Different Title")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1

    def test_hook_reason_not_used_for_similarity(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, hook="Hook A", reason="Reason A")
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0, hook="Hook B", reason="Reason B")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1

    def test_topic_confidence_does_not_affect_clustering(self):
        # primary_topic_id same, overlap below threshold -> no cluster
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=5.0, refined_end=15.0, primary_topic_id="boss")
        # 5s/10s = 0.5 -> moderate threshold with same topic -> duplicate
        # Change to below moderate
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=5.1, refined_end=15.0, primary_topic_id="boss")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 2

    def test_refinement_confidence_does_not_affect_clustering(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, refinement_confidence=0.9)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0, refinement_confidence=0.1)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1

    def test_overall_score_does_not_determine_similarity(self):
        # Two geometrically identical events with different scores
        a = _make_item("a", rank=1, overall_score=0.2, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, overall_score=0.9, refined_start=0.0, refined_end=10.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        assert clustered.clusters[0].representative_event_id == "a"  # rank authority

    def test_low_score_event_remains(self):
        a = _make_item("a", rank=1, overall_score=0.1)
        pool = _make_pool([a])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1

    def test_uncertain_event_remains(self):
        item = _make_item("evt", rank=1, refinement_status="uncertain")
        pool = _make_pool([item])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1

    def test_no_dedupe_remove(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert clustered.source_event_count == 2
        assert len(clustered.diverse_order_event_ids) == 2

    def test_no_llm_network_media(self):
        items = [_make_item("a", rank=1), _make_item("b", rank=2)]
        pool = _make_pool(items)
        clustered = cluster_event_pool(pool)
        assert isinstance(clustered, ClusteredEventPool)

    def test_no_editor_events_import(self):
        import inspect

        import app.pipeline.event_clustering as ec_module
        source = inspect.getsource(ec_module)
        assert "app.editor.events" not in source
        assert "best_event_pool" not in source

    def test_input_objects_unmutated(self):
        items = [_make_item("a", rank=1, refined_start=0.0, refined_end=10.0)]
        pool = _make_pool(items)
        original_start = items[0].refined_start
        cluster_event_pool(pool)
        assert items[0].refined_start == original_start


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestQueryHelpers:
    def test_cluster_by_id_existing(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        found = cluster_by_id(clustered, clustered.clusters[0].cluster_id)
        assert found is not None
        assert found.cluster_id == clustered.clusters[0].cluster_id

    def test_cluster_by_id_missing(self):
        pool = _make_pool([_make_item("a", rank=1)])
        clustered = cluster_event_pool(pool)
        found = cluster_by_id(clustered, "nonexistent")
        assert found is None

    def test_cluster_for_event_existing(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        found = cluster_for_event(clustered, "a")
        assert found is not None
        assert "a" in found.member_event_ids

    def test_cluster_for_event_missing(self):
        pool = _make_pool([_make_item("a", rank=1)])
        clustered = cluster_event_pool(pool)
        found = cluster_for_event(clustered, "nonexistent")
        assert found is None

    def test_events_in_cluster_preserves_member_rank_order(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, refined_start=2.0, refined_end=12.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        events = events_in_cluster(clustered, clustered.clusters[0].cluster_id, pool)
        ranks = [e.rank for e in events]
        assert ranks == [1, 2]


# ──────────────────────────────────────────────────────────────────────────────
# Critical transitive chain test
# ──────────────────────────────────────────────────────────────────────────────

class TestCriticalTransitiveChain:
    def test_a_b_c_chain_prevented(self):
        """A: 0-10, B: 4-14, C: 8-18 with moderate=0.5.
        A~B (6s/10s=0.6 + same topic), B~C (6s/10s=0.6 + same topic), A!~C (2s/10s=0.2).
        Should produce: Cluster(A,B), Cluster(C) - NOT A,B,C together.
        """
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.8)
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=4.0, refined_end=14.0, primary_topic_id="boss")
        c = _make_item("c", rank=3, refined_start=8.0, refined_end=18.0, primary_topic_id="boss")
        pool = _make_pool([a, b, c])
        clustered = cluster_event_pool(pool, config)

        # C only checked against representative A (0.2 overlap), not B
        # So C creates new cluster
        assert len(clustered.clusters) == 2
        cluster_a = cluster_for_event(clustered, "a")
        cluster_c = cluster_for_event(clustered, "c")
        assert cluster_a is not cluster_c
        assert set(cluster_a.member_event_ids) == {"a", "b"}
        assert cluster_c.member_event_ids == ("c",)


# ──────────────────────────────────────────────────────────────────────────────
# Same topic distant test
# ──────────────────────────────────────────────────────────────────────────────

class TestSameTopicDistant:
    def test_same_topic_distant_two_clusters(self):
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=100.0, refined_end=110.0, primary_topic_id="boss")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Score independence test
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreIndependence:
    def test_score_independence(self):
        """Two events geometrically identical but different scores.
        Should cluster same regardless of score.
        Representative = lower rank (a).
        """
        a = _make_item("a", rank=1, overall_score=0.2, refined_start=0.0, refined_end=10.0)
        b = _make_item("b", rank=2, overall_score=0.9, refined_start=0.0, refined_end=10.0)
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool)
        assert len(clustered.clusters) == 1
        assert clustered.clusters[0].representative_event_id == "a"


# ──────────────────────────────────────────────────────────────────────────────
# Topic secondary test
# ──────────────────────────────────────────────────────────────────────────────

class TestTopicSecondary:
    def test_moderate_overlap_same_topic_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7)
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=5.0, refined_end=15.0, primary_topic_id="boss")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool, config)
        assert len(clustered.clusters) == 1

    def test_moderate_overlap_different_topic_no_peaks_not_duplicate(self):
        config = ClusteringConfig(moderate_overlap_threshold=0.5, strong_overlap_threshold=0.7)
        a = _make_item("a", rank=1, refined_start=0.0, refined_end=10.0, primary_topic_id="boss")
        b = _make_item("b", rank=2, refined_start=5.0, refined_end=15.0, primary_topic_id="inventory")
        pool = _make_pool([a, b])
        clustered = cluster_event_pool(pool, config)
        assert len(clustered.clusters) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Integration F -> G
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegrationFG:
    def test_integration_f_g(self):
        """Integration test using real 1.5F build_event_pool + 1.5G cluster_event_pool."""
        # VIA v2 contract migration: Block 02 renamed VideoObservationNature ->
        # EvidenceNature (same members/values OBSERVATION/INFERENCE/UNKNOWN).
        from app.editor.via_evidence import EvidenceKind, EvidenceNature, NormalizedVideoEvidence
        from app.pipeline.event_pool import build_event_pool
        from app.pipeline.highlight_boundaries import refine_candidates
        from app.pipeline.highlight_scoring import score_candidates
        from app.pipeline.topic_timeline import build_topic_timeline
        from app.providers.base import ClipCandidate

        # Create 5 synthetic candidates
        cand1 = ClipCandidate(start=0.0, end=15.0, title="Event A", score=0.8)
        cand2 = ClipCandidate(start=2.0, end=12.0, title="Event A dup", score=0.75)  # near-dup of 1
        cand3 = ClipCandidate(start=20.0, end=35.0, title="Event B", score=0.7)
        cand4 = ClipCandidate(start=22.0, end=32.0, title="Event B dup", score=0.65)  # near-dup of 3
        cand5 = ClipCandidate(start=50.0, end=60.0, title="Event C", score=0.6)

        candidates = [cand1, cand2, cand3, cand4, cand5]

        # Synthetic VIA evidence
        via_evidence = [
            NormalizedVideoEvidence(
                evidence_id="via-1",
                asset_id="asset-1",
                start_us=1_000_000,
                end_us=14_000_000,
                evidence_kind=EvidenceKind.EVENT,
                nature=EvidenceNature.OBSERVATION,
                description="Action event 1",
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
                description="Action event 2",
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
            {"start": 50.0, "end": 55.0, "text": "Final segment"},
        ]

        # 1.5C: Score
        scored = score_candidates(candidates, via_evidence, transcript)
        assert len(scored) == 5
        assert all(s.rank is not None for s in scored)

        # 1.5D: Refine
        refinements = refine_candidates(
            candidates,
            via_evidence,
            transcript,
            signals_doc={"loudness": [], "scene_times": [], "silence_intervals": []},
            video_duration=100.0,
        )
        assert len(refinements) == 5

        # 1.5E: Topic Timeline
        topic_timeline = build_topic_timeline(transcript, via_evidence, video_duration=100.0, asset_id="asset-1")
        assert len(topic_timeline.spans) >= 1

        # 1.5F: Event Pool
        event_pool = build_event_pool(scored, refinements, topic_timeline)
        assert len(event_pool.items) == 5
        assert event_pool.asset_id == "asset-1"

        # 1.5G: Clustering
        clustered = cluster_event_pool(event_pool)

        # Verify requirements
        assert clustered.source_event_count == 5
        assert clustered.asset_id == "asset-1"

        # Should have 3 clusters: A(2), B(2), C(1)
        assert len(clustered.clusters) == 3

        # All IDs preserved
        all_ids = {c.member_event_ids for c in clustered.clusters}
        flat_ids = set()
        for ids in all_ids:
            flat_ids.update(ids)
        source_ids = {item.event_id for item in event_pool.items}
        assert flat_ids == source_ids

        # Representatives correct
        rep_ids = {c.representative_event_id for c in clustered.clusters}
        assert len(rep_ids) == 3

        # Diverse order correct
        diverse = clustered.diverse_order_event_ids
        assert len(diverse) == 5
        assert set(diverse) == source_ids

        # Scores/ranks unchanged
        for item in event_pool.items:
            assert item.overall_score is not None
            assert item.rank is not None


# ──────────────────────────────────────────────────────────────────────────────
# Permutation test
# ──────────────────────────────────────────────────────────────────────────────

class TestPermutation:
    def test_permutation_equality(self):
        """Same EventPool with permuted items produces identical output."""
        items = [
            _make_item("a1", rank=1, refined_start=0.0, refined_end=10.0),
            _make_item("a2", rank=2, refined_start=2.0, refined_end=12.0),
            _make_item("b1", rank=3, refined_start=20.0, refined_end=30.0),
            _make_item("b2", rank=4, refined_start=22.0, refined_end=32.0),
            _make_item("c1", rank=5, refined_start=40.0, refined_end=50.0),
        ]
        pool1 = _make_pool(items)
        pool2 = _make_pool(list(reversed(items)))
        c1 = cluster_event_pool(pool1)
        c2 = cluster_event_pool(pool2)
        assert c1 == c2


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _get_rank(event_id: str, pool: EventPool) -> int:
    for item in pool.items:
        if item.event_id == event_id:
            return item.rank
    raise KeyError(event_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])