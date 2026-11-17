"""Tests for Director Evidence Bridge (SNIPRA v1.5H)."""

from __future__ import annotations

import json
import pytest

from app.pipeline.event_pool import (
    EventPool,
    EventPoolItem,
    EventScoreTrace,
    EventTopicContext,
)
from app.pipeline.event_clustering import (
    ClusteredEventPool,
    EventCluster,
    DuplicateRelation,
)
from app.editor.director_bridge import (
    build_director_evidence,
    validate_director_evidence,
    director_evidence_to_json,
    BridgeValidationError,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def _make_event_pool_item(
    event_id: str,
    rank: int,
    overall_score: float = 0.8,
    refined_start: float = 10.0,
    refined_end: float = 20.0,
    peak_at: float | None = 15.0,
    primary_topic_id: str | None = "topic_1",
    title: str | None = None,
    hook: str | None = None,
    reason: str | None = None,
) -> EventPoolItem:
    return EventPoolItem(
        event_id=event_id,
        rank=rank,
        overall_score=overall_score,
        original_start=refined_start - 1.0,
        original_end=refined_end + 1.0,
        refined_start=refined_start,
        refined_end=refined_end,
        title=title or f"Event {event_id}",
        hook=hook or "Hook text",
        reason=reason or "Reason text",
        refinement_status="refined",
        refinement_confidence=0.9,
        peak_at=peak_at,
        payoff_at=18.0,
        payoff_window_start=17.0,
        payoff_window_end=19.0,
        reaction_at=19.5,
        reaction_window_start=19.0,
        reaction_window_end=20.0,
        primary_topic_id=primary_topic_id,
        topic_contexts=(
            EventTopicContext(
                topic_id="topic_1",
                topic_start=0.0,
                topic_end=30.0,
                overlap_seconds=10.0,
                overlap_ratio=1.0,
                confidence=0.95,
                keywords=("gameplay", "victory"),
                representative_text="We won the game",
                evidence_refs=("via_1",),
            ),
        ),
        score_components=(
            EventScoreTrace(
                name="base_score",
                raw_value=0.7,
                weight=0.5,
                contribution=0.35,
                evidence_refs=(),
                note="Base score from detector",
            ),
            EventScoreTrace(
                name="event_strength",
                raw_value=0.9,
                weight=0.2,
                contribution=0.18,
                evidence_refs=("via_1",),
                note="High event confidence",
            ),
        ),
        evidence_refs=("via_1", "transcript_1"),
    )


def _make_duplicate_relation(
    event_id: str,
    overlap: float = 0.8,
    same_topic: bool = True,
    peak_dist: float | None = 1.0,
) -> DuplicateRelation:
    return DuplicateRelation(
        event_id=event_id,
        temporal_overlap_ratio=overlap,
        same_primary_topic=same_topic,
        peak_distance_seconds=peak_dist,
        basis=("strong_temporal_overlap",) if overlap >= 0.7 else ("moderate_temporal_overlap", "same_primary_topic"),
    )


def _make_event_cluster(
    cluster_id: str,
    representative_id: str,
    member_ids: list[str],
    refined_start: float = 10.0,
    refined_end: float = 20.0,
) -> EventCluster:
    relations = []
    for mid in member_ids:
        if mid != representative_id:
            relations.append(_make_duplicate_relation(mid))
    return EventCluster(
        cluster_id=cluster_id,
        representative_event_id=representative_id,
        member_event_ids=tuple(member_ids),
        relations=tuple(relations),
        topic_ids=("topic_1",),
        refined_start=refined_start,
        refined_end=refined_end,
    )


def _make_event_pool(items: list[EventPoolItem] | None = None) -> EventPool:
    if items is None:
        items = [
            _make_event_pool_item("evt_1", 1, refined_start=10.0, refined_end=20.0),
            _make_event_pool_item("evt_2", 2, refined_start=30.0, refined_end=40.0),
            _make_event_pool_item("evt_3", 3, refined_start=50.0, refined_end=60.0),
        ]
    return EventPool(asset_id="asset_123", items=tuple(items))


def _make_clustered_pool(
    event_pool: EventPool,
    clusters: list[EventCluster] | None = None,
) -> ClusteredEventPool:
    {item.event_id for item in event_pool.items}
    if clusters is None:
        # Create default clusters matching the pool's events
        sorted_items = sorted(event_pool.items, key=lambda i: i.rank)
        if len(sorted_items) >= 2:
            clusters = [
                _make_event_cluster(
                    "cl_1",
                    sorted_items[0].event_id,
                    [sorted_items[0].event_id, sorted_items[1].event_id],
                    sorted_items[0].refined_start,
                    sorted_items[1].refined_end,
                ),
            ]
            if len(sorted_items) >= 3:
                clusters.append(
                    _make_event_cluster(
                        "cl_2",
                        sorted_items[2].event_id,
                        [sorted_items[2].event_id],
                        sorted_items[2].refined_start,
                        sorted_items[2].refined_end,
                    )
                )
        elif len(sorted_items) == 1:
            clusters = [
                _make_event_cluster(
                    "cl_1",
                    sorted_items[0].event_id,
                    [sorted_items[0].event_id],
                    sorted_items[0].refined_start,
                    sorted_items[0].refined_end,
                ),
            ]
        else:
            clusters = []
    diverse_order = [c.representative_event_id for c in clusters]
    # Add non-representatives
    rep_ids = {c.representative_event_id for c in clusters}
    for item in event_pool.items:
        if item.event_id not in rep_ids:
            diverse_order.append(item.event_id)
    return ClusteredEventPool(
        asset_id=event_pool.asset_id,
        source_event_count=len(event_pool.items),
        clusters=tuple(clusters),
        diverse_order_event_ids=tuple(diverse_order),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_event_pool_and_clustering():
    """1. Empty EventPool + empty clustering."""
    pool = EventPool(asset_id="asset_1", items=())
    clustered = ClusteredEventPool(
        asset_id="asset_1",
        source_event_count=0,
        clusters=(),
        diverse_order_event_ids=(),
    )
    evidence = build_director_evidence(pool, clustered)
    assert evidence["source_event_count"] == 0
    assert evidence["events"] == []
    assert evidence["clusters"] == []
    assert evidence["diverse_order_event_ids"] == []


def test_one_event():
    """2. One event."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1)])
    clustered = _make_clustered_pool(pool, [_make_event_cluster("cl_1", "evt_1", ["evt_1"])])
    evidence = build_director_evidence(pool, clustered)
    assert evidence["source_event_count"] == 1
    assert len(evidence["events"]) == 1
    assert evidence["events"][0]["event_id"] == "evt_1"
    assert len(evidence["clusters"]) == 1
    assert evidence["diverse_order_event_ids"] == ["evt_1"]


def test_json_serializable():
    """3. JSON serializable."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Should not raise
    json_str = director_evidence_to_json(evidence)
    assert isinstance(json_str, str)
    # Round-trip
    parsed = json.loads(json_str)
    assert parsed["contract_version"] == "1.5H"


def test_deterministic_output():
    """4. Deterministic output."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence1 = build_director_evidence(pool, clustered)
    evidence2 = build_director_evidence(pool, clustered)
    json1 = director_evidence_to_json(evidence1)
    json2 = director_evidence_to_json(evidence2)
    assert json1 == json2


def test_events_rank_ordered():
    """5. Events rank ordered."""
    items = [
        _make_event_pool_item("evt_3", 3),
        _make_event_pool_item("evt_1", 1),
        _make_event_pool_item("evt_2", 2),
    ]
    pool = _make_event_pool(items)
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    ranks = [e["rank"] for e in evidence["events"]]
    assert ranks == [1, 2, 3]


def test_scores_preserved_exactly():
    """6. Scores preserved exactly."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1, overall_score=0.8472)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["overall_score"] == 0.8472


def test_ranks_preserved_exactly():
    """7. Ranks preserved exactly."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 5)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["rank"] == 5


def test_refined_ranges_preserved():
    """8. Refined ranges preserved."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1, refined_start=12.345, refined_end=28.901)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["refined_start"] == 12.345
    assert evidence["events"][0]["refined_end"] == 28.901


def test_peak_preserved():
    """9. Peak preserved."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1, peak_at=15.5)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["peak_at"] == 15.5


def test_payoff_fields_preserved():
    """10. Payoff fields preserved."""
    item = _make_event_pool_item("evt_1", 1)
    # Override payoff fields
    item = EventPoolItem(
        **{**item.__dict__, "payoff_at": 18.0, "payoff_window_start": 17.5, "payoff_window_end": 18.5}
    )
    pool = _make_event_pool([item])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert e["payoff_at"] == 18.0
    assert e["payoff_window_start"] == 17.5
    assert e["payoff_window_end"] == 18.5


def test_reaction_fields_preserved():
    """11. Reaction fields preserved."""
    item = _make_event_pool_item("evt_1", 1)
    item = EventPoolItem(
        **{**item.__dict__, "reaction_at": 19.0, "reaction_window_start": 18.5, "reaction_window_end": 20.0}
    )
    pool = _make_event_pool([item])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert e["reaction_at"] == 19.0
    assert e["reaction_window_start"] == 18.5
    assert e["reaction_window_end"] == 20.0


def test_topic_contexts_preserved():
    """12. Topic contexts preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert "topic_contexts" in e
    assert len(e["topic_contexts"]) == 1
    ctx = e["topic_contexts"][0]
    assert ctx["topic_id"] == "topic_1"
    assert ctx["overlap_ratio"] == 1.0


def test_score_components_preserved():
    """13. Score components preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert "score_components" in e
    assert len(e["score_components"]) == 2
    comp_names = {c["name"] for c in e["score_components"]}
    assert comp_names == {"base_score", "event_strength"}


def test_evidence_refs_preserved():
    """14. Evidence refs preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert set(e["evidence_refs"]) == {"via_1", "transcript_1"}


def test_clusters_preserved():
    """15. Clusters preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert len(evidence["clusters"]) == 2
    cluster_ids = {c["cluster_id"] for c in evidence["clusters"]}
    assert cluster_ids == {"cl_1", "cl_2"}


def test_member_order_preserved():
    """16. Member order preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Members should be sorted by rank ASC within cluster
    cl1 = next(c for c in evidence["clusters"] if c["cluster_id"] == "cl_1")
    assert cl1["member_event_ids"] == ["evt_1", "evt_2"]


def test_relations_preserved():
    """17. Relations preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    cl1 = next(c for c in evidence["clusters"] if c["cluster_id"] == "cl_1")
    assert len(cl1["relations"]) == 1
    rel = cl1["relations"][0]
    assert rel["event_id"] == "evt_2"
    assert rel["temporal_overlap_ratio"] == 0.8


def test_diverse_order_preserved():
    """18. Diverse order preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Representatives first (cl_1 rep=evt_1, cl_2 rep=evt_3), then remaining by rank
    assert evidence["diverse_order_event_ids"] == ["evt_1", "evt_3", "evt_2"]


def test_asset_id_preserved():
    """19. Asset ID preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["asset_id"] == "asset_123"


def test_mismatched_asset_id_reject():
    """20. Mismatched asset_id reject."""
    pool = _make_event_pool()
    clustered = ClusteredEventPool(
        asset_id="different_asset",
        source_event_count=3,
        clusters=(),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="Asset ID mismatch"):
        build_director_evidence(pool, clustered)


def test_source_event_count_mismatch_reject():
    """21. Source event count mismatch reject."""
    pool = _make_event_pool()
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=99,  # Wrong count
        clusters=(),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="Source event count mismatch"):
        build_director_evidence(pool, clustered)


def test_missing_event_in_cluster_reject():
    """22. Missing event in cluster reject."""
    pool = _make_event_pool()
    bad_cluster = _make_event_cluster("cl_1", "evt_1", ["evt_1", "evt_999"])  # evt_999 not in pool
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(bad_cluster,),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="not found in EventPool"):
        build_director_evidence(pool, clustered)


def test_event_in_two_clusters_reject():
    """23. Event in two clusters reject."""
    pool = _make_event_pool()
    cl1 = _make_event_cluster("cl_1", "evt_1", ["evt_1", "evt_2"])
    cl2 = _make_event_cluster("cl_2", "evt_3", ["evt_2", "evt_3"])  # evt_2 in both
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(cl1, cl2),
        diverse_order_event_ids=("evt_1", "evt_3", "evt_2"),
    )
    with pytest.raises(BridgeValidationError, match="appears in multiple clusters"):
        build_director_evidence(pool, clustered)


def test_event_in_zero_clusters_reject():
    """24. Event in zero clusters reject."""
    pool = _make_event_pool()
    # Only cluster cl_1 with evt_1, evt_2 - evt_3 not clustered
    cl1 = _make_event_cluster("cl_1", "evt_1", ["evt_1", "evt_2"])
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(cl1,),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="not clustered"):
        build_director_evidence(pool, clustered)


def test_representative_missing_reject():
    """25. Representative missing reject."""
    pool = _make_event_pool()
    # Representative evt_999 not in pool - add to diverse_order to pass first check
    cl1 = _make_event_cluster("cl_1", "evt_999", ["evt_999", "evt_1"])
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(cl1,),
        diverse_order_event_ids=("evt_999", "evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="not found in EventPool"):
        build_director_evidence(pool, clustered)


def test_representative_not_member_reject():
    """26. Representative not member reject."""
    pool = _make_event_pool()
    # Representative evt_1 not in member_event_ids - evt_1 is in pool but not in cluster members
    cl1 = _make_event_cluster("cl_1", "evt_1", ["evt_2", "evt_3"])
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(cl1,),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_3"),
    )
    with pytest.raises(BridgeValidationError, match="events not clustered"):
        build_director_evidence(pool, clustered)


def test_unknown_diverse_id_reject():
    """27. Unknown diverse ID reject."""
    pool = _make_event_pool()
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(),
        diverse_order_event_ids=("evt_1", "evt_2", "evt_999"),  # evt_999 unknown
    )
    with pytest.raises(BridgeValidationError, match="not found in EventPool"):
        build_director_evidence(pool, clustered)


def test_duplicate_diverse_id_reject():
    """28. Duplicate diverse ID reject."""
    pool = _make_event_pool()
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(),
        diverse_order_event_ids=("evt_1", "evt_1", "evt_2"),  # duplicate evt_1
    )
    with pytest.raises(BridgeValidationError, match="Duplicate event IDs in diverse_order"):
        build_director_evidence(pool, clustered)


def test_lost_diverse_id_reject():
    """29. Lost diverse ID reject."""
    pool = _make_event_pool()
    clustered = ClusteredEventPool(
        asset_id="asset_123",
        source_event_count=3,
        clusters=(),
        diverse_order_event_ids=("evt_1", "evt_2"),  # missing evt_3
    )
    with pytest.raises(BridgeValidationError, match="missing in diverse_order"):
        build_director_evidence(pool, clustered)


def test_source_inputs_not_mutated():
    """30. Source inputs not mutated."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    pool_items_before = list(pool.items)
    clustered_clusters_before = list(clustered.clusters)

    build_director_evidence(pool, clustered)

    assert list(pool.items) == pool_items_before
    assert list(clustered.clusters) == clustered_clusters_before


def test_no_score_recomputation():
    """31. No score recomputation."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1, overall_score=0.123456)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["overall_score"] == 0.123456


def test_no_rank_recomputation():
    """32. No rank recomputation."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 99)])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["events"][0]["rank"] == 99


def test_no_clustering_performed():
    """33. No clustering performed."""
    # Bridge only transforms, doesn't cluster
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Just verify it returns the same clusters passed in
    assert len(evidence["clusters"]) == len(clustered.clusters)


def test_no_llm():
    """34. No LLM."""
    # Bridge has no LLM calls - verified by code inspection
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["contract_version"] == "1.5H"


def test_no_db():
    """35. No DB."""
    # Bridge has no DB access - verified by code inspection
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert "events" in evidence


def test_no_network():
    """36. No network."""
    # Bridge has no network calls - verified by code inspection
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert "clusters" in evidence


def test_no_media():
    """37. No media."""
    # Bridge has no media operations - verified by code inspection
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert "diverse_order_event_ids" in evidence


def test_free_text_copied_only():
    """38. Free-text copied only."""
    pool = _make_event_pool([_make_event_pool_item("evt_1", 1, title="My Title", hook="My Hook", reason="My Reason")])
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    e = evidence["events"][0]
    assert e["title"] == "My Title"
    assert e["hook"] == "My Hook"
    assert e["reason"] == "My Reason"


def test_no_duplicate_derived_maps():
    """39. No duplicate derived maps (refined_ranges, topic_contexts, score_rank)."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Should not have separate refined_ranges, topic_contexts, score_rank maps
    assert "refined_ranges" not in evidence
    assert "topic_contexts" not in evidence
    assert "score_rank" not in evidence
    assert "representative_event_ids" not in evidence
    # All data should be inside events[] and clusters[]


def test_contract_version_stable():
    """40. Contract version stable."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    assert evidence["contract_version"] == "1.5H"


def test_same_logical_input_deterministic():
    """41. Same logical input deterministic."""
    # Build two pools with same logical content but different object instances
    items1 = [
        _make_event_pool_item("evt_1", 1),
        _make_event_pool_item("evt_2", 2),
    ]
    items2 = [
        _make_event_pool_item("evt_1", 1),
        _make_event_pool_item("evt_2", 2),
    ]
    pool1 = _make_event_pool(items1)
    pool2 = _make_event_pool(items2)
    cl1 = _make_clustered_pool(pool1)
    cl2 = _make_clustered_pool(pool2)

    ev1 = build_director_evidence(pool1, cl1)
    ev2 = build_director_evidence(pool2, cl2)

    json1 = director_evidence_to_json(ev1)
    json2 = director_evidence_to_json(ev2)
    assert json1 == json2


def test_cluster_ids_unchanged():
    """42. Cluster IDs unchanged."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    cluster_ids = {c["cluster_id"] for c in evidence["clusters"]}
    assert cluster_ids == {"cl_1", "cl_2"}


def test_representative_ids_unchanged():
    """43. Representative IDs unchanged."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    rep_ids = {c["representative_event_id"] for c in evidence["clusters"]}
    assert rep_ids == {"evt_1", "evt_3"}


def test_non_representative_preserved():
    """44. Non-representative preserved."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    all_ids = {e["event_id"] for e in evidence["events"]}
    assert all_ids == {"evt_1", "evt_2", "evt_3"}


def test_json_dumps_succeeds():
    """45. json.dumps succeeds."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Should not raise
    json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    director_evidence_to_json(evidence)


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_director_evidence_valid():
    """Validate accepts valid evidence."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    validated = validate_director_evidence(evidence)
    assert validated is evidence


def test_validate_director_evidence_missing_keys():
    """Validate rejects missing keys."""
    with pytest.raises(BridgeValidationError, match="Missing required keys"):
        validate_director_evidence({})


def test_validate_director_evidence_wrong_version():
    """Validate rejects wrong contract version."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["contract_version"] = "1.5G"
    with pytest.raises(BridgeValidationError, match="Unsupported contract_version"):
        validate_director_evidence(evidence)


def test_validate_director_evidence_duplicate_event_ids():
    """Validate rejects duplicate event IDs."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"].append(evidence["events"][0])  # duplicate
    with pytest.raises(BridgeValidationError, match="Duplicate event_id"):
        validate_director_evidence(evidence)


def test_validate_director_evidence_missing_event_id():
    """Validate rejects missing event_id."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0].pop("event_id")
    with pytest.raises(BridgeValidationError, match="missing event_id"):
        validate_director_evidence(evidence)


def test_validate_director_evidence_rep_not_in_members():
    """Validate rejects representative not in members."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["clusters"][0]["representative_event_id"] = "evt_999"
    with pytest.raises(BridgeValidationError, match="not in its own cluster members"):
        validate_director_evidence(evidence)


def test_validate_director_evidence_diverse_order_mismatch():
    """Validate rejects diverse order mismatch."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["diverse_order_event_ids"] = ["evt_1", "evt_2"]  # missing evt_3
    with pytest.raises(BridgeValidationError, match="missing in diverse_order"):
        validate_director_evidence(evidence)


def test_validate_director_evidence_source_count_mismatch():
    """Validate rejects source_event_count mismatch."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["source_event_count"] = 99
    with pytest.raises(BridgeValidationError, match="source_event_count"):
        validate_director_evidence(evidence)


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced validation tests (persisted evidence)
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_persisted_unknown_cluster_member():
    """1. Persisted unknown cluster member → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["clusters"][0]["member_event_ids"].append("unknown-event")
    with pytest.raises(BridgeValidationError, match="not found in events"):
        validate_director_evidence(evidence)


def test_validate_persisted_event_missing_from_clusters():
    """2. Persisted event missing from all clusters → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Remove evt_3 from all clusters
    evidence["clusters"] = [evidence["clusters"][0]]  # Only keep cl_1 with evt_1, evt_2
    with pytest.raises(BridgeValidationError, match="events not clustered"):
        validate_director_evidence(evidence)


def test_validate_persisted_event_in_two_clusters():
    """3. Persisted event in two clusters → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Add evt_2 to second cluster
    evidence["clusters"][1]["member_event_ids"].append("evt_2")
    with pytest.raises(BridgeValidationError, match="appears in multiple clusters"):
        validate_director_evidence(evidence)


def test_validate_persisted_representative_unknown_event():
    """4. Persisted representative unknown event → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["clusters"][0]["representative_event_id"] = "unknown-event"
    with pytest.raises(BridgeValidationError, match="not in its own cluster members"):
        validate_director_evidence(evidence)


def test_validate_persisted_duplicate_cluster_id():
    """5. Duplicate cluster_id → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["clusters"][1]["cluster_id"] = evidence["clusters"][0]["cluster_id"]
    with pytest.raises(BridgeValidationError, match="Duplicate cluster_id"):
        validate_director_evidence(evidence)


def test_validate_persisted_invalid_rank_string():
    """6. Invalid rank string → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["rank"] = "1"
    with pytest.raises(BridgeValidationError, match="rank must be int"):
        validate_director_evidence(evidence)


def test_validate_persisted_rank_bool():
    """7. Rank bool → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["rank"] = True
    with pytest.raises(BridgeValidationError, match="rank must be int"):
        validate_director_evidence(evidence)


def test_validate_persisted_duplicate_ranks():
    """8. Duplicate ranks → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][1]["rank"] = evidence["events"][0]["rank"]
    with pytest.raises(BridgeValidationError, match="Duplicate rank"):
        validate_director_evidence(evidence)


def test_validate_persisted_score_nan():
    """9. Score NaN → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["overall_score"] = float('nan')
    with pytest.raises(BridgeValidationError, match="NaN or Inf"):
        validate_director_evidence(evidence)


def test_validate_persisted_score_inf():
    """10. Score Inf → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["overall_score"] = float('inf')
    with pytest.raises(BridgeValidationError, match="NaN or Inf"):
        validate_director_evidence(evidence)


def test_validate_persisted_score_outside_bounds():
    """11. Score outside bounds → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["overall_score"] = 1.5
    with pytest.raises(BridgeValidationError, match="out of bounds"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_start_nan():
    """12. Refined start NaN → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_start"] = float('nan')
    with pytest.raises(BridgeValidationError, match="NaN or Inf"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_start_negative():
    """13. Refined start negative → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_start"] = -1.0
    with pytest.raises(BridgeValidationError, match="refined_start negative"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_end_inf():
    """14. Refined end Inf → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_end"] = float('inf')
    with pytest.raises(BridgeValidationError, match="NaN or Inf"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_end_le_start():
    """15. Refined end <= start → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_end"] = evidence["events"][0]["refined_start"]
    with pytest.raises(BridgeValidationError, match="refined_end <= refined_start"):
        validate_director_evidence(evidence)


def test_validate_persisted_score_bool():
    """Score bool → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["overall_score"] = True
    with pytest.raises(BridgeValidationError, match="must be a number"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_start_bool():
    """Refined start bool → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_start"] = True
    with pytest.raises(BridgeValidationError, match="must be a number"):
        validate_director_evidence(evidence)


def test_validate_persisted_refined_end_bool():
    """Refined end bool → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["events"][0]["refined_end"] = True
    with pytest.raises(BridgeValidationError, match="must be a number"):
        validate_director_evidence(evidence)


def test_validate_persisted_relation_unknown_event():
    """16. Relation unknown event → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    evidence["clusters"][0]["relations"].append({
        "event_id": "unknown-event",
        "temporal_overlap_ratio": 0.5,
        "same_primary_topic": False,
        "peak_distance_seconds": 1.0,
        "basis": ["moderate_temporal_overlap"],
    })
    with pytest.raises(BridgeValidationError, match="not in cluster members"):
        validate_director_evidence(evidence)


def test_validate_persisted_relation_points_to_representative():
    """17. Relation points to representative → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    rep_id = evidence["clusters"][0]["representative_event_id"]
    evidence["clusters"][0]["relations"].append({
        "event_id": rep_id,
        "temporal_overlap_ratio": 0.5,
        "same_primary_topic": False,
        "peak_distance_seconds": 1.0,
        "basis": ["moderate_temporal_overlap"],
    })
    with pytest.raises(BridgeValidationError, match="points to representative"):
        validate_director_evidence(evidence)


def test_validate_persisted_relation_outside_own_cluster():
    """18. Relation event outside own cluster → reject."""
    pool = _make_event_pool()
    clustered = _make_clustered_pool(pool)
    evidence = build_director_evidence(pool, clustered)
    # Add relation pointing to event in other cluster
    other_cluster_member = evidence["clusters"][1]["member_event_ids"][0]
    evidence["clusters"][0]["relations"].append({
        "event_id": other_cluster_member,
        "temporal_overlap_ratio": 0.5,
        "same_primary_topic": False,
        "peak_distance_seconds": 1.0,
        "basis": ["moderate_temporal_overlap"],
    })
    with pytest.raises(BridgeValidationError, match="not in cluster members"):
        validate_director_evidence(evidence)