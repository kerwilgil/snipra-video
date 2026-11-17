"""Director Evidence Bridge (SNIPRA v1.5H).

Transforms EventPool + ClusteredEventPool into Director-facing evidence.
Pure transformation: no scoring, no clustering, no mutations, no LLM, no DB,
no network, no media operations.
"""

from __future__ import annotations

import json
from typing import Any

from app.pipeline.event_clustering import ClusteredEventPool, DuplicateRelation, EventCluster
from app.pipeline.event_pool import EventPool, EventPoolItem


class BridgeValidationError(ValueError):
    """Raised when cross-contract validation fails."""
    pass


def _validate_cross_contract(
    event_pool: EventPool,
    clustered_pool: ClusteredEventPool,
) -> None:
    """Validate cross-contract invariants between EventPool and ClusteredEventPool.

    Raises:
        BridgeValidationError: If any invariant is violated.
    """
    # Asset ID must match
    if event_pool.asset_id != clustered_pool.asset_id:
        raise BridgeValidationError(
            f"Asset ID mismatch: event_pool.asset_id={event_pool.asset_id!r}, "
            f"clustered_pool.asset_id={clustered_pool.asset_id!r}"
        )

    # Source event count must match
    if clustered_pool.source_event_count != len(event_pool.items):
        raise BridgeValidationError(
            f"Source event count mismatch: clustered_pool.source_event_count="
            f"{clustered_pool.source_event_count}, len(event_pool.items)={len(event_pool.items)}"
        )

    # Build lookup maps
    event_ids_in_pool = {item.event_id for item in event_pool.items}
    {item.event_id: item for item in event_pool.items}

    # All diverse_order_event_ids must exist exactly once in EventPool
    diverse_ids = list(clustered_pool.diverse_order_event_ids)
    if len(diverse_ids) != len(set(diverse_ids)):
        raise BridgeValidationError("Duplicate event IDs in diverse_order_event_ids")

    for eid in diverse_ids:
        if eid not in event_ids_in_pool:
            raise BridgeValidationError(
                f"Event ID {eid!r} in diverse_order_event_ids not found in EventPool"
            )

    if set(diverse_ids) != event_ids_in_pool:
        missing = event_ids_in_pool - set(diverse_ids)
        extra = set(diverse_ids) - event_ids_in_pool
        parts = []
        if missing:
            parts.append(f"missing in diverse_order: {sorted(missing)}")
        if extra:
            parts.append(f"extra in diverse_order: {sorted(extra)}")
        raise BridgeValidationError("; ".join(parts))

    # All member_event_ids across clusters must exist in EventPool
    # and each must appear in exactly one cluster
    seen_member_ids: set[str] = set()
    for cluster in clustered_pool.clusters:
        for member_id in cluster.member_event_ids:
            if member_id not in event_ids_in_pool:
                raise BridgeValidationError(
                    f"Cluster member {member_id!r} not found in EventPool"
                )
            if member_id in seen_member_ids:
                raise BridgeValidationError(
                    f"Event ID {member_id!r} appears in multiple clusters"
                )
            seen_member_ids.add(member_id)

    if seen_member_ids != event_ids_in_pool:
        missing = event_ids_in_pool - seen_member_ids
        extra = seen_member_ids - event_ids_in_pool
        parts = []
        if missing:
            parts.append(f"events not clustered: {sorted(missing)}")
        if extra:
            parts.append(f"clustered events not in pool: {sorted(extra)}")
        raise BridgeValidationError("; ".join(parts))

    # Each representative must exist and belong to its own cluster
    for cluster in clustered_pool.clusters:
        rep_id = cluster.representative_event_id
        if rep_id not in event_ids_in_pool:
            raise BridgeValidationError(
                f"Representative {rep_id!r} not found in EventPool"
            )
        if rep_id not in cluster.member_event_ids:
            raise BridgeValidationError(
                f"Representative {rep_id!r} not a member of its own cluster"
            )


def _serialize_event_topic_context(ctx: Any) -> dict[str, Any]:
    """Serialize EventTopicContext to JSON-safe dict."""
    return {
        "topic_id": ctx.topic_id,
        "topic_start": ctx.topic_start,
        "topic_end": ctx.topic_end,
        "overlap_seconds": ctx.overlap_seconds,
        "overlap_ratio": ctx.overlap_ratio,
        "confidence": ctx.confidence,
        "keywords": list(ctx.keywords),
        "representative_text": ctx.representative_text,
        "evidence_refs": list(ctx.evidence_refs),
    }


def _serialize_event_score_trace(trace: Any) -> dict[str, Any]:
    """Serialize EventScoreTrace to JSON-safe dict."""
    return {
        "name": trace.name,
        "raw_value": trace.raw_value,
        "weight": trace.weight,
        "contribution": trace.contribution,
        "evidence_refs": list(trace.evidence_refs),
        "note": trace.note,
    }


def _serialize_duplicate_relation(rel: DuplicateRelation) -> dict[str, Any]:
    """Serialize DuplicateRelation to JSON-safe dict."""
    return {
        "event_id": rel.event_id,
        "temporal_overlap_ratio": rel.temporal_overlap_ratio,
        "same_primary_topic": rel.same_primary_topic,
        "peak_distance_seconds": rel.peak_distance_seconds,
        "basis": list(rel.basis),
    }


def _serialize_event_pool_item(item: EventPoolItem) -> dict[str, Any]:
    """Serialize EventPoolItem to JSON-safe dict."""
    return {
        "event_id": item.event_id,
        "rank": item.rank,
        "overall_score": item.overall_score,
        "original_start": item.original_start,
        "original_end": item.original_end,
        "refined_start": item.refined_start,
        "refined_end": item.refined_end,
        "title": item.title,
        "hook": item.hook,
        "reason": item.reason,
        "refinement_status": item.refinement_status,
        "refinement_confidence": item.refinement_confidence,
        "peak_at": item.peak_at,
        "payoff_at": item.payoff_at,
        "payoff_window_start": item.payoff_window_start,
        "payoff_window_end": item.payoff_window_end,
        "reaction_at": item.reaction_at,
        "reaction_window_start": item.reaction_window_start,
        "reaction_window_end": item.reaction_window_end,
        "primary_topic_id": item.primary_topic_id,
        "topic_contexts": [_serialize_event_topic_context(ctx) for ctx in item.topic_contexts],
        "score_components": [_serialize_event_score_trace(comp) for comp in item.score_components],
        "evidence_refs": list(item.evidence_refs),
    }


def _serialize_event_cluster(cluster: EventCluster) -> dict[str, Any]:
    """Serialize EventCluster to JSON-safe dict."""
    return {
        "cluster_id": cluster.cluster_id,
        "representative_event_id": cluster.representative_event_id,
        "member_event_ids": list(cluster.member_event_ids),
        "relations": [_serialize_duplicate_relation(rel) for rel in cluster.relations],
        "topic_ids": list(cluster.topic_ids),
        "refined_start": cluster.refined_start,
        "refined_end": cluster.refined_end,
    }


def build_director_evidence(
    event_pool: EventPool,
    clustered_pool: ClusteredEventPool,
) -> dict[str, Any]:
    """Transform EventPool + ClusteredEventPool into Director-facing evidence.

    Args:
        event_pool: EventPool from 1.5F (build_event_pool output)
        clustered_pool: ClusteredEventPool from 1.5G (cluster_event_pool output)

    Returns:
        JSON-safe dict with contract_version, asset_id, source_event_count,
        events (rank-ordered), clusters, and diverse_order_event_ids.

    Raises:
        BridgeValidationError: If cross-contract validation fails.
    """
    # Cross-contract validation (fail-closed)
    _validate_cross_contract(event_pool, clustered_pool)

    # Events are already in rank order in EventPool (by 1.5C authority)
    # Ensure deterministic order: rank ASC
    events_sorted = sorted(event_pool.items, key=lambda item: item.rank)

    # Clusters preserve order from ClusteredEventPool (representative rank ASC)
    clusters_serialized = [_serialize_event_cluster(c) for c in clustered_pool.clusters]

    # Diverse order copied exactly from ClusteredEventPool
    diverse_order = list(clustered_pool.diverse_order_event_ids)

    return {
        "contract_version": "1.5H",
        "asset_id": event_pool.asset_id,
        "source_event_count": clustered_pool.source_event_count,
        "events": [_serialize_event_pool_item(item) for item in events_sorted],
        "clusters": clusters_serialized,
        "diverse_order_event_ids": diverse_order,
    }


def validate_director_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate a previously serialized director evidence dict.

    Used during confirmation to ensure persisted evidence is intact.

    Args:
        evidence: Dict previously returned by build_director_evidence

    Returns:
        The same dict if valid.

    Raises:
        BridgeValidationError: If evidence is corrupt or invalid.
    """
    # Required keys
    required_keys = {
        "contract_version",
        "asset_id",
        "source_event_count",
        "events",
        "clusters",
        "diverse_order_event_ids",
    }
    missing = required_keys - set(evidence.keys())
    if missing:
        raise BridgeValidationError(f"Missing required keys: {sorted(missing)}")

    # Contract version check
    if evidence.get("contract_version") != "1.5H":
        raise BridgeValidationError(
            f"Unsupported contract_version: {evidence.get('contract_version')!r}"
        )

    # Validate events structure
    events = evidence.get("events", [])
    if not isinstance(events, list):
        raise BridgeValidationError("events must be a list")

    event_ids = set()
    ranks = set()
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            raise BridgeValidationError(f"Event at index {i} is not an object")
        eid = event.get("event_id")
        if not isinstance(eid, str) or not eid:
            raise BridgeValidationError(f"Event at index {i} missing event_id")
        if eid in event_ids:
            raise BridgeValidationError(f"Duplicate event_id: {eid}")
        event_ids.add(eid)

        # Validate rank: int (not bool), >= 1, unique
        rank = event.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise BridgeValidationError(f"Event {eid}: rank must be int (not bool)")
        if rank < 1:
            raise BridgeValidationError(f"Event {eid}: rank must be >= 1")
        if rank in ranks:
            raise BridgeValidationError(f"Duplicate rank: {rank}")
        ranks.add(rank)

        # Validate overall_score: finite number, 0 <= score <= 1 (not bool)
        score = event.get("overall_score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise BridgeValidationError(f"Event {eid}: overall_score must be a number (not bool)")
        if score != score or score == float('inf') or score == float('-inf'):
            raise BridgeValidationError(f"Event {eid}: overall_score is NaN or Inf")
        if score < 0.0 or score > 1.0:
            raise BridgeValidationError(f"Event {eid}: overall_score out of bounds [0,1]")

        # Validate refined_start: finite, >= 0 (not bool)
        rs = event.get("refined_start")
        if not isinstance(rs, (int, float)) or isinstance(rs, bool):
            raise BridgeValidationError(f"Event {eid}: refined_start must be a number (not bool)")
        if rs != rs or rs == float('inf') or rs == float('-inf'):
            raise BridgeValidationError(f"Event {eid}: refined_start is NaN or Inf")
        if rs < 0.0:
            raise BridgeValidationError(f"Event {eid}: refined_start negative")

        # Validate refined_end: finite, > refined_start (not bool)
        re = event.get("refined_end")
        if not isinstance(re, (int, float)) or isinstance(re, bool):
            raise BridgeValidationError(f"Event {eid}: refined_end must be a number (not bool)")
        if re != re or re == float('inf') or re == float('-inf'):
            raise BridgeValidationError(f"Event {eid}: refined_end is NaN or Inf")
        if re <= rs:
            raise BridgeValidationError(f"Event {eid}: refined_end <= refined_start")

        # Required fields
        for field in ("rank", "overall_score", "refined_start", "refined_end"):
            if field not in event:
                raise BridgeValidationError(f"Event {eid} missing required field: {field}")

    # Validate clusters structure
    clusters = evidence.get("clusters", [])
    if not isinstance(clusters, list):
        raise BridgeValidationError("clusters must be a list")

    clustered_event_ids = set()
    cluster_ids = set()
    for i, cluster in enumerate(clusters):
        if not isinstance(cluster, dict):
            raise BridgeValidationError(f"Cluster at index {i} is not an object")

        # Check cluster_id unique and non-empty
        cluster_id = cluster.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise BridgeValidationError(f"Cluster at index {i} missing or invalid cluster_id")
        if cluster_id in cluster_ids:
            raise BridgeValidationError(f"Duplicate cluster_id: {cluster_id}")
        cluster_ids.add(cluster_id)

        # Check representative
        rep_id = cluster.get("representative_event_id")
        if not isinstance(rep_id, str) or not rep_id:
            raise BridgeValidationError(f"Cluster {cluster_id} missing representative_event_id")

        # Check members
        members = cluster.get("member_event_ids", [])
        if not isinstance(members, list):
            raise BridgeValidationError(f"Cluster {cluster_id} member_event_ids must be a list")

        for member_id in members:
            if not isinstance(member_id, str):
                raise BridgeValidationError(f"Cluster {cluster_id} has non-string member ID")
            if member_id in clustered_event_ids:
                raise BridgeValidationError(f"Event {member_id} appears in multiple clusters")
            if member_id not in event_ids:
                raise BridgeValidationError(f"Cluster {cluster_id} member {member_id!r} not found in events")
            clustered_event_ids.add(member_id)

        if rep_id not in members:
            raise BridgeValidationError(f"Representative {rep_id} not in its own cluster members")
        if rep_id not in event_ids:
            raise BridgeValidationError(f"Representative {rep_id} not found in events")

        # Validate relations
        relations = cluster.get("relations", [])
        if not isinstance(relations, list):
            raise BridgeValidationError(f"Cluster {cluster_id} relations must be a list")
        for rel in relations:
            if not isinstance(rel, dict):
                raise BridgeValidationError(f"Cluster {cluster_id} relation is not an object")
            rel_eid = rel.get("event_id")
            if not isinstance(rel_eid, str) or not rel_eid:
                raise BridgeValidationError(f"Cluster {cluster_id} relation missing event_id")
            if rel_eid not in members:
                raise BridgeValidationError(f"Cluster {cluster_id} relation event {rel_eid} not in cluster members")
            if rel_eid == rep_id:
                raise BridgeValidationError(f"Cluster {cluster_id} relation points to representative")

    # All events must be clustered exactly once
    if clustered_event_ids != event_ids:
        missing = event_ids - clustered_event_ids
        extra = clustered_event_ids - event_ids
        parts = []
        if missing:
            parts.append(f"events not clustered: {sorted(missing)}")
        if extra:
            parts.append(f"clustered events not in pool: {sorted(extra)}")
        raise BridgeValidationError("; ".join(parts))

    # Validate diverse order
    diverse_order = evidence.get("diverse_order_event_ids", [])
    if not isinstance(diverse_order, list):
        raise BridgeValidationError("diverse_order_event_ids must be a list")

    if len(diverse_order) != len(set(diverse_order)):
        raise BridgeValidationError("Duplicate IDs in diverse_order_event_ids")

    if set(diverse_order) != event_ids:
        missing = event_ids - set(diverse_order)
        extra = set(diverse_order) - event_ids
        parts = []
        if missing:
            parts.append(f"missing in diverse_order: {sorted(missing)}")
        if extra:
            parts.append(f"extra in diverse_order: {sorted(extra)}")
        raise BridgeValidationError("; ".join(parts))

    # Validate source_event_count matches
    if evidence.get("source_event_count") != len(events):
        raise BridgeValidationError(
            f"source_event_count ({evidence.get('source_event_count')}) "
            f"!= len(events) ({len(events)})"
        )

    return evidence


def director_evidence_to_json(evidence: dict[str, Any]) -> str:
    """Serialize director evidence to JSON string.

    Args:
        evidence: Dict from build_director_evidence or validate_director_evidence

    Returns:
        JSON string.
    """
    return json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))