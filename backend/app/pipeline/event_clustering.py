"""Deterministic Clustering + Diversity for Event Pool (SNIPRA v1.5G).

Consumes an EventPool and produces a ClusteredEventPool with near-duplicate
clusters and a recommended diverse order.

Pure intelligence -- no mutations, no editorial decisions, no filtering,
no event removal, no re-scoring, no LLM calls, no network/media operations.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.pipeline.event_pool import EventPool, EventPoolItem

if TYPE_CHECKING:
    from app.pipeline.event_pool import EventPool


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for near-duplicate clustering.

    These are technical thresholds for near-duplicate detection,
    NOT editorial duration rules.
    """
    # Strong temporal overlap threshold: overlap_ratio >= this = near-duplicate
    strong_overlap_threshold: float = 0.70

    # Moderate overlap threshold: needs secondary signal (topic or peak)
    moderate_overlap_threshold: float = 0.50

    # Peak proximity tolerance in seconds
    peak_tolerance_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.moderate_overlap_threshold <= self.strong_overlap_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: 0 <= moderate ({self.moderate_overlap_threshold}) "
                f"<= strong ({self.strong_overlap_threshold}) <= 1 required"
            )
        # peak_tolerance_seconds must be finite and non-negative
        if not math.isfinite(self.peak_tolerance_seconds):
            raise ValueError(f"peak_tolerance_seconds must be finite, got {self.peak_tolerance_seconds}")
        if self.peak_tolerance_seconds < 0.0:
            raise ValueError(f"peak_tolerance_seconds must be >= 0, got {self.peak_tolerance_seconds}")

DEFAULT_CLUSTERING_CONFIG = ClusteringConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _temporal_overlap_ratio(a: EventPoolItem, b: EventPoolItem) -> float:
    """Compute temporal overlap ratio between two events using refined ranges.

    overlap_seconds / min(duration_a, duration_b), bounded 0..1.
    """
    overlap_start = max(a.refined_start, b.refined_start)
    overlap_end = min(a.refined_end, b.refined_end)
    overlap_seconds = max(0.0, overlap_end - overlap_start)

    if overlap_seconds <= 0.0:
        return 0.0

    dur_a = a.refined_end - a.refined_start
    dur_b = b.refined_end - b.refined_start
    shorter = min(dur_a, dur_b)

    if shorter <= 0.0:
        return 0.0

    ratio = overlap_seconds / shorter
    return max(0.0, min(1.0, ratio))


def _same_primary_topic(a: EventPoolItem, b: EventPoolItem) -> bool:
    """Check if both events have the same non-None primary_topic_id."""
    return (
        a.primary_topic_id is not None
        and b.primary_topic_id is not None
        and a.primary_topic_id == b.primary_topic_id
    )


def _close_peaks(a: EventPoolItem, b: EventPoolItem, tolerance: float) -> bool:
    """Check if both events have peak_at within tolerance seconds."""
    if a.peak_at is None or b.peak_at is None:
        return False
    return abs(a.peak_at - b.peak_at) <= tolerance


def _is_near_duplicate(
    a: EventPoolItem,
    b: EventPoolItem,
    config: ClusteringConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Determine if two events are near-duplicates.

    Returns (is_duplicate, basis_tuple).
    """
    overlap_ratio = _temporal_overlap_ratio(a, b)

    # Strong overlap: immediate near-duplicate
    if overlap_ratio >= config.strong_overlap_threshold:
        return True, ("strong_temporal_overlap",)

    # Moderate overlap: needs secondary signal
    if overlap_ratio >= config.moderate_overlap_threshold:
        basis = ["moderate_temporal_overlap"]

        if _same_primary_topic(a, b):
            basis.append("same_primary_topic")
            return True, tuple(basis)

        if _close_peaks(a, b, config.peak_tolerance_seconds):
            basis.append("peak_proximity")
            return True, tuple(basis)

    return False, ()


def _compute_cluster_id(member_event_ids: tuple[str, ...]) -> str:
    """Compute deterministic cluster ID from sorted member event IDs."""
    canonical = "|".join(sorted(member_event_ids))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Output structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DuplicateRelation:
    """Relation of a member to the cluster representative."""
    event_id: str
    temporal_overlap_ratio: float
    same_primary_topic: bool
    peak_distance_seconds: float | None
    basis: tuple[str, ...]


@dataclass(frozen=True)
class EventCluster:
    """A cluster of near-duplicate events."""
    cluster_id: str
    representative_event_id: str
    member_event_ids: tuple[str, ...]
    relations: tuple[DuplicateRelation, ...]
    topic_ids: tuple[str, ...]
    refined_start: float
    refined_end: float


@dataclass(frozen=True)
class ClusteredEventPool:
    """EventPool with clustering and diversity ordering."""
    asset_id: str | None
    source_event_count: int
    clusters: tuple[EventCluster, ...]
    diverse_order_event_ids: tuple[str, ...]


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_pool(pool: EventPool) -> None:
    """Validate EventPool integrity (fail-closed)."""
    if not pool.items:
        return

    # event_id unique
    ids = [item.event_id for item in pool.items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event_id in pool")

    # rank unique, >= 1, contiguous 1..N
    ranks = [item.rank for item in pool.items]
    if any(r is None for r in ranks):
        raise ValueError("Missing rank (None) in pool")
    if any(r < 1 for r in ranks):
        raise ValueError("Invalid rank < 1 in pool")
    if len(ranks) != len(set(ranks)):
        raise ValueError("Duplicate ranks in pool")
    expected_ranks = set(range(1, len(pool.items) + 1))
    if set(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be contiguous 1..{len(pool.items)}, got {sorted(ranks)}")

    # refined range valid
    for item in pool.items:
        rs = item.refined_start
        re = item.refined_end
        if rs != rs or re != re or rs == float('inf') or rs == float('-inf') or re == float('inf') or re == float('-inf'):
            raise ValueError(f"Non-finite refined range for {item.event_id}: start={rs}, end={re}")
        if rs < 0.0:
            raise ValueError(f"Negative refined_start for {item.event_id}: {rs}")
        if re <= rs:
            raise ValueError(f"Invalid refined range for {item.event_id}: {rs} >= {re}")

    # score valid
    for item in pool.items:
        score = item.overall_score
        if score != score or score == float('inf') or score == float('-inf'):
            raise ValueError(f"Invalid score (NaN/Inf) for {item.event_id}: {score}")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score out of bounds [0,1] for {item.event_id}: {score}")


# ──────────────────────────────────────────────────────────────────────────────
# Main clustering function
# ──────────────────────────────────────────────────────────────────────────────

def cluster_event_pool(
    pool: EventPool,
    config: ClusteringConfig | None = None,
) -> ClusteredEventPool:
    """Cluster near-duplicate events and produce diverse order.

    Args:
        pool: EventPool from 1.5F (build_event_pool output)
        config: Optional ClusteringConfig (uses DEFAULT_CLUSTERING_CONFIG if None)

    Returns:
        ClusteredEventPool with clusters and diverse_order_event_ids.

    Raises:
        ValueError: If pool fails validation.
    """
    config = config or DEFAULT_CLUSTERING_CONFIG
    _validate_pool(pool)

    if not pool.items:
        return ClusteredEventPool(
            asset_id=pool.asset_id,
            source_event_count=0,
            clusters=(),
            diverse_order_event_ids=(),
        )

    # Sort items by rank ASC (1.5C authority) for deterministic processing
    items_by_rank = sorted(pool.items, key=lambda item: item.rank)

    # Representative-star clustering
    clusters: list[EventCluster] = []
    representatives: list[EventPoolItem] = []  # representatives of each cluster
    item_to_cluster: dict[str, int] = {}  # event_id -> cluster index

    for item in items_by_rank:
        # Check against existing representatives
        matches: list[tuple[int, float, bool, float | None]] = []  # (cluster_idx, overlap, same_topic, peak_dist)

        for cluster_idx, rep in enumerate(representatives):
            is_dup, basis = _is_near_duplicate(item, rep, config)
            if is_dup:
                overlap = _temporal_overlap_ratio(item, rep)
                same_topic = _same_primary_topic(item, rep)
                peak_dist = None
                if item.peak_at is not None and rep.peak_at is not None:
                    peak_dist = abs(item.peak_at - rep.peak_at)
                matches.append((cluster_idx, overlap, same_topic, peak_dist))

        if not matches:
            # New cluster with this item as representative
            cluster_idx = len(clusters)
            representatives.append(item)
            item_to_cluster[item.event_id] = cluster_idx
            clusters.append(_build_cluster([item], config, pool))
        else:
            # Choose best matching cluster deterministically
            # Sort by: overlap DESC, same_topic True first, peak_dist ASC, rep_rank ASC, rep_event_id ASC
            def match_key(m: tuple[int, float, bool, float | None]) -> tuple:
                cluster_idx, overlap, same_topic, peak_dist = m
                rep = representatives[cluster_idx]
                return (
                    -overlap,
                    0 if same_topic else 1,
                    peak_dist if peak_dist is not None else float('inf'),
                    rep.rank,
                    rep.event_id,
                )
            matches.sort(key=match_key)
            best_cluster_idx = matches[0][0]

            # Add to existing cluster
            item_to_cluster[item.event_id] = best_cluster_idx
            # Rebuild the cluster with the new member
            clusters[best_cluster_idx] = _rebuild_cluster(
                best_cluster_idx, clusters, item_to_cluster, representatives, config, pool
            )

    # Sort clusters by representative rank ASC
    clusters.sort(key=lambda c: _find_representative_rank(c, pool))

    # Build diverse order: representatives first (by cluster order), then remaining by rank ASC
    representative_ids = {c.representative_event_id for c in clusters}
    pass1 = [c.representative_event_id for c in clusters]
    pass2 = [item.event_id for item in items_by_rank if item.event_id not in representative_ids]
    diverse_order = tuple(pass1 + pass2)

    # Verify no loss/duplication
    all_ids = {item.event_id for item in pool.items}
    if set(diverse_order) != all_ids:
        raise ValueError("Diverse order lost or duplicated events")
    if len(diverse_order) != len(pool.items):
        raise ValueError("Diverse order length mismatch")

    return ClusteredEventPool(
        asset_id=pool.asset_id,
        source_event_count=len(pool.items),
        clusters=tuple(clusters),
        diverse_order_event_ids=diverse_order,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cluster building helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_representative_rank(cluster: EventCluster, pool: EventPool) -> int:
    """Find the rank of the cluster's representative."""
    for item in pool.items:
        if item.event_id == cluster.representative_event_id:
            return item.rank
    return float('inf')


def _rebuild_cluster(
    cluster_idx: int,
    clusters: list[EventCluster],
    item_to_cluster: dict[str, int],
    representatives: list[EventPoolItem],
    config: ClusteringConfig,
    pool: EventPool,
) -> EventCluster:
    """Rebuild a cluster after adding a new member."""
    rep = representatives[cluster_idx]
    member_ids = [
        eid for eid, cidx in item_to_cluster.items()
        if cidx == cluster_idx
    ]
    # Sort members by rank ASC
    member_ids.sort(key=lambda eid: _get_event_rank(eid, pool))

    # Build relations for non-representative members
    relations = []
    for member_id in member_ids:
        if member_id == rep.event_id:
            continue
        member = _get_event(member_id, pool)
        overlap = _temporal_overlap_ratio(member, rep)
        same_topic = _same_primary_topic(member, rep)
        peak_dist = None
        if member.peak_at is not None and rep.peak_at is not None:
            peak_dist = abs(member.peak_at - rep.peak_at)
        is_dup, basis = _is_near_duplicate(member, rep, config)
        relations.append(DuplicateRelation(
            event_id=member_id,
            temporal_overlap_ratio=overlap,
            same_primary_topic=same_topic,
            peak_distance_seconds=peak_dist,
            basis=tuple(sorted(basis)),
        ))

    # Sort relations by event rank ASC for determinism
    relations.sort(key=lambda r: _get_event_rank(r.event_id, pool))

    # Topic IDs union from primary_topic_id and topic_contexts
    topic_ids = set()
    for member_id in member_ids:
        member = _get_event(member_id, pool)
        if member.primary_topic_id is not None:
            topic_ids.add(member.primary_topic_id)
        for ctx in member.topic_contexts:
            topic_ids.add(ctx.topic_id)

    # Temporal envelope
    refined_start = min(_get_event(eid, pool).refined_start for eid in member_ids)
    refined_end = max(_get_event(eid, pool).refined_end for eid in member_ids)

    cluster_id = _compute_cluster_id(tuple(member_ids))

    return EventCluster(
        cluster_id=cluster_id,
        representative_event_id=rep.event_id,
        member_event_ids=tuple(member_ids),
        relations=tuple(relations),
        topic_ids=tuple(sorted(topic_ids)),
        refined_start=refined_start,
        refined_end=refined_end,
    )


def _build_cluster(
    members: list[EventPoolItem],
    config: ClusteringConfig,
    pool: EventPool,
) -> EventCluster:
    """Build a new cluster from its members (first member is representative)."""
    rep = members[0]
    member_ids = [m.event_id for m in members]
    member_ids.sort(key=lambda eid: _get_event_rank(eid, pool))

    # For new cluster with single member, no relations yet
    relations = ()
    topic_ids = set()
    if rep.primary_topic_id is not None:
        topic_ids.add(rep.primary_topic_id)
    for ctx in rep.topic_contexts:
        topic_ids.add(ctx.topic_id)

    refined_start = rep.refined_start
    refined_end = rep.refined_end
    cluster_id = _compute_cluster_id(tuple(member_ids))

    return EventCluster(
        cluster_id=cluster_id,
        representative_event_id=rep.event_id,
        member_event_ids=tuple(member_ids),
        relations=relations,
        topic_ids=tuple(sorted(topic_ids)),
        refined_start=refined_start,
        refined_end=refined_end,
    )


def _get_event(event_id: str, pool: EventPool) -> EventPoolItem:
    """Find event by ID in pool."""
    for item in pool.items:
        if item.event_id == event_id:
            return item
    raise KeyError(f"Event {event_id} not found in pool")


def _get_event_rank(event_id: str, pool: EventPool) -> int:
    """Get rank of event by ID."""
    return _get_event(event_id, pool).rank


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers (read-only)
# ──────────────────────────────────────────────────────────────────────────────

def cluster_by_id(clustered: ClusteredEventPool, cluster_id: str) -> EventCluster | None:
    """Find cluster by exact cluster_id match."""
    for cluster in clustered.clusters:
        if cluster.cluster_id == cluster_id:
            return cluster
    return None


def cluster_for_event(clustered: ClusteredEventPool, event_id: str) -> EventCluster | None:
    """Find the cluster containing the given event_id."""
    for cluster in clustered.clusters:
        if event_id in cluster.member_event_ids:
            return cluster
    return None


def events_in_cluster(clustered: ClusteredEventPool, cluster_id: str, pool: EventPool) -> tuple[EventPoolItem, ...]:
    """Get all EventPoolItems in a cluster, ordered by member rank ASC."""
    cluster = cluster_by_id(clustered, cluster_id)
    if not cluster:
        return ()
    result = []
    for eid in cluster.member_event_ids:
        result.append(_get_event(eid, pool))
    return tuple(result)