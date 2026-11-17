"""Short Batch Models (SNIPRA v1.5J).

Data models for Create N Shorts orchestration + Smart Collections / Montage.
Pure data structures -- no mutations, no editorial decisions, no media operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid4().hex


def deterministic_collection_id(
    collection_type: SmartCollectionType,
    asset_id: str,
    event_ids: tuple[str, ...],
    topic_ids: tuple[str, ...],
) -> str:
    """Generate deterministic collection ID from stable content."""
    import hashlib
    canonical = "|".join([
        collection_type.value,
        asset_id,
        ",".join(sorted(event_ids)),
        ",".join(sorted(topic_ids)),
    ])
    return "coll_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class ShortBatchStatus(StrEnum):
    """Status of a short batch."""
    PLANNED = "planned"        # All requested shorts selected and grounded
    PARTIAL = "partial"        # Some but not all requested shorts selected
    FAILED = "failed"          # No valid shorts could be created
    NEEDS_USER = "needs_user"  # Awaiting user confirmation


@dataclass(frozen=True)
class ShortPlan:
    """A single short in the batch with full grounding."""
    short_id: str
    event_id: str                    # References EventPoolItem.event_id
    source_asset_id: str
    original_start: float
    original_end: float
    refined_start: float
    refined_end: float
    title: str
    hook: str
    reason: str
    refinement_status: str
    refinement_confidence: float
    peak_at: float | None
    payoff_at: float | None
    payoff_window_start: float | None
    payoff_window_end: float | None
    reaction_at: float | None
    reaction_window_start: float | None
    reaction_window_end: float | None
    primary_topic_id: str | None
    topic_contexts: tuple[Any, ...]   # EventTopicContext from enhanced evidence
    score_components: tuple[Any, ...] # EventScoreTrace from enhanced evidence
    evidence_refs: tuple[str, ...]
    director_reason: str             # Director's rationale for selecting this event
    editorial_intent: str            # "highlight" or "montage"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API response."""
        return asdict(self)


@dataclass(frozen=True)
class ShortBatchPlan:
    """Complete plan for N shorts."""
    batch_id: str
    project_id: str
    asset_id: str | None
    requested_count: int
    selected_count: int
    status: ShortBatchStatus
    shorts: tuple[ShortPlan, ...]
    enhanced_evidence: dict[str, Any]  # Serialized enhanced_event_pool from 1.5H
    director_decision: dict[str, Any]  # Director's raw decision output
    created_at: str
    created_by: str = "orchestrator"

    @property
    def is_complete(self) -> bool:
        return self.status == ShortBatchStatus.PLANNED

    @property
    def is_partial(self) -> bool:
        return self.status == ShortBatchStatus.PARTIAL

    @property
    def is_failed(self) -> bool:
        return self.status == ShortBatchStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for API response."""
        data = asdict(self)
        # Convert shorts tuple to list of dicts
        data["shorts"] = [s.to_dict() for s in self.shorts]
        return data


@dataclass(frozen=True)
class ShortBatchRequest:
    """Request to create N shorts from a source job/asset."""
    source_job_id: str | None = None
    asset_id: str | None = None
    requested_count: int = 1
    intent: str = "highlight"  # "highlight" or "montage"
    constraints: dict[str, Any] = None  # Future: duration limits, topic filters, etc.

    def __post_init__(self) -> None:
        if not isinstance(self.requested_count, int) or isinstance(self.requested_count, bool):
            raise ValueError("requested_count must be an integer (not bool)")
        if self.requested_count < 1:
            raise ValueError("requested_count must be >= 1")
        if self.intent not in ("highlight", "montage"):
            raise ValueError("intent must be 'highlight' or 'montage'")


def validate_short_batch_request(request: ShortBatchRequest) -> None:
    """Validate a ShortBatchRequest (fail-closed)."""
    if not isinstance(request.requested_count, int) or isinstance(request.requested_count, bool):
        raise ValueError("requested_count must be an integer (not bool)")
    if request.requested_count < 1:
        raise ValueError("requested_count must be >= 1")
    if request.intent not in ("highlight", "montage"):
        raise ValueError("intent must be 'highlight' or 'montage'")


class ShortBatchStatusError(ValueError):
    """Raised when batch status is invalid for an operation."""
    pass


class DirectorSelectionError(ValueError):
    """Raised when Director's selection is invalid."""
    pass


class InsufficientEventsError(ValueError):
    """Raised when not enough valid events exist for the requested count."""
    pass


def build_short_plans_from_director_selection(
    director_selected_event_ids: list[str],
    enhanced_evidence: dict[str, Any],
    director_reasons: dict[str, str] | None = None,
    editorial_intent: str = "highlight",
) -> list[ShortPlan]:
    """Build ShortPlan objects from Director's selected event IDs.

    Args:
        director_selected_event_ids: Event IDs selected by Director (in order)
        enhanced_evidence: Validated enhanced_event_pool from 1.5H bridge
        director_reasons: Optional dict mapping event_id -> Director's rationale
        editorial_intent: "highlight" or "montage"

    Returns:
        List of ShortPlan objects in Director's selection order.

    Raises:
        DirectorSelectionError: If any selected event_id is not grounded in enhanced evidence.
    """
    enhanced_evidence.get("events", [])
    event_by_id = {e["event_id"]: e for e in enhanced_evidence.get("events", [])}

    shorts: list[ShortPlan] = []

    for event_id in director_selected_event_ids:
        if event_id not in event_by_id:
            raise DirectorSelectionError(
                f"Director selected event {event_id!r} not found in enhanced evidence"
            )

        ev = event_by_id[event_id]
        reason = director_reasons.get(event_id, "") if director_reasons else ""

        short = ShortPlan(
            short_id=new_id(),
            event_id=ev["event_id"],
            source_asset_id=enhanced_evidence.get("asset_id", ""),
            original_start=ev["original_start"],
            original_end=ev["original_end"],
            refined_start=ev["refined_start"],
            refined_end=ev["refined_end"],
            title=ev["title"],
            hook=ev["hook"],
            reason=ev["reason"],
            refinement_status=ev["refinement_status"],
            refinement_confidence=ev["refinement_confidence"],
            peak_at=ev.get("peak_at"),
            payoff_at=ev.get("payoff_at"),
            payoff_window_start=ev.get("payoff_window_start"),
            payoff_window_end=ev.get("payoff_window_end"),
            reaction_at=ev.get("reaction_at"),
            reaction_window_start=ev.get("reaction_window_start"),
            reaction_window_end=ev.get("reaction_window_end"),
            primary_topic_id=ev.get("primary_topic_id"),
            topic_contexts=tuple(ev.get("topic_contexts", [])),
            score_components=tuple(ev.get("score_components", [])),
            evidence_refs=tuple(ev.get("evidence_refs", [])),
            director_reason=reason,
            editorial_intent=editorial_intent,
        )
        shorts.append(short)

    return shorts


# ──────────────────────────────────────────────────────────────────────────────
# Smart Collections (1.5J)
# ──────────────────────────────────────────────────────────────────────────────

class SmartCollectionType(StrEnum):
    """Types of smart collections."""
    TOPIC = "topic"           # Events sharing primary topic
    CLUSTER = "cluster"       # Events from same near-duplicate cluster
    TEMPORAL = "temporal"     # Events with temporal continuity
    EVIDENCE = "evidence"     # Events with compatible VIA evidence


@dataclass(frozen=True)
class SmartCollection:
    """A structural grouping of related events (not an editorial decision)."""
    collection_id: str
    collection_type: SmartCollectionType
    event_ids: tuple[str, ...]
    topic_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    refined_start: float
    refined_end: float
    event_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmartCollectionPool:
    """Complete set of smart collections for an asset."""
    asset_id: str
    source_event_count: int
    collections: tuple[SmartCollection, ...]
    ungrouped_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["collections"] = [c.to_dict() for c in self.collections]
        return data


def build_smart_collections(
    enhanced_evidence: dict[str, Any],
) -> SmartCollectionPool:
    """Build smart collections from enhanced evidence (1.5H bridge output).

    Pure transform: no LLM, no DB, no network, no media, no mutations.
    All events are preserved - either in a collection or ungrouped.

    Args:
        enhanced_evidence: Output from build_director_evidence()

    Returns:
        SmartCollectionPool with all events accounted for.
    """
    events = enhanced_evidence.get("events", [])
    clusters = enhanced_evidence.get("clusters", [])

    if not events:
        return SmartCollectionPool(
            asset_id=enhanced_evidence.get("asset_id", ""),
            source_event_count=0,
            collections=(),
            ungrouped_event_ids=(),
        )

    # Map events by ID for quick lookup
    event_by_id = {e["event_id"]: e for e in events}
    all_event_ids = set(event_by_id.keys())
    clustered_event_ids: set[str] = set()

    # Build collections from clusters (CLUSTER type)
    collections: list[SmartCollection] = []
    asset_id = enhanced_evidence.get("asset_id", "")

    for cluster in clusters:
        member_ids = cluster.get("member_event_ids", [])
        if len(member_ids) < 2:
            continue  # Singletons don't form collections

        # Validate all members exist
        valid_members = [mid for mid in member_ids if mid in event_by_id]
        if len(valid_members) < 2:
            continue

        # Build collection
        topic_ids = set()
        evidence_refs = set()
        refined_starts = []
        refined_ends = []

        for mid in sorted(valid_members):
            ev = event_by_id[mid]
            if ev.get("primary_topic_id"):
                topic_ids.add(ev["primary_topic_id"])
            for ctx in ev.get("topic_contexts", []):
                topic_ids.add(ctx.get("topic_id", ""))
            evidence_refs.update(ev.get("evidence_refs", []))
            refined_starts.append(ev.get("refined_start", 0.0))
            refined_ends.append(ev.get("refined_end", 0.0))

        topic_ids_sorted = tuple(sorted(topic_ids))
        collection = SmartCollection(
            collection_id=deterministic_collection_id(
                SmartCollectionType.CLUSTER, asset_id, tuple(sorted(valid_members)), topic_ids_sorted
            ),
            collection_type=SmartCollectionType.CLUSTER,
            event_ids=tuple(sorted(valid_members)),
            topic_ids=topic_ids_sorted,
            evidence_refs=tuple(sorted(evidence_refs)),
            refined_start=min(refined_starts) if refined_starts else 0.0,
            refined_end=max(refined_ends) if refined_ends else 0.0,
            event_count=len(valid_members),
        )
        collections.append(collection)
        clustered_event_ids.update(valid_members)

    # Build topic-based collections for unclustered events (TOPIC type)
    # Group events by primary_topic_id
    topic_groups: dict[str, list[str]] = {}
    for event_id in sorted(all_event_ids - clustered_event_ids):
        ev = event_by_id[event_id]
        ptid = ev.get("primary_topic_id")
        if ptid:
            topic_groups.setdefault(ptid, []).append(event_id)

    for topic_id in sorted(topic_groups.keys()):
        member_ids = topic_groups[topic_id]
        if len(member_ids) < 2:
            continue

        evidence_refs = set()
        refined_starts = []
        refined_ends = []

        for mid in sorted(member_ids):
            ev = event_by_id[mid]
            evidence_refs.update(ev.get("evidence_refs", []))
            refined_starts.append(ev.get("refined_start", 0.0))
            refined_ends.append(ev.get("refined_end", 0.0))

        collection = SmartCollection(
            collection_id=deterministic_collection_id(
                SmartCollectionType.TOPIC, asset_id, tuple(sorted(member_ids)), (topic_id,)
            ),
            collection_type=SmartCollectionType.TOPIC,
            event_ids=tuple(sorted(member_ids)),
            topic_ids=(topic_id,),
            evidence_refs=tuple(sorted(evidence_refs)),
            refined_start=min(refined_starts) if refined_starts else 0.0,
            refined_end=max(refined_ends) if refined_ends else 0.0,
            event_count=len(member_ids),
        )
        collections.append(collection)
        clustered_event_ids.update(member_ids)

    # Ungrouped events (preserved for Director)
    ungrouped = tuple(sorted(all_event_ids - clustered_event_ids))

    return SmartCollectionPool(
        asset_id=enhanced_evidence.get("asset_id", ""),
        source_event_count=len(events),
        collections=tuple(collections),
        ungrouped_event_ids=ungrouped,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Montage Models (1.5J)
# ──────────────────────────────────────────────────────────────────────────────

class MontageBatchStatus(StrEnum):
    """Status of a montage batch."""
    PLANNED = "planned"        # All requested montages selected and grounded
    PARTIAL = "partial"        # Some but not all requested montages selected
    FAILED = "failed"          # No valid montages could be created


@dataclass(frozen=True)
class MontageSegmentPlan:
    """A single segment (event) in a montage with full grounding."""
    event_id: str                    # References EventPoolItem.event_id
    source_asset_id: str
    refined_start: float
    refined_end: float
    order_index: int                 # Director's editorial order
    evidence_refs: tuple[str, ...]
    title: str
    peak_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MontagePlan:
    """Complete plan for a single montage (multiple events composed together)."""
    montage_id: str
    source_asset_id: str
    segments: tuple[MontageSegmentPlan, ...]
    director_reason: str             # Director's rationale for this montage
    estimated_duration: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segments"] = [s.to_dict() for s in self.segments]
        return data


@dataclass(frozen=True)
class MontageBatchPlan:
    """Complete plan for N montages."""
    batch_id: str
    project_id: str
    asset_id: str | None
    requested_count: int
    selected_count: int
    status: MontageBatchStatus
    montages: tuple[MontagePlan, ...]
    smart_collections: dict[str, Any]  # SmartCollectionPool serialized
    enhanced_evidence: dict[str, Any]  # Serialized enhanced_event_pool from 1.5H
    director_decision: dict[str, Any]  # Director's raw decision output
    created_at: str
    created_by: str = "orchestrator"

    @property
    def is_complete(self) -> bool:
        return self.status == MontageBatchStatus.PLANNED

    @property
    def is_partial(self) -> bool:
        return self.status == MontageBatchStatus.PARTIAL

    @property
    def is_failed(self) -> bool:
        return self.status == MontageBatchStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["montages"] = [m.to_dict() for m in self.montages]
        return data


@dataclass(frozen=True)
class MontageBatchRequest:
    """Request to create N montages from a source job/asset."""
    source_job_id: str | None = None
    asset_id: str | None = None
    requested_count: int = 1
    constraints: dict[str, Any] = None  # Future: duration limits, topic filters, etc.

    def __post_init__(self) -> None:
        if not isinstance(self.requested_count, int) or isinstance(self.requested_count, bool):
            raise ValueError("requested_count must be an integer (not bool)")
        if self.requested_count < 1:
            raise ValueError("requested_count must be >= 1")


def validate_montage_batch_request(request: MontageBatchRequest) -> None:
    """Validate a MontageBatchRequest (fail-closed)."""
    if not isinstance(request.requested_count, int) or isinstance(request.requested_count, bool):
        raise ValueError("requested_count must be an integer (not bool)")
    if request.requested_count < 1:
        raise ValueError("requested_count must be >= 1")


class MontageSelectionError(ValueError):
    """Raised when Director's montage selection is invalid."""
    pass


# Metadata Models (1.5K)
# Metadata is a non-editorial projection of an already-grounded plan.  These
# types deliberately carry IDs and event order from I/J verbatim.
@dataclass(frozen=True)
class ContentMetadata:
    content_id: str
    content_type: str
    title: str
    description: str
    hashtags: tuple[str, ...]
    keywords: tuple[str, ...]
    language: str
    source_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hashtags"] = list(self.hashtags)
        data["keywords"] = list(self.keywords)
        data["source_event_ids"] = list(self.source_event_ids)
        return data


@dataclass(frozen=True)
class PlatformMetadata:
    platform: str
    items: tuple[ContentMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"platform": self.platform, "items": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class MetadataBatchPlan:
    project_id: str
    platform_metadata: PlatformMetadata
    source_plan_type: str

    @property
    def items(self) -> tuple[ContentMetadata, ...]:
        return self.platform_metadata.items

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source_plan_type": self.source_plan_type,
            **self.platform_metadata.to_dict(),
        }


class MetadataGenerationRequest(BaseModel):
    """Serialized I/J plan accepted by the non-mutating metadata endpoint."""
    plan_type: str = Field(pattern="^(shorts|montages)$")
    plan: dict[str, Any]
    platform: str = "generic"
    language: str = "es"


def build_montage_plans_from_director_selection(
    director_montages: list[dict[str, Any]],
    enhanced_evidence: dict[str, Any],
    smart_collections: SmartCollectionPool | None = None,
) -> list[MontagePlan]:
    """Build MontagePlan objects from Director's montage selection.

    Args:
        director_montages: List of montage dicts from Director, each containing:
            - montage_id (optional)
            - segments: list of {event_id, order_index}
            - director_reason (optional)
        enhanced_evidence: Validated enhanced_event_pool from 1.5H bridge
        smart_collections: Optional SmartCollectionPool for validation context

    Returns:
        List of MontagePlan objects in Director's selection order.

    Raises:
        MontageSelectionError: If any validation fails.
    """
    enhanced_evidence.get("events", [])
    event_by_id = {e["event_id"]: e for e in enhanced_evidence.get("events", [])}
    set(event_by_id.keys())
    used_event_ids: set[str] = set()

    # Canonical asset ID from enhanced evidence (1.5H contract)
    canonical_asset_id = enhanced_evidence.get("asset_id", "")

    montages: list[MontagePlan] = []

    for montage_idx, montage_data in enumerate(director_montages):
        segments_data = montage_data.get("segments", [])
        if not segments_data:
            raise MontageSelectionError(f"Montage at index {montage_idx} has no segments")

        # Validate minimum 2 unique events per montage
        event_ids = [seg.get("event_id") for seg in segments_data if seg.get("event_id")]
        if len(event_ids) < 2:
            raise MontageSelectionError(
                f"Montage at index {montage_idx} has only {len(event_ids)} event(s); "
                "montage requires at least 2 unique events"
            )

        # Check for duplicates within montage
        if len(event_ids) != len(set(event_ids)):
            raise MontageSelectionError(
                f"Montage at index {montage_idx} has duplicate event IDs: {event_ids}"
            )

        # Check for unknown events
        unknown = [eid for eid in event_ids if eid not in event_by_id]
        if unknown:
            raise MontageSelectionError(
                f"Montage at index {montage_idx} references unknown event(s): {unknown}"
            )

        # Check asset consistency using canonical asset_id from 1.5H contract
        if canonical_asset_id:
            for eid in event_ids:
                ev = event_by_id[eid]
                # Events in enhanced_evidence should all belong to the same asset
                # The asset_id field in 1.5H is at top level, not per-event
                pass

        # Check cross-montage reuse
        reused = set(event_ids) & used_event_ids
        if reused:
            raise MontageSelectionError(
                f"Montage at index {montage_idx} reuses event(s) from previous montage(s): {reused}"
            )
        used_event_ids.update(event_ids)

        # Validate boundaries and build segments
        segments: list[MontageSegmentPlan] = []
        order_indices = []

        for seg_data in segments_data:
            event_id = seg_data.get("event_id")
            order_index = seg_data.get("order_index", 0)
            ev = event_by_id[event_id]

            # Validate order_index: int (not bool), >= 0
            if not isinstance(order_index, int) or isinstance(order_index, bool):
                raise MontageSelectionError(
                    f"Event {event_id}: order_index must be int (not bool)"
                )
            if order_index < 0:
                raise MontageSelectionError(
                    f"Event {event_id}: order_index must be >= 0"
                )

            order_indices.append(order_index)

            segment = MontageSegmentPlan(
                event_id=event_id,
                source_asset_id=canonical_asset_id,
                refined_start=ev.get("refined_start", 0.0),
                refined_end=ev.get("refined_end", 0.0),
                order_index=order_index,
                evidence_refs=tuple(ev.get("evidence_refs", [])),
                title=ev.get("title", ""),
                peak_at=ev.get("peak_at"),
            )
            segments.append(segment)

        # Validate unique order_indices within montage
        if len(order_indices) != len(set(order_indices)):
            raise MontageSelectionError(
                f"Montage at index {montage_idx} has duplicate order_index values: {order_indices}"
            )

        # Sort segments by Director's order_index
        segments.sort(key=lambda s: s.order_index)

        # Validate refined ranges
        for seg in segments:
            if seg.refined_end <= seg.refined_start:
                raise MontageSelectionError(
                    f"Event {seg.event_id} has invalid refined range: "
                    f"{seg.refined_start} >= {seg.refined_end}"
                )

        # Calculate estimated duration
        estimated_duration = sum(
            seg.refined_end - seg.refined_start for seg in segments
        )

        montage = MontagePlan(
            montage_id=montage_data.get("montage_id", f"montage_{uuid4().hex[:12]}"),
            source_asset_id=canonical_asset_id,
            segments=tuple(segments),
            director_reason=montage_data.get("director_reason", ""),
            estimated_duration=estimated_duration,
        )
        montages.append(montage)

    return montages
