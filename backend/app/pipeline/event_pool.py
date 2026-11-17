"""Deterministic Event Pool for highlights (SNIPRA v1.5F).

Consumes scored candidates + refined boundaries + topic timeline and produces
an inspectable pool of events with full contextual metadata.

Pure intelligence -- no mutations, no editorial decisions, no filtering,
no clustering, no re-scoring, no LLM calls, no network/media operations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.pipeline.highlight_boundaries import RefinedBoundaries
from app.pipeline.highlight_scoring import ScoredCandidate
from app.pipeline.topic_timeline import TopicTimeline

if TYPE_CHECKING:
    from app.providers.base import ClipCandidate


# ──────────────────────────────────────────────────────────────────────────────
# Technical constants
# ──────────────────────────────────────────────────────────────────────────────

TIME_EPSILON = 1e-6  # Technical tolerance for boundary validation


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility helper (matches 1.5C/1.5D stable candidate identity)
# ──────────────────────────────────────────────────────────────────────────────

def _stable_candidate_id_from_candidate(candidate: ClipCandidate) -> str:
    """Deterministic identity for candidate — EXACT match to 1.5C/1.5D formula.

    Compatibility identity with accepted 1.5C/1.5D contract.
    DO NOT CHANGE this formula.
    """
    parts = [
        str(candidate.start),
        str(candidate.end),
        candidate.title or "",
        candidate.hook or "",
        candidate.reason or "",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Output structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventScoreTrace:
    """Traceable component score from 1.5C, preserved exactly."""
    name: str
    raw_value: float
    weight: float
    contribution: float
    evidence_refs: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class EventTopicContext:
    """Topic context for an event (read-only, no editorial authority)."""
    topic_id: str
    topic_start: float
    topic_end: float
    overlap_seconds: float
    overlap_ratio: float
    confidence: float
    keywords: tuple[str, ...]
    representative_text: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class EventPoolItem:
    """A single event in the pool with full context."""
    event_id: str

    rank: int
    overall_score: float

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
    topic_contexts: tuple[EventTopicContext, ...]

    score_components: tuple[EventScoreTrace, ...]

    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class EventPool:
    """Complete event pool for a video asset."""
    asset_id: str | None
    items: tuple[EventPoolItem, ...]


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_scored_candidates(scored: list[ScoredCandidate]) -> dict[str, ScoredCandidate]:
    """Validate scored candidates and return map by stable_id.

    Validates:
    - candidate IDs unique
    - rank present, >= 1, unique
    - overall_score in [0, 1], not NaN/Inf
    """
    if not scored:
        return {}

    # Check candidate IDs unique
    ids = [_stable_candidate_id_from_candidate(sc.candidate) for sc in scored]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs in scored_candidates")

    # Validate ranks
    ranks = [sc.rank for sc in scored]
    if any(r is None for r in ranks):
        raise ValueError("Missing rank in scored_candidates")
    if any(r < 1 for r in ranks):
        raise ValueError("Invalid rank < 1 in scored_candidates")
    if len(ranks) != len(set(ranks)):
        raise ValueError("Duplicate ranks in scored_candidates")
    # Ranks must be contiguous 1..N
    expected_ranks = set(range(1, len(scored) + 1))
    if set(ranks) != expected_ranks:
        raise ValueError(f"Ranks must be contiguous 1..{len(scored)}, got {sorted(ranks)}")

    # Validate scores
    for sc in scored:
        score = sc.overall_score
        if score != score or score == float('inf') or score == float('-inf'):  # NaN/Inf check
            raise ValueError(f"Invalid score (NaN/Inf): {score}")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"Score out of bounds [0,1]: {score}")

    return {_stable_candidate_id_from_candidate(sc.candidate): sc for sc in scored}


def _validate_refinements(refinements: list[RefinedBoundaries]) -> dict[str, RefinedBoundaries]:
    """Validate refinements and return map by candidate_id.

    Validates:
    - candidate_ids unique
    - original_start/end match candidate
    - refined range valid (finite, non-negative start, end > start, no NaN/Inf)
    """
    if not refinements:
        return {}

    # Check candidate_ids unique
    ids = [r.candidate_id for r in refinements]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate_ids in refinements")

    result = {}
    for r in refinements:
        # Validate refined range: finite, non-negative start, end > start, no NaN/Inf
        rs = r.refined_start
        re = r.refined_end

        # Check for NaN/Inf
        if rs != rs or re != re or rs == float('inf') or rs == float('-inf') or re == float('inf') or re == float('-inf'):
            raise ValueError(f"Non-finite refined range: start={rs}, end={re}")

        # refined_start must be finite and >= 0.0
        if rs < 0.0:
            raise ValueError(f"Negative refined_start: {rs}")

        # refined_end must be finite and > refined_start
        if re <= rs:
            raise ValueError(f"Invalid refined range: {rs} >= {re}")

        result[r.candidate_id] = r

    return result


def _validate_join(scored_map: dict[str, ScoredCandidate], refinement_map: dict[str, RefinedBoundaries]) -> None:
    """Validate exact match between scored candidates and refinements."""
    scored_ids = set(scored_map.keys())
    refinement_ids = set(refinement_map.keys())

    if scored_ids != refinement_ids:
        missing_in_refinement = scored_ids - refinement_ids
        orphan_in_refinement = refinement_ids - scored_ids
        parts = []
        if missing_in_refinement:
            parts.append(f"missing refinement for: {sorted(missing_in_refinement)}")
        if orphan_in_refinement:
            parts.append(f"orphan refinement: {sorted(orphan_in_refinement)}")
        raise ValueError("; ".join(parts))

    # Validate original boundaries match (technical epsilon)
    for cid in scored_ids:
        sc = scored_map[cid]
        rb = refinement_map[cid]
        if abs(sc.candidate.start - rb.original_start) > TIME_EPSILON:
            raise ValueError(f"Original start mismatch for {cid}: candidate={sc.candidate.start}, refinement={rb.original_start}")
        if abs(sc.candidate.end - rb.original_end) > TIME_EPSILON:
            raise ValueError(f"Original end mismatch for {cid}: candidate={sc.candidate.end}, refinement={rb.original_end}")


# ──────────────────────────────────────────────────────────────────────────────
# Topic association
# ──────────────────────────────────────────────────────────────────────────────

def _associate_topics(
    refined_start: float,
    refined_end: float,
    topic_timeline: TopicTimeline,
) -> tuple[EventTopicContext, ...]:
    """Associate topic spans with an event using REFINED range.

    Returns tuple of EventTopicContext ordered by topic_start ASC.
    """
    if not topic_timeline.spans:
        return ()

    event_duration = refined_end - refined_start
    if event_duration <= 0:
        return ()

    contexts = []

    for span in topic_timeline.spans:
        # Calculate overlap with refined range
        overlap_start = max(refined_start, span.start)
        overlap_end = min(refined_end, span.end)
        overlap_seconds = max(0.0, overlap_end - overlap_start)

        if overlap_seconds > 0:
            overlap_ratio = overlap_seconds / event_duration
            overlap_ratio = max(0.0, min(1.0, overlap_ratio))

            contexts.append(EventTopicContext(
                topic_id=span.topic_id,
                topic_start=span.start,
                topic_end=span.end,
                overlap_seconds=overlap_seconds,
                overlap_ratio=overlap_ratio,
                confidence=span.confidence,
                keywords=span.keywords,
                representative_text=span.representative_text,
                evidence_refs=span.evidence_refs,
            ))

    # Sort chronologically: topic_start ASC, topic_end ASC, topic_id ASC
    contexts.sort(key=lambda c: (c.topic_start, c.topic_end, c.topic_id))
    return tuple(contexts)


def _select_primary_topic(contexts: tuple[EventTopicContext, ...]) -> str | None:
    """Select primary topic deterministically.

    Rules:
    1. Highest overlap_seconds
    2. Highest TopicSpan.confidence
    3. Earliest topic_start
    4. topic_id lexicographically ASC
    """
    if not contexts:
        return None

    # Use min with negated descending criteria for exact 4-way tie-break
    # min(
    #     -overlap_seconds (highest overlap wins),
    #     -confidence (highest confidence wins),
    #     topic_start (earliest wins),
    #     topic_id (lexicographically ASC wins)
    # )
    best = min(
        contexts,
        key=lambda c: (
            -c.overlap_seconds,
            -c.confidence,
            c.topic_start,
            c.topic_id,
        ),
    )

    return best.topic_id


# ──────────────────────────────────────────────────────────────────────────────
# Evidence refs union
# ──────────────────────────────────────────────────────────────────────────────

def _union_evidence_refs(
    scored: ScoredCandidate,
    refinement: RefinedBoundaries,
    topic_contexts: tuple[EventTopicContext, ...],
) -> tuple[str, ...]:
    """Union of all evidence refs, deterministic sorted."""
    refs = set()

    # From component scores
    for comp in scored.component_scores:
        refs.update(comp.evidence_refs)

    # From refinement
    refs.update(refinement.evidence_refs)

    # From topic contexts
    for ctx in topic_contexts:
        refs.update(ctx.evidence_refs)

    return tuple(sorted(refs))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_event_pool(
    scored_candidates: list[ScoredCandidate],
    refinements: list[RefinedBoundaries],
    topic_timeline: TopicTimeline,
) -> EventPool:
    """Build deterministic event pool from scored candidates, refinements, and topic timeline.

    Args:
        scored_candidates: Output from highlight_scoring.score_candidates()
        refinements: Output from highlight_boundaries.refine_candidates()
        topic_timeline: Output from topic_timeline.build_topic_timeline()

    Returns:
        EventPool with items ordered by rank ASC (1.5C authority).

    Raises:
        ValueError: If inputs fail validation (fail-closed).
    """
    # Validate inputs
    scored_map = _validate_scored_candidates(scored_candidates)
    refinement_map = _validate_refinements(refinements)
    _validate_join(scored_map, refinement_map)

    # Build items in rank order (1.5C authority)
    # ScoredCandidate already has rank assigned, but ensure we iterate by rank
    scored_by_rank = sorted(scored_map.values(), key=lambda sc: sc.rank)

    items: list[EventPoolItem] = []

    for sc in scored_by_rank:
        rb = refinement_map[_stable_candidate_id_from_candidate(sc.candidate)]

        # Topic association using REFINED range (authoritative per spec)
        topic_contexts = _associate_topics(rb.refined_start, rb.refined_end, topic_timeline)
        primary_topic_id = _select_primary_topic(topic_contexts)

        # Evidence refs union
        evidence_refs = _union_evidence_refs(sc, rb, topic_contexts)

        # Convert component scores to EventScoreTrace
        score_components = tuple(
            EventScoreTrace(
                name=comp.name,
                raw_value=comp.raw_value,
                weight=comp.weight,
                contribution=comp.contribution,
                evidence_refs=tuple(sorted(comp.evidence_refs)),
                note=comp.note,
            )
            for comp in sc.component_scores
        )

        items.append(EventPoolItem(
            event_id=rb.candidate_id,  # Identity = refinement candidate_id = 1.5C/1.5D stable_id
            rank=sc.rank,
            overall_score=sc.overall_score,
            original_start=sc.candidate.start,
            original_end=sc.candidate.end,
            refined_start=rb.refined_start,
            refined_end=rb.refined_end,
            title=sc.candidate.title,
            hook=sc.candidate.hook,
            reason=sc.candidate.reason,
            refinement_status=rb.status.value if hasattr(rb.status, 'value') else str(rb.status),
            refinement_confidence=rb.confidence,
            peak_at=rb.peak_at,
            payoff_at=rb.payoff_at,
            payoff_window_start=rb.payoff_window_start,
            payoff_window_end=rb.payoff_window_end,
            reaction_at=rb.reaction_at,
            reaction_window_start=rb.reaction_window_start,
            reaction_window_end=rb.reaction_window_end,
            primary_topic_id=primary_topic_id,
            topic_contexts=topic_contexts,
            score_components=score_components,
            evidence_refs=evidence_refs,
        ))

    return EventPool(
        asset_id=topic_timeline.asset_id,
        items=tuple(items),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers (read-only, preserve pool order)
# ──────────────────────────────────────────────────────────────────────────────

def event_by_id(pool: EventPool, event_id: str) -> EventPoolItem | None:
    """Find event by exact event_id match."""
    for item in pool.items:
        if item.event_id == event_id:
            return item
    return None


def events_for_topic(pool: EventPool, topic_id: str) -> tuple[EventPoolItem, ...]:
    """Find events whose topic_contexts contain the given topic_id.

    Preserves original pool order.
    """
    result = []
    for item in pool.items:
        if any(ctx.topic_id == topic_id for ctx in item.topic_contexts):
            result.append(item)
    return tuple(result)


def events_for_range(pool: EventPool, start: float, end: float) -> tuple[EventPoolItem, ...]:
    """Find events whose REFINED range overlaps with query range.

    Preserves original pool order.
    """
    if end <= start:
        return ()

    result = []
    for item in pool.items:
        if item.refined_end > start and item.refined_start < end:
            result.append(item)
    return tuple(result)