"""Read-only adapter from existing Auto candidates to Editor IA event evidence."""

from __future__ import annotations

from sqlmodel import Session, select

from app.editor.via_evidence import NormalizedVideoEvidence, deduplicate_evidence
from app.models import Clip


def events_for_job(
    session: Session, job_id: str | None, via_evidence: list[NormalizedVideoEvidence] | None = None
) -> list[dict]:
    """Get legacy events for a job, optionally enriched with VIA normalized evidence."""
    if not job_id:
        return []
    clips = session.exec(select(Clip).where(Clip.job_id == job_id).order_by(Clip.index)).all()
    legacy_events = [
        {
            "id": clip.id,
            "start_us": round(clip.start * 1_000_000),
            "end_us": round(clip.end * 1_000_000),
            "title": clip.title,
            "reason": clip.reason,
            "legacy_score": clip.score,
            "source_ranges": clip.source_ranges,
            "evidence": clip.score_breakdown,
            "uncertain": any(
                item.get("metadata", {})
                .get("structured_event_features", {})
                .get("outcome")
                == "uncertain"
                for item in clip.score_breakdown
            ),
        }
        for clip in clips
    ]

    if not via_evidence:
        return legacy_events

    # Deduplicate VIA evidence first (conservative)
    deduped_via = deduplicate_evidence(via_evidence)

    # Match VIA evidence to legacy events
    return _merge_via_evidence(legacy_events, deduped_via)


def _merge_via_evidence(
    legacy_events: list[dict], via_evidence: list
) -> dict:
    """
    Conservatively attach VIA evidence to legacy events.

    Matching criteria (all must match):
    - same asset_id
    - temporal overlap >= 50% of shorter duration
    - VIA evidence not already attached

    If multiple legacy events match, select best by:
    1. Highest relative overlap ratio
    2. Closest temporal center distance
    3. Stable event ID tie-breaker

    If no match found, VIA evidence remains as unattached additional evidence.

    Returns dict with "events" (enriched legacy) and "unattached_additional_evidence".
    """
    if not via_evidence:
        return {"events": legacy_events, "unattached_additional_evidence": []}

    # Build a map of via_idx -> list of (legacy_idx, overlap_info)
    # where overlap_info = (relative_overlap, center_distance, legacy_id, legacy_idx)
    matches = {}  # via_idx -> list of (legacy_idx, rel_overlap, center_dist, legacy_id)

    for vi_idx, via in enumerate(via_evidence):
        if via.asset_id is None:
            continue
        via_center = (via.start_us + via.end_us) / 2

        for leg_idx, legacy in enumerate(legacy_events):
            if legacy.get("asset_id") != via.asset_id:
                continue

            legacy_start = legacy.get("start_us", 0)
            legacy_end = legacy.get("end_us", 0)
            overlap = _temporal_overlap_us(legacy_start, legacy_end, via.start_us, via.end_us)
            if overlap == 0:
                continue

            legacy_dur = legacy_end - legacy_start
            via_dur = via.end_us - via.start_us
            min_dur = min(legacy_dur, via_dur)
            if min_dur == 0 or (overlap / min_dur) < 0.5:
                continue

            # Relative overlap of the shorter duration
            rel_overlap = overlap / min_dur
            legacy_center = (legacy_start + legacy_end) / 2
            center_dist = abs(via_center - legacy_center)
            legacy_id = legacy.get("id", f"legacy-{leg_idx}")

            if vi_idx not in matches:
                matches[vi_idx] = []
            matches[vi_idx].append((leg_idx, rel_overlap, center_dist, legacy_id))

    # For each VIA evidence, pick the best match
    matched_via = set()
    legacy_attachments = {leg_idx: [] for leg_idx in range(len(legacy_events))}

    for vi_idx, candidates in matches.items():
        if not candidates:
            continue
        # Sort by: highest relative overlap, then closest center, then lowest legacy ID (stable tie-break)
        candidates.sort(key=lambda x: (-x[1], x[2], x[3]))
        best_leg_idx, _, _, _ = candidates[0]
        legacy_attachments[best_leg_idx].append(via_evidence[vi_idx])
        matched_via.add(vi_idx)

    # Build enriched events
    enriched_events = []
    for leg_idx, legacy in enumerate(legacy_events):
        attached = legacy_attachments.get(leg_idx, [])
        if attached:
            enriched = {**legacy, "additional_evidence": [_via_to_event_evidence(v) for v in attached]}
        else:
            enriched = legacy
        enriched_events.append(enriched)

    # Unattached VIA evidence
    unattached = [_via_to_event_evidence(via_evidence[vi_idx], unattached=True)
                  for vi_idx in range(len(via_evidence)) if vi_idx not in matches or vi_idx not in matched_via]

    return {
        "events": enriched_events,
        "unattached_additional_evidence": unattached,
    }


def _temporal_overlap_us(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Compute temporal overlap in microseconds."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _via_to_event_evidence(via, unattached: bool = False) -> dict:
    """Convert NormalizedVideoEvidence to event-compatible dict."""
    return {
        "evidence_id": via.evidence_id,
        "asset_id": via.asset_id,
        "start_us": via.start_us,
        "end_us": via.end_us,
        "evidence_kind": via.evidence_kind.value,
        "nature": via.nature.value,
        "description": via.description,
        "confidence_original": via.confidence_original,
        "source": via.source,
        "provider": via.provider,
        "model": via.model,
        "chunk_id": via.chunk_id,
        "contract_version": via.contract_version,
        "partial": via.partial,
        "failed_chunks": via.failed_chunks,
        "unattached": unattached,
        # Coverage metadata (Phase 3.4)
        "requested_frames": via.requested_frames,
        "analyzed_frames": via.analyzed_frames,
        "coverage_ratio": via.coverage_ratio,
        "coverage_status": via.coverage_status.value if via.coverage_status else None,
        # Optional enriched fields
        "visual_novelty": via.visual_novelty,
        "payoff_present": via.payoff_present,
        "reaction_present": via.reaction_present,
        "evidence_quality": via.evidence_quality,
        "scene_type": via.scene_type,
        "scene_continuity": via.scene_continuity,
        "object_label": via.object_label,
        "object_track_id": via.object_track_id,
        "object_interaction": via.object_interaction,
    }


def best_event_pool(events: list[dict], intent: str | None = None) -> list[dict]:
    """Inspectable pool; legacy score sorts only as a secondary signal.

    Unattached VIA evidence (unattached=True) is excluded from the pool
    to prevent it from affecting ranking.
    """
    # Filter out unattached VIA evidence
    pool_events = [e for e in events if not e.get("unattached")]
    pool = []
    for event in pool_events:
        ranges = event.get("source_ranges")
        # For unattached VIA evidence (no source_ranges), treat as coherent
        coherent = not ranges or len(ranges) <= 1
        output_suitable = not (intent == "highlight" and not coherent)
        pool.append(
            {
                **event,
                "coherent": coherent,
                "output_suitable": output_suitable,
                "quality_gate": "REWORK" if not coherent else "KEEP",
            }
        )
    return sorted(
        pool,
        key=lambda event: (event.get("output_suitable", True), event.get("legacy_score", 0)),
        reverse=True,
    )
