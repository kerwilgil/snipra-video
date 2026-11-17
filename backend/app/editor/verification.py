"""Deterministic post-operation checks for Editor IA."""

from __future__ import annotations

from app.editor import timeline

# Fraction of an operation's source duration that must be covered by a single
# VIA evidence item's time range for that evidence to be considered a
# verifiable backing of the operation. This is a presence/overlap check, not
# an editorial quality score: confidence, coverage and evidence count are
# deliberately never read here (see Phase 3.5C).
VIA_EVIDENCE_OVERLAP_RATIO = 0.5


def verify(timeline_data: dict, asset_ids: set[str]) -> dict:
    issues: list[str] = []
    for segment in timeline.preview_segments(timeline_data):
        if segment["asset_id"] not in asset_ids:
            issues.append(f"MEDIA_MISSING:{segment['asset_id']}")
        if segment["source_out_us"] <= segment["source_in_us"]:
            issues.append(f"INVALID_RANGE:{segment['clip_id']}")
    integrity_status = "FAIL" if issues else "PASS"
    return {
        "status": integrity_status,
        "deterministic": {"status": integrity_status, "checks": {"issues": issues}},
        "editorial": {"status": "UNCERTAIN", "checks": [], "evidence": []},
        "issues": issues,
        "requires_human": False,
    }


def _operation_source_ranges(operations: list[dict]) -> list[tuple[int, int]]:
    """Extract (source_in_us, source_out_us) for operations that carry one."""
    ranges = []
    for op in operations:
        params = op.get("parameters") or {}
        start, end = params.get("source_in_us"), params.get("source_out_us")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            ranges.append((start, end))
    return ranges


def _via_backed_operations(
    operations: list[dict], via_evidence: list[dict]
) -> tuple[str, list[str]]:
    """Check whether every operation's source range is temporally backed by
    at least one VIA evidence item, without using confidence/coverage/count
    as an editorial score (see Phase 3.5C: verification checks *presence* of
    a verifiable relation, never how important VIA considers a moment)."""
    ranges = _operation_source_ranges(operations)
    if not ranges or not via_evidence:
        return "FAIL", []
    supporting_ids: list[str] = []
    for start, end in ranges:
        duration = end - start
        backed = False
        for item in via_evidence:
            ev_start, ev_end = item.get("start_us"), item.get("end_us")
            if not isinstance(ev_start, int) or not isinstance(ev_end, int):
                continue
            overlap = max(0, min(end, ev_end) - max(start, ev_start))
            if duration > 0 and overlap / duration >= VIA_EVIDENCE_OVERLAP_RATIO:
                backed = True
                evidence_id = item.get("evidence_id")
                if evidence_id:
                    supporting_ids.append(evidence_id)
        if not backed:
            return "FAIL", supporting_ids
    return "PASS", supporting_ids


def verify_editorial(
    plan: dict,
    event_pool: list[dict],
    transcript_segments: list[dict],
    via_evidence: list[dict] | None = None,
    enhanced_event_pool: dict | None = None,
) -> dict:
    """Evaluate only evidence already retrieved; never infer audiovisual facts."""
    intent = str(plan.get("output_intent") or "edit")
    selected = set(plan.get("selected_events") or [])
    via_evidence = via_evidence or []
    checks: list[dict] = []
    pool_by_id = {event["id"]: event for event in event_pool}

    # Enhanced event pool lookup (1.5H) - only structural validation
    enhanced_by_id: dict[str, dict] = {}
    enhanced_clusters: list[dict] = []
    if enhanced_event_pool:
        events_list = enhanced_event_pool.get("events", [])
        enhanced_by_id = {e["event_id"]: e for e in events_list}
        enhanced_clusters = enhanced_event_pool.get("clusters", [])
        enhanced_event_pool.get("diverse_order_event_ids", [])

    if intent == "highlight":
        sel_id = next(iter(selected), "") if len(selected) == 1 else ""
        
        # Determine source of selected event
        selected_in_legacy = sel_id in pool_by_id and sel_id
        selected_in_enhanced = sel_id in enhanced_by_id and sel_id

        # Case C: ID present in both legacy and enhanced -> ambiguity
        if selected_in_legacy and selected_in_enhanced:
            checks.append({
                "name": "ambiguous_event_id",
                "status": "FAIL",
                "event_id": sel_id,
                "reason": "selected event ID present in both legacy and enhanced pools",
            })
        elif selected_in_legacy:
            # Case A: legacy only - preserve exact legacy behavior
            event = pool_by_id.get(sel_id)
            if event and event.get("coherent") and event.get("quality_gate") == "KEEP":
                checks.append({"name": "event_coherence", "status": "PASS"})
            else:
                # Fall back to VIA evidence coherence per Phase 3.5C
                via_status, supporting_ids = _via_backed_operations(
                    plan.get("operations") or [], via_evidence
                )
                check = {"name": "via_evidence_coherence", "status": via_status}
                if supporting_ids:
                    check["supporting_evidence_ids"] = supporting_ids
                checks.append(check)
        elif selected_in_enhanced:
            # Case B: enhanced only - structural grounding only
            enhanced_item = enhanced_by_id[sel_id]
            
            # Structural grounding: selected ID exists in enhanced pool
            checks.append({
                "name": "enhanced_event_grounding",
                "status": "PASS",
                "event_id": sel_id,
            })
            # Refined range validity
            rs = enhanced_item.get("refined_start")
            re = enhanced_item.get("refined_end")
            if isinstance(rs, (int, float)) and isinstance(re, (int, float)) and re > rs:
                checks.append({
                    "name": "enhanced_refined_range_valid",
                    "status": "PASS",
                    "event_id": sel_id,
                })
            # Cluster membership integrity
            for cluster in enhanced_clusters:
                if sel_id in cluster.get("member_event_ids", []):
                    checks.append({
                        "name": "enhanced_cluster_membership",
                        "status": "PASS",
                        "event_id": sel_id,
                        "cluster_id": cluster.get("cluster_id"),
                        "is_representative": sel_id == cluster.get("representative_event_id"),
                    })
                    break
            
            # VIA evidence coherence for enhanced-only (if VIA sufficient)
            # If no VIA evidence, conservatively mark UNCERTAIN rather than FAIL
            via_status, supporting_ids = _via_backed_operations(
                plan.get("operations") or [], via_evidence
            )
            check = {"name": "via_evidence_coherence", "status": via_status if via_status != "FAIL" else "UNCERTAIN"}
            if supporting_ids:
                check["supporting_evidence_ids"] = supporting_ids
            checks.append(check)
        else:
            # Neither legacy nor enhanced - selected ID not found in enhanced pool
            # If enhanced pool was provided, this is a FAIL
            # If no enhanced pool, fall back to original VIA behavior
            if enhanced_event_pool:
                checks.append({
                    "name": "enhanced_event_grounding",
                    "status": "FAIL",
                    "event_id": sel_id,
                    "reason": "selected event ID not found in enhanced pool",
                })
            else:
                # No enhanced pool - use original VIA fallback behavior
                via_status, supporting_ids = _via_backed_operations(
                    plan.get("operations") or [], via_evidence
                )
                check = {"name": "via_evidence_coherence", "status": via_status}
                if supporting_ids:
                    check["supporting_evidence_ids"] = supporting_ids
                checks.append(check)

    elif intent == "montage":
        checks.append(
            {"name": "multiple_events", "status": "PASS" if len(selected) >= 2 else "FAIL"}
        )

        # Enhanced pool structural checks for montage
        if enhanced_event_pool:
            for sel_id in selected:
                if sel_id in enhanced_by_id:
                    checks.append({
                        "name": "enhanced_event_grounding",
                        "status": "PASS",
                        "event_id": sel_id,
                    })
                else:
                    checks.append({
                        "name": "enhanced_event_grounding",
                        "status": "FAIL",
                        "event_id": sel_id,
                        "reason": "selected event ID not found in enhanced pool",
                    })

    elif transcript_segments:
        continuity = (
            "PASS"
            if "continuity_preserved" in set(plan.get("verification_targets") or [])
            else "UNCERTAIN"
        )
        checks.append({"name": "continuity", "status": continuity})
    else:
        checks.append({"name": "timeline_operation_goals", "status": "PASS"})

    statuses = {item["status"] for item in checks}
    status = "FAIL" if "FAIL" in statuses else "UNCERTAIN" if "UNCERTAIN" in statuses else "PASS"
    evidence_sources = ["event_pool"]
    if transcript_segments:
        evidence_sources.append("transcript")
    if via_evidence:
        evidence_sources.append("via_evidence")
    if enhanced_event_pool:
        evidence_sources.append("enhanced_event_pool")
    return {
        "status": status,
        "checks": checks,
        "evidence": evidence_sources,
    }
