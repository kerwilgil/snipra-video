"""Phase 3.5C: verify_editorial() must recognize VIA evidence as a valid
backing for an EditorialPlan, without turning VIA into an editorial or
ranking authority."""

from __future__ import annotations

import copy
import json

from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import agent, service, verification
from app.editor.agent import EditorialPlan
from app.editor.models import EditorAgentRun


def _via_item(evidence_id: str, start_us: int, end_us: int, **overrides) -> dict:
    item = {
        "evidence_id": evidence_id,
        "asset_id": "asset-1",
        "start_us": start_us,
        "end_us": end_us,
        "evidence_kind": "event",
        "nature": "observation",
        "description": "rainbow gradient bar moves downward",
        "confidence_original": 0.95,
        "source": "via",
        "provider": "snipra_vlm",
        "model": "claude-haiku-4-5",
    }
    item.update(overrides)
    return item


def _highlight_plan(operations: list[dict], selected: list[str] | None = None) -> dict:
    return {
        "output_intent": "highlight",
        "selected_events": selected or [],
        "operations": operations,
    }


def _montage_plan(operations: list[dict], selected: list[str] | None = None) -> dict:
    return {
        "output_intent": "montage",
        "selected_events": selected or [],
        "operations": operations,
    }


def _insert_op(source_in_us: int, source_out_us: int) -> dict:
    return {
        "type": "insert",
        "parameters": {
            "asset_id": "asset-1",
            "source_in_us": source_in_us,
            "source_out_us": source_out_us,
            "timeline_start_us": 0,
        },
    }


def _enhanced_event_pool_base() -> dict:
    """Create a valid minimal enhanced_event_pool for testing."""
    return {
        "contract_version": "1.5H",
        "asset_id": "asset-1",
        "source_event_count": 2,
        "events": [
            {
                "event_id": "evt_1",
                "rank": 1,
                "overall_score": 0.9,
                "original_start": 0.0,
                "original_end": 10.0,
                "refined_start": 1.0,
                "refined_end": 9.0,
                "title": "Event 1",
                "hook": "Hook 1",
                "reason": "Reason 1",
                "refinement_status": "refined",
                "refinement_confidence": 0.9,
                "peak_at": 5.0,
                "payoff_at": 8.0,
                "payoff_window_start": 7.5,
                "payoff_window_end": 8.5,
                "reaction_at": 9.0,
                "reaction_window_start": 8.5,
                "reaction_window_end": 9.5,
                "primary_topic_id": "topic_1",
                "topic_contexts": [{
                    "topic_id": "topic_1",
                    "topic_start": 0.0,
                    "topic_end": 20.0,
                    "overlap_seconds": 8.0,
                    "overlap_ratio": 1.0,
                    "confidence": 0.95,
                    "keywords": ["gameplay"],
                    "representative_text": "We won",
                    "evidence_refs": ["via_1"]
                }],
                "score_components": [
                    {"name": "base_score", "raw_value": 0.8, "weight": 0.5, "contribution": 0.4, "evidence_refs": [], "note": "base"}
                ],
                "evidence_refs": ["via_1"]
            },
            {
                "event_id": "evt_2",
                "rank": 2,
                "overall_score": 0.7,
                "original_start": 20.0,
                "original_end": 30.0,
                "refined_start": 21.0,
                "refined_end": 29.0,
                "title": "Event 2",
                "hook": "Hook 2",
                "reason": "Reason 2",
                "refinement_status": "refined",
                "refinement_confidence": 0.8,
                "peak_at": 25.0,
                "payoff_at": 28.0,
                "payoff_window_start": 27.5,
                "payoff_window_end": 28.5,
                "reaction_at": None,
                "reaction_window_start": None,
                "reaction_window_end": None,
                "primary_topic_id": "topic_2",
                "topic_contexts": [],
                "score_components": [
                    {"name": "base_score", "raw_value": 0.6, "weight": 0.5, "contribution": 0.3, "evidence_refs": [], "note": "base"}
                ],
                "evidence_refs": []
            }
        ],
        "clusters": [
            {
                "cluster_id": "cl_1",
                "representative_event_id": "evt_1",
                "member_event_ids": ["evt_1", "evt_2"],
                "relations": [
                    {
                        "event_id": "evt_2",
                        "temporal_overlap_ratio": 0.0,
                        "same_primary_topic": False,
                        "peak_distance_seconds": 20.0,
                        "basis": ["moderate_temporal_overlap", "same_primary_topic"]
                    }
                ],
                "topic_ids": ["topic_1", "topic_2"],
                "refined_start": 1.0,
                "refined_end": 29.0
            }
        ],
        "diverse_order_event_ids": ["evt_1", "evt_2"]
    }


# --- 1. legacy event_pool válido -> PASS como antes -------------------------


def test_legacy_event_pool_pass_preserved():
    plan = _highlight_plan([], selected=["good"])
    pool = [{"id": "good", "coherent": True, "quality_gate": "KEEP"}]
    result = verification.verify_editorial(plan, pool, [])
    assert result["status"] == "PASS"
    assert result["checks"] == [{"name": "event_coherence", "status": "PASS"}]
    assert result["evidence"] == ["event_pool"]


# --- 2. legacy vacío + VIA válida respaldando la operación -> PASS ---------


def test_empty_pool_with_backing_via_evidence_passes():
    plan = _highlight_plan([_insert_op(6_000_000, 7_000_000)])
    via = [_via_item("ev1", 6_000_000, 7_000_000)]
    result = verification.verify_editorial(plan, [], [], via)
    assert result["status"] == "PASS"
    assert result["checks"] == [
        {"name": "via_evidence_coherence", "status": "PASS", "supporting_evidence_ids": ["ev1"]}
    ]
    assert result["evidence"] == ["event_pool", "via_evidence"]


# --- 3. legacy vacío + VIA irrelevante -> FAIL ------------------------------


def test_empty_pool_with_irrelevant_via_evidence_fails():
    plan = _highlight_plan([_insert_op(6_000_000, 7_000_000)])
    via = [_via_item("ev1", 0, 1_000_000)]  # no overlap at all with 6s-7s
    result = verification.verify_editorial(plan, [], [], via)
    assert result["status"] == "FAIL"
    assert result["checks"][0]["name"] == "via_evidence_coherence"
    assert "supporting_evidence_ids" not in result["checks"][0]


def test_via_evidence_below_overlap_threshold_does_not_auto_pass():
    """Case E: unattached VIA evidence exists but only marginally overlaps
    the decided range - must not be treated as a reasonable link."""
    plan = _highlight_plan([_insert_op(6_000_000, 7_000_000)])
    via = [_via_item("ev1", 6_800_000, 7_800_000)]  # 200ms of a 1s op = 20% < 50%
    result = verification.verify_editorial(plan, [], [], via)
    assert result["status"] == "FAIL"


# --- 4. legacy vacío + sin VIA -> FAIL --------------------------------------


def test_empty_pool_without_via_evidence_fails():
    plan = _highlight_plan([_insert_op(0, 1_000_000)])
    result = verification.verify_editorial(plan, [], [], None)
    assert result["status"] == "FAIL"
    assert result["evidence"] == ["event_pool"]


# --- 5. VIA evidence no altera legacy_score / pool ordering / Quality Gate -


def test_via_evidence_does_not_alter_legacy_pool_or_scoring():
    pool = [{"id": "good", "coherent": True, "quality_gate": "KEEP", "legacy_score": 0.42}]
    pool_snapshot = copy.deepcopy(pool)
    plan = _highlight_plan([_insert_op(6_000_000, 7_000_000)], selected=["good"])
    via = [_via_item("ev1", 6_000_000, 7_000_000)]

    result = verification.verify_editorial(plan, pool, [], via)

    assert result["status"] == "PASS"
    # Legacy path decided it, not the VIA fallback - even though VIA evidence
    # was present and would independently have passed too.
    assert result["checks"] == [{"name": "event_coherence", "status": "PASS"}]
    assert pool == pool_snapshot


def test_via_evidence_never_used_as_score_or_bonus():
    """Presence/overlap is binary; confidence, coverage and evidence count
    must never influence the check's outcome."""
    plan = _highlight_plan([_insert_op(6_000_000, 7_000_000)])
    low_confidence = [_via_item("ev1", 6_000_000, 7_000_000, confidence_original=0.01)]
    result_low = verification.verify_editorial(plan, [], [], low_confidence)
    high_confidence = [_via_item("ev2", 6_000_000, 7_000_000, confidence_original=0.99)]
    result_high = verification.verify_editorial(plan, [], [], high_confidence)
    assert result_low["status"] == result_high["status"] == "PASS"

    many_irrelevant = [_via_item(f"ev{i}", 0, 100_000) for i in range(50)]
    result_many = verification.verify_editorial(plan, [], [], many_irrelevant)
    assert result_many["status"] == "FAIL"  # quantity alone never passes


# --- 6. ENHANCED-ONLY HIGHLIGHT TESTS --------------------------------------


def test_enhanced_only_representative_structural_grounding():
    """A. enhanced-only representative: structural grounding PASS, no legacy FAIL."""
    enhanced = _enhanced_event_pool_base()
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    # No legacy pool, no VIA
    result = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced)
    
    # Should have structural grounding checks
    check_names = {c["name"] for c in result["checks"]}
    assert "enhanced_event_grounding" in check_names
    assert "enhanced_refined_range_valid" in check_names
    assert "enhanced_cluster_membership" in check_names
    
    # Should NOT have legacy FAIL
    check_statuses = {c["name"]: c["status"] for c in result["checks"]}
    assert check_statuses.get("enhanced_event_grounding") == "PASS"
    assert check_statuses.get("enhanced_refined_range_valid") == "PASS"
    assert check_statuses.get("enhanced_cluster_membership") == "PASS"
    
    # Representative is NOT editorial PASS
    cluster_check = next(c for c in result["checks"] if c["name"] == "enhanced_cluster_membership")
    assert cluster_check["is_representative"] is True
    # But no automatic editorial PASS from representative role


def test_enhanced_only_non_representative_selectable():
    """B. enhanced-only non-representative: selectable, no FAIL/UNCERTAIN from cluster role."""
    enhanced = _enhanced_event_pool_base()
    plan = _highlight_plan([_insert_op(21_000_000, 29_000_000)], selected=["evt_2"])
    result = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced)
    
    check_names = {c["name"] for c in result["checks"]}
    assert "enhanced_event_grounding" in check_names
    assert "enhanced_refined_range_valid" in check_names
    assert "enhanced_cluster_membership" in check_names
    
    # All structural checks PASS
    check_statuses = {c["name"]: c["status"] for c in result["checks"]}
    assert check_statuses.get("enhanced_event_grounding") == "PASS"
    assert check_statuses.get("enhanced_refined_range_valid") == "PASS"
    assert check_statuses.get("enhanced_cluster_membership") == "PASS"
    
    # Non-representative is NOT penalized
    cluster_check = next(c for c in result["checks"] if c["name"] == "enhanced_cluster_membership")
    assert cluster_check["is_representative"] is False
    # No FAIL/UNCERTAIN from being non-representative


def test_enhanced_only_without_via_uncertain():
    """C. enhanced-only without VIA: no FAIL, conservative UNCERTAIN."""
    enhanced = _enhanced_event_pool_base()
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    result = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced)
    
    # Should have VIA check but conservative (UNCERTAIN not FAIL)
    via_check = next(c for c in result["checks"] if c["name"] == "via_evidence_coherence")
    assert via_check["status"] in ("UNCERTAIN", "PASS")
    # If no VIA evidence, should be UNCERTAIN (conservative)
    if "supporting_evidence_ids" not in via_check:
        assert via_check["status"] == "UNCERTAIN"


def test_enhanced_only_with_via_backing():
    """D. enhanced-only with VIA backing: via_evidence_coherence PASS."""
    enhanced = _enhanced_event_pool_base()
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    via = [_via_item("ev1", 1_000_000, 9_000_000)]
    result = verification.verify_editorial(plan, [], [], via, enhanced)
    
    via_check = next(c for c in result["checks"] if c["name"] == "via_evidence_coherence")
    assert via_check["status"] == "PASS"
    assert via_check["supporting_evidence_ids"] == ["ev1"]


def test_ambiguous_event_id_in_both_pools():
    """E. ID present in both legacy and enhanced -> ambiguous_event_id FAIL."""
    enhanced = _enhanced_event_pool_base()
    legacy_pool = [{"id": "evt_1", "coherent": True, "quality_gate": "KEEP"}]
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    result = verification.verify_editorial(plan, legacy_pool, [], enhanced_event_pool=enhanced)
    
    check = next(c for c in result["checks"] if c["name"] == "ambiguous_event_id")
    assert check["status"] == "FAIL"
    assert check["event_id"] == "evt_1"
    assert "both legacy and enhanced" in check["reason"]


def test_enhanced_score_does_not_authorize():
    """F. Low score vs high score does not change structural result."""
    enhanced_low = _enhanced_event_pool_base()
    enhanced_low["events"][0]["overall_score"] = 0.1
    enhanced_high = _enhanced_event_pool_base()
    enhanced_high["events"][0]["overall_score"] = 0.99
    
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    
    result_low = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced_low)
    result_high = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced_high)
    
    # Structural grounding should be same regardless of score
    for name in ("enhanced_event_grounding", "enhanced_refined_range_valid", "enhanced_cluster_membership"):
        low_check = next(c for c in result_low["checks"] if c["name"] == name)
        high_check = next(c for c in result_high["checks"] if c["name"] == name)
        assert low_check["status"] == high_check["status"] == "PASS"


def test_enhanced_refinement_confidence_does_not_authorize():
    """G. Low refinement_confidence vs high does not change structural result."""
    enhanced_low = _enhanced_event_pool_base()
    enhanced_low["events"][0]["refinement_confidence"] = 0.1
    enhanced_high = _enhanced_event_pool_base()
    enhanced_high["events"][0]["refinement_confidence"] = 0.99
    
    plan = _highlight_plan([_insert_op(1_000_000, 9_000_000)], selected=["evt_1"])
    
    result_low = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced_low)
    result_high = verification.verify_editorial(plan, [], [], enhanced_event_pool=enhanced_high)
    
    for name in ("enhanced_event_grounding", "enhanced_refined_range_valid", "enhanced_cluster_membership"):
        low_check = next(c for c in result_low["checks"] if c["name"] == name)
        high_check = next(c for c in result_high["checks"] if c["name"] == name)
        assert low_check["status"] == high_check["status"] == "PASS"


# --- 7/8. requires_confirmation semantics (untouched, now documented) ------


def _project(session: Session, tmp_path):
    project = service.create_project(session, "Verification E2E")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    service.add_asset(session, project, source, duration_us=20_000_000, has_audio=True)
    return project


def _complete(payload: dict):
    def complete(_system: str, _user: str) -> str:
        return json.dumps(payload)

    return complete


def test_requires_confirmation_false_executes_directly(tmp_path):
    init_db()
    with Session(engine) as session:
        project = _project(session, tmp_path)
        plan = {
            "objective": "Ajuste simple",
            "requires_confirmation": False,
            "operations": [],
        }
        result = agent.run_request(
            session,
            project.id,
            "Haz un ajuste simple.",
            complete=_complete(plan),
            key_prefix="conf-false",
        )
        assert result["status"] != "NEEDS_USER"
        run = session.get(EditorAgentRun, result["run_id"])
        assert run.status != "NEEDS_USER"
        assert run.pending_plan is None


def test_requires_confirmation_true_needs_user_then_confirm_executes(tmp_path):
    init_db()
    with Session(engine) as session:
        project = _project(session, tmp_path)
        plan = {
            "objective": "Cambio que requiere confirmación",
            "requires_confirmation": True,
            "operations": [],
        }
        result = agent.run_request(
            session,
            project.id,
            "Haz un cambio grande.",
            complete=_complete(plan),
            key_prefix="conf-true",
        )
        assert result["status"] == "NEEDS_USER"
        run = session.get(EditorAgentRun, result["run_id"])
        assert run.status == "NEEDS_USER"
        assert run.pending_plan is not None

        # Mirrors POST /agent/{run_id}/confirm (app/api/editor.py) exactly.
        confirmed_plan = EditorialPlan.model_validate(run.pending_plan["plan"])
        executed = agent.execute_plan(
            session, project.id, confirmed_plan, key_prefix=f"confirm:{run.id}"
        )
        assert executed["version"] >= run.base_version
