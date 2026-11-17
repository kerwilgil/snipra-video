"""Phase 3.5D: VIA evidence must survive the NEEDS_USER -> POST /confirm pause.

Phase 3.5C made verify_editorial() VIA-aware, but only the direct-execution
(requires_confirmation=false) path threaded via_evidence through - /confirm
rebuilt evidence from scratch without it, so a plan that legitimately paused
for human confirmation lost the Verification fix entirely.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.db import engine, init_db
from app.editor import agent, service
from app.editor.director_bridge import validate_director_evidence, BridgeValidationError
from app.editor.models import EditorAgentRun
from app.editor.via_evidence import NormalizedVideoEvidence
from app.main import app
from app.video_intelligence.models import (
    CoverageStatus,
    VideoIntelligenceResult,
    VideoObservation,
    VideoProcessingMetadata,
)


def _project(session: Session, tmp_path):
    project = service.create_project(session, "Confirmation VIA Evidence")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    asset = service.add_asset(session, project, source, duration_us=20_000_000, has_audio=True)
    return project, asset


def _director_envelope(director_type: str, **kwargs) -> dict:
    base = {"type": director_type, "evidence_sufficiency": None, "decision_rationale": "test"}
    if director_type == "editorial_plan":
        base["evidence_sufficiency"] = "sufficient"
        base["editorial_plan"] = kwargs["editorial_plan"]
        base["evidence_request"] = None
    else:
        base["evidence_sufficiency"] = "insufficient"
        base["evidence_request"] = kwargs["evidence_request"]
        base["editorial_plan"] = None
    return base


def _via_result(asset_id: str, obs_start_us: int, obs_end_us: int) -> dict:
    return VideoIntelligenceResult(
        asset_id=asset_id,
        start_us=0,
        end_us=20_000_000,
        mode="ask_video",
        observations=[
            VideoObservation(
                asset_id=asset_id,
                start_us=obs_start_us,
                end_us=obs_end_us,
                description="Rainbow gradient shifts downward",
                nature="observation",
                confidence=0.95,
                source="snipra_vlm",
                provider="snipra_vlm",
                model="test",
            )
        ],
        metadata=VideoProcessingMetadata(
            provider="snipra_vlm",
            model="test",
            requested_frames=5,
            analyzed_frames=5,
            coverage_ratio=1.0,
            coverage_status=CoverageStatus.COMPLETE,
        ),
    ).model_dump()


def _needs_user_run(
    session,
    monkeypatch,
    project,
    asset,
    *,
    op_start_us: int,
    op_end_us: int,
    key_prefix: str,
    via_start_us: int | None = None,
    via_end_us: int | None = None,
    requires_confirmation: bool = True,
) -> str:
    """Drive agent.run_request() to a real NEEDS_USER/PASS state, optionally
    threading a real Director evidence_request -> VIA -> editorial_plan
    augmentation round through it."""
    track_id = service.current_version(session, project).timeline["tracks"][0]["id"]
    editorial_plan = {
        "objective": "Highlight the moment",
        "output_intent": "highlight",
        "operations": [
            {
                "type": "insert",
                "parameters": {
                    "asset_id": asset.id,
                    "track_id": track_id,
                    "source_in_us": op_start_us,
                    "source_out_us": op_end_us,
                    "timeline_start_us": 0,
                },
            }
        ],
        "requires_confirmation": requires_confirmation,
    }
    editorial_envelope = _director_envelope("editorial_plan", editorial_plan=editorial_plan)

    if via_start_us is not None:
        evidence_request_envelope = _director_envelope(
            "evidence_request",
            evidence_request={
                "base_version": project.current_version,
                "asset_id": asset.id,
                "start_us": 0,
                "end_us": 20_000_000,
                "mode": "ask_video",
                "reason": "Need to see the clip to confirm the highlight moment.",
                "evidence_gap": "visual_evidence_absent",
                "question": "Where does the moment happen?",
            },
        )
        responses = iter([json.dumps(evidence_request_envelope), json.dumps(editorial_envelope)])

        async def fake_execute_via_request(_evidence_request):
            return _via_result(asset.id, via_start_us, via_end_us)

        monkeypatch.setattr(agent, "_execute_via_request", fake_execute_via_request)
    else:
        responses = iter([json.dumps(editorial_envelope)])

    def complete(_system, _user):
        return next(responses)

    result = agent.run_request(
        session,
        project.id,
        "Crea un highlight del momento en que el gradiente se desplaza.",
        intent="highlight",
        complete=complete,
        key_prefix=key_prefix,
    )
    return result


# --- Section 6: roundtrip -----------------------------------------------


def test_via_evidence_roundtrip_through_confirmation_persistence():
    """NormalizedVideoEvidence -> JSON-safe dict (what pending_plan actually
    stores) -> rehydrate_via_evidence -> re-validated dict, preserving the
    fields Verification/Confirmation depend on."""
    original = NormalizedVideoEvidence(
        evidence_id="ev-1",
        asset_id="asset-1",
        start_us=1_000_000,
        end_us=2_000_000,
        evidence_kind="observation",
        nature="observation",
        description="Rainbow gradient shifts downward",
        confidence_original=0.95,
        provider="snipra_vlm",
        model="claude-haiku-4-5",
    )
    persisted = original.model_dump(mode="json")

    restored = agent.rehydrate_via_evidence([persisted])

    assert len(restored) == 1
    item = restored[0]
    assert item["evidence_id"] == "ev-1"
    assert item["asset_id"] == "asset-1"
    assert item["start_us"] == 1_000_000
    assert item["end_us"] == 2_000_000
    assert item["evidence_kind"] == "observation"
    assert item["nature"] == "observation"
    assert item["description"] == "Rainbow gradient shifts downward"
    assert item["provider"] == "snipra_vlm"
    assert item["model"] == "claude-haiku-4-5"
    assert item["confidence_original"] == 0.95


def test_rehydrate_via_evidence_drops_corrupted_items_without_crashing():
    valid = NormalizedVideoEvidence(
        evidence_id="ev-1",
        asset_id="asset-1",
        start_us=0,
        end_us=1_000_000,
        evidence_kind="observation",
        nature="observation",
        description="ok",
        confidence_original=0.9,
        provider="snipra_vlm",
        model="test",
    ).model_dump(mode="json")
    corrupted = {"evidence_id": "ev-2"}  # missing required fields

    restored = agent.rehydrate_via_evidence([valid, corrupted])

    assert len(restored) == 1
    assert restored[0]["evidence_id"] == "ev-1"


def test_rehydrate_via_evidence_handles_none_and_empty():
    assert agent.rehydrate_via_evidence(None) == []
    assert agent.rehydrate_via_evidence([]) == []


# --- Section 5: regression reproducing the loss --------------------------


def test_confirmation_rehydrates_via_evidence_used_before_pause(monkeypatch, tmp_path):
    """Fails against the Phase 3.5C HEAD (bb46b76/f735e44), where /confirm
    rebuilt `evidence` without a via_evidence key at all: Director Call 1 ->
    EvidenceRequest -> VIA -> normalized evidence -> Director Call 2 ->
    EditorialPlan(requires_confirmation=true) -> NEEDS_USER -> POST /confirm
    must receive exactly the via_evidence that grounded the plan."""
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            via_start_us=1_000_000,
            via_end_us=2_000_000,
            key_prefix="repro",
        )
        assert result["status"] == "NEEDS_USER"
        run_id = result["run_id"]
        run = session.get(EditorAgentRun, run_id)
        assert run.pending_plan["evidence"].get("via_evidence"), (
            "via_evidence must be persisted in pending_plan before NEEDS_USER"
        )
        project_id = project.id

    with TestClient(app) as client:
        resp = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")
    assert resp.status_code == 200
    body = resp.json()
    checks = body["verification"]["editorial"]["checks"]
    assert any(c["name"] == "via_evidence_coherence" and c["status"] == "PASS" for c in checks)
    assert body["verification"]["editorial"]["status"] == "PASS"
    assert "via_evidence" in body["verification"]["editorial"]["evidence"]


# --- Section 7: confirmation=true + relevant VIA -> PASS -----------------


def test_confirmation_true_with_relevant_via_evidence_passes(monkeypatch, tmp_path):
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            via_start_us=1_000_000,
            via_end_us=2_000_000,
            key_prefix="relevant",
        )
        run_id = result["run_id"]
        project_id = project.id

    with TestClient(app) as client:
        resp = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")
    assert resp.status_code == 200
    assert resp.json()["verification"]["editorial"]["status"] == "PASS"


# --- Section 8: confirmation=true + irrelevant VIA -> FAIL ----------------


def test_confirmation_true_with_irrelevant_via_evidence_fails(monkeypatch, tmp_path):
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            via_start_us=15_000_000,  # nowhere near the operation's range
            via_end_us=16_000_000,
            key_prefix="irrelevant",
        )
        run_id = result["run_id"]
        project_id = project.id

    with TestClient(app) as client:
        resp = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")
    assert resp.status_code == 200
    body = resp.json()
    checks = body["verification"]["editorial"]["checks"]
    assert any(c["name"] == "via_evidence_coherence" and c["status"] == "FAIL" for c in checks)
    assert body["verification"]["editorial"]["status"] == "FAIL"


# --- Section 9: confirmation=true + no VIA -> legacy behavior ------------


def test_confirmation_true_without_via_evidence_keeps_legacy_behavior(monkeypatch, tmp_path):
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            key_prefix="no-via",
        )
        run_id = result["run_id"]
        run = session.get(EditorAgentRun, run_id)
        assert "via_evidence" not in run.pending_plan["evidence"]
        project_id = project.id

    with TestClient(app) as client:
        resp = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")
    assert resp.status_code == 200
    body = resp.json()
    assert body["verification"]["editorial"]["status"] == "FAIL"
    assert "via_evidence" not in body["verification"]["editorial"]["evidence"]


# --- Section 10: direct execution (requires_confirmation=false) unchanged -


def test_direct_execution_without_confirmation_still_uses_via_evidence(monkeypatch, tmp_path):
    """The Phase 3.5C direct-execution path never touches pending_plan at
    all - this persistence fix must not change its behavior."""
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            via_start_us=1_000_000,
            via_end_us=2_000_000,
            key_prefix="direct",
            requires_confirmation=False,
        )
        assert result["status"] == "PASS"
        assert result["verification"]["editorial"]["status"] == "PASS"
        run = session.get(EditorAgentRun, result["run_id"])
        assert run.pending_plan is None


# --- Section 11: /confirm idempotency preserved ---------------------------


def test_confirm_is_idempotent_and_does_not_duplicate_operations(monkeypatch, tmp_path):
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        result = _needs_user_run(
            session,
            monkeypatch,
            project,
            asset,
            op_start_us=1_000_000,
            op_end_us=2_000_000,
            via_start_us=1_000_000,
            via_end_us=2_000_000,
            key_prefix="idem",
        )
        run_id = result["run_id"]
        project_id = project.id

    with TestClient(app) as client:
        first = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")
        second = client.post(f"/api/editor/projects/{project_id}/agent/{run_id}/confirm")

    assert first.status_code == 200
    assert second.status_code == 200
    first_body, second_body = first.json(), second.json()
    assert first_body["version"] == second_body["version"]
    assert [op["operation_id"] for op in first_body["operations"]] == [
        op["operation_id"] for op in second_body["operations"]
    ]
    assert second_body["operations"][0]["idempotent"] is True


# --- Section 12: version invariants ---------------------------------------


def _enhanced_event_pool_for_test() -> dict:
    """Create a valid enhanced_event_pool for confirmation tests."""
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


# --- Section 13: 1.5H Enhanced Event Pool Confirmation ----------------------


def test_enhanced_persisted_in_pending_plan(monkeypatch, tmp_path):
    """Enhanced event pool is persisted in pending_plan.evidence."""
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        _enhanced_event_pool_for_test()
        
        # Drive to NEEDS_USER with enhanced
        track_id = service.current_version(session, project).timeline["tracks"][0]["id"]
        editorial_plan = {
            "objective": "Highlight the moment",
            "output_intent": "highlight",
            "operations": [
                {
                    "type": "insert",
                    "parameters": {
                        "asset_id": asset.id,
                        "track_id": track_id,
                        "source_in_us": 1_000_000,
                        "source_out_us": 2_000_000,
                        "timeline_start_us": 0,
                    },
                }
            ],
            "output_intent": "highlight",
            "requires_confirmation": True,
        }
        editorial_envelope = {
            "type": "editorial_plan",
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
            "editorial_plan": editorial_plan,
            "evidence_request": None,
        }
        responses = iter([json.dumps(editorial_envelope)])
        
        def complete(_system, _user):
            return next(responses)
        
        result = agent.run_request(
            session,
            project.id,
            "Test enhanced confirm",
            intent="highlight",
            complete=complete,
            key_prefix="enhanced-confirm",
            enhanced_event_pool=validate_director_evidence(_enhanced_event_pool_for_test()),
        )
        
        assert result["status"] == "NEEDS_USER"
        run_id = result["run_id"]
        run = session.get(EditorAgentRun, run_id)
        
        # Enhanced must be in persisted evidence
        assert run.pending_plan["evidence"].get("enhanced_event_pool") is not None
        persisted = run.pending_plan["evidence"]["enhanced_event_pool"]
        assert persisted["contract_version"] == "1.5H"
        assert len(persisted["events"]) == 2
        assert persisted["diverse_order_event_ids"] == ["evt_1", "evt_2"]


def test_confirm_uses_persisted_enhanced(monkeypatch, tmp_path):
    """Confirm uses persisted enhanced_event_pool, does not reconstruct C-G."""
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        _enhanced_event_pool_for_test()
        
        track_id = service.current_version(session, project).timeline["tracks"][0]["id"]
        editorial_plan = {
            "objective": "Highlight",
            "output_intent": "highlight",
            "selected_events": ["evt_1"],
            "operations": [{
                "type": "insert",
                "parameters": {
                    "asset_id": asset.id,
                    "track_id": track_id,
                    "source_in_us": 1_000_000,
                    "source_out_us": 2_000_000,
                    "timeline_start_us": 0,
                },
            }],
            "output_intent": "highlight",
            "requires_confirmation": True,
        }
        editorial_envelope = {
            "type": "editorial_plan",
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
            "editorial_plan": editorial_plan,
            "evidence_request": None,
        }
        responses = iter([json.dumps(editorial_envelope)])
        
        def complete(_system, _user):
            return next(responses)
        
        validated = validate_director_evidence(_enhanced_event_pool_for_test())
        # Update enhanced pool with correct asset ID
        validated["asset_id"] = asset.id
        for event in validated["events"]:
            for ctx in event.get("topic_contexts", []):
                ctx["evidence_refs"] = [ref.replace("via_1", f"via_{asset.id}") for ref in ctx.get("evidence_refs", [])]
            event["evidence_refs"] = [ref.replace("via_1", f"via_{asset.id}") for ref in event.get("evidence_refs", [])]
            for comp in event.get("score_components", []):
                comp["evidence_refs"] = [ref.replace("via_1", f"via_{asset.id}") for ref in comp.get("evidence_refs", [])]
        
        for cluster in validated.get("clusters", []):
            for rel in cluster.get("relations", []):
                if "basis" in rel:
                    rel["basis"] = [b.replace("via_1", f"via_{asset.id}") for b in rel["basis"]]
        
        result = agent.run_request(
            session,
            project.id,
            "Test",
            intent="highlight",
            complete=lambda *a: next(responses),
            key_prefix="enhanced",
            enhanced_event_pool=validated,
        )
        
        run_id = result["run_id"]
        project_id = project.id
    
    # Confirm directly using same session (bypasses TestClient transaction isolation)
    from app.api.editor import confirm_agent_plan
    from fastapi import HTTPException
    try:
        result = confirm_agent_plan(project_id=project_id, run_id=run_id, session=session)
        body = result
    except HTTPException as e:
        body = e.detail
    
    # Should succeed with enhanced structural checks (PASS or UNCERTAIN without VIA evidence)
    assert body["verification"]["editorial"]["status"] in ("PASS", "UNCERTAIN")
    checks = {c["name"]: c["status"] for c in body["verification"]["editorial"]["checks"]}
    assert "enhanced_event_grounding" in checks
    assert checks["enhanced_event_grounding"] == "PASS"
    # No C-G reconstruction - enhanced came from persisted copy


def test_corrupt_enhanced_confirm_fails_closed(monkeypatch, tmp_path):
    """Corrupt persisted enhanced -> confirm fails with ENHANCED_EVIDENCE_CORRUPT, 0 ops."""
    init_db()
    with Session(engine) as session:
        project, asset = _project(session, tmp_path)
        
        track_id = service.current_version(session, project).timeline["tracks"][0]["id"]
        editorial_plan = {
            "objective": "Test",
            "output_intent": "highlight",
            "selected_events": ["evt_1"],
            "operations": [{
                "type": "insert",
                "parameters": {
                    "asset_id": asset.id,
                    "track_id": track_id,
                    "source_in_us": 1_000_000,
                    "source_out_us": 2_000_000,
                    "timeline_start_us": 0,
                },
            }],
            "output_intent": "highlight",
            "requires_confirmation": True,
        }
        editorial_envelope = {
            "type": "editorial_plan",
            "evidence_sufficiency": "sufficient",
            "decision_rationale": "test",
            "editorial_plan": editorial_plan,
            "evidence_request": None,
        }
        responses = iter([json.dumps(editorial_envelope)])
        
        def complete(_system, _user):
            return next(responses)
        
        validated = validate_director_evidence(_enhanced_event_pool_for_test())
        result = agent.run_request(
            session,
            project.id,
            "Test",
            intent="highlight",
            complete=lambda *a: next(responses),
            key_prefix="corrupt",
            enhanced_event_pool=validated,
        )
        
        run_id = result["run_id"]
        
        # Corrupt the persisted enhanced_event_pool in DB
        run = session.get(EditorAgentRun, run_id)
        evidence = run.pending_plan["evidence"]
        enhanced = evidence.get("enhanced_event_pool")
        if enhanced and "clusters" in enhanced and enhanced["clusters"]:
            enhanced["clusters"][0]["member_event_ids"].append("unknown-event")
        # Force SQLAlchemy to detect the change
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(run, "pending_plan")
        session.add(run)
        session.commit()
        
        project_id = project.id
    
    # Confirm directly using same session
    from app.api.editor import confirm_agent_plan
    from fastapi import HTTPException
    try:
        result = confirm_agent_plan(project_id=project_id, run_id=run_id, session=session)
    except HTTPException as e:
        assert e.status_code == 409
        assert e.detail["code"] == "ENHANCED_EVIDENCE_CORRUPT"
        # No operations executed
        return
    
    assert False, "Expected HTTPException with ENHANCED_EVIDENCE_CORRUPT"
