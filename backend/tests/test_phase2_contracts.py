"""Unit tests for Phase 2 Evidence Request and Director Output contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.editor.evidence_request import (
    DirectorOutput,
    DirectorOutputType,
    EvidenceGap,
    EvidenceRequest,
    EvidenceRequestMode,
    EvidenceSufficiency,
    LoopState,
)
from app.editor.agent import EditorialPlan, PlannedOperation


class TestEvidenceRequest:
    """Tests for EvidenceRequest contract."""

    def test_valid_ask_video_request(self):
        """Valid ASK_VIDEO request with required fields."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Need to identify peak action",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
            question="What happens at the peak?",
        )
        assert req.mode == EvidenceRequestMode.ASK_VIDEO
        assert req.question == "What happens at the peak?"
        assert req.objective is None

    def test_valid_summarize_video_request(self):
        """Valid SUMMARIZE_VIDEO request with required fields."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.SUMMARIZE_VIDEO,
            reason="Need event boundaries",
            evidence_gap=EvidenceGap.EVENT_BOUNDARIES_UNCLEAR,
            objective="Summarize the event structure",
        )
        assert req.mode == EvidenceRequestMode.SUMMARIZE_VIDEO
        assert req.objective == "Summarize the event structure"
        assert req.question is None

    def test_ask_video_requires_question(self):
        """ASK_VIDEO without question should be allowed (validated at VIA layer)."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Test",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
        )
        assert req.question is None

    def test_summarize_video_requires_objective(self):
        """SUMMARIZE_VIDEO without objective should be allowed (validated at VIA layer)."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.SUMMARIZE_VIDEO,
            reason="Test",
            evidence_gap=EvidenceGap.EVENT_BOUNDARIES_UNCLEAR,
        )
        assert req.objective is None

    def test_end_us_must_be_greater_than_start_us(self):
        """end_us <= start_us should fail validation."""
        with pytest.raises(ValidationError):
            EvidenceRequest(
                base_version=1,
                asset_id="asset-123",
                start_us=10_000_000,
                end_us=5_000_000,
                mode=EvidenceRequestMode.ASK_VIDEO,
                reason="Test",
                evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
            )

    def test_base_version_required(self):
        """base_version is required."""
        with pytest.raises(ValidationError):
            EvidenceRequest(
                asset_id="asset-123",
                start_us=0,
                end_us=10_000_000,
                mode=EvidenceRequestMode.ASK_VIDEO,
                reason="Test",
                evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
            )

    def test_evidence_gap_enum_valid(self):
        """All EvidenceGap enum values should be valid."""
        for gap in EvidenceGap:
            req = EvidenceRequest(
                base_version=1,
                asset_id="asset-123",
                start_us=0,
                end_us=10_000_000,
                mode=EvidenceRequestMode.ASK_VIDEO,
                reason="Test",
                evidence_gap=gap,
            )
            assert req.evidence_gap == gap

    def test_to_video_intelligence_request(self):
        """Translation to VIA request works correctly."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Test",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
            question="What happens?",
            events_of_interest=["event-1"],
            objects_of_interest=["person"],
        )
        vi_req = req.to_video_intelligence_request()
        assert vi_req.asset_id == "asset-123"
        assert vi_req.start_us == 0
        assert vi_req.end_us == 10_000_000
        assert vi_req.mode == "ask_video"
        assert vi_req.question == "What happens?"
        assert vi_req.events_of_interest == ["event-1"]
        assert vi_req.objects_of_interest == ["person"]


class TestDirectorOutput:
    """Tests for DirectorOutput discriminated union."""

    def test_editorial_plan_output_valid(self):
        """Valid EditorialPlanOutput with all required fields."""
        plan = EditorialPlan(
            objective="Create highlight",
            output_intent="highlight",
            selected_events=["event-1"],
            operations=[PlannedOperation(type="trim", parameters={"clip_id": "c1", "source_out_us": 5_000_000})],
        )
        output = DirectorOutput(
            type=DirectorOutputType.EDITORIAL_PLAN,
            editorial_plan=plan,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            decision_rationale="Evidence sufficient for highlight",
        )
        assert output.type == DirectorOutputType.EDITORIAL_PLAN
        assert output.editorial_plan is not None
        assert output.evidence_request is None
        assert output.evidence_sufficiency == EvidenceSufficiency.SUFFICIENT

    def test_evidence_request_output_valid(self):
        """Valid EvidenceRequestOutput with all required fields."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="asset-123",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Need peak clarity",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
            question="What happens?",
        )
        output = DirectorOutput(
            type=DirectorOutputType.EVIDENCE_REQUEST,
            evidence_request=req,
            evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
            decision_rationale="Peak action unclear, need VIA",
        )
        assert output.type == DirectorOutputType.EVIDENCE_REQUEST
        assert output.evidence_request is not None
        assert output.editorial_plan is None
        assert output.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT

    def test_editorial_plan_output_requires_editorial_plan(self):
        """EditorialPlanOutput requires editorial_plan field."""
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EDITORIAL_PLAN,
                editorial_plan=None,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                decision_rationale="Test",
            )

    def test_evidence_request_output_requires_evidence_request(self):
        """EvidenceRequestOutput requires evidence_request field."""
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EVIDENCE_REQUEST,
                evidence_request=None,
                evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
                decision_rationale="Test",
            )

    def test_evidence_sufficiency_matches_type_editorial_plan(self):
        """evidence_sufficiency must be sufficient for editorial_plan."""
        plan = EditorialPlan(objective="Test", operations=[])
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EDITORIAL_PLAN,
                editorial_plan=plan,
                evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
                decision_rationale="Test",
            )

    def test_evidence_sufficiency_matches_type_evidence_request(self):
        """evidence_sufficiency must be insufficient for evidence_request."""
        req = EvidenceRequest(
            base_version=1,
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Test",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
        )
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EVIDENCE_REQUEST,
                evidence_request=req,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                decision_rationale="Test",
            )

    def test_decision_rationale_required(self):
        """decision_rationale is required for both types."""
        plan = EditorialPlan(objective="Test", operations=[])
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EDITORIAL_PLAN,
                editorial_plan=plan,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            )

        req = EvidenceRequest(
            base_version=1,
            asset_id="a",
            start_us=0,
            end_us=10_000_000,
            mode=EvidenceRequestMode.ASK_VIDEO,
            reason="Test",
            evidence_gap=EvidenceGap.PEAK_ACTION_UNCLEAR,
        )
        with pytest.raises(ValidationError):
            DirectorOutput(
                type=DirectorOutputType.EVIDENCE_REQUEST,
                evidence_request=req,
                evidence_sufficiency=EvidenceSufficiency.INSUFFICIENT,
            )

    def test_type_discriminator_required(self):
        """type field is required."""
        plan = EditorialPlan(objective="Test", operations=[])
        with pytest.raises(ValidationError):
            DirectorOutput(
                editorial_plan=plan,
                evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
                decision_rationale="Test",
            )


class TestLoopState:
    """Tests for LoopState."""

    def test_default_values(self):
        """LoopState has correct defaults."""
        state = LoopState(base_version=5)
        assert state.round == 0
        assert state.max_rounds == 1
        assert state.base_version == 5
        assert state.prior_evidence_request is None
        assert state.prior_via_result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])