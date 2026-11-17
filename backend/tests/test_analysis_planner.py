"""Tests for the Phase 4.1 Slice 1 Adaptive Analysis Planner.

Pure, unwired: plan_analysis() is not called by VideoIntelligenceService or
ExistingSnipraVLMProvider yet. These tests exercise the planner in isolation.
"""

from __future__ import annotations

import json

import pytest

from app.video_intelligence.chunking import (
    FULL_SCAN_MAX_S,
    MAX_CHUNK_DURATION_S,
    MAX_CHUNKS_PER_REQUEST,
    MAX_FRAMES_PER_CHUNK,
    AnalysisChunk,
    AnalysisStrategy,
    VideoAnalysisPlan,
    plan_analysis,
)
from app.video_intelligence.models import (
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceRequest,
)


def _request(
    duration_s: float,
    mode: str = "summarize_video",
    start_s: float = 0.0,
) -> VideoIntelligenceRequest:
    kwargs = {}
    if mode == "ask_video":
        kwargs["question"] = "what happens?"
    else:
        kwargs["objective"] = "summarize"
    return VideoIntelligenceRequest(
        asset_id="asset-1",
        start_us=int(start_s * 1_000_000),
        end_us=int((start_s + duration_s) * 1_000_000),
        mode=mode,
        **kwargs,
    )


class TestStrategyBoundaries:
    def test_short_range_is_full_scan(self):
        plan = plan_analysis(_request(8, mode="summarize_video"))
        assert plan.strategy == AnalysisStrategy.FULL_SCAN
        assert len(plan.chunks) == 1

    def test_exactly_at_full_scan_ceiling_is_full_scan(self):
        plan = plan_analysis(_request(FULL_SCAN_MAX_S, mode="summarize_video"))
        assert plan.strategy == AnalysisStrategy.FULL_SCAN
        assert len(plan.chunks) == 1

    def test_just_past_full_scan_ceiling_ask_video_is_targeted(self):
        plan = plan_analysis(_request(FULL_SCAN_MAX_S + 1, mode="ask_video"))
        assert plan.strategy == AnalysisStrategy.TARGETED_SCAN

    def test_just_past_full_scan_ceiling_summarize_is_coarse(self):
        plan = plan_analysis(_request(FULL_SCAN_MAX_S + 1, mode="summarize_video"))
        assert plan.strategy == AnalysisStrategy.COARSE_SCAN

    def test_long_ask_video_is_targeted_not_rejected(self):
        """Sign-off #1: no hard 10-minute cutoff for ask_video. A long
        targeted range must still produce a valid multi-chunk plan."""
        plan = plan_analysis(_request(1200, mode="ask_video"))  # 20 min
        assert plan.strategy == AnalysisStrategy.TARGETED_SCAN
        assert len(plan.chunks) > 1


class TestBudgetInvariants:
    def test_no_chunk_exceeds_max_frames(self):
        for duration_s in (8, 60, 300, 1800, 7200):
            plan = plan_analysis(_request(duration_s))
            for chunk in plan.chunks:
                assert chunk.frames_planned <= MAX_FRAMES_PER_CHUNK

    def test_chunk_count_never_exceeds_request_cap(self):
        for duration_s in (8, 60, 300, 1800, 7200, 36000):
            plan = plan_analysis(_request(duration_s))
            assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST

    def test_total_planned_frames_equals_sum_of_chunks(self):
        for duration_s in (8, 60, 300, 1800, 7200):
            plan = plan_analysis(_request(duration_s))
            assert plan.total_planned_frames == sum(
                c.frames_planned for c in plan.chunks
            )

    def test_plan_model_rejects_mismatched_total(self):
        chunk = AnalysisChunk(
            index=0, start_us=0, end_us=1_000_000, frames_planned=1, sampling_fps=1.0
        )
        with pytest.raises(ValueError, match="does not match"):
            VideoAnalysisPlan(
                strategy=AnalysisStrategy.FULL_SCAN,
                chunks=[chunk],
                total_planned_frames=99,
            )

    def test_analysis_chunk_rejects_frames_over_cap(self):
        with pytest.raises(ValueError):
            AnalysisChunk(
                index=0,
                start_us=0,
                end_us=1_000_000,
                frames_planned=MAX_FRAMES_PER_CHUNK + 1,
                sampling_fps=1.0,
            )

    def test_analysis_chunk_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            AnalysisChunk(
                index=0, start_us=1_000_000, end_us=0, frames_planned=1, sampling_fps=1.0
            )


class TestRanges:
    def test_nonzero_start(self):
        plan = plan_analysis(_request(10, start_s=500))
        assert plan.chunks[0].start_us == 500_000_000
        assert plan.chunks[-1].end_us == 510_000_000

    def test_subrange_boundaries_exact(self):
        req = _request(90, mode="summarize_video", start_s=100)
        plan = plan_analysis(req)
        assert plan.chunks[0].start_us == req.start_us
        assert plan.chunks[-1].end_us == req.end_us

    def test_very_short_range(self):
        plan = plan_analysis(_request(0.5))
        assert len(plan.chunks) == 1
        assert plan.chunks[0].frames_planned >= 1

    def test_large_range_within_budget(self):
        plan = plan_analysis(_request(36000))  # 10h, exact boundary - must succeed
        assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST


class TestScale:
    @pytest.mark.parametrize(
        "duration_s",
        [8, 60, 300, 1800, 7200],
        ids=["8sec", "1min", "5min", "30min", "2hr"],
    )
    def test_scale_produces_valid_plan(self, duration_s):
        plan = plan_analysis(_request(duration_s))
        assert plan.chunks[0].start_us == 0
        assert plan.chunks[-1].end_us == int(duration_s * 1_000_000)
        assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST
        assert plan.total_planned_frames > 0

    def test_30min_matches_todays_default_chunking_shape(self):
        """30 min sits exactly at MAX_CHUNKS_PER_REQUEST with TARGET_CHUNK_S=30,
        so this must reproduce today's implicit 30s/1fps chunking, unchanged."""
        plan = plan_analysis(_request(1800))
        assert len(plan.chunks) == MAX_CHUNKS_PER_REQUEST
        assert all(c.frames_planned == 30 for c in plan.chunks)

    def test_2hr_degrades_density_instead_of_exploding_chunk_count(self):
        plan = plan_analysis(_request(7200))
        assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST
        assert all(c.sampling_fps < 1.0 for c in plan.chunks[:-1])

    def test_long_ask_video_scale(self):
        plan = plan_analysis(_request(3600, mode="ask_video"))  # 1h
        assert plan.strategy == AnalysisStrategy.TARGETED_SCAN
        assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST
        assert plan.total_planned_frames > 0


class TestPlanInfeasible:
    def test_range_past_ten_hours_is_infeasible(self):
        """Derived infeasibility, not an arbitrary cutoff: beyond
        MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST (10h at current
        defaults) no chunk plan fits the call-count budget even at the
        existing per-chunk duration ceiling."""
        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        with pytest.raises(VideoIntelligenceError) as exc_info:
            plan_analysis(_request(boundary_s + 3600))
        assert exc_info.value.code == VideoIntelligenceErrorCode.PLAN_INFEASIBLE

    def test_exact_ten_hour_boundary_still_succeeds(self):
        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        plan = plan_analysis(_request(boundary_s))
        assert len(plan.chunks) == MAX_CHUNKS_PER_REQUEST


class TestProviderCapabilitiesOverride:
    def test_lower_provider_cap_reduces_frames_per_chunk(self):
        plan = plan_analysis(
            _request(8), provider_capabilities={"max_frames_per_call": 3}
        )
        assert all(c.frames_planned <= 3 for c in plan.chunks)

    def test_missing_capabilities_uses_default_cap(self):
        plan = plan_analysis(_request(8), provider_capabilities=None)
        assert plan.chunks[0].frames_planned <= MAX_FRAMES_PER_CHUNK


class TestDeterminism:
    def test_same_request_same_plan(self):
        req = _request(300)
        plan_a = plan_analysis(req)
        plan_b = plan_analysis(req)
        assert plan_a.model_dump() == plan_b.model_dump()

    def test_equivalent_requests_same_plan(self):
        plan_a = plan_analysis(_request(300))
        plan_b = plan_analysis(_request(300))
        assert plan_a == plan_b


class TestNoEditorialLeakage:
    def test_editorial_plan_schema_has_no_planner_vocabulary(self):
        from app.editor.agent import EditorialPlan

        schema = json.dumps(EditorialPlan.model_json_schema())
        for term in (
            "frames_planned",
            "sampling_fps",
            "AnalysisStrategy",
            "AnalysisChunk",
            "VideoAnalysisPlan",
            "total_planned_frames",
        ):
            assert term not in schema

    def test_evidence_request_schema_has_no_planner_vocabulary(self):
        from app.editor.evidence_request import EvidenceRequest

        schema = json.dumps(EvidenceRequest.model_json_schema())
        for term in (
            "frames_planned",
            "sampling_fps",
            "AnalysisStrategy",
            "AnalysisChunk",
            "VideoAnalysisPlan",
            "total_planned_frames",
        ):
            assert term not in schema
