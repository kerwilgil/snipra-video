"""Phase 4.2a - pure scene-aware plan refinement.

refine_plan_with_scenes() is pure and deterministic: given an already-valid
VideoAnalysisPlan (Phase 4.1) and a list of SceneBoundary, it only nudges
INTERNAL chunk boundaries toward real cuts within a small tolerance. No
scene detector exists yet (Phase 4.2b) - boundaries are hand-supplied here.
"""

from __future__ import annotations

import json
import math

import pytest

from app.video_intelligence.chunking import (
    MAX_FRAMES_PER_CHUNK,
    AnalysisChunk,
    AnalysisStrategy,
    SceneBoundary,
    VideoAnalysisPlan,
    plan_analysis,
    refine_plan_with_scenes,
)
from app.video_intelligence.models import VideoIntelligenceRequest


def _base_plan(duration_s: float, start_s: float = 0.0, mode: str = "summarize_video"):
    kwargs = {"objective": "test"} if mode == "summarize_video" else {"question": "test?"}
    request = VideoIntelligenceRequest(
        asset_id="scene-refine-asset",
        start_us=int(start_s * 1_000_000),
        end_us=int((start_s + duration_s) * 1_000_000),
        mode=mode,
        **kwargs,
    )
    return plan_analysis(request)


def _boundary(seconds: float) -> SceneBoundary:
    return SceneBoundary(timestamp_us=int(seconds * 1_000_000))


class TestNoOp:
    def test_a_empty_boundaries_is_noop(self):
        plan = _base_plan(90)
        result = refine_plan_with_scenes(plan, [])
        assert result is plan

    def test_q_single_chunk_is_noop(self):
        plan = _base_plan(20)  # FULL_SCAN, 1 chunk
        assert plan.strategy == AnalysisStrategy.FULL_SCAN
        assert len(plan.chunks) == 1
        result = refine_plan_with_scenes(plan, [_boundary(10)])
        assert result is plan


class TestSnapping:
    def test_b_boundary_exact_at_technical_cut_same_plan(self):
        plan = _base_plan(90)  # 3 chunks: [0,30) [30,60) [60,90)
        boundaries = [_boundary(30), _boundary(60)]
        refined = refine_plan_with_scenes(plan, boundaries)

        assert len(refined.chunks) == len(plan.chunks)
        for base, ref in zip(plan.chunks, refined.chunks, strict=True):
            assert ref.start_us == base.start_us
            assert ref.end_us == base.end_us
            assert ref.frames_planned == base.frames_planned
            assert ref.sampling_fps == pytest.approx(base.sampling_fps)

    def test_c_nearby_boundary_snaps(self):
        plan = _base_plan(90)
        refined = refine_plan_with_scenes(plan, [_boundary(31.5)])  # 1.5s from 30s cut

        assert refined.chunks[0].end_us == 31_500_000
        assert refined.chunks[1].start_us == 31_500_000
        assert refined.chunks[1].end_us == 60_000_000  # untouched, no nearby boundary
        assert refined.chunks[2].start_us == 60_000_000

    def test_d_out_of_tolerance_boundary_does_not_snap(self):
        plan = _base_plan(90)
        refined = refine_plan_with_scenes(plan, [_boundary(35)])  # 5s away, tol=2.5s
        assert refined.chunks[0].end_us == 30_000_000
        assert refined.chunks[1].start_us == 30_000_000

    def test_e_tie_break_prefers_earlier_timestamp(self):
        plan = _base_plan(90)
        # both 1.5s from the 30s cut
        refined = refine_plan_with_scenes(plan, [_boundary(28.5), _boundary(31.5)])
        assert refined.chunks[0].end_us == 28_500_000
        assert refined.chunks[1].start_us == 28_500_000

    def test_f_nonzero_start_us(self):
        plan = _base_plan(90, start_s=500)
        refined = refine_plan_with_scenes(plan, [_boundary(531.5)])  # 1.5s from 530s cut
        assert refined.chunks[0].start_us == 500_000_000
        assert refined.chunks[-1].end_us == 590_000_000
        assert refined.chunks[0].end_us == 531_500_000

    def test_g_multiple_chunks_multiple_snaps(self):
        plan = _base_plan(300)  # 10 chunks of 30s: cuts at 30,60,...,270
        boundaries = [_boundary(s + 1) for s in range(30, 270 + 1, 30)]  # +1s each
        refined = refine_plan_with_scenes(plan, boundaries)

        assert len(refined.chunks) == len(plan.chunks) == 10
        for i, base_cut in enumerate(range(30, 270 + 1, 30)):
            assert refined.chunks[i].end_us == int((base_cut + 1) * 1_000_000)


class TestContiguity:
    def test_h_no_gaps(self):
        plan = _base_plan(300)
        boundaries = [_boundary(s + 1) for s in range(30, 270 + 1, 30)]
        refined = refine_plan_with_scenes(plan, boundaries)
        for i in range(len(refined.chunks) - 1):
            assert refined.chunks[i].end_us == refined.chunks[i + 1].start_us

    def test_i_no_overlaps(self):
        plan = _base_plan(300)
        boundaries = [_boundary(s - 1) for s in range(30, 270 + 1, 30)]
        refined = refine_plan_with_scenes(plan, boundaries)
        for i in range(len(refined.chunks) - 1):
            assert refined.chunks[i].end_us <= refined.chunks[i + 1].start_us

    def test_j_chunk_never_inverted(self):
        plan = _base_plan(300)
        boundaries = [_boundary(s + 2) for s in range(30, 270 + 1, 30)]
        refined = refine_plan_with_scenes(plan, boundaries)
        for chunk in refined.chunks:
            assert chunk.start_us < chunk.end_us

    def test_request_range_preserved(self):
        plan = _base_plan(300, start_s=17)
        boundaries = [_boundary(17 + s + 1) for s in range(30, 270 + 1, 30)]
        refined = refine_plan_with_scenes(plan, boundaries)
        assert refined.chunks[0].start_us == plan.chunks[0].start_us
        assert refined.chunks[-1].end_us == plan.chunks[-1].end_us
        assert refined.strategy == plan.strategy


class TestFrameBudgetRecalculation:
    def test_k_extended_chunk_recalculates_honestly(self):
        plan = _base_plan(90)  # chunk0 base: 30s, 30 frames, fps=1.0
        refined = refine_plan_with_scenes(plan, [_boundary(31.5)])
        chunk0 = refined.chunks[0]
        assert chunk0.end_us - chunk0.start_us == 31_500_000
        expected_frames = math.floor(31.5 * 1.0 + 1e-9)
        assert chunk0.frames_planned == expected_frames
        assert chunk0.sampling_fps == pytest.approx(expected_frames / 31.5)
        assert chunk0.sampling_fps <= plan.chunks[0].sampling_fps

    def test_l_shortened_chunk_metadata_coherent(self):
        plan = _base_plan(90)  # chunk1 base: 30s, 30 frames, fps=1.0
        refined = refine_plan_with_scenes(plan, [_boundary(31.5)])
        chunk1 = refined.chunks[1]
        assert chunk1.end_us - chunk1.start_us == 28_500_000
        expected_frames = math.floor(28.5 * 1.0 + 1e-9)
        assert chunk1.frames_planned == expected_frames
        assert chunk1.sampling_fps == pytest.approx(expected_frames / 28.5)
        assert chunk1.sampling_fps <= plan.chunks[1].sampling_fps

    @pytest.mark.parametrize("duration_s", [90, 300, 1800, 7200])
    def test_m_no_chunk_exceeds_max_frames(self, duration_s):
        plan = _base_plan(duration_s)
        boundaries = [
            _boundary(c.end_us / 1_000_000 + 1.0) for c in plan.chunks[:-1]
        ]
        refined = refine_plan_with_scenes(plan, boundaries)
        for chunk in refined.chunks:
            assert chunk.frames_planned <= MAX_FRAMES_PER_CHUNK

    @pytest.mark.parametrize("duration_s", [90, 300, 1800, 7200])
    def test_n_sampling_fps_never_increases(self, duration_s):
        plan = _base_plan(duration_s)
        boundaries = [
            _boundary(c.end_us / 1_000_000 - 1.0) for c in plan.chunks[:-1]
        ]
        refined = refine_plan_with_scenes(plan, boundaries)
        for base, ref in zip(plan.chunks, refined.chunks, strict=True):
            assert ref.sampling_fps <= base.sampling_fps + 1e-9

    def test_o_total_planned_frames_recomputed(self):
        plan = _base_plan(300)
        # Fractional-second shift so floor() actually drops frames overall
        # (an integer-second shift between two 1.0fps chunks conserves the
        # sum exactly and wouldn't demonstrate a real recomputation).
        boundaries = [_boundary(270 + 1.7)]
        refined = refine_plan_with_scenes(plan, boundaries)
        assert refined.total_planned_frames == sum(c.frames_planned for c in refined.chunks)
        assert refined.total_planned_frames != plan.total_planned_frames


class TestDeterminism:
    def test_p_same_inputs_same_output(self):
        plan = _base_plan(300)
        boundaries = [_boundary(s + 1) for s in range(30, 270 + 1, 30)]
        result_a = refine_plan_with_scenes(plan, boundaries)
        result_b = refine_plan_with_scenes(plan, boundaries)
        assert result_a.model_dump() == result_b.model_dump()


class TestManyBoundaries:
    def test_r_hundreds_of_boundaries_chunk_count_unchanged(self):
        plan = _base_plan(1800)  # 30min -> 60 chunks
        boundaries = [_boundary(t) for t in range(0, 1800, 6)]  # 300 boundaries
        refined = refine_plan_with_scenes(plan, boundaries)
        assert len(refined.chunks) == len(plan.chunks) == 60


class TestCriticalBudgetRegression:
    """Section 11: a chunk already at the frame cap must never end up
    declaring duration*sampling_fps > MAX_FRAMES_PER_CHUNK after extension.
    This is the exact bug the naive "move start/end, freeze frames_planned
    and sampling_fps" design would have produced."""

    def test_chunk_at_cap_extended_stays_capped(self):
        base_cap_fps = MAX_FRAMES_PER_CHUNK / 120.0  # 40 frames over 120s
        chunk0 = AnalysisChunk(
            index=0, start_us=0, end_us=120_000_000,
            frames_planned=MAX_FRAMES_PER_CHUNK, sampling_fps=base_cap_fps,
        )
        chunk1 = AnalysisChunk(
            index=1, start_us=120_000_000, end_us=240_000_000,
            frames_planned=MAX_FRAMES_PER_CHUNK, sampling_fps=base_cap_fps,
        )
        plan = VideoAnalysisPlan(
            strategy=AnalysisStrategy.COARSE_SCAN,
            chunks=[chunk0, chunk1],
            total_planned_frames=2 * MAX_FRAMES_PER_CHUNK,
        )
        assert plan.chunks[0].frames_planned == MAX_FRAMES_PER_CHUNK

        # Snap extends chunk0 by 2.5s (within tolerance of the 120s cut).
        refined = refine_plan_with_scenes(plan, [_boundary(122.5)])

        extended = refined.chunks[0]
        assert extended.end_us - extended.start_us == 122_500_000
        duration_s = (extended.end_us - extended.start_us) / 1_000_000
        assert extended.frames_planned <= MAX_FRAMES_PER_CHUNK
        assert duration_s * extended.sampling_fps <= MAX_FRAMES_PER_CHUNK + 1e-9
        assert extended.sampling_fps <= base_cap_fps + 1e-9

        shrunk = refined.chunks[1]
        assert shrunk.end_us - shrunk.start_us == 117_500_000
        assert shrunk.frames_planned <= MAX_FRAMES_PER_CHUNK
        assert shrunk.sampling_fps <= base_cap_fps + 1e-9

    def test_snap_rejected_when_it_would_break_min_duration_for_cap(self):
        """A chunk whose base density is already very low (needs >=20s for
        even 1 frame at low_fps=0.05) cannot be shrunk below that floor by a
        tolerance-sized snap - the snap must be rejected outright rather
        than letting frames_planned's >=1 floor silently exceed the base
        cap."""
        low_fps = 0.05  # needs >=20s for even 1 frame
        chunk0 = AnalysisChunk(
            index=0, start_us=0, end_us=60_000_000, frames_planned=3, sampling_fps=low_fps,
        )
        # 22s: comfortably above the 20s floor on its own, but a 2.5s
        # rightward snap of the shared boundary shrinks it to 19.5s < 20s.
        chunk1 = AnalysisChunk(
            index=1, start_us=60_000_000, end_us=82_000_000, frames_planned=1,
            sampling_fps=low_fps,
        )
        plan = VideoAnalysisPlan(
            strategy=AnalysisStrategy.COARSE_SCAN, chunks=[chunk0, chunk1],
            total_planned_frames=4,
        )
        refined = refine_plan_with_scenes(plan, [_boundary(62.5)])

        # Rejected: original boundary preserved, both chunks unchanged.
        assert refined.chunks[0].end_us == 60_000_000
        assert refined.chunks[1].start_us == 60_000_000
        for base, ref in zip(plan.chunks, refined.chunks, strict=True):
            assert ref.sampling_fps <= base.sampling_fps + 1e-9


class TestNoEditorialLeakage:
    def test_editorial_plan_schema_has_no_scene_vocabulary(self):
        from app.editor.agent import EditorialPlan

        schema = json.dumps(EditorialPlan.model_json_schema())
        for term in ("SceneBoundary", "timestamp_us", "refine_plan_with_scenes"):
            assert term not in schema

    def test_evidence_request_schema_has_no_scene_vocabulary(self):
        from app.editor.evidence_request import EvidenceRequest

        schema = json.dumps(EvidenceRequest.model_json_schema())
        for term in ("SceneBoundary", "timestamp_us", "refine_plan_with_scenes"):
            assert term not in schema
