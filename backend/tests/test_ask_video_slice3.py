"""Phase 4.1 Slice 3 — ask_video wired to the AnalysisPlanner.

FULL_SCAN (<=40s) takes the exact pre-Slice-3 single-call path. TARGETED_SCAN
(>40s) splits into multiple AnalysisChunk, each answering the same question
over its own absolute window, deterministically aggregated (no second LLM
call). No real ffmpeg/Claude calls here - _extract_frames and
_call_vlm_with_frames are monkeypatched. See test_planner_wiring.py for the
service->provider wiring tests, and test_analysis_planner.py for the pure
planner.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.video_intelligence.chunking import (
    MAX_CHUNK_DURATION_S,
    MAX_CHUNKS_PER_REQUEST,
    MAX_FRAMES_PER_CHUNK,
    AnalysisStrategy,
    plan_analysis,
)
from app.video_intelligence.models import (
    CoverageStatus,
    ExtractedFrame,
    VideoAnswer,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceRequest,
)
from app.video_intelligence.providers import ExistingSnipraVLMProvider
from app.video_intelligence.registry import get_registry
from app.video_intelligence.service import VideoIntelligenceService


def _fake_extracted_frames(
    start_us: int, end_us: int, sampling_fps: float = 1.0, count: int | None = None
) -> list[ExtractedFrame]:
    """VIA v2 contract migration.

    old expectation: `_extract_frames` fakes returned raw ``[b"frame"] * n``.
    new VIA v2 expectation: `_extract_frames` returns ``list[ExtractedFrame]``
      (frame_id + timestamp_us + data) - the parsers now build a
      ``frame_id_to_ts`` map from it for evidence provenance, so a bytes list
      raises ``'bytes' object has no attribute 'frame_id'`` and fails the chunk.
    reason: Block 02 (grounded VIA provenance contract) made frame identity /
      timestamps first-class. This helper mirrors
      ``ExistingSnipraVLMProvider._extract_frames_blocking``'s deterministic
      ``frame_{idx:04d}`` id and ``start_us + idx/sampling_fps`` timestamp
      scheme so fake extraction is contract-accurate, not just non-crashing.
    """
    if count is None:
        duration_s = (end_us - start_us) / 1_000_000
        count = max(1, round(duration_s * sampling_fps))
    frames: list[ExtractedFrame] = []
    for idx in range(count):
        ts = (
            start_us + int((idx / sampling_fps) * 1_000_000)
            if sampling_fps > 0
            else start_us
        )
        frames.append(
            ExtractedFrame(frame_id=f"frame_{idx:04d}", timestamp_us=ts, data=b"frame")
        )
    return frames


def _ask_request(duration_s: float, start_s: float = 0.0, question: str = "what happens?"):
    return VideoIntelligenceRequest(
        asset_id="ask-slice3-asset",
        start_us=int(start_s * 1_000_000),
        end_us=int((start_s + duration_s) * 1_000_000),
        mode="ask_video",
        question=question,
    )


def _grounded_provider(monkeypatch) -> ExistingSnipraVLMProvider:
    from app.core.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "claude")
    return ExistingSnipraVLMProvider()


def _install_fakes(monkeypatch, provider, responses=None, prompts_seen=None):
    """responses: optional JSON string returned for every chunk call.
    prompts_seen: optional list that collects each prompt actually sent, in
    call order."""
    extract_calls: list[tuple[int, int, float]] = []

    async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
        extract_calls.append((start_us, end_us, sampling_fps))
        return _fake_extracted_frames(start_us, end_us, sampling_fps)

    def _fake_vlm(system, prompt, frames):
        if prompts_seen is not None:
            prompts_seen.append(prompt)
        return responses or json.dumps(
            {"answer": "ok", "confidence": 0.9, "observations": [], "events": []}
        )

    monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
    monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)
    return extract_calls


class TestFullScanUnchanged:
    """Section 3: FULL_SCAN must stay practically equivalent to the original
    single-call ask_video path, including the 8s smoke shape."""

    def test_8sec_is_full_scan_single_call(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        calls = _install_fakes(
            monkeypatch,
            provider,
            responses=json.dumps(
                {
                    "answer": "the gradient shifts at 3 seconds",
                    "confidence": 0.85,
                    "observations": [
                        {"start": "3.0s", "end": "3.0s", "description": "shift", "confidence": 0.9}
                    ],
                    "events": [],
                }
            ),
        )

        request = _ask_request(8)
        plan = plan_analysis(request)
        assert plan.strategy == AnalysisStrategy.FULL_SCAN
        assert len(plan.chunks) == 1

        result = asyncio.run(provider.ask_video(request, plan))

        assert len(calls) == 1
        assert result.answer.answer == "the gradient shifts at 3 seconds"
        assert result.metadata.chunks_processed == 1
        assert result.metadata.chunks_failed == 0
        assert result.partial is False
        assert result.metadata.requested_frames == plan.total_planned_frames

    def test_full_scan_raises_on_zero_frames_not_partial(self, monkeypatch):
        """FULL_SCAN keeps the old all-or-nothing failure contract - it must
        NOT silently degrade into a partial/empty-answer result."""
        provider = _grounded_provider(monkeypatch)

        async def _empty_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return []

        monkeypatch.setattr(provider, "_extract_frames", _empty_extract)

        request = _ask_request(5)
        with pytest.raises(VideoIntelligenceError) as exc:
            asyncio.run(provider.ask_video(request))
        assert exc.value.code == VideoIntelligenceErrorCode.FRAME_EXTRACTION_FAILED


class TestTargetedScanMultiChunk:
    def test_60sec_is_targeted_scan_valid_plan(self, monkeypatch):
        request = _ask_request(60)
        plan = plan_analysis(request)
        assert plan.strategy == AnalysisStrategy.TARGETED_SCAN
        for chunk in plan.chunks:
            assert chunk.frames_planned <= MAX_FRAMES_PER_CHUNK
        assert len(plan.chunks) <= MAX_CHUNKS_PER_REQUEST

    def test_5min_multi_chunk_exactly_n_calls(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        prompts = []
        calls = _install_fakes(monkeypatch, provider, prompts_seen=prompts)

        request = _ask_request(300)
        plan = plan_analysis(request)
        assert len(plan.chunks) > 1

        result = asyncio.run(provider.ask_video(request, plan))

        assert len(calls) == len(plan.chunks)
        assert len(prompts) == len(plan.chunks)
        assert result.metadata.chunks_processed == len(plan.chunks)
        assert result.metadata.chunks_failed == 0

    def test_no_single_call_exceeds_frame_budget(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        calls = _install_fakes(monkeypatch, provider)

        request = _ask_request(1200)  # 20 min
        plan = plan_analysis(request)
        asyncio.run(provider.ask_video(request, plan))

        for (start_us, end_us, fps), chunk in zip(calls, plan.chunks, strict=True):
            duration_s = (end_us - start_us) / 1_000_000
            frames = max(1, round(duration_s * fps))
            assert frames <= MAX_FRAMES_PER_CHUNK
            assert frames <= chunk.frames_planned

    def test_same_question_sent_to_every_chunk(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        prompts = []
        _install_fakes(monkeypatch, provider, prompts_seen=prompts)

        request = _ask_request(90, question="does the player win?")
        plan = plan_analysis(request)
        asyncio.run(provider.ask_video(request, plan))

        assert len(prompts) == len(plan.chunks) > 1
        for prompt in prompts:
            assert "does the player win?" in prompt


class TestAggregation:
    def test_single_chunk_answer_used_verbatim(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        _install_fakes(
            monkeypatch,
            provider,
            responses=json.dumps(
                {"answer": "yes, clearly", "confidence": 0.77, "observations": [], "events": []}
            ),
        )
        request = _ask_request(5)
        result = asyncio.run(provider.ask_video(request))
        assert result.answer.answer == "yes, clearly"
        assert result.answer.confidence == 0.77

    def test_multi_chunk_answer_is_labeled_join_not_silent_overwrite(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        seq = [
            json.dumps({"answer": "player is reloading", "confidence": 0.6,
                        "observations": [], "events": []}),
            json.dumps({"answer": "player wins the round", "confidence": 0.9,
                        "observations": [], "events": []}),
        ]
        prompts = []

        def _fake_vlm(system, prompt, frames):
            prompts.append(prompt)
            return seq[min(len(prompts) - 1, len(seq) - 1)]

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)

        request = _ask_request(60)  # -> 2 chunks (0-30s, 30-60s)
        plan = plan_analysis(request)
        assert len(plan.chunks) == 2

        result = asyncio.run(provider.ask_video(request, plan))

        assert "player is reloading" in result.answer.answer
        assert "player wins the round" in result.answer.answer
        assert "0.0s" in result.answer.answer and "30.0s" in result.answer.answer
        assert result.answer.confidence == pytest.approx((0.6 + 0.9) / 2)

    def test_supporting_observation_ids_unioned_without_duplicates(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        seq = [
            json.dumps({"answer": "a", "confidence": 0.5,
                        "supporting_observation_ids": ["obs-1", "obs-2"],
                        "observations": [], "events": []}),
            json.dumps({"answer": "b", "confidence": 0.5,
                        "supporting_observation_ids": ["obs-2", "obs-3"],
                        "observations": [], "events": []}),
        ]
        prompts = []

        def _fake_vlm(system, prompt, frames):
            prompts.append(prompt)
            return seq[min(len(prompts) - 1, len(seq) - 1)]

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)

        request = _ask_request(60)
        result = asyncio.run(provider.ask_video(request))

        assert result.answer.supporting_observation_ids == ["obs-1", "obs-2", "obs-3"]

    def test_citations_concatenated_and_stay_list_of_dict(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        seq = [
            json.dumps({"answer": "a", "confidence": 0.5,
                        "citations": [{"source": "frame_1", "timestamp_us": 1_000_000}],
                        "observations": [], "events": []}),
            json.dumps({"answer": "b", "confidence": 0.5,
                        "citations": [{"source": "frame_2", "timestamp_us": 31_000_000}],
                        "observations": [], "events": []}),
        ]
        prompts = []

        def _fake_vlm(system, prompt, frames):
            prompts.append(prompt)
            return seq[min(len(prompts) - 1, len(seq) - 1)]

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)

        request = _ask_request(60)
        result = asyncio.run(provider.ask_video(request))

        assert isinstance(result.answer.citations, list)
        assert all(isinstance(c, dict) for c in result.answer.citations)
        assert len(result.answer.citations) == 2

    def test_aggregated_answer_roundtrips_through_json(self, monkeypatch):
        """Section 7: construct -> model_dump(mode='json') -> model_validate()
        must preserve citations as list[dict] (the Phase 3.5B.2 regression)."""
        answer = VideoAnswer(
            answer="[0.0s-30.0s] a\n[30.0s-60.0s] b",
            confidence=0.7,
            supporting_observation_ids=["obs-1", "obs-2"],
            citations=[
                {"source": "frame_1", "timestamp_us": 1_000_000},
                {"source": "frame_2", "timestamp_us": 31_000_000},
            ],
        )
        payload = answer.model_dump(mode="json")
        validated = VideoAnswer.model_validate(payload)
        assert validated.citations == answer.citations
        assert isinstance(validated.citations, list)
        assert all(isinstance(c, dict) for c in validated.citations)


class TestTimestampGrounding:
    def test_nonzero_start_multi_chunk_observations_use_absolute_time(self, monkeypatch):
        """Section 8: chunk 2's observation at 'start: 5.0s' in its own prompt
        must be grounded relative to ITS OWN absolute window, not
        request.start_us - reusing _offset_us exactly like summarize_video."""
        provider = _grounded_provider(monkeypatch)
        seq = [
            json.dumps({"answer": "a", "confidence": 0.5, "observations": [
                {"start": "5.0s", "end": "5.0s", "description": "first chunk event",
                 "confidence": 0.8}
            ], "events": []}),
            json.dumps({"answer": "b", "confidence": 0.5, "observations": [
                {"start": "5.0s", "end": "5.0s", "description": "second chunk event",
                 "confidence": 0.8}
            ], "events": []}),
        ]
        prompts = []

        def _fake_vlm(system, prompt, frames):
            prompts.append(prompt)
            return seq[min(len(prompts) - 1, len(seq) - 1)]

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)

        request = _ask_request(60, start_s=1000)  # request starts at 1000s absolute
        plan = plan_analysis(request)
        assert len(plan.chunks) == 2

        result = asyncio.run(provider.ask_video(request, plan))

        obs_by_desc = {o.description: o for o in result.observations}
        # chunk 0 covers [1000s, 1030s) absolute -> "5.0s" offset -> 1005s
        assert obs_by_desc["first chunk event"].start_us == 1_005_000_000
        # chunk 1 covers [1030s, 1060s) absolute -> "5.0s" offset -> 1035s
        assert obs_by_desc["second chunk event"].start_us == 1_035_000_000


class TestFailureSemantics:
    def test_partial_success_preserves_successful_chunk_answer(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        outcomes = [
            json.dumps({"answer": "real answer", "confidence": 0.8,
                        "observations": [], "events": []}),
            RuntimeError("VLM upstream error"),
        ]
        calls = {"n": 0}

        def _fake_vlm(system, prompt, frames):
            outcome = outcomes[min(calls["n"], len(outcomes) - 1)]
            calls["n"] += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)

        request = _ask_request(60)  # 2 chunks
        result = asyncio.run(provider.ask_video(request))

        assert result.partial is True
        assert len(result.metadata.failed_chunks) == 1
        assert result.answer is not None
        assert result.answer.answer == "real answer"
        assert result.metadata.coverage_status == CoverageStatus.PARTIAL

    def test_all_chunks_fail_returns_no_fabricated_answer(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)

        def _failing_vlm(system, prompt, frames):
            raise RuntimeError("boom")

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(provider, "_call_vlm_with_frames", _failing_vlm)

        request = _ask_request(60)  # 2 chunks
        result = asyncio.run(provider.ask_video(request))

        assert result.answer is None
        assert result.partial is True
        assert len(result.metadata.failed_chunks) == 2
        assert result.metadata.analyzed_frames == 0
        assert result.metadata.coverage_status == CoverageStatus.INSUFFICIENT
        assert result.metadata.coverage_ratio == 0.0

    def test_all_success_no_partial(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        _install_fakes(monkeypatch, provider)

        request = _ask_request(90)
        result = asyncio.run(provider.ask_video(request))

        assert result.partial is False
        assert result.metadata.failed_chunks == []
        assert result.metadata.coverage_status == CoverageStatus.COMPLETE


class TestRequestedAnalyzedFrames:
    def test_requested_frames_from_plan_not_duration_times_fps(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        _install_fakes(monkeypatch, provider)

        for duration_s in (8, 300, 3600):
            request = _ask_request(duration_s)
            plan = plan_analysis(request)
            result = asyncio.run(provider.ask_video(request, plan))
            assert result.metadata.requested_frames == plan.total_planned_frames

        # 1h: naive_chunks=120 exceeds MAX_CHUNKS_PER_REQUEST, so density
        # degrades below 1fps - this is where requested_frames must diverge
        # from the old naive duration*1fps estimate to prove it's plan-sourced.
        long_request = _ask_request(3600)
        long_plan = plan_analysis(long_request)
        naive_duration_times_fps = int(3600 * 1.0)
        assert long_plan.total_planned_frames != naive_duration_times_fps


class TestScale:
    @pytest.mark.parametrize(
        "duration_s",
        [8, 60, 300, 1200, 3600],
        ids=["8sec", "1min", "5min", "20min", "1hr"],
    )
    def test_scale_respects_budget(self, monkeypatch, duration_s):
        provider = _grounded_provider(monkeypatch)
        calls = _install_fakes(monkeypatch, provider)

        request = _ask_request(duration_s)
        plan = plan_analysis(request)
        result = asyncio.run(provider.ask_video(request, plan))

        assert len(calls) <= MAX_CHUNKS_PER_REQUEST
        assert all(frames_planned <= MAX_FRAMES_PER_CHUNK for frames_planned in
                   [c.frames_planned for c in plan.chunks])
        assert result.metadata.chunks_failed == 0


class TestPlanInfeasibleAskVideo:
    def test_infeasible_ask_video_range_never_reaches_provider_or_ffmpeg(self, monkeypatch):
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")

        provider_calls = {"n": 0}
        extract_calls = {"n": 0}

        async def _counting_ask(self, request, analysis_plan=None):
            provider_calls["n"] += 1
            raise AssertionError("provider.ask_video must not be called")

        async def _counting_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            extract_calls["n"] += 1
            raise AssertionError("_extract_frames must not be called")

        monkeypatch.setattr(ExistingSnipraVLMProvider, "ask_video", _counting_ask)
        monkeypatch.setattr(provider, "_extract_frames", _counting_extract)

        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        request = _ask_request(boundary_s + 3600)  # past the 10h ceiling

        with pytest.raises(VideoIntelligenceError) as exc_info:
            asyncio.run(service.ask_video(request))

        assert exc_info.value.code == VideoIntelligenceErrorCode.PLAN_INFEASIBLE
        assert provider_calls["n"] == 0
        assert extract_calls["n"] == 0

    def test_no_arbitrary_ten_minute_cutoff(self, monkeypatch):
        """Sign-off #1 (Slice 1) still holds for ask_video: a 20-minute
        targeted range must produce a valid multi-chunk plan, not fail."""
        provider = _grounded_provider(monkeypatch)
        _install_fakes(monkeypatch, provider)

        request = _ask_request(1200)  # 20 min
        result = asyncio.run(provider.ask_video(request))
        assert result.partial is False
