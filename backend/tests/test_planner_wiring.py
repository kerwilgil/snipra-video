"""Phase 4.1 Slice 2 — wiring tests.

Verifies VideoIntelligenceService and ExistingSnipraVLMProvider actually use
the AnalysisPlanner (chunking.plan_analysis) for summarize_video, and that
ask_video remains completely untouched by it. No real ffmpeg or Claude API
calls: _extract_frames and _call_vlm_with_frames are monkeypatched.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.video_intelligence.chunking import (
    MAX_CHUNK_DURATION_S,
    MAX_CHUNKS_PER_REQUEST,
    plan_analysis,
)
from app.video_intelligence.models import (
    ExtractedFrame,
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
    new VIA v2 expectation: `_extract_frames` returns ``list[ExtractedFrame]``;
      Block 02's parsers build a ``frame_id_to_ts`` map from it for evidence
      provenance, so a bytes list raises ``'bytes' object has no attribute
      'frame_id'`` and every chunk fails (``partial=True``).
    reason: mirrors ``ExistingSnipraVLMProvider._extract_frames_blocking``'s
      deterministic ``frame_{idx:04d}`` id / ``start_us + idx/sampling_fps``
      timestamp scheme so fake extraction is contract-accurate.
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


def _summarize_request(duration_s: float, start_s: float = 0.0) -> VideoIntelligenceRequest:
    return VideoIntelligenceRequest(
        asset_id="wiring-asset",
        start_us=int(start_s * 1_000_000),
        end_us=int((start_s + duration_s) * 1_000_000),
        mode="summarize_video",
        objective="Summarize",
    )


def _ask_request(duration_s: float) -> VideoIntelligenceRequest:
    return VideoIntelligenceRequest(
        asset_id="wiring-asset",
        start_us=0,
        end_us=int(duration_s * 1_000_000),
        mode="ask_video",
        question="what happens?",
    )


@pytest.fixture(autouse=True)
def _via_available(monkeypatch):
    """N.3: estos tests ejercitan wiring de VIA, no disponibilidad/costo.
    Tras N.1 la disponibilidad depende del proveedor VISUAL resuelto; sin
    esto, settings.llm_provider='naive' (default seguro nuevo) hace que VIA
    falle fail-closed antes de llegar al código bajo prueba. Configura un
    proveedor visual claro (claude auto) con credencial falsa; las llamadas
    peores todavía quedan mockeadas vía _call_vlm_with_frames."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude")
    monkeypatch.setattr(settings, "llm_model", "claude-haiku-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")


def _grounded_provider(monkeypatch) -> ExistingSnipraVLMProvider:
    from app.core.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "claude")
    return ExistingSnipraVLMProvider()


def _empty_evidence_json() -> str:
    return json.dumps({"observations": [], "events": [], "scenes": [], "objects": []})


def _install_always_succeeding_fakes(monkeypatch, provider):
    """Fake _extract_frames that actually respects the requested sampling_fps
    (like real ffmpeg would: frames = round(duration_s * fps)), and a fake VLM
    call that always succeeds with empty evidence. Records every
    (start_us, end_us, sampling_fps) call so budget/density can be asserted."""
    calls: list[tuple[int, int, float]] = []

    async def _fake_extract_frames(asset_id, start_us, end_us, sampling_fps=1.0):
        calls.append((start_us, end_us, sampling_fps))
        return _fake_extracted_frames(start_us, end_us, sampling_fps)

    monkeypatch.setattr(provider, "_extract_frames", _fake_extract_frames)
    monkeypatch.setattr(
        provider, "_call_vlm_with_frames", lambda system, prompt, frames: _empty_evidence_json()
    )
    return calls


class TestServiceObtainsAndPassesPlan:
    """Items 1-2: the service computes the plan and hands it to the provider -
    not a value the provider derives on its own from the request."""

    def test_service_passes_a_plan_to_provider_summarize_video(self, monkeypatch):
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        received = {}

        real_summarize = ExistingSnipraVLMProvider.summarize_video

        async def _spy(self, request, analysis_plan=None):
            received["plan"] = analysis_plan
            received["request"] = request
            return await real_summarize(self, request, analysis_plan)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _spy)

        request = _summarize_request(90)
        result = asyncio.run(service.summarize_video(request))

        expected_plan = plan_analysis(request)
        assert received["plan"] is not None
        assert received["plan"] == expected_plan
        assert result.metadata.requested_frames == expected_plan.total_planned_frames


class TestChunkExecutionMatchesPlan:
    """Items 3-5: the provider iterates exactly the plan's chunks and forwards
    each chunk's own sampling_fps into frame extraction - it does not
    recompute chunk boundaries or density itself."""

    def test_extract_frames_called_once_per_planned_chunk_with_its_fps(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        calls = _install_always_succeeding_fakes(monkeypatch, provider)

        request = _summarize_request(65)  # COARSE_SCAN -> 3 chunks
        plan = plan_analysis(request)
        asyncio.run(provider.summarize_video(request, plan))

        assert len(calls) == len(plan.chunks) == 3
        for (start_us, end_us, fps), chunk in zip(calls, plan.chunks, strict=True):
            assert start_us == chunk.start_us
            assert end_us == chunk.end_us
            assert fps == chunk.sampling_fps

    def test_sampling_fps_is_not_silently_hardcoded_to_one(self, monkeypatch):
        """Regression guard for section 6: if _extract_frames() ever goes back
        to ignoring sampling_fps, a long/degraded-density plan's chunks would
        all report fps=1.0 here and this test would fail."""
        provider = _grounded_provider(monkeypatch)
        calls = _install_always_succeeding_fakes(monkeypatch, provider)

        request = _summarize_request(7200)  # 2h -> degraded density expected
        plan = plan_analysis(request)
        asyncio.run(provider.summarize_video(request, plan))

        fps_values = {round(fps, 6) for _, _, fps in calls}
        assert fps_values != {1.0}
        assert all(fps < 1.0 for fps in fps_values)

    def test_frames_extracted_never_exceeds_frames_planned(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        calls = _install_always_succeeding_fakes(monkeypatch, provider)

        request = _summarize_request(7200)
        plan = plan_analysis(request)
        asyncio.run(provider.summarize_video(request, plan))

        for (start_us, end_us, _fps), chunk in zip(calls, plan.chunks, strict=True):
            duration_s = (end_us - start_us) / 1_000_000
            frames_returned = max(1, round(duration_s * chunk.sampling_fps))
            assert frames_returned <= chunk.frames_planned


class TestRequestedFramesFromPlan:
    """Item 6: requested_frames must come from the plan, not duration*1fps,
    not evidence-object counts, not VLM response counts."""

    def test_requested_frames_equals_plan_total(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        _install_always_succeeding_fakes(monkeypatch, provider)

        for duration_s in (8, 300, 7200):
            request = _summarize_request(duration_s)
            plan = plan_analysis(request)
            result = asyncio.run(provider.summarize_video(request, plan))
            assert result.metadata.requested_frames == plan.total_planned_frames


class TestScale:
    """Items 9-13: prove runtime respects the budget at each scale, and that
    2h genuinely uses <=60 chunks instead of the ~257 the legacy per-chunk
    30s/2s-overlap chunking would have produced. No real ffmpeg/VLM calls."""

    @pytest.mark.parametrize(
        "duration_s",
        [8, 60, 300, 1800, 7200],
        ids=["8sec", "1min", "5min", "30min", "2hr"],
    )
    def test_runtime_respects_budget(self, monkeypatch, duration_s):
        provider = _grounded_provider(monkeypatch)
        calls = _install_always_succeeding_fakes(monkeypatch, provider)

        request = _summarize_request(duration_s)
        plan = plan_analysis(request)
        result = asyncio.run(provider.summarize_video(request, plan))

        assert len(calls) <= MAX_CHUNKS_PER_REQUEST
        assert result.metadata.chunks_failed == 0
        assert result.metadata.partial is False

    def test_2hr_does_not_reproduce_legacy_257_chunk_count(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        calls = _install_always_succeeding_fakes(monkeypatch, provider)

        request = _summarize_request(7200)
        plan = plan_analysis(request)
        asyncio.run(provider.summarize_video(request, plan))

        assert len(calls) <= MAX_CHUNKS_PER_REQUEST
        assert len(calls) < 100  # legacy behavior would have been ~257


class TestPlanInfeasibleFailsFast:
    """Item 14: PLAN_INFEASIBLE must stop the service before any provider
    call or frame extraction - zero side effects."""

    def test_infeasible_range_never_reaches_provider_or_ffmpeg(self, monkeypatch):
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")

        provider_calls = {"n": 0}
        extract_calls = {"n": 0}

        async def _counting_summarize(self, request, analysis_plan=None):
            provider_calls["n"] += 1
            raise AssertionError("provider.summarize_video must not be called")

        async def _counting_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            extract_calls["n"] += 1
            raise AssertionError("_extract_frames must not be called")

        monkeypatch.setattr(ExistingSnipraVLMProvider, "summarize_video", _counting_summarize)
        monkeypatch.setattr(provider, "_extract_frames", _counting_extract)

        boundary_s = MAX_CHUNK_DURATION_S * MAX_CHUNKS_PER_REQUEST
        request = _summarize_request(boundary_s + 3600)  # past the 10h ceiling

        with pytest.raises(VideoIntelligenceError) as exc_info:
            asyncio.run(service.summarize_video(request))

        assert exc_info.value.code == VideoIntelligenceErrorCode.PLAN_INFEASIBLE
        assert provider_calls["n"] == 0
        assert extract_calls["n"] == 0


class TestAskVideoNowPlanned:
    """Phase 4.1 Slice 3: ask_video is wired to the planner - superseding
    Slice 2's guard that it must NOT be touched. See test_ask_video_slice3.py
    for the full TARGETED_SCAN/FULL_SCAN behavior; these two just confirm the
    service->provider wiring itself, mirroring TestServiceObtainsAndPassesPlan
    for summarize_video."""

    def test_ask_video_now_calls_plan_analysis(self, monkeypatch):
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")
        _install_always_succeeding_fakes(monkeypatch, provider)

        request = _ask_request(90)  # > FULL_SCAN_MAX_S -> TARGETED_SCAN, multi-chunk
        expected_plan = plan_analysis(request)

        monkeypatch.setattr(
            provider,
            "_call_vlm_with_frames",
            lambda system, prompt, frames: json.dumps(
                {"answer": "ok", "confidence": 0.9, "observations": [], "events": []}
            ),
        )

        result = asyncio.run(service.ask_video(request))

        assert result.metadata.requested_frames == expected_plan.total_planned_frames
        assert "ok" in result.answer.answer  # labeled/joined across chunks, see providers.py

    def test_ask_video_provider_call_receives_the_plan(self, monkeypatch):
        service = VideoIntelligenceService()
        provider = get_registry().get("snipra_vlm")

        received_args = {}

        real_ask = ExistingSnipraVLMProvider.ask_video

        async def _spy(self, request, *args, **kwargs):
            received_args["args"] = args
            received_args["kwargs"] = kwargs
            return await real_ask(self, request, *args, **kwargs)

        monkeypatch.setattr(ExistingSnipraVLMProvider, "ask_video", _spy)

        async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
            return _fake_extracted_frames(start_us, end_us, sampling_fps, count=1)

        monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
        monkeypatch.setattr(
            provider,
            "_call_vlm_with_frames",
            lambda system, prompt, frames: json.dumps(
                {"answer": "ok", "confidence": 0.9, "observations": [], "events": []}
            ),
        )

        request = _ask_request(10)  # FULL_SCAN, single call
        asyncio.run(service.ask_video(request))

        assert received_args["args"] == (plan_analysis(request),)
        assert received_args["kwargs"] == {}
