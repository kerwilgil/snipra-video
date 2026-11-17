"""Phase 4.3A - bounded parallel chunk execution tests.

Two layers: unit-level tests against app.video_intelligence.chunk_executor.
run_chunks_bounded() directly, using controlled asyncio.sleep()-based fake
workers and asyncio.Event (no real ffmpeg subprocess, no real Claude API
call, no asyncio.to_thread() involved); and a smaller set of provider-level
integration tests (coroutine-object guard, concurrency=1/3 parity, mock
performance smoke) against the real ExistingSnipraVLMProvider with
_extract_frames_blocking/_call_vlm_with_frames monkeypatched, exactly like
test_video_intelligence.py/test_ask_video_slice3.py already do for the
pre-4.3A sequential path.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time

import pytest

from app.core import media
from app.video_intelligence import chunk_executor as chunk_executor_module
from app.video_intelligence.chunk_executor import MAX_CONCURRENT_CHUNKS, run_chunks_bounded
from app.video_intelligence.chunking import AnalysisChunk, plan_analysis
from app.video_intelligence.models import ExtractedFrame, VideoIntelligenceRequest
from app.video_intelligence.providers import ExistingSnipraVLMProvider


def _chunk(index: int) -> AnalysisChunk:
    return AnalysisChunk(
        index=index,
        start_us=index * 1_000_000,
        end_us=(index + 1) * 1_000_000,
        frames_planned=1,
        sampling_fps=1.0,
    )


def _chunks(n: int) -> list[AnalysisChunk]:
    return [_chunk(i) for i in range(n)]


class TestEmptyInput:
    async def test_no_chunks_returns_empty_list_without_touching_worker(self):
        calls = []

        async def worker(chunk):
            calls.append(chunk.index)
            return "x"

        outcomes = await run_chunks_bounded([], worker)
        assert outcomes == []
        assert calls == []


class TestDeterministicOrdering:
    """Section 17/18: real completion order 2, 0, 1 must not change the
    output order, which must always be 0, 1, 2 (chunk.index order)."""

    async def test_output_ordered_by_index_despite_out_of_order_completion(self):
        delays = {0: 0.03, 1: 0.05, 2: 0.01}
        completion_order: list[int] = []

        async def worker(chunk):
            await asyncio.sleep(delays[chunk.index])
            completion_order.append(chunk.index)
            return f"value-{chunk.index}"

        outcomes = await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        assert completion_order == [2, 0, 1]  # sanity: proves real out-of-order completion
        assert [o.index for o in outcomes] == [0, 1, 2]
        assert [o.value for o in outcomes] == ["value-0", "value-1", "value-2"]
        assert all(o.ok for o in outcomes)


class TestRealConcurrency:
    """Section 18: must prove concurrency ACTUALLY happened (max_active > 1),
    not merely that a Semaphore object exists, and that it never exceeds the
    configured bound."""

    @pytest.mark.parametrize("limit", [2, 3])
    async def test_max_active_between_1_and_limit(self, limit):
        active = 0
        max_active = 0

        async def worker(chunk):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return chunk.index

        outcomes = await run_chunks_bounded(_chunks(6), worker, max_concurrency=limit)

        assert max_active > 1
        assert max_active <= limit
        assert [o.value for o in outcomes] == list(range(6))

    async def test_default_max_concurrency_is_respected_when_unspecified(self):
        active = 0
        max_active = 0

        async def worker(chunk):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return chunk.index

        outcomes = await run_chunks_bounded(_chunks(6), worker)  # no max_concurrency passed

        assert max_active > 1
        assert max_active <= MAX_CONCURRENT_CHUNKS
        assert len(outcomes) == 6

    async def test_monkeypatched_module_constant_takes_effect_at_call_time(self, monkeypatch):
        """Guards against the classic 'default evaluated once' bug: the
        constant must be re-read on every call, not baked into the
        function's default parameter at import time."""
        monkeypatch.setattr(chunk_executor_module, "MAX_CONCURRENT_CHUNKS", 1)
        active = 0
        max_active = 0

        async def worker(chunk):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return chunk.index

        await run_chunks_bounded(_chunks(4), worker)  # no explicit max_concurrency
        assert max_active == 1


class TestFailureIsolation:
    """Section 15: one chunk failing must not stop or cancel the others."""

    async def test_one_failure_others_complete(self):
        async def worker(chunk):
            if chunk.index == 1:
                raise ValueError("boom")
            return f"ok-{chunk.index}"

        outcomes = await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        assert [o.ok for o in outcomes] == [True, False, True]
        assert outcomes[0].value == "ok-0"
        assert outcomes[1].error == "boom"
        assert outcomes[2].value == "ok-2"


class TestAllFail:
    async def test_all_chunks_fail_all_outcomes_report_failure(self):
        async def worker(chunk):
            raise RuntimeError(f"fail-{chunk.index}")

        outcomes = await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        assert all(not o.ok for o in outcomes)
        assert [o.error for o in outcomes] == ["fail-0", "fail-1", "fail-2"]


class TestErrorOrdering:
    """Section 20/22: failed_chunks-equivalent output must be ordered by
    chunk.index, never by real failure timing."""

    async def test_failed_chunks_ordered_by_index_not_failure_time(self):
        # High index fails FAST, low index fails SLOW.
        delays = {0: 0.05, 1: 0.03, 2: 0.01}

        async def worker(chunk):
            await asyncio.sleep(delays[chunk.index])
            raise ValueError(f"err-{chunk.index}")

        outcomes = await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        assert [o.index for o in outcomes] == [0, 1, 2]
        assert [o.error for o in outcomes] == ["err-0", "err-1", "err-2"]


class TestParentCancellation:
    """Section 21: cancelling the caller mid-flight must propagate cleanly
    and leave no pending executor task behind."""

    async def test_cancellation_propagates_and_leaves_no_pending_tasks(self):
        started = asyncio.Event()
        release = asyncio.Event()
        seen_tasks: list[asyncio.Task] = []

        async def worker(chunk):
            seen_tasks.append(asyncio.current_task())
            started.set()
            await release.wait()  # would hang forever if not cancelled
            return chunk.index

        async def run():
            await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        outer = asyncio.create_task(run())
        await started.wait()  # at least one worker is now mid-flight
        outer.cancel()

        with pytest.raises(asyncio.CancelledError):
            await outer

        # Give the event loop a couple of ticks to finish unwinding.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert seen_tasks, "expected at least one worker to have started"
        for task in seen_tasks:
            assert task.done(), "no chunk_executor task should remain pending after cancellation"


class TestFFmpegCancellationPropagation:
    """Section 22: a real media.FFmpegCanceledError from one chunk must
    propagate as itself - never appear as an ordinary ChunkOutcome
    failure, and must not orphan still-running siblings."""

    async def test_ffmpeg_canceled_error_propagates_not_a_chunk_failure(self):
        release_others = asyncio.Event()
        other_finished = {"count": 0}

        async def worker(chunk):
            if chunk.index == 1:
                raise media.FFmpegCanceledError("job canceled")
            await release_others.wait()
            other_finished["count"] += 1
            return chunk.index

        async def run():
            return await run_chunks_bounded(_chunks(3), worker, max_concurrency=3)

        task = asyncio.create_task(run())
        # Let chunk 1 raise, then unblock chunks 0/2 so gather can settle
        # (return_exceptions=True waits for everyone before this function
        # re-raises) - if the exception path orphaned them, this would hang.
        await asyncio.sleep(0.01)
        release_others.set()

        with pytest.raises(media.FFmpegCanceledError):
            await task

        assert other_finished["count"] == 2, (
            "siblings must run to completion (not be orphaned) even though "
            "one chunk raised a request-global exception"
        )


class TestCancelCheckWiring:
    """Section 4: media.set_cancel_check() must actually be armed for the
    duration of the call and reset afterward - verified directly, not
    assumed."""

    async def test_cancel_check_is_armed_during_and_reset_after(self):
        observed_during = {"check": None}

        async def worker(chunk):
            observed_during["check"] = media._cancel_check.get()
            return chunk.index

        assert media._cancel_check.get() is None
        await run_chunks_bounded(_chunks(1), worker, max_concurrency=1)

        assert observed_during["check"] is not None
        assert media._cancel_check.get() is None  # reset after the call

    async def test_cancel_event_flips_on_parent_cancellation(self):
        started = asyncio.Event()
        release = asyncio.Event()
        seen_cancel_check: list = []

        async def worker(chunk):
            seen_cancel_check.append(media._cancel_check.get())
            started.set()
            await release.wait()
            return chunk.index

        async def run():
            await run_chunks_bounded(_chunks(1), worker, max_concurrency=1)

        outer = asyncio.create_task(run())
        await started.wait()
        check = seen_cancel_check[0]
        assert check() is False  # not requested yet

        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer

        assert check() is True  # cancel_event.set() happened before re-raise


def _grounded_provider(monkeypatch) -> ExistingSnipraVLMProvider:
    from app.core.config import settings

    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_provider", "claude")
    return ExistingSnipraVLMProvider()


class TestExtractFramesRealBoundary:
    """Section 2: the async _extract_frames() -> asyncio.to_thread(
    _extract_frames_blocking) boundary must hand back real frames (a
    list[bytes]), never an unawaited coroutine object. Only
    _extract_frames_blocking is monkeypatched here - _extract_frames()
    itself is the REAL method, so this exercises the actual to_thread
    plumbing, not a fake replacement of it."""

    async def test_extract_frames_returns_real_list_not_a_coroutine(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)

        def _fake_blocking(asset_id, start_us, end_us, sampling_fps=1.0):
            return [b"frame-a", b"frame-b"]

        monkeypatch.setattr(provider, "_extract_frames_blocking", _fake_blocking)

        result = await provider._extract_frames("asset-1", 0, 1_000_000, sampling_fps=1.0)

        assert not asyncio.iscoroutine(result)
        assert isinstance(result, list)
        assert result == [b"frame-a", b"frame-b"]

    async def test_extract_frames_blocking_runs_off_the_event_loop(self, monkeypatch):
        """If _extract_frames() ever regressed to calling
        _extract_frames_blocking directly (no asyncio.to_thread), this
        wouldn't prove much; the real regression this guards is the
        opposite one from section 2 - double-wrapping an async
        _extract_frames in asyncio.to_thread from a caller. That is
        exercised structurally: _extract_frames_blocking is a plain `def`,
        not `async def` - asserted here so nobody re-adds `async` to it
        without also breaking this test's monkeypatch (a plain callable is
        what asyncio.to_thread requires)."""
        provider = _grounded_provider(monkeypatch)
        assert not inspect.iscoroutinefunction(provider._extract_frames_blocking)
        assert inspect.iscoroutinefunction(provider._extract_frames)


def _content_addressed_fakes(monkeypatch, provider, frames_by_chunk: dict, vlm_by_frames: dict):
    """Deterministic-by-content fakes (not by call order): _extract_frames
    is keyed by (start_us, end_us), and _call_vlm_with_frames is keyed by
    the actual frames bytes it receives - both robust to whatever real
    order concurrent chunks happen to execute in, unlike a stateful
    call-counter fake."""

    async def _fake_extract(asset_id, start_us, end_us, sampling_fps=1.0):
        frames = frames_by_chunk[(start_us, end_us)]
        return [ExtractedFrame(frame_id=f"frame_{i}", timestamp_us=start_us + int(i / sampling_fps * 1_000_000), data=frame_data) for i, frame_data in enumerate(frames)]

    def _fake_vlm(system, prompt, frames):
        # Use the first frame's data as the key
        frame_key = frames[0].data if frames else b""
        outcome = vlm_by_frames.get(frame_key, vlm_by_frames.get(frames[0].data if frames else b""))
        if outcome is None:
            outcome = vlm_by_frames.get(frames[0].data if frames else b"")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(provider, "_extract_frames", _fake_extract)
    monkeypatch.setattr(provider, "_call_vlm_with_frames", _fake_vlm)


class TestConcurrencyParity:
    """Section 19 - CRITICAL: max_concurrency=1 vs max_concurrency=3 must
    produce byte-identical VideoIntelligenceResult.model_dump(mode="json")
    for the same deterministic inputs. This is what proves the Phase 4.3A
    refactor did not change aggregation semantics before relying on N=3 in
    production."""

    async def test_summarize_video_parity_mixed_success_and_failure(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        request = VideoIntelligenceRequest(
            asset_id="asset-1", start_us=0, end_us=65_000_000,
            mode="summarize_video", objective="Summarize",
        )
        plan = plan_analysis(request)  # COARSE_SCAN, 3 chunks (0-30s,30-60s,60-65s)
        assert len(plan.chunks) == 3

        ok_response = json.dumps({
            "observations": [{"start": "0.0s", "end": "1.0s",
                               "description": "seen", "confidence": 0.8, "nature": "observation", "frame_ids": ["frame_0", "frame_1"]}],
            "events": [], "scenes": [], "objects": [],
        })
        frames_by_chunk = {
            (c.start_us, c.end_us): [b"frame-data-" + str(c.index).encode()] for c in plan.chunks
        }
        vlm_by_frames = {
            b"frame-data-0": ok_response,
            b"frame-data-1": RuntimeError("boom"),
            b"frame-data-2": ok_response,
        }

        async def _run(max_concurrency):
            monkeypatch.setattr(chunk_executor_module, "MAX_CONCURRENT_CHUNKS", max_concurrency)
            _content_addressed_fakes(monkeypatch, provider, frames_by_chunk, vlm_by_frames)
            return await provider.summarize_video(request, plan)

        result_n1 = await _run(1)
        result_n3 = await _run(3)

        assert result_n1.model_dump(mode="json") == result_n3.model_dump(mode="json")
        assert result_n1.metadata.failed_chunks == [{"chunk_index": 1, "error": "boom"}]

    async def test_ask_video_parity_all_success(self, monkeypatch):
        provider = _grounded_provider(monkeypatch)
        request = VideoIntelligenceRequest(
            asset_id="asset-1", start_us=0, end_us=90_000_000,
            mode="ask_video", question="what happens?",
        )
        plan = plan_analysis(request)  # TARGETED_SCAN, multi-chunk
        assert len(plan.chunks) > 1

        frames_by_chunk = {
            (c.start_us, c.end_us): [b"frame-data-" + str(c.index).encode()] for c in plan.chunks
        }
        vlm_by_frames = {
            b"frame-data-" + str(c.index).encode(): json.dumps({
                "answer": f"answer-{c.index}", "confidence": 0.7,
                "observations": [{"start": "0.0s", "end": "1.0s", "description": "obs", "confidence": 0.7, "nature": "observation", "frame_ids": ["frame_0", "frame_1"]}],
                "events": [],
            })
            for c in plan.chunks
        }

        async def _run(max_concurrency):
            monkeypatch.setattr(chunk_executor_module, "MAX_CONCURRENT_CHUNKS", max_concurrency)
            _content_addressed_fakes(monkeypatch, provider, frames_by_chunk, vlm_by_frames)
            return await provider.ask_video(request, plan)

        result_n1 = await _run(1)
        result_n3 = await _run(3)

        assert result_n1.model_dump(mode="json") == result_n3.model_dump(mode="json")
        assert result_n1.metadata.chunks_failed == 0


class TestMockPerformanceSmoke:
    """Section 30: 6 chunks, ~200ms simulated VLM latency each. PASS is
    decided by max_active/waves/ordering equality, never by a rigid wall-
    clock threshold - wall time is recorded as information only."""

    async def test_concurrency_3_uses_fewer_waves_than_concurrency_1(self):
        active = 0
        max_active_by_run: dict[int, int] = {}

        async def worker(chunk):
            nonlocal active
            active += 1
            max_active_by_run[chunk.index] = max(max_active_by_run.get(chunk.index, 0), active)
            await asyncio.sleep(0.2)
            active -= 1
            return chunk.index

        async def run(max_concurrency):
            nonlocal active, max_active_by_run
            active = 0
            max_active_by_run = {}
            t0 = time.perf_counter()
            outcomes = await run_chunks_bounded(_chunks(6), worker, max_concurrency=max_concurrency)
            wall_s = time.perf_counter() - t0
            return outcomes, wall_s, max(max_active_by_run.values())

        outcomes_seq, wall_seq, max_active_seq = await run(1)
        outcomes_conc, wall_conc, max_active_conc = await run(3)

        # Decisive assertions - not timing-sensitive.
        assert max_active_seq == 1
        assert max_active_conc > 1
        assert max_active_conc <= 3
        assert [o.value for o in outcomes_seq] == list(range(6))
        assert [o.value for o in outcomes_conc] == list(range(6))

        # Informational only (printed, not asserted rigidly - CI timing varies).
        print(f"\n[mock perf smoke] concurrency=1 wall={wall_seq:.3f}s "
              f"concurrency=3 wall={wall_conc:.3f}s")
