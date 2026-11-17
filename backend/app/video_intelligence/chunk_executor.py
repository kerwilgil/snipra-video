"""Phase 4.3A - bounded parallel execution of independent AnalysisChunk work.

Runs up to MAX_CONCURRENT_CHUNKS chunk workers at once instead of Phase 4.1's
one-at-a-time sequential loop. Each AnalysisChunk is an independent unit of
work over a disjoint temporal range (see chunking.py) - there is no
cross-chunk state to coordinate, so this module's only job is: bound how
much work runs at once, preserve deterministic index ordering in the output
regardless of real completion order, isolate one chunk's failure from the
rest, and propagate a real cancellation (asyncio.CancelledError or a genuine
media.FFmpegCanceledError) cleanly instead of masking it as an ordinary
per-chunk failure.

Deliberately NOT here: any editorial authority, any Director-facing
contract, any VLM/ffmpeg-specific logic. This module only knows how to run
an arbitrary async `worker(chunk) -> value` callable per chunk, bounded and
ordered - ExistingSnipraVLMProvider supplies the worker and does its own,
completely unchanged aggregation over the returned ChunkOutcome[] (see the
Phase 4.3 design gate report - VideoAnswer/VideoProcessingMetadata
aggregation semantics are out of scope for this module to touch).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core import media
from app.video_intelligence.chunking import AnalysisChunk

logger = logging.getLogger(__name__)

# Internal constant - never exposed to the Director, never a
# ProviderCapabilitiesModel field (Phase 4.3 design gate section 12: a
# single real provider exists today; promote to a capability only if/when a
# second provider with meaningfully different concurrency tolerance exists).
MAX_CONCURRENT_CHUNKS = 3


@dataclass(frozen=True)
class ChunkOutcome:
    """One chunk worker's result. `value` is whatever the caller's worker
    returned on success (opaque to this module); `error` is str(exception)
    on failure - the same shape the pre-4.3A sequential loops already used
    for their `failed_chunks` entries."""

    index: int
    ok: bool
    value: object = None
    error: str | None = None


async def run_chunks_bounded(
    chunks: list[AnalysisChunk],
    worker: Callable[[AnalysisChunk], Awaitable[object]],
    *,
    max_concurrency: int | None = None,
) -> list[ChunkOutcome]:
    """Run `worker(chunk)` for every chunk, at most `max_concurrency` at a
    time, and return one ChunkOutcome per chunk in chunks[i].index order -
    regardless of real completion order (asyncio.gather's own documented
    ordering guarantee: results correspond to the input awaitables' order,
    not to completion order).

    Failure isolation: any Exception raised by `worker(chunk)` becomes that
    chunk's ChunkOutcome(ok=False, error=...) - it never cancels sibling
    chunks and never escapes this function. The two exceptions that DO
    escape (request-global, not chunk-local) are asyncio.CancelledError and
    media.FFmpegCanceledError. gather() is always called with
    return_exceptions=True precisely so a chunk raising one of these can
    never leave its still-running siblings orphaned/detached (gather's
    default, return_exceptions=False, propagates the first exception
    immediately without waiting for the rest) - every task is always
    awaited to completion before this function returns or re-raises.

    The Semaphore and the ffmpeg cancel-check are created fresh on every
    call (never module/self state) - concurrency is bounded per-request,
    not globally. This function does not know or care that
    ExistingSnipraVLMProvider is a process-wide singleton shared across
    concurrent requests (see the Phase 4.3 design gate's FAIRNESS section).
    """
    if not chunks:
        return []

    # Resolved at call time (not baked into the signature's default), so
    # tests can monkeypatch MAX_CONCURRENT_CHUNKS itself and have it take
    # effect - a plain `= MAX_CONCURRENT_CHUNKS` default would snapshot the
    # value once, at function-definition time.
    if max_concurrency is None:
        max_concurrency = MAX_CONCURRENT_CHUNKS
    semaphore = asyncio.Semaphore(max_concurrency)

    # Bridges asyncio-level cancellation to app.core.media.run()'s own
    # poll-based subprocess kill switch (media.py: _cancel_check is a
    # ContextVar, checked every ~0.1s inside run()'s wait loop).
    # asyncio.to_thread() copies the calling coroutine's contextvars context
    # into the worker thread at call time, so setting this here - before any
    # chunk task is created - is enough for every chunk's own to_thread-
    # wrapped ffmpeg call to observe it, without per-chunk wiring.
    cancel_event = threading.Event()
    cancel_token = media.set_cancel_check(cancel_event.is_set)

    async def _run_one(chunk: AnalysisChunk) -> ChunkOutcome:
        async with semaphore:
            try:
                value = await worker(chunk)
                return ChunkOutcome(index=chunk.index, ok=True, value=value)
            except media.FFmpegCanceledError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("chunk %s failed: %s", chunk.index, exc)
                return ChunkOutcome(index=chunk.index, ok=False, error=str(exc))

    tasks = [asyncio.ensure_future(_run_one(chunk)) for chunk in chunks]

    try:
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        # Our own await was cancelled directly (the caller's task was
        # cancelled). Flip the ffmpeg cancel-check immediately so any
        # in-flight extraction can stop within one poll cycle, then cancel
        # and drain every child task before propagating, so none is left
        # pending once this function actually returns control.
        cancel_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        media.reset_cancel_check(cancel_token)

    # return_exceptions=True already waited for every task to settle, so
    # nothing is left running at this point. Chunk-local failures never
    # reach here as exception objects (caught inside _run_one and returned
    # as ChunkOutcome); anything that IS an exception here is
    # request-global - re-raise the first one, preserving its real type
    # (CancelledError or FFmpegCanceledError).
    for result in raw_results:
        if isinstance(result, BaseException):
            raise result

    return list(raw_results)  # type: ignore[return-value]
