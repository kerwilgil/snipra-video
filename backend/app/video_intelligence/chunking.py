"""Provider-neutral chunking for long-form video analysis.

Also home to the Phase 4.1 Adaptive Analysis Planner (`plan_analysis`, and the
`AnalysisStrategy` / `AnalysisChunk` / `VideoAnalysisPlan` contracts it
produces). The planner is a PURE, DETERMINISTIC sizing function: given a
VideoIntelligenceRequest, it decides how many frames to ask for and how to
split the range into chunks. It has no editorial authority - it never reads
or produces confidence, coverage, event/highlight scores, or anything else
that belongs to the Director/Quality Gate/ranking layers, and its output
types must never appear in EditorialPlan or EvidenceRequest schemas. Slice 1
of Phase 4.1 wires none of this into the runtime pipeline yet: the planner
exists and is fully tested, but VideoIntelligenceService and
ExistingSnipraVLMProvider are untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.video_intelligence.models import (
    VideoAnalysisConfig,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
)

if TYPE_CHECKING:
    from app.video_intelligence.models import VideoIntelligenceRequest


@dataclass(frozen=True)
class VideoChunk:
    """A single video chunk with temporal bounds."""
    index: int
    start_us: int
    end_us: int
    duration_us: int

    @property
    def duration_s(self) -> float:
        return self.duration_us / 1_000_000

    @property
    def start_s(self) -> float:
        return self.start_us / 1_000_000

    @property
    def end_s(self) -> float:
        return self.end_us / 1_000_000


def create_chunk_plan(
    start_us: int,
    end_us: int,
    config: VideoAnalysisConfig,
) -> list[VideoChunk]:
    """
    Create a deterministic chunk plan for a video range.

    Rules:
    - Covers 100% of the requested range
    - No gaps between chunks
    - Overlap never >= chunk duration
    - Final chunk clamped to end_us
    - Small durations produce 1 chunk
    - Exact microsecond precision
    - Deterministic
    """
    if start_us >= end_us:
        raise ValueError("end_us must be greater than start_us")

    chunk_duration_us = config.chunk_duration_s * 1_000_000
    overlap_us = config.overlap_s * 1_000_000

    if overlap_us >= chunk_duration_us:
        raise ValueError("overlap_s must be less than chunk_duration_s")

    total_duration_us = end_us - start_us
    chunk_duration_us = min(chunk_duration_us, total_duration_us)

    chunks = []
    current_start = start_us
    index = 0

    while current_start < end_us:
        chunk_end = min(current_start + chunk_duration_us, end_us)
        chunk_duration = chunk_end - current_start

        chunk = VideoChunk(
            index=index,
            start_us=current_start,
            end_us=chunk_end,
            duration_us=chunk_duration,
        )
        chunks.append(chunk)

        index += 1

        if chunk_end >= end_us:
            break

        # Next chunk starts at (chunk_end - overlap)
        next_start = chunk_end - (config.overlap_s * 1_000_000)
        if next_start <= current_start:
            # Overlap too large or chunk too small - advance minimally
            next_start = current_start + 1
        current_start = next_start

        if current_start >= end_us:
            break

    return chunks


def chunk_video_range(
    start_us: int,
    end_us: int,
    chunk_duration_s: int = 30,
    overlap_s: int = 2,
) -> list[tuple[int, int, int]]:
    """
    Legacy-compatible chunking function.
    Returns list of (index, start_us, end_us).
    """
    config = VideoAnalysisConfig(
        chunk_duration_s=chunk_duration_s,
        overlap_s=overlap_s,
    )
    create_chunk_plan(start_us, end_us, config)
    return [(c.index, c.start_us, c.end_us) for c in config]


def get_chunk_info(chunks: list[VideoChunk]) -> dict:
    """Get summary info about a chunk plan."""
    if not chunks:
        return {
            "num_chunks": 0,
            "total_duration_s": 0,
            "has_overlap": False,
        }

    total_duration_s = sum(c.duration_us for c in chunks) / 1_000_000
    has_overlap = len(chunks) > 1

    return {
        "num_chunks": len(chunks),
        "total_duration_s": total_duration_s,
        "has_overlap": has_overlap,
        "chunk_duration_s": chunks[0].duration_s if chunks else 0,
        "overlap_s": chunks[0].overlap_s if has_overlap and len(chunks) > 1 else 0,
    }


def validate_chunk_plan(
    chunks: list[VideoChunk],
    start_us: int,
    end_us: int,
) -> bool:
    """Validate that a chunk plan is correct."""
    if not chunks:
        return start_us >= end_us

    # Check coverage
    if chunks[0].start_us != start_us:
        return False
    if chunks[-1].end_us != end_us:
        return False

    # Check no gaps
    for i in range(1, len(chunks)):
        if chunks[i].start_us > chunks[i - 1].end_us:
            return False

    # Check no chunk exceeds end
    if any(c.end_us > chunks[-1].end_us for c in chunks):
        return False

    # Check overlap < chunk duration
    for i in range(1, len(chunks)):
        overlap = chunks[i - 1].end_us - chunks[i].start_us
        if overlap < 0:
            return False  # Gap

    return True


# ---------------------------------------------------------------------------
# Phase 4.1 Slice 1 - Adaptive Analysis Planner (pure, unwired)
# ---------------------------------------------------------------------------
#
# Internal planner defaults. These are NOT part of VideoAnalysisConfig and are
# never read from a request: VideoAnalysisConfig is Director/EvidenceRequest-
# facing surface (see app/editor/evidence_request.py) and its chunk_duration_s
# /overlap_s/max_frames_per_chunk fields are legacy and were never actually
# enforced by the pipeline. The planner intentionally does not read
# request.analysis_config - it owns sampling decisions itself, using only the
# request's temporal range and mode, so these constants can be retuned after
# load testing without touching editorial-facing contracts.
MAX_SAMPLING_FPS = 1.0
MAX_FRAMES_PER_CHUNK = 40
TARGET_CHUNK_S = 30
MAX_CHUNKS_PER_REQUEST = 60
# Mirrors VideoAnalysisConfig.chunk_duration_s's own validator ceiling
# (app/video_intelligence/models.py) - not a new arbitrary limit, but the
# already-established technical maximum for a single chunk.
MAX_CHUNK_DURATION_S = 600

# Derived, not hardcoded: the longest range a single chunk can cover at max
# density without exceeding MAX_FRAMES_PER_CHUNK.
FULL_SCAN_MAX_S = MAX_FRAMES_PER_CHUNK / MAX_SAMPLING_FPS


class AnalysisStrategy(StrEnum):
    FULL_SCAN = "full_scan"
    TARGETED_SCAN = "targeted_scan"
    COARSE_SCAN = "coarse_scan"


class AnalysisChunk(BaseModel):
    """One VLM call's worth of work: a temporal window plus how many frames
    to extract from it. Frame count is the primary decision; sampling_fps is
    derived from it for observability, not the other way around, so the
    MAX_FRAMES_PER_CHUNK cap can never be missed by a rounding round-trip."""

    index: int = Field(ge=0)
    start_us: int = Field(ge=0)
    end_us: int
    frames_planned: int = Field(ge=1, le=MAX_FRAMES_PER_CHUNK)
    sampling_fps: float = Field(gt=0)

    @model_validator(mode="after")
    def _validate_range(self) -> AnalysisChunk:
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be greater than start_us")
        return self


class VideoAnalysisPlan(BaseModel):
    """The output of plan_analysis(): an executable, budget-respecting plan.
    Never exposed to the Director - see the module docstring."""

    strategy: AnalysisStrategy
    chunks: list[AnalysisChunk] = Field(min_length=1)
    total_planned_frames: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_plan(self) -> VideoAnalysisPlan:
        if len(self.chunks) > MAX_CHUNKS_PER_REQUEST:
            raise ValueError(
                f"plan has {len(self.chunks)} chunks, exceeds "
                f"MAX_CHUNKS_PER_REQUEST={MAX_CHUNKS_PER_REQUEST}"
            )

        expected_total = sum(c.frames_planned for c in self.chunks)
        if self.total_planned_frames != expected_total:
            raise ValueError(
                f"total_planned_frames={self.total_planned_frames} does not "
                f"match sum(chunk.frames_planned)={expected_total}"
            )

        # No gaps, no overlap. Deliberate: the legacy chunking.create_chunk_plan
        # overlap_s exists to give the VLM cross-chunk continuity context for
        # narrative/scene work, which is not this planner's job - it only
        # decides sampling density and call count. Carrying that overlap
        # forward would silently inflate total_planned_frames beyond what the
        # frame budget actually allows. The planner always calls
        # create_chunk_plan with overlap_s=0, and this validator makes that
        # an explicit, checked invariant of every plan rather than an
        # accident of the current implementation.
        for i in range(1, len(self.chunks)):
            prev, cur = self.chunks[i - 1], self.chunks[i]
            if cur.start_us < prev.end_us:
                raise ValueError(f"chunk {i} overlaps chunk {i - 1}")
            if cur.start_us > prev.end_us:
                raise ValueError(f"gap between chunk {i - 1} and chunk {i}")

        return self


def _sized_chunk_duration_s(range_s: float) -> int:
    """How large should each chunk be so the plan needs at most
    MAX_CHUNKS_PER_REQUEST chunks? Grows the chunk beyond TARGET_CHUNK_S only
    when the naive 30s chunking would need more calls than the budget allows,
    capped at MAX_CHUNK_DURATION_S (the same ceiling VideoAnalysisConfig
    already enforces)."""
    naive_chunks = math.ceil(range_s / TARGET_CHUNK_S)
    if naive_chunks <= MAX_CHUNKS_PER_REQUEST:
        return TARGET_CHUNK_S
    return min(MAX_CHUNK_DURATION_S, math.ceil(range_s / MAX_CHUNKS_PER_REQUEST))


def plan_analysis(
    request: VideoIntelligenceRequest,
    provider_capabilities: dict | None = None,
) -> VideoAnalysisPlan:
    """Pure, deterministic sizing of a VideoIntelligenceRequest into an
    executable VideoAnalysisPlan.

    No I/O, no ffmpeg, no DB, no LLM, no provider calls, no editor state.
    Depends only on request.start_us/end_us/mode and, optionally, a
    provider's own frame-count ceiling passed as
    provider_capabilities={"max_frames_per_call": N} - a soft, duck-typed
    override so this slice doesn't need to add a field to
    ProviderCapabilitiesModel before there's a real consumer for it.

    Strategy selection:
    - range fits one chunk at max density (<= FULL_SCAN_MAX_S)  -> FULL_SCAN
    - longer range, mode == ask_video                            -> TARGETED_SCAN
    - longer range, mode == summarize_video                      -> COARSE_SCAN

    TARGETED_SCAN and COARSE_SCAN currently share identical chunk-sizing
    math (see _sized_chunk_duration_s) - the strategy label distinguishes
    "this is a Director-directed question" from "this is a broad summary"
    for observability/telemetry, not because the sampling formula differs
    today. A single ask_video request can still resolve into multiple
    AnalysisChunk entries: this planner has no opinion on how a future
    provider-side implementation synthesizes multiple chunks into one
    VideoIntelligenceResult, and does not introduce a second Director round
    to do it.

    Raises VideoIntelligenceError(PLAN_INFEASIBLE) only when no chunk-count
    fits within MAX_CHUNKS_PER_REQUEST even at MAX_CHUNK_DURATION_S (the
    existing per-chunk ceiling) - i.e. range_s > MAX_CHUNK_DURATION_S *
    MAX_CHUNKS_PER_REQUEST (10h at current defaults). This is a composition
    of two already-justified technical limits, not a new arbitrary cutoff.
    """
    max_frames_per_chunk = MAX_FRAMES_PER_CHUNK
    if provider_capabilities:
        cap = provider_capabilities.get("max_frames_per_call")
        if cap:
            max_frames_per_chunk = min(max_frames_per_chunk, cap)

    range_us = request.end_us - request.start_us
    range_s = range_us / 1_000_000

    if range_s <= FULL_SCAN_MAX_S:
        strategy = AnalysisStrategy.FULL_SCAN
        chunk_duration_s = max(1, math.ceil(range_s))
    else:
        strategy = (
            AnalysisStrategy.TARGETED_SCAN
            if request.mode == "ask_video"
            else AnalysisStrategy.COARSE_SCAN
        )
        chunk_duration_s = _sized_chunk_duration_s(range_s)

    config = VideoAnalysisConfig(chunk_duration_s=chunk_duration_s, overlap_s=0)
    raw_chunks = create_chunk_plan(request.start_us, request.end_us, config)

    if len(raw_chunks) > MAX_CHUNKS_PER_REQUEST:
        raise VideoIntelligenceError(
            VideoIntelligenceErrorCode.PLAN_INFEASIBLE,
            f"Range of {range_s:.1f}s cannot be planned within "
            f"{MAX_CHUNKS_PER_REQUEST} chunks even at the "
            f"{MAX_CHUNK_DURATION_S}s max chunk duration "
            f"({len(raw_chunks)} chunks required). Narrow the requested range.",
            details={"range_s": range_s, "chunks_required": len(raw_chunks)},
        )

    chunks: list[AnalysisChunk] = []
    for raw in raw_chunks:
        frames_planned = max(
            1,
            min(max_frames_per_chunk, math.ceil(raw.duration_s * MAX_SAMPLING_FPS)),
        )
        sampling_fps = (
            frames_planned / raw.duration_s if raw.duration_s > 0 else MAX_SAMPLING_FPS
        )
        chunks.append(
            AnalysisChunk(
                index=raw.index,
                start_us=raw.start_us,
                end_us=raw.end_us,
                frames_planned=frames_planned,
                sampling_fps=sampling_fps,
            )
        )

    return VideoAnalysisPlan(
        strategy=strategy,
        chunks=chunks,
        total_planned_frames=sum(c.frames_planned for c in chunks),
    )


# ---------------------------------------------------------------------------
# Phase 4.2a - Pure scene-aware plan refinement (no detector, no I/O yet)
# ---------------------------------------------------------------------------
#
# SceneBoundary is intentionally minimal - just a timestamp. No score,
# confidence, or importance field: Phase 4.2's design gate was explicit that
# a scene-change signal (if one is ever attached) is a technical transition
# strength, never editorial relevance, and this slice doesn't need one at
# all since there's no real detector yet (Phase 4.2b).
class SceneBoundary(BaseModel):
    timestamp_us: int = Field(ge=0)


# Matches app/pipeline/detect.py::_snap_to_scene's existing tol=2.5 default -
# not a new arbitrary value, the same tolerance already proven for "snap to
# nearest real cut" elsewhere in this codebase.
SCENE_SNAP_TOLERANCE_US = int(2.5 * 1_000_000)


def _best_boundary_snap(original_us: int, boundary_us: list[int]) -> int | None:
    """Nearest boundary timestamp to original_us within
    SCENE_SNAP_TOLERANCE_US, or None if none qualifies. Tie-break: smaller
    distance wins; among equal distances, the earlier (smaller) timestamp
    wins - deterministic by construction of the (distance, timestamp) sort
    key, no separate tie-break branch needed."""
    best: int | None = None
    best_key: tuple[int, int] | None = None
    for ts in boundary_us:
        distance = abs(ts - original_us)
        if distance > SCENE_SNAP_TOLERANCE_US:
            continue
        key = (distance, ts)
        if best_key is None or key < best_key:
            best_key = key
            best = ts
    return best


def _min_duration_us_for_cap(base_fps_cap: float) -> int:
    """Shortest duration (us) that can honor >=1 frame without exceeding
    base_fps_cap. A candidate snap that would shrink either neighboring
    chunk below this is rejected outright (chunking.py:
    refine_plan_with_scenes) - AnalysisChunk.frames_planned can never be
    forced to 0 (Field(ge=1)), so a too-short chunk would otherwise be the
    one situation where sampling_fps could exceed the base chunk's own
    density despite floor-based rounding everywhere else."""
    if base_fps_cap <= 0:
        return 0
    return math.ceil(1_000_000 / base_fps_cap)


def _refined_frames_for_duration(
    duration_s: float, base_fps_cap: float, max_frames_per_chunk: int
) -> tuple[int, float]:
    """Recompute (frames_planned, sampling_fps) for a chunk whose duration
    changed after scene-boundary snapping. Never denser than base_fps_cap
    (the corresponding base chunk's own sampling_fps) - scene-awareness can
    only ever reduce or preserve density, never increase it.

    Deliberately NOT the same helper plan_analysis() uses inline (math.ceil
    against MAX_SAMPLING_FPS): that formula's round-up-for-coverage behavior
    is already tested/frozen Phase 4.1 output, and reusing it here would
    let a fractional duration_s round sampling_fps slightly ABOVE
    base_fps_cap (ceil(duration_s*cap) can exceed duration_s*cap). This
    helper uses math.floor instead, which can only round DOWN, so the
    never-exceed invariant holds by construction rather than by afterward
    clamping a value that was already computed too high.
    """
    # +1e-9 guards against float round-trip noise (base_fps_cap is itself
    # derived as frames/duration_s elsewhere, so duration_s*base_fps_cap can
    # land a hair under an integer, e.g. 29.999999999999996) - negligible
    # next to the frame-count granularity this rounds to, but without it an
    # UNCHANGED boundary could spuriously lose a frame to floor().
    frames_planned = max(
        1, min(max_frames_per_chunk, math.floor(duration_s * base_fps_cap + 1e-9))
    )
    sampling_fps = frames_planned / duration_s if duration_s > 0 else 0.0
    return frames_planned, sampling_fps


def refine_plan_with_scenes(
    plan: VideoAnalysisPlan,
    boundaries: list[SceneBoundary],
) -> VideoAnalysisPlan:
    """Pure, deterministic refinement of an already-valid VideoAnalysisPlan:
    nudges INTERNAL chunk boundaries toward real scene cuts within
    SCENE_SNAP_TOLERANCE_US. No I/O, no ffmpeg, no LLM - boundaries must
    already be detected (Phase 4.2b, not implemented yet).

    Guarantees, always:
    - len(result.chunks) == len(plan.chunks) - scene count never changes
      chunk count / VLM call count (that stays plan_analysis()'s job).
    - result.strategy == plan.strategy
    - result.chunks[0].start_us == plan.chunks[0].start_us and
      result.chunks[-1].end_us == plan.chunks[-1].end_us - the original
      request range is never touched, only chunk boundaries BETWEEN chunks.
    - contiguous, no gaps, no overlaps (VideoAnalysisPlan's own validator
      checks this on the way out).
    - every chunk's sampling_fps <= its base-plan counterpart's
      sampling_fps - scene-awareness can only reduce or preserve density.

    If boundaries is empty or the plan has a single chunk (nothing internal
    to refine - this is what FULL_SCAN plans always look like), returns
    plan unchanged. If anything about the candidate refinement can't satisfy
    the guarantees above, falls back to returning plan unchanged rather
    than emit a partially-refined or invalid result - scene-awareness is a
    safe optimization, never a required one.
    """
    if not boundaries or len(plan.chunks) <= 1:
        return plan

    boundary_us = sorted({b.timestamp_us for b in boundaries})

    # original_edges[0] = request start, original_edges[-1] = request end -
    # both fixed, never candidates for snapping. original_edges[1:-1] are
    # the N-1 internal boundaries that may move.
    original_edges = [c.start_us for c in plan.chunks] + [plan.chunks[-1].end_us]

    new_internal: list[int] = []
    for i in range(1, len(plan.chunks)):
        original = original_edges[i]
        left_bound = original_edges[i - 1]
        right_bound = original_edges[i + 1]

        candidate = _best_boundary_snap(original, boundary_us)
        if candidate is None or not (left_bound < candidate < right_bound):
            new_internal.append(original)
            continue

        left_cap = plan.chunks[i - 1].sampling_fps
        right_cap = plan.chunks[i].sampling_fps
        if (candidate - left_bound) < _min_duration_us_for_cap(left_cap):
            new_internal.append(original)
            continue
        if (right_bound - candidate) < _min_duration_us_for_cap(right_cap):
            new_internal.append(original)
            continue

        new_internal.append(candidate)

    new_edges = [original_edges[0], *new_internal, original_edges[-1]]

    # Defense in depth: today's constants make a middle chunk shorter than
    # 2x tolerance structurally impossible (min chunk size is
    # TARGET_CHUNK_S=30s, tolerance is 2.5s), so the per-boundary checks
    # above already guarantee monotonicity. Fail safe instead of proving it.
    if any(new_edges[i] >= new_edges[i + 1] for i in range(len(new_edges) - 1)):
        return plan

    refined_chunks: list[AnalysisChunk] = []
    for i, base_chunk in enumerate(plan.chunks):
        start_us, end_us = new_edges[i], new_edges[i + 1]
        duration_s = (end_us - start_us) / 1_000_000
        frames_planned, sampling_fps = _refined_frames_for_duration(
            duration_s, base_chunk.sampling_fps, MAX_FRAMES_PER_CHUNK
        )
        refined_chunks.append(
            AnalysisChunk(
                index=i,
                start_us=start_us,
                end_us=end_us,
                frames_planned=frames_planned,
                sampling_fps=sampling_fps,
            )
        )

    # Final safety net for the never-exceed-base-density guarantee - should
    # never trigger given the per-boundary admissibility checks above, but
    # this is what actually enforces the invariant rather than merely
    # relying on those checks having covered every case correctly.
    for base_chunk, refined_chunk in zip(plan.chunks, refined_chunks, strict=True):
        if refined_chunk.sampling_fps > base_chunk.sampling_fps + 1e-9:
            return plan

    try:
        return VideoAnalysisPlan(
            strategy=plan.strategy,
            chunks=refined_chunks,
            total_planned_frames=sum(c.frames_planned for c in refined_chunks),
        )
    except ValidationError:
        return plan