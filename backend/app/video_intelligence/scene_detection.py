"""Phase 4.2b - real scene boundary detection (I/O boundary, no wiring yet).

Sole responsibility: media -> temporal visual boundaries. Pure ffmpeg, no
cv2, no PySceneDetect, no LLM. detect_scene_boundaries() is the only public
symbol - the AnalysisPlanner (chunking.py) and everything above it stays
unaware of how boundaries were found.

Deliberately does NOT import app.pipeline.signals, even though that module
already has an equivalent ffmpeg scene-change scan (_scene_times): importing
it would make video_intelligence depend on the legacy auto-clipper pipeline
module. The ffmpeg command/regex are small and stable enough that
duplicating them here is the better trade-off - see the Phase 4.2 design
gate report for the reasoning.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core import media
from app.core.config import settings
from app.video_intelligence.chunking import SceneBoundary

# Real precedent (app/pipeline/signals.py::_scene_times), not a new arbitrary
# value. This is a TECHNICAL visual-transition threshold - NOT calibrated
# for VIA's asset mix and NOT an editorial-importance knob. It is not
# exposed to the Director, EvidenceRequest, EditorialPlan, or any UI.
DEFAULT_SCENE_CHANGE_THRESHOLD = 0.35

# Same value as _scene_times' timeout - deliberately not scaled to range
# duration in this slice (matches the existing precedent rather than
# inventing a new formula); revisit if real measurement shows it matters.
DEFAULT_DETECTION_TIMEOUT_S = 600

# Same cap and same "keep the first N after sorting" semantics as
# _scene_times - NOT the "N strongest" (no score is stored - see
# chunking.SceneBoundary - so there is nothing to rank by).
MAX_SCENE_BOUNDARIES = 500

_PTS_TIME_RE = re.compile(r"pts_time:(?P<time>\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class SceneDetectionResult:
    """detect_scene_boundaries_detailed()'s result: the boundaries actually
    found, plus whether the run completed cleanly enough to persist (Phase
    4.2D). `cacheable` is True only when ffmpeg genuinely ran to completion -
    including completing with zero real cuts, which is a legitimate result
    worth caching - and False for any environmental failure (ffmpeg missing,
    non-zero exit, timeout), where an empty `boundaries` would otherwise be
    indistinguishable from "we don't actually know". Not exposed to the
    Director/EvidenceRequest/EditorialPlan - purely a video_intelligence-
    internal detail."""

    boundaries: list[SceneBoundary]
    cacheable: bool


def detect_scene_boundaries(
    asset_path: str | Path,
    start_us: int,
    end_us: int,
    *,
    threshold: float = DEFAULT_SCENE_CHANGE_THRESHOLD,
) -> list[SceneBoundary]:
    """Detect visual scene cuts strictly inside (start_us, end_us).

    Runs `ffmpeg -ss <start> -i <asset> -t <duration> -vf
    "select='gt(scene,threshold)',showinfo" -an -f null -` and parses
    `pts_time:` from stderr. Input seeking (-ss before -i) is used for
    speed on large files - empirically verified (not assumed) that this
    resets pts_time to 0 at the seek point rather than reporting the
    original file's absolute timeline, so every parsed timestamp is
    converted back to an absolute asset timestamp via `start_us + pts_time`
    before being returned. Only detects VISUAL structure - no relation to
    editorial importance, and does not read or produce it.

    This is a pure I/O boundary: given the same media bytes and ffmpeg
    build, it is deterministic, but it is not a pure function of its
    arguments the way chunking.plan_analysis()/refine_plan_with_scenes()
    are.

    Failure semantics - scene detection is an optimization, never required
    for VIA to function:
    - invalid threshold or empty/inverted range: raises ValueError
      immediately, before any subprocess call - this is a caller bug, not
      an environment failure, so it does not silently degrade to [].
    - ffmpeg unavailable, non-zero exit, or timeout: returns [].
    - a real cancellation request (app.core.media.set_cancel_check) is NOT
      swallowed - FFmpegCanceledError propagates, matching how the rest of
      this codebase already expects cancellation to work.
    - malformed stderr lines are ignored (only well-formed `pts_time:`
      matches are parsed; anything else is simply not matched).
    - more than MAX_SCENE_BOUNDARIES real cuts: kept sorted, capped, never
      "top N strongest" (no score exists to rank by).

    Returns absolute-timestamp SceneBoundary, sorted ascending, deduplicated
    by exact microsecond, strictly excluding start_us and end_us themselves
    (those already belong to the AnalysisPlan, not to scene structure).

    Thin wrapper over detect_scene_boundaries_detailed() (Phase 4.2D) for
    every caller that only needs the boundaries, not whether the run is
    safe to persist - the public contract/signature is unchanged.
    """
    return detect_scene_boundaries_detailed(
        asset_path, start_us, end_us, threshold=threshold
    ).boundaries


def detect_scene_boundaries_detailed(
    asset_path: str | Path,
    start_us: int,
    end_us: int,
    *,
    threshold: float = DEFAULT_SCENE_CHANGE_THRESHOLD,
) -> SceneDetectionResult:
    """Same detection as detect_scene_boundaries(), but also reports whether
    the run is safe to persist (Phase 4.2D scene cache): see
    SceneDetectionResult. Cancellation is never wrapped into either outcome -
    FFmpegCanceledError still propagates unchanged."""
    if not (0 < threshold <= 1):
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    if end_us <= start_us:
        raise ValueError(f"end_us ({end_us}) must be greater than start_us ({start_us})")

    start_s = start_us / 1_000_000
    duration_s = (end_us - start_us) / 1_000_000

    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-ss", f"{start_s:.3f}",
        "-i", str(asset_path),
        "-t", f"{duration_s:.3f}",
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f", "null",
        "-",
    ]

    try:
        proc = media.run(cmd, timeout=DEFAULT_DETECTION_TIMEOUT_S)
    except media.FFmpegCanceledError:
        raise
    except (media.FFmpegError, subprocess.TimeoutExpired, OSError):
        return SceneDetectionResult(boundaries=[], cacheable=False)

    absolute_us: set[int] = set()
    for match in _PTS_TIME_RE.finditer(proc.stderr or ""):
        relative_us = start_us + round(float(match.group("time")) * 1_000_000)
        if start_us < relative_us < end_us:
            absolute_us.add(relative_us)

    sorted_us = sorted(absolute_us)[:MAX_SCENE_BOUNDARIES]
    return SceneDetectionResult(
        boundaries=[SceneBoundary(timestamp_us=ts) for ts in sorted_us],
        cacheable=True,
    )
