"""Phase 4.2D - persistent scene boundary cache.

Sole responsibility: avoid repeating a real ffmpeg scene detection run
(measured ~41s on a ~19min asset, see the Phase 4.2C slice report) when a
validated result for the same asset content + threshold + exact temporal
range already exists. Storage is EditorAsset.asset_metadata - already
persistent, already belongs to the asset, no migration needed (see the
Phase 4.2D design gate: app.editor.auto_import already updates a sibling JSON
column, EditorProject.agent_metadata, with the exact read-merge-reassign-
commit pattern this module follows).

This is a cache for scene DETECTION RESULTS only - never for
VideoAnalysisPlan, refined plans, sampling_fps, frames_planned, Director
output, or VIA evidence. Those stay cheap and deterministic to recompute
every request (chunking.plan_analysis/refine_plan_with_scenes). Also
deliberately separate from app.video_intelligence.cache.VideoIntelligenceCache
(that caches whole VIA request/result pairs; this caches derived temporal
structure of media/range) - the two must never be merged into one key space.

Fail-safe by construction: every public function here returns a null-ish
result (None/False) instead of raising on any DB or data problem. A cache
miss, a corrupt entry, or a failed write must never fail a VIA request - the
caller always still has the option to run detect_scene_boundaries_detailed()
for real.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlmodel import Session

from app.core.db import engine
from app.editor.models import EditorAsset
from app.video_intelligence.chunking import SceneBoundary
from app.video_intelligence.scene_detection import MAX_SCENE_BOUNDARIES

# Bumping this invalidates every previously persisted entry across every
# asset (compared against the "version" stored alongside "entries" - see
# write_cached_boundaries) without a migration - only bump if the detector's
# parser, timestamp semantics, or output semantics actually change. Never a
# git commit hash or a date: a small number under this module's own control.
SCENE_DETECTOR_CACHE_VERSION = 1

# Namespaced so scene detection's cache never collides with unrelated
# asset_metadata written by app.editor.auto_import/service (e.g. "origin",
# "source_job_id") or by any future video_intelligence sub-feature that also
# wants a slot under "video_intelligence".
_METADATA_NAMESPACE = "video_intelligence"
_METADATA_KEY = "scene_detection"

# Deliberately small and FIFO (oldest entry evicted first, by list order -
# no separate timestamp field needed for that). A single VIA asset is
# realistically probed over a handful of distinct ranges (whole-asset
# summarize_video reruns, a few narrower re-analyses); 8 comfortably covers
# that without letting asset_metadata grow unbounded for a heavily-reused
# asset.
MAX_SCENE_CACHE_ENTRIES_PER_ASSET = 8


@dataclass(frozen=True)
class SceneCacheLookup:
    """One EditorAsset read, resolved once per request and reused for both
    cache lookup and (on miss) the real detector call - avoids the two
    separate DB round trips Phase 4.2C's asset-path-only resolution and a
    naive Phase 4.2D fingerprint/metadata lookup would otherwise require."""

    asset_id: str
    original_path: str
    fingerprint: str
    asset_metadata: dict


def resolve_scene_cache_lookup(asset_id: str) -> SceneCacheLookup | None:
    """Best-effort EditorAsset read. Returns None (never raises) when the
    asset or DB is unavailable - same fallback contract as the rest of
    scene-aware composition (Phase 4.2C): a resolution failure here must
    fall back to running detection without a cache, never fail the request."""
    try:
        with Session(engine) as session:
            asset = session.get(EditorAsset, asset_id)
            if not asset:
                return None
            return SceneCacheLookup(
                asset_id=asset.id,
                original_path=asset.original_path,
                fingerprint=asset.fingerprint,
                asset_metadata=dict(asset.asset_metadata or {}),
            )
    except Exception:
        return None


def _entry_matches(
    entry, *, fingerprint: str, threshold: float, start_us: int, end_us: int
) -> bool:
    if not isinstance(entry, dict):
        return False
    return (
        entry.get("fingerprint") == fingerprint
        and entry.get("threshold") == threshold
        and entry.get("start_us") == start_us
        and entry.get("end_us") == end_us
    )


def _validate_boundaries(raw, start_us: int, end_us: int) -> list[SceneBoundary] | None:
    """Never trusts persisted JSON blindly (section 7): rejects anything
    that isn't a flat list of in-range integers, and reconstructs through
    SceneBoundary so the return value carries the same guarantees a real
    detect_scene_boundaries() call would. Returns None (not a partial/
    repaired list) for anything that fails validation - the caller treats
    that identically to a cache miss."""
    if not isinstance(raw, list):
        return None
    try:
        timestamps = {int(ts) for ts in raw}
    except (TypeError, ValueError):
        return None
    if any(ts <= start_us or ts >= end_us for ts in timestamps):
        return None
    if len(timestamps) > MAX_SCENE_BOUNDARIES:
        return None
    return [SceneBoundary(timestamp_us=ts) for ts in sorted(timestamps)]


def read_cached_boundaries(
    asset_metadata: dict,
    *,
    fingerprint: str,
    threshold: float,
    start_us: int,
    end_us: int,
) -> list[SceneBoundary] | None:
    """Returns validated cached boundaries on a hit - possibly an empty list,
    meaning "detection genuinely ran and found zero cuts" (see
    write_cached_boundaries/cacheable) - or None on any kind of miss: no
    namespace, version mismatch, no matching entry (asset content, threshold,
    or exact range differs), or a corrupt payload. Pure function of the
    caller-supplied asset_metadata dict - no I/O, so it's safe to call with
    metadata already in hand from resolve_scene_cache_lookup()."""
    try:
        namespace = asset_metadata.get(_METADATA_NAMESPACE) or {}
        cached = namespace.get(_METADATA_KEY) or {}
        if cached.get("version") != SCENE_DETECTOR_CACHE_VERSION:
            return None
        for entry in cached.get("entries") or []:
            if _entry_matches(
                entry,
                fingerprint=fingerprint,
                threshold=threshold,
                start_us=start_us,
                end_us=end_us,
            ):
                return _validate_boundaries(entry.get("boundaries_us"), start_us, end_us)
        return None
    except Exception:
        return None


def write_cached_boundaries(
    asset_id: str,
    *,
    fingerprint: str,
    threshold: float,
    start_us: int,
    end_us: int,
    boundaries: list[SceneBoundary],
) -> bool:
    """Best-effort persistence of a successful (possibly empty) detection
    result. Opens its own session and re-reads the asset immediately before
    writing - right after detection finishes, not a copy held across the
    ~41s ffmpeg call - so the read-modify-write race window is the size of
    this one commit, not the whole detection run (see the Phase 4.2D design
    gate's concurrency note: this repo has no optimistic-locking primitive
    for JSON columns, so a wider window would make lost updates far more
    likely). Never raises - cache writes are best-effort and must never
    become a request dependency; returns False on any DB failure."""
    try:
        with Session(engine) as session:
            asset = session.get(EditorAsset, asset_id)
            if not asset:
                return False

            metadata = dict(asset.asset_metadata or {})
            namespace = dict(metadata.get(_METADATA_NAMESPACE) or {})
            cached = namespace.get(_METADATA_KEY) or {}
            entries = (
                list(cached.get("entries") or [])
                if cached.get("version") == SCENE_DETECTOR_CACHE_VERSION
                else []
            )

            entries = [
                e
                for e in entries
                if not _entry_matches(
                    e,
                    fingerprint=fingerprint,
                    threshold=threshold,
                    start_us=start_us,
                    end_us=end_us,
                )
            ]
            entries.append(
                {
                    "fingerprint": fingerprint,
                    "threshold": threshold,
                    "start_us": start_us,
                    "end_us": end_us,
                    "boundaries_us": [b.timestamp_us for b in boundaries],
                }
            )
            if len(entries) > MAX_SCENE_CACHE_ENTRIES_PER_ASSET:
                entries = entries[-MAX_SCENE_CACHE_ENTRIES_PER_ASSET:]

            namespace[_METADATA_KEY] = {
                "version": SCENE_DETECTOR_CACHE_VERSION,
                "entries": entries,
            }
            metadata[_METADATA_NAMESPACE] = namespace

            # Reassignment (not in-place mutation) is required: asset_metadata
            # is a plain sa.Column(JSON) with no MutableDict tracking, so
            # SQLAlchemy only detects the change via the attribute's own
            # __set__ - mirrors app.editor.auto_import's
            # `project.agent_metadata = {**project.agent_metadata, ...}`
            # pattern for the sibling EditorProject.agent_metadata column.
            asset.asset_metadata = metadata
            session.add(asset)
            session.commit()
            return True
    except Exception:
        return False
