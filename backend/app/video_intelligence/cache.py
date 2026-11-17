"""Video Intelligence deterministic cache."""

from __future__ import annotations

import hashlib

from app.video_intelligence.models import VideoAnalysisConfig, VideoIntelligenceRequest

VIDEO_INTELLIGENCE_CONTRACT_VERSION = "via_v2"


def compute_cache_key(
    request: VideoIntelligenceRequest,
    contract_version: str = "via_v2",
    provider: str | None = None,
    model: str | None = None,
    frame_set_fingerprint: str | None = None,
    asset_fingerprint: str | None = None,
) -> str:
    """
    Compute a deterministic cache key for a video intelligence request.

    Includes all parameters that affect the result:
    - Asset fingerprint (not just ID, because content could change)
    - Temporal range
    - Mode and question/objective
    - Analysis configuration
    - Contract version
    - Provider and model (actual visual provider)
    - Frame set fingerprint (for frame-level cache isolation)
    """

    # Build the canonical key components
    # Use asset_fingerprint if provided for true content-based caching, otherwise fall back to asset_id
    asset_identity = asset_fingerprint if asset_fingerprint else str(request.asset_id)
    key_parts = [
        asset_identity,
        str(request.start_us),
        str(request.end_us),
        request.mode,
        request.question or "",
        request.objective or "",
        request.scenario or "",
        ",".join(sorted(request.events_of_interest)),
        ",".join(sorted(request.objects_of_interest)),
    ]

    # Add analysis config if present
    if request.analysis_config:
        config = request.analysis_config
        config_parts = [
            str(config.chunk_duration_s),
            str(config.overlap_s),
            config.sampling_strategy,
            str(config.frames_per_second),
            str(config.max_frames_per_chunk),
            str(config.scene_detection).lower(),
            str(config.dense_captioning).lower(),
        ]
        key_parts.append(",".join(config_parts))

    # Add provider and model for cache isolation
    key_parts.append(provider or "unknown")
    key_parts.append(model or "unknown")

    # Add frame set fingerprint for frame-level cache isolation
    key_parts.append(frame_set_fingerprint or "no-frames")

    # Add contract version
    key_parts.append(contract_version)

    # Create deterministic hash
    canonical = "|".join(key_parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_cache_key_from_parts(
    asset_fingerprint: str,
    start_us: int,
    end_us: int,
    mode: str,
    question: str,
    objective: str,
    scenario: str,
    events_of_interest: list[str],
    objects_of_interest: list[str],
    config: VideoAnalysisConfig | None = None,
    contract_version: str = "via_v2",
    provider: str | None = None,
    model: str | None = None,
    frame_set_fingerprint: str | None = None,
) -> str:
    """Compute cache key from individual components."""

    key_parts = [
        asset_fingerprint,
        str(start_us),
        str(end_us),
        mode,
        question or "",
        objective or "",
        scenario or "",
        ",".join(sorted(events_of_interest)),
        ",".join(sorted(objects_of_interest)),
    ]

    if config:
        config_parts = [
            str(config.chunk_duration_s),
            str(config.overlap_s),
            config.sampling_strategy,
            str(config.frames_per_second),
            str(config.max_frames_per_chunk),
            str(config.scene_detection).lower(),
            str(config.dense_captioning).lower(),
        ]
        key_parts.append(",".join(config_parts))

    # Add provider and model for cache isolation
    key_parts.append(provider or "unknown")
    key_parts.append(model or "unknown")

    # Add frame set fingerprint for frame-level cache isolation
    key_parts.append(frame_set_fingerprint or "no-frames")

    key_parts.append(contract_version)

    canonical = "|".join(key_parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VideoIntelligenceCache:
    """Simple in-memory cache for Video Intelligence results."""

    def __init__(self, max_size: int = 100):
        self._cache: dict[str, dict] = {}
        self._max_size = max_size
        self._access_order: list[str] = []

    def get(self, key: str) -> dict | None:
        if key in self._cache:
            # Move to end (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Evict LRU
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache


# Global cache instance
_cache = VideoIntelligenceCache()


def get_cache() -> VideoIntelligenceCache:
    return _cache


def cached_result(key: str) -> dict | None:
    return _cache.get(key)


def cache_result(key: str, value: dict) -> None:
    _cache.set(key, value)