"""Video Intelligence service — orchestrates requests, providers, chunking, and caching."""

from __future__ import annotations

import time

from app.video_intelligence.cache import cache_result, cached_result, compute_cache_key
from app.video_intelligence.chunking import (
    AnalysisStrategy,
    plan_analysis,
    refine_plan_with_scenes,
)
from app.video_intelligence.models import (
    VideoAnalysisConfig,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceMode,
    VideoIntelligenceRequest,
    VideoIntelligenceResult,
)
from app.video_intelligence.registry import get_registry
from app.video_intelligence.scene_cache import (
    read_cached_boundaries,
    resolve_scene_cache_lookup,
    write_cached_boundaries,
)
from app.video_intelligence.scene_detection import (
    DEFAULT_SCENE_CHANGE_THRESHOLD,
    detect_scene_boundaries_detailed,
)

VIDEO_INTELLIGENCE_CONTRACT_VERSION = "via_v2"


class VideoIntelligenceService:
    """Main service for Video Intelligence operations."""

    def __init__(self):
        self.registry = get_registry()
        self._ensure_providers()

    def _ensure_providers(self) -> None:
        """Ensure providers are registered."""
        # The registry auto-registers ExistingSnipraVLMProvider on init
        pass

    async def ask_video(
        self,
        request: VideoIntelligenceRequest,
        provider_name: str | None = None,
    ) -> VideoIntelligenceResult:
        """Answer a question about a video."""
        return await self._execute_request(request, provider_name="snipra_vlm")

    async def summarize_video(
        self,
        request: VideoIntelligenceRequest,
        provider_name: str | None = None,
    ) -> VideoIntelligenceResult:
        """Summarize a video or video segment."""
        return await self._execute_request(request, provider_name="snipra_vlm")

    async def _execute_request(
        self,
        request: VideoIntelligenceRequest,
        provider_name: str | None = None,
    ) -> VideoIntelligenceResult:
        from app.video_intelligence.models import (
            VideoIntelligenceError,
            VideoIntelligenceErrorCode,
            VideoIntelligenceResult,
        )

        # Validate request
        self._validate_request(request)

        # Select provider
        mode = request.mode
        provider_name = provider_name or "snipra_vlm"

        try:
            provider = self.registry.select(mode, provider_name)
        except ValueError as err:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
                f"No provider available for mode: {request.mode}",
            ) from err

        if request.mode == "ask_video" and not provider.capabilities.ask_video:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.CAPABILITY_UNAVAILABLE,
                f"Provider {provider.name} does not support ask_video",
            )
        if request.mode == "summarize_video" and not provider.capabilities.summarize_video:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.CAPABILITY_UNAVAILABLE,
                f"Provider {provider.name} does not support summarize_video",
            )

        # Plan sampling BEFORE dispatch, outside the try/except below, so a
        # PLAN_INFEASIBLE VideoIntelligenceError fails fast - before any frame
        # extraction or provider VLM call - and propagates with its own error
        # code instead of being wrapped into INVALID_OUTPUT. Every VIA request
        # (ask_video and summarize_video alike) now has an effective plan
        # (Phase 4.1 Slice 3).
        analysis_plan = plan_analysis(request)

        # Get provider's actual visual provider and model for cache key
        # Use the public API to resolve visual identity without making API calls
        visual_provider, visual_model = provider.resolve_visual_identity()

        # Fetch asset fingerprint for true content-based cache isolation
        asset_fingerprint = None
        try:
            from sqlmodel import Session

            from app.core.db import engine
            from app.editor.models import EditorAsset
            with Session(engine) as session:
                asset = session.get(EditorAsset, request.asset_id)
                if asset and asset.fingerprint:
                    asset_fingerprint = asset.fingerprint
        except Exception:
            # Fallback gracefully if asset lookup fails
            pass

        # Phase 4.2C/4.2D - scene-aware composition, COARSE_SCAN only.
        # Best-effort optimization: never required for VIA to function.
        # FULL_SCAN and TARGETED_SCAN are untouched (zero cache lookups, zero
        # detector invocations) - see module docstring / Phase 4.2 design
        # gate. Any resolution/cache/detector failure simply leaves
        # analysis_plan as the base plan_analysis() output. A real
        # cancellation from detect_scene_boundaries_detailed is NOT caught
        # here and propagates.
        if (
            analysis_plan.strategy == AnalysisStrategy.COARSE_SCAN
            and len(analysis_plan.chunks) > 1
        ):
            lookup = resolve_scene_cache_lookup(request.asset_id)
            if lookup is not None:
                boundaries = read_cached_boundaries(
                    lookup.asset_metadata,
                    fingerprint=lookup.fingerprint,
                    threshold=DEFAULT_SCENE_CHANGE_THRESHOLD,
                    start_us=request.start_us,
                    end_us=request.end_us,
                )
                if boundaries is None:
                    # Cache miss (or corrupt/mismatched entry) - detect for
                    # real. Only a genuinely completed run (cacheable=True,
                    # see SceneDetectionResult) is persisted - an
                    # environmental failure must not poison the cache with a
                    # false "no scenes here" that would outlive the outage.
                    detection = detect_scene_boundaries_detailed(
                        lookup.original_path, request.start_us, request.end_us
                    )
                    boundaries = detection.boundaries
                    if detection.cacheable:
                        write_cached_boundaries(
                            lookup.asset_id,
                            fingerprint=lookup.fingerprint,
                            threshold=DEFAULT_SCENE_CHANGE_THRESHOLD,
                            start_us=request.start_us,
                            end_us=request.end_us,
                            boundaries=boundaries,
                        )
                if boundaries:
                    analysis_plan = refine_plan_with_scenes(analysis_plan, boundaries)

        # Compute detailed plan fingerprint for cache isolation
        # Include chunk boundaries, sampling_fps, frames_planned per chunk, strategy
        # This is computed AFTER scene refinement so the key represents the actual plan to be executed
        plan_parts = [analysis_plan.strategy.value]
        for chunk in analysis_plan.chunks:
            plan_parts.append(f"{chunk.index}:{chunk.start_us}:{chunk.end_us}:{chunk.sampling_fps}:{chunk.frames_planned}")
        import hashlib
        plan_canonical = "|".join(plan_parts)
        frame_set_fingerprint = hashlib.sha256(plan_canonical.encode("utf-8")).hexdigest()[:16]

        # Check cache with provider/model/frame/asset isolation
        cache_key = compute_cache_key(
            request,
            contract_version="via_v2",
            provider=visual_provider,
            model=visual_model,
            frame_set_fingerprint=frame_set_fingerprint,
            asset_fingerprint=asset_fingerprint,
        )
        cached = cached_result(cache_key)
        if cached:
            cached["metadata"]["cache_hit"] = True
            return VideoIntelligenceResult.model_validate(cached)

        # Execute request
        start_time = time.time()
        try:
            if request.mode == "ask_video":
                result = await provider.ask_video(request, analysis_plan)
            elif request.mode == "summarize_video":
                result = await provider.summarize_video(request, analysis_plan)
            else:
                raise ValueError(f"Unknown mode: {request.mode}")
        except VideoIntelligenceError:
            # Re-raise VIA errors with their original error code
            raise
        except Exception as e:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.INVALID_OUTPUT,
                f"Provider execution failed: {e}",
            ) from e

        # Add processing metadata
        processing_time_ms = int((time.time() - start_time) * 1000)
        if hasattr(result, "metadata"):
            result.metadata.processing_time_ms = processing_time_ms

        # Cache the result
        cache_key = compute_cache_key(
            request,
            contract_version="via_v2",
            provider=visual_provider,
            model=visual_model,
            frame_set_fingerprint=frame_set_fingerprint,
            asset_fingerprint=asset_fingerprint,
        )
        cache_result(cache_key, result.model_dump())

        return result

    def _validate_request(self, request: VideoIntelligenceRequest) -> None:
        if request.end_us <= request.start_us:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.INVALID_RANGE,
                "end_us must be greater than start_us",
            )

        if request.mode == "ask_video" and not request.question:
            raise VideoIntelligenceError(
                VideoIntelligenceErrorCode.INVALID_RANGE,
                "question is required for ask_video mode",
            )


# Re-export for convenience

__all__ = [
    "VideoIntelligenceService",
    "VideoIntelligenceRequest",
    "VideoIntelligenceResult",
    "VideoIntelligenceMode",
    "VideoAnalysisConfig",
    "VideoIntelligenceError",
    "VideoIntelligenceErrorCode",
]

