"""Video Intelligence — provider-neutral video analysis for Snipra."""

from __future__ import annotations

from app.video_intelligence.cache import (
    VIDEO_INTELLIGENCE_CONTRACT_VERSION,
    VideoIntelligenceCache,
    cache_result,
    cached_result,
    compute_cache_key,
    get_cache,
)
from app.video_intelligence.chunking import (
    AnalysisChunk,
    AnalysisStrategy,
    SceneBoundary,
    VideoAnalysisPlan,
    VideoChunk,
    chunk_video_range,
    create_chunk_plan,
    get_chunk_info,
    plan_analysis,
    refine_plan_with_scenes,
    validate_chunk_plan,
)
from app.video_intelligence.models import (
    VideoAnalysisConfig,
    VideoAnswer,
    VideoEventCandidate,
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceMode,
    VideoIntelligenceRequest,
    VideoIntelligenceResult,
    VideoObjectEvidence,
    VideoObservation,
    VideoProcessingMetadata,
    VideoScene,
)
from app.video_intelligence.providers import (
    ExistingSnipraVLMProvider,
    ProviderCapabilities,
    VideoIntelligenceProvider,
)
from app.video_intelligence.registry import (
    VIAProviderRegistry,
    get_provider_for_mode,
    get_registry,
    list_providers_for_mode,
)
from app.video_intelligence.scene_detection import (
    DEFAULT_SCENE_CHANGE_THRESHOLD,
    MAX_SCENE_BOUNDARIES,
    detect_scene_boundaries,
)
from app.video_intelligence.service import VideoIntelligenceService

__all__ = [
    # Models
    "VIDEO_INTELLIGENCE_CONTRACT_VERSION",
    "VideoIntelligenceRequest",
    "VideoIntelligenceResult",
    "VideoIntelligenceMode",
    "VideoAnalysisConfig",
    "VideoIntelligenceError",
    "VideoIntelligenceErrorCode",
    "VideoIntelligenceError",
    "VideoObservation",
    "VideoAnswer",
    "VideoEventCandidate",
    "VideoScene",
    "VideoObjectEvidence",
    "VideoProcessingMetadata",
    "VideoIntelligenceResult",
    "VideoAnswer",
    "VideoEventCandidate",
    "VideoScene",
    "VideoObjectEvidence",
    "VideoProcessingMetadata",
    # Service
    "VideoIntelligenceService",
    # Registry
    "VIAProviderRegistry",
    "get_registry",
    "get_provider_for_mode",
    "list_providers_for_mode",
    # Providers
    "VideoIntelligenceProvider",
    "ProviderCapabilities",
    "ExistingSnipraVLMProvider",
    # Chunking
    "VideoChunk",
    "create_chunk_plan",
    "chunk_video_range",
    "validate_chunk_plan",
    "get_chunk_info",
    # Adaptive Analysis Planner (Phase 4.1 Slice 1 - unwired)
    "AnalysisStrategy",
    "AnalysisChunk",
    "VideoAnalysisPlan",
    "plan_analysis",
    # Scene-aware plan refinement (Phase 4.2a - pure, unwired)
    "SceneBoundary",
    "refine_plan_with_scenes",
    # Scene boundary detector (Phase 4.2b - I/O, unwired)
    "DEFAULT_SCENE_CHANGE_THRESHOLD",
    "MAX_SCENE_BOUNDARIES",
    "detect_scene_boundaries",
    # Cache
    "VIDEO_INTELLIGENCE_CONTRACT_VERSION",
    "compute_cache_key",
    "VideoIntelligenceCache",
    "get_cache",
    "cached_result",
    "cache_result",
    # Chunking (duplicate removed in actual use)
    "VideoChunk",
    "create_chunk_plan",
    "chunk_video_range",
    "validate_chunk_plan",
    "get_chunk_info",
]

__version__ = "1.0.0"