"""Video Intelligence contracts — provider-neutral, versioned, bounded."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Mode(StrEnum):
    ASK_VIDEO = "ask_video"
    SUMMARIZE_VIDEO = "summarize_video"


class EvidenceNature(StrEnum):
    OBSERVATION = "observation"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


class SamplingStrategy(StrEnum):
    FIXED_FPS = "fixed_fps"
    ADAPTIVE_SCENE = "adaptive_scene"
    UNIFORM = "uniform"


class VideoIntelligenceMode(StrEnum):
    ASK_VIDEO = "ask_video"
    SUMMARIZE_VIDEO = "summarize_video"


class VideoIntelligenceProviderName(StrEnum):
    SNIPRA_VLM = "snipra_vlm"


class VideoIntelligenceErrorCode(StrEnum):
    PROVIDER_UNAVAILABLE = "VIA_PROVIDER_UNAVAILABLE"
    PROVIDER_UNKNOWN = "VIA_PROVIDER_UNKNOWN"
    CAPABILITY_UNAVAILABLE = "VIA_CAPABILITY_UNAVAILABLE"
    INVALID_OUTPUT = "VIA_INVALID_OUTPUT"
    ASSET_NOT_FOUND = "VIA_ASSET_NOT_FOUND"
    INVALID_RANGE = "VIA_INVALID_RANGE"
    FRAME_EXTRACTION_FAILED = "VIA_FRAME_EXTRACTION_FAILED"
    TIMEOUT = "VIA_TIMEOUT"
    PARTIAL_RESULT = "VIA_PARTIAL_RESULT"
    PLAN_INFEASIBLE = "VIA_PLAN_INFEASIBLE"
    VISUAL_PROVIDER_NOT_CONFIGURED = "VIA_VISUAL_PROVIDER_NOT_CONFIGURED"
    MISSING_PROVIDER_CREDENTIALS = "VIA_MISSING_PROVIDER_CREDENTIALS"
    VISUAL_PROVIDER_UNSUPPORTED = "VIA_VISUAL_PROVIDER_UNSUPPORTED"


class CoverageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class ConfidenceSource(StrEnum):
    MODEL_EXPLICIT = "model_explicit"
    DERIVED = "derived"
    UNKNOWN = "unknown"


class NatureSource(StrEnum):
    MODEL_EXPLICIT = "model_explicit"
    ADAPTER_DERIVED = "adapter_derived"
    UNKNOWN = "unknown"


class TemporalSource(StrEnum):
    MODEL_GROUNDED = "model_grounded"
    FRAME_MAPPED = "frame_mapped"
    CHUNK_WINDOW = "chunk_window"
    UNKNOWN = "unknown"


class VideoIntelligenceError(Exception):
    def __init__(self, code: VideoIntelligenceErrorCode, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code.value}: {message}")


# Alias for backward compatibility during transition
VideoObservationNature = EvidenceNature


@dataclass(frozen=True)
class ExtractedFrame:
    frame_id: str
    timestamp_us: int
    data: bytes


class EvidenceProvenance(BaseModel):
    provider: str
    model: str | None = None
    frame_ids: list[str] = Field(default_factory=list)
    frame_timestamps_us: list[int] = Field(default_factory=list)
    analysis_window_start_us: int
    analysis_window_end_us: int
    temporal_source: TemporalSource
    confidence_source: ConfidenceSource
    nature_source: NatureSource
    contract_version: str = "via_v2"


class VideoAnalysisConfig(BaseModel):
    chunk_duration_s: int = 30
    overlap_s: int = 2
    sampling_strategy: str = "fixed_fps"
    frames_per_second: float = 1.0
    max_frames_per_chunk: int = 100
    scene_detection: bool = False
    dense_captioning: bool = False

    @field_validator("chunk_duration_s")
    @classmethod
    def validate_chunk_duration(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_duration_s must be positive")
        if v > 600:
            raise ValueError("chunk_duration_s must not exceed 600s")
        return v

    @field_validator("overlap_s")
    @classmethod
    def validate_overlap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("overlap_s must be non-negative")
        return v

    @field_validator("frames_per_second")
    @classmethod
    def validate_fps(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("frames_per_second must be positive")
        if v > 30:
            raise ValueError("frames_per_second must not exceed 30")
        return v

    @field_validator("max_frames_per_chunk")
    @classmethod
    def validate_max_frames(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_frames_per_chunk must be positive")
        if v > 500:
            raise ValueError("max_frames_per_chunk must not exceed 500")
        return v

    @field_validator("overlap_s")
    @classmethod
    def validate_overlap_lt_chunk(cls, v: int, info) -> int:
        if "chunk_duration_s" in info.data and v >= info.data["chunk_duration_s"]:
            raise ValueError("overlap_s must be less than chunk_duration_s")
        return v


class VideoIntelligenceRequest(BaseModel):
    asset_id: str = Field(min_length=1)
    start_us: int = Field(ge=0)
    end_us: int = Field(gt=0)
    mode: Literal["ask_video", "summarize_video"]
    question: str | None = Field(default=None, validate_default=True)
    objective: str | None = Field(default=None, validate_default=True)
    scenario: str | None = None
    events_of_interest: list[str] = Field(default_factory=list)
    objects_of_interest: list[str] = Field(default_factory=list)
    analysis_config: VideoAnalysisConfig | None = None

    @field_validator("end_us")
    @classmethod
    def validate_end_after_start(cls, v: int, info) -> int:
        if "start_us" in info.data and v <= info.data["start_us"]:
            raise ValueError("end_us must be greater than start_us")
        return v

    @field_validator("question")
    @classmethod
    def validate_question_for_ask(cls, v: str | None, info) -> str | None:
        if info.data.get("mode") == "ask_video" and not v:
            raise ValueError("question is required for ask_video mode")
        return v

    @field_validator("objective")
    @classmethod
    def validate_objective_for_summarize(cls, v: str | None, info) -> str | None:
        if info.data.get("mode") == "summarize_video" and not v:
            return "Summarize the video content"
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("ask_video", "summarize_video"):
            raise ValueError("mode must be 'ask_video' or 'summarize_video'")
        return v

    def model_post_init(self, __context) -> None:
        if self.analysis_config is None:
            self.analysis_config = VideoAnalysisConfig()


class VideoObservation(BaseModel):
    asset_id: str
    start_us: int
    end_us: int
    description: str
    nature: EvidenceNature
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    nature_source: NatureSource = NatureSource.UNKNOWN
    source: str
    provider: str
    model: str | None = None
    chunk_id: str | None = None
    provenance: EvidenceProvenance | None = None
    metadata: dict = Field(default_factory=dict)


class VideoAnswer(BaseModel):
    answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    supporting_observation_ids: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


class VideoEventCandidate(BaseModel):
    asset_id: str
    start_us: int
    end_us: int
    type: str
    description: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    nature: EvidenceNature
    nature_source: NatureSource = NatureSource.UNKNOWN
    provider: str
    model: str | None = None
    chunk_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    observed_action: str | None = None
    subjects: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    provenance: EvidenceProvenance | None = None
    metadata: dict = Field(default_factory=dict)


class VideoScene(BaseModel):
    asset_id: str
    start_us: int
    end_us: int
    description: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    nature: EvidenceNature
    nature_source: NatureSource = NatureSource.UNKNOWN
    provider: str
    model: str | None = None
    chunk_id: str | None = None
    subjects: list[str] = Field(default_factory=list)
    scene_type: str | None = None
    continuity: str | None = None
    dominant_objects: list[str] = Field(default_factory=list)
    provenance: EvidenceProvenance | None = None
    metadata: dict = Field(default_factory=dict)


class VideoObjectEvidence(BaseModel):
    asset_id: str
    label: str
    timestamps_us: list[int]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_source: ConfidenceSource = ConfidenceSource.UNKNOWN
    nature: EvidenceNature
    nature_source: NatureSource = NatureSource.UNKNOWN
    provider: str
    model: str | None = None
    first_seen_us: int | None = None
    last_seen_us: int | None = None
    track_id: str | None = None
    interactions: list[str] = Field(default_factory=list)
    provenance: EvidenceProvenance | None = None
    metadata: dict = Field(default_factory=dict)


class VideoProcessingMetadata(BaseModel):
    chunks_processed: int = 0
    chunks_failed: int = 0
    total_frames_analyzed: int = 0
    provider: str
    model: str
    processing_time_ms: int = 0
    partial: bool = False
    failed_chunks: list[dict] = Field(default_factory=list)

    # Coverage metadata (Phase 3.4)
    requested_frames: int | None = Field(default=None, ge=0)
    analyzed_frames: int | None = Field(default=None, ge=0)
    coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage_status: CoverageStatus | None = Field(default=None)

    @classmethod
    def compute_coverage(
        cls, requested: int | None, analyzed: int | None
    ) -> tuple[float | None, CoverageStatus | None]:
        """Compute coverage ratio and status from frame counts.

        Returns (coverage_ratio, coverage_status) or (None, CoverageStatus.UNKNOWN)
        if requested_frames is None, 0, or unknown.

        Status thresholds (deliberate, no partial-vs-insufficient magic cutoff):
        - ratio == 1.0        -> COMPLETE
        - 0.0 < ratio < 1.0   -> PARTIAL (any analysis happened but not all of it)
        - ratio == 0.0        -> INSUFFICIENT (nothing was analyzed)
        """
        if requested is None or requested <= 0:
            return None, CoverageStatus.UNKNOWN

        if analyzed is None:
            analyzed = 0

        # Clamp analyzed to requested max
        analyzed = min(analyzed, requested)
        ratio = analyzed / requested

        # Clamp ratio to [0.0, 1.0]
        ratio = max(0.0, min(1.0, ratio))

        if ratio >= 1.0:
            status = CoverageStatus.COMPLETE
        elif ratio > 0.0:
            status = CoverageStatus.PARTIAL
        else:
            status = CoverageStatus.INSUFFICIENT

        return ratio, status


class VideoIntelligenceResult(BaseModel):
    asset_id: str
    start_us: int
    end_us: int
    mode: str
    question: str | None = None
    objective: str | None = None
    summary: str | None = None
    answer: VideoAnswer | None = None
    observations: list[VideoObservation] = Field(default_factory=list)
    events: list[VideoEventCandidate] = Field(default_factory=list)
    scenes: list[VideoScene] = Field(default_factory=list)
    objects: list[VideoObjectEvidence] = Field(default_factory=list)
    metadata: VideoProcessingMetadata = Field(default_factory=lambda: VideoProcessingMetadata(
        provider="", model=""
    ))
    partial: bool = False
    contract_version: str = "via_v2"

    @classmethod
    def model_json_schema(cls, **kwargs):
        return super().model_json_schema(**kwargs)


VideoIntelligenceResult.model_rebuild()
VideoAnswer = VideoAnswer
VideoEventCandidate = VideoEventCandidate
VideoScene = VideoScene
VideoObjectEvidence = VideoObjectEvidence
VideoIntelligenceResult = VideoIntelligenceResult