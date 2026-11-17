"""Versioned decision contracts. Existing scores are not model probabilities."""

from __future__ import annotations

from enum import StrEnum
from math import isclose, isfinite
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.providers.base import SourceRange

Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PROBABILITY_SUM_TOLERANCE = 1e-6
RecommendedAction = Literal["keep", "trim", "merge", "discard", "review"]
CONTENT_TYPES = (
    "hook",
    "context",
    "explanation",
    "demonstration",
    "reaction",
    "punchline",
    "call_to_action",
    "ending",
    "filler",
)


class DecisionTask(StrEnum):
    clip_quality = "clip_quality"
    content_type = "content_type"
    hook_detection = "hook_detection"
    engagement_candidate = "engagement_candidate"
    recommended_action = "recommended_action"


DEFAULT_OPTIONS: dict[DecisionTask, tuple[str, ...]] = {
    DecisionTask.clip_quality: ("excellent", "adequate", "review"),
    DecisionTask.content_type: CONTENT_TYPES,
    DecisionTask.hook_detection: ("hook", "not_hook"),
    DecisionTask.engagement_candidate: ("candidate", "review"),
    DecisionTask.recommended_action: ("keep", "trim", "merge", "discard", "review"),
}


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def validate_finite_tree(value: Any) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("non-finite evidence/metadata")
    if isinstance(value, dict):
        for item in value.values():
            validate_finite_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_finite_tree(item)


class AudioEvidence(Contract):
    energy: Finite | None = None  # RMS dB, NOT a normalized probability
    silence_ratio: Probability | None = None
    speech_ratio: Probability | None = None  # transcript coverage, not active audio


class VisualEvidence(Contract):
    scene_changes: int | None = Field(default=None, ge=0)
    motion_intensity: Nonnegative | None = None
    face_presence: Probability | None = None


class SpeechEvidence(Contract):
    transcript: str | None = None
    words_per_second: Nonnegative | None = None
    sentiment: str | None = None


class ClipEvidence(Contract):
    schema_version: Literal["1"] = "1"
    start_ms: int = Field(ge=0, strict=True)
    end_ms: int = Field(gt=0, strict=True)
    duration_ms: int = Field(gt=0, strict=True)
    # Existing Snipra candidate score uses 0..1; UI may display it as 0..100.
    existing_score: Probability | None = None
    source_ranges: list[SourceRange] = Field(default_factory=list)
    audio: AudioEvidence = Field(default_factory=AudioEvidence)
    visual: VisualEvidence = Field(default_factory=VisualEvidence)
    speech: SpeechEvidence = Field(default_factory=SpeechEvidence)
    signals: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_time(self) -> ClipEvidence:
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must exceed start_ms")
        if self.source_ranges:
            for span in self.source_ranges:
                if not isfinite(span.start) or not isfinite(span.end):
                    raise ValueError("source timestamps must be finite")
                if (
                    not self.start_ms
                    <= round(span.start * 1000)
                    < (round(span.end * 1000))
                    <= self.end_ms
                ):
                    raise ValueError("source range outside evidence bounds")
            expected = round(sum(span.duration for span in self.source_ranges) * 1000)
        else:
            expected = self.end_ms - self.start_ms
        if self.duration_ms != expected:
            raise ValueError("duration_ms does not match source time")
        # Also rejects non-finite values nested in extensible signals.
        validate_finite_tree(self.signals)
        return self


class DecisionRequest(Contract):
    task: DecisionTask
    video_id: str = Field(min_length=1)
    segment_id: str | None = None
    evidence: ClipEvidence
    options: list[str] = Field(min_length=1)
    mutually_exclusive: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_options(self) -> DecisionRequest:
        validate_finite_tree(self.metadata)
        if any(not option.strip() for option in self.options):
            raise ValueError("options must not contain empty labels")
        if len(set(self.options)) != len(self.options):
            raise ValueError("duplicate options")
        if self.task == DecisionTask.recommended_action and any(
            option not in DEFAULT_OPTIONS[self.task] for option in self.options
        ):
            raise ValueError("unknown recommended action")
        return self


class DecisionResult(Contract):
    schema_version: Literal["1"] = "1"
    engine: str = Field(min_length=1)
    model: str | None = None
    task: DecisionTask
    classification: str | None = None
    probabilities: dict[str, Probability] = Field(default_factory=dict)
    mutually_exclusive: bool = True
    confidence: Probability | None = None
    recommended_action: RecommendedAction | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    duration_ms: Nonnegative = 0
    status: Literal["success", "disabled", "unavailable", "timeout", "error"]

    @model_validator(mode="after")
    def valid_distribution(self) -> DecisionResult:
        if self.status == "success":
            if not self.probabilities:
                raise ValueError("successful classification requires probabilities")
            if self.mutually_exclusive and not isclose(
                sum(self.probabilities.values()),
                1.0,
                rel_tol=0,
                abs_tol=PROBABILITY_SUM_TOLERANCE,
            ):
                raise ValueError("exclusive probabilities must sum to one")
            if self.classification not in self.probabilities:
                raise ValueError("classification must be a probability option")
            if self.recommended_action is not None and (
                self.task != DecisionTask.recommended_action
                or self.recommended_action != self.classification
            ):
                raise ValueError("action must match recommended_action classification")
        elif (
            self.probabilities
            or self.classification
            or self.recommended_action
            or (self.confidence is not None)
        ):
            raise ValueError("non-success results cannot contain decisions")
        return self
