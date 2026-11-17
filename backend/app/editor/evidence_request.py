"""Evidence Request contracts for Director ↔ VIA integration."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, model_validator

from app.editor.models import new_id
from app.video_intelligence.models import VideoAnalysisConfig

if TYPE_CHECKING:
    from app.editor.agent import EditorialPlan
    from app.video_intelligence.models import VideoIntelligenceRequest


class EvidenceRequestMode(StrEnum):
    ASK_VIDEO = "ask_video"
    SUMMARIZE_VIDEO = "summarize_video"


class EvidenceGap(StrEnum):
    PEAK_ACTION_UNCLEAR = "peak_action_unclear"
    OUTCOME_UNCERTAIN = "outcome_uncertain"
    SUBJECT_IDENTITY_AMBIGUOUS = "subject_identity_ambiguous"
    CAUSAL_CHAIN_BROKEN = "causal_chain_broken"
    PAYOFF_MISSING = "payoff_missing"
    EVENT_BOUNDARIES_UNCLEAR = "event_boundaries_unclear"
    EVENT_TYPE_UNCLASSIFIED = "event_type_unclassified"
    COHERENCE_INSUFFICIENT = "coherence_insufficient"
    COMPLETENESS_UNKNOWN = "completeness_unknown"
    VISUAL_EVIDENCE_ABSENT = "visual_evidence_absent"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class DirectorOutputType(StrEnum):
    EDITORIAL_PLAN = "editorial_plan"
    EVIDENCE_REQUEST = "evidence_request"
    MONTAGE_PLAN = "montage_plan"


class EvidenceRequest(BaseModel):
    request_id: str = Field(default_factory=new_id)
    base_version: int
    asset_id: str
    start_us: int = Field(ge=0)
    end_us: int = Field(gt=0)
    mode: EvidenceRequestMode
    reason: str
    evidence_gap: EvidenceGap
    question: str | None = None
    objective: str | None = None
    events_of_interest: list[str] = Field(default_factory=list)
    objects_of_interest: list[str] = Field(default_factory=list)
    analysis_config: VideoAnalysisConfig | None = None

    @model_validator(mode="after")
    def validate_range(self) -> EvidenceRequest:
        if self.end_us <= self.start_us:
            raise ValueError("end_us must be greater than start_us")
        return self

    def to_video_intelligence_request(self) -> VideoIntelligenceRequest:
        from app.video_intelligence.models import VideoIntelligenceRequest
        return VideoIntelligenceRequest(
            asset_id=self.asset_id,
            start_us=self.start_us,
            end_us=self.end_us,
            mode=self.mode.value,
            question=self.question,
            objective=self.objective,
            events_of_interest=self.events_of_interest,
            objects_of_interest=self.objects_of_interest,
            analysis_config=self.analysis_config,
        )


class MontageSegmentOutput(BaseModel):
    """A segment in a montage plan from Director."""
    event_id: str
    order_index: int


class MontageOutput(BaseModel):
    """A montage plan from Director."""
    montage_id: str | None = None
    segments: list[MontageSegmentOutput]
    director_reason: str = ""


class MontagePlanOutput(BaseModel):
    type: Literal[DirectorOutputType.MONTAGE_PLAN] = DirectorOutputType.MONTAGE_PLAN
    montages: list[MontageOutput]
    evidence_sufficiency: Literal[EvidenceSufficiency.SUFFICIENT] = EvidenceSufficiency.SUFFICIENT
    decision_rationale: str


class EditorialPlanOutput(BaseModel):
    type: Literal[DirectorOutputType.EDITORIAL_PLAN] = DirectorOutputType.EDITORIAL_PLAN
    editorial_plan: EditorialPlan
    evidence_sufficiency: Literal[EvidenceSufficiency.SUFFICIENT] = EvidenceSufficiency.SUFFICIENT
    decision_rationale: str


class EvidenceRequestOutput(BaseModel):
    type: Literal[DirectorOutputType.EVIDENCE_REQUEST] = DirectorOutputType.EVIDENCE_REQUEST
    evidence_request: EvidenceRequest
    evidence_sufficiency: Literal[EvidenceSufficiency.INSUFFICIENT] = (
        EvidenceSufficiency.INSUFFICIENT
    )
    decision_rationale: str


class DirectorOutput(BaseModel):
    type: DirectorOutputType
    editorial_plan: EditorialPlan | None = None
    evidence_request: EvidenceRequest | None = None
    montages: list[MontageOutput] | None = None
    evidence_sufficiency: EvidenceSufficiency
    decision_rationale: str

    @model_validator(mode="after")
    def validate_discriminated_union(self) -> DirectorOutput:
        if self.type == DirectorOutputType.EDITORIAL_PLAN:
            if self.editorial_plan is None:
                raise ValueError("editorial_plan required when type=editorial_plan")
            if self.evidence_sufficiency != EvidenceSufficiency.SUFFICIENT:
                raise ValueError("evidence_sufficiency must be 'sufficient' for editorial_plan")
        elif self.type == DirectorOutputType.EVIDENCE_REQUEST:
            if self.evidence_request is None:
                raise ValueError("evidence_request required when type=evidence_request")
            if self.evidence_sufficiency != EvidenceSufficiency.INSUFFICIENT:
                raise ValueError("evidence_sufficiency must be 'insufficient' for evidence_request")
        elif self.type == DirectorOutputType.MONTAGE_PLAN:
            if not self.montages:
                raise ValueError("montages required when type=montage_plan")
            if self.evidence_sufficiency != EvidenceSufficiency.SUFFICIENT:
                raise ValueError("evidence_sufficiency must be 'sufficient' for montage_plan")
        return self


class LoopState(BaseModel):
    round: int = 0
    max_rounds: int = 1
    base_version: int
    prior_evidence_request: EvidenceRequest | None = None
    prior_via_result: dict | None = None