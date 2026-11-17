"""Small offline baseline and asynchronous adapter contract."""

from __future__ import annotations

from typing import Protocol

from app.decision.models import DecisionRequest, DecisionResult, DecisionTask


class DecisionEngine(Protocol):
    # Async methods MUST be nonblocking and cancellation-cooperative.
    # A future network/local adapter owns its I/O/executor and resource limits.
    def id(self) -> str: ...
    async def available(self) -> bool: ...
    async def classify(self, request: DecisionRequest) -> DecisionResult: ...


class BatchDecisionEngine(DecisionEngine, Protocol):
    async def batch_classify(self, requests: list[DecisionRequest]) -> list[DecisionResult]: ...


class RulesDecisionEngine:
    def id(self) -> str:
        return "rules"

    async def available(self) -> bool:
        return True

    async def classify(self, request: DecisionRequest) -> DecisionResult:
        score = request.evidence.existing_score
        label = None
        if score is not None:
            if request.task == DecisionTask.clip_quality:
                label = "excellent" if score >= 0.8 else "adequate" if score >= 0.5 else "review"
            elif request.task == DecisionTask.engagement_candidate:
                label = "candidate" if score >= 0.8 else "review"
            elif request.task == DecisionTask.recommended_action:
                label = "keep" if score >= 0.65 else "review"
        # Semantic tasks abstain without an explicit upstream label; never infer
        # a hook/content type from loudness, title or a candidate's proposed hook.
        if request.task in {DecisionTask.content_type, DecisionTask.hook_detection}:
            label = request.evidence.signals.get(request.task.value)
        if label not in request.options:
            return DecisionResult(
                engine=self.id(),
                task=request.task,
                status="unavailable",
                evidence_refs=request.evidence.evidence_refs,
            )
        return DecisionResult(
            engine=self.id(),
            task=request.task,
            status="success",
            classification=label,
            probabilities={option: float(option == label) for option in request.options},
            mutually_exclusive=request.mutually_exclusive,
            # Certainty of rule execution, NOT calibrated predictive confidence.
            confidence=1.0,
            evidence_refs=request.evidence.evidence_refs,
            recommended_action=label if request.task == DecisionTask.recommended_action else None,
        )

    async def batch_classify(self, requests: list[DecisionRequest]) -> list[DecisionResult]:
        return [await self.classify(request) for request in requests]
