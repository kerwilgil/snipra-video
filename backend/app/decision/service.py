"""Bounded optional batch execution; all provider failures become inert results."""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter

from app.core.config import Settings
from app.decision.engines import RulesDecisionEngine
from app.decision.models import DEFAULT_OPTIONS, DecisionRequest, DecisionResult, DecisionTask
from app.decision.registry import resolve_decision_engine

logger = logging.getLogger(__name__)


async def classify_batch(
    requests: list[DecisionRequest],
    config: Settings,
    *,
    job_provider: str | None = None,
    job_model: str | None = None,
) -> list[DecisionResult]:
    started = perf_counter()
    provider = config.decision_engine_provider

    def fallback(status: str) -> list[DecisionResult]:
        return [
            DecisionResult(
                engine=provider,
                task=request.task,
                status=status,
                evidence_refs=request.evidence.evidence_refs,
                duration_ms=(perf_counter() - started) * 1000,
            )
            for request in requests
        ]

    if not config.decision_engine_enabled:
        return fallback("disabled")

    engine = resolve_decision_engine(provider, job_provider=job_provider, job_model=job_model)
    provider = engine.id() if engine is not None else provider

    async def rules_fallback(status):
        # Preserve foundation error statuses for explicitly registered engines;
        # AI/auto resolution gets the offline fallback chain.
        if config.decision_engine_provider == "auto" or provider in {"openai", "claude", "nvidia"}:
            return await RulesDecisionEngine().batch_classify(requests)
        return fallback(status)

    async def execute() -> list[DecisionResult] | None:
        if engine is None or not await engine.available():
            return None
        # Provider cannot mutate caller evidence or the existing candidate.
        inputs = [request.model_copy(deep=True) for request in requests]
        batch = getattr(engine, "batch_classify", None)
        if batch is not None:
            return await batch(inputs)
        return [await engine.classify(request) for request in inputs]

    try:
        results = await asyncio.wait_for(execute(), config.decision_engine_timeout_ms / 1000)
        if results is None:
            return await rules_fallback("unavailable")
        if len(results) != len(requests):
            raise ValueError("provider batch length mismatch")
        validated = []
        for request, raw in zip(requests, results, strict=True):
            result = DecisionResult.model_validate(raw.model_dump())
            if result.task != request.task or result.engine != provider:
                raise ValueError("provider result identity mismatch")
            if result.evidence_refs != request.evidence.evidence_refs:
                raise ValueError("provider altered provenance")
            if hasattr(engine, "model") and result.model != engine.model:
                raise ValueError("provider model mismatch")
            if result.status != "success" and provider in {"openai", "claude", "nvidia"}:
                result = (await rules_fallback("unavailable"))[len(validated)]
            if result.status == "success":
                if set(result.probabilities) != set(request.options):
                    raise ValueError("provider result options mismatch")
                if result.mutually_exclusive != request.mutually_exclusive:
                    raise ValueError("provider exclusivity mismatch")
                if result.confidence is None or (
                    result.confidence < config.decision_engine_min_confidence
                ):
                    result = (await rules_fallback("unavailable"))[len(validated)]
            validated.append(
                result.model_copy(
                    update={
                        "duration_ms": (perf_counter() - started) * 1000,
                    }
                )
            )
        logger.debug(
            "decision engine=%s batch_size=%d duration_ms=%.3f",
            provider,
            len(requests),
            (perf_counter() - started) * 1000,
        )
        return validated
    except TimeoutError:
        return await rules_fallback("timeout")
    except Exception:
        logger.warning("Decision batch failed; existing pipeline continues")
        return await rules_fallback("error")


def evaluate_candidates(
    candidates: list,
    transcript_doc: dict,
    signals_doc: dict,
    *,
    video_id: str,
    config: Settings,
    job_provider: str | None = None,
    job_model: str | None = None,
) -> list[list[dict]]:
    """Worker seam. Disabled = no normalization/provider work and no API additions."""
    if not config.decision_engine_enabled:
        return []
    from app.decision.evidence import normalize_candidate

    output = []
    requests = []
    positions = []
    for index, candidate in enumerate(candidates):
        output.append([])
        try:
            evidence = normalize_candidate(
                candidate,
                transcript_doc,
                signals_doc,
                video_id=video_id,
            )
            for task in DecisionTask:
                requests.append(
                    DecisionRequest(
                        task=task,
                        video_id=video_id,
                        segment_id=str(index),
                        evidence=evidence,
                        options=list(DEFAULT_OPTIONS[task]),
                    )
                )
                positions.append(index)
        except Exception:
            output[index] = [
                DecisionResult(
                    engine=config.decision_engine_provider,
                    task=task,
                    status="error",
                ).model_dump(mode="json")
                for task in DecisionTask
            ]
    results = asyncio.run(classify_batch(
        requests, config, job_provider=job_provider, job_model=job_model,
    ))
    for index, result in zip(positions, results, strict=True):
        output[index].append(result.model_dump(mode="json"))
    return output
