"""Short Orchestrator (SNIPRA v1.5J).

Orchestrates Create N Shorts / Montages: runs C-G pipeline once, builds enhanced evidence,
invokes Director for editorial decisions, produces ShortBatchPlan / MontageBatchPlan.

Pure orchestration: no C-G logic, no rendering, no publishing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session

from app.core.sanitize import sanitize_error_text
from app.editor import agent, service, transcript
from app.editor import events as events_module
from app.editor.director_bridge import (
    build_director_evidence,
)
from app.editor.short_models import (
    DirectorSelectionError,
    MontageBatchPlan,
    MontageBatchRequest,
    MontageBatchStatus,
    MontagePlan,
    MontageSelectionError,
    ShortBatchPlan,
    ShortBatchRequest,
    ShortBatchStatus,
    ShortPlan,
    SmartCollectionPool,
    build_montage_plans_from_director_selection,
    build_short_plans_from_director_selection,
    build_smart_collections,
    new_id,
    validate_montage_batch_request,
    validate_short_batch_request,
)
from app.editor.via_evidence import (
    NormalizedVideoEvidence,
    annotate_continuity,
    deduplicate_evidence,
    normalize_via_result,
)
from app.pipeline import (
    event_clustering,
    highlight_boundaries,
    highlight_scoring,
    topic_timeline,
)
from app.pipeline import (
    event_pool as event_pool_module,
)
from app.pipeline.event_clustering import ClusteredEventPool
from app.pipeline.event_pool import EventPool
from app.providers.base import ClipCandidate
from app.video_intelligence.models import VideoIntelligenceError, VideoIntelligenceResult


def _legacy_events_to_candidates(events: list[dict[str, Any]]) -> list[ClipCandidate]:
    """Adapt the read-only legacy event bridge to the canonical C scorer input."""
    return [
        ClipCandidate.model_validate(
            {
                "start": event.get("start_us", 0) / 1_000_000,
                "end": event.get("end_us", 0) / 1_000_000,
                "title": event.get("title", ""),
                "reason": event.get("reason", ""),
                "score": event.get("legacy_score", 0.5),
                "source_ranges": event.get("source_ranges") or [],
                "score_breakdown": event.get("evidence") or [],
            }
        )
        for event in events
    ]


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for the Short Orchestrator."""
    max_requested_count: int = 50  # Safety ceiling
    min_event_duration: float = 0.5  # seconds
    max_event_duration: float = 300.0  # 5 minutes


@dataclass(frozen=True)
class SourceContext:
    """Authoritative source context resolved once per batch."""
    project: Any
    source_job_id: str
    source_asset_id: str
    asset_id: str
    video_duration_us: int
    video_duration_seconds: float
    transcript_segments: list[dict]
    normalized_via_evidence: list[NormalizedVideoEvidence]
    via_available: bool
    via_degradation_reason: str | None


class ShortOrchestrator:
    """Orchestrates Create N Shorts: C-G pipeline + Director decisions.

    Single entry point: create_shorts(request) -> ShortBatchPlan
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig()

    async def create_shorts(
        self,
        session: Session,
        project_id: str,
        request: Any,
        *,
        intent: str = "highlight",
        complete_fn = None,
        via_evidence: list[dict] | None = None,
    ) -> ShortBatchPlan | MontageBatchPlan:
        """Main entry point: Create N Shorts or Montages from a source job/asset.

        Args:
            session: Database session
            project_id: Editor project ID
            request: ShortBatchRequest or MontageBatchRequest or dict with source_job_id/asset_id, requested_count, intent
            intent: "highlight" or "montage"
            complete_fn: Optional Director completion function for testing
            via_evidence: Optional pre-fetched VIA evidence (dicts from persisted storage)

        Returns:
            ShortBatchPlan (for highlight) or MontageBatchPlan (for montage) with full grounding

        Raises:
            ValueError: Invalid request
            InsufficientEventsError: Not enough valid events for requested count
            DirectorSelectionError: Director selected unknown/ungrounded events
            MontageSelectionError: Director's montage selection invalid
            BridgeValidationError: Enhanced evidence validation failed
        """
        # Parse and validate request based on intent
        if intent == "highlight":
            if isinstance(request, dict):
                request = ShortBatchRequest(**request)
            validate_short_batch_request(request)
        elif intent == "montage":
            if isinstance(request, dict):
                request = MontageBatchRequest(**request)
            validate_montage_batch_request(request)
        else:
            raise ValueError(f"Unsupported intent: {intent}")

        if request.requested_count > self.config.max_requested_count:
            raise ValueError(f"requested_count exceeds maximum: {self.config.max_requested_count}")

        # Resolve project and source
        project = service.project_or_404(session, project_id)
        source_job_id = request.source_job_id or project.source_job_id
        if not source_job_id:
            raise ValueError("Project has no source_job_id and none provided in request")

        # Step 0: Resolve authoritative source context ONCE
        source_ctx = self._resolve_source_context(session, project, source_job_id, request.asset_id, via_evidence)

        # Step 1: Run C-G pipeline ONCE per batch with real inputs
        cg_outputs = await self._run_cg_pipeline(
            session=session,
            source_ctx=source_ctx,
            intent=intent,
        )

        # Step 2: Short-circuit empty pool BEFORE Director
        if not cg_outputs["event_pool"].items:
            return self._empty_batch_plan(
                project_id=project_id,
                asset_id=source_ctx.asset_id,
                requested=request.requested_count,
                intent=intent,
            )

        # Step 3: Build enhanced evidence via 1.5H bridge
        enhanced_evidence = self._build_enhanced_evidence(
            event_pool=cg_outputs["event_pool"],
            clustered_pool=cg_outputs["clustered_pool"],
        )

        # Step 4: Build smart collections (1.5J)
        smart_collections = build_smart_collections(enhanced_evidence)

        # Step 5: Build evidence package for Director (includes smart collections)
        evidence = self._build_director_evidence(
            session=session,
            project_id=project_id,
            intent=intent,
            cg_outputs=cg_outputs,
            enhanced_evidence=enhanced_evidence,
            smart_collections=smart_collections,
            source_ctx=source_ctx,
        )

        # Step 6: Invoke Director for editorial decisions
        director_decision = await self._invoke_director(
            session=session,
            project_id=project_id,
            intent=intent,
            request=f"Create {request.requested_count} {'shorts' if intent == 'highlight' else 'montages'} from the best moments",
            evidence=evidence,
            complete_fn=complete_fn,
            source_ctx=source_ctx,
        )

        # Step 7: Validate and build plans from Director's selection
        if intent == "highlight":
            plans = self._build_short_plans(
                director_decision=director_decision,
                enhanced_evidence=enhanced_evidence,
                intent=intent,
                requested_count=request.requested_count,
            )
            status = self._determine_batch_status(
                requested=request.requested_count,
                selected=len(plans),
            )

            batch_plan = ShortBatchPlan(
                batch_id=f"batch_{new_id()}",
                project_id=project_id,
                asset_id=cg_outputs["event_pool"].asset_id,
                requested_count=request.requested_count,
                selected_count=len(plans),
                status=status,
                shorts=tuple(plans),
                enhanced_evidence=self._serialize_enhanced_evidence(enhanced_evidence),
                director_decision={
                    "selected_event_ids": [p.event_id for p in plans],
                    "intent": intent,
                    "requested_count": request.requested_count,
                },
                created_at=datetime.now(UTC).isoformat(),
            )
            return batch_plan

        else:  # montage
            montages = self._build_montage_plans(
                director_decision=director_decision,
                enhanced_evidence=enhanced_evidence,
                smart_collections=smart_collections,
                requested_count=request.requested_count,
            )
            status = self._determine_montage_batch_status(
                requested=request.requested_count,
                selected=len(montages),
            )

            batch_plan = MontageBatchPlan(
                batch_id=f"batch_{new_id()}",
                project_id=project_id,
                asset_id=cg_outputs["event_pool"].asset_id,
                requested_count=request.requested_count,
                selected_count=len(montages),
                status=status,
                montages=tuple(montages),
                smart_collections=smart_collections.to_dict(),
                enhanced_evidence=self._serialize_enhanced_evidence(enhanced_evidence),
                director_decision={
                    "montages": [
                        {
                            "montage_id": m.montage_id,
                            "segments": [
                                {"event_id": seg.event_id, "order_index": seg.order_index}
                                for seg in m.segments
                            ],
                            "director_reason": m.director_reason,
                        }
                        for m in montages
                    ],
                    "intent": intent,
                    "requested_count": request.requested_count,
                },
                created_at=datetime.now(UTC).isoformat(),
            )
            return batch_plan

    def _resolve_source_context(
        self,
        session: Session,
        project: Any,
        source_job_id: str,
        asset_id: str | None,
        via_evidence: list[dict] | None = None,
    ) -> SourceContext:
        """Resolve authoritative source context once per batch.

        Returns:
            SourceContext with all real source data (asset, duration, transcript, VIA).

        Raises:
            ValueError: If source asset cannot be determined or duration is invalid.
        """
        # Get project assets to find the source asset
        assets = service.project_assets(session, project.id)
        if not assets:
            raise ValueError("SOURCE_ASSET_MISSING: Project has no assets")

        # If asset_id provided in request, use exactly that asset
        if asset_id:
            source_asset = None
            for asset in assets:
                if asset.id == asset_id:
                    source_asset = asset
                    break
            if source_asset is None:
                raise ValueError(f"ASSET_NOT_FOUND: Asset {asset_id!r} not found in project")
            if source_asset.kind != "video":
                raise ValueError(f"INVALID_ASSET_KIND: Asset {asset_id!r} is not a video asset")
        else:
            # No asset_id provided: find the primary video asset
            # Prefer asset linked to source_job_id if possible, else first video asset
            source_asset = None
            video_assets = [a for a in assets if a.kind == "video" and a.duration_us > 0]
            if not video_assets:
                raise ValueError("SOURCE_ASSET_MISSING: No video asset with valid duration found")
            if len(video_assets) > 1:
                # Multiple video assets - ambiguous, fail closed
                raise ValueError("AMBIGUOUS_SOURCE_ASSET: Multiple video assets found; provide asset_id in request")
            source_asset = video_assets[0]

        # Validate duration
        if source_asset.duration_us <= 0:
            raise ValueError("INVALID_ASSET_DURATION: Source asset duration_us must be > 0")

        resolved_asset_id = source_asset.id
        video_duration_us = source_asset.duration_us
        video_duration_seconds = video_duration_us / 1_000_000.0

        # Get FULL transcript (0 to duration_us) - single source of truth
        transcript_segments = transcript.get_range(
            source_job_id,
            0,
            video_duration_us,
        )

        # Resolve VIA evidence
        normalized_via_evidence, via_available, via_degradation_reason = self._resolve_via_evidence(
            via_evidence=via_evidence,
            asset_id=resolved_asset_id,
            video_duration_us=video_duration_us,
        )

        return SourceContext(
            project=project,
            source_job_id=source_job_id,
            source_asset_id=resolved_asset_id,
            asset_id=resolved_asset_id,
            video_duration_us=video_duration_us,
            video_duration_seconds=video_duration_seconds,
            transcript_segments=transcript_segments,
            normalized_via_evidence=normalized_via_evidence,
            via_available=via_available,
            via_degradation_reason=via_degradation_reason,
        )

    def _resolve_via_evidence(
        self,
        via_evidence: list[dict] | None,
        asset_id: str,
        video_duration_us: int,
    ) -> tuple[list[NormalizedVideoEvidence], bool, str | None]:
        """Resolve VIA evidence for the source asset.

        Priority:
        1. Pre-fetched via_evidence (from caller/upstream) - normalize and validate
        2. Explicit degraded path - no VIA available for this asset

        Returns:
            Tuple of (normalized_via_evidence, via_available, degradation_reason)
        """
        if via_evidence:
            # Caller provided pre-fetched VIA evidence (e.g., from persisted enhanced_event_pool)
            normalized = []
            for item in via_evidence:
                try:
                    normalized.append(NormalizedVideoEvidence.model_validate(item))
                except Exception:
                    # Drop invalid items (fail-closed for individual items)
                    continue

            # Filter to this asset_id
            normalized = [v for v in normalized if v.asset_id == asset_id]

            # Apply canonical pipeline: normalize -> deduplicate -> annotate_continuity
            if normalized:
                deduplicated = deduplicate_evidence(normalized)
                # Note: annotate_continuity requires VideoIntelligenceResult for scene cache lookup
                # We don't have the full result here, so we pass None (continuity will abstain safely)
                annotated = annotate_continuity(deduplicated, None)
                return annotated, True, None

            return [], False, "VIA_EVIDENCE_EMPTY_AFTER_FILTER"

        # No pre-fetched VIA - explicit degraded path
        # C-G pipeline is designed to work with optional VIA
        return [], False, "VIA_NOT_AVAILABLE"

    async def _run_cg_pipeline(
        self,
        session: Session,
        source_ctx: SourceContext,
        intent: str,
    ) -> dict[str, Any]:
        """Run C-G pipeline exactly ONCE per batch with real source context.

        Returns dict with all intermediate outputs for transparency.
        """
        # 1.5C: Score candidates with real VIA and transcript
        candidates_result = events_module.events_for_job(session, source_ctx.source_job_id, source_ctx.normalized_via_evidence)
        # events_for_job returns list when no VIA, dict with "events" when VIA provided
        legacy_events = candidates_result.get("events", candidates_result) if isinstance(candidates_result, dict) else candidates_result
        candidates = _legacy_events_to_candidates(legacy_events)
        scored_candidates = highlight_scoring.score_candidates(
            candidates=candidates,
            via_evidence=source_ctx.normalized_via_evidence,
            transcript_segments=source_ctx.transcript_segments,
        )

        # 1.5D: Refine boundaries with real VIA, transcript, and video_duration
        refinements = highlight_boundaries.refine_candidates(
            candidates=[sc.candidate for sc in scored_candidates],
            via_evidence=source_ctx.normalized_via_evidence,
            transcript_segments=source_ctx.transcript_segments,
            video_duration=source_ctx.video_duration_seconds,
        )

        # 1.5E: Build topic timeline with real inputs
        topic_timeline_result = topic_timeline.build_topic_timeline(
            transcript_segments=source_ctx.transcript_segments,
            via_evidence=source_ctx.normalized_via_evidence,
            video_duration=source_ctx.video_duration_seconds,
            asset_id=source_ctx.asset_id,
        )

        # 1.5F: Build event pool
        event_pool = event_pool_module.build_event_pool(
            scored_candidates=scored_candidates,
            refinements=refinements,
            topic_timeline=topic_timeline_result,
        )

        # 1.5G: Cluster events
        clustered_pool = event_clustering.cluster_event_pool(event_pool)

        return {
            "scored_candidates": scored_candidates,
            "refinements": refinements,
            "topic_timeline": topic_timeline_result,
            "event_pool": event_pool,
            "clustered_pool": clustered_pool,
        }

    def _build_enhanced_evidence(
        self,
        event_pool: EventPool,
        clustered_pool: ClusteredEventPool,
    ) -> dict[str, Any]:
        """Build enhanced evidence via 1.5H bridge."""
        return build_director_evidence(event_pool, clustered_pool)

    def _empty_batch_plan(
        self,
        project_id: str,
        asset_id: str,
        requested: int,
        intent: str = "highlight",
    ) -> ShortBatchPlan | MontageBatchPlan:
        """Return a FAILED/NO_VALID_EVENTS batch plan without calling Director."""
        if intent == "highlight":
            return ShortBatchPlan(
                batch_id=f"batch_{new_id()}",
                project_id=project_id,
                asset_id=asset_id,
                requested_count=requested,
                selected_count=0,
                status=ShortBatchStatus.FAILED,
                shorts=(),
                enhanced_evidence={
                    "contract_version": "1.5H",
                    "asset_id": asset_id,
                    "source_event_count": 0,
                    "events": [],
                    "clusters": [],
                    "diverse_order_event_ids": [],
                },
                director_decision={
                    "selected_event_ids": [],
                    "intent": "highlight",
                    "requested_count": requested,
                    "failure_reason": "NO_VALID_EVENTS",
                },
                created_at=datetime.now(UTC).isoformat(),
            )
        else:  # montage
            return MontageBatchPlan(
                batch_id=f"batch_{new_id()}",
                project_id=project_id,
                asset_id=asset_id,
                requested_count=requested,
                selected_count=0,
                status=MontageBatchStatus.FAILED,
                montages=(),
                smart_collections={
                    "asset_id": asset_id,
                    "source_event_count": 0,
                    "collections": [],
                    "ungrouped_event_ids": [],
                },
                enhanced_evidence={
                    "contract_version": "1.5H",
                    "asset_id": asset_id,
                    "source_event_count": 0,
                    "events": [],
                    "clusters": [],
                    "diverse_order_event_ids": [],
                },
                director_decision={
                    "montages": [],
                    "intent": "montage",
                    "requested_count": requested,
                    "failure_reason": "NO_VALID_EVENTS",
                },
                created_at=datetime.now(UTC).isoformat(),
            )

    def _build_director_evidence(
        self,
        session: Session,
        project_id: str,
        intent: str,
        cg_outputs: dict[str, Any],
        enhanced_evidence: dict[str, Any],
        smart_collections: SmartCollectionPool,
        source_ctx: SourceContext,
    ) -> dict[str, Any]:
        """Build evidence package for Director using resolved source context."""
        project = service.project_or_404(session, project_id)
        events_result = events_module.events_for_job(session, project.source_job_id, source_ctx.normalized_via_evidence)
        # events_for_job returns list when no VIA, dict with "events" when VIA provided
        legacy_events = events_result.get("events", events_result) if isinstance(events_result, dict) else events_result
        legacy_pool = events_module.best_event_pool(legacy_events, "highlight")

        # Use the SAME transcript as C-G pipeline (full range from source_ctx)
        transcript_segments = source_ctx.transcript_segments

        # Pass VIA evidence in Director-facing format (already deduplicated + annotated)
        director_via = source_ctx.normalized_via_evidence
        director_via_serialized = [item.model_dump(mode="json") for item in director_via] if director_via else []

        return {
            "best_event_pool": legacy_pool,
            "transcript": transcript_segments,
            "via_evidence": director_via_serialized,
            "via_available": source_ctx.via_available,
            "via_degradation_reason": source_ctx.via_degradation_reason,
            "enhanced_event_pool": enhanced_evidence,
            "smart_collections": smart_collections.to_dict(),
        }

    async def _invoke_director(
        self,
        session: Session,
        project_id: str,
        intent: str,
        request: str,
        evidence: dict[str, Any],
        complete_fn,
        source_ctx: SourceContext,
    ) -> dict[str, Any]:
        """Invoke Director with evidence and get editorial decisions.

        Handles evidence request loop (Director can request more evidence).
        Uses proper async/await for VIA execution.
        """
        if complete_fn is None:
            from app.core.config import settings
            from app.providers.registry import get_detector
            detector = get_detector(settings.llm_provider, settings.llm_model)
            complete_fn = getattr(detector, "_complete", None)

        if not callable(complete_fn):
            raise RuntimeError("Director provider unavailable")

        # Get real base version from project
        project = service.project_or_404(session, source_ctx.project.id)
        base_version = project.current_version
        loop_state = agent.LoopState(base_version=base_version)

        while True:
            if loop_state.round > loop_state.max_rounds:
                raise RuntimeError("AGENT_EVIDENCE_AUGMENTATION_EXHAUSTED")

            # Build Director prompt
            system = agent._system_context(intent)
            user = json.dumps({
                "user_request": request,
                "project": {"project_id": source_ctx.project.id, "current_version": base_version},
                "targeted_evidence": evidence,
                "media_evidence": agent.director.untrusted_evidence(""),
            })

            # Call Director with retry logic
            raw = agent._complete_with_retry(complete_fn, system, user)
            output = agent._parse_director_output(raw)

            if output.type == "evidence_request":
                # Director needs more evidence - run VIA and retry
                evidence_request = output.evidence_request
                if evidence_request.base_version != base_version:
                    raise RuntimeError("STALE_VERSION")

                # No ciclar sin progreso: la misma peticion repetida tras haber
                # aportado via_evidence no aporta informacion nueva.
                if (
                    loop_state.prior_evidence_request is not None
                    and agent._evidence_request_signature(evidence_request)
                    == agent._evidence_request_signature(loop_state.prior_evidence_request)
                ):
                    raise RuntimeError(
                        "AGENT_EVIDENCE_REQUEST_REPEATED: Director repitio la misma "
                        "solicitud de evidencia; no hay progreso"
                    )
                try:
                    # Execute VIA request using proper async/await
                    via_result_raw = await agent._execute_via_request(evidence_request)
                except VideoIntelligenceError as exc:
                    # Evidencia visual no disponible/fallida: error determinista.
                    # Mensaje sanitizado: puede contener texto del SDK/CLI.
                    raise RuntimeError(
                        f"AGENT_VIA_EVIDENCE_UNAVAILABLE: {exc.code.value}: "
                        f"{sanitize_error_text(exc.message)}"
                    ) from exc
                
                # Normalize VIA result using canonical pipeline
                via_result = VideoIntelligenceResult.model_validate(via_result_raw)
                normalized_via = normalize_via_result(via_result)
                deduplicated_via = deduplicate_evidence(normalized_via)
                annotated_via = annotate_continuity(deduplicated_via, via_result)
                
                # Serialize for JSON evidence package
                via_evidence_serialized = [item.model_dump(mode="json") for item in annotated_via]
                
                loop_state.round += 1
                loop_state.prior_evidence_request = evidence_request
                loop_state.prior_via_result = via_result_raw

                # Update evidence with new VIA data (properly normalized)
                evidence["via_evidence"] = via_evidence_serialized
                evidence["via_available"] = True
                evidence["via_degradation_reason"] = None
                continue

            if output.type not in ("editorial_plan", "montage_plan"):
                raise RuntimeError("Director returned invalid output type")

            if output.type == "editorial_plan":
                return {
                    "type": "editorial_plan",
                    "selected_events": output.editorial_plan.selected_events,
                    "excluded_events": output.editorial_plan.excluded_events,
                    "operations": [op.model_dump() for op in output.editorial_plan.operations],
                    "decision_rationale": output.decision_rationale,
                }
            else:  # montage_plan
                return {
                    "type": "montage_plan",
                    "montages": [
                        {
                            "montage_id": m.montage_id,
                            "segments": [
                                {"event_id": seg.event_id, "order_index": seg.order_index}
                                for seg in m.segments
                            ],
                            "director_reason": m.director_reason,
                        }
                        for m in output.montages
                    ],
                    "decision_rationale": output.decision_rationale,
                }

    def _build_short_plans(
        self,
        director_decision: dict[str, Any],
        enhanced_evidence: dict[str, Any],
        intent: str,
        requested_count: int,
    ) -> list[ShortPlan]:
        """Build ShortPlans from Director's selection."""
        selected_event_ids = director_decision.get("selected_events", [])

        # P1-2: Empty pool fail closed guard (should be caught earlier but defense in depth)
        if not selected_event_ids:
            return []

        # P1-3: Duplicate Director selections validation
        if len(selected_event_ids) != len(set(selected_event_ids)):
            raise DirectorSelectionError(
                "Director returned duplicate event IDs in selection"
            )

        # P1-4: Director selections must not exceed requested count
        if len(selected_event_ids) > requested_count:
            raise DirectorSelectionError(
                f"Director selected {len(selected_event_ids)} events, exceeding requested_count={requested_count}"
            )

        # Get Director's rationale if available
        if "decision_rationale" in director_decision:
            # Parse rationale for per-event reasons if structured
            pass

        shorts = build_short_plans_from_director_selection(
            director_selected_event_ids=selected_event_ids,
            enhanced_evidence=enhanced_evidence,
            director_reasons={},
            editorial_intent=intent,
        )

        return shorts

    def _build_montage_plans(
        self,
        director_decision: dict[str, Any],
        enhanced_evidence: dict[str, Any],
        smart_collections: SmartCollectionPool,
        requested_count: int,
    ) -> list[MontagePlan]:
        """Build MontagePlans from Director's selection."""
        montages_data = director_decision.get("montages", [])

        if not montages_data:
            return []

        # P1-4: Director montage selections must not exceed requested count
        if len(montages_data) > requested_count:
            raise MontageSelectionError(
                f"Director selected {len(montages_data)} montages, exceeding requested_count={requested_count}"
            )

        montages = build_montage_plans_from_director_selection(
            director_montages=montages_data,
            enhanced_evidence=enhanced_evidence,
            smart_collections=smart_collections,
        )

        return montages

    def _determine_batch_status(self, requested: int, selected: int) -> ShortBatchStatus:
        """Determine batch status from counts."""
        if selected == 0:
            return ShortBatchStatus.FAILED
        if selected < requested:
            return ShortBatchStatus.PARTIAL
        return ShortBatchStatus.PLANNED

    def _determine_montage_batch_status(self, requested: int, selected: int) -> MontageBatchStatus:
        """Determine montage batch status from counts."""
        if selected == 0:
            return MontageBatchStatus.FAILED
        if selected < requested:
            return MontageBatchStatus.PARTIAL
        return MontageBatchStatus.PLANNED

    def _serialize_enhanced_evidence(self, enhanced_evidence: dict[str, Any]) -> dict[str, Any]:
        """Serialize enhanced evidence for storage in batch plan."""
        # Already JSON-serializable from bridge
        return enhanced_evidence


async def create_shorts(
    session: Session,
    project_id: str,
    request,
    *,
    intent: str = "highlight",
    complete_fn = None,
    via_evidence = None,
) -> ShortBatchPlan | MontageBatchPlan:
    """Convenience function to create N shorts or montages."""
    orchestrator = ShortOrchestrator()
    return await orchestrator.create_shorts(
        session=session,
        project_id=project_id,
        request=request,
        intent=intent,
        complete_fn=complete_fn,
        via_evidence=via_evidence,
    )
