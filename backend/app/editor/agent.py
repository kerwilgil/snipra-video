"""Version-safe Editor IA orchestration over the canonical Director and operation engine."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from app.core.config import settings
from app.core.sanitize import sanitize_error_text
from app.editor import capabilities, director, events, service, transcript, verification
from app.editor.director_bridge import BridgeValidationError, validate_director_evidence
from app.editor.evidence_request import (
    DirectorOutput,
    DirectorOutputType,
    EvidenceRequest,
    EvidenceRequestMode,
    EvidenceSufficiency,
    LoopState,
)
from app.editor.models import EditorAgentRun, EditorOperation, new_id
from app.editor.via_evidence import (
    NormalizedVideoEvidence,
    annotate_continuity,
    deduplicate_evidence,
    normalize_via_result,
)
from app.providers.registry import get_detector, get_structured_completion_adapter
from app.video_intelligence.models import (
    VideoIntelligenceError,
    VideoIntelligenceErrorCode,
    VideoIntelligenceResult,
)
from app.video_intelligence.service import VideoIntelligenceService

ALLOWED_MUTATIONS = frozenset(
    {
        "split",
        "trim",
        "move",
        "delete",
        "duplicate",
        "insert",
        "clip_properties",
        "add_caption",
        "edit_caption",
        "add_text",
        "edit_text",
        "set_volume",
        "set_mute",
        "set_fade",
        "set_crop",
        "set_position",
        "set_scale",
        "change_speed",
        "track_properties",
        "undo",
    }
)
DEFAULT_MAX_REWORK_ROUNDS = 1


class PlannedOperation(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class EditorialPlan(BaseModel):
    objective: str
    output_intent: str = "edit"
    target_duration_us: int | None = None
    selected_events: list[str] = Field(default_factory=list)
    excluded_events: list[str] = Field(default_factory=list)
    operations: list[PlannedOperation] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    verification_targets: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    base_version: int | None = None


EditorialPlan.model_rebuild()
DirectorOutput.model_rebuild()


_EDITORIAL_PLAN_SCHEMA = EditorialPlan.model_json_schema()
_DIRECTOR_OUTPUT_SCHEMA = DirectorOutput.model_json_schema()


def _output_contract_for_intent(intent: str | None) -> str:
    """Generate output contract based on editorial intent."""
    sufficient_type = "editorial_plan" if intent != "montage" else "montage_plan"
    sufficient_fields = "editorial_plan populated" if intent != "montage" else "montages populated"
    second_call_type = "editorial_plan" if intent != "montage" else "montage_plan"

    return f"""OUTPUT CONTRACT:
Return exactly one JSON object matching the DirectorOutput JSON Schema below.
No Markdown.
No code fences.
No commentary before or after the JSON.
Use only capabilities reported as available.
Do not invent IDs, timestamps or assets.

DECISION LOGIC:
- If visual evidence is SUFFICIENT to make an editorial decision:
  Return type="{sufficient_type}" with {sufficient_fields}.
  Set evidence_sufficiency="sufficient".
  Explain rationale in decision_rationale.

- If visual evidence is INSUFFICIENT (gap in observations):
  Return type="evidence_request" with evidence_request populated.
  Set evidence_sufficiency="insufficient".
  Specify evidence_gap from the enum.
  Use ask_video for specific questions, summarize_video for range summaries.
  Explain rationale in decision_rationale.

- If targeted_evidence contains "via_evidence" (second call):
  This is Video Intelligence evidence from a prior evidence_request.
  Distinguish observations (nature="observation") from inferences (nature="inference").
  Use confidence and partial/failed_chunks to weigh evidence.
  NEVER treat absence of observation as observation of absence.
  Then return type="{second_call_type}" (evidence must now be sufficient).

DIRECTOROUTPUT JSON SCHEMA:
{json.dumps(_DIRECTOR_OUTPUT_SCHEMA, ensure_ascii=False)}"""


def inspect_project(session: Session, project_id: str) -> dict:
    project = service.project_or_404(session, project_id)
    version = service.current_version(session, project)
    return {
        "project_id": project.id,
        "timeline_version": project.current_version,
        "timeline": version.timeline,
        "assets": [service.serialize_asset(a) for a in service.project_assets(session, project.id)],
        "capabilities": capabilities.public_capabilities(),
        "source_job_id": project.source_job_id,
    }


def _system_context(intent: str | None) -> str:
    context = director.load_director_context(intent)
    workflow_and_refs = "\n\n".join([context["workflow"], *context["references"].values()])
    return f"{workflow_and_refs}\n\n{_output_contract_for_intent(intent)}"


def _evidence_request_signature(request: EvidenceRequest) -> tuple:
    """Firma estable para detectar la misma peticion repetida sin progreso."""
    return (
        request.mode,
        request.asset_id,
        request.start_us,
        request.end_us,
        (getattr(request, "question", "") or "").strip().lower(),
    )


async def _execute_via_request(evidence_request: EvidenceRequest) -> dict:
    via_service = VideoIntelligenceService()
    vi_request = evidence_request.to_video_intelligence_request()
    try:
        if evidence_request.mode == EvidenceRequestMode.ASK_VIDEO:
            result = await via_service.ask_video(vi_request)
        else:
            result = await via_service.summarize_video(vi_request)
        return result.model_dump()
    except VideoIntelligenceError as e:
        if e.code in (
            VideoIntelligenceErrorCode.PROVIDER_UNAVAILABLE,
            VideoIntelligenceErrorCode.CAPABILITY_UNAVAILABLE,
            VideoIntelligenceErrorCode.INVALID_OUTPUT,
            VideoIntelligenceErrorCode.TIMEOUT,
            VideoIntelligenceErrorCode.ASSET_NOT_FOUND,
            VideoIntelligenceErrorCode.INVALID_RANGE,
            VideoIntelligenceErrorCode.FRAME_EXTRACTION_FAILED,
        ):
            raise
        raise


def _extract_json(raw: str) -> str:
    """Extract JSON from raw response. Accepts pure JSON or a single ```json fence."""
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    m = re.search(r"^```json\s*(\{.*?\})\s*```$", stripped, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"^```\s*(\{.*?\})\s*```$", stripped, re.DOTALL)
    if m:
        return m.group(1)
    raise ValueError("NO_VALID_JSON_BLOCK")


def _parse_plan(raw: str) -> EditorialPlan:
    json_str = _extract_json(raw)
    try:
        plan = EditorialPlan.model_validate_json(json_str)
    except Exception as exc:
        raise ValueError("AGENT_PLAN_INVALID") from exc
    unsupported = {
        item.type
        for item in plan.operations
        if not capabilities.CAPABILITIES.get(item.type, {}).get("available")
    } | ({item.type for item in plan.operations} - ALLOWED_MUTATIONS)
    if unsupported:
        raise ValueError(f"CAPABILITY_UNAVAILABLE: {sorted(unsupported)!r}")  # noqa: B904
    return plan


def _parse_director_output(raw: str) -> DirectorOutput:
    json_str = _extract_json(raw)
    try:
        output = DirectorOutput.model_validate_json(json_str)
    except Exception:
        # Backward compatibility: if Director returns old EditorialPlan format
        # (without type/evidence_sufficiency/decision_rationale), wrap it
        try:
            plan = EditorialPlan.model_validate_json(json_str)
        except Exception as exc:
            raise ValueError("AGENT_PLAN_INVALID") from exc
        unsupported = {
            item.type
            for item in plan.operations
            if not capabilities.CAPABILITIES.get(item.type, {}).get("available")
        } | ({item.type for item in plan.operations} - ALLOWED_MUTATIONS)
        if unsupported:
            raise ValueError(f"CAPABILITY_UNAVAILABLE: {sorted(unsupported)!r}")  # noqa: B904
        return DirectorOutput(
            type=DirectorOutputType.EDITORIAL_PLAN,
            editorial_plan=plan,
            evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
            decision_rationale="Legacy EditorialPlan format (backward compat)",
        )
    if output.type == DirectorOutputType.EDITORIAL_PLAN:
        if output.editorial_plan is None:
            raise ValueError("AGENT_PLAN_INVALID: editorial_plan required")
        unsupported = {
            item.type
            for item in output.editorial_plan.operations
            if not capabilities.CAPABILITIES.get(item.type, {}).get("available")
        } | ({item.type for item in output.editorial_plan.operations} - ALLOWED_MUTATIONS)
        if unsupported:
            raise ValueError(f"CAPABILITY_UNAVAILABLE: {sorted(unsupported)!r}")
    elif output.type == DirectorOutputType.EVIDENCE_REQUEST:
        if output.evidence_request is None:
            raise ValueError("AGENT_PLAN_INVALID: evidence_request required")
    elif output.type == DirectorOutputType.MONTAGE_PLAN:
        if not output.montages:
            raise ValueError("AGENT_PLAN_INVALID: montages required")
        # Validate montage segments reference valid event_ids (will be validated in orchestrator)
    return output


_RETRY_INSTRUCTION = (
    "Your previous response did not satisfy the DirectorOutput JSON schema.\n"
    "Return only one valid JSON object matching the schema."
)


def _intent(request: str, requested: str | None) -> str:
    if requested:
        return requested
    lowered = request.lower()
    return "highlight" if "highlight" in lowered else "montage" if "montage" in lowered else "edit"


def _evidence(
    session: Session,
    snapshot: dict,
    intent: str,
    request: str,
    via_evidence: dict | list | None = None,
    enhanced_event_pool: dict | None = None,
) -> dict:
    # Normalize VIA evidence if provided as raw dict
    normalized_via: list = []
    via_result: VideoIntelligenceResult | None = None
    if isinstance(via_evidence, dict):
        # Raw VIA result dict - normalize it
        via_result = VideoIntelligenceResult.model_validate(via_evidence)
        normalized_via = normalize_via_result(via_result)
    elif isinstance(via_evidence, list):
        # Already normalized
        normalized_via = via_evidence

    pool = []
    if intent in {"highlight", "montage"}:
        # events.py's own Event Map matching (best_event_pool/legacy_score/
        # Quality Gate) reads the RAW normalized_via, untouched by the
        # Director-facing dedup+continuity pipeline below - it already runs
        # its own independent deduplicate_evidence() internally (section 24:
        # continuity/dedup-for-Director must never interfere with it).
        ev_result = events.events_for_job(session, snapshot["source_job_id"], normalized_via)
        # events_for_job() returns a plain list of legacy events when there is no VIA
        # evidence to merge yet (the common case on the first loop iteration), and a
        # dict with an "events" key only once _merge_via_evidence() has run.
        event_items = ev_result.get("events", []) if isinstance(ev_result, dict) else ev_result
        pool = events.best_event_pool(event_items, intent)
    wants_transcript = any(
        word in request.lower() for word in ("repet", "texto", "palabra", "frase")
    )
    segments = (
        transcript.get_range(
            snapshot["source_job_id"],
            0,
            min(int(snapshot["timeline"].get("duration_us", 0)), 30_000_000),
        )
        if wants_transcript
        else []
    )
    # Director-facing pipeline (section 3, mandatory order):
    # normalize_via_result() [above] -> deduplicate_evidence() -> annotate_continuity().
    # Continuity must never run over duplicates, so dedup always precedes it.
    director_via = normalized_via
    if director_via:
        director_via = deduplicate_evidence(director_via)
        director_via = annotate_continuity(director_via, via_result)

    evidence = {"best_event_pool": pool, "transcript": segments}
    if director_via:
        # Serialize to plain dicts: plan_request()/_rework_plan() embed `evidence`
        # in a json.dumps() prompt payload, which cannot encode NormalizedVideoEvidence
        # pydantic instances directly.
        evidence["via_evidence"] = [item.model_dump(mode="json") for item in director_via]

    # Enhanced Event Pool (1.5H) - only for highlight/montage intents
    if enhanced_event_pool is not None and intent in {"highlight", "montage"}:
        # Validate the enhanced evidence structure (fail-closed)
        try:
            validated = validate_director_evidence(enhanced_event_pool)
            evidence["enhanced_event_pool"] = validated
        except BridgeValidationError as exc:
            raise ValueError(f"ENHANCED_EVIDENCE_INVALID: {exc}") from exc

    return evidence


def rehydrate_via_evidence(raw_items: list[dict] | None) -> list[dict]:
    """Re-validate persisted VIA evidence against the canonical model before reuse.

    Used by POST /confirm to restore the via_evidence a NEEDS_USER plan was
    actually grounded in, without trusting the persisted dicts blindly and
    without re-calling VIA/Director. Items that fail canonical validation
    (corrupted/incompatible persisted data) are dropped rather than crashing
    confirmation.
    """
    if not raw_items:
        return []
    restored = []
    for item in raw_items:
        try:
            restored.append(NormalizedVideoEvidence.model_validate(item).model_dump(mode="json"))
        except Exception:
            continue
    return restored


def _complete_with_retry(complete: Callable[[str, str], str], system: str, user: str) -> str:
    """Call provider with at most one retry for invalid plan format/schema.

    Validates against the current DirectorOutput contract (_parse_director_output),
    matching what the system prompt (_OUTPUT_CONTRACT) actually asks the model to
    return - the legacy bare-EditorialPlan parser (_parse_plan) rejects every
    well-formed envelope response, including a correct evidence_request.
    """
    raw = complete(system, user)
    try:
        _parse_director_output(raw)
        return raw
    except ValueError as exc:
        if "AGENT_PLAN_INVALID" in str(exc):
            raw2 = complete(system + "\n\n" + _RETRY_INSTRUCTION, user)
            try:
                _parse_director_output(raw2)
                return raw2
            except ValueError:
                pass
        raise


def plan_request(
    session: Session,
    project_id: str,
    request: str,
    *,
    intent: str | None = None,
    complete: Callable[[str, str], str] | None = None,
    enhanced_event_pool: dict | None = None,
) -> tuple[EditorialPlan, dict, dict]:
    snapshot = inspect_project(session, project_id)
    resolved_intent = _intent(request, intent)
    base_version = snapshot["timeline_version"]
    loop_state = LoopState(base_version=base_version)
    via_evidence: dict | None = None

    while True:
        if loop_state.round > loop_state.max_rounds:
            raise RuntimeError("AGENT_EVIDENCE_AUGMENTATION_EXHAUSTED")

        evidence = _evidence(
            session, snapshot, resolved_intent, request, via_evidence, enhanced_event_pool
        )
        user = json.dumps(
            {
                "user_request": request,
                "project": snapshot,
                "targeted_evidence": evidence,
                "media_evidence": director.untrusted_evidence(""),
            }
        )
        if complete is None:
            adapter = get_structured_completion_adapter(
                settings.llm_provider, settings.llm_model
            )
            if adapter and adapter.capabilities.supports_native_json_schema:
                # Use native JSON Schema path
                system = _system_context(resolved_intent)
                result = adapter.complete(
                    system,
                    user,
                    json_schema=DirectorOutput.model_json_schema(),
                )
                raw = result.content
                # Validate structured output
                try:
                    output = _parse_director_output(raw)
                except ValueError as exc:
                    if "AGENT_PLAN_INVALID" in str(exc):
                        # One retry with schema reminder
                        result = adapter.complete(
                            system + "\n\n" + _RETRY_INSTRUCTION,
                            user,
                            json_schema=DirectorOutput.model_json_schema()
                        )
                        raw = result.content
                        output = _parse_director_output(raw)
                    else:
                        raise
            else:
                # Fallback path: use legacy _complete
                detector = get_detector(settings.llm_provider, settings.llm_model)
                complete = getattr(detector, "_complete", None)
                if not callable(complete):
                    raise RuntimeError("AGENT_PROVIDER_UNAVAILABLE")
                system = _system_context(resolved_intent)
                raw = _complete_with_retry(complete, system, user)
                output = _parse_director_output(raw)
        else:
            # Use provided complete function (legacy or test)
            system = _system_context(resolved_intent)
            raw = _complete_with_retry(complete, system, user)
            output = _parse_director_output(raw)

        if output.type == DirectorOutputType.EDITORIAL_PLAN:
            plan = output.editorial_plan
            plan.base_version = base_version
            return plan, snapshot, evidence

        evidence_request = output.evidence_request
        if evidence_request.base_version != base_version:
            raise RuntimeError("STALE_VERSION")

        # No ciclar sin progreso: repetir exactamente la misma peticion tras
        # haber aportado via_evidence no aporta informacion nueva.
        if (
            loop_state.prior_evidence_request is not None
            and _evidence_request_signature(evidence_request)
            == _evidence_request_signature(loop_state.prior_evidence_request)
        ):
            raise RuntimeError(
                "AGENT_EVIDENCE_REQUEST_REPEATED: Director repitio la misma "
                "solicitud de evidencia tras aportar via_evidence; no hay progreso"
            )
        try:
            via_result = asyncio.run(_execute_via_request(evidence_request))
        except VideoIntelligenceError as exc:
            # Evidencia visual no disponible/fallida: fallo determinista con
            # codigo, no un retry ciclico ni un 500 opaco. El mensaje puede
            # contener texto del SDK/CLI: sanitizar antes de exponer.
            raise RuntimeError(
                f"AGENT_VIA_EVIDENCE_UNAVAILABLE: {exc.code.value}: "
                f"{sanitize_error_text(exc.message)}"
            ) from exc
        via_evidence = via_result
        loop_state.round += 1
        loop_state.prior_evidence_request = evidence_request
        loop_state.prior_via_result = via_result


def execute_plan(
    session: Session, project_id: str, plan: EditorialPlan, *, key_prefix: str
) -> dict:
    project = service.project_or_404(session, project_id)
    if plan.base_version is None or project.current_version != plan.base_version:
        keys = [f"{key_prefix}:{index}" for index, _ in enumerate(plan.operations)]
        prior = session.exec(
            select(EditorOperation).where(
                EditorOperation.project_id == project_id, EditorOperation.idempotency_key.in_(keys)
            )
        ).all()
        if keys and len(prior) == len(keys):
            return {
                "version": project.current_version,
                "operations": [
                    {"operation_id": item.id, "version": item.version_number, "idempotent": True}
                    for item in sorted(prior, key=lambda item: item.version_number)
                ],
                "plan": plan.model_dump(),
            }
        raise service.error(409, "STALE_VERSION", "La timeline cambió; reinspecciona y replanifica")
    version, results = plan.base_version, []
    for index, operation in enumerate(plan.operations):
        if operation.type == "undo":
            result = service.undo(session, project_id, generated_by="agent")
            result["operation_id"] = f"undo:{result['version']}"
        else:
            result = service.apply_operation(
                session,
                project_id,
                {
                    "type": operation.type,
                    "parameters": operation.parameters,
                    "base_version": version,
                    "idempotency_key": f"{key_prefix}:{index}",
                    "generated_by": "agent",
                },
            )
        version = result["version"]
        results.append(result)
    return {"version": version, "operations": results, "plan": plan.model_dump()}


def _rework_plan(
    session: Session,
    project_id: str,
    original_request: str,
    failed_plan: EditorialPlan,
    failure: dict,
    evidence: dict,
    complete: Callable[[str, str], str] | None,
) -> EditorialPlan | None:
    """Ask the same Director-backed contract for the smallest safe correction."""
    # Resolve snapshot once, used by both structured and legacy paths
    snapshot = inspect_project(session, project_id)

    if complete is None:
        adapter = get_structured_completion_adapter(
            settings.llm_provider, settings.llm_model
        )
        if adapter and adapter.capabilities.supports_native_json_schema:
            system = _system_context(failed_plan.output_intent)
            user = json.dumps(
                {
                    "user_request": original_request,
                    "rework": True,
                    "failed_plan": failed_plan.model_dump(),
                    "verification": failure,
                    "project": snapshot,
                    "targeted_evidence": evidence,
                }
            )
            result = adapter.complete(
                system, user, json_schema=EditorialPlan.model_json_schema()
            )
            raw = result.content
            try:
                plan = _parse_plan(raw)
            except ValueError:
                # One retry
                result = adapter.complete(
                    system + "\n\n" + _RETRY_INSTRUCTION,
                    user,
                    json_schema=EditorialPlan.model_json_schema()
                )
                raw = result.content
                plan = _parse_plan(raw)
            return plan
        # Fallback to legacy
        complete = getattr(
            get_detector(settings.llm_provider, settings.llm_model), "_complete", None
        )
        if not callable(complete):
            return None
    if not callable(complete):
        return None
    raw = _complete_with_retry(
        complete,
        _system_context(failed_plan.output_intent),
        json.dumps(
            {
                "user_request": original_request,
                "rework": True,
                "failed_plan": failed_plan.model_dump(),
                "verification": failure,
                "project": snapshot,
                "targeted_evidence": evidence,
            }
        ),
    )
    plan = _parse_plan(raw)
    plan.base_version = snapshot["timeline_version"]
    return plan if plan.operations else None


def _check(session: Session, project_id: str, plan: EditorialPlan, evidence: dict) -> dict:
    project = service.project_or_404(session, project_id)
    timeline_data = service.current_version(session, project).timeline
    checked = verification.verify(
        timeline_data, {asset.id for asset in service.project_assets(session, project.id)}
    )
    checked["editorial"] = verification.verify_editorial(
        plan.model_dump(),
        evidence["best_event_pool"],
        evidence["transcript"],
        evidence.get("via_evidence"),
        evidence.get("enhanced_event_pool"),
    )
    checked["status"] = (
        "FAIL" if checked["deterministic"]["status"] == "FAIL" else checked["editorial"]["status"]
    )
    checked["requires_human"] = checked["status"] == "UNCERTAIN"
    return checked


def run_request(
    session: Session,
    project_id: str,
    request: str,
    *,
    intent: str | None = None,
    execute: bool = True,
    key_prefix: str | None = None,
    complete: Callable[[str, str], str] | None = None,
    enhanced_event_pool: dict | None = None,
) -> dict:
    resolved_intent = _intent(request, intent)
    from app.providers.nvidia_catalog import capture_model_usage

    with capture_model_usage() as model_usage:
        plan, snapshot, evidence = plan_request(
            session,
            project_id,
            request,
            intent=intent,
            complete=complete,
            enhanced_event_pool=enhanced_event_pool,
        )
    run = EditorAgentRun(
        project_id=project_id, base_version=snapshot["timeline_version"], status="PLANNING",
        verification={"model_usage": model_usage},
    )
    session.add(run)
    session.commit()
    if not execute:
        run.status = "READY"
        session.add(run)
        session.commit()
        return {"status": "READY", "plan": plan.model_dump(), "run_id": run.id}
    if plan.requires_confirmation:
        run.status = "NEEDS_USER"
        pending_evidence: dict[str, Any] = {
            "needs_transcript": any(
                word in request.lower() for word in ("repet", "texto", "palabra", "frase")
            )
        }
        # Preserve the VIA evidence the Director actually used to reach this plan
        # (already JSON-safe: _evidence() serializes it via model_dump(mode="json")),
        # so /confirm can rehydrate it instead of losing it across the pause.
        if evidence.get("via_evidence"):
            pending_evidence["via_evidence"] = evidence["via_evidence"]
        # Preserve enhanced event pool if present (1.5H)
        if evidence.get("enhanced_event_pool"):
            pending_evidence["enhanced_event_pool"] = evidence["enhanced_event_pool"]
        run.pending_plan = {
            "plan": plan.model_dump(),
            "intent": resolved_intent,
            "evidence": pending_evidence,
        }
        flag_modified(run, "pending_plan")
        session.add(run)
        session.commit()
        return {"status": "NEEDS_USER", "plan": plan.model_dump(), "run_id": run.id}
    try:
        result = execute_plan(session, project_id, plan, key_prefix=key_prefix or new_id())
        checked = _check(session, project_id, plan, evidence)
        rounds = 0
        failures = {json.dumps(checked, sort_keys=True)}
        max_rounds = int(
            service.project_or_404(session, project_id).settings.get(
                "max_rework_rounds", DEFAULT_MAX_REWORK_ROUNDS
            )
        )
        while checked["status"] == "FAIL" and rounds < max(0, max_rounds):
            correction = _rework_plan(
                session, project_id, request, plan, checked, evidence, complete
            )
            if correction is None:
                break
            rounds += 1
            repaired = execute_plan(
                session,
                project_id,
                correction,
                key_prefix=f"{key_prefix or run.id}:rework:{rounds}",
            )
            result["operations"].extend(repaired["operations"])
            result["version"] = repaired["version"]
            plan = correction
            checked = _check(session, project_id, plan, evidence)
            fingerprint = json.dumps(checked, sort_keys=True)
            if fingerprint in failures:
                break
            failures.add(fingerprint)
        project = service.project_or_404(session, project_id)
        result["timeline"] = service.current_version(session, project).timeline
        if checked["status"] == "FAIL":
            checked["recommended_action"] = "NEEDS_USER"
        run.repair_rounds = rounds
        run.status, run.resulting_version = checked["status"], result["version"]
        run.operation_ids, run.verification = (
            [item["operation_id"] for item in result["operations"]],
            {**checked, "model_usage": model_usage} if model_usage else checked,
        )
        session.add(run)
        session.commit()
        return {
            "status": "PASS" if checked["status"] == "PASS" else "NEEDS_USER",
            "state": "VERIFYING",
            **result,
            "verification": checked,
            "report": {
                "changes": len(result["operations"]),
                "repair_rounds": rounds,
                "duration_us": result["timeline"].get("duration_us", 0),
                "verification": checked["status"],
                "undo_available": bool(result["operations"]),
            },
            "run_id": run.id,
        }
    except Exception:
        run.status = "FAILED"
        session.add(run)
        session.commit()
        raise
