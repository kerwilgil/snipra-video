"""AssetTask state machine + approval binding (SNIPRA 1.6C Block 01).

Long enhancement / (future) generation operations do NOT reuse the source
`Job` pipeline. `AssetTask` (table in app.editor.models) is a small dedicated
entity. This module owns its enums, valid transitions, and the
approval-binding rule that 1.6D's paid operations will depend on.

No engine runs here. No external/API calls.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from app.editor.enhancement import EnhancementEstimate
from app.editor.generation import GenerationEstimate


class AssetTaskKind(StrEnum):
    ENHANCEMENT = "ENHANCEMENT"
    GENERATION = "GENERATION"


class AssetTaskStatus(StrEnum):
    ESTIMATING = "ESTIMATING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AssetTaskApprovalState(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


_TERMINAL: frozenset[AssetTaskStatus] = frozenset(
    {AssetTaskStatus.COMPLETED, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED}
)

# Every non-terminal state may fail or be cancelled. Forward progress is
# strictly linear otherwise.
VALID_TRANSITIONS: dict[AssetTaskStatus, frozenset[AssetTaskStatus]] = {
    AssetTaskStatus.ESTIMATING: frozenset(
        {AssetTaskStatus.AWAITING_APPROVAL, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED}
    ),
    AssetTaskStatus.AWAITING_APPROVAL: frozenset(
        {AssetTaskStatus.QUEUED, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED}
    ),
    AssetTaskStatus.QUEUED: frozenset(
        {AssetTaskStatus.RUNNING, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED}
    ),
    AssetTaskStatus.RUNNING: frozenset(
        {AssetTaskStatus.COMPLETED, AssetTaskStatus.FAILED, AssetTaskStatus.CANCELLED}
    ),
    AssetTaskStatus.COMPLETED: frozenset(),
    AssetTaskStatus.FAILED: frozenset(),
    AssetTaskStatus.CANCELLED: frozenset(),
}


def is_terminal(status: AssetTaskStatus | str) -> bool:
    return AssetTaskStatus(status) in _TERMINAL


def can_transition(current: AssetTaskStatus | str, nxt: AssetTaskStatus | str) -> bool:
    return AssetTaskStatus(nxt) in VALID_TRANSITIONS[AssetTaskStatus(current)]


class InvalidTaskTransition(ValueError):
    """Raised on an illegal AssetTask status transition. Fail closed."""


class EnhancementValidationError(ValueError):
    """Raised when an enhancement request fails validation. Fail closed."""


def assert_transition(current: AssetTaskStatus | str, nxt: AssetTaskStatus | str) -> None:
    if not can_transition(current, nxt):
        raise InvalidTaskTransition(
            f"illegal AssetTask transition {AssetTaskStatus(current)} -> {AssetTaskStatus(nxt)}"
        )


def compute_request_hash(payload: dict) -> str:
    """Canonical SHA-256 of a request payload. Key-order independent."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approval_is_valid(
    *,
    approval_state: AssetTaskApprovalState | str,
    approved_request_hash: str | None,
    current_request_hash: str,
) -> bool:
    """An approval is only valid while it is bound to the EXACT request that
    will run. If the request changed after approval (hash differs), or it was
    never approved, the approval is void."""
    if AssetTaskApprovalState(approval_state) != AssetTaskApprovalState.APPROVED:
        return False
    if not approved_request_hash:
        return False
    return approved_request_hash == current_request_hash


def estimate_permits_execution(
    cost_estimate: dict | EnhancementEstimate | GenerationEstimate | None,
) -> bool:
    """Fail closed: no estimate, or a paid estimate with unknown monetary cost,
    must not proceed to QUEUED/RUNNING."""
    if cost_estimate is None:
        return False
    est = cost_estimate
    if isinstance(cost_estimate, dict):
        # Discriminate by provider field
        provider = cost_estimate.get("provider")
        if provider == "local_ltx":
            est = GenerationEstimate.model_validate(cost_estimate)
        else:
            est = EnhancementEstimate.model_validate(cost_estimate)
    return est.permits_execution()


def guard_ready_to_queue(
    *,
    approval_state: AssetTaskApprovalState | str,
    approved_request_hash: str | None,
    current_request_hash: str,
    cost_estimate: dict | EnhancementEstimate | GenerationEstimate | None,
) -> None:
    """All conditions that must hold before AWAITING_APPROVAL -> QUEUED.
    Raises (fail closed) if any fails."""
    if not approval_is_valid(
        approval_state=approval_state,
        approved_request_hash=approved_request_hash,
        current_request_hash=current_request_hash,
    ):
        raise InvalidTaskTransition("approval is missing or no longer bound to the current request")
    if not estimate_permits_execution(cost_estimate):
        raise InvalidTaskTransition("cost estimate does not permit execution (unknown paid cost)")
