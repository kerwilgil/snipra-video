from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> datetime:
    return datetime.now(UTC)


class EditorProject(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    name: str = "Untitled project"
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    source_job_id: str | None = Field(default=None, index=True)
    # A result from Auto is a specific Clip, not merely its parent job.  This
    # stable association makes opening the same result idempotent.
    source_clip_id: str | None = Field(default=None, index=True)
    current_version: int = 0
    settings: dict = Field(default_factory=dict, sa_column=Column(JSON))
    agent_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))


class EditorAsset(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    project_id: str = Field(index=True, foreign_key="editorproject.id")
    created_at: datetime = Field(default_factory=now)
    kind: str = "video"
    original_path: str
    filename: str
    fingerprint: str = ""
    duration_us: int = 0
    width: int = 0
    height: int = 0
    fps_num: int = 0
    fps_den: int = 1
    has_audio: bool = False
    asset_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))


class AssetTask(SQLModel, table=True):
    """Dedicated entity for long enhancement / (1.6D) generation operations.

    Deliberately NOT the source-processing Job. Enums live in
    app.editor.asset_tasks; columns store their string values. Never stores
    secrets or API keys - `request` holds a typed EnhancementRequest payload,
    `cost_estimate` a typed EnhancementEstimate/GenerationEstimate payload.
    """

    id: str = Field(default_factory=new_id, primary_key=True)
    project_id: str = Field(index=True, foreign_key="editorproject.id")
    kind: str = "ENHANCEMENT"
    status: str = "ESTIMATING"

    provider: str = ""
    model: str = ""

    request: dict = Field(default_factory=dict, sa_column=Column(JSON))
    request_hash: str = ""

    cost_estimate: dict = Field(default_factory=dict, sa_column=Column(JSON))
    approval_state: str = "PENDING"
    approved_request_hash: str | None = None

    progress: float = 0.0

    # External generation task tracking (for GENERATION kind)
    external_task_id: str | None = None
    external_task_state: str | None = None
    external_task_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # Next scheduled poll time for durable polling (UTC)
    next_poll_at: datetime | None = None
    # Counter for submit attempts (exactly-once semantics)
    submit_count: int = 0
    # Cancel request flag for durable cancellation
    cancel_requested: bool = False

    output_asset_id: str | None = Field(default=None, foreign_key="editorasset.id")
    error: str | None = None

    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)


class EditorVersion(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("project_id", "number", name="uq_editor_version"),)
    id: str = Field(default_factory=new_id, primary_key=True)
    project_id: str = Field(index=True, foreign_key="editorproject.id")
    number: int
    parent_number: int | None = None
    created_at: datetime = Field(default_factory=now)
    created_by: str = "user"
    timeline: dict = Field(default_factory=dict, sa_column=Column(JSON))
    summary: str = ""


class EditorOperation(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("project_id", "idempotency_key", name="uq_editor_idempotency"),
    )
    id: str = Field(default_factory=new_id, primary_key=True)
    project_id: str = Field(index=True, foreign_key="editorproject.id")
    version_number: int
    idempotency_key: str
    kind: str
    created_at: datetime = Field(default_factory=now)
    generated_by: str = "user"
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))


class EditorAgentRun(SQLModel, table=True):
    """Compact audit trail; prompts and full transcript evidence are never stored."""

    id: str = Field(default_factory=new_id, primary_key=True)
    project_id: str = Field(index=True, foreign_key="editorproject.id")
    base_version: int
    resulting_version: int | None = None
    status: str = "INSPECTING"
    repair_rounds: int = 0
    operation_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    verification: dict = Field(default_factory=dict, sa_column=Column(JSON))
    pending_plan: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
