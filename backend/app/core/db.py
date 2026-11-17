"""Capa de base de datos SQLite vía SQLModel."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import inspect, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

_connect_args = {"check_same_thread": False}
engine = create_engine(
    settings.resolved_database_url,
    echo=False,
    connect_args=_connect_args,
    poolclass=StaticPool if settings.resolved_database_url.endswith(":memory:") else None,
)


def init_db() -> None:
    # importa modelos para que se registren en SQLModel.metadata
    from app import models  # noqa: F401
    from app.editor import models as editor_models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _migrate_existing_sqlite()


def _migrate_existing_sqlite() -> None:
    if not settings.resolved_database_url.startswith("sqlite"):
        return

    inspector = inspect(engine)
    if "job" not in inspector.get_table_names():
        return

    columns = {col["name"] for col in inspector.get_columns("job")}
    with engine.begin() as conn:
        if "processing_mode" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE job ADD COLUMN processing_mode "
                    "VARCHAR DEFAULT 'auto_smart' NOT NULL"
                )
            )
        if "caption_style" not in columns:
            conn.execute(
                text(
                    "ALTER TABLE job ADD COLUMN caption_style VARCHAR DEFAULT 'viral_bold' NOT NULL"
                )
            )
        if "detection_source" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN detection_source VARCHAR DEFAULT '' NOT NULL")
            )
        if "output_mode" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN output_mode VARCHAR DEFAULT 'clips' NOT NULL")
            )
        if "captions_enabled" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN captions_enabled BOOLEAN DEFAULT 1 NOT NULL")
            )
        if "download_quality" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN download_quality VARCHAR DEFAULT '' NOT NULL")
            )
        if "source_width" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN source_width INTEGER DEFAULT 0 NOT NULL"))
        if "source_height" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN source_height INTEGER DEFAULT 0 NOT NULL")
            )
        if "audio_format" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN audio_format VARCHAR DEFAULT '' NOT NULL")
            )
        if "sponsorblock_enabled" not in columns:
            conn.execute(
                text("ALTER TABLE job ADD COLUMN sponsorblock_enabled BOOLEAN DEFAULT 1 NOT NULL")
            )
        if "sponsorblock_segments" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN sponsorblock_segments JSON DEFAULT '[]'"))
        if "stage_started_at" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN stage_started_at DATETIME"))
            conn.execute(
                text("UPDATE job SET stage_started_at = updated_at WHERE stage_started_at IS NULL")
            )
        if "editorial_options" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN editorial_options JSON DEFAULT '{}'"))
        if "media_options" not in columns:
            conn.execute(text("ALTER TABLE job ADD COLUMN media_options JSON DEFAULT '{}'"))

    if "editorproject" in inspector.get_table_names():
        editor_columns = {col["name"] for col in inspector.get_columns("editorproject")}
        with engine.begin() as conn:
            if "source_clip_id" not in editor_columns:
                conn.execute(text("ALTER TABLE editorproject ADD COLUMN source_clip_id VARCHAR"))
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_editorproject_source_clip_id "
                        "ON editorproject (source_clip_id)"
                    )
                )

    if "editoragentrun" in inspector.get_table_names():
        agent_run_columns = {col["name"] for col in inspector.get_columns("editoragentrun")}
        with engine.begin() as conn:
            if "pending_plan" not in agent_run_columns:
                conn.execute(text("ALTER TABLE editoragentrun ADD COLUMN pending_plan JSON"))

    if "clip" in inspector.get_table_names():
        clip_columns = {col["name"] for col in inspector.get_columns("clip")}
        with engine.begin() as conn:
            if "detection_source" not in clip_columns:
                conn.execute(
                    text("ALTER TABLE clip ADD COLUMN detection_source VARCHAR DEFAULT '' NOT NULL")
                )
            if "hashtags" not in clip_columns:
                conn.execute(text("ALTER TABLE clip ADD COLUMN hashtags JSON DEFAULT '[]'"))
            if "description" not in clip_columns:
                conn.execute(
                    text("ALTER TABLE clip ADD COLUMN description VARCHAR DEFAULT '' NOT NULL")
                )
            if "source_ranges" not in clip_columns:
                conn.execute(text("ALTER TABLE clip ADD COLUMN source_ranges JSON DEFAULT '[]'"))
            for name, default in (
                ("editorial_cards", "[]"),
                ("privacy_suggestions", "[]"),
                ("privacy_masks", "[]"),
                ("protected_zones", "[]"),
                ("editorial_qa", "{}"),
                ("score_breakdown", "[]"),
                ("decision_results", "[]"),
            ):
                if name not in clip_columns:
                    conn.execute(
                        text(f"ALTER TABLE clip ADD COLUMN {name} JSON DEFAULT '{default}'")
                    )

    if "compilation" in inspector.get_table_names():
        compilation_columns = {col["name"] for col in inspector.get_columns("compilation")}
        with engine.begin() as conn:
            if "bgm_auto_level" not in compilation_columns:
                conn.execute(
                    text(
                        "ALTER TABLE compilation ADD COLUMN bgm_auto_level "
                        "BOOLEAN DEFAULT 1 NOT NULL"
                    )
                )
            if "bgm_duck_lu" not in compilation_columns:
                conn.execute(
                    text("ALTER TABLE compilation ADD COLUMN bgm_duck_lu FLOAT DEFAULT 12 NOT NULL")
                )
            if "audio_report" not in compilation_columns:
                conn.execute(
                    text("ALTER TABLE compilation ADD COLUMN audio_report JSON DEFAULT '{}'")
                )

    if "trackedchannel" in inspector.get_table_names():
        channel_columns = {col["name"] for col in inspector.get_columns("trackedchannel")}
        with engine.begin() as conn:
            if "last_check_status" not in channel_columns:
                conn.execute(
                    text(
                        "ALTER TABLE trackedchannel ADD COLUMN last_check_status "
                        "VARCHAR DEFAULT '' NOT NULL"
                    )
                )
            if "last_check_error" not in channel_columns:
                conn.execute(text("ALTER TABLE trackedchannel ADD COLUMN last_check_error VARCHAR"))
            if "last_success_at" not in channel_columns:
                conn.execute(text("ALTER TABLE trackedchannel ADD COLUMN last_success_at DATETIME"))
            if "consecutive_failures" not in channel_columns:
                conn.execute(
                    text(
                        "ALTER TABLE trackedchannel ADD COLUMN consecutive_failures "
                        "INTEGER DEFAULT 0 NOT NULL"
                    )
                )
            if "videos_seen_last_check" not in channel_columns:
                conn.execute(
                    text(
                        "ALTER TABLE trackedchannel ADD COLUMN videos_seen_last_check "
                        "INTEGER DEFAULT 0 NOT NULL"
                    )
                )
            if "new_videos_last_check" not in channel_columns:
                conn.execute(
                    text(
                        "ALTER TABLE trackedchannel ADD COLUMN new_videos_last_check "
                        "INTEGER DEFAULT 0 NOT NULL"
                    )
                )
            if "canonical_url" not in channel_columns:
                conn.execute(text("ALTER TABLE trackedchannel ADD COLUMN canonical_url VARCHAR"))
            if "channel_title" not in channel_columns:
                conn.execute(text("ALTER TABLE trackedchannel ADD COLUMN channel_title VARCHAR"))

    if "assettask" in inspector.get_table_names():
        asset_task_columns = {col["name"] for col in inspector.get_columns("assettask")}
        with engine.begin() as conn:
            if "external_task_id" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_id VARCHAR"))
            if "external_task_state" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_state VARCHAR"))
            if "external_task_metadata" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN external_task_metadata JSON DEFAULT '{}'"))
            if "next_poll_at" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN next_poll_at DATETIME"))
            if "submit_count" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN submit_count INTEGER DEFAULT 0 NOT NULL"))
            if "cancel_requested" not in asset_task_columns:
                conn.execute(text("ALTER TABLE assettask ADD COLUMN cancel_requested BOOLEAN DEFAULT 0 NOT NULL"))


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
