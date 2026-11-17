"""Modelos de datos (SQLModel) para jobs, momentos detectados y assets renderizados."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    queued = "queued"
    ingesting = "ingesting"
    transcribing = "transcribing"
    detecting = "detecting"
    rendering = "rendering"
    done = "done"
    failed = "failed"
    canceled = "canceled"


class SourceType(StrEnum):
    upload = "upload"
    url = "url"


class ClipOperationStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class CompilationStatus(StrEnum):
    queued = "queued"
    rendering = "rendering"
    done = "done"
    failed = "failed"


class Job(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    # Se actualiza solo cuando cambia la etapa principal (status), no con cada
    # porcentaje. Permite mostrar un cronometro real aunque se recargue la UI.
    stage_started_at: datetime = Field(default_factory=_now)

    source_type: SourceType
    source_name: str  # nombre de archivo original o URL
    title: str = ""

    status: JobStatus = Field(default=JobStatus.queued, index=True)
    stage: str = ""  # mensaje legible de la etapa actual
    progress: float = 0.0  # 0..1
    error: str | None = None

    duration: float | None = None  # duración del source en segundos
    # resolución real del video fuente tras descargar/sondear (0 = aún no sondeado).
    # No es lo mismo que download_quality (el techo pedido): esto es lo que realmente llegó.
    source_width: int = 0
    source_height: int = 0

    # parámetros del job
    num_clips: int = 5
    formats: str = "9x16"  # csv: 9x16,1x1,16x9
    processing_mode: str = "auto_smart"
    caption_style: str = "viral_bold"
    output_mode: str = "clips"  # clips | download | audio
    captions_enabled: bool = True
    download_quality: str = ""  # auto | 720 | 1080 | 1440 | 2160
    audio_format: str = ""  # mp3 | wav | m4a | opus | flac (solo output_mode=audio)
    provider: str = "naive"
    model: str = ""
    detection_source: str = ""  # llm | fallback | mixed | music | signal
    sponsorblock_enabled: bool = True
    # tramos [{start, end, category}] en segundos devueltos por SponsorBlock (solo YouTube)
    sponsorblock_segments: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    # Funciones editoriales avanzadas (edición compacta, tarjetas, privacidad,
    # zonas protegidas, QA y precisión de transcripción). JSON evita una columna
    # por interruptor y mantiene el contrato extensible.
    editorial_options: dict = Field(default_factory=dict, sa_column=Column(JSON))
    # Mejoras de medios 1.0 (destination_profile, color_handling, stabilization,
    # video_denoise, audio_cleanup). Mismo patrón JSON del contrato extensible.
    media_options: dict = Field(default_factory=dict, sa_column=Column(JSON))

    clips: list["Clip"] = Relationship(
        back_populates="job",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    @property
    def format_list(self) -> list[str]:
        return [f.strip() for f in self.formats.split(",") if f.strip()]


class Clip(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    index: int = 0

    start: float
    end: float
    title: str = ""
    hook: str = ""
    score: float = 0.0
    reason: str = ""
    detection_source: str = ""
    hashtags: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    description: str = ""
    source_ranges: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    editorial_cards: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    privacy_suggestions: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    privacy_masks: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    protected_zones: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    editorial_qa: dict = Field(default_factory=dict, sa_column=Column(JSON))
    score_breakdown: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    decision_results: list[dict] = Field(default_factory=list, sa_column=Column(JSON))

    job: Job | None = Relationship(back_populates="clips")
    assets: list["ClipAsset"] = Relationship(
        back_populates="clip",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    @property
    def duration(self) -> float:
        if self.source_ranges:
            return sum(
                max(0.0, float(item.get("end", 0)) - float(item.get("start", 0)))
                for item in self.source_ranges
            )
        return max(0.0, self.end - self.start)


class ClipAsset(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)

    format: str  # 9x16 | 1x1 | 16x9
    path: str
    thumb_path: str | None = None
    width: int = 0
    height: int = 0
    duration: float = 0.0
    size_bytes: int = 0

    clip: Clip | None = Relationship(back_populates="assets")


class JobLog(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    created_at: datetime = Field(default_factory=_now, index=True)
    level: str = "info"
    message: str


class ClipOperation(SQLModel, table=True):
    """Edicion masiva persistente consumida por el worker local."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    job_id: str = Field(foreign_key="job.id", index=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)
    created_at: datetime = Field(default_factory=_now, index=True)
    updated_at: datetime = Field(default_factory=_now)
    kind: str
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: ClipOperationStatus = Field(default=ClipOperationStatus.queued, index=True)
    progress: float = 0.0
    error: str | None = None


class Compilation(SQLModel, table=True):
    """Varios ClipAsset ya renderizados unidos en un video nuevo.

    Deliberadamente NO es un Job: un Job asume que existe un video fuente
    (source_type upload|url, probe_source, ingest/transcribe/detect). Una
    compilacion no tiene fuente, solo concatena mp4 que ya existen, asi que
    comparte cero etapas con ese pipeline y vive en su propia tabla.
    """

    id: str = Field(default_factory=_uuid, primary_key=True)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    title: str = ""
    # Formato heredado de los assets elegidos; todos deben coincidir para poder
    # concatenar sin recortar ni deformar.
    format: str = "9x16"
    # Orden de la lista = orden final en el video.
    asset_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    transition: str = "none"  # none | fade
    transition_seconds: float = 0.4

    bgm_path: str = ""  # ruta local del audio de fondo; vacio = sin musica
    bgm_volume: float = 0.18  # 0..1
    bgm_ducking: bool = True  # baja la musica cuando hay voz (sidechaincompress)
    bgm_auto_level: bool = True  # mide voz/música y calcula la ganancia por LUFS
    bgm_duck_lu: float = 12.0  # relación objetivo: música N LU por debajo de la voz
    audio_report: dict = Field(default_factory=dict, sa_column=Column(JSON))

    status: CompilationStatus = Field(default=CompilationStatus.queued, index=True)
    progress: float = 0.0
    error: str | None = None

    output_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    size_bytes: int = 0


class TrackedChannel(SQLModel, table=True):
    """Canal/playlist de YouTube vigilado por el worker para auto-encolar videos nuevos."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    created_at: datetime = Field(default_factory=_now)

    url: str
    label: str = ""
    enabled: bool = True
    poll_interval_minutes: int = 60
    last_checked_at: datetime | None = None

    # defaults aplicados a cada job creado automáticamente para este canal
    # (mismo shape que los parámetros de creación de Job vía URL)
    num_clips: int = 5
    formats: str = "9x16"  # csv: 9x16,1x1,16x9
    processing_mode: str = "auto_smart"
    caption_style: str = "viral_bold"
    output_mode: str = "clips"
    captions_enabled: bool = True
    download_quality: str = ""
    provider: str = "naive"
    model: str = ""
    sponsorblock_enabled: bool = True

    # Observabilidad y salud del canal
    last_check_status: str = Field(default="")  # healthy | warning | error | ""
    last_check_error: str | None = Field(default=None)
    last_success_at: datetime | None = Field(default=None)
    consecutive_failures: int = Field(default=0)
    videos_seen_last_check: int = Field(default=0)
    new_videos_last_check: int = Field(default=0)
    canonical_url: str | None = Field(default=None)
    channel_title: str | None = Field(default=None)

    @property
    def format_list(self) -> list[str]:
        return [f.strip() for f in self.formats.split(",") if f.strip()]


class TrackedChannelVideo(SQLModel, table=True):
    """Marca un video de un canal trackeado como ya visto (idempotencia del polling)."""

    __table_args__ = (UniqueConstraint("channel_id", "video_id", name="uq_tracked_channel_video"),)

    id: str = Field(default_factory=_uuid, primary_key=True)
    channel_id: str = Field(foreign_key="trackedchannel.id", index=True)
    video_id: str = Field(index=True)
    job_id: str | None = Field(default=None, foreign_key="job.id")
    created_at: datetime = Field(default_factory=_now)
