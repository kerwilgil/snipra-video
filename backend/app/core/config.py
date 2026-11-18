"""Configuración central cargada desde .env (pydantic-settings)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Snipra"
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    # Frontend compilado (vite build). Vacío = modo desarrollo: la UI la sirve Vite.
    # La app macOS empaquetada lo define para servir la UI desde el mismo origen.
    frontend_dist_dir: Path | None = None

    # --- Almacenamiento ---
    storage_dir: Path = PROJECT_DIR / "storage"
    # 0 = nunca borrar automaticamente; la limpieza solo ocurre si el usuario la activa
    storage_retention_days: int = 0
    database_url: str = ""  # se autocompleta a sqlite en storage_dir si queda vacío

    # --- Proveedor de IA (detección de momentos) ---
    # naive | claude | openai | nvidia | claude_code | codex
    # DEFAULT SEGURO: "naive" (local, offline). Una instalación nueva NUNCA debe
    # consumir una API pagada solo por defaults. Claude/OpenAI/NVIDIA o los CLI
    # de suscripción únicamente se usan si el usuario los elige explícitamente
    # (LLM_PROVIDER en .env o Ajustes). Tener una *_API_KEY en el entorno NO
    # selecciona nada por sí sola.
    llm_provider: str = "naive"
    llm_model: str = ""  # vacío => default por proveedor (ver providers/registry.py)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # --- Transcripción (faster-whisper) ---
    whisper_model: str = "base"  # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"  # cpu | cuda | auto (auto puede fallar sin DLLs de CUDA)
    whisper_compute_type: str = "int8"  # int8 | float16 | float32

    # --- Clipping ---
    default_num_clips: int = 5
    clip_duration_preset: str = "standard"
    min_clip_seconds: float = 45.0
    max_clip_seconds: float = 90.0
    enable_media_signals: bool = True
    # Re-rank visual con modelo multimodal. Usa más tokens.
    enable_visual_rerank: bool = False
    # P1.1: only records structured visual diagnostics; never changes ranking/output.
    enable_structured_visual_shadow: bool = False
    # Optional advisory layer; never controls detection, score, selection or rendering.
    decision_engine_enabled: bool = False
    decision_engine_provider: str = Field(default="rules", min_length=1)
    decision_engine_min_confidence: float = Field(default=0.65, ge=0, le=1, allow_inf_nan=False)
    decision_engine_timeout_ms: int = Field(default=2000, gt=0)
    # Proveedor de visión, INDEPENDIENTE del de texto: se puede analizar el
    # transcript con uno y las imágenes con otro (p. ej. texto con Claude Code y
    # visión con Codex, que redacta mejor). "auto" = el mismo que el de texto,
    # para que elegir una suscripción no termine llamando a una API de pago.
    vision_provider: str = "auto"
    # Modelo de visión concreto. Vacío = el que corresponda al proveedor. Se
    # expone en Ajustes para que sea visible cuál se está usando.
    vision_model: str = ""
    # Modelo de visión para NVIDIA NIM (los de texto no ven imágenes).
    # NVIDIA es opción EXPLÍCITA de VIA (grounding, free-tier) desde N.1;
    # está DESHABILITADA como rerank de clips (alucina) y NUNCA se elige por
    # "auto" (ver app/providers/vision.py::resolve_vision).
    nvidia_vision_model: str = "auto"
    # Super-resolución IA (Real-ESRGAN ncnn-vulkan). Requiere GPU + binario. Lento.
    enable_ai_upscale: bool = False
    realesrgan_path: str = "realesrgan-ncnn-vulkan"  # binario en PATH o ruta absoluta
    realesrgan_model: str = "realesrgan-x4plus"
    # formatos destino por defecto: 9x16 | 1x1 | 16x9
    default_formats: Annotated[list[str], NoDecode] = ["9x16"]
    # categorías SponsorBlock excluidas del pool de clips por defecto (solo YouTube)
    sponsorblock_categories: Annotated[list[str], NoDecode] = [
        "sponsor",
        "selfpromo",
        "interaction",
        "preview",
    ]

    # --- Reframe ---
    # auto | preserve | center | blur | track
    reframe_mode: str = "auto"
    enable_tracking: bool = True  # requiere extra [vision]; degrada a center si falta
    caption_style: str = "viral_bold"

    # --- Media foundation 1.0 (mejoras de medios) ---
    # destination_profile: "auto" (genérico según formato) o id de platform
    # profile (ver app/core/platform_profiles.py).
    destination_profile: str = "auto"
    # auto | preserve | sdr — manejo de color HDR→SDR (auto: tone map si la
    # fuente es HDR, no tocar si es SDR).
    color_handling: str = "auto"
    # off | auto | light | medium | strong — estabilización opcional (off por defecto)
    stabilization: str = "off"
    # off | light | medium | strong — denoise de video opcional (off por defecto)
    video_denoise: str = "off"
    # off | voice_light | voice | voice_strong — limpieza de audio opcional
    audio_cleanup: str = "off"

    # --- Binarios (override opcional) ---
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # --- Descarga URL (yt-dlp) ---
    # Sesiones existentes: auto|none|chrome|brave|edge|firefox|cookies_file.
    # Solo se persisten el modo y la ruta local de un cookies.txt; nunca sus valores.
    yt_dlp_cookies_path: str = ""
    yt_dlp_cookies_from_browser: str = "auto"
    # Opt-in solo para usuarios que descargan deliberadamente desde su propia LAN.
    allow_private_downloads: bool = False
    # calidad de descarga: auto (mejor disponible) | 720 | 1080 | 1440 (2K)
    download_quality: str = "1080"
    # formato de audio para el modo "Extraer audio": mp3 | wav | m4a | opus | flac
    audio_format: str = "mp3"

    # --- Extension de navegador (popup MV3, uso local personal) ---
    # Token compartido requerido en el header X-Snipra-Extension-Token para que la
    # extension pueda mutar la API. Vacio = feature deshabilitada (fail-closed): sin
    # token configurado, ninguna peticion con origin chrome-extension://* puede mutar.
    # Se genera desde Ajustes -> Extension de navegador (POST /settings/extension-token/regenerate).
    extension_token: str = ""

    @field_validator("cors_origins", "default_formats", "sponsorblock_categories", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.storage_dir / 'mediashorts.db').as_posix()}"

    @property
    def work_dir(self) -> Path:
        return self.storage_dir / "work"

    @property
    def compositions_dir(self) -> Path:
        """Salida de las compilaciones. Separada de work/ porque no pertenece a
        ningun job y no debe borrarla la retencion de artefactos de jobs."""
        return self.storage_dir / "compositions"

    @property
    def editor_dir(self) -> Path:
        return self.storage_dir / "editor"

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.compositions_dir.mkdir(parents=True, exist_ok=True)
        self.editor_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
