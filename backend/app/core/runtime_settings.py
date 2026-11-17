"""Overrides editables desde la UI, persistidos en storage/app_settings.json."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from app.core.config import settings
from app.providers.base import DURATION_PRESETS

RUNTIME_SETTINGS_PATH = settings.storage_dir / "app_settings.json"

EDITABLE_FIELDS = {
    "llm_provider",
    "llm_model",
    "anthropic_api_key",
    "openai_api_key",
    "nvidia_api_key",
    "whisper_model",
    "default_num_clips",
    "clip_duration_preset",
    "min_clip_seconds",
    "max_clip_seconds",
    "enable_media_signals",
    "enable_visual_rerank",
    "vision_provider",
    "vision_model",
    "enable_ai_upscale",
    "nvidia_vision_model",
    "default_formats",
    "sponsorblock_categories",
    "reframe_mode",
    "yt_dlp_cookies_from_browser",
    "yt_dlp_cookies_path",
    "caption_style",
    "download_quality",
    "audio_format",
    "storage_retention_days",
    # Media foundation 1.0: defaults globales de mejoras de medios
    # (job-specific options ganan sobre estos).
    "destination_profile",
    "color_handling",
    "stabilization",
    "video_denoise",
    "audio_cleanup",
    # "extension_token" es persistible (para sobrevivir un restart) pero NO forma parte
    # de `SettingsUpdate` en api/settings.py: solo se escribe vía
    # POST /settings/extension-token/regenerate (secrets.token_urlsafe), nunca desde un
    # valor arbitrario de un PUT de cliente.
    "extension_token",
}

DOWNLOAD_QUALITIES = {"auto", "720", "1080", "1440", "2160"}
AUDIO_FORMATS = {"mp3", "wav", "m4a", "opus", "flac"}

CAPTION_STYLES = [
    {
        "value": "viral_bold",
        "label": "Viral Bold",
        "description": "Mayusculas, extra bold, contorno fuerte y palabra clave resaltada.",
    },
    {"value": "clean", "label": "Clean", "description": "Subtitulo legible y menos llamativo."},
    {"value": "neon", "label": "Neon", "description": "Subtitulo brillante con entrada rapida."},
    {"value": "minimal", "label": "Minimal", "description": "Texto simple con menos decoracion."},
    {"value": "gaming", "label": "Gaming", "description": "Colores fuertes para clips de accion."},
    {
        "value": "corporate",
        "label": "Corporate",
        "description": "Estilo sobrio para contenido profesional.",
    },
]

CAPTION_STYLE_VALUES = {item["value"] for item in CAPTION_STYLES}
CAPTION_STYLE_ALIASES = {
    "opus": "viral_bold",
    "podcast": "viral_bold",
    "bold": "viral_bold",
}


def _coerce_value(key: str, value: Any) -> Any:
    if key == "nvidia_vision_model":
        return str(value or "auto").strip() or "auto"
    if key == "llm_provider":
        # Fail-safe: un provider vacío o no listado NO puede terminar en una API
        # pagada. Catálogo = fuente de verdad del registry (+ "naive").
        from app.providers.registry import DEFAULT_MODELS

        provider = str(value or "naive").strip().lower()
        return provider if provider in DEFAULT_MODELS else "naive"
    if key == "default_num_clips":
        return max(1, min(12, int(value)))
    if key == "storage_retention_days":
        return max(0, min(3650, int(value)))
    if key == "clip_duration_preset":
        v = str(value or "standard").strip().lower()
        return v if v in DURATION_PRESETS else "standard"
    if key == "min_clip_seconds":
        return max(3.0, min(120.0, float(value)))
    if key == "max_clip_seconds":
        return max(8.0, min(300.0, float(value)))
    if key in {"enable_media_signals", "enable_visual_rerank", "enable_ai_upscale"}:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "si"}
    if key == "vision_provider":
        provider = str(value or "auto").strip().lower()
        # "nvidia" es free-tier (no pago): se puede elegir a proposito para VIA.
        return provider if provider in {"auto", "claude_code", "codex", "nvidia"} else "auto"
    if key in {"default_formats", "sponsorblock_categories"}:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]
    if key == "caption_style":
        style = str(value or "viral_bold").strip().lower()
        style = CAPTION_STYLE_ALIASES.get(style, style)
        return style if style in CAPTION_STYLE_VALUES else "viral_bold"
    if key == "download_quality":
        q = str(value or "1080").strip().lower()
        q = {"2k": "1440", "4k": "2160"}.get(q, q)
        return q if q in DOWNLOAD_QUALITIES else "1080"
    if key in {
        "destination_profile",
        "color_handling",
        "stabilization",
        "video_denoise",
        "audio_cleanup",
    }:
        # Los defaults de medios se validan con el MISMO contrato de media
        # options (fuente única de verdad para los enums).
        from app.core.media_options import normalize_media_options

        return normalize_media_options({key: value})[key]
    return str(value or "").strip()


def load_runtime_settings() -> dict[str, Any]:
    data = _read_runtime_settings()
    apply_runtime_settings(data, persist=False)
    return data


def _read_runtime_settings() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in EDITABLE_FIELDS}


def apply_runtime_settings(data: dict[str, Any], *, persist: bool = True) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in data.items():
        if key not in EDITABLE_FIELDS:
            continue
        clean[key] = _coerce_value(key, value)

    if clean.get("llm_provider", settings.llm_provider) == "nvidia" and (
        "llm_model" in clean or "llm_provider" in clean
    ):
        previous_model = settings.llm_model if settings.llm_provider == "nvidia" else "auto"
        clean["llm_model"] = clean.get("llm_model", previous_model) or "auto"

    # El modelo de vision pertenece al proveedor elegido. Al cambiar de proveedor,
    # un modelo anterior (p. ej. Claude en Codex) debe resetearse al default en vez
    # de quedar guardado y hacer fallar silenciosamente cada job.
    if (
        "vision_provider" in clean
        and "vision_model" not in clean
        and clean["vision_provider"] != getattr(settings, "vision_provider", "auto")
    ):
        clean["vision_model"] = ""
    if "vision_model" in clean and clean["vision_model"]:
        from app.providers.vision import vision_models_for

        configured = clean.get("vision_provider", getattr(settings, "vision_provider", "auto"))
        text_provider = clean.get("llm_provider", settings.llm_provider)
        effective_provider = text_provider if configured == "auto" else configured
        if clean["vision_model"] not in vision_models_for(effective_provider):
            clean["vision_model"] = ""

    # Cuando se selecciona un preset, sobreescribe min/max con sus valores canónicos
    if "clip_duration_preset" in clean:
        preset = DURATION_PRESETS.get(clean["clip_duration_preset"])
        if preset:
            clean["min_clip_seconds"] = preset["min"]
            clean["max_clip_seconds"] = preset["max"]

    for key, value in clean.items():
        setattr(settings, key, value)

    if (
        "min_clip_seconds" in clean or "max_clip_seconds" in clean
    ) and settings.max_clip_seconds < settings.min_clip_seconds:
        settings.max_clip_seconds = settings.min_clip_seconds
        clean["max_clip_seconds"] = settings.max_clip_seconds

    if persist:
        current = _read_runtime_settings()
        current.update(clean)
        settings.ensure_dirs()
        # Readers in other processes must see a complete old or new snapshot.
        fd, temporary = tempfile.mkstemp(dir=RUNTIME_SETTINGS_PATH.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(current, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, RUNTIME_SETTINGS_PATH)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return clean


def caption_style_options() -> list[dict[str, str]]:
    return CAPTION_STYLES


def effective_caption_style() -> str:
    return _coerce_value("caption_style", getattr(settings, "caption_style", "viral_bold"))
