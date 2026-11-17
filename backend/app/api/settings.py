"""Endpoint de settings efectivos (solo lectura en F1). PUT en F4."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.core.deps import check_ffmpeg
from app.core.download_auth import COOKIE_MODES, cookie_mode, validate_cookie_path
from app.core.platform_profiles import catalog
from app.core.processing_modes import PODCAST_VIRAL_CONFIG, processing_modes_payload
from app.core.runtime_settings import (
    AUDIO_FORMATS,
    DURATION_PRESETS,
    apply_runtime_settings,
    caption_style_options,
    effective_caption_style,
)
from app.pipeline.sponsorblock import DEFAULT_CATEGORIES as SPONSORBLOCK_DEFAULT_CATEGORIES
from app.pipeline.subtitles import DEFAULT_STYLE_CONFIG, HIGHLIGHT_COLORS
from app.pipeline.upscale import upscale_available as _upscale_available
from app.providers.cli_subscription import subscription_cli_available
from app.providers.registry import DEFAULT_MODELS
from app.providers.vision import (
    VISION_DEFAULT_MODELS,
    VISION_PROVIDER_CHOICES,
    resolve_vision,
    vision_models_for,
)
from app.video_intelligence.registry import get_registry

# Categorias soportadas en la UI: las de descarte por defecto de SponsorBlock mas
# intro/outro (tambien validas para excluir del pool de clips). Deliberadamente NO
# incluye poi_highlight: es señal positiva de highlight, no de tramo a descartar.
SPONSORBLOCK_CATEGORIES_AVAILABLE = [*SPONSORBLOCK_DEFAULT_CATEGORIES, "intro", "outro"]

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/nvidia-models")
def get_nvidia_models(refresh: bool = False) -> dict:
    from app.providers.nvidia_catalog import NvidiaCatalogError, get_catalog

    try:
        return get_catalog(refresh=refresh)
    except NvidiaCatalogError as exc:
        raise HTTPException(503, str(exc)) from exc


class SettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    whisper_model: str | None = None
    default_num_clips: int | None = None
    clip_duration_preset: str | None = None
    min_clip_seconds: float | None = None
    max_clip_seconds: float | None = None
    enable_media_signals: bool | None = None
    enable_visual_rerank: bool | None = None
    vision_provider: str | None = None
    vision_model: str | None = None
    enable_ai_upscale: bool | None = None
    nvidia_vision_model: str | None = None
    default_formats: list[str] | None = None
    sponsorblock_categories: list[str] | None = None
    reframe_mode: str | None = None
    yt_dlp_cookies_from_browser: str | None = None
    yt_dlp_cookies_path: str | None = None
    caption_style: str | None = None
    download_quality: str | None = None
    storage_retention_days: int | None = None
    # Media foundation 1.0: defaults globales de mejoras de medios.
    destination_profile: str | None = None
    color_handling: str | None = None
    stabilization: str | None = None
    video_denoise: str | None = None
    audio_cleanup: str | None = None


class ApiKeysUpdate(BaseModel):
    """Solo mutacion: los valores crudos nunca forman parte del GET de Ajustes."""

    claude: str | None = None
    openai: str | None = None
    nvidia: str | None = None


def _vision_effective() -> tuple[str, str]:
    """(proveedor, modelo) de vision que se usarian ahora mismo."""
    return resolve_vision(
        settings.llm_provider,
        settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider, ""),
    )


@router.get("")
def get_settings() -> dict:
    bins = check_ffmpeg()
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider, ""),
        "providers": list(DEFAULT_MODELS.keys()),
        "default_models": DEFAULT_MODELS,
        # Vision: proveedor y modelo EFECTIVOS, no solo lo configurado, para que
        # en Ajustes se vea cual se va a usar de verdad ("auto" no dice nada).
        "vision_provider": getattr(settings, "vision_provider", "auto"),
        "vision_model": getattr(settings, "vision_model", ""),
        "vision_providers": list(VISION_PROVIDER_CHOICES),
        "vision_default_models": VISION_DEFAULT_MODELS,
        "vision_models_by_provider": {
            provider: vision_models_for(provider) for provider in VISION_DEFAULT_MODELS
        },
        # Modelos SOLO del proveedor de vision efectivo. Devolver la union de
        # todos hacia que el selector mostrara claude-sonnet-5 dos veces y
        # gpt-4o-mini incluso habiendo elegido codex.
        "vision_models": vision_models_for(_vision_effective()[0]),
        "vision_effective": dict(zip(("provider", "model"), _vision_effective(), strict=True)),
        "whisper_model": settings.whisper_model,
        "default_num_clips": settings.default_num_clips,
        "clip_duration_preset": getattr(settings, "clip_duration_preset", "standard"),
        "duration_presets": [
            {
                "value": k,
                "label": v["label"],
                "description": v["description"],
                "min": v["min"],
                "max": v["max"],
            }
            for k, v in DURATION_PRESETS.items()
        ],
        "min_clip_seconds": settings.min_clip_seconds,
        "max_clip_seconds": settings.max_clip_seconds,
        "enable_media_signals": settings.enable_media_signals,
        "enable_visual_rerank": getattr(settings, "enable_visual_rerank", False),
        "enable_ai_upscale": getattr(settings, "enable_ai_upscale", False),
        "ai_upscale_available": _upscale_available(),
        "nvidia_vision_model": getattr(settings, "nvidia_vision_model", "") or "auto",
        "default_formats": settings.default_formats,
        "sponsorblock_categories": settings.sponsorblock_categories,
        "sponsorblock_categories_available": SPONSORBLOCK_CATEGORIES_AVAILABLE,
        "reframe_mode": settings.reframe_mode,
        "available_formats": ["9x16", "1x1", "16x9"],
        "processing_modes": processing_modes_payload(),
        "caption_style": effective_caption_style(),
        "caption_styles": caption_style_options(),
        "highlight_colors": HIGHLIGHT_COLORS,
        "default_style_config": DEFAULT_STYLE_CONFIG,
        "podcast_viral_config": PODCAST_VIRAL_CONFIG,
        "default_for_9_16": {
            "processing_mode": "podcast_viral",
            "subtitle_style": "viral_bold",
            "output_width": 1080,
            "output_height": 1920,
            "resize_mode": "cover",
            "allow_letterbox": False,
            "allow_black_bars": False,
        },
        "presets": {
            "default_preset_name": "Podcast Viral Premium",
            "allow_save_custom_preset": True,
            "allow_restore_default": True,
        },
        "yt_dlp_cookies_from_browser": settings.yt_dlp_cookies_from_browser,
        "yt_dlp_cookies_path": settings.yt_dlp_cookies_path,
        "download_quality": settings.download_quality,
        "audio_format": settings.audio_format,
        "available_audio_formats": sorted(AUDIO_FORMATS),
        "storage_retention_days": settings.storage_retention_days,
        # Media foundation 1.0: defaults globales + catálogo de destinos.
        "media_defaults": {
            "destination_profile": getattr(settings, "destination_profile", "auto"),
            "color_handling": getattr(settings, "color_handling", "auto"),
            "stabilization": getattr(settings, "stabilization", "off"),
            "video_denoise": getattr(settings, "video_denoise", "off"),
            "audio_cleanup": getattr(settings, "audio_cleanup", "off"),
        },
        "destination_profiles": catalog(),
        "extension_token_configured": bool(settings.extension_token),
        "yt_dlp_cookies_configured": cookie_mode(
            settings.yt_dlp_cookies_from_browser, settings.yt_dlp_cookies_path
        ) != "none",
        "keys_present": {
            "claude": bool(settings.anthropic_api_key),
            "openai": bool(settings.openai_api_key),
            "nvidia": bool(settings.nvidia_api_key),
        },
        # Disponibilidad de los proveedores de suscripción (CLI instalado en esta
        # maquina). No verifica login, solo que el binario exista.
        "subscription_available": {
            "claude_code": subscription_cli_available("claude_code"),
            "codex": subscription_cli_available("codex"),
        },
        "binaries": {
            name: {"available": s.available, "version": s.version, "path": s.path}
            for name, s in bins.items()
        },
    }


@router.get("/via-availability")
def get_via_availability() -> dict:
    """Get VIA (Video Intelligence) availability status."""
    registry = get_registry()
    provider = registry.get("snipra_vlm")
    
    if provider is None:
        return {
            "available": False,
            "code": "VIA_PROVIDER_UNAVAILABLE",
            "reason": "VIA provider not registered",
        }
    
    # Get availability from provider
    avail = provider.get_availability()
    return {
        "available": avail["available"],
        "code": avail["code"].value if avail["code"] else None,
        "reason": avail["reason"],
    }


@router.put("")
def update_settings(body: SettingsUpdate) -> dict:
    data = body.model_dump(exclude_unset=True)
    if "yt_dlp_cookies_from_browser" in data:
        raw = data["yt_dlp_cookies_from_browser"] or ""
        from app.pipeline.ingest import VideoDownloadError, _parse_browser_cookie_source

        mode = cookie_mode(raw, data.get("yt_dlp_cookies_path") or settings.yt_dlp_cookies_path)
        if mode not in COOKIE_MODES:
            try:
                _parse_browser_cookie_source(raw)
            except VideoDownloadError:
                raise HTTPException(400, "Modo de cookies no válido.") from None
        data["yt_dlp_cookies_from_browser"] = mode
    if data.get("yt_dlp_cookies_path"):
        try:
            data["yt_dlp_cookies_path"] = validate_cookie_path(data["yt_dlp_cookies_path"])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    mapped = {
        "llm_provider": data.get("provider"),
        "llm_model": data.get("model"),
        "vision_provider": data.get("vision_provider"),
        "vision_model": data.get("vision_model"),
        "whisper_model": data.get("whisper_model"),
        "default_num_clips": data.get("default_num_clips"),
        "clip_duration_preset": data.get("clip_duration_preset"),
        "min_clip_seconds": data.get("min_clip_seconds"),
        "max_clip_seconds": data.get("max_clip_seconds"),
        "enable_media_signals": data.get("enable_media_signals"),
        "enable_visual_rerank": data.get("enable_visual_rerank"),
        "enable_ai_upscale": data.get("enable_ai_upscale"),
        "nvidia_vision_model": data.get("nvidia_vision_model"),
        "default_formats": data.get("default_formats"),
        "sponsorblock_categories": data.get("sponsorblock_categories"),
        "reframe_mode": data.get("reframe_mode"),
        "yt_dlp_cookies_from_browser": data.get("yt_dlp_cookies_from_browser"),
        "yt_dlp_cookies_path": data.get("yt_dlp_cookies_path"),
        "caption_style": data.get("caption_style"),
        "download_quality": data.get("download_quality"),
        "audio_format": data.get("audio_format"),
        "storage_retention_days": data.get("storage_retention_days"),
        "destination_profile": data.get("destination_profile"),
        "color_handling": data.get("color_handling"),
        "stabilization": data.get("stabilization"),
        "video_denoise": data.get("video_denoise"),
        "audio_cleanup": data.get("audio_cleanup"),
    }
    apply_runtime_settings({k: v for k, v in mapped.items() if v is not None})
    return get_settings()


@router.put("/api-keys")
def update_api_keys(body: ApiKeysUpdate) -> dict:
    data = body.model_dump(exclude_unset=True)
    mapping = {
        "claude": "anthropic_api_key",
        "openai": "openai_api_key",
        "nvidia": "nvidia_api_key",
    }
    apply_runtime_settings({mapping[key]: value for key, value in data.items()})
    # Devuelve solo presencia/ausencia mediante get_settings(); nunca hace eco
    # de una clave aunque el cliente la acabe de enviar.
    return get_settings()


@router.post("/extension-token/regenerate")
def regenerate_extension_token() -> dict:
    """Genera (o rota) el token compartido que autentica a la extension de navegador.

    Server-only: nunca se acepta un valor de token desde el body de un cliente, para
    que no pueda imponerse uno débil/predecible. Vacío = feature deshabilitada.

    El valor crudo del token solo se revela UNA VEZ, en esta respuesta: el GET general
    de /api/settings solo expone `extension_token_configured` (bool), nunca el valor,
    para que un GET sin protección de Origin no pueda filtrarlo (ver hallazgo de
    seguridad 2026-07-21: una extension de terceros con host_permissions amplios
    puede leer respuestas cross-origin incluso cuando el guard de mutaciones bloquea
    escrituras).
    """
    apply_runtime_settings({"extension_token": secrets.token_urlsafe(32)})
    payload = get_settings()
    payload["extension_token"] = settings.extension_token
    return payload
