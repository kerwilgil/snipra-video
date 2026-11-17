"""Contrato de opciones de medios por job (media foundation 1.0).

Backwards-compatible: los jobs existentes que no envían media_options
reciben los defaults globales (todo OFF salvo color auto), de modo que un
SDR común con todo en OFF produce exactamente el mismo resultado que antes.

Job-specific options ganan sobre Settings (Bloque J1).
"""

from __future__ import annotations

from app.core.config import settings
from app.core.platform_profiles import PROFILES

COLOR_HANDLING_MODES = ("auto", "preserve", "sdr")
STABILIZATION_MODES = ("off", "auto", "light", "medium", "strong")
VIDEO_DENOISE_MODES = ("off", "light", "medium", "strong")
AUDIO_CLEANUP_MODES = ("off", "voice_light", "voice", "voice_strong")


def _clean_choice(value: object, allowed: tuple[str, ...], fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip().lower()
    return normalized if normalized in allowed else fallback


def normalize_media_options(value: dict | None) -> dict:
    """Normaliza y valida las opciones de medios. Descarta claves desconocidas."""
    raw = value if isinstance(value, dict) else {}
    profile = str(raw.get("destination_profile") or "").strip().lower()
    if profile not in PROFILES:
        profile = "auto"
    return {
        "destination_profile": profile,
        "color_handling": _clean_choice(
            raw.get("color_handling"), COLOR_HANDLING_MODES, settings.color_handling
        ),
        "stabilization": _clean_choice(
            raw.get("stabilization"), STABILIZATION_MODES, settings.stabilization
        ),
        "video_denoise": _clean_choice(
            raw.get("video_denoise"), VIDEO_DENOISE_MODES, settings.video_denoise
        ),
        "audio_cleanup": _clean_choice(
            raw.get("audio_cleanup"), AUDIO_CLEANUP_MODES, settings.audio_cleanup
        ),
    }


def all_options_off(options: dict | None) -> bool:
    """True cuando ninguna mejora opcional está activa (regression contract)."""
    normalized = normalize_media_options(options)
    return (
        normalized["stabilization"] == "off"
        and normalized["video_denoise"] == "off"
        and normalized["audio_cleanup"] == "off"
    )
