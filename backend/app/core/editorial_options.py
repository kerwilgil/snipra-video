"""Contrato de opciones editoriales avanzadas por trabajo."""

from __future__ import annotations

TRANSCRIPTION_QUALITIES = frozenset({"default", "fast", "balanced", "precise"})

DEFAULT_EDITORIAL_OPTIONS: dict[str, bool | str] = {
    "compact_editing": False,
    "semantic_cards": False,
    "privacy_assist": False,
    "protected_zones": True,
    "overlay_qa": True,
    "transcription_quality": "default",
}


def normalize_editorial_options(value: dict | None) -> dict[str, bool | str]:
    """Devuelve un shape estable y descarta claves desconocidas."""
    source = value if isinstance(value, dict) else {}
    quality = str(source.get("transcription_quality") or "default").strip().lower()
    if quality not in TRANSCRIPTION_QUALITIES:
        quality = "default"
    return {
        "compact_editing": bool(source.get("compact_editing", False)),
        "semantic_cards": bool(source.get("semantic_cards", False)),
        "privacy_assist": bool(source.get("privacy_assist", False)),
        "protected_zones": bool(source.get("protected_zones", True)),
        "overlay_qa": bool(source.get("overlay_qa", True)),
        "transcription_quality": quality,
    }


def whisper_model_for_quality(quality: str, configured_model: str) -> str:
    """Traduce el perfil legible a un modelo; default respeta Ajustes."""
    return {
        "fast": "base",
        "balanced": "small",
        "precise": "medium",
    }.get((quality or "default").strip().lower(), configured_model)
