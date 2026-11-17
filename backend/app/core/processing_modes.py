"""Perfiles de procesamiento para ajustar el pipeline por tipo de video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingMode:
    value: str
    label: str
    description: str


PROCESSING_MODES: tuple[ProcessingMode, ...] = (
    ProcessingMode(
        "auto_smart",
        "Auto",
        "El LLM detecta el tipo de contenido del transcript y elige el enfoque más adecuado.",
    ),
    ProcessingMode(
        "podcast_viral",
        "Podcast / Entrevista",
        "2 o más personas, debate, entrevista, mesa redonda. "
        "Prioriza ideas virales y el hablante activo.",
    ),
    ProcessingMode(
        "vlog",
        "Vlog / Directo a cámara",
        "Una persona hablando, estilo personal y cotidiano. "
        "Busca anécdotas, humor y momentos de conexión.",
    ),
    ProcessingMode(
        "gaming",
        "Gaming",
        "Gameplay, kills, reacciones en tiempo real. "
        "Prioriza acción, tensión y victorias o derrotas.",
    ),
    ProcessingMode(
        "gaming_auto",
        "Gaming Automatico",
        "Decide entre clips independientes, un hook o un montaje de highlights.",
    ),
    ProcessingMode(
        "gaming_hook",
        "Gaming Hook",
        "Un edit de 8-25 s con accion inmediata, desenlace y repeticion si existe.",
    ),
    ProcessingMode(
        "gaming_highlights",
        "Gaming Highlights",
        "Un montaje de 30-45 s con 4-8 jugadas breves, sin tiempos muertos.",
    ),
    ProcessingMode(
        "tutorial_screen",
        "Tutorial / Educativo",
        "Instrucciones paso a paso o contenido con pantalla. Protege texto y UI visible.",
    ),
    ProcessingMode(
        "music",
        "Música / Ambiental",
        "Sin diálogo principal o con música como protagonista. "
        "Corta por escena o energía, sin subtítulos.",
    ),
    ProcessingMode(
        "keep_original",
        "Técnico: conservar encuadre",
        "Respeta el cuadro completo siempre. "
        "Útil para contenido con texto en bordes o composición fija.",
    ),
)

PROCESSING_MODE_VALUES = {mode.value for mode in PROCESSING_MODES}
HIDDEN_PROCESSING_MODES = {"gaming_auto", "gaming_hook", "gaming_highlights"}
PRESERVE_FRAME_MODES = {"tutorial_screen", "keep_original"}
PODCAST_VIRAL_MODES = {"auto_smart", "podcast_viral"}
VLOG_MODES = {"vlog"}
# Modos que por defecto usan face tracking (persona a cámara, sin pantallas)
FACE_TRACK_MODES = PODCAST_VIRAL_MODES | VLOG_MODES
MUSIC_MODES = {"music"}
GAMING_MODES = {"gaming", "gaming_auto", "gaming_hook", "gaming_highlights"}
GAMING_EDIT_MODES = {"gaming_auto", "gaming_hook", "gaming_highlights"}
PROCESSING_MODE_ALIASES = {
    "auto": "auto_smart",
    "podcast": "podcast_viral",
    "entrevista": "podcast_viral",
    "vertical": "podcast_viral",
    "vlogs": "vlog",
    "gaming": "gaming",
    "gaming auto": "gaming_auto",
    "gaming hook": "gaming_hook",
    "gaming highlights": "gaming_highlights",
    "tutorial": "tutorial_screen",
    "educativo": "tutorial_screen",
    "original": "keep_original",
    "musica": "music",
    "musical": "music",
    "instrumental": "music",
}


def normalize_processing_mode(value: str | None) -> str:
    mode = (value or "auto_smart").strip().lower()
    mode = PROCESSING_MODE_ALIASES.get(mode, mode)
    if mode not in PROCESSING_MODE_VALUES:
        raise ValueError(f"Modo de procesamiento no soportado: {value}")
    return mode


def processing_modes_payload() -> list[dict[str, str]]:
    return [
        {"value": mode.value, "label": mode.label, "description": mode.description}
        for mode in PROCESSING_MODES
        if mode.value not in HIDDEN_PROCESSING_MODES
    ]


def reframe_request_for_processing_mode(processing_mode: str | None, configured: str) -> str:
    try:
        mode = normalize_processing_mode(processing_mode)
    except ValueError:
        mode = "auto"

    # tutorial/keep_original siempre fuerzan preserve (el contenido lo requiere).
    if mode in PRESERVE_FRAME_MODES:
        return "preserve"

    # Modos con persona a cámara: si el usuario no eligió explícitamente, usar track.
    explicit = (configured or "").strip().lower()
    if mode in FACE_TRACK_MODES:
        return explicit if (explicit and explicit != "auto") else "track"

    return explicit or "auto"


def is_podcast_viral_mode(processing_mode: str | None) -> bool:
    try:
        return normalize_processing_mode(processing_mode) in PODCAST_VIRAL_MODES
    except ValueError:
        return False


def is_music_mode(processing_mode: str | None) -> bool:
    try:
        return normalize_processing_mode(processing_mode) in MUSIC_MODES
    except ValueError:
        return False


def is_gaming_mode(processing_mode: str | None) -> bool:
    try:
        return normalize_processing_mode(processing_mode) in GAMING_MODES
    except ValueError:
        return False


def is_gaming_edit_mode(processing_mode: str | None) -> bool:
    try:
        return normalize_processing_mode(processing_mode) in GAMING_EDIT_MODES
    except ValueError:
        return False


PODCAST_VIRAL_CONFIG = {
    "smart_crop": True,
    "face_tracking": True,
    "speaker_priority": "high",
    "keep_microphone_visible": True,
    "target_speaker_height_ratio": 0.68,
    "min_speaker_height_ratio": 0.60,
    "max_speaker_height_ratio": 0.75,
    "max_headroom_ratio": 0.10,
    "dynamic_zoom": True,
    "zoom_intensity": "soft",
    "zoom_min": 1.05,
    "zoom_max": 1.18,
    "min_seconds_between_zooms": 3.0,
}
