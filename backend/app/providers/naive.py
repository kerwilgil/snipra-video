"""Detector local sin IA basado en ventanas y señales de hook.

Sirve como fallback offline y como red de seguridad cuando el proveedor LLM falla. No intenta
entender el video como un modelo multimodal, pero evita cortes uniformes: puntua ventanas por
frases de gancho, numeros, preguntas, intensidad y senales por modo de procesamiento.
"""

from __future__ import annotations

import re

from app.providers.base import ClipCandidate, ClipDetector, DetectParams, overlap_ratio
from app.providers.capabilities import get_provider_capabilities
from app.providers.structured_completion import (
    CompletionResult,
    StructuredCompletionAdapter,
)

HOOK_TERMS = {
    "no vas a creer",
    "nadie",
    "nunca",
    "siempre",
    "error",
    "errores",
    "secreto",
    "clave",
    "truco",
    "verdad",
    "problema",
    "solucion",
    "cambio",
    "mejor",
    "peor",
    "importante",
    "atencion",
    "cuidado",
    "increible",
    "brutal",
    "facil",
    "rapido",
    "gratis",
}

MODE_TERMS = {
    "auto_smart": {"mira", "rapido", "facil", "antes", "despues", "resultado", "historia"},
    "podcast_viral": {
        "pienso",
        "historia",
        "aprendi",
        "experiencia",
        "personas",
        "mentalidad",
    },
    "gaming": {"kill", "mato", "gane", "perdi", "boss", "nivel", "partida", "enemigo", "jugada"},
    "gaming_hook": {"kill", "mato", "gane", "victoria", "ultimo", "enemigo", "jugada"},
    "gaming_auto": {"kill", "mato", "gane", "victoria", "ultimo", "enemigo", "jugada"},
    "gaming_highlights": {
        "kill",
        "mato",
        "gane",
        "victoria",
        "enemigo",
        "jugada",
    },
    "tutorial_screen": {
        "paso",
        "primero",
        "segundo",
        "configura",
        "haz clic",
        "error",
        "solucion",
        "pantalla",
    },
    "keep_original": set(),
}


class NaiveDetector(ClipDetector, StructuredCompletionAdapter):
    name = "naive"

    def __init__(self):
        self._caps = get_provider_capabilities("naive", "")

    @property
    def capabilities(self):
        return self._caps

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        """Naive detector does not support structured completion."""
        raise RuntimeError("NaiveDetector does not support structured completion")

    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        clean_segments = [s for s in segments if (s.get("text") or "").strip()]
        if not clean_segments:
            return []

        windows = self._build_windows(clean_segments, params)
        if not windows:
            return []

        scored = [self._make(window, idx, params) for idx, window in enumerate(windows)]
        scored.sort(key=lambda c: (c.score, -c.start), reverse=True)

        selected: list[ClipCandidate] = []
        for cand in scored:
            if any(overlap_ratio(cand, prev) > 0.45 for prev in selected):
                continue
            selected.append(cand)
            if len(selected) >= params.num_clips:
                break

        return selected

    @staticmethod
    def _build_windows(segments: list[dict], params: DetectParams) -> list[list[dict]]:
        windows: list[list[dict]] = []
        start_idx = 0
        while start_idx < len(segments):
            win: list[dict] = []
            for seg in segments[start_idx:]:
                win.append(seg)
                duration = win[-1]["end"] - win[0]["start"]
                if duration >= params.min_seconds and _ends_on_sentence(seg.get("text", "")):
                    break
                if duration >= min(params.max_seconds, max(params.min_seconds, 34.0)):
                    break
            if win and win[-1]["end"] - win[0]["start"] >= params.min_seconds * 0.6:
                windows.append(win)
            step = max(1, len(win) // 2)
            start_idx += step
        return windows

    @staticmethod
    def _make(window: list[dict], idx: int, params: DetectParams) -> ClipCandidate:
        text = " ".join((s.get("text") or "").strip() for s in window if s.get("text")).strip()
        start = round(float(window[0]["start"]), 3)
        end = round(float(window[-1]["end"]), 3)
        score, signals = _score_text(text, params.processing_mode)
        title = _title_from_text(text, idx)
        hook = _hook_from_text(text, title)
        reason_detail = ", ".join(signals[:3]) if signals else "ventana clara del transcript"
        reason = f"Fallback local: {reason_detail}"
        return ClipCandidate(
            start=start,
            end=end,
            title=title,
            hook=hook,
            score=score,
            reason=reason,
            hashtags=_hashtags_from_text(text, params.processing_mode),
            description=_description_from_text(hook, title),
        )


GENERIC_HASHTAGS = ["#shorts", "#viral", "#fyp", "#parati"]
MODE_HASHTAGS = {
    "auto_smart": "#shorts",
    "podcast_viral": "#podcast",
    "gaming": "#gaming",
    "gaming_auto": "#gaming",
    "gaming_hook": "#gaming",
    "gaming_highlights": "#gaming",
    "tutorial_screen": "#tutorial",
}
TOPIC_HASHTAGS = ("dinero", "exito", "secreto", "error", "motivacion", "negocio", "clave")


def _hashtags_from_text(text: str, processing_mode: str) -> list[str]:
    lower = _normalize(text)
    tags: list[str] = []
    mode_tag = MODE_HASHTAGS.get(processing_mode)
    if mode_tag:
        tags.append(mode_tag)
    for term in TOPIC_HASHTAGS:
        if term in lower and f"#{term}" not in tags:
            tags.append(f"#{term}")
    for generic in GENERIC_HASHTAGS:
        if len(tags) >= 5:
            break
        if generic not in tags:
            tags.append(generic)
    return tags[:5]


def _description_from_text(hook: str, title: str) -> str:
    base = (hook or title).strip()
    if not base:
        return ""
    if not base.endswith((".", "!", "?", "…")):
        base += "."
    return base


def _score_text(text: str, processing_mode: str) -> tuple[float, list[str]]:
    lower = _normalize(text)
    signals: list[str] = []
    score = 0.42

    hook_hits = sum(1 for term in HOOK_TERMS if term in lower)
    if hook_hits:
        score += min(0.22, hook_hits * 0.055)
        signals.append("palabras de gancho")

    if re.search(r"\b\d+([.,]\d+)?\b|%|x\b", lower):
        score += 0.08
        signals.append("dato numerico")

    if "?" in text or "¿" in text:
        score += 0.07
        signals.append("pregunta")

    if "!" in text or "¡" in text:
        score += 0.06
        signals.append("intensidad")

    terms = MODE_TERMS.get(processing_mode, set()) | MODE_TERMS.get("auto_smart", set())
    mode_hits = sum(1 for term in terms if term in lower)
    if mode_hits:
        score += min(0.18, mode_hits * 0.045)
        signals.append(f"senal {processing_mode}")

    word_count = len(lower.split())
    if 45 <= word_count <= 170:
        score += 0.05
        signals.append("duracion conversacional")

    return round(min(0.96, score), 3), signals


def _normalize(text: str) -> str:
    replacements = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    return text.translate(replacements).lower()


def _title_from_text(text: str, idx: int) -> str:
    sentence = _first_sentence(text)
    if not sentence:
        return f"Clip {idx + 1}"
    return sentence[:68].rstrip(" ,.;:") + ("..." if len(sentence) > 68 else "")


def _hook_from_text(text: str, fallback: str) -> str:
    for sentence in _sentences(text):
        normalized = _normalize(sentence)
        if any(term in normalized for term in HOOK_TERMS) or "?" in sentence or "¿" in sentence:
            return sentence[:120].strip()
    return fallback


def _first_sentence(text: str) -> str:
    sentences = _sentences(text)
    return sentences[0] if sentences else text.strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _ends_on_sentence(text: str) -> bool:
    return text.strip().endswith((".", "?", "!", "…"))


