"""Caption rewrite helpers for social subtitles."""

from __future__ import annotations

import json
import re

from app.providers.registry import get_detector

SYSTEM_PROMPT = (
    "Eres editor de captions para clips sociales. Reescribe subtitulos para que "
    "se lean naturales, cortos y con gancho, sin cambiar el idioma ni inventar "
    "datos. Responde solo JSON."
)

TRANSLATE_SYSTEM_PROMPT = (
    "Eres traductor profesional de subtitulos para clips sociales. Traduces "
    "cada caption al idioma pedido manteniendo el sentido exacto, sin agregar "
    "ni quitar informacion. Responde solo JSON."
)


class TranslationError(RuntimeError):
    """Se lanza cuando no fue posible traducir los subtitulos con IA.

    A diferencia de rewrite_subtitle_blocks, no existe traductor local
    offline confiable en este proyecto: si el proveedor es naive/vacio o el
    LLM falla, se levanta este error en vez de fingir una traduccion falsa.
    """


def rewrite_subtitle_blocks(
    blocks: list[dict],
    *,
    provider: str,
    model: str | None,
) -> list[dict]:
    if not blocks:
        return []

    original = [str(block.get("text", "")).strip() for block in blocks]
    fallback = [_local_rewrite(text) for text in original]
    if provider in {"", "naive"}:
        return _apply_rewrites(blocks, fallback)

    try:
        detector = get_detector(provider, model)
        complete = getattr(detector, "_complete", None)
        if complete is None:
            return _apply_rewrites(blocks, fallback)
        payload = {
            "rules": [
                "Devuelve un array JSON de strings, mismo largo y mismo orden.",
                "Cada string debe tener maximo 10 palabras.",
                "No agregues hashtags, emojis ni explicaciones.",
                "Mantiene el sentido exacto; mejora ritmo y claridad.",
            ],
            "captions": original,
        }
        raw = complete(SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        rewritten = _parse_rewrite_response(raw, len(original))
        return _apply_rewrites(blocks, rewritten or fallback)
    except Exception:  # noqa: BLE001
        return _apply_rewrites(blocks, fallback)


def translate_subtitle_blocks(
    blocks: list[dict],
    *,
    target_language: str,
    provider: str,
    model: str | None,
) -> list[dict]:
    if not blocks:
        return []

    target = (target_language or "").strip()
    if not target:
        raise TranslationError("Debes indicar el idioma destino para traducir")

    if provider in {"", "naive"}:
        raise TranslationError(
            "No hay traductor local disponible: elige un proveedor de IA para traducir"
        )

    original = [str(block.get("text", "")).strip() for block in blocks]

    try:
        detector = get_detector(provider, model)
        complete = getattr(detector, "_complete", None)
        if complete is None:
            raise TranslationError(f"El proveedor '{provider}' no soporta traduccion por IA")
        payload = {
            "rules": [
                f"Traduce cada caption al idioma: {target}.",
                "Devuelve un array JSON de strings, mismo largo y mismo orden.",
                "No agregues hashtags, emojis ni explicaciones.",
                "Mantiene el sentido exacto; no inventes contenido nuevo.",
            ],
            "captions": original,
        }
        raw = complete(TRANSLATE_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
    except TranslationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranslationError(f"Fallo la traduccion con IA: {exc}") from exc

    translated = _parse_rewrite_response(raw, len(original))
    if not translated:
        raise TranslationError("La IA no devolvio una traduccion valida")

    return _apply_rewrites(blocks, translated)


def _parse_rewrite_response(raw: str, expected: int) -> list[str] | None:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != expected:
        return None
    out = [str(item).strip() for item in data]
    if any(not item for item in out):
        return None
    return out


def _local_rewrite(text: str) -> str:
    value = " ".join(text.split())
    replacements = {
        " o sea ": " ",
        " este ": " ",
        " pues ": " ",
        " eh ": " ",
        " ehh ": " ",
        " emm ": " ",
        " como que ": " ",
    }
    lowered = f" {value.lower()} "
    for src, dst in replacements.items():
        lowered = lowered.replace(src, dst)
    cleaned = " ".join(lowered.split())
    if not cleaned:
        return value
    words = cleaned.split()
    if len(words) > 10:
        cleaned = " ".join(words[:10])
    return cleaned[:1].upper() + cleaned[1:]


def _apply_rewrites(blocks: list[dict], rewrites: list[str]) -> list[dict]:
    out: list[dict] = []
    for block, text in zip(blocks, rewrites, strict=False):
        next_block = dict(block)
        start = float(next_block.get("start", 0.0))
        end = max(start + 0.3, float(next_block.get("end", start + 0.8)))
        next_block["text"] = text
        next_block["words"] = _timed_words(text, start, end)
        out.append(next_block)
    return out


def _timed_words(text: str, start: float, end: float) -> list[dict]:
    parts = [part for part in text.split() if part]
    if not parts:
        return []
    step = (end - start) / len(parts)
    return [
        {
            "word": part,
            "start": round(start + idx * step, 3),
            "end": round(start + (idx + 1) * step, 3),
        }
        for idx, part in enumerate(parts)
    ]
