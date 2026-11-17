"""SponsorBlock: fetch de tramos sponsor/auto-promo de YouTube y filtrado de candidatos.

Todo el fetch de red es best-effort: cualquier error (timeout, red caída, respuesta
inesperada) devuelve una lista vacía en vez de romper el job (mismo criterio que el
resto del pipeline al degradar a fallbacks locales).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import parse_qs, urlparse

import httpx

logger = logging.getLogger(__name__)

SPONSORBLOCK_API_URL = "https://sponsor.ajay.app/api/skipSegments"
DEFAULT_CATEGORIES = ("sponsor", "selfpromo", "interaction", "preview")
# fracción máxima de la duración del candidato que puede solaparse con un tramo
# sponsor antes de descartarlo directamente en vez de intentar recortarlo.
_DISCARD_OVERLAP_RATIO = 0.4

_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com"}
_VIDEO_ID_RE = re.compile(r"^[\w-]{11}$")


def extract_youtube_video_id(url: str) -> str | None:
    """Extrae el ID de video de una URL de YouTube. None si no es de YouTube."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host not in _YOUTUBE_HOSTS:
        return None

    video_id: str | None = None
    if host == "youtu.be":
        parts = [p for p in parsed.path.split("/") if p]
        video_id = parts[0] if parts else None
    elif parsed.path == "/watch":
        video_id = (parse_qs(parsed.query).get("v") or [None])[0]
    elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
        parts = [p for p in parsed.path.split("/") if p]
        video_id = parts[1] if len(parts) > 1 else None

    if not video_id or not _VIDEO_ID_RE.match(video_id):
        return None
    return video_id


def fetch_sponsor_segments(
    video_id: str,
    categories: list[str] | None = None,
    *,
    timeout: float = 5.0,
) -> list[dict]:
    """Consulta la API pública de SponsorBlock. Best-effort: nunca lanza excepciones."""
    cats = [c for c in (categories or DEFAULT_CATEGORIES) if c]
    if not cats:
        return []
    params = [("videoID", video_id), *[("category", c) for c in cats]]
    try:
        response = httpx.get(SPONSORBLOCK_API_URL, params=params, timeout=timeout)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - fail-open: cualquier error de red/parseo
        logger.warning("SponsorBlock: fetch falló para %s (%s)", video_id, exc)
        return []

    if not isinstance(data, list):
        return []

    segments: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        segment = item.get("segment")
        if not isinstance(segment, list | tuple) or len(segment) != 2:
            continue
        try:
            start, end = float(segment[0]), float(segment[1])
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "category": str(item.get("category", "")),
            }
        )
    return segments


def fetch_segments_for_url(url: str, categories: list[str] | None = None) -> list[dict]:
    """Atajo: extrae el video ID de la URL y consulta SponsorBlock. [] si no es YouTube."""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return []
    return fetch_sponsor_segments(video_id, categories)


def _merge_segments(segments: list[dict]) -> list[tuple[float, float]]:
    """Une tramos sponsor solapados/adyacentes en intervalos (start, end) ordenados."""
    intervals = sorted(
        (
            (float(s["start"]), float(s["end"]))
            for s in segments
            if s.get("start") is not None and s.get("end") is not None and s["end"] > s["start"]
        ),
        key=lambda pair: pair[0],
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _trim_or_drop(candidate, merged: list[tuple[float, float]], min_seconds: float):
    """Recorta o descarta un candidato según su solape con tramos sponsor fusionados."""
    start, end = candidate.start, candidate.end
    duration = end - start
    if duration <= 0:
        return candidate

    overlaps = sorted(
        (max(start, s), min(end, e)) for s, e in merged if max(start, s) < min(end, e)
    )
    if not overlaps:
        return candidate

    total_overlap = sum(o_end - o_start for o_start, o_end in overlaps)
    if (total_overlap / duration) > _DISCARD_OVERLAP_RATIO:
        return None

    new_start, new_end = start, end
    for o_start, o_end in overlaps:
        if o_start <= new_start + 1e-6:
            new_start = max(new_start, o_end)
        elif o_end >= new_end - 1e-6:
            new_end = min(new_end, o_start)

    if new_end - new_start < min_seconds * 0.85:
        return None
    if new_start == start and new_end == end:
        return candidate
    return candidate.model_copy(update={"start": round(new_start, 3), "end": round(new_end, 3)})


def filter_candidates_by_sponsor_segments(
    candidates: list,
    sponsor_segments: list[dict] | None,
    min_seconds: float,
) -> list:
    """Descarta/recorta candidatos que caen dentro de tramos sponsor.

    Función pura: no hace red ni logging, fácil de testear. El caller decide qué
    hacer si el resultado queda vacío (ver detect.py: nunca se debe romper el job
    por este filtro).
    """
    if not sponsor_segments or not candidates:
        return candidates
    merged = _merge_segments(sponsor_segments)
    if not merged:
        return candidates

    out = []
    for candidate in candidates:
        trimmed = _trim_or_drop(candidate, merged, min_seconds)
        if trimmed is not None:
            out.append(trimmed)
    return out
