"""Funciones editoriales nativas y verificables para clips de Snipra."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from app.pipeline.reedit import compute_kept_ranges
from app.providers.base import ClipCandidate, EditorialCard, ScoreContribution, SourceRange

_WORD_RE = re.compile(r"[a-z0-9áéíóúüñ]+", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SENSITIVE_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.IGNORECASE),
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"(?:api[ _-]?key|token|secret|password|contraseña)\s*[:=]", re.IGNORECASE),
    re.compile(r"(?:[a-z]:\\|/(?:users|home)/)\S+", re.IGNORECASE),
    re.compile(r"\b(?:sk|pk|ghp|xoxb)-[a-z0-9_-]{12,}\b", re.IGNORECASE),
)


def _normal(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(_WORD_RE.findall(ascii_text.lower()))


def _clip_segments(segments: list[dict], start: float, end: float) -> list[dict]:
    return [
        segment
        for segment in segments
        if float(segment.get("end", 0.0)) > start and float(segment.get("start", 0.0)) < end
    ]


def _retake_ranges(segments: list[dict], start: float, end: float) -> list[tuple[float, float]]:
    """Quita la primera de dos tomas consecutivas casi iguales; conserva la última."""
    clipped = _clip_segments(segments, start, end)
    remove: list[tuple[float, float]] = []
    for index, left in enumerate(clipped[:-1]):
        left_text = _normal(str(left.get("text", "")))
        if len(left_text) < 18:
            continue
        for right in clipped[index + 1 : index + 4]:
            gap = float(right.get("start", 0.0)) - float(left.get("end", 0.0))
            if gap > 4.0:
                break
            right_text = _normal(str(right.get("text", "")))
            if len(right_text) < 18:
                continue
            if SequenceMatcher(None, left_text, right_text).ratio() >= 0.82:
                remove.append(
                    (
                        max(start, float(left.get("start", start))),
                        min(end, float(left.get("end", start))),
                    )
                )
                break
    return remove


def compact_candidate(
    candidate: ClipCandidate,
    segments: list[dict],
    silence_intervals: list[list[float] | tuple[float, float]],
) -> ClipCandidate:
    """Recorta pausas internas y primeras tomas repetidas con límites conservadores."""
    if candidate.source_ranges or candidate.duration < 10.0:
        return candidate
    remove: list[tuple[float, float]] = []
    for value in silence_intervals:
        if len(value) < 2:
            continue
        silence_start, silence_end = float(value[0]), float(value[1])
        if silence_end - silence_start < 0.55:
            continue
        # Conserva respiración alrededor del corte para que no suene mecánico.
        cut_start = max(candidate.start + 0.5, silence_start + 0.18)
        cut_end = min(candidate.end - 0.5, silence_end - 0.18)
        if cut_end - cut_start >= 0.18:
            remove.append((cut_start, cut_end))
    remove.extend(_retake_ranges(segments, candidate.start, candidate.end))
    kept = compute_kept_ranges(candidate.start, candidate.end, remove)
    kept_duration = sum(end - start for start, end in kept)
    if len(kept) <= 1 or kept_duration < max(8.0, candidate.duration * 0.60):
        return candidate
    note = "edición compacta"
    return candidate.model_copy(
        update={
            "source_ranges": [
                SourceRange(start=start, end=end, kind="compact", score=candidate.score)
                for start, end in kept
            ],
            "reason": f"{candidate.reason} · {note}".strip(" ·"),
            "score_breakdown": [
                *candidate.score_breakdown,
                ScoreContribution(
                    signal="edicion_compacta",
                    stage="editorial.compact_candidate",
                    score_before=candidate.score,
                    contribution=0.0,
                    score_after=candidate.score,
                    note=note,
                ),
            ],
        }
    )


def sanitize_cards(candidate: ClipCandidate, segments: list[dict]) -> list[EditorialCard]:
    """Acepta sólo tarjetas sustentadas y sin cifras ajenas al transcript."""
    transcript = " ".join(
        str(segment.get("text", ""))
        for segment in _clip_segments(segments, candidate.start, candidate.end)
    )
    transcript_norm = _normal(transcript)
    source_numbers = set(_NUMBER_RE.findall(transcript))
    accepted: list[EditorialCard] = []
    for card in candidate.editorial_cards[:3]:
        evidence = _normal(card.evidence)
        card_text = _normal(" ".join((card.kicker, card.text, card.secondary)))
        if not evidence or evidence not in transcript_norm:
            continue
        if not card_text or card.at < candidate.start or card.at > candidate.end:
            continue
        if set(_NUMBER_RE.findall(card.text + " " + card.secondary)) - source_numbers:
            continue
        # Si es prácticamente una repetición del subtítulo no aporta valor editorial.
        if SequenceMatcher(None, card_text, evidence).ratio() > 0.88:
            continue
        accepted.append(card)
    return accepted


def source_time_to_output(source_time: float, ranges: list[SourceRange]) -> float | None:
    if not ranges:
        return source_time
    elapsed = 0.0
    for source_range in ranges:
        if source_range.start <= source_time <= source_range.end:
            return elapsed + source_time - source_range.start
        elapsed += source_range.duration
    return None


def remap_cards(cards: list[EditorialCard], ranges: list[SourceRange], offset: float) -> list[dict]:
    out: list[dict] = []
    for card in cards:
        mapped = source_time_to_output(card.at, ranges)
        if mapped is None:
            continue
        payload = card.model_dump()
        payload["at"] = round(max(0.0, mapped - (0.0 if ranges else offset)), 3)
        out.append(payload)
    return out


def ass_time(seconds: float) -> str:
    value = max(0.0, seconds)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _ass_escape(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_cards_ass(cards: list[dict], width: int, height: int, position: str = "top") -> str:
    """Genera tarjetas con ASS; sin runtime web ni dependencia de Remotion."""
    alignment = 8 if position == "top" else 2
    margin_v = int(height * (0.10 if position == "top" else 0.20))
    font_size = max(30, int(height * 0.036))
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
        "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,"
        "Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        (
            f"Style: Card,Arial,{font_size},&H00FFFFFF,&H0000FFFF,&HCC111827,"
            f"&HCC111827,-1,0,0,0,100,100,0,0,3,2,0,{alignment},"
            f"{int(width * 0.08)},{int(width * 0.08)},{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for card in cards:
        start = float(card.get("at", 0.0))
        end = start + max(2.0, min(6.0, float(card.get("duration", 4.0))))
        kicker = _ass_escape(str(card.get("kicker", "")).upper())
        main = _ass_escape(str(card.get("text", "")))
        secondary = _ass_escape(str(card.get("secondary", "")))
        text = "\\N".join(
            part
            for part in (
                f"{{\\fs{int(font_size * 0.62)}\\c&H58E7FF&}}{kicker}",
                f"{{\\fs{font_size}\\c&HFFFFFF&}}{main}",
                f"{{\\fs{int(font_size * 0.62)}\\c&HD1D5DB&}}{secondary}",
            )
            if part.strip("{}")
        )
        lines.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Card,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def protected_zones(
    width: int,
    height: int,
    *,
    captions: bool,
    subject: bool,
    profile: object | None = None,
) -> list[dict]:
    """Zonas protegidas del frame.

    Con `profile` (PlatformProfile) la zona de subtítulos respeta el margen
    seguro inferior del destino (single source of truth de safe zones).
    Sin profile se mantiene el comportamiento histórico (banda 62%-90%).
    """
    zones: list[dict] = []
    if captions:
        band_height = int(height * 0.28)
        safe_bottom = float(getattr(profile, "safe_bottom", 0.10))
        zones.append(
            {
                "kind": "captions",
                "x": 0,
                "y": max(0, int(height * (1.0 - safe_bottom - 0.28))),
                "width": width,
                "height": band_height,
            }
        )
    if subject:
        zones.append(
            {
                "kind": "subject",
                "x": int(width * 0.22),
                "y": int(height * 0.12),
                "width": int(width * 0.56),
                "height": int(height * 0.55),
            }
        )
    return zones


def card_position(zones: list[dict], height: int) -> str:
    top_occupied = any(zone["y"] < height * 0.35 for zone in zones if zone["kind"] != "subject")
    return "bottom" if top_occupied else "top"


def privacy_suggestions(regions: list[dict], *, at: float, window: float = 2.0) -> list[dict]:
    suggestions: list[dict] = []
    for region in regions:
        text = str(region.get("text", ""))
        if not any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS):
            continue
        suggestions.append(
            {
                "text": text[:120],
                "confidence": float(region.get("confidence", 0.0)),
                "x": int(region.get("x", 0)),
                "y": int(region.get("y", 0)),
                "width": int(region.get("width", 0)),
                "height": int(region.get("height", 0)),
                "start": round(max(0.0, at - window / 2), 3),
                "end": round(at + window / 2, 3),
                "confirmed": False,
            }
        )
    return suggestions


def overlay_qa(cards: list[dict], masks: list[dict], duration: float) -> dict:
    issues: list[str] = []
    for overlay in [*cards, *masks]:
        start = float(overlay.get("at", overlay.get("start", 0.0)))
        end = float(overlay.get("end", start + float(overlay.get("duration", 0.0))))
        if start < 0 or end <= start or end > duration + 0.25:
            issues.append(f"overlay fuera de límites ({start:.2f}-{end:.2f}s)")
    return {"passed": not issues, "issues": issues, "checked_overlays": len(cards) + len(masks)}
