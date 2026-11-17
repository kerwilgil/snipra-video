"""Contrato común de la capa de detección de momentos (multi-proveedor)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol

from pydantic import BaseModel, Field

DURATION_PRESETS: dict[str, dict] = {
    "social": {
        "min": 30.0,
        "max": 60.0,
        "label": "Corto (30-60s)",
        "description": "Ideal para TikTok/Reels/Shorts rápido.",
    },
    "standard": {
        "min": 45.0,
        "max": 90.0,
        "label": "Estándar (45-90s)",
        "description": "TikTok moderno — el más viral. Recomendado.",
    },
    "long": {
        "min": 60.0,
        "max": 120.0,
        "label": "Largo (60-120s)",
        "description": "YouTube Shorts largo o contenido en profundidad.",
    },
}


class SourceRange(BaseModel):
    """Tramo de la fuente que forma parte del edit final."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    kind: str = "moment"
    score: float = Field(default=0.5, ge=0, le=1)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class EditorialCard(BaseModel):
    """Tarjeta breve respaldada por una frase concreta del transcript."""

    kind: Literal["contrast", "consequence", "derived"]
    at: float = Field(ge=0)
    duration: float = Field(default=4.0, ge=2.0, le=6.0)
    kicker: str = ""
    text: str
    secondary: str = ""
    evidence: str = ""


class ScoreContribution(BaseModel):
    """Un aporte de una señal al score de un candidato (observabilidad, P0.1).

    Aditivo: no cambia ninguna decisión de scoring/selección, solo la explica.
    `reason` (texto libre) sigue existiendo intacto para compatibilidad; esto
    es una capa estructurada paralela para poder auditar "qué señal aportó
    qué puntaje" sin parsear texto en español.
    """

    signal: str
    stage: str = ""
    score_before: float = 0.0
    contribution: float = 0.0
    score_after: float = 0.0
    at: float | None = None
    note: str = ""
    metadata: dict = Field(default_factory=dict)


class ClipCandidate(BaseModel):
    """Un momento detectado como candidato a clip."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    title: str = ""
    hook: str = ""
    score: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""
    hashtags: list[str] = Field(default_factory=list)
    description: str = ""
    source_ranges: list[SourceRange] = Field(default_factory=list)
    editorial_cards: list[EditorialCard] = Field(default_factory=list)
    score_breakdown: list[ScoreContribution] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        if self.source_ranges:
            return sum(item.duration for item in self.source_ranges)
        return max(0.0, self.end - self.start)


class DetectParams(BaseModel):
    num_clips: int = 5
    min_seconds: float = 8.0
    max_seconds: float = 90.0
    formats: list[str] = ["9x16"]
    language: str = ""
    processing_mode: str = "auto"
    semantic_cards: bool = False


class ClipDetector(ABC):
    """Interfaz que implementan todos los proveedores (naive, claude, openai, nvidia)."""

    name: str = "base"

    @abstractmethod
    def detect(self, segments: list[dict], params: DetectParams) -> list[ClipCandidate]:
        """Recibe los segmentos del transcript y devuelve candidatos a clip."""
        raise NotImplementedError


def clamp_candidates(
    candidates: list[ClipCandidate],
    params: DetectParams,
    total_duration: float,
) -> list[ClipCandidate]:
    """Sanea candidatos: límites válidos, dentro del video, duración aceptable, ordenados."""
    out: list[ClipCandidate] = []
    for c in candidates:
        start = max(0.0, min(c.start, total_duration))
        end = max(0.0, min(c.end, total_duration))
        if end <= start:
            continue
        dur = end - start
        if dur < params.min_seconds * 0.85:
            continue
        if dur > params.max_seconds:
            end = start + params.max_seconds
        out.append(c.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
    out.sort(key=lambda x: x.score, reverse=True)
    return out[: params.num_clips]


class _TimeSpan(Protocol):
    """Cualquier objeto con start/end/duration (ClipCandidate o SourceRange)."""

    start: float
    end: float

    @property
    def duration(self) -> float: ...


def overlap_ratio(a: _TimeSpan, b: _TimeSpan) -> float:
    """Fracción de solape relativa a la duración del tramo más corto.

    Consolida las implementaciones que antes vivían duplicadas en
    providers/naive.py (_overlap_ratio, para ClipCandidate) y en
    pipeline/gaming_edits.py (_range_overlap, para SourceRange) — misma
    fórmula letra por letra en ambas. No reemplaza la deduplicación
    geométrica de pipeline/signals.py::action_windows, que rechaza
    cualquier solape sin fracción tolerada y es semánticamente distinta.
    """
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    shortest = max(0.001, min(a.duration, b.duration))
    return overlap / shortest


def dedupe_by_overlap(candidates: list, *, ratio: float) -> list:
    """Ordena por score descendente y descarta solapes > ratio con lo ya elegido.

    Reemplaza pipeline/gaming_edits.py::_dedupe, mismo criterio exacto.
    """
    selected: list = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(overlap_ratio(candidate, previous) > ratio for previous in selected):
            continue
        selected.append(candidate)
    return selected


_SENTENCE_END = frozenset(".!?")


def snap_to_segment_boundaries(
    candidates: list[ClipCandidate],
    segments: list[dict],
    params: DetectParams,
    total_duration: float,
) -> list[ClipCandidate]:
    """Ajusta end de cada candidato al cierre de oración más cercano.

    Prioridad: fin de segmento cuyo texto termina en . ! ?
    Fallback: límite de segmento Whisper más cercano al end elegido.
    Esto evita que el clip termine a mitad de frase (p. ej. "OFRECES," en vez de
    esperar al "." al final de la idea completa).
    """
    if not segments:
        return candidates

    segs = sorted(
        [s for s in segments if s.get("start") is not None and s.get("end") is not None],
        key=lambda s: float(s["start"]),
    )
    seg_starts = [float(s["start"]) for s in segs]
    seg_ends_text = [(float(s["end"]), (s.get("text") or "").strip()) for s in segs]

    out: list[ClipCandidate] = []
    for c in candidates:
        # Start → último inicio de segmento ≤ c.start + 0.5 s (dentro de ventana de 1.5 s)
        start_cands = [ss for ss in seg_starts if c.start - 1.5 <= ss <= c.start + 0.5]
        new_start = max(start_cands) if start_cands else c.start
        budget = new_start + params.max_seconds

        # End → escanear desde c.end - 1.5 s hacia adelante hasta el presupuesto.
        # Se elige el PRIMER segmento que cierra oración (. ! ?).
        # Si no hay ninguno dentro del presupuesto, ampliar hasta c.end + 8 s para
        # capturar frases que quedan justo fuera del límite de max_seconds.
        sentence_end: float | None = None
        nearest_end: float | None = None
        extended_budget = max(budget, c.end + 8.0)

        for se, text in seg_ends_text:
            if se < c.end - 1.5:
                continue
            if se > extended_budget:
                break
            if nearest_end is None:
                nearest_end = se
            if text and text[-1] in _SENTENCE_END and (se <= budget or sentence_end is None):
                sentence_end = se
                break

        new_end = (
            sentence_end
            if sentence_end is not None
            else (nearest_end if nearest_end is not None else c.end)
        )
        new_end = min(new_end, total_duration)

        if new_end > new_start:
            out.append(
                c.model_copy(update={"start": round(new_start, 3), "end": round(new_end, 3)})
            )
        else:
            out.append(c)

    return out
