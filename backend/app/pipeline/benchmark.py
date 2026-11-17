"""Harness de benchmark del motor de detección (Paso 7 del plan P0).

No acepta "generó un clip" como prueba de éxito. Compara una lista de
candidatos (venga de donde venga: build_event_candidates, NaiveDetector, un
LLM, o el pipeline completo) contra un ground truth marcado manualmente, y
calcula recall/precision/overlap/boundary error — las métricas que la
auditoría identificó como prioritarias (Candidate Recall primero: si el
evento correcto nunca entra al pool, ningún ranker puede recuperarlo).

Formato de ground truth (un archivo JSON por video):
{
  "video": "ruta o descripción",
  "mode": "gaming",
  "output_type": "hook",
  "ground_truth": [{"start": 37.0, "peak": 41.2, "end": 46.0, "type": "hook"}],
  "annotation_status": "ai_assisted_pending_human_confirmation" | "human_confirmed",
  "evidence": "descripción de por qué se eligió este timestamp (audio/escena/frame)"
}

`annotation_status` es obligatorio y explícito: si no fue confirmado por un
humano, cualquier reporte que use este ground truth debe advertirlo — nunca
se trata un timestamp generado por IA como verdad de referencia silenciosa.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.providers.base import ClipCandidate


@dataclass
class GroundTruthEvent:
    start: float
    end: float
    peak: float | None = None
    type: str = "hook"


@dataclass
class GroundTruth:
    video: str
    mode: str
    output_type: str
    events: list[GroundTruthEvent]
    annotation_status: str
    evidence: str = ""


@dataclass
class MatchResult:
    """Resultado de emparejar UN evento de ground truth contra un pool de candidatos."""

    event: GroundTruthEvent
    matched: bool
    rank: int | None  # posición 1-indexada por score desc del mejor candidato que solapa
    overlap_ratio: float  # IoU del mejor candidato que solapa (0 si no matched)
    start_error: float | None  # |candidate.start - event.start| del mejor match
    end_error: float | None  # |candidate.end - event.end| del mejor match


@dataclass
class BenchmarkReport:
    candidate_count: int
    matches: list[MatchResult]
    false_positive_count: int  # candidatos que no solapan ningún ground truth

    @property
    def candidate_recall(self) -> float:
        """Fracción de eventos de ground truth presentes en el pool (con CUALQUIER
        overlap > 0), sin importar el ranking. Métrica prioritaria: si esto es bajo,
        ningún ranker puede compensarlo."""
        if not self.matches:
            return 0.0
        return sum(1 for m in self.matches if m.matched) / len(self.matches)

    def top_k_recall(self, k: int) -> float:
        if not self.matches:
            return 0.0
        hits = sum(1 for m in self.matches if m.matched and m.rank is not None and m.rank <= k)
        return hits / len(self.matches)

    @property
    def precision(self) -> float:
        if self.candidate_count == 0:
            return 0.0
        true_positive_candidates = self.candidate_count - self.false_positive_count
        return true_positive_candidates / self.candidate_count

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_recall": round(self.candidate_recall, 3),
            "top1_recall": round(self.top_k_recall(1), 3),
            "top3_recall": round(self.top_k_recall(3), 3),
            "top5_recall": round(self.top_k_recall(5), 3),
            "precision": round(self.precision, 3),
            "false_positive_count": self.false_positive_count,
            "matches": [
                {
                    "ground_truth": f"{m.event.start:.1f}-{m.event.end:.1f} ({m.event.type})",
                    "matched": m.matched,
                    "rank": m.rank,
                    "overlap_ratio": round(m.overlap_ratio, 3) if m.matched else None,
                    "start_error_s": round(m.start_error, 2) if m.start_error is not None else None,
                    "end_error_s": round(m.end_error, 2) if m.end_error is not None else None,
                }
                for m in self.matches
            ],
        }


def load_ground_truth(path: str | Path) -> GroundTruth:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "annotation_status" not in data:
        raise ValueError(
            f"{path}: falta 'annotation_status' — todo ground truth debe declarar "
            "explícitamente si fue confirmado por un humano o es AI-assisted pendiente."
        )
    events = [
        GroundTruthEvent(
            start=float(item["start"]),
            end=float(item["end"]),
            peak=float(item["peak"]) if item.get("peak") is not None else None,
            type=item.get("type", "hook"),
        )
        for item in data.get("ground_truth", [])
    ]
    return GroundTruth(
        video=data.get("video", ""),
        mode=data.get("mode", ""),
        output_type=data.get("output_type", ""),
        events=events,
        annotation_status=data["annotation_status"],
        evidence=data.get("evidence", ""),
    )


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def evaluate(candidates: list[ClipCandidate], ground_truth: GroundTruth) -> BenchmarkReport:
    """Compara candidatos (ya ordenados por score desc, o se ordenan aquí) contra
    el ground truth. No requiere que los candidatos vengan de ningún proveedor
    en particular — funciona igual con NaiveDetector, build_event_candidates,
    un LLM real, o el pool final tras rerank visual."""
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    matched_candidate_indices: set[int] = set()

    matches: list[MatchResult] = []
    for event in ground_truth.events:
        best_rank: int | None = None
        best_overlap = 0.0
        best_candidate: ClipCandidate | None = None
        for rank, cand in enumerate(ranked, start=1):
            overlap = _iou(cand.start, cand.end, event.start, event.end)
            if overlap > 0.0 and overlap > best_overlap:
                best_overlap = overlap
                best_rank = rank
                best_candidate = cand
        if best_candidate is not None:
            matched_candidate_indices.add(id(best_candidate))
            matches.append(
                MatchResult(
                    event=event,
                    matched=True,
                    rank=best_rank,
                    overlap_ratio=best_overlap,
                    start_error=abs(best_candidate.start - event.start),
                    end_error=abs(best_candidate.end - event.end),
                )
            )
        else:
            matches.append(
                MatchResult(
                    event=event,
                    matched=False,
                    rank=None,
                    overlap_ratio=0.0,
                    start_error=None,
                    end_error=None,
                )
            )

    false_positives = sum(
        1
        for cand in ranked
        if id(cand) not in matched_candidate_indices
        and not any(_iou(cand.start, cand.end, e.start, e.end) > 0.0 for e in ground_truth.events)
    )

    return BenchmarkReport(
        candidate_count=len(candidates), matches=matches, false_positive_count=false_positives
    )
