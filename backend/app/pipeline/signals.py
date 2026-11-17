"""Señales ligeras de audio/video para ranking y modo rapido de videos largos."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.pipeline.ingest import job_dir
from app.providers.base import ClipCandidate, ScoreContribution

SCENE_RE = re.compile(r"pts_time:(?P<time>\d+(?:\.\d+)?)")
SILENCE_START_RE = re.compile(r"silence_start:\s*(?P<time>\d+(?:\.\d+)?)")
SILENCE_END_RE = re.compile(r"silence_end:\s*(?P<time>\d+(?:\.\d+)?)")
PTS_TIME_RE = re.compile(r"pts_time:(?P<time>\d+(?:\.\d+)?)")
RMS_RE = re.compile(r"RMS_level=(?P<db>-?\d+(?:\.\d+)?|-?inf)")


def signals_path(job_id: str) -> Path:
    return job_dir(job_id) / "signals.json"


def analyze_media(job_id: str, src: str | Path, duration: float) -> dict[str, Any]:
    """Analiza escenas, silencios y energia de audio. Si FFmpeg falla, señales vacías."""
    scene_times = _scene_times(src)
    silence_intervals = _silence_intervals(src)
    active_intervals = _invert_intervals(silence_intervals, duration)
    # Energia de audio por segundo: es la unica señal que capta disparos, explosiones
    # o gritos en gameplay sin narracion. `silence_intervals` solo dice si hay o no
    # sonido, no cuanta intensidad, asi que no sirve para rankear accion.
    loudness = _loudness_series(src)
    result = {
        "duration": round(duration, 3),
        "scene_times": scene_times,
        "silence_intervals": silence_intervals,
        "active_intervals": active_intervals,
        "loudness": loudness,
    }
    signals_path(job_id).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_signals(job_id: str) -> dict[str, Any]:
    path = signals_path(job_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_candidate(
    candidate: ClipCandidate,
    signals_doc: dict[str, Any] | None,
    *,
    processing_mode: str,
) -> tuple[float, list[str]]:
    if not signals_doc:
        return 0.0, []

    labels: list[str] = []
    bonus = 0.0
    scene_times = [float(t) for t in signals_doc.get("scene_times", [])]
    active_intervals = [_pair(item) for item in signals_doc.get("active_intervals", [])]
    silence_intervals = [_pair(item) for item in signals_doc.get("silence_intervals", [])]

    scene_count = sum(1 for t in scene_times if candidate.start <= t <= candidate.end)
    if scene_count:
        scene_bonus = min(0.12 if processing_mode == "gaming" else 0.07, scene_count * 0.025)
        bonus += scene_bonus
        labels.append("cambios visuales")

    coverage = _coverage(candidate.start, candidate.end, active_intervals)
    if coverage >= 0.55:
        bonus += min(0.08, coverage * 0.06)
        labels.append("audio activo")

    if _near_interval_boundary(candidate.start, silence_intervals) or _near_interval_boundary(
        candidate.end, silence_intervals
    ):
        bonus += 0.035
        labels.append("corte cercano a pausa")

    return round(min(0.18, bonus), 3), labels


def apply_signal_scores(
    candidates: list[ClipCandidate],
    signals_doc: dict[str, Any] | None,
    *,
    processing_mode: str,
) -> list[ClipCandidate]:
    adjusted: list[ClipCandidate] = []
    for candidate in candidates:
        bonus, labels = score_candidate(candidate, signals_doc, processing_mode=processing_mode)
        if not bonus:
            adjusted.append(candidate)
            continue
        reason = candidate.reason
        suffix = "señales: " + ", ".join(labels[:3])
        reason = f"{reason}; {suffix}" if reason else suffix
        new_score = min(1.0, candidate.score + bonus)
        adjusted.append(
            candidate.model_copy(
                update={
                    "score": new_score,
                    "reason": reason,
                    "score_breakdown": [
                        *candidate.score_breakdown,
                        ScoreContribution(
                            signal="señales_audio_escena",
                            stage="signals.apply_signal_scores",
                            score_before=candidate.score,
                            contribution=round(new_score - candidate.score, 3),
                            score_after=new_score,
                            note=suffix,
                        ),
                    ],
                }
            )
        )
    return adjusted


def _loudness_series(src: str | Path, *, timeout: int = 600) -> list[list[float]]:
    """RMS en dB por segundo: [[t, db], ...]. Lista vacia si FFmpeg falla.

    `-vn` + `asetnsamples` es lo que lo hace barato: sin decodificar video y con
    ventanas de 1s exactas tarda ~3s en un video de 36 min (decodificando video
    tardaba ~80s y emitia ~95.000 lineas por frame de audio en vez de una por
    segundo).
    """
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-v",
        "error",
        "-vn",
        "-i",
        str(src),
        "-af",
        (
            "aresample=8000,asetnsamples=8000,astats=metadata=1:reset=1,"
            "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    series: list[list[float]] = []
    current: float | None = None
    for line in (proc.stdout or "").splitlines():
        if match := PTS_TIME_RE.search(line):
            current = float(match.group("time"))
            continue
        if (match := RMS_RE.search(line)) and current is not None:
            raw = match.group("db")
            if raw.endswith("inf"):  # silencio absoluto
                current = None
                continue
            db = float(raw)
            # -120 dB o menos es silencio digital: no aporta y ensucia la media.
            if db > -120:
                series.append([round(current, 2), round(db, 2)])
            current = None
    return series[:20000]


def _scene_times(src: str | Path) -> list[float]:
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-i",
        str(src),
        "-vf",
        "select='gt(scene,0.35)',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    return sorted(set(_parse_times(cmd, SCENE_RE, timeout=600)))[:500]


def _silence_intervals(src: str | Path) -> list[tuple[float, float]]:
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-i",
        str(src),
        "-af",
        "silencedetect=noise=-35dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for line in (proc.stderr or "").splitlines():
        if match := SILENCE_START_RE.search(line):
            start = float(match.group("time"))
        elif match := SILENCE_END_RE.search(line):
            end = float(match.group("time"))
            if start is not None and end > start:
                intervals.append((round(start, 3), round(end, 3)))
            start = None
    return intervals[:1000]


def _parse_times(cmd: list[str], pattern: re.Pattern[str], *, timeout: int) -> list[float]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [round(float(match.group("time")), 3) for match in pattern.finditer(proc.stderr or "")]


def _invert_intervals(
    intervals: list[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in _merge_intervals(intervals):
        if start > cursor:
            out.append((round(cursor, 3), round(min(start, duration), 3)))
        cursor = max(cursor, end)
    if cursor < duration:
        out.append((round(cursor, 3), round(duration, 3)))
    return [(s, e) for s, e in out if e - s >= 0.25]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((max(0.0, s), max(0.0, e)) for s, e in intervals if e > s):
        if not merged or start > merged[-1][1] + 0.25:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return [(round(s, 3), round(e, 3)) for s, e in merged]


def _coverage(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    duration = max(0.001, end - start)
    overlap = sum(max(0.0, min(end, b) - max(start, a)) for a, b in intervals)
    return max(0.0, min(1.0, overlap / duration))


def _near_interval_boundary(
    t: float, intervals: list[tuple[float, float]], *, tolerance: float = 1.2
) -> bool:
    return any(abs(t - start) <= tolerance or abs(t - end) <= tolerance for start, end in intervals)


def _pair(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


# --- Etapa 1: candidatos sin transcript ----------------------------------------
# Para gameplay el transcript no alcanza: 33 chars/min en Warzone contra ~1080 de
# un podcast (medido sobre videos reales de Kerwil). El detector de texto no puede
# proponer momentos que nadie comento, asi que aca se generan candidatos a partir
# de lo que el video HACE: picos de audio (disparos, gritos) y rafagas de cambio
# de escena (accion en pantalla).

TRANSCRIPT_DENSITY_THRESHOLD = 250.0
"""Caracteres de transcript por minuto por debajo de los cuales se activan los
candidatos por señal. Calibrado con datos reales: podcasts 1085 y 1066, gameplay
33 y 60. Cualquier umbral entre ~150 y ~500 separa igual; 250 deja margen de 4x
hacia ambos lados para no tocar un podcast que hable menos de lo normal."""


def transcript_density(segments: list[dict], duration: float) -> float:
    """Caracteres de transcript por minuto. 0.0 si no hay duracion util."""
    if duration <= 0:
        return 0.0
    chars = sum(len(str(s.get("text") or "")) for s in segments)
    return chars / (duration / 60.0)


def _peak_seconds(loudness: list[list[float]], *, k: float = 1.0) -> list[float]:
    """Segundos cuyo RMS supera media + k*desvio. Lista vacia si no hay señal."""
    values = [v for _, v in loudness if v is not None]
    if len(values) < 10:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    sd = var**0.5
    if sd <= 0:
        return []
    threshold = mean + k * sd
    return [t for t, v in loudness if v > threshold]


def action_windows(
    signals_doc: dict[str, Any] | None,
    duration: float,
    *,
    window: float,
    top_n: int,
) -> list[tuple[float, float, float, list[str]]]:
    """Ventanas ordenadas por densidad de accion: (start, end, score01, etiquetas).

    Combina picos de audio y cambios de escena. Se necesitan LAS DOS señales: en
    el video de prueba, los dos clips que el LLM puntuo mas alto por ruido (6m22s
    y 33m35s) estaban en el top-5 de picos de audio, pero el mejor de todos
    (26m05s, una puñalada sigilosa, score 0.94) era silencioso y solo se detecta
    por cambio de escena. Con audio solo, ese clip se pierde.
    """
    if not signals_doc or duration <= 0 or window <= 0:
        return []

    loudness = [
        [float(t), float(v)] for t, v in (signals_doc.get("loudness") or []) if v is not None
    ]
    scene_times = [float(t) for t in (signals_doc.get("scene_times") or [])]
    peaks = _peak_seconds(loudness)
    if not peaks and not scene_times:
        return []

    step = max(1.0, window / 2.0)  # ventanas solapadas al 50%: no parte la accion al medio
    starts: list[float] = []
    t = 0.0
    while t + window <= duration + step:
        starts.append(min(t, max(0.0, duration - window)))
        t += step

    raw: list[tuple[float, float, float, float]] = []
    for start in starts:
        end = min(start + window, duration)
        if end - start < window * 0.6:
            continue
        n_peaks = sum(1 for p in peaks if start <= p < end)
        n_scenes = sum(1 for s in scene_times if start <= s < end)
        raw.append((start, end, float(n_peaks), float(n_scenes)))

    if not raw:
        return []

    max_peaks = max((p for _, _, p, _ in raw), default=0.0) or 1.0
    max_scenes = max((s for _, _, _, s in raw), default=0.0) or 1.0

    scored: list[tuple[float, float, float, list[str]]] = []
    for start, end, n_peaks, n_scenes in raw:
        audio_n = n_peaks / max_peaks
        scene_n = n_scenes / max_scenes
        # 0.6/0.4: el audio discrimina mejor la accion, pero la escena es la que
        # rescata los momentos sigilosos, asi que no puede pesar poco.
        score = 0.6 * audio_n + 0.4 * scene_n
        if score <= 0:
            continue
        labels: list[str] = []
        if n_peaks:
            labels.append("picos de audio")
        if n_scenes:
            labels.append("cambios de escena")
        scored.append((start, end, round(score, 4), labels))

    scored.sort(key=lambda x: x[2], reverse=True)

    # No devolver ventanas solapadas: son el mismo momento contado dos veces.
    selected: list[tuple[float, float, float, list[str]]] = []
    for cand in scored:
        if len(selected) >= top_n:
            break
        if any(cand[0] < s[1] and s[0] < cand[1] for s in selected):
            continue
        selected.append(cand)
    return selected


def build_action_candidates(
    signals_doc: dict[str, Any] | None,
    duration: float,
    *,
    num_clips: int,
    min_seconds: float,
    max_seconds: float,
) -> list[ClipCandidate]:
    """Candidatos derivados de señal, para videos con poco o ningun transcript."""
    window = max(min_seconds, min(max_seconds, (min_seconds + max_seconds) / 2))
    windows = action_windows(signals_doc, duration, window=window, top_n=max(num_clips * 2, 8))
    out: list[ClipCandidate] = []
    for start, end, score, labels in windows:
        detail = " + ".join(labels) if labels else "actividad"
        out.append(
            ClipCandidate(
                start=round(start, 3),
                end=round(end, 3),
                # Score deliberadamente en 0.40-0.75: por debajo de un clip que el
                # LLM justifico con contenido real, por encima del relleno naive.
                score=round(0.40 + 0.35 * score, 3),
                title="",
                reason=f"señal: {detail}",
            )
        )
    return out
