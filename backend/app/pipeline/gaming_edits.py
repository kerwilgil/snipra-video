"""Seleccion y composicion de hooks y montajes de gameplay.

Los modos gaming tradicionales devuelven ventanas continuas. Estos helpers
modelan el edit final como una lista ordenada de tramos de la fuente para poder
representar accion -> recompensa -> repeticion y montajes de microjugadas.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from app.core.config import settings
from app.pipeline.menu_filter import frame_texts, looks_like_defeat, looks_like_menu
from app.pipeline.signals import action_windows
from app.providers.base import (
    ClipCandidate,
    ScoreContribution,
    SourceRange,
    dedupe_by_overlap,
    overlap_ratio,
)

logger = logging.getLogger(__name__)

REWARD_TERMS = frozenset(
    {
        "victoria",
        "victory",
        "ganaste",
        "winner",
        "baja final",
        "final kill",
        "killcam",
        "jugada de la partida",
        "play of the game",
        "elimination",
    }
)

TEASER_REJECTION_HINTS = (
    "poco clara",
    "sin momento de máxima intensidad",
    "sin momento de maxima intensidad",
    "estático",
    "estatico",
    "pantalla de muerte",
    "pantalla de derrota",
    "resultado visual: death",
    "resultado visual: menu",
    "resultado visual: scoreboard",
    "resultado visual: idle",
    "marcador",
    "scoreboard",
    "menú",
    "menu",
    "lobby",
)
TRANSCRIPT_PAYOFF_TERMS = frozenset(
    {
        "estoy a uno",
        "ganamos",
        "gané",
        "gane",
        "victoria",
        "último",
        "ultimo",
        "ellos, papá",
        "ellos papa",
        "lo maté",
        "lo mate",
        "clutch",
    }
)


def gaming_detect_params(mode: str, requested_outputs: int) -> tuple[int, float, float]:
    """Devuelve tamaño del pool y limites de las ventanas que compiten."""
    if mode == "gaming_auto":
        return min(24, max(16, requested_outputs * 4)), 6.0, 25.0
    if mode == "gaming_hook":
        return max(12, requested_outputs * 6), 8.0, 25.0
    if mode == "gaming_highlights":
        return max(16, requested_outputs * 8), 6.0, 18.0
    return requested_outputs, settings.min_clip_seconds, settings.max_clip_seconds


def build_event_candidates(
    signals_doc: dict | None,
    segments: list[dict],
    duration: float,
    *,
    top_n: int,
) -> list[ClipCandidate]:
    """Crea ventanas cortas con cola suficiente para no perder el desenlace."""
    out: list[ClipCandidate] = []
    for start, end, score, labels in action_windows(
        signals_doc, duration, window=12.0, top_n=max(top_n, 16)
    ):
        # Dos segundos de anticipacion y hasta diez de cola: la señal de disparos
        # suele terminar antes de que aparezca VICTORIA o la killcam.
        event_start = max(0.0, start - 2.0)
        event_end = min(duration, end + 10.0)
        if event_end - event_start > 25.0:
            event_start = event_end - 25.0
        detail = " + ".join(labels) if labels else "actividad"
        out.append(
            ClipCandidate(
                start=round(event_start, 3),
                end=round(event_end, 3),
                score=round(min(0.88, 0.44 + 0.4 * score), 3),
                title="Jugada decisiva",
                hook="La jugada cambia en segundos",
                reason=f"señal gaming: {detail}; cola de desenlace protegida",
                hashtags=["#gaming", "#gameplay", "#gamingclips"],
            )
        )

    for seg in segments:
        text = str(seg.get("text") or "").strip()
        normalized = _normalize(text)
        hits = sorted(term for term in TRANSCRIPT_PAYOFF_TERMS if term in normalized)
        if not hits:
            continue
        center_start = float(seg.get("start", 0.0))
        center_end = float(seg.get("end", center_start))
        start = max(0.0, center_start - 6.0)
        end = min(duration, max(center_end + 10.0, start + 8.0))
        if end - start > 25.0:
            start = end - 25.0
        decisive = any(term in normalized for term in {"estoy a uno", "ellos papa", "ellos, papa"})
        out.append(
            ClipCandidate(
                start=round(start, 3),
                end=round(end, 3),
                score=0.92 if decisive else 0.82,
                title="El momento decisivo",
                hook=text,
                reason=f"payoff en comentario: {', '.join(hits[:2])}",
                hashtags=["#gaming", "#clutch", "#gamingclips"],
                description=text,
            )
        )
    return dedupe_by_overlap(out, ratio=0.82)


def reward_text(texts: list[str]) -> tuple[bool, str]:
    joined = " | ".join(_normalize(text) for text in texts)
    hits = sorted(term for term in REWARD_TERMS if term in joined)
    return bool(hits), ", ".join(hits[:3])


def annotate_visual_rewards(
    candidates: list[ClipCandidate],
    video_path: str | None,
    *,
    limit: int = 12,
    progress_callback=None,
) -> list[ClipCandidate]:
    """Bonifica recompensas verificadas por OCR en la cola de eventos.

    Es un no-op seguro si RapidOCR no esta disponible o FFmpeg no puede extraer
    el frame. Sólo revisa el pool superior para mantener acotado el coste.
    """
    if not video_path or not candidates:
        return candidates
    annotated = list(candidates)
    total = min(len(annotated), limit)
    with tempfile.TemporaryDirectory() as tmp:
        for idx, candidate in enumerate(annotated[:limit]):
            samples = [
                candidate.start + candidate.duration * 0.65,
                candidate.start + candidate.duration * 0.9,
            ]
            labels: dict[str, float] = {}
            defeat_label = ""
            for sample_idx, timestamp in enumerate(samples):
                out = Path(tmp) / f"reward-{idx}-{sample_idx}.jpg"
                cmd = [
                    settings.ffmpeg_path,
                    "-v",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=960:-2",
                    str(out),
                    "-y",
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=60)
                except (OSError, subprocess.SubprocessError):
                    continue
                if not out.exists():
                    continue
                texts = frame_texts(str(out))
                defeated, defeated_why = looks_like_defeat(texts)
                if defeated:
                    defeat_label = defeated_why
                    continue
                found, label = reward_text(texts)
                if found:
                    for part in label.split(","):
                        if part.strip():
                            labels.setdefault(part.strip(), timestamp)
            if defeat_label:
                new_score = round(max(0.0, candidate.score - 0.22), 3)
                note = f"{defeat_label} verificada"
                annotated[idx] = candidate.model_copy(
                    update={
                        "score": new_score,
                        "reason": f"{candidate.reason}; {note}",
                        "score_breakdown": [
                            *candidate.score_breakdown,
                            ScoreContribution(
                                signal="recompensa_ocr",
                                stage="gaming_edits.annotate_visual_rewards",
                                score_before=candidate.score,
                                contribution=round(new_score - candidate.score, 3),
                                score_after=new_score,
                                note=note,
                            ),
                        ],
                    }
                )
            elif labels:
                suffix = "recompensa visual: " + ", ".join(
                    f"{label} @{labels[label]:.3f}s" for label in sorted(labels)
                )
                new_score = round(min(1.0, candidate.score + 0.16), 3)
                annotated[idx] = candidate.model_copy(
                    update={
                        "score": new_score,
                        "reason": f"{candidate.reason}; {suffix}",
                        "score_breakdown": [
                            *candidate.score_breakdown,
                            ScoreContribution(
                                signal="recompensa_ocr",
                                stage="gaming_edits.annotate_visual_rewards",
                                score_before=candidate.score,
                                contribution=round(new_score - candidate.score, 3),
                                score_after=new_score,
                                note=suffix,
                            ),
                        ],
                    }
                )
            if progress_callback:
                progress_callback(
                    (idx + 1) / total,
                    f"Verificando recompensas: candidato {idx + 1}/{total}",
                )
    return annotated


def compose_gaming_edit(
    candidates: list[ClipCandidate],
    mode: str,
    duration: float,
    *,
    requested_outputs: int = 1,
    signals_doc: dict | None = None,
) -> list[ClipCandidate]:
    """Convierte el pool puntuado en el único edit que se exportará."""
    pool = dedupe_by_overlap(
        sorted(candidates, key=lambda item: item.score, reverse=True), ratio=0.72
    )
    if not pool:
        return []
    if mode == "gaming_hook":
        return [_compose_hook(pool, duration, signals_doc)]
    if mode == "gaming_highlights":
        return [_compose_highlights(pool, duration, signals_doc)]
    if mode == "gaming_auto":
        chosen = choose_gaming_output(pool, duration)
        if chosen == "hook":
            result = _compose_hook(pool, duration, signals_doc)
            return [
                result.model_copy(update={"reason": f"{result.reason}; automatico eligio hook"})
            ]
        if chosen == "highlights":
            result = _compose_highlights(pool, duration, signals_doc)
            return [
                result.model_copy(
                    update={"reason": f"{result.reason}; automatico eligio highlights"}
                )
            ]
        return [
            item.model_copy(
                update={"reason": f"{item.reason}; automatico eligio clip independiente"}
            )
            for item in pool[: max(1, requested_outputs)]
        ]
    return candidates


def sanitize_gaming_outcomes(
    edits: list[ClipCandidate],
    video_path: str | None,
) -> list[ClipCandidate]:
    """Recorta derrotas, menús o marcadores dentro del montaje final.

    El OCR inicial revisa ventanas candidatas amplias. Una ventana puede incluir
    una baja y, varios segundos después, una muerte; la visión puede valorar la
    baja y el compositor escoger justo ese subtramo. Esta QA mira los cortes
    exactos y termina el beat antes de ``Te eliminó``.
    """
    if not video_path:
        return edits

    sanitized: list[ClipCandidate] = []
    with tempfile.TemporaryDirectory() as tmp:
        for edit_index, edit in enumerate(edits):
            if not edit.source_ranges:
                sanitized.append(edit)
                continue
            ranges: list[SourceRange] = []
            changed = False
            for range_index, source_range in enumerate(edit.source_ranges):
                is_payoff = source_range.kind == "payoff"
                blocked_at: float | None = None
                blocked_fraction = 1.0
                blocked_kind = ""
                fractions = (0.08, 0.20, 0.32) if is_payoff else (0.35, 0.65, 0.90)
                for sample_index, fraction in enumerate(fractions):
                    timestamp = source_range.start + source_range.duration * fraction
                    out = Path(tmp) / f"qa-{edit_index}-{range_index}-{sample_index}.jpg"
                    cmd = [
                        settings.ffmpeg_path,
                        "-v",
                        "error",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        video_path,
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=720:-2",
                        str(out),
                        "-y",
                    ]
                    try:
                        subprocess.run(cmd, capture_output=True, timeout=60)
                    except (OSError, subprocess.SubprocessError):
                        continue
                    if not out.exists():
                        continue
                    texts = frame_texts(str(out))
                    defeated, _ = looks_like_defeat(texts)
                    if defeated:
                        blocked_at = timestamp
                        blocked_fraction = fraction
                        blocked_kind = "derrota"
                        break
                    menu, _ = looks_like_menu(texts)
                    if menu:
                        blocked_at = timestamp
                        blocked_fraction = fraction
                        blocked_kind = "interfaz"
                        break
                if blocked_at is None:
                    ranges.append(source_range)
                    continue
                if is_payoff:
                    # El cierre puede comenzar con un marcador breve y volver
                    # enseguida a la accion/recompensa. Se limpia solo la
                    # entrada; una derrota no autoriza a recortar un payoff
                    # verificado sin evidencia de dónde termina.
                    if blocked_kind != "interfaz":
                        ranges.append(source_range)
                        continue
                    safe_start = round(min(source_range.end, blocked_at + 0.35), 3)
                    if source_range.end - safe_start >= 3.0:
                        changed = True
                        ranges.append(source_range.model_copy(update={"start": safe_start}))
                    else:
                        ranges.append(source_range)
                    continue
                changed = True
                # Si la interfaz ocupa ya el primer tercio/centro, el beat no
                # tiene acción suficiente y se elimina entero. En la cola se
                # conserva lo anterior, cortando antes de que aparezca.
                if blocked_fraction <= 0.65:
                    continue
                backoff = 1.5 if blocked_kind == "derrota" else 0.7
                safe_end = round(max(source_range.start, blocked_at - backoff), 3)
                if safe_end - source_range.start >= 1.2:
                    ranges.append(source_range.model_copy(update={"end": safe_end}))

            if not ranges:
                sanitized.append(edit)
                continue
            if not changed:
                sanitized.append(edit)
                continue
            weighted_duration = sum(item.duration for item in ranges)
            weighted_score = round(
                sum(item.score * item.duration for item in ranges) / max(0.1, weighted_duration),
                3,
            )
            note = "QA OCR recorto derrotas, menus o marcadores"
            sanitized.append(
                edit.model_copy(
                    update={
                        "start": min(item.start for item in ranges),
                        "end": max(item.end for item in ranges),
                        "score": weighted_score,
                        "reason": (f"{edit.reason}; {note}"),
                        "source_ranges": ranges,
                        "score_breakdown": [
                            *edit.score_breakdown,
                            ScoreContribution(
                                signal="qa_ocr_final",
                                stage="gaming_edits.sanitize_gaming_outcomes",
                                score_before=edit.score,
                                contribution=round(weighted_score - edit.score, 3),
                                score_after=weighted_score,
                                note=note,
                            ),
                        ],
                    }
                )
            )
    return sanitized


def choose_gaming_output(pool: list[ClipCandidate], duration: float) -> str:
    """Elige hook, highlights o clips usando evidencia del pool completo."""
    if not pool:
        return "clips"

    top = pool[0]
    top_reason = top.reason.lower()
    has_verified_payoff = any(
        marker in top_reason
        for marker in (
            "recompensa visual",
            "payoff en comentario",
            "baja final",
            "killcam",
        )
    )
    if top.score >= 0.86 and has_verified_payoff:
        return "hook"

    strong = [item for item in pool if item.score >= 0.62]
    if len(strong) >= 4:
        span = max(item.start for item in strong) - min(item.start for item in strong)
        required_span = min(180.0, max(45.0, duration * 0.20))
        if span >= required_span:
            return "highlights"
    return "clips"


def _compose_hook(
    pool: list[ClipCandidate],
    duration: float,
    signals_doc: dict | None = None,
) -> ClipCandidate:
    eligible = [item for item in pool if not _is_weak_beat(item)] or pool
    verified_payoffs = [item for item in eligible if _is_verified_payoff(item)]
    best = (
        max(verified_payoffs, key=_payoff_priority)
        if verified_payoffs
        else max(eligible, key=_teaser_priority)
    )
    primary_kind = "payoff" if _is_verified_payoff(best) else "hook"
    primary_duration = _beat_duration(best, primary_kind, 0, 1)
    ranges = [
        _micro_fragment(
            best,
            primary_duration,
            primary_kind,
            duration,
            anchor=0.68,
            signals_doc=signals_doc,
        )
    ]

    # Una recompensa/repeticion posterior cercana se conserva como segundo beat.
    replay = next(
        (
            item
            for item in pool
            if item is not best
            if item.end >= best.end
            and item.start - best.end <= 20.0
            and (
                "recompensa visual" in item.reason.lower()
                or "baja final" in item.reason.lower()
                or "killcam" in item.reason.lower()
            )
        ),
        None,
    )
    if replay is not None:
        timestamp_match = re.search(r"baja final @(\d+(?:\.\d+)?)s", replay.reason.lower())
        if timestamp_match:
            replay_start = max(best.end, float(timestamp_match.group(1)) - 1.0)
            replay_end = min(duration, replay_start + 10.0)
        else:
            replay_start = max(best.end, replay.start)
            replay_end = min(replay.end, replay_start + 10.0)
        if replay_end - replay_start >= 2.0:
            replay_target = min(
                _beat_duration(replay, "replay", 0, 1),
                replay_end - replay_start,
            )
            if timestamp_match:
                reward_at = float(timestamp_match.group(1))
                replay_end = min(duration, replay.end, reward_at + 2.0)
                replay_start = max(
                    best.end,
                    replay.end - replay.duration,
                    replay_end - replay_target,
                )
                replay_range = _range(
                    replay_start,
                    replay_end,
                    "replay",
                    replay.score,
                    duration,
                )
                replay_range = _snap_range_to_scenes(replay_range, signals_doc, duration)
            else:
                replay_range = _micro_fragment(
                    replay,
                    replay_target,
                    "replay",
                    duration,
                    anchor=0.55,
                    signals_doc=signals_doc,
                )
            if not any(overlap_ratio(replay_range, item) > 0.20 for item in ranges):
                ranges.append(replay_range)

    ranges = [item for item in ranges if item.duration >= 0.5]
    start = min(item.start for item in ranges)
    end = max(item.end for item in ranges)
    edit_score = sum(item.score * item.duration for item in ranges) / sum(
        item.duration for item in ranges
    )
    reason = (
        f"{best.reason}; gancho gaming compacto centrado en el pico "
        f"({len(ranges)} tramo(s), ritmo variable)"
    )
    return best.model_copy(
        update={
            "start": round(start, 3),
            "end": round(end, 3),
            "title": best.title or "Gaming Hook",
            "score": round(edit_score, 3),
            "reason": reason,
            "source_ranges": ranges,
        }
    )


def _compose_highlights(
    pool: list[ClipCandidate],
    duration: float,
    signals_doc: dict | None = None,
) -> ClipCandidate:
    selected: list[ClipCandidate] = []
    for candidate in pool:
        full_range = _range(
            candidate.start,
            candidate.end,
            "highlight",
            candidate.score,
            duration,
        )
        if full_range.duration < 2.0:
            continue
        if any(
            overlap_ratio(
                full_range,
                _range(item.start, item.end, "highlight", item.score, duration),
            )
            > 0.35
            for item in selected
        ):
            continue
        selected.append(candidate)
        # Se guardan reservas porque dos ventanas amplias distintas pueden
        # converger en el mismo microfragmento al construir el trailer.
        if len(selected) >= 12:
            break

    if len(selected) < 2:
        return _compose_hook(pool, duration, signals_doc)

    # Un desenlace verificado manda sobre la mera posicion temporal. Elegir el
    # candidato mas tardio convertia cualquier menu posterior en el final del
    # trailer y podia dejar fuera una victoria confirmada unos segundos antes.
    verified_payoffs = [item for item in selected if _is_verified_payoff(item)]
    ending = (
        max(verified_payoffs, key=_payoff_priority)
        if verified_payoffs
        else max(selected, key=lambda item: (item.start, item.score))
    )
    desired_count = min(8, len(selected))
    teaser_count = 2 if desired_count >= 6 else 1
    all_teaser_candidates = [item for item in selected if item is not ending]
    teaser_pool = [
        item for item in selected if item is not ending and not _is_weak_beat(item)
    ] or all_teaser_candidates
    eligible_teasers = [
        item
        for item in teaser_pool
        if not any(hint in item.reason.lower() for hint in TEASER_REJECTION_HINTS)
    ]
    teaser_candidates = sorted(
        eligible_teasers if len(eligible_teasers) >= teaser_count else teaser_pool,
        key=_teaser_priority,
        reverse=True,
    )[:teaser_count]
    teaser_ids = {id(item) for item in teaser_candidates}
    teaser_previews = [
        _micro_fragment(
            item,
            _beat_duration(item, "teaser", index, teaser_count),
            "teaser",
            duration,
            anchor=0.65,
            signals_doc=signals_doc,
        )
        for index, item in enumerate(teaser_candidates)
    ]

    # El no-solape se valida sobre los fragmentos que realmente se exportan,
    # no sólo sobre las ventanas candidatas originales. Esas ventanas pueden
    # ser distintas y aun así terminar centradas en la misma jugada.
    body_slots = max(0, desired_count - teaser_count - 1)
    storyline_pool: list[ClipCandidate] = []
    body_previews: list[SourceRange] = []
    for item in sorted(
        (
            candidate
            for candidate in selected
            if id(candidate) not in teaser_ids
            and candidate is not ending
            and not _is_weak_beat(candidate)
            and (not verified_payoffs or candidate.start < ending.start)
        ),
        key=lambda candidate: candidate.start,
    ):
        preview = _micro_fragment(
            item,
            _beat_duration(item, "escalation", 0, max(1, body_slots)),
            "escalation",
            duration,
            anchor=0.62,
            signals_doc=signals_doc,
        )
        if any(overlap_ratio(preview, seen) > 0.20 for seen in teaser_previews + body_previews):
            continue
        storyline_pool.append(item)
        body_previews.append(preview)

    storyline = _select_distributed_storyline(storyline_pool, body_slots)
    planned: list[tuple[ClipCandidate, SourceRange]] = list(
        zip(teaser_candidates, teaser_previews, strict=False)
    )
    for index, candidate in enumerate(storyline):
        kind = "setup" if index == 0 else "escalation"
        planned.append(
            (
                candidate,
                _micro_fragment(
                    candidate,
                    _beat_duration(candidate, kind, index, len(storyline)),
                    kind,
                    duration,
                    anchor=0.55 if kind == "setup" else 0.65,
                    signals_doc=signals_doc,
                ),
            )
        )
    planned.append(
        (
            ending,
            _micro_fragment(
                ending,
                _beat_duration(ending, "payoff", 0, 1),
                "payoff",
                duration,
                anchor=1.0,
                signals_doc=signals_doc,
            ),
        )
    )

    # Elimina fragmentos demasiado cortos y limita el montaje a 45 s recortando
    # el beat regular mas largo, nunca el payoff final.
    planned = [(candidate, item) for candidate, item in planned if item.duration >= 1.4]
    total = sum(item.duration for _, item in planned)
    if total > 45.0:
        overflow = total - 45.0
        for index in range(len(planned) - 2, -1, -1):
            candidate, item = planned[index]
            minimum = 1.4 if item.kind == "teaser" else 2.4
            removable = max(0.0, item.duration - minimum)
            trim = min(overflow, removable)
            if trim:
                planned[index] = (
                    candidate,
                    item.model_copy(update={"start": round(item.start + trim, 3)}),
                )
                overflow -= trim
            if overflow <= 0:
                break

    ranges = [item for _, item in planned]
    weighted = planned
    score = sum(candidate.score * item.duration for candidate, item in weighted) / max(
        0.1, sum(item.duration for _, item in weighted)
    )
    best = max(selected, key=lambda item: item.score)
    return best.model_copy(
        update={
            "start": round(min(item.start for item in ranges), 3),
            "end": round(max(item.end for item in ranges), 3),
            "title": "Gaming Highlights",
            "hook": best.hook or "Las mejores jugadas, sin tiempos muertos",
            "score": round(min(1.0, score + 0.03), 3),
            "reason": (
                f"montaje gaming narrativo: {len(teaser_previews)} teaser(s), "
                f"{len(storyline)} beat(s) de preparacion/escalada y payoff protegido; "
                "duraciones variables y score ponderado del edit"
            ),
            "source_ranges": ranges,
            "hashtags": ["#gaming", "#highlights", "#gamingclips", "#shorts"],
        }
    )


def _micro_fragment(
    candidate: ClipCandidate,
    target_duration: float,
    kind: str,
    video_duration: float,
    *,
    anchor: float,
    signals_doc: dict | None = None,
) -> SourceRange:
    if kind == "payoff":
        reward_times = [
            float(match) for match in re.findall(r"@(\d+(?:\.\d+)?)s", candidate.reason.lower())
        ]
        if reward_times:
            # OCR mira la cola del evento y puede confirmar la victoria unos
            # segundos despues del end saneado por max_seconds. El cierre debe
            # incluir la evidencia que justifico llamarlo payoff.
            end = min(video_duration, max(candidate.end, max(reward_times) + 2.2))
            start = max(0.0, end - min(target_duration, end))
            result = _range(start, end, kind, candidate.score, video_duration)
            return _snap_range_to_scenes(result, signals_doc, video_duration)

    available = max(0.0, candidate.end - candidate.start)
    length = min(target_duration, available)
    visual_peak = re.search(r"pico visual frame ([1-4])/4", candidate.reason.lower())
    if visual_peak:
        # Los cuatro frames se extraen a 1/5, 2/5, 3/5 y 4/5 de la ventana.
        # Se deja ~1 s de anticipación para que la baja tenga preparación, pero
        # el clímax ocurre dentro de los primeros segundos del fragmento.
        peak = candidate.start + available * int(visual_peak.group(1)) / 5.0
        lead = min(1.0, length * 0.35)
        start = max(candidate.start, min(peak - lead, candidate.end - length))
        result = _range(start, start + length, kind, candidate.score, video_duration)
        return _snap_range_to_scenes(result, signals_doc, video_duration)

    signal_peak = _signal_peak_time(candidate, signals_doc)
    center = signal_peak or candidate.start + available * max(0.0, min(1.0, anchor))
    if signal_peak is not None:
        lead = min(0.9 if kind == "teaser" else 1.4, length * 0.35)
        start = max(candidate.start, min(center - lead, candidate.end - length))
        result = _range(start, start + length, kind, candidate.score, video_duration)
        return _snap_range_to_scenes(result, signals_doc, video_duration)
    start = max(candidate.start, min(center - length / 2, candidate.end - length))
    end = min(candidate.end, start + length)
    result = _range(start, end, kind, candidate.score, video_duration)
    return _snap_range_to_scenes(result, signals_doc, video_duration)


def _is_verified_payoff(candidate: ClipCandidate) -> bool:
    reason = _normalize(candidate.reason)
    has_evidence = any(
        marker in reason
        for marker in (
            "recompensa visual",
            "payoff en comentario",
            "resultado visual: victory",
        )
    )
    has_outcome = any(
        marker in reason
        for marker in (
            "victoria",
            "victory",
            "winner",
            "ganaste",
            "baja final",
            "final kill",
            "killcam",
        )
    )
    return has_evidence and has_outcome


def _payoff_priority(candidate: ClipCandidate) -> tuple[float, int, int, float]:
    """Ordena candidatos que YA pasaron _is_verified_payoff (evidencia + outcome).

    P0.4 (auditoría, hallazgo de Hook mediocre): antes `victory`/`timestamped`
    iban primero en la tupla, lo que hacía que esas banderas (0/1) dominaran
    lexicográficamente sobre `candidate.score`. Dentro del subconjunto ya
    verificado, el score real (que ya incorpora LLM + señales + OCR) es el
    criterio primario; `victory`/`timestamped` quedan como desempate entre
    candidatos de fuerza equivalente. Nunca compite contra un candidato NO
    verificado con score mayor: ese filtro ocurre antes, en `verified_payoffs`.
    """
    reason = _normalize(candidate.reason)
    victory = int(any(marker in reason for marker in ("victoria", "victory", "winner", "ganaste")))
    timestamped = int(bool(re.search(r"@\d+(?:\.\d+)?s", reason)))
    return round(candidate.score, 2), victory, timestamped, candidate.start


def _teaser_priority(candidate: ClipCandidate) -> tuple[float, int, int, int, float]:
    """Ordena teasers/hooks SIN payoff verificado disponible.

    P0.4: antes `action_strength` (conteo de 8 frases fijas en `reason`) iba
    primero en la tupla y dominaba sobre `candidate.score` — un `reason`
    verboso podía ganarle a un score genuinamente mejor. Ahora el score real
    manda; `evento verificado individualmente` (relevancia), `fuerza
    semántica` (conteo de marcadores) y `pico visual/timestamp` quedan como
    desempate entre candidatos de score equivalente (redondeado a 2
    decimales, para no exigir igualdad exacta de floats).
    """
    reason = _normalize(candidate.reason)
    action_markers = (
        "resultado visual: kill",
        "baja final",
        "final kill",
        "killcam",
        "pico visual frame",
        "tiroteo",
        "accion clara",
        "jugada decisiva",
    )
    action_strength = sum(marker in reason for marker in action_markers)
    has_peak = int("pico visual frame" in reason or "@" in reason)
    verified = int(_is_verified_payoff(candidate))
    # A igualdad de score, un evento compacto abre mejor que una ventana
    # larga que necesita demasiada preparacion.
    return round(candidate.score, 2), verified, action_strength, has_peak, -candidate.duration


def _is_weak_beat(candidate: ClipCandidate) -> bool:
    reason = _normalize(candidate.reason)
    return any(_normalize(hint) in reason for hint in TEASER_REJECTION_HINTS)


def _select_distributed_storyline(
    candidates: list[ClipCandidate],
    count: int,
) -> list[ClipCandidate]:
    """Escoge beats fuertes repartidos en el tiempo y conserva su cronologia."""
    ordered = sorted(candidates, key=lambda item: item.start)
    if count <= 0:
        return []
    if len(ordered) <= count:
        return ordered

    first = ordered[0].start
    span = max(0.001, ordered[-1].start - first)
    selected: list[ClipCandidate] = []
    selected_ids: set[int] = set()
    for slot in range(count):
        lower = first + span * slot / count
        upper = first + span * (slot + 1) / count
        bucket = [
            item
            for item in ordered
            if id(item) not in selected_ids
            and lower <= item.start <= (upper if slot == count - 1 else upper - 0.001)
        ]
        if not bucket:
            center = (lower + upper) / 2.0
            remaining = [item for item in ordered if id(item) not in selected_ids]
            if not remaining:
                break
            bucket = sorted(remaining, key=lambda item: abs(item.start - center))[:2]
        chosen = max(bucket, key=lambda item: (_teaser_priority(item), item.score))
        selected.append(chosen)
        selected_ids.add(id(chosen))
    return sorted(selected, key=lambda item: item.start)


def _beat_duration(
    candidate: ClipCandidate,
    kind: str,
    index: int,
    total: int,
) -> float:
    """Duracion editorial por rol; evita una cadencia mecanica de ventanas iguales."""
    quality = max(0.0, min(1.0, (candidate.score - 0.35) / 0.60))
    if kind == "teaser":
        return round((1.65 + quality * 0.85) * (1.0 + min(index, 2) * 0.08), 3)
    if kind == "hook":
        return round(7.5 + quality * 3.0, 3)
    if kind == "payoff":
        return round(6.5 + quality * 1.8, 3)
    if kind == "setup":
        return round(5.5 + (1.0 - quality) * 0.9, 3)
    if kind == "replay":
        return round(4.5 + quality * 1.2, 3)

    cadence = (0.90, 1.08, 0.96, 1.12, 1.0)
    multiplier = cadence[index % len(cadence)]
    progression = 0.60 * (index / max(1, total - 1))
    return round((4.5 + quality * 2.0 + progression) * multiplier, 3)


def _signal_peak_time(
    candidate: ClipCandidate,
    signals_doc: dict | None,
) -> float | None:
    if not signals_doc:
        return None
    samples: list[tuple[float, float]] = []
    for item in signals_doc.get("loudness") or []:
        try:
            timestamp, level = float(item[0]), float(item[1])
        except (IndexError, TypeError, ValueError):
            continue
        if candidate.start + 0.35 <= timestamp <= candidate.end - 0.35:
            samples.append((timestamp, level))
    if not samples:
        return None
    return max(samples, key=lambda item: item[1])[0]


def _snap_range_to_scenes(
    source_range: SourceRange,
    signals_doc: dict | None,
    video_duration: float,
) -> SourceRange:
    if not signals_doc:
        return source_range
    try:
        scenes = [float(value) for value in signals_doc.get("scene_times") or []]
    except (TypeError, ValueError):
        return source_range
    if not scenes:
        return source_range

    def nearest(value: float) -> float:
        matches = [scene for scene in scenes if abs(scene - value) <= 0.45]
        return min(matches, key=lambda scene: abs(scene - value)) if matches else value

    start = max(0.0, nearest(source_range.start))
    end = min(video_duration, nearest(source_range.end))
    minimum = max(1.2, source_range.duration * 0.75)
    if end - start < minimum:
        return source_range
    return source_range.model_copy(update={"start": round(start, 3), "end": round(end, 3)})


def _range(start: float, end: float, kind: str, score: float, duration: float) -> SourceRange:
    return SourceRange(
        start=round(max(0.0, min(start, duration)), 3),
        end=round(max(0.0, min(end, duration)), 3),
        kind=kind,
        score=score,
    )


def _normalize(text: str) -> str:
    table = str.maketrans("áéíóúüñ", "aeiouun")
    return re.sub(r"\s+", " ", text.translate(table).lower()).strip()
