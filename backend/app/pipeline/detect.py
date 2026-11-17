"""Detección de momentos: orquesta transcript → proveedor → candidatos saneados."""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.core.config import settings
from app.core.processing_modes import is_gaming_edit_mode, is_gaming_mode
from app.core.sanitize import sanitize_error_text
from app.pipeline import sponsorblock
from app.pipeline.gaming_edits import (
    annotate_visual_rewards,
    build_event_candidates,
    compose_gaming_edit,
    gaming_detect_params,
)
from app.pipeline.signals import (
    TRANSCRIPT_DENSITY_THRESHOLD,
    apply_signal_scores,
    build_action_candidates,
    transcript_density,
)
from app.providers.base import (
    ClipCandidate,
    DetectParams,
    clamp_candidates,
    dedupe_by_overlap,
    overlap_ratio,
    snap_to_segment_boundaries,
)
from app.providers.naive import NaiveDetector
from app.providers.registry import get_detector

logger = logging.getLogger(__name__)

DetectionEvent = Callable[[str, str], None]
DetectionProgress = Callable[[float, str], None]


def _visual_pool_size(
    candidate_count: int,
    requested_count: int,
    processing_mode: str,
) -> int:
    """Acota la fase cara de visión priorizando recall (P0.2 / hallazgo A de la auditoría).

    Antes: cap FIJO de 4 (hook) / 6 (highlights, auto) sin importar cuántos
    candidatos había disponibles -> el mejor candidato real en la posición
    7+ del pool pre-visión nunca se evaluaba. Ahora escala con el pool
    competidor real (requested_count, ya expandido por gaming_detect_params),
    con piso >= al cap anterior (no degrada el caso ya cubierto) y techo
    absoluto para controlar latencia con proveedores de suscripción (Codex
    CLI mide ~75s/llamada, ver providers/vision.py).
    """
    if is_gaming_edit_mode(processing_mode):
        floor = 6 if processing_mode == "gaming_hook" else 10
        ceiling = 10 if processing_mode == "gaming_hook" else 14
        cap = max(floor, min(ceiling, round(requested_count * 0.6)))
        return min(candidate_count, cap)
    return min(candidate_count, max(requested_count * 2, requested_count + 4, 8))


def _progress_slice(
    callback: DetectionProgress | None,
    start: float,
    end: float,
) -> DetectionProgress | None:
    if callback is None:
        return None

    def update(progress: float, message: str) -> None:
        callback(start + (end - start) * max(0.0, min(1.0, progress)), message)

    return update


def _merge_visual_composition_pool(
    visual_pool: list[ClipCandidate],
    composition_pool: list[ClipCandidate],
) -> list[ClipCandidate]:
    merged = list(visual_pool)
    for candidate in composition_pool:
        if not any(overlap_ratio(candidate, seen) > 0.72 for seen in merged):
            merged.append(candidate)
    return sorted(merged, key=lambda item: item.score, reverse=True)


def _apply_sponsorblock_filter(
    candidates: list[ClipCandidate],
    sponsorblock_segments: list[dict] | None,
    min_seconds: float,
    event_callback: DetectionEvent | None = None,
) -> list[ClipCandidate]:
    """Filtra candidatos por tramos SponsorBlock. Nunca deja el pool en 0 por esto."""
    if not sponsorblock_segments or not candidates:
        return candidates
    filtered = sponsorblock.filter_candidates_by_sponsor_segments(
        candidates, sponsorblock_segments, min_seconds
    )
    if not filtered:
        logger.warning(
            "SponsorBlock dejó el pool de candidatos en 0 tras filtrar; usando el pool sin filtrar."
        )
        if event_callback:
            event_callback(
                "warning",
                "SponsorBlock excluyó todos los clips candidatos; se usó el pool sin filtrar.",
            )
        return candidates
    if len(filtered) < len(candidates) and event_callback:
        event_callback(
            "info",
            f"SponsorBlock excluyó/recortó {len(candidates) - len(filtered)} "
            "clip(s) que caían en tramos de sponsor.",
        )
    return filtered


def _snap_to_scene(t: float, scene_times: list[float], tol: float = 2.5) -> float:
    """Acerca el inicio al cambio de escena más cercano dentro de ±tol (corte limpio)."""
    best = t
    best_d = tol
    for s in scene_times:
        d = abs(s - t)
        if d <= best_d:
            best = s
            best_d = d
    return best


def detect_music_clips(
    duration: float,
    signals_doc: dict | None,
    *,
    num_clips: int,
    min_seconds: float,
    max_seconds: float,
    sponsorblock_segments: list[dict] | None = None,
    event_callback: DetectionEvent | None = None,
) -> list[ClipCandidate]:
    """Genera clips para video musical / sin voz: sin transcripción ni LLM.

    Reparte num_clips tramos por el video (duración ~media del rango pedido) y los
    alinea al cambio de escena más cercano para que el corte caiga en un punto visual
    limpio. No produce subtítulos (no hay texto fiable en música).
    """
    if duration <= 0:
        return []

    clip_len = max(min_seconds, min(max_seconds, (min_seconds + max_seconds) / 2))
    clip_len = min(clip_len, duration)
    max_fit = max(1, int(duration // max(1.0, min_seconds)))
    n = max(1, min(num_clips, max_fit))
    scene_times = sorted(float(t) for t in (signals_doc or {}).get("scene_times", []))

    span = max(0.0, duration - clip_len)
    out: list[ClipCandidate] = []
    seen: set[float] = set()
    for i in range(n):
        start = (span * i / (n - 1)) if n > 1 else 0.0
        start = max(0.0, min(_snap_to_scene(start, scene_times), span))
        end = min(start + clip_len, duration)
        key = round(start, 1)
        if key in seen or end - start < min_seconds * 0.85:
            continue
        seen.add(key)
        out.append(
            ClipCandidate(
                start=round(start, 3),
                end=round(end, 3),
                title=f"Clip musical {len(out) + 1}",
                score=0.5,
                reason="modo música: corte por escena/tiempo",
            )
        )
    return _apply_sponsorblock_filter(out, sponsorblock_segments, min_seconds, event_callback)


def detect_clips(
    segments_doc: dict,
    *,
    provider: str,
    model: str,
    num_clips: int,
    formats: list[str],
    processing_mode: str = "auto",
    signals_doc: dict | None = None,
    video_path: str | None = None,
    event_callback: DetectionEvent | None = None,
    progress_callback: DetectionProgress | None = None,
    sponsorblock_segments: list[dict] | None = None,
    semantic_cards: bool = False,
) -> list[ClipCandidate]:
    segments = segments_doc.get("segments", [])
    total = segments_doc.get("duration") or (segments[-1]["end"] if segments else 0.0)
    requested_outputs = num_clips
    special_gaming_edit = is_gaming_edit_mode(processing_mode)
    pool_size, min_seconds, max_seconds = gaming_detect_params(processing_mode, requested_outputs)

    params = DetectParams(
        num_clips=pool_size,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        formats=formats,
        language=segments_doc.get("language", ""),
        processing_mode=processing_mode,
        semantic_cards=semantic_cards,
    )

    if progress_callback:
        progress_callback(0.02, "Consultando detector de momentos")
    detector = get_detector(provider, model)
    try:
        candidates = detector.detect(segments, params)
    except Exception as exc:  # noqa: BLE001
        fallback_note = (
            "intentando Claude Code antes del fallback local"
            if provider == "codex"
            else "usando fallback local"
        )
        logger.warning(
            "Proveedor '%s' falló (%s); %s.", provider, sanitize_error_text(str(exc)), fallback_note
        )
        if event_callback:
            event_callback(
                "warning",
                f"Proveedor {provider}/{model or 'default'} falló durante la detección; "
                f"{fallback_note}. Error: {type(exc).__name__}: "
                f"{sanitize_error_text(str(exc))[:220]}",
            )
        candidates = []
        if provider == "codex":
            from app.providers.cli_subscription import resolve_provider_cli
            from app.providers.registry import DEFAULT_MODELS

            if resolve_provider_cli("claude_code") is not None:
                try:
                    if event_callback:
                        event_callback(
                            "warning",
                            "Detección Codex falló; se usa Claude Code (suscripción) "
                            "antes del fallback local.",
                        )
                    detector = get_detector(
                        "claude_code",
                        DEFAULT_MODELS["claude_code"],
                    )
                    candidates = detector.detect(segments, params)
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.warning(
                        "Fallback Claude Code también falló (%s). Usando naive.",
                        sanitize_error_text(str(fallback_exc)),
                    )
                    if event_callback:
                        event_callback(
                            "warning",
                            "Claude Code también falló durante la detección; "
                            f"usando fallback local. Error: {type(fallback_exc).__name__}: "
                            f"{sanitize_error_text(str(fallback_exc))[:180]}",
                        )
        if not candidates:
            candidates = NaiveDetector().detect(segments, params)

    if not candidates and not isinstance(detector, NaiveDetector):
        if event_callback:
            event_callback(
                "warning",
                f"Proveedor {provider}/{model or 'default'} devolvió 0 clips; "
                "usando fallback local.",
            )
        candidates = NaiveDetector().detect(segments, params)

    candidates = apply_signal_scores(candidates, signals_doc, processing_mode=processing_mode)
    if progress_callback:
        progress_callback(0.12, "Candidatos iniciales listos")

    # Etapa 1: cuando casi no hay transcript, el detector de texto no puede proponer
    # momentos (gameplay sin narracion da ~33 chars/min contra ~1080 de un podcast).
    # Se agregan candidatos derivados de audio/escena AL POOL, sin reemplazar a los
    # del texto: si el LLM encontro momentos reales, siguen compitiendo por score.
    density = transcript_density(segments, total)
    if special_gaming_edit:
        action = build_event_candidates(
            signals_doc,
            segments,
            total,
            top_n=pool_size,
        )
        candidates = candidates + action
        if candidates:
            from app.pipeline.menu_filter import filter_menu_candidates

            candidates = filter_menu_candidates(
                candidates,
                video_path,
                event_callback=event_callback,
                progress_callback=_progress_slice(progress_callback, 0.12, 0.35),
            )
        if event_callback:
            event_callback(
                "info",
                f"Edit gaming: {len(candidates)} eventos compiten antes del montaje final.",
            )
    elif density < TRANSCRIPT_DENSITY_THRESHOLD:
        action = build_action_candidates(
            signals_doc,
            total,
            num_clips=num_clips,
            min_seconds=params.min_seconds,
            max_seconds=params.max_seconds,
        )
        # Se filtra el pool combinado para cubrir LLM, fallback y senales en una
        # sola pasada. Asi el seguro "nunca dejar el pool vacio" no reintroduce un
        # menu cuando uno de los origenes solo aporto un candidato.
        if is_gaming_mode(processing_mode) and (candidates or action):
            from app.pipeline.menu_filter import filter_menu_candidates

            combined = candidates + action
            filtered = filter_menu_candidates(
                combined,
                video_path,
                event_callback=event_callback,
                progress_callback=_progress_slice(progress_callback, 0.12, 0.35),
            )
            kept_ids = {id(candidate) for candidate in filtered}
            candidates = [candidate for candidate in candidates if id(candidate) in kept_ids]
            action = [candidate for candidate in action if id(candidate) in kept_ids]
        # Solo los que no pisan un candidato existente: si el texto ya cubrio ese
        # momento, gana el del texto porque trae justificacion de contenido.
        added = [a for a in action if not any(overlap_ratio(a, c) > 0.30 for c in candidates)]
        if added:
            candidates = candidates + added
            if event_callback:
                event_callback(
                    "info",
                    f"Transcript de baja densidad ({density:.0f} caracteres/min): "
                    f"se añadieron {len(added)} candidatos por señal de audio/escena.",
                )
        elif event_callback and not action:
            event_callback(
                "warning",
                f"Transcript de baja densidad ({density:.0f} caracteres/min) y sin señales "
                "de audio/escena utilizables; la seleccion puede ser pobre.",
            )
    elif is_gaming_mode(processing_mode) and candidates:
        from app.pipeline.menu_filter import filter_menu_candidates

        candidates = filter_menu_candidates(
            candidates,
            video_path,
            event_callback=event_callback,
            progress_callback=_progress_slice(progress_callback, 0.12, 0.35),
        )

    # Re-rank visual opcional: amplía el pool, deja que el modelo con visión reordene
    # por lo que se VE, y recién entonces recorta a num_clips. Así la visión puede
    # rescatar clips que el texto no priorizó (p. ej. acción de gameplay sin comentario).
    if special_gaming_edit:
        candidates = annotate_visual_rewards(
            sorted(candidates, key=lambda item: item.score, reverse=True),
            video_path,
            progress_callback=_progress_slice(progress_callback, 0.35, 0.50),
        )
    elif progress_callback:
        progress_callback(0.45, "Candidatos ordenados por señales")

    if getattr(settings, "enable_visual_rerank", False) and video_path:
        from app.providers.vision import VISION_PROVIDERS, rerank_by_vision, resolve_vision

        # El proveedor de vision es independiente del de texto (Ajustes). Con
        # "auto" sigue al de texto, para que elegir una suscripcion no termine
        # llamando a una API de pago sin que el usuario lo haya pedido.
        vision_provider, vision_model = resolve_vision(provider, model)
        if vision_provider:
            composition_pool = (
                clamp_candidates(candidates, params, total_duration=total)
                if special_gaming_edit
                else []
            )
            if event_callback:
                event_callback(
                    "info",
                    f"Analisis visual: {vision_provider} / {vision_model or 'modelo por defecto'}.",
                )
            # Pre-filtro barato de diversidad: sin esto, 3-4 candidatos casi
            # idénticos (mismo pico, ventanas desplazadas) pueden ocupar todo
            # el cap ampliado y dejar fuera un momento distinto con score
            # similar. Umbral 0.5, calibrar con score_breakdown/benchmark
            # antes de tocarlo de nuevo (no preserva ningún call site previo).
            diversity_pool = dedupe_by_overlap(candidates, ratio=0.5)
            visual_pool_size = _visual_pool_size(
                len(diversity_pool),
                params.num_clips,
                processing_mode,
            )
            if event_callback:
                event_callback(
                    "info",
                    f"Visión IA: evaluando {visual_pool_size} candidato(s) preseleccionados.",
                )
            pool_params = params.model_copy(update={"num_clips": visual_pool_size})
            pool = clamp_candidates(diversity_pool, pool_params, total_duration=total)
            if not pool:
                logger.warning("Visual rerank: pool vacío tras clamp. Usando fallback naive.")
                if event_callback:
                    event_callback(
                        "warning",
                        "Re-rank visual quedó sin candidatos válidos; usando fallback local.",
                    )
                pool = clamp_candidates(
                    NaiveDetector().detect(segments, params), params, total_duration=total
                )
            else:
                before = [(c.start, c.score) for c in pool]
                pool = rerank_by_vision(
                    pool,
                    video_path,
                    provider=vision_provider,
                    model=vision_model,
                    progress_callback=_progress_slice(progress_callback, 0.50, 0.98),
                    event_callback=event_callback,
                )
                # rerank_by_vision nunca rompe el job: ante cualquier error
                # devuelve la lista intacta. Sin este aviso, quedarse sin cuota
                # de la suscripcion se ve exactamente igual que un exito.
                if [(c.start, c.score) for c in pool] == before and event_callback:
                    event_callback(
                        "warning",
                        f"El analisis visual con {vision_provider} no modifico ningun clip. "
                        "Suele indicar que el CLI fallo o que no queda cuota; los clips "
                        "quedaron ordenados solo por texto/señal.",
                    )
            if special_gaming_edit:
                # Vision solo puntua 4-6 finalistas por velocidad, pero Highlights
                # necesita mas beats para un montaje con ritmo. Se reincorporan
                # candidatos saneados que no dupliquen los ya evaluados.
                pool = _merge_visual_composition_pool(pool, composition_pool)
                pool = _apply_sponsorblock_filter(
                    pool, sponsorblock_segments, params.min_seconds, event_callback
                )
                if progress_callback:
                    progress_callback(1.0, "Detección completada")
                from app.pipeline.gaming_edits import sanitize_gaming_outcomes

                return sanitize_gaming_outcomes(
                    compose_gaming_edit(
                        pool,
                        processing_mode,
                        total,
                        requested_outputs=requested_outputs,
                        signals_doc=signals_doc,
                    ),
                    video_path,
                )
            candidates = pool[:requested_outputs]
            snapped = snap_to_segment_boundaries(candidates, segments, params, total_duration=total)
            if progress_callback:
                progress_callback(1.0, "Detección completada")
            return _apply_sponsorblock_filter(
                snapped, sponsorblock_segments, params.min_seconds, event_callback
            )
        # El analisis visual estaba PEDIDO pero el proveedor no lo soporta. Sin este
        # aviso el job termina normal y nadie se entera de que la opcion no hizo
        # nada: es el mismo fallo silencioso que tenia NVIDIA con frames de 512px.
        logger.warning(
            "Analisis visual pedido pero el proveedor '%s' no tiene vision; se omite.", provider
        )
        if event_callback:
            event_callback(
                "warning",
                f"El analisis visual esta activado, pero el proveedor '{provider}' no lo "
                "soporta y se omitio. Proveedores con vision: "
                f"{', '.join(sorted(VISION_PROVIDERS))}.",
            )

    if special_gaming_edit:
        pool = clamp_candidates(candidates, params, total_duration=total)
        if not pool:
            pool = clamp_candidates(
                NaiveDetector().detect(segments, params), params, total_duration=total
            )
        pool = _apply_sponsorblock_filter(
            pool, sponsorblock_segments, params.min_seconds, event_callback
        )
        if progress_callback:
            progress_callback(1.0, "Detección completada")
        from app.pipeline.gaming_edits import sanitize_gaming_outcomes

        return sanitize_gaming_outcomes(
            compose_gaming_edit(
                pool,
                processing_mode,
                total,
                requested_outputs=requested_outputs,
                signals_doc=signals_doc,
            ),
            video_path,
        )

    candidates = clamp_candidates(candidates, params, total_duration=total)
    if not candidates:
        logger.warning("Pipeline principal devolvió 0 clips. Usando fallback naive.")
        if event_callback:
            event_callback("warning", "Pipeline principal devolvió 0 clips; usando fallback local.")
        candidates = clamp_candidates(
            NaiveDetector().detect(segments, params), params, total_duration=total
        )
    elif len(candidates) < params.num_clips:
        # LLM devolvió menos clips que los pedidos → completa con naive por zona libre
        logger.warning(
            "LLM devolvió %d/%d clips. Completando con naive.", len(candidates), params.num_clips
        )
        if event_callback:
            event_callback(
                "warning",
                f"LLM devolvió {len(candidates)}/{params.num_clips} clips; "
                "completando con fallback local.",
            )
        remaining = params.num_clips - len(candidates)

        # Zonas del video no cubiertas por ningún candidato existente.
        # Se opera zona a zona para evitar que naive construya ventanas que cruzan
        # huecos y terminen solapando clips del LLM.
        sorted_cands = sorted(candidates, key=lambda c: c.start)
        boundaries = [0.0]
        for c in sorted_cands:
            boundaries += [c.start, c.end]
        boundaries.append(total)
        free_zones = [
            (boundaries[i], boundaries[i + 1])
            for i in range(0, len(boundaries) - 1, 2)
            if boundaries[i + 1] - boundaries[i] >= params.min_seconds * 0.8
        ]

        filler: list[ClipCandidate] = []
        for zone_start, zone_end in free_zones:
            if len(filler) >= remaining:
                break
            zone_segs = [
                s
                for s in segments
                if float(s.get("start", 0)) < zone_end
                and float(s.get("end", s.get("start", 0))) > zone_start
            ]
            if not zone_segs:
                continue
            zone_params = params.model_copy(update={"num_clips": 3})
            zone_clips = clamp_candidates(
                NaiveDetector().detect(zone_segs, zone_params),
                zone_params,
                total_duration=total,
            )
            if zone_clips:
                filler.append(zone_clips[0])

        # Fallback: si aún faltan, búsqueda amplia sobre todo el video
        if len(filler) < remaining:
            extra_params = params.model_copy(update={"num_clips": remaining * 4})
            existing = candidates + filler
            for nc in clamp_candidates(
                NaiveDetector().detect(segments, extra_params), extra_params, total_duration=total
            ):
                if len(filler) >= remaining:
                    break
                if not any(overlap_ratio(nc, c) > 0.30 for c in existing):
                    filler.append(nc)

        for nc in filler:
            if len(candidates) >= params.num_clips:
                break
            if not any(overlap_ratio(nc, c) > 0.30 for c in candidates):
                candidates.append(nc)
        if is_gaming_mode(processing_mode) and filler:
            from app.pipeline.menu_filter import filter_menu_candidates

            candidates = filter_menu_candidates(
                candidates, video_path, event_callback=event_callback
            )
    snapped = snap_to_segment_boundaries(candidates, segments, params, total_duration=total)
    if progress_callback:
        progress_callback(1.0, "Detección completada")
    return _apply_sponsorblock_filter(
        snapped, sponsorblock_segments, params.min_seconds, event_callback
    )
