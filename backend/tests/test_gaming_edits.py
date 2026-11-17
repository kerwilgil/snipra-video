import pytest

from app.pipeline.detect import _merge_visual_composition_pool, _visual_pool_size
from app.pipeline.gaming_edits import (
    _compose_hook,
    _is_verified_payoff,
    _payoff_priority,
    _teaser_priority,
    annotate_visual_rewards,
    build_event_candidates,
    choose_gaming_output,
    compose_gaming_edit,
    gaming_detect_params,
    reward_text,
    sanitize_gaming_outcomes,
)
from app.providers.base import ClipCandidate, SourceRange


def _candidate(start: float, end: float, score: float) -> ClipCandidate:
    return ClipCandidate(
        start=start,
        end=end,
        score=score,
        title=f"Jugada {start}",
        reason="señal gaming",
    )


def test_gaming_modes_use_short_event_pools():
    assert gaming_detect_params("gaming_auto", 5) == (20, 6.0, 25.0)
    assert gaming_detect_params("gaming_hook", 1) == (12, 8.0, 25.0)
    assert gaming_detect_params("gaming_highlights", 1) == (16, 6.0, 18.0)


def test_gaming_vision_pool_scales_with_candidate_pool_for_recall():
    """P0.2 (auditoría, hallazgo A): el cap era fijo (4/6) sin importar cuántos
    candidatos había disponibles, así que el mejor candidato real en la
    posición 7+ del pool pre-visión nunca se evaluaba. Ahora escala con el
    pool competidor real (requested_count), con piso >= al cap anterior."""
    assert _visual_pool_size(20, 12, "gaming_hook") == 7
    assert _visual_pool_size(20, 16, "gaming_highlights") == 10
    assert _visual_pool_size(20, 20, "gaming_auto") == 12
    assert _visual_pool_size(3, 16, "gaming_highlights") == 3  # nunca > candidatos disponibles
    assert _visual_pool_size(20, 2, "podcast_viral") == 8  # rama no-gaming sin cambios


def test_gaming_vision_pool_never_exceeds_latency_ceiling():
    """Techo duro para no disparar la latencia (Codex CLI ~75s/llamada)."""
    assert _visual_pool_size(60, 100, "gaming_hook") == 10
    assert _visual_pool_size(60, 100, "gaming_highlights") == 14
    assert _visual_pool_size(60, 100, "gaming_auto") == 14


def test_visual_pool_diversity_prefilter_drops_near_duplicate_candidates():
    """P0.2: el pre-filtro de diversidad (dedupe_by_overlap, ratio=0.5) que
    detect.py aplica ANTES de calcular _visual_pool_size no debe gastar el
    cap ampliado en variantes del mismo pico. Reproduce exactamente la misma
    composición que detect.py::detect_clips (dedupe_by_overlap -> _visual_pool_size)
    sin correr el pipeline completo."""
    from app.providers.base import dedupe_by_overlap

    near_duplicates = [
        _candidate(100 + i, 120 + i, 0.90 - i * 0.01) for i in range(4)
    ]  # mismo pico, ventanas desplazadas 1s -> overlap_ratio > 0.5 entre si
    distinct = _candidate(500, 520, 0.50)

    diversity_pool = dedupe_by_overlap(near_duplicates + [distinct], ratio=0.5)

    assert len(diversity_pool) == 2  # 1 sobreviviente del cluster + el distinto
    assert diversity_pool[0].score == pytest.approx(0.90)  # el mejor del cluster gana
    assert any(c.start == 500 for c in diversity_pool)  # el distinto siempre sobrevive

    cap = _visual_pool_size(len(diversity_pool), 12, "gaming_hook")
    assert cap == len(diversity_pool) == 2  # ningun slot desperdiciado en duplicados


def test_visual_pool_diversity_prefilter_keeps_genuinely_distinct_candidates():
    from app.providers.base import dedupe_by_overlap

    distinct_candidates = [_candidate(i * 100, i * 100 + 15, 0.6) for i in range(5)]

    diversity_pool = dedupe_by_overlap(distinct_candidates, ratio=0.5)

    assert len(diversity_pool) == 5  # ninguno se solapa, todos sobreviven


def test_visual_finalists_do_not_discard_unscored_montage_beats():
    visual = [_candidate(10, 20, 0.90), _candidate(30, 40, 0.85)]
    composition = [
        _candidate(10, 20, 0.70),
        _candidate(30, 40, 0.65),
        _candidate(50, 60, 0.60),
        _candidate(70, 80, 0.55),
    ]

    merged = _merge_visual_composition_pool(visual, composition)

    assert [item.start for item in merged] == [10, 30, 50, 70]


def test_reward_text_recognizes_victory_and_final_kill():
    assert reward_text(["VICTORIA", "PUNTUACION"])[0] is True
    found, label = reward_text(["BAJA FINAL"])
    assert found is True
    assert "baja final" in label
    assert reward_text(["municion", "equipo"])[0] is False


def test_reward_ocr_penalizes_own_defeat_instead_of_rewarding_elimination(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-2]).write_bytes(b"frame")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", lambda path: ["eliminacion", "teelimino:"])
    candidate = _candidate(400, 430, 0.70)

    result = annotate_visual_rewards([candidate], "fake.mp4")

    assert result[0].score == pytest.approx(0.48)
    assert "pantalla de derrota" in result[0].reason
    assert "recompensa visual" not in result[0].reason


def test_signal_event_keeps_tail_for_payoff():
    signals = {
        "loudness": [[float(i), -12.0 if i in {10, 11, 12} else -40.0] for i in range(60)],
        "scene_times": [9.0, 12.0, 18.0],
    }
    events = build_event_candidates(signals, [], 60.0, top_n=8)
    assert events
    event = events[0]
    assert event.end - event.start <= 25.0
    assert event.end >= 22.0
    assert "desenlace" in event.reason


def test_hook_has_real_edit_duration_and_source_range():
    result = compose_gaming_edit([_candidate(100, 122, 0.91)], "gaming_hook", 180)[0]
    assert 8 <= result.duration <= 11
    assert len(result.source_ranges) == 1
    assert result.source_ranges[0].kind == "hook"
    assert 100 <= result.source_ranges[0].start < result.source_ranges[0].end <= 122
    assert "centrado en el pico" in result.reason


def test_hook_appends_verified_final_kill_replay():
    event = _candidate(523.66, 545.83, 0.95).model_copy(
        update={"reason": "payoff; recompensa visual: victoria @543.200s"}
    )
    replay = _candidate(532.0, 556.0, 0.75).model_copy(
        update={"reason": "recompensa visual: baja final @553.600s"}
    )

    result = compose_gaming_edit([event, replay], "gaming_hook", 576.969)[0]

    assert len(result.source_ranges) == 2
    assert result.source_ranges[0].kind == "payoff"
    assert result.source_ranges[0].start <= 543.2 <= result.source_ranges[0].end
    assert result.source_ranges[1].kind == "replay"
    assert result.source_ranges[1].start <= 553.6 <= result.source_ranges[1].end
    assert 12 <= result.duration <= 16


def test_highlights_builds_microfragment_montage_and_weighted_score():
    candidates = [
        _candidate(170, 182, 0.95),
        _candidate(20, 32, 0.80),
        _candidate(50, 62, 0.75),
        _candidate(100, 112, 0.70),
        _candidate(140, 152, 0.65),
    ]
    result = compose_gaming_edit(candidates, "gaming_highlights", 200)[0]

    assert result.title == "Gaming Highlights"
    assert 4 <= len(result.source_ranges) <= 8
    assert 25 <= result.duration <= 45
    assert result.source_ranges[0].kind == "teaser"
    assert 1.5 <= result.source_ranges[0].duration < 3.0
    assert result.source_ranges[0].score == 0.80
    assert result.source_ranges[-1].kind == "payoff"
    assert result.source_ranges[-1].end == 182
    assert result.source_ranges[1].kind == "setup"
    assert all(item.kind == "escalation" for item in result.source_ranges[2:-1])
    assert [item.start for item in result.source_ranges[1:]] == sorted(
        item.start for item in result.source_ranges[1:]
    )
    assert len({round(item.duration, 1) for item in result.source_ranges}) > 1
    assert "score ponderado" in result.reason
    for index, left in enumerate(result.source_ranges):
        for right in result.source_ranges[index + 1 :]:
            assert min(left.end, right.end) <= max(left.start, right.start)


def test_highlights_never_opens_with_finale_then_jumps_to_start():
    candidates = [
        _candidate(531.66, 541.66, 0.829),
        _candidate(84.0, 94.0, 0.782),
        _candidate(420.0, 430.0, 0.662),
        _candidate(516.0, 526.0, 0.675),
    ]

    result = compose_gaming_edit(candidates, "gaming_highlights", 576.969)[0]

    assert result.source_ranges[0].start < 500
    assert result.source_ranges[-1].start >= 531
    assert result.source_ranges[-1].kind == "payoff"
    assert [item.start for item in result.source_ranges[1:]] == sorted(
        item.start for item in result.source_ranges[1:]
    )


def test_highlights_payoff_includes_verified_reward_timestamp():
    candidates = [
        _candidate(84.0, 94.0, 0.78),
        _candidate(420.0, 430.0, 0.70),
        _candidate(523.66, 541.66, 0.95).model_copy(
            update={"reason": "recompensa visual: victoria @543.613s"}
        ),
    ]

    result = compose_gaming_edit(candidates, "gaming_highlights", 576.969)[0]
    payoff = result.source_ranges[-1]

    assert payoff.kind == "payoff"
    assert 6.5 <= payoff.duration <= 8.5
    assert payoff.start <= 543.613 <= payoff.end


def test_highlights_rejects_visually_weak_death_screen_as_teaser():
    candidates = [
        _candidate(420, 430, 0.90).model_copy(
            update={
                "reason": (
                    "visual: kill cam, pero acción de gameplay poco clara y "
                    "sin momento de máxima intensidad visual"
                )
            }
        ),
        _candidate(290, 300, 0.80).model_copy(update={"reason": "visual: tiroteo claro"}),
        _candidate(220, 230, 0.75).model_copy(update={"reason": "visual: acción clara"}),
        _candidate(530, 546, 0.95).model_copy(
            update={"reason": "recompensa visual: victoria @543.6s"}
        ),
    ]

    result = compose_gaming_edit(candidates, "gaming_highlights", 576.969)[0]

    assert result.source_ranges[0].kind == "teaser"
    assert 290 <= result.source_ranges[0].start < 300


def test_highlights_places_visual_peak_near_start_of_fragment():
    candidates = [
        _candidate(100, 120, 0.88).model_copy(
            update={"reason": "visual: baja clara; pico visual frame 4/4"}
        ),
        _candidate(200, 220, 0.75),
        _candidate(300, 320, 0.70),
    ]

    result = compose_gaming_edit(candidates, "gaming_highlights", 360)[0]
    teaser = result.source_ranges[0]
    visual_peak = 100 + 20 * 4 / 5

    assert 0 < visual_peak - teaser.start <= 1.0
    assert teaser.start < visual_peak < teaser.end


def test_highlights_prefers_verified_payoff_over_later_neutral_candidate():
    candidates = [
        _candidate(20, 36, 0.78),
        _candidate(90, 106, 0.74),
        _candidate(150, 170, 0.82).model_copy(
            update={"reason": "recompensa visual: victoria @166.0s"}
        ),
        _candidate(230, 246, 0.90),
    ]

    result = compose_gaming_edit(candidates, "gaming_highlights", 260)[0]
    payoff = result.source_ranges[-1]

    assert payoff.kind == "payoff"
    assert payoff.start <= 166.0 <= payoff.end
    assert payoff.end < 200


def test_highlights_uses_audio_peak_to_place_action_near_teaser_start():
    candidates = [
        _candidate(100, 120, 0.90),
        _candidate(200, 220, 0.75),
        _candidate(300, 320, 0.70),
    ]
    signals = {
        "loudness": [[float(t), -8.0 if t == 118 else -35.0] for t in range(360)],
        "scene_times": [],
    }

    result = compose_gaming_edit(
        candidates,
        "gaming_highlights",
        360,
        signals_doc=signals,
    )[0]
    teaser = result.source_ranges[0]

    assert teaser.start <= 118 <= teaser.end
    assert 118 - teaser.start <= 0.9


def test_final_qa_trims_scoreboard_from_tail(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-2]).write_bytes(b"frame")

    def fake_texts(path):
        return ["scoreboard"] if "qa-0-0-2" in path else []

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", fake_texts)
    edit = _candidate(10, 80, 0.8).model_copy(
        update={
            "source_ranges": [
                SourceRange(start=10, end=20, kind="setup", score=0.75),
                SourceRange(start=70, end=80, kind="payoff", score=0.85),
            ]
        }
    )

    result = sanitize_gaming_outcomes([edit], "fake.mp4")[0]

    assert len(result.source_ranges) == 2
    assert result.source_ranges[0].end < 20
    assert result.source_ranges[-1].kind == "payoff"
    assert "menus o marcadores" in result.reason


def test_final_qa_cleans_scoreboard_at_payoff_entry(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-2]).write_bytes(b"frame")

    def fake_texts(path):
        return ["marcador", "puntuacion"] if "qa-0-0-1" in path else []

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", fake_texts)
    payoff = SourceRange(start=100, end=110, kind="payoff", score=0.90)
    edit = _candidate(100, 110, 0.9).model_copy(update={"source_ranges": [payoff]})

    result = sanitize_gaming_outcomes([edit], "fake.mp4")[0]
    cleaned = result.source_ranges[0]

    assert cleaned.kind == "payoff"
    assert cleaned.start > payoff.start
    assert cleaned.end == payoff.end
    assert cleaned.duration >= 3.0


def test_final_qa_trims_teaser_before_own_defeat(monkeypatch):
    import app.pipeline.gaming_edits as mod

    def fake_run(cmd, **kwargs):
        from pathlib import Path

        Path(cmd[-2]).write_bytes(b"frame")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    monkeypatch.setattr(mod, "frame_texts", lambda path: ["teelimino:"])
    edit = compose_gaming_edit(
        [_candidate(80, 96, 0.8), _candidate(200, 216, 0.7)],
        "gaming_highlights",
        240,
    )[0]
    original = edit.source_ranges[0]

    result = sanitize_gaming_outcomes([edit], "fake.mp4")[0]

    matching = [item for item in result.source_ranges if item.start == original.start]
    assert not matching or matching[0].end < original.end
    assert all(item.duration >= 1.2 for item in result.source_ranges)
    assert "QA OCR" in result.reason


def test_gaming_auto_chooses_verified_hook():
    payoff = _candidate(520, 545, 0.92).model_copy(
        update={"reason": "payoff en comentario; recompensa visual: victoria"}
    )
    assert choose_gaming_output([payoff], 576) == "hook"

    result = compose_gaming_edit([payoff], "gaming_auto", 576, requested_outputs=5)
    assert len(result) == 1
    assert "automatico eligio hook" in result[0].reason


def test_gaming_auto_chooses_distributed_highlights():
    pool = [
        _candidate(10, 22, 0.80),
        _candidate(90, 102, 0.75),
        _candidate(180, 192, 0.72),
        _candidate(270, 282, 0.68),
    ]
    assert choose_gaming_output(pool, 300) == "highlights"


def test_gaming_auto_keeps_independent_clips_without_enough_evidence():
    pool = [_candidate(10, 22, 0.75), _candidate(60, 72, 0.70)]
    result = compose_gaming_edit(pool, "gaming_auto", 120, requested_outputs=2)
    assert len(result) == 2
    assert all(not item.source_ranges for item in result)
    assert all("clip independiente" in item.reason for item in result)


# ---------------------------------------------------------------------------
# P0.4: _teaser_priority / _payoff_priority — score real antes que conteo
# léxico, semántica diferenciada entre teaser y payoff (ver docs/auditoria-
# motor-deteccion-2026-08-11.md, hallazgo 6).
# ---------------------------------------------------------------------------


def test_teaser_priority_prefers_real_score_over_keyword_count():
    """Caso A: reason verboso + score bajo NO debe ganarle a reason corto + score alto."""
    verbose_low_score = _candidate(10, 20, 0.55).model_copy(
        update={"reason": "visual: tiroteo claro; accion clara; jugada decisiva; killcam"}
    )
    terse_high_score = _candidate(50, 60, 0.92).model_copy(
        update={"reason": "visual: buena jugada"}
    )

    assert _teaser_priority(terse_high_score) > _teaser_priority(verbose_low_score)


def test_teaser_priority_uses_keywords_only_as_tiebreak_at_equal_score():
    """Caso B: a igual score, las keywords sí desempatan."""
    with_markers = _candidate(10, 20, 0.80).model_copy(update={"reason": "visual: tiroteo claro"})
    without_markers = _candidate(50, 60, 0.80).model_copy(
        update={"reason": "visual: sin marcadores"}
    )

    assert _teaser_priority(with_markers) > _teaser_priority(without_markers)


def test_compose_hook_prefers_verified_payoff_over_higher_scoring_unverified_candidate():
    """Caso C: un payoff visual REALMENTE verificado gana aunque otro candidato
    tenga score mayor sin verificación. La garantía viene de que _compose_hook
    solo compara _payoff_priority DENTRO del subconjunto ya verificado
    (verified_payoffs) — el candidato genérico nunca entra a esa comparación."""
    verified = _candidate(100, 115, 0.80).model_copy(
        update={"reason": "recompensa visual: victoria verificada @110.0s"}
    )
    generic_higher_score = _candidate(200, 215, 0.95).model_copy(
        update={"reason": "visual: buena jugada"}
    )
    assert _is_verified_payoff(verified)
    assert not _is_verified_payoff(generic_higher_score)

    result = _compose_hook([verified, generic_higher_score], duration=300.0)

    assert result.source_ranges[0].kind == "payoff"
    assert result.reason.startswith(verified.reason)


def test_compose_hook_does_not_auto_win_with_unverified_payoff_mention():
    """Caso D: mencionar "payoff" en el texto sin evidencia real no debe ganar
    automáticamente — debe competir por score como cualquier otro candidato."""
    unverified_mention = _candidate(100, 115, 0.50).model_copy(
        update={"reason": "posible payoff, sin confirmar"}
    )
    stronger_candidate = _candidate(200, 215, 0.90).model_copy(
        update={"reason": "visual: buena jugada"}
    )
    assert not _is_verified_payoff(unverified_mention)

    result = _compose_hook([unverified_mention, stronger_candidate], duration=300.0)

    assert result.source_ranges[0].kind == "hook"
    assert result.reason.startswith(stronger_candidate.reason)


def test_payoff_priority_ranks_by_score_among_verified_candidates():
    """Dentro de ya-verificados, el score real decide, no victory/timestamped
    como primer criterio (bug de tupla lexicográfica corregido)."""
    weaker_but_timestamped = _candidate(10, 20, 0.55).model_copy(
        update={"reason": "recompensa visual: victoria verificada @15.0s"}
    )
    stronger_verified = _candidate(50, 60, 0.90).model_copy(
        update={"reason": "recompensa visual: victoria verificada"}
    )
    assert _is_verified_payoff(weaker_but_timestamped)
    assert _is_verified_payoff(stronger_verified)

    assert _payoff_priority(stronger_verified) > _payoff_priority(weaker_but_timestamped)
