"""Etapa 1: candidatos por señal cuando el transcript no alcanza.

Calibracion con datos reales de Kerwil (chars/min): podcast 1085 y 1066,
gameplay Warzone 33, gameplay AC Shadows 60.
"""

from app.pipeline.signals import (
    TRANSCRIPT_DENSITY_THRESHOLD,
    action_windows,
    build_action_candidates,
    transcript_density,
)


def _segments(text: str, n: int, span: float) -> list[dict]:
    return [{"start": i * span, "end": (i + 1) * span, "text": text} for i in range(n)]


def test_densidad_separa_podcast_de_gameplay():
    # Podcast real: ~1085 chars/min -> muy por encima del umbral
    podcast = _segments("x" * 180, 200, 10.0)  # 36000 chars en 2000s = 1080/min
    assert transcript_density(podcast, 2000) > TRANSCRIPT_DENSITY_THRESHOLD

    # Gameplay real: ~33 chars/min -> muy por debajo
    gameplay = _segments("Contra todos", 60, 36.7)  # ~720 chars en 2202s
    assert transcript_density(gameplay, 2202) < TRANSCRIPT_DENSITY_THRESHOLD


def test_densidad_sin_duracion_no_divide_por_cero():
    assert transcript_density([{"text": "hola"}], 0) == 0.0


def test_action_windows_prioriza_zona_con_audio_y_escena():
    # Accion concentrada entre 100-160s; el resto plano.
    loudness = [[float(t), -40.0] for t in range(300)]
    for t in range(100, 160):
        loudness[t][1] = -5.0
    signals = {"loudness": loudness, "scene_times": [float(t) for t in range(100, 160, 5)]}

    wins = action_windows(signals, 300.0, window=30.0, top_n=3)

    assert wins, "deberia proponer ventanas"
    best_start, best_end, score, labels = wins[0]
    assert 90 <= best_start <= 160, (
        f"la mejor ventana deberia caer en la zona de accion, dio {best_start}"
    )
    assert score > 0
    assert "picos de audio" in labels


def test_action_windows_no_devuelve_solapadas():
    loudness = [[float(t), -40.0 if t % 50 else -3.0] for t in range(600)]
    signals = {"loudness": loudness, "scene_times": []}
    wins = action_windows(signals, 600.0, window=60.0, top_n=6)
    for i, a in enumerate(wins):
        for b in wins[i + 1 :]:
            assert not (a[0] < b[1] and b[0] < a[1]), f"ventanas solapadas: {a[:2]} y {b[:2]}"


def test_action_windows_sin_señales_devuelve_vacio():
    assert action_windows({}, 300.0, window=30.0, top_n=5) == []
    assert action_windows(None, 300.0, window=30.0, top_n=5) == []
    assert action_windows({"loudness": [], "scene_times": []}, 300.0, window=30.0, top_n=5) == []


def test_action_windows_audio_plano_no_inventa_picos():
    """Sin variacion de volumen no hay picos: no debe proponer por audio."""
    loudness = [[float(t), -30.0] for t in range(300)]
    wins = action_windows({"loudness": loudness, "scene_times": []}, 300.0, window=30.0, top_n=5)
    assert wins == []


def test_build_action_candidates_respeta_duracion_y_rango_de_score():
    loudness = [[float(t), -40.0] for t in range(400)]
    for t in range(50, 120):
        loudness[t][1] = -6.0
    signals = {"loudness": loudness, "scene_times": [60.0, 70.0, 80.0]}

    cands = build_action_candidates(signals, 400.0, num_clips=6, min_seconds=20.0, max_seconds=60.0)

    assert cands
    for c in cands:
        assert c.end > c.start
        assert 20.0 * 0.6 <= (c.end - c.start) <= 60.0 + 0.01
        # Por debajo de un clip justificado por el LLM, por encima del relleno naive.
        assert 0.40 <= c.score <= 0.75
        assert c.reason.startswith("señal:")


def test_podcast_denso_no_recibe_candidatos_por_señal(monkeypatch):
    """El riesgo real del cambio: romper el podcast, que hoy funciona bien.

    Con densidad alta, detect_clips no debe llamar a build_action_candidates.
    """
    from app.pipeline import detect as detect_mod

    llamado = {"n": 0}

    def _spy(*a, **k):
        llamado["n"] += 1
        return []

    monkeypatch.setattr(detect_mod, "build_action_candidates", _spy)

    segments = [
        {"start": i * 5.0, "end": (i + 1) * 5.0, "text": "palabra " * 20} for i in range(100)
    ]
    detect_mod.detect_clips(
        {"segments": segments, "duration": 500.0, "language": "es"},
        provider="naive",
        model="",
        num_clips=3,
        formats=["9x16"],
        processing_mode="podcast_viral",
        signals_doc={"loudness": [[float(t), -20.0] for t in range(500)], "scene_times": []},
    )
    assert llamado["n"] == 0, "un podcast denso no debe activar candidatos por señal"


def test_gameplay_disperso_si_recibe_candidatos_por_señal(monkeypatch):
    from app.pipeline import detect as detect_mod

    llamado = {"n": 0}
    original = detect_mod.build_action_candidates

    def _spy(*a, **k):
        llamado["n"] += 1
        return original(*a, **k)

    monkeypatch.setattr(detect_mod, "build_action_candidates", _spy)

    # 60 segmentos cortos en 2200s = densidad tipo Warzone
    segments = [
        {"start": i * 36.0, "end": i * 36.0 + 3.0, "text": "Contra todos"} for i in range(60)
    ]
    loudness = [[float(t), -40.0] for t in range(2200)]
    for t in range(1500, 1600):
        loudness[t][1] = -5.0
    detect_mod.detect_clips(
        {"segments": segments, "duration": 2200.0, "language": "es"},
        provider="naive",
        model="",
        num_clips=6,
        formats=["9x16"],
        processing_mode="gaming",
        signals_doc={"loudness": loudness, "scene_times": [1520.0, 1540.0, 1560.0]},
    )
    assert llamado["n"] == 1, "gameplay disperso debe activar candidatos por señal"


def test_gameplay_filtra_menus_en_texto_y_senales(monkeypatch):
    """El OCR debe cubrir LLM/fallback y tambien candidatos por senal."""
    from app.core.config import settings
    from app.pipeline import detect as detect_mod
    from app.pipeline import menu_filter
    from app.providers.base import ClipCandidate

    monkeypatch.setattr(settings, "enable_visual_rerank", False, raising=False)
    calls: list[list[str]] = []

    def fake_filter(candidates, video_path, **kwargs):
        calls.append([c.reason for c in candidates])
        return candidates

    monkeypatch.setattr(menu_filter, "filter_menu_candidates", fake_filter)
    monkeypatch.setattr(
        detect_mod,
        "build_action_candidates",
        lambda *args, **kwargs: [
            ClipCandidate(start=120, end=180, score=0.6, reason="señal: audio")
        ],
    )

    segments = [
        {"start": 0.0, "end": 50.0, "text": "¿Que fue eso?"},
        {"start": 60.0, "end": 110.0, "text": "Vamos otra vez."},
    ]
    detect_mod.detect_clips(
        {"segments": segments, "duration": 200.0, "language": "es"},
        provider="naive",
        model="",
        num_clips=2,
        formats=["9x16"],
        processing_mode="gaming",
        signals_doc={"loudness": [[0.0, -20.0]], "scene_times": [120.0]},
        video_path="fake.mp4",
    )

    assert any(any(reason.startswith("Fallback local:") for reason in group) for group in calls)
    assert any(any(reason.startswith("señal:") for reason in group) for group in calls)
