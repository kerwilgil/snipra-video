"""Tests de F3: reframe, subtítulos .ass y render real por formato (FFmpeg)."""

import json

from app.core.processing_modes import reframe_request_for_processing_mode
from app.pipeline.reframe import auto_reframe_mode, build_filtergraph, resolve_mode, target_size
from app.pipeline.render import (
    QualityReport,
    _encode_with_quality_gate,
    _final_filtergraph,
    _legacy_slug,
    _normalized_block_words,
    _slug,
    _write_timeline_artifact,
    render_clip,
)
from app.pipeline.subtitles import build_ass
from app.pipeline.tracking import SubjectTrack, build_split_graph, build_track_graph
from app.providers.base import ClipCandidate, EditorialCard, ScoreContribution, SourceRange
from tests.conftest import requires_ffmpeg, requires_vision

SEGMENTS = [
    {
        "start": 1.0,
        "end": 4.0,
        "text": "Hola esto es una prueba",
        "words": [
            {"start": 1.0, "end": 1.5, "word": "Hola"},
            {"start": 1.6, "end": 2.0, "word": "esto"},
            {"start": 2.1, "end": 2.4, "word": "es"},
            {"start": 2.5, "end": 3.0, "word": "una"},
            {"start": 3.1, "end": 3.8, "word": "prueba"},
        ],
    }
]


def test_target_sizes():
    assert target_size("9x16") == (1080, 1920)
    assert target_size("1x1") == (1080, 1080)
    assert target_size("16x9") == (1920, 1080)


def test_slug_is_ascii_and_legacy_slug_remains_available():
    assert _slug(0, "Niño: acción 🎬") == "01-nino-accion"
    assert _slug(1, "🎬") == "02-clip"
    assert _legacy_slug(0, "Niño") == "01-niño"


def test_normalized_block_words_supports_word_and_text_keys():
    block_from_auto_detect = {"words": [{"word": "Hola", "start": 0.0, "end": 0.4}]}
    block_from_manual_edit = {"words": [{"text": "Chau", "start": 1.0, "end": 1.4}]}

    assert _normalized_block_words(block_from_auto_detect) == [
        {"word": "Hola", "start": 0.0, "end": 0.4}
    ]
    assert _normalized_block_words(block_from_manual_edit) == [
        {"word": "Chau", "start": 1.0, "end": 1.4}
    ]


def test_write_timeline_artifact_includes_words_per_subtitle_block(tmp_path):
    cand = ClipCandidate(start=0.0, end=5.0, title="Bloques", score=0.5)
    blocks = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Hola mundo",
            "words": [
                {"word": "Hola", "start": 0.0, "end": 0.4},
                {"word": "mundo", "start": 0.5, "end": 1.0},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "Chau",
            "words": [{"text": "Chau", "start": 1.0, "end": 2.0}],
        },
    ]

    _write_timeline_artifact(
        tmp_path,
        candidate=cand,
        formats=[],
        blocks=blocks,
        style_cfg={"style": "viral_bold"},
        processing_mode="auto_smart",
        requested_mode="center",
        effective_start=0.0,
        trim_offset=0.0,
    )

    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert payload["subtitles"][0]["words"] == [
        {"word": "Hola", "start": 0.0, "end": 0.4},
        {"word": "mundo", "start": 0.5, "end": 1.0},
    ]
    assert payload["subtitles"][1]["words"] == [{"word": "Chau", "start": 1.0, "end": 2.0}]
    # el "words" plano de todo el clip también soporta ambas claves por palabra.
    assert {w["word"] for w in payload["words"]} == {"Hola", "mundo", "Chau"}


def test_write_timeline_artifact_includes_score_breakdown(tmp_path):
    """P0.1 (debug timeline pedido en la auditoría): score_breakdown debe
    salir en timeline.json, ya expuesto por GET /clips/{id}/timeline sin
    endpoint nuevo."""
    cand = ClipCandidate(
        start=0.0,
        end=5.0,
        title="Con breakdown",
        score=0.87,
        score_breakdown=[
            ScoreContribution(
                signal="vision_rerank",
                stage="semantic_validation",
                score_before=0.71,
                contribution=0.16,
                score_after=0.87,
                at=38.4,
                note="verified payoff",
            )
        ],
    )

    _write_timeline_artifact(
        tmp_path,
        candidate=cand,
        formats=[],
        blocks=[],
        style_cfg={"style": "viral_bold"},
        processing_mode="gaming_hook",
        requested_mode="center",
        effective_start=0.0,
        trim_offset=0.0,
    )

    payload = json.loads((tmp_path / "timeline.json").read_text(encoding="utf-8"))
    assert payload["clip"]["score_breakdown"] == [
        {
            "signal": "vision_rerank",
            "stage": "semantic_validation",
            "score_before": 0.71,
            "contribution": 0.16,
            "score_after": 0.87,
            "at": 38.4,
            "note": "verified payoff",
        }
    ]


def test_filtergraph_center_and_blur():
    center = build_filtergraph("9x16", "center")
    assert "crop=1080:1920" in center and "[v]" in center
    blur = build_filtergraph("9x16", "blur")
    assert "crop=1080:1920" in blur and "boxblur" not in blur and "overlay" not in blur
    preserve = build_filtergraph("9x16", "preserve")
    assert "force_original_aspect_ratio=increase" in preserve and "crop=1080:1920" in preserve


def test_final_filtergraph_burns_subtitles_after_overlays(monkeypatch):
    from app.pipeline import render

    monkeypatch.setattr(render, "_has_subtitles_filter", lambda: True)
    overlay_graph = "[0:v]scale=1080:1920[bg];[bg][0:v]overlay=(W-w)/2:(H-h)/2[v]"

    graph, vmap = _final_filtergraph(overlay_graph, "subtitles-9x16.ass")

    assert vmap == "[vout]"
    assert graph.index("overlay") < graph.index("unsharp") < graph.index("subtitles=filename")


def test_quality_gate_retries_until_pass(monkeypatch, tmp_path):
    from app.pipeline import render

    attempts = {"n": 0}
    events = []

    def fake_encode(*args, **kwargs):
        attempts["n"] += 1

    def fake_evaluate(*args, **kwargs):
        if attempts["n"] == 1:
            return QualityReport(
                False,
                attempts["n"],
                ["audio faltante"],
                {"attempt": attempts["n"]},
            )
        return QualityReport(True, attempts["n"], [], {"attempt": attempts["n"]})

    monkeypatch.setattr(render, "_encode_format", fake_encode)
    monkeypatch.setattr(render, "_validate_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(render, "_evaluate_render_quality", fake_evaluate)

    report = _encode_with_quality_gate(
        tmp_path,
        "base.mp4",
        "[0:v]scale=1080:1920[v]",
        "out.mp4",
        None,
        fmt="9x16",
        expected_duration=5.0,
        expected_has_audio=True,
        event_callback=lambda level, message: events.append((level, message)),
    )

    assert report.passed is True
    assert attempts["n"] == 2
    assert any(level == "warning" for level, _ in events)
    assert (tmp_path / "quality-9x16.json").exists()


def test_quality_gate_fails_after_three_attempts(monkeypatch, tmp_path):
    from app.pipeline import render

    attempts = {"n": 0}

    def fake_encode(*args, **kwargs):
        attempts["n"] += 1

    monkeypatch.setattr(render, "_encode_format", fake_encode)
    monkeypatch.setattr(render, "_validate_export", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        render,
        "_evaluate_render_quality",
        lambda *args, **kwargs: QualityReport(
            False, attempts["n"], ["segmento negro largo"], {"attempt": attempts["n"]}
        ),
    )

    import pytest

    with pytest.raises(RuntimeError, match="Quality gate fallo tras 3 intentos"):
        _encode_with_quality_gate(
            tmp_path,
            "base.mp4",
            "[0:v]scale=1080:1920[v]",
            "out.mp4",
            None,
            fmt="9x16",
            expected_duration=5.0,
            expected_has_audio=True,
        )

    assert attempts["n"] == 3
    assert (tmp_path / "quality-9x16.json").exists()


def test_auto_reframe_by_aspect_ratio(monkeypatch):
    from app.pipeline import reframe

    # aspecto compatible -> preserve
    assert auto_reframe_mode(1080, 1920, "9x16", "auto") == "preserve"
    assert auto_reframe_mode(1920, 1080, "9x16", "center") == "center"

    # cambio de aspecto grande: usa track; sin [vision], track usa motion-energy fallback.
    monkeypatch.setattr(reframe, "_vision_available", lambda: True)
    monkeypatch.setattr(reframe.settings, "enable_tracking", True)
    assert auto_reframe_mode(1920, 1080, "9x16", "auto") == "track"

    monkeypatch.setattr(reframe, "_vision_available", lambda: False)
    assert auto_reframe_mode(1920, 1080, "9x16", "auto") == "track"


def test_track_degrades_when_disabled(monkeypatch):
    # si enable_tracking=False, track degrada a center aunque haya libs
    from app.core import config

    monkeypatch.setattr(config.settings, "enable_tracking", False)
    assert resolve_mode("track") == "center"


def test_modes_passthrough():
    assert resolve_mode("center") == "center"
    assert resolve_mode("blur") == "blur"


def test_processing_mode_reframe_requests():
    assert reframe_request_for_processing_mode("tutorial", "auto") == "preserve"
    assert reframe_request_for_processing_mode("original", "blur") == "preserve"
    assert reframe_request_for_processing_mode("gaming", "auto") == "auto"


def test_track_graph_holds_stable_crop_for_small_jitter():
    track = SubjectTrack(
        width=1920,
        height=1080,
        samples=[(0.0, 950.0, 540.0), (1.0, 980.0, 540.0), (2.0, 970.0, 540.0)],
    )

    graph = build_track_graph(track, "9x16")

    assert graph is not None
    assert "crop=608:1080" in graph
    assert "if(" not in graph


def test_build_split_graph_produces_two_independent_panes():
    """split-screen: 2 crops independientes (uno por persona) apilados verticalmente."""
    track_top = SubjectTrack(
        width=1920, height=1080, samples=[(0.0, 480.0, 540.0), (1.0, 480.0, 540.0)]
    )
    track_bottom = SubjectTrack(
        width=1920, height=1080, samples=[(0.0, 1440.0, 540.0), (1.0, 1440.0, 540.0)]
    )

    graph = build_split_graph(track_top, track_bottom, "9x16")

    assert graph is not None
    assert "split=2" in graph
    assert "vstack=inputs=2" in graph
    assert graph.count("crop=") == 2
    assert "scale=1080:960" in graph  # cada mitad es la mitad del alto del target


def test_track_graph_step_function_for_large_motion():
    """Movimiento grande → múltiples keyframes pero función escalón, sin paneo lineal."""
    track = SubjectTrack(
        width=1920,
        height=1080,
        samples=[
            (0.0, 420.0, 540.0),
            (1.0, 700.0, 540.0),
            (2.0, 1100.0, 540.0),
            (3.0, 1500.0, 540.0),
        ],
    )

    graph = build_track_graph(track, "9x16")

    assert graph is not None
    assert "if(" in graph  # múltiples posiciones → condicional en la expresión
    # 3 keyframes (420, 1100, 1500; el intermedio 700 no llega a sostenerse el
    # tiempo minimo) * 2 "if(" por transición con ease = 4. Menos que el peor
    # caso sin estabilización (3 transiciones crudas * 2 = 6).
    assert graph.count("if(") < 2 * (len(track.samples) - 1)
    # función escalón: sin interpolación lineal (no hay multiplicación de t)
    import re

    x_expr = re.search(r"x='([^']+)'", graph)
    if x_expr:
        assert "*((t-" not in x_expr.group(1), "la expresión no debe interpolar linealmente"


@requires_vision
def test_analyze_subject_sets_trim_start_for_wide_shot_open(monkeypatch):
    """Gap inicial largo → trim_start señala cuánto recortar; samples reindexados a 0."""
    import cv2
    import numpy as np

    from app.pipeline import tracking

    call_count = {"n": 0}

    def fake_pick(model, frame, width, height, prev_cx):
        call_count["n"] += 1
        # Primeras 5 llamadas: sin detección (plano general)
        if call_count["n"] <= 5:
            return None, None
        # A partir de la 6ª: cara en x=400, y=200
        return 400.0, 200.0

    monkeypatch.setattr(tracking, "_pick_subject", fake_pick)

    frames_given = {"n": 0}

    class FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 25 * 30  # 30s clip
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 360
            return 0

        def set(self, prop, val):
            pass

        def read(self):
            frames_given["n"] += 1
            return True, np.zeros((360, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCap())
    monkeypatch.setattr(tracking, "_get_model", lambda: None)

    result = tracking.analyze_subject("fake.mp4")

    assert result is not None
    assert result.trim_start > 0, "debe señalar trim cuando el inicio no tiene cara"
    # Los samples NO se reindexan — el pipeline de render decide si recortar
    # y re-ejecuta analyze_subject sobre el clip recortado si aplica.
    assert result.samples[0][0] == 0.0, "primer sample tiene t=0 (backfill desde t=0)"
    # La posición del primer sample es la de la primera detección real (backfill)
    assert result.samples[0][1] == 400.0, "backfill usa cx de primera detección"


@requires_vision
def test_analyze_subject_backfills_short_gap(monkeypatch):
    """Gap inicial corto (<2s) → backfill, sin trim."""
    import cv2
    import numpy as np

    from app.pipeline import tracking

    call_count = {"n": 0}

    def fake_pick(model, frame, width, height, prev_cx):
        call_count["n"] += 1
        if call_count["n"] <= 1:  # Solo 1 frame sin detección (~0.8s en clip de 30s/36 samples)
            return None, None
        return 960.0, 540.0

    monkeypatch.setattr(tracking, "_pick_subject", fake_pick)

    class FakeCap:
        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 25 * 30
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 1920
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 1080
            return 0

        def set(self, prop, val):
            pass

        def read(self):
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: FakeCap())
    monkeypatch.setattr(tracking, "_get_model", lambda: None)

    result = tracking.analyze_subject("fake.mp4")

    assert result is not None
    assert result.trim_start == 0.0, "gap corto no debe generar trim"
    assert result.samples[0][0] == 0.0, "primer sample sigue en t=0"


def test_pick_subject_prefers_face_with_headroom(monkeypatch):
    """Con cara visible, el encuadre sigue el rostro (no el torso) + headroom."""
    import numpy as np

    from app.pipeline import tracking

    # Simula YOLO detectando una persona con cara estimada en x=960, y=216
    # (persona cuya cara queda en x=0.5*1920=960, y=0.2*1080=216 aprox.)
    fake_face = (960.0, 216.0, 300.0 * 300, 810.0, 108.0, 1110.0, 324.0)
    monkeypatch.setattr(tracking, "_detect_faces", lambda *a: [fake_face])
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # model=None: _detect_face usa _detect_faces (mockeado), no llega a YOLO
    cx, cy = tracking._pick_subject(None, frame, 1920, 1080, prev_cx=960)
    assert abs(cx - 960) < 1  # centro x de la cara = 960
    # cy cara = 216; + headroom 0.10 * 1080 = 108 → 324
    assert abs(cy - 324) < 1


def test_motion_energy_centers_follow_active_region():
    from app.pipeline import tracking

    width, height = 12, 6
    base = bytearray(width * height)
    moved = bytearray(width * height)
    for y in range(2, 4):
        for x in range(8, 11):
            moved[y * width + x] = 255

    centers = tracking._motion_centers(
        [bytes(base), bytes(moved)],
        source_width=1920,
        source_height=1080,
        sample_width=width,
        sample_height=height,
    )

    assert centers
    cx, cy = centers[0]
    assert cx > 1200
    assert cy == 1080 * 0.42


def test_pick_subject_falls_back_to_body_without_face(monkeypatch):
    """Sin cara detectable, cae a YOLO person box (centro = torso)."""
    import numpy as np

    from app.pipeline import tracking

    monkeypatch.setattr(tracking, "_detect_face", lambda *a: None)

    class _Boxes:
        @staticmethod
        def __len__():
            return 1

        class xyxy:  # noqa: N801
            @staticmethod
            def tolist():
                return [[800.0, 200.0, 1100.0, 900.0]]  # centro x=950, y=550

    class _FakeModel:
        def __call__(self, _frame, classes=None, verbose=False):
            return [type("R", (), {"boxes": _Boxes()})()]

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cx, cy = tracking._pick_subject(_FakeModel(), frame, 1920, 1080, prev_cx=960)
    assert abs(cx - 950) < 1
    assert abs(cy - 550) < 1


def test_pick_subject_active_speaker_picks_open_mouth(monkeypatch):
    """Con varias caras, sigue a la que tiene la boca abierta (la que habla)."""
    import numpy as np

    from app.pipeline import tracking

    # dos caras: izquierda (x=480) callada, derecha (x=1440) hablando
    faces = [
        (480.0, 540.0, 300.0 * 300, 330.0, 390.0, 630.0, 690.0),
        (1440.0, 540.0, 300.0 * 300, 1290.0, 390.0, 1590.0, 690.0),
    ]
    monkeypatch.setattr(tracking, "_detect_faces", lambda *a: faces)
    # la cara derecha (cx≈1440) abre la boca; la izquierda no
    monkeypatch.setattr(
        tracking, "_mouth_openness", lambda _frame, face: 0.6 if face[0] > 1000 else 0.02
    )
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cx, _cy = tracking._pick_subject(None, frame, 1920, 1080, prev_cx=960)
    assert cx == 1440.0  # eligió a quien habla, no a la más cercana al centro


def test_build_ass_from_words():
    ass = build_ass(SEGMENTS, clip_start=0.0, clip_end=5.0, width=1080, height=1920)
    assert "[Events]" in ass
    assert "Dialogue:" in ass
    assert "PRUEBA" in ass  # va en mayúsculas


def test_build_ass_empty_when_no_text():
    assert build_ass([], 0.0, 5.0, 1080, 1920) == ""


def _dialogue_times(ass: str) -> list[tuple[float, float]]:
    def _t(s: str) -> float:
        h, m, rest = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    out = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",")
            out.append((_t(parts[1]), _t(parts[2])))
    return out


def test_build_ass_lines_do_not_overlap():
    # linea corta cuya extension MIN_LINE_SECONDS pisaria a la siguiente -> no debe solapar
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "uno dos tres cuatro cinco seis",
            "words": [
                {"start": 0.0, "end": 0.1, "word": "uno"},
                {"start": 0.1, "end": 0.2, "word": "dos"},
                {"start": 0.2, "end": 0.3, "word": "tres"},
                {"start": 0.3, "end": 0.4, "word": "cuatro"},
                {"start": 0.45, "end": 0.6, "word": "cinco"},
                {"start": 0.6, "end": 0.8, "word": "seis"},
            ],
        }
    ]
    times = sorted(_dialogue_times(build_ass(segments, 0.0, 1.0, 1080, 1920, style="opus")))
    for (_s1, e1), (s2, _e2) in zip(times, times[1:], strict=False):
        assert e1 <= s2 + 0.001, f"subtitulos solapados: fin {e1} > inicio {s2}"


EMOJI_SEGMENTS = [
    {
        "start": 1.0,
        "end": 3.0,
        "text": "Gane mucho dinero",
        "words": [
            {"start": 1.0, "end": 1.5, "word": "Gane"},
            {"start": 1.6, "end": 2.0, "word": "mucho"},
            {"start": 2.1, "end": 2.8, "word": "dinero"},
        ],
    }
]


def test_build_ass_active_word_has_color_and_bold():
    # Animación \t() eliminada — la palabra activa usa solo color + bold.
    opus = build_ass(SEGMENTS, 0.0, 5.0, 1080, 1920, style="opus")
    assert r"\b1}" in opus  # bold en palabra activa
    assert r"\t(" not in opus  # sin animación de escala


def test_build_ass_injects_emoji_for_keyword():
    opus = build_ass(
        EMOJI_SEGMENTS,
        0.0,
        4.0,
        1080,
        1920,
        style="gaming",
        style_config={"emoji_enabled": True, "style": "gaming"},
    )
    assert "💰" in opus  # 'dinero' -> emoji
    clean = build_ass(EMOJI_SEGMENTS, 0.0, 4.0, 1080, 1920, style="clean")
    assert "💰" not in clean  # estilo clean no inyecta emoji


def test_build_ass_social_style_and_fallback_chunks():
    styled = build_ass(SEGMENTS, clip_start=0.0, clip_end=5.0, width=1080, height=1920)
    assert "BorderStyle" in styled
    assert r"\c&H001FD2FF&" in styled

    fallback = build_ass(
        [{"start": 0.0, "end": 5.0, "text": "No vas a creer este truco facil"}],
        0.0,
        5.0,
        1080,
        1920,
    )
    assert "NO" in fallback
    assert "VAS" in fallback
    assert "CREER" in fallback
    assert r"\c&H001FD2FF&" in fallback


@requires_ffmpeg
def test_render_clip_multiformat(sample_video, monkeypatch, tmp_path):
    # redirige el storage a tmp para no ensuciar
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)

    cand = ClipCandidate(start=2.0, end=8.0, title="Clip de prueba", score=0.9)
    assets = render_clip("jobtest", sample_video, 0, cand, ["9x16", "1x1"], segments=SEGMENTS)
    assert len(assets) == 2
    fmts = {a.format for a in assets}
    assert fmts == {"9x16", "1x1"}
    for a in assets:
        assert a.size_bytes > 0
        assert a.width and a.height
    vertical = next(a for a in assets if a.format == "9x16")
    assert (vertical.width, vertical.height) == (1080, 1920)
    quality_report = (
        tmp_path / "work" / "jobtest" / "clips" / "01-clip-de-prueba" / "quality-9x16.json"
    )
    assert quality_report.exists()
    timeline = tmp_path / "work" / "jobtest" / "clips" / "01-clip-de-prueba" / "timeline.json"
    assert timeline.exists()
    frame = (
        tmp_path / "work" / "jobtest" / "clips" / "01-clip-de-prueba" / "timeline-9x16-frame-1.jpg"
    )
    assert frame.exists()


@requires_ffmpeg
def test_render_clip_concatenates_source_ranges_and_reports_real_duration(
    sample_video, monkeypatch, tmp_path
):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    cand = ClipCandidate(
        start=1.0,
        end=8.0,
        title="Montaje gaming",
        score=0.9,
        source_ranges=[
            SourceRange(start=1.0, end=3.0, kind="event", score=0.9),
            SourceRange(start=6.0, end=8.0, kind="replay", score=0.8),
        ],
    )

    assets = render_clip(
        "job-montage", sample_video, 0, cand, ["9x16"], segments=SEGMENTS
    )

    assert len(assets) == 1
    assert abs(assets[0].duration - 4.0) < 0.25
    clip_dir = tmp_path / "work" / "job-montage" / "clips" / "01-montaje-gaming"
    ranges = json.loads((clip_dir / "source_ranges.json").read_text(encoding="utf-8"))
    timeline = json.loads((clip_dir / "timeline.json").read_text(encoding="utf-8"))
    assert [item["kind"] for item in ranges] == ["event", "replay"]
    assert timeline["clip"]["source_ranges"] == ranges
    assert timeline["clip"]["duration"] == 4.0


@requires_ffmpeg
def test_render_clip_without_captions_does_not_write_ass(sample_video, monkeypatch, tmp_path):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    cand = ClipCandidate(start=2.0, end=8.0, title="Sin captions", score=0.9)

    render_clip(
        "job-no-captions",
        sample_video,
        0,
        cand,
        ["9x16"],
        segments=SEGMENTS,
        captions_enabled=False,
    )

    clip_dir = tmp_path / "work" / "job-no-captions" / "clips" / "01-sin-captions"
    assert not (clip_dir / "subtitles-9x16.ass").exists()


@requires_ffmpeg
def test_render_clip_burns_editorial_card_and_confirmed_privacy_mask(
    sample_video, monkeypatch, tmp_path
):
    from app.core import config

    monkeypatch.setattr(config.settings, "storage_dir", tmp_path)
    candidate = ClipCandidate(
        start=2.0,
        end=8.0,
        title="Editorial nativo",
        score=0.8,
        editorial_cards=[
            EditorialCard(
                kind="derived",
                at=3.0,
                duration=2.0,
                kicker="Conclusión",
                text="Resultado verificado",
                evidence="prueba",
            )
        ],
    )

    assets = render_clip(
        "job-editorial",
        sample_video,
        0,
        candidate,
        ["9x16"],
        segments=SEGMENTS,
        editorial_options={"protected_zones": False, "overlay_qa": True},
        privacy_masks=[
            {
                "format": "9x16",
                "x": 20,
                "y": 20,
                "width": 120,
                "height": 60,
                "start": 0.5,
                "end": 1.5,
            }
        ],
    )

    assert assets[0].size_bytes > 0
    clip_dir = tmp_path / "work" / "job-editorial" / "clips" / "01-editorial-nativo"
    assert (clip_dir / "editorial-cards-9x16.ass").exists()
    timeline = json.loads((clip_dir / "timeline.json").read_text(encoding="utf-8"))
    assert timeline["editorial"]["cards"][0]["text"] == "Resultado verificado"
    assert timeline["editorial"]["privacy_masks"][0]["format"] == "9x16"
    assert timeline["editorial"]["qa"]["boundary_frames"] > 0
