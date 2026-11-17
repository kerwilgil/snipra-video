"""Tests unitarios del dispatcher de deteccion de cara (sin modelos reales)."""

from dataclasses import dataclass

from app.pipeline import tracking


@dataclass
class _FakeFace:
    bbox: tuple[float, float, float, float]


class _FakeDetector:
    def __init__(self, faces):
        self._faces = faces

    def detect(self, frame):
        return self._faces


def test_detect_faces_prefers_uniface_when_available(monkeypatch):
    sentinel = [(1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setattr(tracking, "_detect_faces_uniface", lambda frame: sentinel)
    monkeypatch.setattr(
        tracking,
        "_detect_faces_yolo",
        lambda frame, w, h: (_ for _ in ()).throw(AssertionError("no deberia llamarse")),
    )
    assert tracking._detect_faces(None, 100, 100) == sentinel


def test_detect_faces_falls_back_to_yolo_when_uniface_returns_none(monkeypatch):
    sentinel = [(5.0, 6.0, 7.0, 0.0, 0.0, 1.0, 1.0)]
    monkeypatch.setattr(tracking, "_detect_faces_uniface", lambda frame: None)
    monkeypatch.setattr(tracking, "_detect_faces_yolo", lambda frame, w, h: sentinel)
    assert tracking._detect_faces(None, 100, 100) == sentinel


def test_detect_faces_uniface_returns_none_on_import_error(monkeypatch):
    def _raise():
        raise ImportError("uniface no instalado")

    monkeypatch.setattr(tracking, "_get_face_detector", _raise)
    assert tracking._detect_faces_uniface(None) is None


def test_detect_faces_uniface_returns_none_when_no_faces_found(monkeypatch):
    monkeypatch.setattr(tracking, "_get_face_detector", lambda: _FakeDetector([]))
    assert tracking._detect_faces_uniface(object()) is None


def test_detect_faces_uniface_converts_bbox_to_center_and_area(monkeypatch):
    face = _FakeFace(bbox=(10.0, 20.0, 30.0, 60.0))
    monkeypatch.setattr(tracking, "_get_face_detector", lambda: _FakeDetector([face]))
    faces = tracking._detect_faces_uniface(object())
    assert faces == [(20.0, 40.0, 800.0, 10.0, 20.0, 30.0, 60.0)]


def test_detect_faces_uniface_skips_degenerate_bbox(monkeypatch):
    face = _FakeFace(bbox=(10.0, 20.0, 10.0, 60.0))
    monkeypatch.setattr(tracking, "_get_face_detector", lambda: _FakeDetector([face]))
    assert tracking._detect_faces_uniface(object()) is None


# --- _stabilize_offsets: no debe congelarse cuando 2 personas alternan ---------------


def test_stabilize_offsets_does_not_freeze_on_alternating_speakers():
    """Caso real: podcast de 2 personas en plano de los 2 / multi-camara.

    Valores tomados de una reproduccion real (backend/context, F37/F38): antes de
    este fix, el deadband de movimiento comparaba cada muestra contra el PRIMER
    valor (382, cerca del punto medio entre las 2 posiciones reales ~266 y ~424)
    y ninguna diferencia individual lo superaba -> el crop quedaba fijo en 382
    para todo el clip, un punto que no corresponde a ninguna de las 2 personas.
    """
    offsets = [
        (0.00, 382.0), (1.53, 261.6), (3.06, 246.1), (4.58, 261.1),
        (6.11, 382.9), (7.64, 455.0), (9.17, 433.3), (10.70, 401.7),
        (12.23, 466.1), (13.75, 263.9), (15.28, 266.1), (16.81, 264.7),
        (18.34, 462.7), (19.87, 445.5), (21.39, 262.7), (22.92, 425.0),
        (24.45, 424.4), (25.98, 441.2), (27.51, 387.7), (29.03, 385.2),
        (30.56, 387.9), (32.09, 368.4), (33.62, 414.3), (35.15, 417.3),
        (36.68, 397.8), (38.20, 429.1), (39.73, 517.2), (41.26, 432.4),
        (42.79, 465.0), (44.32, 270.3), (45.84, 270.8), (47.37, 269.4),
        (48.90, 267.5), (50.43, 392.9), (51.96, 394.0), (53.49, 379.8),
        (54.51, 415.7),
    ]  # fmt: skip

    stable = tracking._stabilize_offsets(offsets, crop_extent=404)

    assert len(stable) > 1, "el crop no debe quedar fijo en un solo punto"
    values = [v for _, v in stable]
    assert max(values) - min(values) > 100, "debe alternar entre posiciones realmente distintas"


def test_stabilize_offsets_captures_short_sustained_plateau_soon_after_keyframe():
    """Un plano sostenido 3s (t=1.53..4.58) que llega poco despues del primer
    keyframe (t=0) debe registrarse aunque no hayan pasado los 2s de espera desde
    el ULTIMO keyframe -- su propia duracion (3s) ya demuestra que no es ruido.
    """
    offsets = [
        (0.00, 380.0), (1.53, 260.0), (3.06, 260.0), (4.58, 260.0), (6.11, 380.0),
    ]  # fmt: skip

    stable = tracking._stabilize_offsets(offsets, crop_extent=404)

    times = [t for t, _ in stable]
    assert 1.53 in times, "el plano sostenido debe capturarse aunque llegue pronto"


def test_stabilize_offsets_still_collapses_single_speaker_jitter():
    """Regresion: micro-movimiento de una sola persona sigue colapsando a un
    unico crop estatico (no debe volverse mas twitchy con el fix de plateaus).
    """
    offsets = [(0.0, 640.0), (1.0, 660.0), (2.0, 650.0), (3.0, 645.0)]

    stable = tracking._stabilize_offsets(offsets, crop_extent=1200)

    assert len(stable) == 1


# --- split-screen: asignacion de caras a slots A/B a lo largo del tiempo ------------


def test_assign_dual_slots_seeds_both_from_first_two_face_frame():
    """Bug real (F38): mientras solo se ve 1 cara, slot_b nunca se siembra. La
    primera vez que aparecen 2 caras reales, la condicion de guardia
    "slot_b is None" caia en la rama de 1-sola-cara y la segunda cara se
    descartaba para siempre -- B nunca llegaba a sembrarse en todo el clip.
    """
    slot_a, slot_b = None, None
    # Frames iniciales con 1 sola cara visible (host1 unicamente).
    for _ in range(5):
        slot_a, slot_b = tracking._assign_dual_slots([(400.0, 300.0, 900.0)], slot_a, slot_b)
    assert slot_a == (400.0, 300.0)
    assert slot_b is None

    # Primer frame con 2 caras reales (plano de los 2).
    faces = [(350.0, 300.0, 900.0), (900.0, 310.0, 950.0)]
    slot_a, slot_b = tracking._assign_dual_slots(faces, slot_a, slot_b)

    assert slot_a is not None and slot_b is not None
    assert slot_a != slot_b


def test_assign_dual_slots_keeps_identity_across_frames():
    """Una vez sembrados los 2 slots, cada persona se mantiene en su slot
    aunque ambas se muevan un poco (no deben "saltar" de slot por ruido)."""
    slot_a, slot_b = (300.0, 300.0), (900.0, 300.0)

    faces = [(310.0, 305.0, 900.0), (890.0, 295.0, 950.0)]
    slot_a, slot_b = tracking._assign_dual_slots(faces, slot_a, slot_b)

    assert slot_a[0] < 500  # sigue siendo la persona de la izquierda
    assert slot_b[0] > 500  # sigue siendo la persona de la derecha


def test_assign_dual_slots_holds_last_position_when_no_faces():
    slot_a, slot_b = (300.0, 300.0), (900.0, 300.0)
    result = tracking._assign_dual_slots([], slot_a, slot_b)
    assert result == (slot_a, slot_b)
