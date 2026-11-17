"""Seguimiento de sujeto para reframe dinámico.

Detecta personas con YOLO en frames muestreados del clip, elige al sujeto
dominante y construye un recorte que lo sigue (paneo suave) hacia el formato
destino. Si no detecta nada, devuelve None y el render cae a center-crop.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from statistics import median

from app.core.config import settings
from app.core.media import probe
from app.pipeline.reframe import target_size

logger = logging.getLogger(__name__)

_model = None
_face_mesh = None
_face_detector = None
STATIC_RANGE_RATIO = 0.22
MOVE_DEADBAND_RATIO = 0.35
MIN_PAN_HOLD_SECONDS = 2.0
# Salto de hablante (>50% del crop): el cambio es real, reaccionar más rápido.
SPEAKER_SWITCH_RATIO = 0.50
SPEAKER_SWITCH_MIN_HOLD = 1.0
# Fracción del alto del bbox de persona donde está la cara (aprox. 22% desde arriba).
FACE_Y_RATIO = 0.22
# Headroom: desplaza el objetivo Y para dejar aire sobre la cabeza.
FACE_HEADROOM_RATIO = 0.10
SPEAKER_MOUTH_WEIGHT = 0.5
TRANSITION_EASE_SECONDS = 0.35
MOTION_FRAME_WIDTH = 96
MOTION_FRAME_HEIGHT = 54
MOTION_MIN_ENERGY_RATIO = 0.08


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO("yolov8n.pt")
    return _model


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp

        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.45,
        )
    return _face_mesh


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        from uniface import SCRFD
        from uniface.detection.scrfd import SCRFDWeights

        _face_detector = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS)
    return _face_detector


def _detect_faces_uniface(frame):
    """Cara real (uniface/SCRFD), no una aproximacion sobre el bbox de una persona.

    Devuelve None -- no [] -- cuando no hay resultado utilizable (uniface no
    instalado, o cero caras encontradas), para que el caller distinga "no hay
    nadie" de "no pude preguntar" y caiga al heuristico YOLO-persona, que sigue
    encontrando sujeto en perfiles/planos lejanos donde un detector de cara
    estricto no encuentra nada.
    """
    try:
        faces = _get_face_detector().detect(frame)
    except ImportError:
        return None
    except Exception:
        logger.warning(
            "uniface fallo detectando caras, cae a heuristica YOLO-persona", exc_info=True
        )
        return None
    if not faces:
        return None

    out = []
    for face in faces:
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        out.append(((x1 + x2) / 2, (y1 + y2) / 2, bw * bh, x1, y1, x2, y2))
    return out or None


def _detect_faces_yolo(frame, width, height):
    """Estima posición de caras usando YOLO (persona → tercio superior del bbox).

    Heuristico de respaldo para cuando no hay deteccion real de cara: YOLO
    detecta personas correctamente incluso en perfiles y planos medios/lejanos
    donde un detector de cara estricto no encuentra nada.
    Devuelve lista de (cx, cy, area, x1, y1, x2, y2) igual que _detect_faces_uniface.
    """
    try:
        res = _get_model()(frame, classes=[0], verbose=False)
        boxes = res[0].boxes.xyxy.tolist()
    except Exception:
        return []

    faces = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        cx = (x1 + x2) / 2
        # cara ≈ 22% desde la parte superior del bbox de la persona
        cy = y1 + bh * FACE_Y_RATIO
        area = bw * bh
        faces.append((cx, cy, area, x1, y1, x2, y2))
    return faces


def _detect_faces(frame, width, height):
    """Posiciones de cara del frame: (cx, cy, area, x1, y1, x2, y2) por cara.

    Prueba primero deteccion real de cara (uniface/SCRFD, mas precisa: centra en
    el ojo, no en una fraccion fija del bbox de persona). Si uniface no esta
    instalado o no encuentra ninguna cara -- perfil cerrado, de espaldas, plano
    muy lejano -- cae al heuristico YOLO-persona que cubria esos casos antes.
    """
    uniface_faces = _detect_faces_uniface(frame)
    if uniface_faces is not None:
        return uniface_faces
    return _detect_faces_yolo(frame, width, height)


def _mouth_openness(frame, face) -> float:
    """Estimate mouth openness with MediaPipe FaceMesh.

    Returns a normalized mouth aperture ratio. If MediaPipe/cv2 is unavailable
    or landmarks cannot be read, returns 0 so the caller falls back to stable
    face selection by area/proximity.
    """
    try:
        import cv2
    except ImportError:
        return 0.0

    try:
        _cx, _cy, _area, x1, y1, x2, y2 = face
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, int(x1)))
        x2 = max(x1 + 2, min(width, int(x2)))
        y1 = max(0, min(height - 1, int(y1)))
        # YOLO gives the person box; the face is in the upper part.
        y2 = max(y1 + 2, min(height, int(y1 + (y2 - y1) * 0.42)))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = _get_face_mesh().process(rgb)
        if not result.multi_face_landmarks:
            return 0.0

        landmarks = result.multi_face_landmarks[0].landmark
        upper = landmarks[13]
        lower = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        mouth_height = abs(lower.y - upper.y)
        mouth_width = max(0.001, abs(right.x - left.x))
        return max(0.0, min(1.0, mouth_height / mouth_width))
    except Exception:  # noqa: BLE001
        return 0.0


def _detect_face(frame, width, height, prev_cx):
    """Centro (cx, cy) de la cara a encuadrar, o None si no hay cara visible.

    1 cara -> esa. Varias caras -> si FaceMesh esta disponible, la apertura
    de boca pesa como senal de hablante; si no, se prioriza estabilidad y area.
    """
    faces = _detect_faces(frame, width, height)
    if not faces:
        return None
    if len(faces) == 1:
        cx, cy, *_ = faces[0]
        return (cx, cy)

    best = None
    best_score = -1.0
    for face in faces:
        cx, cy, area, *_ = face
        mouth = _mouth_openness(frame, face)
        proximity = 1.0 - min(1.0, abs(cx - prev_cx) / width)
        # boca abierta (habla) domina cuando esté disponible; mientras tanto,
        # redistribuir su peso a proximidad para evitar oscilación entre hablantes
        # de tamaño similar en podcasts de 3+ personas.
        mouth_bonus = SPEAKER_MOUTH_WEIGHT * min(1.0, mouth * 2.0)
        if mouth_bonus < 1e-9:
            # sin detección de boca → sesgo de estabilidad: mantener al hablante actual
            score = 0.35 * (area / (width * height)) + 0.65 * proximity
        else:
            score = mouth_bonus + 0.3 * (area / (width * height)) + 0.2 * proximity
        if score > best_score:
            best_score = score
            best = (cx, cy)
    return best


@dataclass
class SubjectTrack:
    width: int
    height: int
    samples: list[tuple[float, float, float]]  # (t, center_x, center_y) en px de la fuente
    trim_start: float = 0.0  # segundos a recortar del inicio del clip base (plano general / intro)


def _even(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 0 else n - 1


def analyze_face_anchor(path: str, max_samples: int = 7) -> SubjectTrack | None:
    """Detecta la posición mediana de la cara y devuelve un crop ESTÁTICO centrado en ella.

    A diferencia de analyze_subject (que anima el paneo), esta función devuelve un
    SubjectTrack con un único punto → build_track_graph produce un crop fijo sin movimiento.
    Ideal para 'center' cuando el hablante está descentrado en el encuadre 16:9 original.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    if duration <= 0 or width == 0 or height == 0:
        cap.release()
        return None

    step = max(duration / max_samples, 0.5)
    cx_list: list[float] = []
    cy_list: list[float] = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        faces = _detect_faces(frame, width, height)
        if faces:
            cx, cy, *_ = max(faces, key=lambda f: f[2])  # cara más grande
            cx_list.append(cx)
            cy_list.append(cy + height * FACE_HEADROOM_RATIO)
        t += step

    cap.release()
    if not cx_list:
        return None

    # posición mediana → único keyframe → crop fijo
    med_cx = float(sorted(cx_list)[len(cx_list) // 2])
    med_cy = float(sorted(cy_list)[len(cy_list) // 2])
    return SubjectTrack(width=width, height=height, samples=[(0.0, med_cx, med_cy)])


def analyze_subject(path: str, max_samples: int = 36) -> SubjectTrack | None:
    """Muestrea el clip, detecta al sujeto dominante y devuelve su trayectoria."""
    try:
        import cv2
    except ImportError:
        return analyze_motion_energy(path, max_samples=max_samples)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return analyze_motion_energy(path, max_samples=max_samples)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    if duration <= 0 or width == 0 or height == 0:
        cap.release()
        return None

    step = max(duration / max_samples, 0.5)
    try:
        model = _get_model()
    except Exception as exc:  # noqa: BLE001
        logger.info("Vision tracking no disponible (%s). Usando motion-energy fallback.", exc)
        cap.release()
        return analyze_motion_energy(path, max_samples=max_samples)

    samples: list[tuple[float, float, float]] = []
    has_detection: list[bool] = []
    prev_cx = width / 2.0
    prev_cy = height / 2.0
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = _pick_subject(model, frame, width, height, prev_cx)
        detected = cx is not None
        if detected:
            prev_cx = cx
            prev_cy = cy
        # Siempre registra la posición — si no hay detección usa la última conocida.
        # Esto evita que gaps de detección creen arcos de interpolación largos y erráticos.
        samples.append((round(t, 2), prev_cx, prev_cy))
        has_detection.append(detected)
        t += step

    # Sample cerca del final del clip para capturar cambios de hablante
    # que ocurren en el último intervalo (donde el loop regular no alcanza).
    t_final = max(0.0, duration - 0.5)
    if samples and t_final > samples[-1][0] + 0.3:
        cap.set(cv2.CAP_PROP_POS_MSEC, t_final * 1000)
        ok, frame = cap.read()
        if ok:
            cx_f, cy_f = _pick_subject(model, frame, width, height, prev_cx)
            if cx_f is not None:
                prev_cx, prev_cy = cx_f, cy_f
            samples.append((round(t_final, 2), prev_cx, prev_cy))

    cap.release()

    # Ajuste inicial cuando los primeros frames no tienen cara detectable.
    # Siempre hace backfill: rellena los samples previos a la primera detección
    # con la posición de esa primera detección real (mejor que center-fallback).
    # Si el gap es ≥ _TRIM_THRESHOLD s, además marca trim_start para que el
    # pipeline de render pueda recortar el clip base — la decisión de trim la
    # toma render_from_base, NO esta función.  Los tiempos de los samples no se
    # reindexan aquí; el pipeline re-ejecuta analyze_subject sobre el clip ya
    # recortado si aplica.
    _TRIM_THRESHOLD = 2.0
    trim_start = 0.0
    if any(has_detection):
        first_det = next(i for i, d in enumerate(has_detection) if d)
        if first_det > 0:
            t_first = samples[first_det][0]
            if t_first >= _TRIM_THRESHOLD:
                trim_start = t_first
            # Backfill siempre (tanto para gaps largos como cortos)
            _, fx, fy = samples[first_det]
            samples[:first_det] = [(t, fx, fy) for t, _, _ in samples[:first_det]]

    if len(samples) < 2:
        return None
    return SubjectTrack(width=width, height=height, samples=samples, trim_start=trim_start)


def _assign_dual_slots(
    faces: list[tuple],
    slot_a: tuple[float, float] | None,
    slot_b: tuple[float, float] | None,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Asigna hasta 2 caras detectadas a los slots A/B por cercanía a su última
    posición conocida. Si falta una cara en el frame, el slot correspondiente
    mantiene su última posición (igual que analyze_subject con 1 sola persona).
    """
    if not faces:
        return slot_a, slot_b

    if len(faces) >= 2:
        a0, a1 = (faces[0][0], faces[0][1]), (faces[1][0], faces[1][1])
        if slot_a is None or slot_b is None:
            # Primera vez que se ven 2 caras reales a la vez: siembra los 2
            # slots de una sola vez (izquierda -> derecha). Sin este caso
            # aparte, "slot_b is None" caia en la rama de 1-sola-cara de abajo
            # y la segunda cara detectada se descartaba para siempre -- B
            # nunca llegaba a sembrarse aunque el video SI tuviera 2 personas.
            return a0, a1
        # Ya sembrados ambos: la asignacion (directa o cruzada) que minimiza
        # la distancia total a las posiciones previas -- evita que las 2
        # personas "salten" de slot cuando ambas se mueven un poco.
        cost_direct = abs(a0[0] - slot_a[0]) + abs(a1[0] - slot_b[0])
        cost_swapped = abs(a1[0] - slot_a[0]) + abs(a0[0] - slot_b[0])
        return (a0, a1) if cost_direct <= cost_swapped else (a1, a0)

    # 1 sola cara detectada.
    cx, cy = faces[0][0], faces[0][1]
    if slot_b is None:
        # Todavia no se ha visto a una 2a persona: esta cara es la 1a (A).
        return (cx, cy), slot_b
    da = abs(cx - slot_a[0]) if slot_a is not None else float("inf")
    db = abs(cx - slot_b[0])
    return ((cx, cy), slot_b) if da <= db else (slot_a, (cx, cy))


def analyze_dual_subjects(
    path: str, max_samples: int = 36
) -> tuple[SubjectTrack, SubjectTrack] | None:
    """Como analyze_subject, pero sigue a 2 personas en paralelo (split-screen).

    Asigna cada cara detectada al slot (A/B) mas cercano a su posicion anterior:
    funciona porque en un plano de podcast la camara es estatica y las personas
    no cambian de asiento. None si nunca se detectan 2 caras simultaneamente --
    el caller debe caer a track/center de una sola persona en ese caso.
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    if duration <= 0 or width == 0 or height == 0:
        cap.release()
        return None

    step = max(duration / max_samples, 0.5)
    samples_a: list[tuple[float, float, float]] = []
    samples_b: list[tuple[float, float, float]] = []
    slot_a: tuple[float, float] | None = None
    slot_b: tuple[float, float] | None = None
    ever_had_two = False
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        faces = _detect_faces(frame, width, height)
        # Las 2 caras mas grandes: evita que una deteccion espuria de fondo
        # (objeto, mano) entre como "tercera persona" en un frame con ruido.
        faces = sorted(faces, key=lambda f: f[2], reverse=True)[:2]
        if len(faces) >= 2:
            ever_had_two = True
            faces = sorted(faces, key=lambda f: f[0])  # izquierda -> derecha

        slot_a, slot_b = _assign_dual_slots(faces, slot_a, slot_b)
        if slot_a is not None:
            samples_a.append((round(t, 2), slot_a[0], slot_a[1]))
        if slot_b is not None:
            samples_b.append((round(t, 2), slot_b[0], slot_b[1]))
        t += step

    cap.release()

    if not ever_had_two or len(samples_a) < 2 or len(samples_b) < 2:
        return None

    track_a = SubjectTrack(width=width, height=height, samples=samples_a)
    track_b = SubjectTrack(width=width, height=height, samples=samples_b)
    return track_a, track_b


def _read_motion_frames(
    path: str,
    *,
    duration: float,
    max_samples: int,
    width: int = MOTION_FRAME_WIDTH,
    height: int = MOTION_FRAME_HEIGHT,
) -> list[bytes]:
    fps = max(0.2, min(4.0, max_samples / max(duration, 0.1)))
    cmd = [
        settings.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps:.3f},scale={width}:{height}:flags=fast_bilinear,format=gray",
        "-frames:v",
        str(max_samples),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=180)
    if proc.returncode != 0:
        logger.info("Motion-energy fallback no pudo leer frames con FFmpeg.")
        return []
    frame_size = width * height
    payload = proc.stdout or b""
    return [
        payload[i : i + frame_size]
        for i in range(0, len(payload), frame_size)
        if len(payload[i : i + frame_size]) == frame_size
    ]


def _motion_centers(
    frames: list[bytes],
    *,
    source_width: int,
    source_height: int,
    sample_width: int = MOTION_FRAME_WIDTH,
    sample_height: int = MOTION_FRAME_HEIGHT,
) -> list[tuple[float, float]]:
    if len(frames) < 2:
        return []

    y_start = int(sample_height * 0.22)
    y_end = int(sample_height * 0.76)
    row_width = sample_width
    centers: list[tuple[float, float]] = []
    for idx in range(1, len(frames)):
        prev = frames[idx - 1]
        current = frames[idx]
        column_energy = [0] * sample_width
        total = 0
        for y in range(y_start, y_end):
            row = y * row_width
            for x in range(sample_width):
                delta = abs(current[row + x] - prev[row + x])
                if delta < 8:
                    continue
                column_energy[x] += delta
                total += delta
        if total <= 0:
            continue
        avg = total / sample_width
        if avg < MOTION_MIN_ENERGY_RATIO * 255:
            continue
        weighted_x = sum(x * energy for x, energy in enumerate(column_energy)) / total
        centers.append(
            (
                (weighted_x + 0.5) / sample_width * source_width,
                source_height * 0.42,
            )
        )
    return centers


def analyze_motion_energy(path: str, max_samples: int = 36) -> SubjectTrack | None:
    """Fallback liviano: sigue la zona con mayor energia de movimiento sin YOLO/OpenCV."""
    try:
        info = probe(path)
    except Exception as exc:  # noqa: BLE001
        logger.info("Motion-energy fallback no pudo sondear video (%s).", exc)
        return None
    if info.duration <= 0 or info.width <= 0 or info.height <= 0:
        return None

    frames = _read_motion_frames(str(path), duration=info.duration, max_samples=max_samples)
    centers = _motion_centers(frames, source_width=info.width, source_height=info.height)
    if len(centers) < 2:
        return None

    step = info.duration / max(len(centers) - 1, 1)
    samples = [(round(i * step, 2), cx, cy) for i, (cx, cy) in enumerate(centers)]
    return SubjectTrack(width=info.width, height=info.height, samples=samples)


def _pick_subject(model, frame, width, height, prev_cx):
    """Elige el objetivo del encuadre.

    Prioridad 1: la CARA dominante (encuadra el rostro del hablante, con algo de
    aire sobre la cabeza). Prioridad 2 (fallback si no hay cara visible): la
    persona dominante por YOLO, centrada en el torso como antes.
    """
    face = _detect_face(frame, width, height, prev_cx)
    if face is not None:
        cx, cy = face
        # baja un poco el objetivo Y para dejar headroom (la cara queda sobre el centro)
        return cx, cy + height * FACE_HEADROOM_RATIO

    res = model(frame, classes=[0], verbose=False)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    best = None
    best_score = -1.0
    for b in boxes.xyxy.tolist():
        x1, y1, x2, y2 = b
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2
        # score: área normalizada penalizando saltos respecto al sujeto previo
        proximity = 1.0 - min(1.0, abs(cx - prev_cx) / width)
        score = (area / (width * height)) * 0.7 + proximity * 0.3
        if score > best_score:
            best_score = score
            best = ((x1 + x2) / 2, (y1 + y2) / 2)
    return best if best else (None, None)


def _median_filter(values: list[float], window: int = 3) -> list[float]:
    """Filtro de mediana deslizante con relleno de borde.

    Elimina outliers de detección (frames donde YOLO se equivoca) sin mezclar
    posiciones de hablantes distintos como haría un EMA. Una mediana de ventana 3
    descarta el valor extremo y devuelve el valor intermedio real, preservando
    transiciones abruptas de hablante → exactamente lo que necesita la función escalón.
    """
    if len(values) <= 2:
        return list(values)
    half = window // 2
    padded = [values[0]] * half + list(values) + [values[-1]] * half
    return [float(median(padded[i : i + window])) for i in range(len(values))]


def _collapse_plateaus(
    times: list[float], values: list[float], gap: float
) -> list[tuple[float, float, float]]:
    """Agrupa muestras consecutivas dentro de `gap` entre si en (t_inicio, t_fin, mediana).

    Sin esto, el deadband de `_stabilize_offsets` compara cada muestra contra el ULTIMO
    keyframe aceptado: si dos personas alternan en un plano de los 2 y cada salto
    individual queda por debajo del deadband (aunque la diferencia acumulada entre sus
    posiciones sea grande), el ancla nunca se mueve y el crop queda fijo en el primer
    punto para todo el clip -- que puede no ser ninguna de las dos personas si ese punto
    cae entre ambas. Agrupar en plateaus antes del walk deja que el deadband compare
    posiciones SOSTENIDAS entre si, no una muestra aislada contra un ancla vieja.
    t_fin (tiempo de la ultima muestra del grupo) deja medir cuanto duro cada
    plateau por si solo, independiente de cuando ocurrio el ultimo keyframe aceptado.
    """
    plateaus: list[tuple[float, float, float]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or abs(values[i] - values[start]) > gap:
            group = values[start:i]
            plateaus.append((times[start], times[i - 1], float(median(group))))
            start = i
    return plateaus


def _bimodal_split(values: list[float], min_gap: float) -> tuple[float, float] | None:
    """Si los valores se separan en 2 grupos por un salto >= min_gap, devuelve sus
    2 centros (mediana de cada grupo). None si no hay separacion clara.

    Detecta el caso "2 personas alternando en un plano de los 2": sin esto, el
    walk de `_stabilize_offsets` compara cada plateau contra el ULTIMO keyframe
    aceptado, y si ese keyframe resulta ser un plateau cercano al borde de su
    propio grupo (equidistante de ambos clusters por casualidad), CADA plateau
    del otro grupo puede quedar individualmente bajo el deadband aunque la
    distancia real entre los 2 grupos sea grande -- el crop nunca se mueve.
    Snapear cada plateau al centro de su cluster antes del walk garantiza que
    los cambios de cluster siempre traen la distancia completa entre centros.
    """
    if len(values) < 2:
        return None
    ordered = sorted(values)
    best_gap, best_idx = 0.0, -1
    for i in range(1, len(ordered)):
        gap = ordered[i] - ordered[i - 1]
        if gap > best_gap:
            best_gap, best_idx = gap, i
    if best_gap < min_gap:
        return None
    low, high = ordered[:best_idx], ordered[best_idx:]
    return float(median(low)), float(median(high))


def _stabilize_offsets(
    samples: list[tuple[float, float]],
    crop_extent: int,
) -> list[tuple[float, float]]:
    """Reduce keyframes: fija el crop si el movimiento real es pequeño.

    Usa filtro de mediana (no EMA) para limpiar outliers de YOLO sin mezclar
    posiciones de hablantes distintos. Combinar con _expr() escalón produce
    cortes de cámara instantáneos en vez de paneos graduales.
    """
    if len(samples) <= 1:
        return samples

    times = [t for t, _ in samples]
    smoothed = _median_filter([v for _, v in samples])
    static_range = max(36.0, crop_extent * STATIC_RANGE_RATIO)
    if max(smoothed) - min(smoothed) <= static_range:
        return [(times[0], float(round(median(smoothed))))]

    move_deadband = max(28.0, crop_extent * MOVE_DEADBAND_RATIO)
    speaker_switch_threshold = crop_extent * SPEAKER_SWITCH_RATIO
    plateaus = _collapse_plateaus(times, smoothed, static_range)

    # _bimodal_split solo decide SI relajar el deadband, no a que valor snapear --
    # un clip editado multi-camara puede tener 3+ encuadres reales distintos (plano
    # de los 2, close-up A, close-up B), no solo 2, y forzar cada plateau a uno de
    # 2 centros mezclaria posiciones de tomas distintas. Si hay señal de que esto
    # no es el drift gradual de una sola persona (aunque sea solo 2 clusters
    # detectables), se compara plateau contra plateau con static_range: el ruido
    # de una muestra ya lo absorbio el filtro de mediana + el agrupamiento en
    # plateaus, asi que cualquier plateau que difiera del ultimo keyframe aceptado
    # en mas de static_range ES una posicion sostenida distinta, no ruido. Sin
    # este relajamiento, el deadband de movimiento (mucho mas grande, pensado para
    # filtrar micro-paneo de una sola persona muestra a muestra) puede dejar el
    # crop fijo para siempre en el primer plateau si el resto alterna dentro de
    # esa banda sin superarla nunca contra ese ancla especifica.
    has_distinct_positions = (
        _bimodal_split([v for _, _, v in plateaus], static_range) is not None
    )
    effective_deadband = static_range if has_distinct_positions else move_deadband

    stable: list[tuple[float, float]] = [(plateaus[0][0], plateaus[0][2])]
    last_t, last_v = stable[0]
    for p_start, p_end, value in plateaus[1:]:
        change = abs(value - last_v)
        # Salto grande (>50% del crop) = cambio real de hablante → min hold más corto
        min_hold = (
            SPEAKER_SWITCH_MIN_HOLD if change > speaker_switch_threshold else MIN_PAN_HOLD_SECONDS
        )
        # Un plateau cuenta como sostenido si paso suficiente tiempo desde el ULTIMO
        # keyframe aceptado, O si el propio plateau se mantuvo detectado por min_hold
        # segundos seguidos. Sin el segundo criterio, un corte de camara real (p.ej.
        # un plano ancho de 3s) que ocurre poco despues del keyframe anterior se
        # descarta solo por llegar "pronto", aunque su propia duracion demuestre que
        # no es ruido de una sola muestra.
        sustained = (p_start - last_t >= min_hold) or (p_end - p_start >= min_hold)
        if not sustained or change < effective_deadband:
            continue
        stable.append((p_start, value))
        last_t, last_v = p_start, value

    return [(t, float(round(v))) for t, v in stable]


def _expr(samples: list[tuple[float, float]], ease_seconds: float = TRANSITION_EASE_SECONDS) -> str:
    """Expresión FFmpeg con cubic ease-in-out entre keyframes.

    El crop se mantiene quieto en cada posición y transiciona suavemente al siguiente
    keyframe durante `ease_seconds` (curva 3x²−2x³). Disimula el cambio de encuadre
    sin crear un paneo largo visible.
    """
    if len(samples) == 1:
        return str(int(samples[0][1]))

    e = str(int(samples[-1][1]))
    for i in range(len(samples) - 2, -1, -1):
        t0, v0 = samples[i]
        t1, v1 = samples[i + 1]
        # Clamp para que la ease termine antes del siguiente keyframe
        td = min(ease_seconds, (t1 - t0) * 0.8)
        delta = int(v1) - int(v0)
        if abs(delta) < 8 or td < 0.05:
            # Movimiento mínimo o ventana demasiado corta: escalón directo
            e = f"if(lt(t,{t1:.2f}),{int(v0)},{e})"
        else:
            # ease: v0 + delta*(3x²−2x³), x=(t−t1)/td, t∈[t1, t1+td]
            ease = (
                f"{int(v0)}+{delta}*(3*pow((t-{t1:.2f})/{td:.3f},2)-2*pow((t-{t1:.2f})/{td:.3f},3))"
            )
            e = f"if(lt(t,{t1:.2f}),{int(v0)},if(lt(t,{t1 + td:.3f}),{ease},{e}))"
    t0, v0 = samples[0]
    if t0 > 0:
        return f"if(lt(t,{t0:.2f}),{int(v0)},{e})"
    return e


def _pane_crop_filter(
    track: SubjectTrack, tw: int, th: int, in_label: str, out_label: str
) -> str:
    """Filtergraph crop(paneo)+scale de UN track hacia un lienzo tw x th.

    Compartido por build_track_graph (lienzo completo) y build_split_graph (cada
    mitad del split-screen es un lienzo tw x th/2 con su propio track).
    """
    W, H = track.width, track.height
    ta = tw / th
    sa = W / H

    if abs(sa - ta) < 0.02:
        # mismo aspecto: solo escalar
        return f"[{in_label}]scale={tw}:{th},setsar=1[{out_label}]"

    if sa > ta:
        cw, ch = _even(H * ta), _even(H)
        max_off = max(0, W - cw)
        offsets = [(t, max(0.0, min(max_off, cx - cw / 2))) for t, cx, _ in track.samples]
        offsets = _stabilize_offsets(offsets, cw)
        x_expr = f"'{_expr(offsets)}'"
        crop = f"crop={cw}:{ch}:x={x_expr}:y=0"
    else:
        cw, ch = _even(W), _even(W / ta)
        max_off = max(0, H - ch)
        offsets = [(t, max(0.0, min(max_off, cy - ch / 2))) for t, _, cy in track.samples]
        offsets = _stabilize_offsets(offsets, ch)
        y_expr = f"'{_expr(offsets)}'"
        crop = f"crop={cw}:{ch}:x=0:y={y_expr}"

    return f"[{in_label}]{crop},scale={tw}:{th},setsar=1[{out_label}]"


def build_track_graph(
    track: SubjectTrack, fmt: str, in_label: str = "0:v", out_label: str = "v"
) -> str | None:
    """Construye el filtergraph crop(paneo)+scale para seguir al sujeto."""
    tw, th = target_size(fmt)
    return _pane_crop_filter(track, tw, th, in_label, out_label)


def build_split_graph(
    track_top: SubjectTrack,
    track_bottom: SubjectTrack,
    fmt: str,
    in_label: str = "0:v",
    out_label: str = "v",
) -> str:
    """2 crops independientes apilados: persona 1 arriba, persona 2 abajo.

    Cada mitad sigue a su propio sujeto con el mismo paneo suave que el modo
    track normal, sobre un lienzo de la mitad de alto. Uso: planos de podcast
    con 2 personas donde track/auto solo puede seguir a una y el resto del
    encuadre queda vacio o entre las dos (ver F38 en context-snipra.md).
    """
    tw, th = target_size(fmt)
    half_h = _even(th // 2)
    top_in, bot_in = f"{in_label}split_top", f"{in_label}split_bot"
    top_graph = _pane_crop_filter(track_top, tw, half_h, top_in, "vsplit_top")
    bottom_graph = _pane_crop_filter(track_bottom, tw, half_h, bot_in, "vsplit_bot")
    return (
        f"[{in_label}]split=2[{top_in}][{bot_in}];"
        f"{top_graph};{bottom_graph};"
        f"[vsplit_top][vsplit_bot]vstack=inputs=2[{out_label}]"
    )
