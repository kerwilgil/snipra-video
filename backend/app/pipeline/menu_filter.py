"""Filtro de pantallas de menu/lobby por OCR.

Por que existe: en gameplay, el fotograma con MAS texto suele ser el menu, que
es el momento menos publicable posible. Medido sobre un video real de Warzone:
un frame de lobby dio 35 textos (`VOTACION EN CURSO`, `PASE DE BATALLA`,
`RESERVAR SALA`, nicks de jugadores) contra 14-16 de los frames de partida.

Por eso el OCR se usa como señal NEGATIVA y no positiva. Usarlo al reves
--"mas texto = mas interesante"-- haria que Snipra eligiera menus.

RapidOCR y no Tesseract/Surya: es pip puro sobre onnxruntime, sin binario del
sistema, y Snipra corre en Windows y macOS. Tesseract exige instalar un binario
por maquina; Surya ademas necesita vllm (GPU) o llama-server (CPU) y sus pesos
llevan licencia Open Rail-M con limite de facturacion. Coste medido de RapidOCR:
~2.1s por fotograma en CPU.

La dependencia es OPCIONAL: si no esta instalada, el filtro se desactiva solo y
el pipeline sigue igual (nunca debe tumbar un job por no poder leer texto).
"""

from __future__ import annotations

import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Terminos de menu/lobby/pausa. En español e ingles porque el juego puede estar
# en cualquiera de los dos (el AC Shadows de prueba estaba en japones con
# subtitulos en español, y el HUD de Warzone en español).
MENU_TERMS = frozenset(
    {
        # es
        "pase de batalla",
        "votacion",
        "votación",
        "reservar sala",
        "recompensas",
        "operadores",
        "tienda",
        "ajustes",
        "opciones",
        "armamento",
        "personalizado",
        "pausa",
        "continuar",
        "reanudar",
        "salir",
        "partida privada",
        "buscar partida",
        "cargando",
        "marcador",
        "puntuacion",
        "puntuación",
        "menu principal",
        "menú principal",
        # en
        "battle pass",
        "loadout",
        "settings",
        "options",
        "main menu",
        "resume",
        "quit",
        "lobby",
        "store",
        "rewards",
        "matchmaking",
        "loading",
        "scoreboard",
        "match summary",
        "press start",
        "continue",
    }
)

# Una baja propia puede ser un buen momento; una pantalla que dice que el
# jugador fue eliminado no lo es. Se buscan frases dirigidas al jugador y no la
# palabra genérica "eliminación", que también aparece al conseguir una baja.
DEFEAT_SCREEN_TERMS = frozenset(
    {
        "te elimino",
        "te eliminó",
        "te eliminaron",
        "killed by",
        "you were eliminated",
        "you died",
    }
)

MIN_CONFIDENCE = 0.60
"""Por debajo de esto RapidOCR devuelve mucho ruido (fragmentos como 'ARTIOA',
'口周' aparecieron en frames de gameplay normales)."""

MENU_TEXT_COUNT = 25
"""Cantidad de textos distintos a partir de la cual un fotograma se considera
menu aunque no matchee ningun termino conocido. Calibrado con el video real:
lobby 35, gameplay 14-16."""

_HUD_FRACTION_RE = re.compile(r"\d+[.,]?\d*\s*/\s*\d+")
"""Patron de barra de vida tipo '397.41 / 1000'. Presente en gameplay real
incluso en fatalities/X-rays (verificado con frames reales de Mortal Kombat 1);
ausente en dialogo de modo historia."""

CUTSCENE_TEXT_MIN_LEN = 20
"""Largo minimo (caracteres) de un texto OCR no-HUD para tratarlo como posible
subtitulo de dialogo. RapidOCR a veces funde una frase entera de subtitulo en
un solo bloque sin espacios detectables ('harumiestanferozcomoel.', 24
caracteres) -- por eso el umbral es de caracteres, no de palabras.

Calibrado con dos casos reales que apuntan en direcciones opuestas: un
subtitulo de dialogo real de MK1 (24 caracteres, DEBE marcarse) y un nombre de
zona real de Warzone, "calle principal" (15 caracteres, NO debe marcarse -- ya
cubierto por test_gameplay_no_es_menu). El umbral en 20 deja margen a ambos
lados, pero sigue siendo un solo ejemplo por caso: un nombre de zona o de arma
mas largo en otro juego podria seguir dando falso positivo. _HUD_FRACTION_RE
tambien es especifico de juegos con barra de vida en fraccion (MK); en juegos
sin ese patron (como Warzone) esta señal no aporta y todo el peso cae en el
largo de texto, con el riesgo de falso positivo que eso implica."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


class _Engine:
    """Carga perezosa y unica del motor OCR (tarda ~1.6s en arrancar)."""

    _instance = None
    _failed = False

    @classmethod
    def get(cls):
        if cls._failed:
            return None
        if cls._instance is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                cls._instance = RapidOCR()
            except Exception as exc:  # noqa: BLE001
                # Dependencia opcional ausente o rota: se desactiva el filtro.
                logger.info("Filtro de menus desactivado (RapidOCR no disponible): %s", exc)
                cls._failed = True
                return None
        return cls._instance


def frame_texts(image_path: str) -> list[str]:
    """Textos detectados por encima del umbral de confianza. Vacio si falla."""
    return [item["text"] for item in frame_regions(image_path)]


def frame_regions(image_path: str) -> list[dict]:
    """Texto OCR con caja final; sirve para sugerir zonas privadas sin aplicarlas."""
    engine = _Engine.get()
    if engine is None:
        return []
    try:
        result, _ = engine(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR fallo sobre %s: %s", image_path, exc)
        return []
    out: list[dict] = []
    for item in result or []:
        if len(item) < 3:
            continue
        box, text, confidence = item[0], item[1], item[2]
        try:
            if float(confidence) < MIN_CONFIDENCE:
                continue
        except (TypeError, ValueError):
            continue
        norm = _normalize(str(text))
        if norm:
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            out.append(
                {
                    "text": norm,
                    "confidence": round(float(confidence), 4),
                    "x": max(0, int(min(xs))),
                    "y": max(0, int(min(ys))),
                    "width": max(1, int(max(xs) - min(xs))),
                    "height": max(1, int(max(ys) - min(ys))),
                }
            )
    return out


def looks_like_menu(texts: list[str]) -> tuple[bool, str]:
    """Detecta interfaces no publicables: menú, lobby, derrota o cutscene de historia."""
    if not texts:
        return False, ""
    joined = " | ".join(texts)
    defeated, defeat_label = looks_like_defeat(texts)
    if defeated:
        return True, defeat_label
    hits = sorted({term for term in MENU_TERMS if term in joined})
    if hits:
        return True, f"texto de menu: {', '.join(hits[:3])}"
    if len(set(texts)) >= MENU_TEXT_COUNT:
        return True, f"densidad de texto de interfaz ({len(set(texts))} elementos)"
    cutscene, cutscene_label = looks_like_cutscene(texts)
    if cutscene:
        return True, cutscene_label
    return False, ""


def looks_like_cutscene(texts: list[str]) -> tuple[bool, str]:
    """Detecta dialogo de modo historia/cutscene mezclado con gameplay real.

    Sin esto, un montaje "Highlights" podia elegir una escena de dialogo entre
    2 personajes (sin HUD de combate) como si fuera un beat narrativo mas,
    dejando un salto sin sentido entre jugadas reales de partidas distintas.
    Señal: hay un texto largo (subtitulo) y NO hay ninguna barra de vida tipo
    fraccion en el frame -- el HUD de combate esta casi siempre presente en
    gameplay real, incluso en fatalities/X-rays, pero desaparece en dialogo.
    """
    if not texts:
        return False, ""
    joined = " | ".join(texts)
    if _HUD_FRACTION_RE.search(joined):
        return False, ""
    for text in texts:
        if len(text) >= CUTSCENE_TEXT_MIN_LEN and not any(term in text for term in MENU_TERMS):
            return True, f'posible dialogo/cutscene (sin HUD de combate): "{text[:40]}"'
    return False, ""


def looks_like_defeat(texts: list[str]) -> tuple[bool, str]:
    """Distingue una muerte propia de la palabra genérica «eliminación»."""
    if not texts:
        return False, ""
    joined = " | ".join(texts)
    compact = re.sub(r"[^a-z0-9áéíóúüñ]+", "", joined)
    defeat_hits = sorted(
        {
            term
            for term in DEFEAT_SCREEN_TERMS
            if term in joined or re.sub(r"[^a-z0-9áéíóúüñ]+", "", term) in compact
        }
    )
    # RapidOCR confundió la T inicial de "Te eliminó" como F en un frame real,
    # pero conservó el sufijo unido ("feelimino").
    standalone_elimino = any(re.sub(r"[^a-z0-9áéíóúüñ]+", "", text) == "elimino" for text in texts)
    if defeat_hits or standalone_elimino or re.search(r"(?:te|fe)elimino", compact):
        label = defeat_hits[0] if defeat_hits else "te eliminó"
        return True, f"pantalla de derrota: {label}"
    return False, ""


def is_menu_frame(image_path: str) -> tuple[bool, str]:
    return looks_like_menu(frame_texts(image_path))


def filter_menu_candidates(
    candidates: list,
    video_path: str | None,
    *,
    event_callback=None,
    progress_callback=None,
) -> list:
    """Descarta candidatos cuya apertura o centro es un menu/lobby.

    La apertura importa mas que el centro: un clip que empieza con armamento ya
    perdio el gancho aunque luego tenga gameplay. Se revisa aproximadamente al
    10% de la duracion (maximo 5s) y despues el centro. Si la apertura ya es menu,
    no se paga el segundo OCR. Nunca deja el pool vacio.
    """
    if not candidates or not video_path:
        return candidates
    if _Engine.get() is None:
        return candidates

    import tempfile
    from pathlib import Path

    from app.core.config import settings

    kept: list = []
    dropped: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for idx, cand in enumerate(candidates):
            duration = max(0.0, cand.end - cand.start)
            opening = cand.start + min(5.0, max(0.5, duration * 0.10))
            middle = (cand.start + cand.end) / 2.0
            sample_times = [opening]
            if abs(middle - opening) >= 1.0:
                sample_times.append(middle)

            rejected = False
            for sample_idx, timestamp in enumerate(sample_times):
                out = Path(tmp) / f"c{idx}-s{sample_idx}.jpg"
                cmd = [
                    settings.ffmpeg_path,
                    "-v",
                    "error",
                    "-ss",
                    f"{max(0.0, timestamp):.3f}",
                    "-i",
                    str(video_path),
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
                menu, why = is_menu_frame(str(out))
                if menu:
                    dropped.append(f"{int(timestamp) // 60}m{int(timestamp) % 60:02d}s ({why})")
                    rejected = True
                    break
            if not rejected:
                kept.append(cand)
            if progress_callback:
                progress_callback(
                    (idx + 1) / len(candidates),
                    f"Filtrando menús: candidato {idx + 1}/{len(candidates)}",
                )

    if not kept:
        logger.warning("El filtro de menus dejaria el pool en 0; se usa sin filtrar.")
        return candidates
    if dropped and event_callback:
        event_callback(
            "info",
            f"Filtro de menus descarto {len(dropped)} candidato(s): {'; '.join(dropped[:3])}.",
        )
    return kept
